# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic]
# :relates-to: uses: "crawl4ai.AsyncWebCrawler", uses: "config_models.CrawlerConfigYAML", uses: "presets.crawling_factory", uses: "presets.extraction_factory"
# :rationale: "Encapsulates the core business logic of configuring and executing the crawl4ai crawler. Creates strategy instances from parameters using factory functions."
# :references: PRD: "F001, F002, F003", SPEC: "SPEC-F001, SPEC-F002, SPEC-F003"
# :contract: pre: "Valid RunnerInput (config may contain strategy_params)", post: "Returns RunnerOutput with markdown or error"
# :decision_cache: "Separated logic from server code to allow for easier testing and potential CLI usage [ARCH-003]. Refactored to parameterized configuration to avoid serialization issues [ARCH-010]"
# LLM:END

import asyncio
import logging
import sys
import traceback
from datetime import datetime

# Import crawl4ai components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult

from .config_models import CrawlerConfigYAML
from .models import RunnerInput, RunnerOutput
from .presets import crawling_factory, extraction_factory
from .crawler_registry import register as register_crawler, deregister as deregister_crawler, close_crawler

# Configure logger with hierarchy path format
logger = logging.getLogger(__name__)


class LoggerWriter:
    """Stream writer that redirects output to a logger in real-time.

    This replaces StringIO buffering to enable live log streaming during
    long-running operations like deep crawls.
    """

    def __init__(self, logger_instance: logging.Logger, level: int = logging.DEBUG):
        """Initialize LoggerWriter.

        Args:
            logger_instance: Logger to write to
            level: Log level for messages (default: DEBUG)
        """
        self.logger = logger_instance
        self.level = level
        self.buffer = ""
        self._last_line: str | None = None

    def write(self, message: str) -> int:
        """Write message to logger.

        Buffers partial lines and logs complete lines immediately.

        Args:
            message: Text to write

        Returns:
            Number of characters written
        """
        if message and message != "\n":
            # Buffer the message
            self.buffer += message

            # If we have complete lines, log them
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                # Log all complete lines
                for line in lines[:-1]:
                    cleaned = line.strip()
                    if not cleaned:
                        continue
                    # Skip immediate duplicates to avoid double-logging the same crawl4ai progress line.
                    if cleaned == self._last_line:
                        continue
                    self._last_line = cleaned
                    self.logger.log(self.level, "[crawl4ai] %s", cleaned)
                # Keep the incomplete line in buffer
                self.buffer = lines[-1]

        return len(message)

    def flush(self) -> None:
        """Flush any remaining buffered content."""
        if self.buffer.strip():
            cleaned = self.buffer.strip()
            if cleaned != self._last_line:
                self.logger.log(self.level, "[crawl4ai] %s", cleaned)
                self._last_line = cleaned
            self.buffer = ""


class CrawlRunner:
    """
    Executes crawl sessions using crawl4ai based on provided configuration.

    Attributes:
        default_crawler_config: Default crawler configuration from YAML
        browser_config: Pre-created browser configuration
    """

    def __init__(
        self,
        default_crawler_config: CrawlerConfigYAML,
        browser_config: BrowserConfig,
    ):
        """Initialize CrawlRunner with default configs.

        Args:
            default_crawler_config: Default crawler settings (from YAML + overrides)
            browser_config: Pre-created browser configuration
        """
        self.default_crawler_config = default_crawler_config
        self.browser_config = browser_config
        logger.info(
            "[C4A-MCP | Logic] CrawlRunner initialized | " "data: {browser_type: %s, headless: %s}",
            browser_config.browser_type,
            browser_config.headless,
        )

    async def run(self, inputs: RunnerInput) -> RunnerOutput:
        """
        Executes the crawl logic.

        Args:
            inputs: Validated input parameters.

        Returns:
            RunnerOutput containing markdown or error info.
        """
        run_config = None  # Initialize to avoid UnboundLocalError
        try:
            logger.debug(
                "[C4A-MCP | Logic | Run] Starting crawl | data: {url: %s, has_script: %s}",
                inputs.url,
                inputs.script is not None,
            )
            # Flush logger handlers to ensure logs appear immediately
            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            # Build CrawlerRunConfig with 3-layer merge:
            # defaults → file overrides → tool overrides
            logger.debug("[C4A-MCP | Logic | Run] Building run config")
            run_config = self._build_run_config(inputs.config)
            logger.debug(
                "[C4A-MCP | Logic | Run] Run config built | data: {has_deep_crawl: %s, has_extraction: %s, timeout: %s}",
                run_config.deep_crawl_strategy is not None,
                run_config.extraction_strategy is not None,
                run_config.page_timeout,
            )
            # Flush after config build
            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            # NOTE(REVIEWER): crawl4ai supports c4a-script DSL directly via js_code parameter.
            # The library automatically processes c4a-script commands (GO, CLICK, WAIT, etc.)
            # as shown in doc/c4a.md examples.
            if inputs.script:
                run_config.js_code = inputs.script  # Pass c4a-script DSL to js_code
                logger.debug(
                    "[C4A-MCP | Logic | Run] Script attached | data: {script_length: %d}",
                    len(inputs.script),
                )

            # Use pre-created browser_config from constructor
            # NOTE(REVIEWER): Reusing browser_config is efficient.
            # Verified: AsyncWebCrawler treats config as read-only during initialization.

            # CRITICAL: Redirect stdout/stderr to logger for real-time streaming
            # This prevents crawl4ai progress messages from breaking MCP JSON-RPC protocol
            # while still allowing visibility into crawl progress
            logger.debug("[C4A-MCP | Logic | Run] Setting up log streaming")
            stdout_logger = LoggerWriter(logger, logging.DEBUG)
            stderr_logger = LoggerWriter(logger, logging.WARNING)

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            crawler = AsyncWebCrawler(config=self.browser_config)
            register_crawler(crawler)
            entered = False

            try:
                sys.stdout = stdout_logger
                sys.stderr = stderr_logger

                # Flush before starting crawler
                for handler in logger.handlers:
                    if hasattr(handler, "flush"):
                        handler.flush()

                logger.debug("[C4A-MCP | Logic | Run] Initializing AsyncWebCrawler")
                try:
                    async with crawler as active_crawler:
                        entered = True
                        logger.debug("[C4A-MCP | Logic | Run] Starting crawler.arun()")
                        # Flush before arun
                        for handler in logger.handlers:
                            if hasattr(handler, "flush"):
                                handler.flush()

                        crawl_result: CrawlResult = await active_crawler.arun(
                            inputs.url, config=run_config
                        )
                        logger.debug("[C4A-MCP | Logic | Run] crawler.arun() completed")
                except asyncio.CancelledError:
                    logger.warning("[C4A-MCP | Logic | Run] Cancelled during crawl; forcing cleanup")
                    raise
            finally:
                # Ensure crawler is closed if context failed to enter; otherwise __aexit__ already ran
                try:
                    if not entered:
                        await close_crawler(crawler, logger, timeout=10.0)
                finally:
                    deregister_crawler(crawler)

                # Restore original stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                # Flush any remaining buffered content
                stdout_logger.flush()
                stderr_logger.flush()
                logger.debug("[C4A-MCP | Logic | Run] Log streaming restored")
                # Final flush
                for handler in logger.handlers:
                    if hasattr(handler, "flush"):
                        handler.flush()

            # Normalize list responses from deep crawl (returns list[CrawlResult])
            if isinstance(crawl_result, list):
                logger.debug(
                    "[C4A-MCP | Logic] Received list crawl_result | data: {count: %d}",
                    len(crawl_result),
                )
                if not crawl_result:
                    return RunnerOutput(markdown="", error="Deep crawl returned no results")

                markdown_parts: list[str] = []
                for idx, res in enumerate(crawl_result):
                    if getattr(res, "markdown", None):
                        if isinstance(res.markdown, str):
                            markdown_parts.append(res.markdown)
                        else:
                            markdown_obj = res.markdown
                            content = getattr(markdown_obj, "raw_markdown", "") or getattr(
                                markdown_obj, "fit_markdown", ""
                            )
                            if content:
                                markdown_parts.append(content)
                # Add a clear delimiter between pages to avoid merging content silently
                markdown_content = "\n\n---\n\n".join(markdown_parts)

                # Use the first result for metadata to preserve behavior
                crawl_result = crawl_result[0]
            else:
                markdown_content = ""

            # Process result into RunnerOutput
            # CrawlResult has metadata dict for title, status_code directly, no timestamp

            # Diagnostic logging for markdown extraction
            logger.debug(
                "[C4A-MCP | Logic] Extracting markdown | "
                "data: {has_markdown: %s, markdown_is_none: %s, markdown_type: %s}",
                hasattr(crawl_result, "markdown"),
                not hasattr(crawl_result, "markdown") or crawl_result.markdown is None,
                (
                    type(getattr(crawl_result, "markdown", None)).__name__
                    if hasattr(crawl_result, "markdown")
                    else "N/A"
                ),
            )

            if crawl_result.markdown:
                # markdown can be str or MarkdownGenerationResult
                if isinstance(crawl_result.markdown, str):
                    markdown_content = crawl_result.markdown
                    logger.debug(
                        "[C4A-MCP | Logic] Markdown is string | " "data: {length: %d, preview: %s}",
                        len(markdown_content),
                        markdown_content[:200] if markdown_content else "",
                    )
                else:
                    # MarkdownGenerationResult has raw_markdown attribute
                    markdown_obj = crawl_result.markdown
                    logger.debug(
                        "[C4A-MCP | Logic] Markdown is object | "
                        "data: {type: %s, has_raw_markdown: %s, has_fit_markdown: %s}",
                        type(markdown_obj).__name__,
                        hasattr(markdown_obj, "raw_markdown"),
                        hasattr(markdown_obj, "fit_markdown"),
                    )
                    markdown_content = getattr(markdown_obj, "raw_markdown", "")
                    if not markdown_content:
                        markdown_content = getattr(markdown_obj, "fit_markdown", "")
                    logger.debug(
                        "[C4A-MCP | Logic] Extracted markdown from object | "
                        "data: {length: %d, preview: %s}",
                        len(markdown_content) if markdown_content else 0,
                        markdown_content[:200] if markdown_content else "",
                    )
            else:
                logger.warning(
                    "[C4A-MCP | Logic] No markdown found in crawl_result | "
                    "data: {url: %s, status: %s, has_html: %s, html_length: %d}",
                    crawl_result.url,
                    crawl_result.status_code,
                    hasattr(crawl_result, "html"),
                    (
                        len(crawl_result.html)
                        if hasattr(crawl_result, "html") and crawl_result.html
                        else 0
                    ),
                )

            return RunnerOutput(
                markdown=markdown_content,
                metadata={
                    "url": crawl_result.url,
                    "title": (
                        crawl_result.metadata.get("title") if crawl_result.metadata else None
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "status": crawl_result.status_code,
                },
                error=None,
            )
        except Exception as e:
            # Catch any unexpected errors and try to categorize them
            error_message = str(e).lower()
            formatted_error = ""

            # Log full traceback for debugging
            logger.error(
                "[C4A-MCP | Logic] Error during crawl execution | "
                "data: {error_type: %s, error: %s}",
                type(e).__name__,
                str(e),
            )
            logger.debug(
                "[C4A-MCP | Logic] Full traceback | data: {traceback: %s}",
                traceback.format_exc(),
            )

            if "timeout" in error_message:
                # Extract timeout from page_timeout (milliseconds) if available
                timeout_seconds = (
                    (run_config.page_timeout / 1000)
                    if run_config
                    and hasattr(run_config, "page_timeout")
                    and run_config.page_timeout
                    else 60
                )
                formatted_error = f"Timeout after {int(timeout_seconds)} seconds: {e}"
            elif (
                "network" in error_message
                or "connection" in error_message
                or "http error" in error_message
            ):
                formatted_error = f"Network error: {e}"
            elif "script execution failed" in error_message or "js error" in error_message:
                formatted_error = f"Script execution failed: {e}"
            else:
                formatted_error = f"An unexpected error occurred: {e}"

            # Return only sanitized error message to client
            return RunnerOutput(markdown="", error=formatted_error)

    def _build_run_config(self, tool_config_dict: dict | None) -> CrawlerRunConfig:
        """
        Build CrawlerRunConfig with strategy creation from parameters.

        Implements 3-layer merge: defaults → file overrides → tool overrides.
        If strategy parameters are present, creates strategy instances using factory functions.

        Args:
            tool_config_dict: Optional config dict from tool call (may contain strategy_params)

        Returns:
            CrawlerRunConfig instance ready for crawl (with strategies created from parameters)
        """
        logger.debug(
            "[C4A-MCP | Logic | Build Config] Starting config build | "
            "data: {has_tool_config: %s}",
            tool_config_dict is not None,
        )

        # Start with defaults (already includes file overrides from server.py)
        merged = self.default_crawler_config.model_copy()

        # Initialize strategy params (may be None)
        crawling_params = None
        extraction_params = None

        if tool_config_dict:
            # Extract strategy parameters (if present) before merging
            # These are not part of CrawlerConfigYAML, so we handle them separately
            crawling_params = tool_config_dict.pop("deep_crawl_strategy_params", None)
            extraction_params = tool_config_dict.pop("extraction_strategy_params", None)

            logger.debug(
                "[C4A-MCP | Logic | Build Config] Extracted strategy params | "
                "data: {has_crawling_params: %s, has_extraction_params: %s}",
                crawling_params is not None,
                extraction_params is not None,
            )

            logger.debug(
                "[C4A-MCP | Logic | Build Config] Applying tool-level config overrides | "
                "data: {overrides: %s}",
                tool_config_dict,
            )

            # Merge remaining config with defaults
            tool_overrides = CrawlerConfigYAML.from_dict(tool_config_dict)
            merged = merged.merge(tool_overrides)

        # Convert to CrawlerRunConfig kwargs (excludes None values, converts types)
        kwargs = merged.to_crawler_run_config_kwargs()

        # Create strategies from parameters using factory functions
        if crawling_params:
            strategy_type = crawling_params.pop("strategy_type")
            logger.debug(
                "[C4A-MCP | Logic | Build Config] Creating crawling strategy | "
                "data: {strategy_type: %s, params: %s}",
                strategy_type,
                crawling_params,
            )
            kwargs["deep_crawl_strategy"] = crawling_factory.create_crawling_strategy(
                strategy_type, crawling_params
            )
            logger.debug(
                "[C4A-MCP | Logic | Build Config] Crawling strategy created | "
                "data: {strategy_type: %s, instance: %s}",
                strategy_type,
                type(kwargs["deep_crawl_strategy"]).__name__,
            )

        if extraction_params:
            strategy_type = extraction_params.pop("strategy_type")
            config = extraction_params.get("config")

            # Log extraction schema details for CSS strategy
            # NOTE: config is a Pydantic model (ExtractionConfigCss), not a dict
            if strategy_type == "css" and config:
                # Access Pydantic attribute directly (has alias "schema" too)
                schema = config.extraction_schema
                if schema:
                    logger.debug(
                        "[C4A-MCP | Logic | Build Config] CSS extraction schema | "
                        "data: {base_selector: %s, fields_count: %d, field_names: %s}",
                        schema.get("baseSelector"),
                        len(schema.get("fields", [])),
                        [f.get("name") for f in schema.get("fields", [])],
                    )

            logger.debug(
                "[C4A-MCP | Logic | Build Config] Creating extraction strategy | "
                "data: {strategy_type: %s}",
                strategy_type,
            )
            kwargs["extraction_strategy"] = extraction_factory.create_extraction_strategy(
                strategy_type, config
            )
            logger.debug(
                "[C4A-MCP | Logic | Build Config] Extraction strategy created | "
                "data: {strategy_type: %s, instance: %s}",
                strategy_type,
                type(kwargs["extraction_strategy"]).__name__,
            )

        logger.debug(
            "[C4A-MCP | Logic | Build Config] Config build complete | "
            "data: {has_deep_crawl: %s, has_extraction: %s, timeout: %s}",
            "deep_crawl_strategy" in kwargs,
            "extraction_strategy" in kwargs,
            kwargs.get("page_timeout"),
        )

        # Create CrawlerRunConfig
        return CrawlerRunConfig(**kwargs)
