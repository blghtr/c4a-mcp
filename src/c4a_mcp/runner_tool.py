# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic]
# :relates-to: uses: "crawl4ai.AsyncWebCrawler", uses: "config_models.CrawlerConfigYAML", uses: "presets.crawling_factory", uses: "presets.extraction_factory"
# :rationale: "Encapsulates the core business logic of configuring and executing the crawl4ai crawler. Creates strategy instances from parameters using factory functions."
# :references: PRD: "F001, F002, F003", SPEC: "SPEC-F001, SPEC-F002, SPEC-F003"
# :contract: pre: "Valid RunnerInput (config may contain strategy_params)", post: "Returns RunnerOutput with markdown or error"
# :decision_cache: "Separated logic from server code to allow for easier testing and potential CLI usage [ARCH-003]. Refactored to parameterized configuration to avoid serialization issues [ARCH-010]"
# LLM:END

import contextlib
import io
import logging
import traceback
from datetime import datetime
from typing import Any

# Import crawl4ai components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult

from .config_models import CrawlerConfigYAML
from .models import RunnerInput, RunnerOutput
from .presets import crawling_factory, extraction_factory

# Configure logger with hierarchy path format
logger = logging.getLogger(__name__)


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
            # Build CrawlerRunConfig with 3-layer merge:
            # defaults → file overrides → tool overrides
            run_config = self._build_run_config(inputs.config)

            # NOTE(REVIEWER): crawl4ai supports c4a-script DSL directly via js_code parameter.
            # The library automatically processes c4a-script commands (GO, CLICK, WAIT, etc.)
            # as shown in doc/c4a.md examples.
            if inputs.script:
                run_config.js_code = inputs.script  # Pass c4a-script DSL to js_code

            # Use pre-created browser_config from constructor
            # NOTE(REVIEWER): Reusing browser_config is efficient.
            # Verified: AsyncWebCrawler treats config as read-only during initialization.
            # CRITICAL: Redirect stdout/stderr to prevent crawl4ai progress messages
            # from breaking MCP JSON-RPC protocol (which expects only JSON on stdout)
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                async with AsyncWebCrawler(config=self.browser_config) as crawler:
                    crawl_result: CrawlResult = await crawler.arun(inputs.url, config=run_config)
            
            # Log captured output for debugging (but don't send to stdout/stderr)
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()
            if captured_stdout:
                logger.debug(
                    "[C4A-MCP | Logic] Captured crawl4ai stdout | data: {output: %s}",
                    captured_stdout[:500],  # Limit log size
                )
            if captured_stderr:
                logger.debug(
                    "[C4A-MCP | Logic] Captured crawl4ai stderr | data: {output: %s}",
                    captured_stderr[:500],  # Limit log size
                )

            # Process result into RunnerOutput
            # CrawlResult has metadata dict for title, status_code directly, no timestamp
            markdown_content = ""
            
            # Diagnostic logging for markdown extraction
            logger.debug(
                "[C4A-MCP | Logic] Extracting markdown | "
                "data: {has_markdown: %s, markdown_is_none: %s, markdown_type: %s}",
                hasattr(crawl_result, "markdown"),
                not hasattr(crawl_result, "markdown") or crawl_result.markdown is None,
                type(getattr(crawl_result, "markdown", None)).__name__ if hasattr(crawl_result, "markdown") else "N/A",
            )
            
            if crawl_result.markdown:
                # markdown can be str or MarkdownGenerationResult
                if isinstance(crawl_result.markdown, str):
                    markdown_content = crawl_result.markdown
                    logger.debug(
                        "[C4A-MCP | Logic] Markdown is string | "
                        "data: {length: %d, preview: %s}",
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
                    len(crawl_result.html) if hasattr(crawl_result, "html") and crawl_result.html else 0,
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
                "[C4A-MCP | Logic] Applying tool-level config overrides | "
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
            kwargs["deep_crawl_strategy"] = crawling_factory.create_crawling_strategy(
                strategy_type, crawling_params
            )
            logger.debug(
                "[C4A-MCP | Logic] Created crawling strategy from parameters | "
                "data: {strategy_type: %s}",
                strategy_type,
            )

        if extraction_params:
            strategy_type = extraction_params.pop("strategy_type")
            config = extraction_params.get("config")
            kwargs["extraction_strategy"] = extraction_factory.create_extraction_strategy(
                strategy_type, config
            )
            logger.debug(
                "[C4A-MCP | Logic] Created extraction strategy from parameters | "
                "data: {strategy_type: %s}",
                strategy_type,
            )

        logger.debug(
            "[C4A-MCP | Logic] Built CrawlerRunConfig | data: {kwargs: %s}",
            kwargs,
        )

        # Create CrawlerRunConfig
        return CrawlerRunConfig(**kwargs)
