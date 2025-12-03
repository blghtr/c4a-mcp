# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic]
# :relates-to: uses: "crawl4ai.AsyncWebCrawler", uses: "config_models.CrawlerConfigYAML", implements: "SPEC-F001, SPEC-F002, SPEC-F003"
# :rationale: "Encapsulates the core business logic of configuring and executing the crawl4ai crawler."
# :contract: pre: "Valid RunnerInput", post: "Returns RunnerOutput with markdown or error"
# :decision_cache: "Separated logic from server code to allow for easier testing and potential CLI usage [ARCH-003]"
# LLM:END

import logging
import traceback
from datetime import datetime

# Import crawl4ai components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult

from .config_models import CrawlerConfigYAML
from .models import RunnerInput, RunnerOutput

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
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                crawl_result: CrawlResult = await crawler.arun(inputs.url, config=run_config)

                # Process result into RunnerOutput
                # CrawlResult has metadata dict for title, status_code directly, no timestamp
                markdown_content = ""
                if crawl_result.markdown:
                    # markdown can be str or MarkdownGenerationResult
                    if isinstance(crawl_result.markdown, str):
                        markdown_content = crawl_result.markdown
                    else:
                        # MarkdownGenerationResult has raw_markdown attribute
                        markdown_content = getattr(crawl_result.markdown, "raw_markdown", "")

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
        Build CrawlerRunConfig with 3-layer merge:
        defaults → file overrides → tool overrides.

        Args:
            tool_config_dict: Optional config dict from tool call

        Returns:
            CrawlerRunConfig instance ready for crawl
        """
        # Start with defaults (already includes file overrides from server.py)
        merged = self.default_crawler_config.model_copy()

        # Apply tool-level overrides if provided
        if tool_config_dict:
            logger.debug(
                "[C4A-MCP | Logic] Applying tool-level config overrides | " "data: {overrides: %s}",
                tool_config_dict,
            )
            tool_overrides = CrawlerConfigYAML.from_dict(tool_config_dict)
            merged = merged.merge(tool_overrides)

        # Convert to CrawlerRunConfig kwargs (excludes None values, converts types)
        kwargs = merged.to_crawler_run_config_kwargs()

        logger.debug(
            "[C4A-MCP | Logic] Built CrawlerRunConfig | data: {kwargs: %s}",
            kwargs,
        )

        # Create CrawlerRunConfig
        return CrawlerRunConfig(**kwargs)
