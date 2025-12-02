# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic]
# :relates-to: uses: "crawl4ai.AsyncWebCrawler", implements: "SPEC-F001, SPEC-F002, SPEC-F003"
# :rationale: "Encapsulates the core business logic of configuring and executing the crawl4ai crawler."
# :contract: pre: "Valid RunnerInput", post: "Returns RunnerOutput with markdown or error"
# :decision_cache: "Separated logic from server code to allow for easier testing and potential CLI usage [ARCH-003]"
# LLM:END

import logging
import traceback
from datetime import datetime
from typing import Any

# Import crawl4ai components
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)

from .models import RunnerInput, RunnerOutput

# Configure logger with hierarchy path format
logger = logging.getLogger(__name__)


class CrawlRunner:
    """
    Executes crawl sessions using crawl4ai based on provided configuration.
    """

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
            # Map input config to CrawlerRunConfig
            run_config = self._map_config(inputs.config)

            # NOTE(REVIEWER): crawl4ai supports c4a-script DSL directly via js_code parameter.
            # The library automatically processes c4a-script commands (GO, CLICK, WAIT, etc.)
            # as shown in doc/c4a.md examples.
            if inputs.script:
                run_config.js_code = inputs.script  # Pass c4a-script DSL to js_code

            async with AsyncWebCrawler() as crawler:
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

    def _map_config(self, config_dict: dict[str, Any] | None) -> CrawlerRunConfig:
        """
        Helper to convert dictionary config to CrawlerRunConfig.
        """
        if not config_dict:
            return CrawlerRunConfig(cache_mode=CacheMode.BYPASS)  # Default config

        # Convert bypass_cache to cache_mode if provided
        cache_mode = CacheMode.BYPASS  # Default
        if "bypass_cache" in config_dict:
            cache_mode = (
                CacheMode.BYPASS if config_dict.get("bypass_cache", True) else CacheMode.ENABLED
            )

        # Convert timeout from seconds to milliseconds for page_timeout
        # PRD specifies default 60s timeout (Non-Functional Requirements)
        timeout_seconds = config_dict.get("timeout", 60)
        # page_timeout is required int, default 60000 ms (60 seconds)
        page_timeout_ms = int(timeout_seconds * 1000)

        # Build config dict with only non-None values for Optional fields
        config_kwargs = {
            "cache_mode": cache_mode,
            "page_timeout": page_timeout_ms,  # Required int, always set
            "exclude_external_links": config_dict.get("exclude_external_links", False),
            "exclude_social_media_links": config_dict.get("exclude_social_media_links", False),
        }

        # Add Optional[str] fields only if not None
        css_selector = config_dict.get("css_selector")
        if css_selector is not None:
            config_kwargs["css_selector"] = css_selector

        wait_for = config_dict.get("wait_for")
        if wait_for is not None:
            config_kwargs["wait_for"] = wait_for

        # word_count_threshold is int, only add if provided (otherwise use crawl4ai default ~200)
        word_count_threshold = config_dict.get("word_count_threshold")
        if word_count_threshold is not None:
            config_kwargs["word_count_threshold"] = int(word_count_threshold)

        run_config = CrawlerRunConfig(**config_kwargs)

        # Handle extraction strategy if provided
        # JsonCssExtractionStrategy requires schema dict for initialization
        extraction_strategy_str = config_dict.get("extraction_strategy")
        if extraction_strategy_str:
            extraction_strategy_schema = config_dict.get("extraction_strategy_schema")
            if extraction_strategy_str.lower() == "jsoncss":
                if extraction_strategy_schema:
                    # Create instance with provided schema
                    run_config.extraction_strategy = JsonCssExtractionStrategy(
                        extraction_strategy_schema
                    )
                else:
                    # Log warning if schema not provided
                    logger.warning(
                        "[C4A-MCP | Logic] extraction_strategy='jsoncss' specified "
                        "but extraction_strategy_schema not provided. "
                        "Skipping extraction strategy setup. | data: {config: %s}",
                        config_dict,
                    )

        return run_config
