# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic]
# :relates-to: uses: "crawl4ai.AsyncWebCrawler", uses: "config_models.CrawlerConfigYAML", uses: "presets.crawling_factory", uses: "presets.extraction_factory", extends: "base_runner.BaseRunner"
# :rationale: "Encapsulates the core business logic of configuring and executing the crawl4ai crawler. Creates strategy instances from parameters using factory functions."
# :references: PRD: "F001, F002, F003", SPEC: "SPEC-F001, SPEC-F002, SPEC-F003"
# :contract: pre: "Valid RunnerInput (config may contain strategy_params)", post: "Returns RunnerOutput with markdown or error"
# :decision_cache: "Refactored to inherit from BaseRunner to eliminate code duplication [ARCH-020]"
# LLM:END

import logging
from typing import Any

# Import crawl4ai components
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from .base_runner import BaseRunner
from .config_models import CrawlerConfigYAML
from .models import RunnerInput
from .presets import crawling_factory, extraction_factory

# Configure logger with hierarchy path format
logger = logging.getLogger(__name__)


class CrawlRunner(BaseRunner):
    """
    Executes crawl sessions using crawl4ai based on provided configuration.

    Inherits lifecycle management from BaseRunner and implements specific
    configuration assembly and execution logic for standard crawls.

    Attributes:
        default_crawler_config: Default crawler configuration from YAML
        browser_config: Pre-created browser configuration
    """

    def _build_config(self, inputs: RunnerInput) -> CrawlerRunConfig:
        """
        Build CrawlerRunConfig with strategy creation from parameters.

        Implements 3-layer merge: defaults → file overrides → tool overrides.
        If strategy parameters are present, creates strategy instances using factory functions.

        Args:
            inputs: RunnerInput containing URL, script, and config dict

        Returns:
            CrawlerRunConfig instance ready for crawl
        """
        tool_config_dict = inputs.config

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
            # Make a copy to avoid mutating the input dict if it's reused
            config_copy = tool_config_dict.copy()
            crawling_params = config_copy.pop("deep_crawl_strategy_params", None)
            extraction_params = config_copy.pop("extraction_strategy_params", None)

            # Use the cleaned copy for merging
            tool_config_dict = config_copy

            logger.debug(
                "[C4A-MCP | Logic | Build Config] Extracted strategy params | "
                "data: {has_crawling_params: %s, has_extraction_params: %s}",
                crawling_params is not None,
                extraction_params is not None,
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

        if extraction_params:
            strategy_type = extraction_params.pop("strategy_type")
            config = extraction_params.get("config")

            # Log extraction schema details for CSS strategy
            if strategy_type == "css" and config:
                schema = getattr(config, "extraction_schema", None)
                if schema:
                    logger.debug(
                        "[C4A-MCP | Logic | Build Config] CSS extraction schema | "
                        "data: {base_selector: %s, fields_count: %d}",
                        schema.get("baseSelector"),
                        len(schema.get("fields", [])),
                    )

            kwargs["extraction_strategy"] = extraction_factory.create_extraction_strategy(
                strategy_type, config
            )

        # Handle c4a-script DSL via js_code parameter
        if inputs.script:
            kwargs["js_code"] = inputs.script
            logger.debug(
                "[C4A-MCP | Logic | Build Config] Script attached | data: {script_length: %d}",
                len(inputs.script),
            )

        logger.debug(
            "[C4A-MCP | Logic | Build Config] Config build complete | "
            "data: {has_deep_crawl: %s, has_extraction: %s, timeout: %s}",
            "deep_crawl_strategy" in kwargs,
            "extraction_strategy" in kwargs,
            kwargs.get("page_timeout"),
        )

        return CrawlerRunConfig(**kwargs)

    async def _execute_core(
        self, crawler: AsyncWebCrawler, config: CrawlerRunConfig, inputs: RunnerInput
    ) -> Any:
        """
        Executes the standard crawl logic using arun().

        Args:
            crawler: Initialized AsyncWebCrawler
            config: Configured CrawlerRunConfig
            inputs: RunnerInput parameters

        Returns:
            CrawlResult or list[CrawlResult]
        """
        logger.debug("[C4A-MCP | Logic | Run] Starting crawler.arun()")

        # arun returns CrawlResult or list[CrawlResult] (for deep crawl)
        result = await crawler.arun(inputs.url, config=config)

        logger.debug("[C4A-MCP | Logic | Run] crawler.arun() completed")
        return result
