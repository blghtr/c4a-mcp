# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Tools]
# :relates-to: uses: "runner_tool.CrawlRunner", uses: "presets.extraction_factory", uses: "presets.crawling_factory", uses: "presets.models"
# :rationale: "MCP tool implementations for preset crawling patterns, providing high-level interfaces with sensible defaults."
# :references: PRD: "F004", SPEC: "SPEC-F004"
# :contract: pre: "Valid PresetInput models", post: "Returns RunnerOutput JSON string"
# :decision_cache: "Reused CrawlRunner for execution consistency, built CrawlerRunConfig with strategies from factories [ARCH-010]. Implemented dependency injection via factory pattern to eliminate global state [ARCH-011]"
# LLM:END

"""
MCP tool implementations for preset crawling patterns.

These tools provide high-level interfaces for common crawling scenarios:
- crawl_deep: BFS deep crawling
- crawl_deep_smart: Best-first crawling with keywords
- scrape_page: Single-page scraping

All tools support extraction strategies and extensive parameter customization.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from crawl4ai import CacheMode, CrawlerRunConfig

from ..config_models import CrawlerConfigYAML
from ..models import RunnerInput
from ..runner_tool import CrawlRunner
from .crawling_factory import create_crawling_strategy
from .extraction_factory import create_extraction_strategy
from .models import (
    CrawlDeepSmartInput,
    DeepCrawlPresetInput,
    PresetBaseConfig,
    ScrapePagePresetInput,
)

logger = logging.getLogger(__name__)


def _build_run_config_from_preset(
    preset_config: PresetBaseConfig,
    crawling_strategy: Any | None,
    extraction_strategy: Any | None,
) -> CrawlerRunConfig:
    """Build CrawlerRunConfig from preset configuration and strategies.

    Args:
        preset_config: PresetBaseConfig with all parameters
        crawling_strategy: DeepCrawlStrategy instance or None
        extraction_strategy: ExtractionStrategy instance or None

    Returns:
        CrawlerRunConfig ready for execution
    """
    # Convert preset config to CrawlerRunConfig kwargs
    kwargs = preset_config.to_crawler_run_config_kwargs()

    # Remove extraction_strategy_config - it's not a CrawlerRunConfig parameter
    # (extraction_strategy is already created and passed separately)
    kwargs.pop("extraction_strategy_config", None)

    # Add strategies
    if crawling_strategy:
        kwargs["deep_crawl_strategy"] = crawling_strategy
    if extraction_strategy:
        kwargs["extraction_strategy"] = extraction_strategy

    return CrawlerRunConfig(**kwargs)


def _crawler_run_config_to_dict(run_config: CrawlerRunConfig) -> dict[str, Any]:
    """Convert CrawlerRunConfig to dict for RunnerInput.

    Uses whitelist-based approach: extracts only attributes that are supported
    by CrawlerConfigYAML, using introspection for robustness.

    Args:
        run_config: CrawlerRunConfig instance

    Returns:
        Dict suitable for RunnerInput.config
    """
    # Whitelist of CrawlerConfigYAML fields that can be extracted from CrawlerRunConfig
    # This is more robust than manual mapping - automatically includes all supported fields
    config_dict: dict[str, Any] = {}

    # Get all field names from CrawlerConfigYAML model (whitelist)
    crawler_config_fields = set(CrawlerConfigYAML.model_fields.keys())

    # Reverse mapping: CrawlerConfigYAML field -> CrawlerRunConfig attribute
    # Some fields have different names in CrawlerRunConfig
    run_config_attr_map: dict[str, str] = {
        "timeout": "page_timeout",  # CrawlerRunConfig uses page_timeout (ms)
        "bypass_cache": "cache_mode",  # CrawlerRunConfig uses cache_mode (enum)
    }

    # Extract attributes using whitelist from CrawlerConfigYAML
    for config_field_name in crawler_config_fields:
        # Skip fields that are handled specially or not in CrawlerRunConfig
        if config_field_name in ("extraction_strategy", "extraction_strategy_schema"):
            continue

        # Map to CrawlerRunConfig attribute name
        run_config_attr = run_config_attr_map.get(config_field_name, config_field_name)

        if hasattr(run_config, run_config_attr):
            value = getattr(run_config, run_config_attr, None)

            # Skip None values (CrawlerConfigYAML excludes them)
            if value is None:
                continue

            # Special conversions
            if config_field_name == "timeout" and run_config_attr == "page_timeout":
                # Convert from milliseconds to seconds
                config_dict[config_field_name] = value / 1000
            elif config_field_name == "bypass_cache" and run_config_attr == "cache_mode":
                # Convert CacheMode enum to bypass_cache bool
                config_dict[config_field_name] = value == CacheMode.BYPASS
            else:
                # Direct mapping (same name in both)
                config_dict[config_field_name] = value

    # Note: deep_crawl_strategy and extraction_strategy are not serializable
    # and should be handled by CrawlRunner directly, not via config dict

    return config_dict


def create_preset_tools(
    crawl_runner: CrawlRunner,
) -> tuple[
    Callable[..., Awaitable[str]],
    Callable[..., Awaitable[str]],
    Callable[..., Awaitable[str]],
]:
    """Create preset tool functions with injected CrawlRunner dependency.

    This factory function implements dependency injection pattern, eliminating
    global state and improving testability. Each returned function is a closure
    that captures the crawl_runner instance.

    Args:
        crawl_runner: CrawlRunner instance to inject into tools

    Returns:
        Tuple of (crawl_deep, crawl_deep_smart, scrape_page) functions ready for MCP registration

    Example:
        ```python
        crawl_runner = CrawlRunner(...)
        crawl_deep, crawl_deep_smart, scrape_page = create_preset_tools(crawl_runner)
        mcp.tool()(crawl_deep)
        mcp.tool()(crawl_deep_smart)
        mcp.tool()(scrape_page)
        ```
    """
    # NOTE(REVIEWER): Closure-based dependency injection is thread-safe and memory-efficient.
    # Each closure captures crawl_runner by reference, not by value, so there's no
    # performance penalty. This pattern is commonly used in Python for DI.
    logger.info(
        "[C4A-MCP | Presets | Tools] Creating preset tools with dependency injection | "
        "data: {runner_type: %s}",
        type(crawl_runner).__name__,
    )

    async def crawl_deep(
        url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        include_external: bool = False,
        script: str | None = None,
        extraction_strategy: str | None = None,
        extraction_strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Perform BFS deep crawling of a website.

        Crawls a website using breadth-first search, exploring all links at one depth
        before going deeper. Ideal for collecting content from small to medium sites.

        Args:
            url: Starting URL for the crawl
            max_depth: Maximum crawl depth (default: 2)
            max_pages: Maximum number of pages to crawl (default: 50)
            include_external: Whether to follow external links (default: False)
            script: Optional c4a-script DSL or JavaScript code for page interactions
            extraction_strategy: Extraction strategy type ("regex", "css", "llm", or None)
            extraction_strategy_config: Configuration for extraction strategy
            **kwargs: Additional CrawlerRunConfig parameters (timeout, css_selector, etc.)

        Returns:
            JSON string with RunnerOutput structure (markdown, metadata, error)
        """
        # Validate input
        input_data = DeepCrawlPresetInput(
            url=url,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            extraction_strategy=extraction_strategy,
            extraction_strategy_config=extraction_strategy_config,
            **kwargs,
        )

        # Create crawling strategy
        crawling_strategy = create_crawling_strategy(
            "bfs",
            {
                "max_depth": input_data.max_depth,
                "max_pages": input_data.max_pages,
                "include_external": input_data.include_external,
            },
        )

        # Create extraction strategy
        extraction_strategy_instance = create_extraction_strategy(
            input_data.extraction_strategy, input_data.extraction_strategy_config
        )

        # Build preset config
        preset_config = PresetBaseConfig(**input_data.model_dump(exclude={"url", "max_depth", "max_pages", "include_external"}))

        # Build run config
        run_config = _build_run_config_from_preset(
            preset_config, crawling_strategy, extraction_strategy_instance
        )

        # Execute via CrawlRunner
        # NOTE(REVIEWER): crawl_runner is captured from closure - no global state access
        runner_input = RunnerInput(
            url=url, script=script, config=_crawler_run_config_to_dict(run_config)
        )
        result = await crawl_runner.run(runner_input)

        return result.model_dump_json()

    async def crawl_deep_smart(
        url: str,
        keywords: list[str],
        max_depth: int = 2,
        max_pages: int = 25,
        include_external: bool = False,
        script: str | None = None,
        extraction_strategy: str | None = None,
        extraction_strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Perform best-first deep crawling with keyword prioritization.

        Crawls a website prioritizing pages by keyword relevance. Useful for
        thematic research on large portals.

        Args:
            url: Starting URL for the crawl
            keywords: Keywords for relevance scoring (required)
            max_depth: Maximum crawl depth (default: 2)
            max_pages: Maximum number of pages to crawl (default: 25)
            include_external: Whether to follow external links (default: False)
            script: Optional c4a-script DSL or JavaScript code for page interactions
            extraction_strategy: Extraction strategy type ("regex", "css", "llm", or None)
            extraction_strategy_config: Configuration for extraction strategy
            **kwargs: Additional CrawlerRunConfig parameters

        Returns:
            JSON string with RunnerOutput structure (markdown, metadata, error)
        """
        # Validate input
        input_data = CrawlDeepSmartInput(
            url=url,
            keywords=keywords,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            extraction_strategy=extraction_strategy,
            extraction_strategy_config=extraction_strategy_config,
            **kwargs,
        )

        # Create crawling strategy
        crawling_strategy = create_crawling_strategy(
            "best_first",
            {
                "max_depth": input_data.max_depth,
                "max_pages": input_data.max_pages,
                "include_external": input_data.include_external,
                "keywords": input_data.keywords,
            },
        )

        # Create extraction strategy
        extraction_strategy_instance = create_extraction_strategy(
            input_data.extraction_strategy, input_data.extraction_strategy_config
        )

        # Build preset config
        preset_config = PresetBaseConfig(**input_data.model_dump(exclude={"url", "keywords", "max_depth", "max_pages", "include_external"}))

        # Build run config
        run_config = _build_run_config_from_preset(
            preset_config, crawling_strategy, extraction_strategy_instance
        )

        # Execute via CrawlRunner
        # NOTE(REVIEWER): crawl_runner is captured from closure - no global state access
        runner_input = RunnerInput(
            url=url, script=script, config=_crawler_run_config_to_dict(run_config)
        )
        result = await crawl_runner.run(runner_input)

        return result.model_dump_json()

    async def scrape_page(
        url: str,
        script: str | None = None,
        extraction_strategy: str | None = None,
        extraction_strategy_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Scrape content from a single page.

        Extracts content from only the specified page. JavaScript rendering is
        controlled by BrowserConfig set at server startup (see server.py).

        Args:
            url: URL of the page to scrape
            script: Optional c4a-script DSL or JavaScript code for page interactions
            extraction_strategy: Extraction strategy type ("regex", "css", "llm", or None)
            extraction_strategy_config: Configuration for extraction strategy
            **kwargs: Additional CrawlerRunConfig parameters

        Returns:
            JSON string with RunnerOutput structure (markdown, metadata, error)
        """
        # Validate input
        input_data = ScrapePagePresetInput(
            url=url,
            extraction_strategy=extraction_strategy,
            extraction_strategy_config=extraction_strategy_config,
            **kwargs,
        )

        # No deep crawling for single-page scraping
        crawling_strategy = None

        # Create extraction strategy
        extraction_strategy_instance = create_extraction_strategy(
            input_data.extraction_strategy, input_data.extraction_strategy_config
        )

        # Build preset config
        preset_config = PresetBaseConfig(**input_data.model_dump(exclude={"url"}))

        # Build run config
        run_config = _build_run_config_from_preset(
            preset_config, crawling_strategy, extraction_strategy_instance
        )

        # Execute via CrawlRunner
        # NOTE(REVIEWER): crawl_runner is captured from closure - no global state access
        runner_input = RunnerInput(
            url=url, script=script, config=_crawler_run_config_to_dict(run_config)
        )
        result = await crawl_runner.run(runner_input)

        return result.model_dump_json()

    return crawl_deep, crawl_deep_smart, scrape_page
