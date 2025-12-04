# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Tools]
# :relates-to: uses: "fastmcp.Context", uses: "runner_tool.CrawlRunner", uses: "presets.extraction_factory", uses: "presets.crawling_factory"
# :rationale: "MCP tool implementations for preset crawling patterns using Context injection pattern."
# :references: PRD: "F004", SPEC: "SPEC-F004"
# :contract: pre: "Valid PresetInput models, Context with crawl_runner in state", post: "Returns RunnerOutput JSON string"
# :decision_cache: "Migrated from factory pattern to Context injection for FastMCP standalone compatibility [ARCH-013]"
# LLM:END

"""
MCP tool implementations for preset crawling patterns.

These tools provide high-level interfaces for common crawling scenarios:
- crawl_deep: BFS deep crawling
- crawl_deep_smart: Best-first crawling with keywords
- scrape_page: Single-page scraping

All tools use Context injection to access CrawlRunner from lifespan state.
"""

import logging
from typing import Any

from crawl4ai import CrawlerRunConfig
from fastmcp import Context

from ..models import RunnerInput
from .crawling_factory import create_crawling_strategy
from .extraction_factory import create_extraction_strategy
from .models import (
    CrawlDeepSmartInput,
    DeepCrawlPresetInput,
    ExtractionConfig,
    PresetBaseConfig,
    ScrapePagePresetInput,
)

logger = logging.getLogger(__name__)


def _build_run_config_from_preset(
    preset_config: PresetBaseConfig,
    crawling_strategy: Any | None,
    extraction_strategy: Any | None,
) -> dict[str, Any]:
    """Build serialized config dict from preset configuration and strategies.

    Uses CrawlerRunConfig.dump() to properly serialize strategies using
    crawl4ai's to_serializable_dict() mechanism.

    Args:
        preset_config: PresetBaseConfig with all parameters
        crawling_strategy: DeepCrawlStrategy instance or None
        extraction_strategy: ExtractionStrategy instance or None

    Returns:
        Serialized dict ready for RunnerInput.config (can be deserialized with CrawlerRunConfig.load())
    """
    # Convert preset config to CrawlerRunConfig kwargs
    kwargs = preset_config.to_crawler_run_config_kwargs()

    # Remove extraction_strategy_config - it's not a CrawlerRunConfig parameter
    kwargs.pop("extraction_strategy_config", None)

    # Add strategies to kwargs
    if crawling_strategy:
        kwargs["deep_crawl_strategy"] = crawling_strategy
    if extraction_strategy:
        kwargs["extraction_strategy"] = extraction_strategy

    # Create CrawlerRunConfig and serialize it using dump()
    # This properly handles strategy serialization via to_serializable_dict()
    run_config = CrawlerRunConfig(**kwargs)
    return run_config.dump()


async def crawl_deep(
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    include_external: bool = False,
    script: str | None = None,
    extraction_strategy: str | None = None,
    extraction_strategy_config: ExtractionConfig | None = None,
    config: dict[str, Any] | None = None,
    ctx: Context | None = None,
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
        extraction_strategy: Extraction strategy type ("regex", "css", or None)
        extraction_strategy_config: Configuration for extraction strategy
        config: Additional CrawlerRunConfig parameters (timeout, css_selector, etc.)

    Returns:
        JSON string with RunnerOutput structure (markdown, metadata, error)
    """
    # Access crawl_runner from lifespan state
    if ctx is None:
        raise ValueError("Context is required for preset tools")
    crawl_runner = ctx.get_state("crawl_runner")
    if crawl_runner is None:
        raise ValueError(
            "crawl_runner not found in context state. "
            "Ensure the server lifespan properly initializes crawl_runner."
        )

    # Validate input
    input_data = DeepCrawlPresetInput(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        extraction_strategy=extraction_strategy,
        extraction_strategy_config=extraction_strategy_config,
        **(config or {}),
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
    preset_config = PresetBaseConfig(
        **input_data.model_dump(exclude={"url", "max_depth", "max_pages", "include_external"})
    )

    # Build serialized config dict using CrawlerRunConfig.dump()
    # This properly serializes strategies via to_serializable_dict()
    config_dict = _build_run_config_from_preset(
        preset_config, crawling_strategy, extraction_strategy_instance
    )

    # Execute via CrawlRunner
    runner_input = RunnerInput(
        url=url, script=script, config=config_dict
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
    extraction_strategy_config: ExtractionConfig | None = None,
    config: dict[str, Any] | None = None,
    ctx: Context | None = None,
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
        extraction_strategy: Extraction strategy type ("regex", "css", or None)
        extraction_strategy_config: Configuration for extraction strategy
        config: Additional CrawlerRunConfig parameters

    Returns:
        JSON string with RunnerOutput structure (markdown, metadata, error)
    """
    # Access crawl_runner from lifespan state
    if ctx is None:
        raise ValueError("Context is required for preset tools")
    crawl_runner = ctx.get_state("crawl_runner")
    if crawl_runner is None:
        raise ValueError(
            "crawl_runner not found in context state. "
            "Ensure the server lifespan properly initializes crawl_runner."
        )

    # Validate input
    input_data = CrawlDeepSmartInput(
        url=url,
        keywords=keywords,
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        extraction_strategy=extraction_strategy,
        extraction_strategy_config=extraction_strategy_config,
        **(config or {}),
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
    preset_config = PresetBaseConfig(
        **input_data.model_dump(
            exclude={"url", "keywords", "max_depth", "max_pages", "include_external"}
        )
    )

    # Build serialized config dict using CrawlerRunConfig.dump()
    # This properly serializes strategies via to_serializable_dict()
    config_dict = _build_run_config_from_preset(
        preset_config, crawling_strategy, extraction_strategy_instance
    )

    # Execute via CrawlRunner
    runner_input = RunnerInput(
        url=url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()


async def scrape_page(
    url: str,
    script: str | None = None,
    extraction_strategy: str | None = None,
    extraction_strategy_config: ExtractionConfig | None = None,
    config: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> str:
    """
    Scrape content from a single page.

    Extracts content from only the specified page. JavaScript rendering is
    controlled by BrowserConfig set at server startup (see server.py).

    Args:
        url: URL of the page to scrape
        script: Optional c4a-script DSL or JavaScript code for page interactions
        extraction_strategy: Extraction strategy type ("regex", "css", or None)
        extraction_strategy_config: Configuration for extraction strategy
        config: Additional CrawlerRunConfig parameters

    Returns:
        JSON string with RunnerOutput structure (markdown, metadata, error)
    """
    # Access crawl_runner from lifespan state
    if ctx is None:
        raise ValueError("Context is required for preset tools")
    crawl_runner = ctx.get_state("crawl_runner")
    if crawl_runner is None:
        raise ValueError(
            "crawl_runner not found in context state. "
            "Ensure the server lifespan properly initializes crawl_runner."
        )

    # Validate input
    input_data = ScrapePagePresetInput(
        url=url,
        extraction_strategy=extraction_strategy,
        extraction_strategy_config=extraction_strategy_config,
        **(config or {}),
    )

    # No deep crawling for single-page scraping
    crawling_strategy = None

    # Create extraction strategy
    extraction_strategy_instance = create_extraction_strategy(
        input_data.extraction_strategy, input_data.extraction_strategy_config
    )

    # Build preset config
    preset_config = PresetBaseConfig(**input_data.model_dump(exclude={"url"}))

    # Build serialized config dict using CrawlerRunConfig.dump()
    # This properly serializes strategies via to_serializable_dict()
    config_dict = _build_run_config_from_preset(
        preset_config, crawling_strategy, extraction_strategy_instance
    )

    # Execute via CrawlRunner
    runner_input = RunnerInput(
        url=url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()
