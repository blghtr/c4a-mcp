# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Tools]
# :relates-to: uses: "fastmcp.Context", uses: "runner_tool.CrawlRunner"
# :rationale: "MCP tool implementations for preset crawling patterns using Context injection pattern. Passes strategy parameters (not objects) for JSON-serializable configuration."
# :references: PRD: "F004", SPEC: "SPEC-F004"
# :contract: pre: "Valid PresetInput models, Context with crawl_runner in state", post: "Returns RunnerOutput JSON string with config containing strategy_params"
# :decision_cache: "Migrated from factory pattern to Context injection for FastMCP standalone compatibility [ARCH-013]. Refactored to parameterized configuration to avoid serialization issues [ARCH-010]"
# LLM:END

"""
MCP tool implementations for preset crawling patterns.

These tools provide high-level interfaces for common crawling scenarios:
- crawl_deep: BFS deep crawling
- crawl_deep_smart: Best-first crawling with keywords
- scrape_page: Single-page scraping

All tools use Context injection to access CrawlRunner from lifespan state.
"""

import json
import logging
from typing import Any

from fastmcp import Context

logger = logging.getLogger(__name__)

from ..adaptive_runner import AdaptiveCrawlRunner, AdaptiveRunnerInput
from ..models import RunnerInput
from .models import (
    AdaptiveEmbeddingInput,
    AdaptiveStatisticalInput,
    CrawlDeepSmartInput,
    DeepCrawlPresetInput,
    ExtractionConfig,
    PresetBaseConfig,
    ScrapePagePresetInput,
)

logger = logging.getLogger(__name__)


def _build_run_config_from_preset(
    preset_config: PresetBaseConfig,
    crawling_strategy_params: dict[str, Any] | None,
    extraction_strategy_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build config dict with strategy parameters (not serialized objects).

    Args:
        preset_config: PresetBaseConfig with all parameters
        crawling_strategy_params: Dict with 'strategy_type' and strategy-specific params
        extraction_strategy_params: Dict with 'strategy_type' and extraction config

    Returns:
        Config dict with strategy parameters (JSON-serializable, standard Python types only)
    """
    # Convert preset config to CrawlerRunConfig kwargs
    kwargs = preset_config.to_crawler_run_config_kwargs()

    # Remove extraction_strategy fields - we use extraction_strategy_params instead
    # These fields would cause validation errors in CrawlerConfigYAML (only "jsoncss" is allowed)
    kwargs.pop("extraction_strategy", None)
    kwargs.pop("extraction_strategy_config", None)

    # Add strategy parameters (not objects) - these will be used to create strategies at execution time
    if crawling_strategy_params:
        kwargs["deep_crawl_strategy_params"] = crawling_strategy_params
    if extraction_strategy_params:
        kwargs["extraction_strategy_params"] = extraction_strategy_params

    # Return plain dict (no serialization needed - all values are JSON-serializable)
    return kwargs


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

    # Prepare crawling strategy parameters (don't create strategy object)
    crawling_strategy_params = {
        "strategy_type": "bfs",
        "max_depth": input_data.max_depth,
        "max_pages": input_data.max_pages,
        "include_external": input_data.include_external,
    }

    # Prepare extraction strategy parameters (don't create strategy object)
    extraction_strategy_params = None
    if input_data.extraction_strategy:
        extraction_strategy_params = {
            "strategy_type": input_data.extraction_strategy,
            "config": input_data.extraction_strategy_config,
        }

    # Build preset config
    preset_config = PresetBaseConfig(
        **input_data.model_dump(exclude={"url", "max_depth", "max_pages", "include_external"})
    )

    # Build config dict with strategy parameters (JSON-serializable)
    config_dict = _build_run_config_from_preset(
        preset_config, crawling_strategy_params, extraction_strategy_params
    )

    # Execute via CrawlRunner
    runner_input = RunnerInput(
        url=url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()


async def adaptive_crawl_statistical(
    url: str,
    query: str,
    confidence_threshold: float = 0.7,
    max_pages: int = 20,
    top_k_links: int = 3,
    min_gain_threshold: float = 0.1,
    config: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> str:
    """
    Perform adaptive crawling using statistical strategy.
    
    Uses term-based analysis and information theory to determine when
    sufficient information has been gathered. Fast, efficient, no external
    dependencies. Best for well-defined queries with specific terminology.
    
    Args:
        url: Starting URL for the adaptive crawl
        query: Query string for relevance-based crawling (required)
        confidence_threshold: Stop when confidence reaches this threshold (0.0-1.0, default: 0.7)
        max_pages: Maximum number of pages to crawl (default: 20)
        top_k_links: Number of links to follow per page (default: 3)
        min_gain_threshold: Minimum expected information gain to continue (default: 0.1)
        config: Additional configuration parameters (timeout, css_selector, etc.)
        ctx: FastMCP Context (injected automatically)
    
    Returns:
        JSON string with RunnerOutput structure (markdown, metadata with confidence/metrics, error)
    """
    # Access adaptive_crawl_runner from lifespan state
    if ctx is None:
        raise ValueError("Context is required for preset tools")
    adaptive_runner = ctx.get_state("adaptive_crawl_runner")
    if adaptive_runner is None:
        raise ValueError(
            "adaptive_crawl_runner not found in context state. "
            "Ensure the server lifespan properly initializes adaptive_crawl_runner."
        )
    
    # Validate input
    input_data = AdaptiveStatisticalInput(
        url=url,
        query=query,
        confidence_threshold=confidence_threshold,
        max_pages=max_pages,
        top_k_links=top_k_links,
        min_gain_threshold=min_gain_threshold,
        **(config or {}),
    )
    
    # Prepare adaptive config parameters
    adaptive_config_params = {
        "confidence_threshold": input_data.confidence_threshold,
        "max_pages": input_data.max_pages,
        "top_k_links": input_data.top_k_links,
        "min_gain_threshold": input_data.min_gain_threshold,
    }
    
    # Build config dict with strategy and adaptive config params
    # Include timeout and other common config parameters
    config_dict: dict[str, Any] = {
        "strategy": "statistical",
        "adaptive_config_params": adaptive_config_params,
    }
    
    # Pass through timeout and other config parameters if provided
    if input_data.timeout is not None:
        config_dict["timeout"] = int(input_data.timeout)
    
    # Execute via AdaptiveCrawlRunner
    runner_input = AdaptiveRunnerInput(
        url=url, query=query, config=config_dict
    )
    result = await adaptive_runner.run(runner_input)
    
    return result.model_dump_json()


async def adaptive_crawl_embedding(
    url: str,
    query: str,
    confidence_threshold: float = 0.7,
    max_pages: int = 20,
    top_k_links: int = 3,
    min_gain_threshold: float = 0.1,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_llm_config: dict[str, Any] | None = None,
    n_query_variations: int = 10,
    embedding_coverage_radius: float = 0.2,
    embedding_k_exp: float = 3.0,
    embedding_min_relative_improvement: float = 0.1,
    embedding_validation_min_score: float = 0.3,
    embedding_min_confidence_threshold: float = 0.1,
    embedding_overlap_threshold: float = 0.85,
    embedding_quality_min_confidence: float = 0.7,
    embedding_quality_max_confidence: float = 0.95,
    config: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> str:
    """
    Perform adaptive crawling using embedding strategy.
    
    Uses semantic embeddings for deeper understanding. Captures meaning
    beyond exact term matches. Requires embedding model or LLM API.
    Best for complex queries, ambiguous topics, conceptual understanding.
    
    Args:
        url: Starting URL for the adaptive crawl
        query: Query string for relevance-based crawling (required)
        confidence_threshold: Stop when confidence reaches this threshold (0.0-1.0, default: 0.7)
        max_pages: Maximum number of pages to crawl (default: 20)
        top_k_links: Number of links to follow per page (default: 3)
        min_gain_threshold: Minimum expected information gain to continue (default: 0.1)
        embedding_model: Embedding model identifier (default: sentence-transformers/all-MiniLM-L6-v2)
        embedding_llm_config: LLM config for query expansion: {provider: str, api_token: str, ...}
        n_query_variations: Number of query variations to generate (default: 10)
        embedding_coverage_radius: Distance threshold for semantic coverage (default: 0.2)
        embedding_k_exp: Exponential decay factor for coverage (default: 3.0)
        embedding_min_relative_improvement: Minimum relative improvement to continue (default: 0.1)
        embedding_validation_min_score: Minimum validation score threshold (default: 0.3)
        embedding_min_confidence_threshold: Below this confidence = irrelevant (default: 0.1)
        embedding_overlap_threshold: Similarity threshold for deduplication (default: 0.85)
        embedding_quality_min_confidence: Minimum confidence for quality display (default: 0.7)
        embedding_quality_max_confidence: Maximum confidence for quality display (default: 0.95)
        config: Additional configuration parameters (timeout, css_selector, etc.)
        ctx: FastMCP Context (injected automatically)
    
    Returns:
        JSON string with RunnerOutput structure (markdown, metadata with confidence/metrics, error)
    """
    # Access adaptive_crawl_runner from lifespan state
    # Check context FIRST before checking dependencies
    if ctx is None:
        raise ValueError("Context is required for preset tools")
    adaptive_runner = ctx.get_state("adaptive_crawl_runner")
    if adaptive_runner is None:
        raise ValueError(
            "adaptive_crawl_runner not found in context state. "
            "Ensure the server lifespan properly initializes adaptive_crawl_runner."
        )
    
    # Check if sentence-transformers is available when using local embeddings
    # This check happens AFTER context validation to allow proper error handling
    if embedding_llm_config is None:
        try:
            import sentence_transformers  # noqa: F401
            logger.info(
                "[C4A-MCP | Presets | Tools] sentence-transformers available, "
                "will use local embeddings. First model load may take time."
            )
        except ImportError:
            return json.dumps({
                "markdown": "",
                "metadata": {},
                "error": (
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: uv pip install --group embeddings or "
                    "pip install sentence-transformers"
                )
            })
    
    # Validate input
    input_data = AdaptiveEmbeddingInput(
        url=url,
        query=query,
        confidence_threshold=confidence_threshold,
        max_pages=max_pages,
        top_k_links=top_k_links,
        min_gain_threshold=min_gain_threshold,
        embedding_model=embedding_model,
        embedding_llm_config=embedding_llm_config,
        n_query_variations=n_query_variations,
        embedding_coverage_radius=embedding_coverage_radius,
        embedding_k_exp=embedding_k_exp,
        embedding_min_relative_improvement=embedding_min_relative_improvement,
        embedding_validation_min_score=embedding_validation_min_score,
        embedding_min_confidence_threshold=embedding_min_confidence_threshold,
        embedding_overlap_threshold=embedding_overlap_threshold,
        embedding_quality_min_confidence=embedding_quality_min_confidence,
        embedding_quality_max_confidence=embedding_quality_max_confidence,
        **(config or {}),
    )
    
    # Prepare adaptive config parameters
    adaptive_config_params = {
        "confidence_threshold": input_data.confidence_threshold,
        "max_pages": input_data.max_pages,
        "top_k_links": input_data.top_k_links,
        "min_gain_threshold": input_data.min_gain_threshold,
        "embedding_model": input_data.embedding_model,
        "embedding_llm_config": input_data.embedding_llm_config,
        "n_query_variations": input_data.n_query_variations,
        "embedding_coverage_radius": input_data.embedding_coverage_radius,
        "embedding_k_exp": input_data.embedding_k_exp,
        "embedding_min_relative_improvement": input_data.embedding_min_relative_improvement,
        "embedding_validation_min_score": input_data.embedding_validation_min_score,
        "embedding_min_confidence_threshold": input_data.embedding_min_confidence_threshold,
        "embedding_overlap_threshold": input_data.embedding_overlap_threshold,
        "embedding_quality_min_confidence": input_data.embedding_quality_min_confidence,
        "embedding_quality_max_confidence": input_data.embedding_quality_max_confidence,
    }
    
    # Build config dict with strategy and adaptive config params
    # Include timeout and other common config parameters
    config_dict: dict[str, Any] = {
        "strategy": "embedding",
        "adaptive_config_params": adaptive_config_params,
    }
    
    # Pass through timeout and other config parameters if provided
    if input_data.timeout is not None:
        config_dict["timeout"] = int(input_data.timeout)
    
    # Execute via AdaptiveCrawlRunner
    runner_input = AdaptiveRunnerInput(
        url=url, query=query, config=config_dict
    )
    result = await adaptive_runner.run(runner_input)
    
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

    # Prepare crawling strategy parameters (don't create strategy object)
    crawling_strategy_params = {
        "strategy_type": "best_first",
        "max_depth": input_data.max_depth,
        "max_pages": input_data.max_pages,
        "include_external": input_data.include_external,
        "keywords": input_data.keywords,
    }

    # Prepare extraction strategy parameters (don't create strategy object)
    extraction_strategy_params = None
    if input_data.extraction_strategy:
        extraction_strategy_params = {
            "strategy_type": input_data.extraction_strategy,
            "config": input_data.extraction_strategy_config,
        }

    # Build preset config
    preset_config = PresetBaseConfig(
        **input_data.model_dump(
            exclude={"url", "keywords", "max_depth", "max_pages", "include_external"}
        )
    )

    # Build config dict with strategy parameters (JSON-serializable)
    config_dict = _build_run_config_from_preset(
        preset_config, crawling_strategy_params, extraction_strategy_params
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
    crawling_strategy_params = None

    # Prepare extraction strategy parameters (don't create strategy object)
    extraction_strategy_params = None
    if input_data.extraction_strategy:
        extraction_strategy_params = {
            "strategy_type": input_data.extraction_strategy,
            "config": input_data.extraction_strategy_config,
        }

    # Build preset config
    preset_config = PresetBaseConfig(**input_data.model_dump(exclude={"url"}))

    # Build config dict with strategy parameters (JSON-serializable)
    config_dict = _build_run_config_from_preset(
        preset_config, crawling_strategy_params, extraction_strategy_params
    )

    # Execute via CrawlRunner
    runner_input = RunnerInput(
        url=url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()
