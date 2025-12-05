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

import importlib.util
import json
import logging
from typing import Any, Annotated

from fastmcp import Context
from pydantic import Field

logger = logging.getLogger(__name__)

from ..models import RunnerInput
from .models import (
    AdaptiveEmbeddingInput,
    AdaptiveStatisticalInput,
    CrawlDeepSmartInput,
    DeepCrawlPresetInput,
    PresetBaseConfig,
    ScrapePagePresetInput,
)


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
    params: Annotated[
        DeepCrawlPresetInput,
        Field(description="Validated deep crawl parameters (url, depth, pages, config fields)"),
    ],
    script: Annotated[
        str | None,
        Field(
            description=(
                "Optional c4a-script DSL or JavaScript code for page interactions "
                "(GO, WAIT, CLICK, etc.)"
            )
        ),
    ] = None,
    ctx: Context | None = None,
) -> str:
    """
    Perform BFS deep crawling of a website.

    Pydantic payload (`DeepCrawlPresetInput`) — все ключи доступны в схеме:
    - url (http/https), max_depth, max_pages, include_external
    - extraction_strategy: regex | css | None
    - extraction_strategy_config:
        * regex: built_in_patterns | custom_patterns, input_format
          built_in_patterns допустимы: Email, PhoneUS, PhoneIntl, Url, IPv4, IPv6,
          Uuid, Currency, Percentage, Number, DateIso, DateUS, Time24h, PostalUS,
          PostalUK, HexColor, TwitterHandle, Hashtag, MacAddr, Iban, CreditCard, All.
        * css: schema/extraction_schema {name, baseSelector, fields}
    - content/navigation: timeout, css_selector, word_count_threshold, wait_for,
      excluded_tags/excluded_selector, only_text, remove_forms, process_iframes,
      exclude_external_links, exclude_social_media_links, wait_until,
      wait_for_images, delay_before_return_html, check_robots_txt, scan_full_page,
      scroll_delay, remove_overlay_elements, simulate_user, override_navigator,
      magic, bypass_cache
    script: опциональный c4a-script/JS для взаимодействия.
    ctx: FastMCP context (инжектируется).

    Returns: JSON RunnerOutput (markdown, metadata, error).
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

    # Normalize/validate input
    input_data = (
        params
        if isinstance(params, DeepCrawlPresetInput)
        else DeepCrawlPresetInput.model_validate(params)
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
        url=input_data.url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()


async def adaptive_crawl_statistical(
    params: Annotated[
        AdaptiveStatisticalInput,
        Field(description="Adaptive crawl parameters (url, query, thresholds, config)"),
    ],
    ctx: Context | None = None,
) -> str:
    """
    Perform adaptive crawling using statistical strategy with structured params.
    
    Параметры (доступны в схеме):
    - url, query
    - confidence_threshold, max_pages, top_k_links, min_gain_threshold
    - content/navigation: timeout, css_selector, word_count_threshold, wait_for,
      exclude_external_links, exclude_social_media_links, bypass_cache
    - config (опционально): strategy/ adaptive_config_params/ timeout override
    ctx: FastMCP Context.
    
    Returns: JSON RunnerOutput (markdown, metadata c confidence/metrics, error).
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
    
    # Normalize/validate input
    input_data = (
        params
        if isinstance(params, AdaptiveStatisticalInput)
        else AdaptiveStatisticalInput.model_validate(params)
    )
    
    # Prepare adaptive config parameters
    adaptive_config_params = {
        "confidence_threshold": input_data.confidence_threshold,
        "max_pages": input_data.max_pages,
        "top_k_links": input_data.top_k_links,
        "min_gain_threshold": input_data.min_gain_threshold,
    }
    
    # Build config dict with strategy and adaptive config params
    # Include any caller-provided config (fields already consolidated in model)
    config_dict: dict[str, Any] = input_data.config.copy() if input_data.config else {}
    config_dict.update(
        {
            "strategy": "statistical",
            "adaptive_config_params": adaptive_config_params,
        }
    )
    
    # Execute via AdaptiveCrawlRunner (accepts Pydantic input directly)
    result = await adaptive_runner.run(
        input_data.model_copy(update={"config": config_dict})
    )
    
    return result.model_dump_json()


async def adaptive_crawl_embedding(
    params: Annotated[
        AdaptiveEmbeddingInput,
        Field(description="Adaptive crawl parameters (url, query, embedding, thresholds)"),
    ],
    ctx: Context | None = None,
) -> str:
    """
    Perform adaptive crawling using embedding strategy with structured params.
    
    Параметры (доступны в схеме):
    - url, query
    - confidence_threshold, max_pages, top_k_links, min_gain_threshold, timeout
    - embedding_model, embedding_llm_config, n_query_variations,
      embedding_coverage_radius, embedding_k_exp,
      embedding_min_relative_improvement, embedding_validation_min_score,
      embedding_min_confidence_threshold, embedding_overlap_threshold,
      embedding_quality_min_confidence, embedding_quality_max_confidence
    - content/navigation: css_selector, word_count_threshold, wait_for,
      exclude_external_links, exclude_social_media_links, bypass_cache
    - config (опционально): strategy/ adaptive_config_params/ timeout override
    ctx: FastMCP Context.
    
    Returns: JSON RunnerOutput (markdown, metadata c confidence/metrics, error).
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
    
    # Normalize/validate input
    input_data = (
        params
        if isinstance(params, AdaptiveEmbeddingInput)
        else AdaptiveEmbeddingInput.model_validate(params)
    )

    # Check if sentence-transformers is available when using local embeddings
    # This check happens AFTER validation so ValueError surfaces for bad input first
    if input_data.embedding_llm_config is None:
        if importlib.util.find_spec("sentence_transformers") is None:
            return json.dumps({
                "markdown": "",
                "metadata": {},
                "error": (
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: uv pip install --group embeddings or "
                    "pip install sentence-transformers"
                )
            })
        logger.info(
            "[C4A-MCP | Presets | Tools] sentence-transformers available, "
            "will use local embeddings. First model load may take time."
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
    # Include any caller-provided config (fields already consolidated in model)
    config_dict: dict[str, Any] = input_data.config.copy() if input_data.config else {}
    config_dict.update(
        {
            "strategy": "embedding",
            "adaptive_config_params": adaptive_config_params,
        }
    )
    
    # Execute via AdaptiveCrawlRunner (accepts Pydantic input directly)
    result = await adaptive_runner.run(
        input_data.model_copy(update={"config": config_dict})
    )
    
    return result.model_dump_json()


async def crawl_deep_smart(
    params: Annotated[
        CrawlDeepSmartInput,
        Field(description="Smart deep crawl parameters (url, keywords, config)"),
    ],
    script: Annotated[
        str | None,
        Field(description="Optional c4a-script DSL or JavaScript code for page interactions"),
    ] = None,
    ctx: Context | None = None,
) -> str:
    """
    Perform best-first deep crawling with keyword prioritization.

    Параметры (схема):
    - url, keywords, max_depth, max_pages, include_external
    - extraction_strategy + extraction_strategy_config (regex/css, как в crawl_deep)
      * regex built_in_patterns допустимы: Email, PhoneUS, PhoneIntl, Url, IPv4, IPv6,
        Uuid, Currency, Percentage, Number, DateIso, DateUS, Time24h, PostalUS, PostalUK,
        HexColor, TwitterHandle, Hashtag, MacAddr, Iban, CreditCard, All.
      * css: schema/extraction_schema {name, baseSelector, fields}
    - content/navigation: timeout, css_selector, word_count_threshold, wait_for,
      excluded_tags/excluded_selector, only_text, remove_forms, process_iframes,
      exclude_external_links, exclude_social_media_links, wait_until,
      wait_for_images, delay_before_return_html, check_robots_txt, scan_full_page,
      scroll_delay, remove_overlay_elements, simulate_user, override_navigator,
      magic, bypass_cache
    script: опциональный c4a-script/JS.
    ctx: FastMCP context.

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

    # Normalize/validate input
    input_data = (
        params
        if isinstance(params, CrawlDeepSmartInput)
        else CrawlDeepSmartInput.model_validate(params)
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
        url=input_data.url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()


async def scrape_page(
    params: Annotated[
        ScrapePagePresetInput,
        Field(description="Scrape parameters (url plus optional extraction config)"),
    ],
    script: Annotated[
        str | None,
        Field(description="Optional c4a-script DSL or JavaScript code for page interactions"),
    ] = None,
    ctx: Context | None = None,
) -> str:
    """
    Scrape content from a single page.

    Параметры (схема):
    - url
    - extraction_strategy + extraction_strategy_config (regex/css)
      * regex built_in_patterns: Email, PhoneUS, PhoneIntl, Url, IPv4, IPv6, Uuid,
        Currency, Percentage, Number, DateIso, DateUS, Time24h, PostalUS, PostalUK,
        HexColor, TwitterHandle, Hashtag, MacAddr, Iban, CreditCard, All.
      * css: schema/extraction_schema {name, baseSelector, fields}
    - content/navigation: timeout, css_selector, word_count_threshold, wait_for,
      excluded_tags/excluded_selector, only_text, remove_forms, process_iframes,
      exclude_external_links, exclude_social_media_links, wait_until,
      wait_for_images, delay_before_return_html, check_robots_txt, scan_full_page,
      scroll_delay, remove_overlay_elements, simulate_user, override_navigator,
      magic, bypass_cache
    script: опциональный c4a-script/JS.
    ctx: FastMCP context.

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

    # Normalize/validate input
    input_data = (
        params
        if isinstance(params, ScrapePagePresetInput)
        else ScrapePagePresetInput.model_validate(params)
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
        url=input_data.url, script=script, config=config_dict
    )
    result = await crawl_runner.run(runner_input)

    return result.model_dump_json()
