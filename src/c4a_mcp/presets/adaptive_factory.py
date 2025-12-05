# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Adaptive Factory]
# :relates-to: uses: "crawl4ai.AdaptiveConfig", depends-on: "presets.models"
# :rationale: "Factory for creating AdaptiveConfig instances from validated parameters, isolating strategy creation for testability."
# :references: PRD: "F004 extension", SPEC: "SPEC-F004 extension"
# :contract: pre: "Valid strategy type and params", post: "Returns AdaptiveConfig instance"
# :decision_cache: "Factory pattern chosen to isolate AdaptiveConfig creation, enable testing without crawl4ai imports [ARCH-017]"
# LLM:END

"""
Factory for creating adaptive crawling configuration instances.

This module creates crawl4ai AdaptiveConfig objects from validated parameters
for both statistical and embedding strategies.
"""

import logging
from typing import Any

import crawl4ai

logger = logging.getLogger(__name__)


def create_adaptive_config(
    strategy: str,
    params: dict[str, Any],
) -> Any:  # Returns AdaptiveConfig, but avoiding import
    """
    Create an AdaptiveConfig instance from parameters.
    
    Args:
        strategy: Type of adaptive strategy ("statistical", "embedding")
        params: Parameters dictionary for the strategy
    
    Returns:
        AdaptiveConfig instance ready for AdaptiveCrawler
    
    Raises:
        ValueError: If strategy is invalid or params are malformed
        ImportError: If crawl4ai AdaptiveConfig cannot be imported
    """
    strategy_lower = strategy.lower()
    
    logger.debug(
        "[C4A-MCP | Presets | Adaptive Factory] Creating adaptive config | "
        "data: {strategy: %s, params: %s}",
        strategy_lower,
        params,
    )
    
    try:
        if strategy_lower == "statistical":
            return _create_statistical_config(params)
        elif strategy_lower == "embedding":
            return _create_embedding_config(params)
        else:
            raise ValueError(
                f"Unsupported adaptive strategy: {strategy}. "
                "Supported: 'statistical', 'embedding'"
            )
    except ImportError as e:
        logger.error(
            "[C4A-MCP | Presets | Adaptive Factory] Failed to import crawl4ai AdaptiveConfig | "
            "data: {error: %s}",
            str(e),
        )
        raise ImportError(
            "Failed to import crawl4ai AdaptiveConfig. "
            "Ensure crawl4ai is installed with adaptive crawling support."
        ) from e


def _create_statistical_config(params: dict[str, Any]) -> Any:
    """Create AdaptiveConfig for statistical strategy."""
    # Extract common adaptive parameters
    confidence_threshold = params.get("confidence_threshold", 0.7)
    max_pages = params.get("max_pages", 20)
    top_k_links = params.get("top_k_links", 3)
    min_gain_threshold = params.get("min_gain_threshold", 0.1)
    
    return crawl4ai.AdaptiveConfig(
        strategy="statistical",
        confidence_threshold=confidence_threshold,
        max_pages=max_pages,
        top_k_links=top_k_links,
        min_gain_threshold=min_gain_threshold,
    )


def _create_embedding_config(params: dict[str, Any]) -> Any:
    """Create AdaptiveConfig for embedding strategy."""
    # Extract common adaptive parameters
    confidence_threshold = params.get("confidence_threshold", 0.7)
    max_pages = params.get("max_pages", 20)
    top_k_links = params.get("top_k_links", 3)
    min_gain_threshold = params.get("min_gain_threshold", 0.1)
    
    # Extract embedding-specific parameters
    embedding_model = params.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_llm_config_dict = params.get("embedding_llm_config")
    n_query_variations = params.get("n_query_variations", 10)
    embedding_coverage_radius = params.get("embedding_coverage_radius", 0.2)
    embedding_k_exp = params.get("embedding_k_exp", 3.0)
    embedding_min_relative_improvement = params.get("embedding_min_relative_improvement", 0.1)
    embedding_validation_min_score = params.get("embedding_validation_min_score", 0.3)
    embedding_min_confidence_threshold = params.get("embedding_min_confidence_threshold", 0.1)
    embedding_overlap_threshold = params.get("embedding_overlap_threshold", 0.85)
    embedding_quality_min_confidence = params.get("embedding_quality_min_confidence", 0.7)
    embedding_quality_max_confidence = params.get("embedding_quality_max_confidence", 0.95)
    
    # Create LLMConfig if provided
    embedding_llm_config = None
    if embedding_llm_config_dict:
        if not isinstance(embedding_llm_config_dict, dict):
            raise ValueError(
                f"embedding_llm_config must be a dict, got: {type(embedding_llm_config_dict).__name__}"
            )
        if "provider" not in embedding_llm_config_dict:
            raise ValueError(
                "embedding_llm_config must contain 'provider' key. "
                "Required keys: provider, api_token (optional)"
            )
        embedding_llm_config = crawl4ai.LLMConfig(**embedding_llm_config_dict)
    
    return crawl4ai.AdaptiveConfig(
        strategy="embedding",
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
    )

