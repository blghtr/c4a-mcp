# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Crawling Factory]
# :relates-to: uses: "crawl4ai deep crawling strategies", depends-on: "presets.models"
# :rationale: "Factory for creating crawling strategy instances from validated parameters, isolating strategy creation for testability."
# :references: PRD: "F004", SPEC: "SPEC-F004"
# :contract: pre: "Valid strategy_type and params", post: "Returns DeepCrawlStrategy instance or None"
# :decision_cache: "Factory pattern chosen to isolate strategy creation, enable testing without crawl4ai imports, and centralize KeywordRelevanceScorer creation [ARCH-009]"
# LLM:END

"""
Factory for creating crawling strategy instances.

This module creates crawl4ai crawling strategies (BFSDeepCrawlStrategy,
BestFirstCrawlingStrategy) from validated parameters.
"""

import logging
from typing import Any

from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

logger = logging.getLogger(__name__)


def create_crawling_strategy(
    strategy_type: str, params: dict[str, Any]
) -> Any:  # Returns DeepCrawlStrategy | None, but avoiding import
    """
    Create a crawling strategy instance from parameters.

    Args:
        strategy_type: Type of crawling strategy ("bfs", "best_first", "none")
        params: Parameters dictionary for the strategy

    Returns:
        DeepCrawlStrategy instance or None for single-page scraping

    Raises:
        ValueError: If strategy_type is invalid or params are malformed
        ImportError: If crawl4ai crawling strategies cannot be imported
    """
    strategy_type_lower = strategy_type.lower()

    logger.debug(
        "[C4A-MCP | Presets | Crawling Factory] Creating crawling strategy | "
        "data: {strategy_type: %s, params: %s}",
        strategy_type_lower,
        params,
    )

    try:
        if strategy_type_lower == "bfs":
            return _create_bfs_strategy(params)
        elif strategy_type_lower == "best_first":
            return _create_best_first_strategy(params)
        elif strategy_type_lower == "none":
            return None  # No deep crawling for single-page scraping
        else:
            raise ValueError(
                f"Unsupported crawling strategy: {strategy_type}. "
                "Supported: 'bfs', 'best_first', 'none'"
            )
    except ImportError as e:
        logger.error(
            "[C4A-MCP | Presets | Crawling Factory] Failed to import crawl4ai strategies | "
            "data: {error: %s}",
            str(e),
        )
        raise ImportError(
            "Failed to import crawl4ai crawling strategies. "
            "Ensure crawl4ai is installed with required dependencies."
        ) from e


def _create_bfs_strategy(params: dict[str, Any]) -> Any:
    """Create BFSDeepCrawlStrategy from parameters."""
    max_depth = params.get("max_depth", 2)
    max_pages = params.get("max_pages", 50)
    include_external = params.get("include_external", False)
    score_threshold = params.get("score_threshold", 0.3)

    return BFSDeepCrawlStrategy(
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        score_threshold=score_threshold,
    )


def _create_best_first_strategy(params: dict[str, Any]) -> Any:
    """Create BestFirstCrawlingStrategy from parameters."""
    max_depth = params.get("max_depth", 2)
    max_pages = params.get("max_pages", 25)
    include_external = params.get("include_external", False)
    keywords = params.get("keywords", [])

    if not keywords:
        raise ValueError("keywords required for best_first crawling strategy")

    # Create KeywordRelevanceScorer
    # keywords must be positional argument per API signature
    scorer = KeywordRelevanceScorer(keywords, weight=0.7)
    
    # Note: Serialization/deserialization is handled in runner_tool._fix_keyword_scorer_deserialization()
    # to_serializable_dict() may not properly serialize _keywords, but we fix it during deserialization

    return BestFirstCrawlingStrategy(
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        url_scorer=scorer,
    )



