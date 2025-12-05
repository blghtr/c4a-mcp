# LLM:METADATA
# :hierarchy: [C4A-MCP | Tests | Crawling Factory]
# :relates-to: tests: "presets.crawling_factory", implements: "SPEC-F003 extension"
# :rationale: "Validates crawling strategy factory creates correct strategy instances from parameters."
# :contract: pre: "Valid strategy_type and params", post: "Returns correct DeepCrawlStrategy instance"
# :decision_cache: "Unit tests with mocks to avoid crawl4ai imports in test environment [TEST-003]"
# LLM:END

"""
Tests for crawling strategy factory.

Validates that crawling strategies are created correctly from
parameter dictionaries.
"""

import pytest
from unittest.mock import MagicMock, patch

from c4a_mcp.presets.crawling_factory import create_crawling_strategy


@patch("c4a_mcp.presets.crawling_factory._create_bfs_strategy")
def test_create_crawling_strategy_bfs(mock_create_bfs):
    """Test creating BFS crawling strategy."""
    mock_strategy = MagicMock()
    mock_create_bfs.return_value = mock_strategy

    params = {"max_depth": 2, "max_pages": 50, "include_external": False}
    result = create_crawling_strategy("bfs", params)

    assert result == mock_strategy
    mock_create_bfs.assert_called_once_with(params)


@patch("c4a_mcp.presets.crawling_factory._create_best_first_strategy")
def test_create_crawling_strategy_best_first(mock_create_best_first):
    """Test creating best-first crawling strategy."""
    mock_strategy = MagicMock()
    mock_create_best_first.return_value = mock_strategy

    params = {
        "max_depth": 2,
        "max_pages": 25,
        "include_external": False,
        "keywords": ["test", "example"],
    }
    result = create_crawling_strategy("best_first", params)

    assert result == mock_strategy
    mock_create_best_first.assert_called_once_with(params)




def test_create_crawling_strategy_none():
    """Test creating crawling strategy with 'none' type."""
    result = create_crawling_strategy("none", {})
    assert result is None


def test_create_crawling_strategy_invalid_type():
    """Test creating crawling strategy with invalid type."""
    with pytest.raises(ValueError) as exc_info:
        create_crawling_strategy("invalid", {})
    assert "Unsupported crawling strategy" in str(exc_info.value)


def test_create_crawling_strategy_case_insensitive():
    """Test that crawling strategy type is case-insensitive."""
    with patch("c4a_mcp.presets.crawling_factory._create_bfs_strategy") as mock_create:
        mock_create.return_value = MagicMock()
        create_crawling_strategy("BFS", {"max_depth": 2})
        mock_create.assert_called_once()

    with patch("c4a_mcp.presets.crawling_factory._create_best_first_strategy") as mock_create:
        mock_create.return_value = MagicMock()
        create_crawling_strategy("BEST_FIRST", {"keywords": ["test"]})
        mock_create.assert_called_once()


@patch("c4a_mcp.presets.crawling_factory.BFSDeepCrawlStrategy")
def test_create_bfs_strategy(mock_bfs_class):
    """Test creating BFS strategy with parameters."""
    mock_strategy = MagicMock()
    mock_bfs_class.return_value = mock_strategy

    from c4a_mcp.presets.crawling_factory import _create_bfs_strategy

    params = {
        "max_depth": 3,
        "max_pages": 100,
        "include_external": True,
        "score_threshold": 0.5,
    }
    result = _create_bfs_strategy(params)

    assert result == mock_strategy
    mock_bfs_class.assert_called_once_with(
        max_depth=3, max_pages=100, include_external=True, score_threshold=0.5
    )


@patch("c4a_mcp.presets.crawling_factory.BFSDeepCrawlStrategy")
def test_create_bfs_strategy_defaults(mock_bfs_class):
    """Test creating BFS strategy with default parameters."""
    mock_strategy = MagicMock()
    mock_bfs_class.return_value = mock_strategy

    from c4a_mcp.presets.crawling_factory import _create_bfs_strategy

    params = {}
    result = _create_bfs_strategy(params)

    assert result == mock_strategy
    mock_bfs_class.assert_called_once_with(
        max_depth=2, max_pages=50, include_external=False, score_threshold=float("-inf")
    )


@patch("c4a_mcp.presets.crawling_factory.BestFirstCrawlingStrategy")
@patch("c4a_mcp.presets.crawling_factory.KeywordRelevanceScorer")
def test_create_best_first_strategy(mock_scorer_class, mock_best_first_class):
    """Test creating best-first strategy with keywords."""
    mock_strategy = MagicMock()
    mock_best_first_class.return_value = mock_strategy
    mock_scorer = MagicMock()
    mock_scorer_class.return_value = mock_scorer

    from c4a_mcp.presets.crawling_factory import _create_best_first_strategy

    params = {
        "max_depth": 2,
        "max_pages": 25,
        "include_external": False,
        "keywords": ["test", "example"],
    }
    result = _create_best_first_strategy(params)

    assert result == mock_strategy
    # KeywordRelevanceScorer is called with keywords as positional arg
    mock_scorer_class.assert_called_once_with(["test", "example"], weight=0.7)
    mock_best_first_class.assert_called_once_with(
        max_depth=2,
        max_pages=25,
        include_external=False,
        url_scorer=mock_scorer,
    )


def test_create_best_first_strategy_no_keywords():
    """Test creating best-first strategy without keywords (invalid)."""
    from c4a_mcp.presets.crawling_factory import _create_best_first_strategy

    params = {"max_depth": 2, "max_pages": 25}
    with pytest.raises(ValueError) as exc_info:
        _create_best_first_strategy(params)
    assert "keywords required" in str(exc_info.value)



