# LLM:METADATA
# :hierarchy: [C4A-MCP | Tests | Adaptive Factory]
# :relates-to: tests: "presets.adaptive_factory", implements: "SPEC-F004 extension"
# :rationale: "Validates adaptive config factory, ensuring AdaptiveConfig instances are created correctly from parameters."
# :contract: pre: "Valid strategy type and params", post: "All factory functions return AdaptiveConfig instances"
# :decision_cache: "Unit tests with mocked crawl4ai imports to verify factory logic without dependencies [TEST-006]"
# LLM:END

"""
Tests for adaptive crawling configuration factory.

Validates that adaptive_factory correctly creates AdaptiveConfig instances
from parameters for both statistical and embedding strategies.
"""

import pytest
from unittest.mock import MagicMock, patch

from c4a_mcp.presets.adaptive_factory import (
    create_adaptive_config,
    _create_statistical_config,
    _create_embedding_config,
)


# --- Test Fixtures ---
@pytest.fixture
def mock_adaptive_config():
    """Create a mock AdaptiveConfig for testing."""
    return MagicMock()


@pytest.fixture
def mock_llm_config():
    """Create a mock LLMConfig for testing."""
    return MagicMock()


# --- Test create_adaptive_config ---
@patch("c4a_mcp.presets.adaptive_factory._create_statistical_config")
def test_create_adaptive_config_statistical(mock_create_statistical):
    """Test creating statistical adaptive config."""
    mock_config = MagicMock()
    mock_create_statistical.return_value = mock_config
    
    params = {
        "confidence_threshold": 0.8,
        "max_pages": 30,
        "top_k_links": 5,
        "min_gain_threshold": 0.15,
    }
    
    result = create_adaptive_config("statistical", params)
    
    assert result == mock_config
    mock_create_statistical.assert_called_once_with(params)


@patch("c4a_mcp.presets.adaptive_factory._create_embedding_config")
def test_create_adaptive_config_embedding(mock_create_embedding):
    """Test creating embedding adaptive config."""
    mock_config = MagicMock()
    mock_create_embedding.return_value = mock_config
    
    params = {
        "confidence_threshold": 0.75,
        "max_pages": 25,
        "embedding_model": "test-model",
    }
    
    result = create_adaptive_config("embedding", params)
    
    assert result == mock_config
    mock_create_embedding.assert_called_once_with(params)


def test_create_adaptive_config_case_insensitive():
    """Test that strategy type is case-insensitive."""
    with patch("c4a_mcp.presets.adaptive_factory._create_statistical_config") as mock_create:
        mock_config = MagicMock()
        mock_create.return_value = mock_config
        
        params = {}
        result = create_adaptive_config("STATISTICAL", params)
        
        assert result == mock_config
        mock_create.assert_called_once_with(params)


def test_create_adaptive_config_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported adaptive strategy"):
        create_adaptive_config("invalid", {})


# --- Test _create_statistical_config ---
@patch("crawl4ai.AdaptiveConfig")
def test_create_statistical_config_defaults(mock_adaptive_config_class):
    """Test creating statistical config with default values."""
    mock_config = MagicMock()
    mock_adaptive_config_class.return_value = mock_config
    
    params = {}
    result = _create_statistical_config(params)
    
    assert result == mock_config
    mock_adaptive_config_class.assert_called_once_with(
        strategy="statistical",
        confidence_threshold=0.7,
        max_pages=20,
        top_k_links=3,
        min_gain_threshold=0.1,
    )


@patch("crawl4ai.AdaptiveConfig")
def test_create_statistical_config_custom_values(mock_adaptive_config_class):
    """Test creating statistical config with custom values."""
    mock_config = MagicMock()
    mock_adaptive_config_class.return_value = mock_config
    
    params = {
        "confidence_threshold": 0.85,
        "max_pages": 50,
        "top_k_links": 7,
        "min_gain_threshold": 0.2,
    }
    result = _create_statistical_config(params)
    
    assert result == mock_config
    mock_adaptive_config_class.assert_called_once_with(
        strategy="statistical",
        confidence_threshold=0.85,
        max_pages=50,
        top_k_links=7,
        min_gain_threshold=0.2,
    )


# --- Test _create_embedding_config ---
@patch("crawl4ai.LLMConfig")
@patch("crawl4ai.AdaptiveConfig")
def test_create_embedding_config_defaults(mock_adaptive_config_class, mock_llm_config_class):
    """Test creating embedding config with default values."""
    mock_config = MagicMock()
    mock_adaptive_config_class.return_value = mock_config
    
    params = {}
    result = _create_embedding_config(params)
    
    assert result == mock_config
    # Should not create LLMConfig if not provided
    mock_llm_config_class.assert_not_called()
    mock_adaptive_config_class.assert_called_once()
    call_kwargs = mock_adaptive_config_class.call_args[1]
    assert call_kwargs["strategy"] == "embedding"
    assert call_kwargs["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert call_kwargs["embedding_llm_config"] is None


@patch("crawl4ai.LLMConfig")
@patch("crawl4ai.AdaptiveConfig")
def test_create_embedding_config_with_llm(mock_adaptive_config_class, mock_llm_config_class):
    """Test creating embedding config with LLM config."""
    mock_config = MagicMock()
    mock_llm_config = MagicMock()
    mock_adaptive_config_class.return_value = mock_config
    mock_llm_config_class.return_value = mock_llm_config
    
    params = {
        "embedding_llm_config": {
            "provider": "openai/gpt-4",
            "api_token": "test-token",
        },
        "embedding_model": "custom-model",
        "n_query_variations": 15,
    }
    result = _create_embedding_config(params)
    
    assert result == mock_config
    mock_llm_config_class.assert_called_once_with(
        provider="openai/gpt-4",
        api_token="test-token",
    )
    call_kwargs = mock_adaptive_config_class.call_args[1]
    assert call_kwargs["embedding_llm_config"] == mock_llm_config
    assert call_kwargs["embedding_model"] == "custom-model"
    assert call_kwargs["n_query_variations"] == 15


@patch("crawl4ai.LLMConfig")
@patch("crawl4ai.AdaptiveConfig")
def test_create_embedding_config_all_params(mock_adaptive_config_class, mock_llm_config_class):
    """Test creating embedding config with all parameters."""
    mock_config = MagicMock()
    mock_llm_config = MagicMock()
    mock_adaptive_config_class.return_value = mock_config
    mock_llm_config_class.return_value = mock_llm_config
    
    params = {
        "confidence_threshold": 0.8,
        "max_pages": 30,
        "top_k_links": 5,
        "min_gain_threshold": 0.15,
        "embedding_model": "test-model",
        "embedding_llm_config": {"provider": "test", "api_token": "token"},
        "n_query_variations": 12,
        "embedding_coverage_radius": 0.3,
        "embedding_k_exp": 4.0,
        "embedding_min_relative_improvement": 0.15,
        "embedding_validation_min_score": 0.4,
        "embedding_min_confidence_threshold": 0.15,
        "embedding_overlap_threshold": 0.9,
        "embedding_quality_min_confidence": 0.75,
        "embedding_quality_max_confidence": 0.98,
    }
    result = _create_embedding_config(params)
    
    assert result == mock_config
    call_kwargs = mock_adaptive_config_class.call_args[1]
    assert call_kwargs["confidence_threshold"] == 0.8
    assert call_kwargs["max_pages"] == 30
    assert call_kwargs["embedding_model"] == "test-model"
    assert call_kwargs["n_query_variations"] == 12
    assert call_kwargs["embedding_coverage_radius"] == 0.3


def test_create_embedding_config_invalid_llm_config_type():
    """Test creating embedding config with invalid LLM config type."""
    params = {
        "embedding_llm_config": "not_a_dict",  # Should be dict
    }
    
    with pytest.raises(ValueError, match="embedding_llm_config must be a dict"):
        _create_embedding_config(params)


def test_create_embedding_config_missing_provider():
    """Test creating embedding config with LLM config missing provider."""
    params = {
        "embedding_llm_config": {
            "api_token": "test-token",
            # Missing "provider" key
        },
    }
    
    with pytest.raises(ValueError, match="embedding_llm_config must contain 'provider' key"):
        _create_embedding_config(params)

