# LLM:METADATA
# :hierarchy: [C4A-MCP | Tests | Preset Models]
# :relates-to: tests: "presets.models", implements: "SPEC-F003 extension"
# :rationale: "Validates input models for preset tools, ensuring type safety and business rule compliance."
# :contract: pre: "Valid test data", post: "All validation rules verified"
# :decision_cache: "Unit tests for Pydantic models to catch validation errors early [TEST-001]"
# LLM:END

"""
Tests for preset tool input models.

Validates Pydantic models for preset tools, ensuring proper validation
of URLs, parameters, and extraction configurations.
"""

import pytest
from pydantic import ValidationError

from c4a_mcp.presets.models import (
    AdaptiveEmbeddingInput,
    AdaptiveStatisticalInput,
    CrawlDeepSmartInput,
    DeepCrawlPresetInput,
    ExtractionConfigCss,
    ExtractionConfigRegex,
    PresetBaseConfig,
    ScrapePagePresetInput,
)


# --- Test DeepCrawlPresetInput ---
def test_deep_crawl_input_valid():
    """Test valid DeepCrawlPresetInput."""
    input_data = DeepCrawlPresetInput(
        url="https://example.com",
        max_depth=3,
        max_pages=100,
        include_external=True,
    )
    assert input_data.url == "https://example.com"
    assert input_data.max_depth == 3
    assert input_data.max_pages == 100
    assert input_data.include_external is True


def test_deep_crawl_input_defaults():
    """Test DeepCrawlPresetInput with default values."""
    input_data = DeepCrawlPresetInput(url="https://example.com")
    assert input_data.max_depth == 2
    assert input_data.max_pages == 50
    assert input_data.include_external is False


def test_deep_crawl_input_invalid_url():
    """Test DeepCrawlPresetInput with invalid URL."""
    with pytest.raises(ValidationError) as exc_info:
        DeepCrawlPresetInput(url="not-a-url")
    assert "URL must use http:// or https://" in str(exc_info.value)


def test_deep_crawl_input_invalid_max_depth():
    """Test DeepCrawlPresetInput with invalid max_depth."""
    with pytest.raises(ValidationError) as exc_info:
        DeepCrawlPresetInput(url="https://example.com", max_depth=0)
    assert "max_depth must be positive" in str(exc_info.value)

    with pytest.raises(ValidationError):
        DeepCrawlPresetInput(url="https://example.com", max_depth=-1)


def test_deep_crawl_input_invalid_max_pages():
    """Test DeepCrawlPresetInput with invalid max_pages."""
    with pytest.raises(ValidationError) as exc_info:
        DeepCrawlPresetInput(url="https://example.com", max_pages=0)
    assert "max_pages must be positive" in str(exc_info.value)


# --- Test CrawlDeepSmartInput ---
def test_crawl_deep_smart_input_valid():
    """Test valid CrawlDeepSmartInput."""
    input_data = CrawlDeepSmartInput(
        url="https://example.com",
        keywords=["test", "example"],
        max_depth=2,
        max_pages=25,
    )
    assert input_data.keywords == ["test", "example"]
    assert input_data.max_depth == 2


def test_crawl_deep_smart_input_empty_keywords():
    """Test CrawlDeepSmartInput with empty keywords."""
    with pytest.raises(ValidationError) as exc_info:
        CrawlDeepSmartInput(url="https://example.com", keywords=[])
    assert "keywords list cannot be empty" in str(exc_info.value)


# --- Test ScrapePagePresetInput ---
def test_scrape_page_input_valid():
    """Test valid ScrapePagePresetInput."""
    input_data = ScrapePagePresetInput(url="https://example.com")
    assert input_data.url == "https://example.com"


# --- Test PresetBaseConfig ---
def test_preset_base_config_valid():
    """Test valid PresetBaseConfig."""
    config = PresetBaseConfig(
        timeout=30,
        css_selector="article",
        word_count_threshold=100,
    )
    assert config.timeout == 30
    assert config.css_selector == "article"
    assert config.word_count_threshold == 100


def test_preset_base_config_extraction_strategy_none():
    """Test PresetBaseConfig with no extraction strategy."""
    config = PresetBaseConfig()
    assert config.extraction_strategy is None
    assert config.extraction_strategy_config is None


def test_preset_base_config_extraction_strategy_without_config():
    """Test PresetBaseConfig with extraction_strategy but no config."""
    with pytest.raises(ValidationError) as exc_info:
        PresetBaseConfig(extraction_strategy="regex")
    assert "extraction_strategy_config required" in str(exc_info.value)


def test_preset_base_config_extraction_strategy_invalid_type():
    """Test PresetBaseConfig with invalid extraction_strategy type."""
    with pytest.raises(ValidationError) as exc_info:
        PresetBaseConfig(extraction_strategy="invalid")
        assert "must be 'regex', 'css', or None" in str(exc_info.value)


# --- Test ExtractionConfig models ---
def test_extraction_config_regex_built_in():
    """Test ExtractionConfigRegex with built-in patterns."""
    config = ExtractionConfigRegex(
        type="regex",
        built_in_patterns=["Email", "PhoneUS", "Url"],
        input_format="fit_html",
    )
    assert config.type == "regex"
    assert config.built_in_patterns == ["Email", "PhoneUS", "Url"]
    assert config.custom_patterns is None


def test_extraction_config_regex_custom():
    """Test ExtractionConfigRegex with custom patterns."""
    config = ExtractionConfigRegex(
        type="regex",
        custom_patterns={"price": r"\$\d+\.\d{2}"},
        input_format="html",
    )
    assert config.custom_patterns == {"price": r"\$\d+\.\d{2}"}
    assert config.built_in_patterns is None


def test_extraction_config_regex_both_patterns():
    """Test ExtractionConfigRegex with both built-in and custom patterns (invalid)."""
    with pytest.raises(ValidationError) as exc_info:
        ExtractionConfigRegex(
            type="regex",
            built_in_patterns=["Email"],
            custom_patterns={"price": r"\$\d+"},
        )
    assert "Cannot specify both built_in_patterns and custom_patterns" in str(
        exc_info.value
    )


def test_extraction_config_regex_no_patterns():
    """Test ExtractionConfigRegex with no patterns (invalid)."""
    with pytest.raises(ValidationError) as exc_info:
        ExtractionConfigRegex(type="regex")
    assert "Must specify either built_in_patterns or custom_patterns" in str(
        exc_info.value
    )


def test_extraction_config_css():
    """Test ExtractionConfigCss."""
    schema = {
        "name": "Product",
        "baseSelector": ".product-card",
        "fields": [{"name": "title", "selector": "h2", "type": "text"}],
    }
    config = ExtractionConfigCss(type="css", extraction_schema=schema)
    assert config.type == "css"
    assert config.extraction_schema == schema


# --- Test AdaptiveCrawlInput ---
def test_adaptive_crawl_input_valid():
    """Test valid AdaptiveCrawlInput."""
    input_data = AdaptiveStatisticalInput(
        url="https://example.com",
        query="test query",
        confidence_threshold=0.8,
        max_pages=30,
        top_k_links=5,
        min_gain_threshold=0.15,
    )
    assert input_data.url == "https://example.com"
    assert input_data.query == "test query"
    assert input_data.confidence_threshold == 0.8
    assert input_data.max_pages == 30
    assert input_data.top_k_links == 5
    assert input_data.min_gain_threshold == 0.15


def test_adaptive_crawl_input_defaults():
    """Test AdaptiveCrawlInput with default values."""
    input_data = AdaptiveStatisticalInput(
        url="https://example.com",
        query="test query",
    )
    assert input_data.confidence_threshold == 0.7
    assert input_data.max_pages == 20
    assert input_data.top_k_links == 3
    assert input_data.min_gain_threshold == 0.1


def test_adaptive_crawl_input_invalid_url():
    """Test AdaptiveCrawlInput with invalid URL."""
    with pytest.raises(ValidationError) as exc_info:
        AdaptiveStatisticalInput(url="not-a-url", query="test")
    assert "URL must use http:// or https://" in str(exc_info.value)


def test_adaptive_crawl_input_empty_query():
    """Test AdaptiveCrawlInput with empty query."""
    with pytest.raises(ValidationError) as exc_info:
        AdaptiveStatisticalInput(url="https://example.com", query="")
    assert "query cannot be empty" in str(exc_info.value)

    with pytest.raises(ValidationError):
        AdaptiveStatisticalInput(url="https://example.com", query="   ")


def test_adaptive_crawl_input_query_trimmed():
    """Test that query is trimmed of whitespace."""
    input_data = AdaptiveStatisticalInput(
        url="https://example.com",
        query="  test query  ",
    )
    assert input_data.query == "test query"


def test_adaptive_crawl_input_invalid_confidence_threshold():
    """Test AdaptiveCrawlInput with invalid confidence_threshold."""
    with pytest.raises(ValidationError):
        AdaptiveStatisticalInput(
            url="https://example.com",
            query="test",
            confidence_threshold=1.5,  # > 1.0
        )

    with pytest.raises(ValidationError):
        AdaptiveStatisticalInput(
            url="https://example.com",
            query="test",
            confidence_threshold=-0.1,  # < 0.0
        )


def test_adaptive_crawl_input_invalid_max_pages():
    """Test AdaptiveCrawlInput with invalid max_pages."""
    with pytest.raises(ValidationError):
        AdaptiveStatisticalInput(
            url="https://example.com",
            query="test",
            max_pages=0,  # <= 0
        )


def test_adaptive_crawl_input_invalid_top_k_links():
    """Test AdaptiveCrawlInput with invalid top_k_links."""
    with pytest.raises(ValidationError):
        AdaptiveStatisticalInput(
            url="https://example.com",
            query="test",
            top_k_links=0,  # <= 0
        )


# --- Test AdaptiveStatisticalInput ---
def test_adaptive_statistical_input_valid():
    """Test valid AdaptiveStatisticalInput."""
    input_data = AdaptiveStatisticalInput(
        url="https://example.com",
        query="test query",
    )
    assert isinstance(input_data, AdaptiveStatisticalInput)


# --- Test AdaptiveEmbeddingInput ---
def test_adaptive_embedding_input_valid():
    """Test valid AdaptiveEmbeddingInput."""
    input_data = AdaptiveEmbeddingInput(
        url="https://example.com",
        query="test query",
        embedding_model="custom-model",
        n_query_variations=15,
    )
    assert input_data.embedding_model == "custom-model"
    assert input_data.n_query_variations == 15
    assert input_data.embedding_llm_config is None


def test_adaptive_embedding_input_defaults():
    """Test AdaptiveEmbeddingInput with default values."""
    input_data = AdaptiveEmbeddingInput(
        url="https://example.com",
        query="test query",
    )
    assert input_data.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert input_data.n_query_variations == 10
    assert input_data.embedding_coverage_radius == 0.2
    assert input_data.embedding_k_exp == 3.0


def test_adaptive_embedding_input_with_llm_config():
    """Test AdaptiveEmbeddingInput with LLM config."""
    llm_config = {
        "provider": "openai/gpt-4",
        "api_token": "test-token",
    }
    input_data = AdaptiveEmbeddingInput(
        url="https://example.com",
        query="test query",
        embedding_llm_config=llm_config,
    )
    assert input_data.embedding_llm_config == llm_config


def test_adaptive_embedding_input_invalid_embedding_params():
    """Test AdaptiveEmbeddingInput with invalid embedding parameters."""
    with pytest.raises(ValidationError):
        AdaptiveEmbeddingInput(
            url="https://example.com",
            query="test",
            n_query_variations=0,  # <= 0
        )

    with pytest.raises(ValidationError):
        AdaptiveEmbeddingInput(
            url="https://example.com",
            query="test",
            embedding_validation_min_score=1.5,  # > 1.0
        )



