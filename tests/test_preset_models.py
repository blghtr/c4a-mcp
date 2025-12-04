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



