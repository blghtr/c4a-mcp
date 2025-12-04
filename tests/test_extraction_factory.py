# LLM:METADATA
# :hierarchy: [C4A-MCP | Tests | Extraction Factory]
# :relates-to: tests: "presets.extraction_factory", implements: "SPEC-F003 extension"
# :rationale: "Validates extraction strategy factory creates correct strategy instances from config."
# :contract: pre: "Valid strategy_type and config", post: "Returns correct ExtractionStrategy instance"
# :decision_cache: "Unit tests with mocks to avoid crawl4ai imports in test environment [TEST-002]"
# LLM:END

"""
Tests for extraction strategy factory.

Validates that extraction strategies are created correctly from
configuration dictionaries.
"""

import pytest
from unittest.mock import MagicMock, patch

from c4a_mcp.presets.extraction_factory import create_extraction_strategy
from c4a_mcp.presets.models import ExtractionConfigCss, ExtractionConfigRegex


def test_create_extraction_strategy_none():
    """Test creating extraction strategy with None type."""
    result = create_extraction_strategy(None, None)
    assert result is None


def test_create_extraction_strategy_none_with_config():
    """Test creating extraction strategy with None type but config provided."""
    result = create_extraction_strategy(None, {"some": "config"})
    assert result is None


def test_create_extraction_strategy_without_config():
    """Test creating extraction strategy without config (should raise error)."""
    with pytest.raises(ValueError) as exc_info:
        create_extraction_strategy("regex", None)
    assert "extraction_strategy_config required" in str(exc_info.value)


@patch("c4a_mcp.presets.extraction_factory._create_regex_strategy")
def test_create_extraction_strategy_regex(mock_create_regex):
    """Test creating regex extraction strategy."""
    mock_strategy = MagicMock()
    mock_create_regex.return_value = mock_strategy

    config = ExtractionConfigRegex(built_in_patterns=["Email", "PhoneUS"])
    result = create_extraction_strategy("regex", config)

    assert result == mock_strategy
    mock_create_regex.assert_called_once_with(config)


@patch("c4a_mcp.presets.extraction_factory._create_css_strategy")
def test_create_extraction_strategy_css(mock_create_css):
    """Test creating CSS extraction strategy."""
    mock_strategy = MagicMock()
    mock_create_css.return_value = mock_strategy

    config = ExtractionConfigCss(extraction_schema={"name": "Test", "baseSelector": "div"})
    result = create_extraction_strategy("css", config)

    assert result == mock_strategy
    mock_create_css.assert_called_once_with(config)


def test_create_extraction_strategy_invalid_type():
    """Test creating extraction strategy with invalid type."""
    with pytest.raises(ValueError) as exc_info:
        create_extraction_strategy("invalid", {"some": "config"})
    assert "Unsupported extraction_strategy" in str(exc_info.value)


def test_create_extraction_strategy_case_insensitive():
    """Test that extraction strategy type is case-insensitive."""
    with patch("c4a_mcp.presets.extraction_factory._create_regex_strategy") as mock_create:
        mock_create.return_value = MagicMock()
        config = ExtractionConfigRegex(built_in_patterns=["Email"])
        create_extraction_strategy("REGEX", config)
        mock_create.assert_called_once()

    with patch("c4a_mcp.presets.extraction_factory._create_css_strategy") as mock_create:
        mock_create.return_value = MagicMock()
        config = ExtractionConfigCss(extraction_schema={})
        create_extraction_strategy("CSS", config)
        mock_create.assert_called_once()


@patch("c4a_mcp.presets.extraction_factory.RegexExtractionStrategy")
def test_create_regex_strategy_built_in(mock_regex_class):
    """Test creating regex strategy with built-in patterns."""
    from c4a_mcp.presets.extraction_factory import _create_regex_strategy
    
    mock_strategy = MagicMock()
    mock_regex_class.return_value = mock_strategy

    config = ExtractionConfigRegex(built_in_patterns=["Email", "PhoneUS"], input_format="fit_html")
    result = _create_regex_strategy(config)

    assert result == mock_strategy
    mock_regex_class.assert_called_once()
    # Verify pattern flags were used
    call_kwargs = mock_regex_class.call_args[1]
    assert "pattern" in call_kwargs
    assert call_kwargs["input_format"] == "fit_html"


@patch("c4a_mcp.presets.extraction_factory.RegexExtractionStrategy")
def test_create_regex_strategy_custom(mock_regex_class):
    """Test creating regex strategy with custom patterns."""
    from c4a_mcp.presets.extraction_factory import _create_regex_strategy
    
    mock_strategy = MagicMock()
    mock_regex_class.return_value = mock_strategy

    config = ExtractionConfigRegex(custom_patterns={"price": r"\$\d+"}, input_format="html")
    result = _create_regex_strategy(config)

    assert result == mock_strategy
    mock_regex_class.assert_called_once()
    call_kwargs = mock_regex_class.call_args[1]
    assert call_kwargs["custom"] == {"price": r"\$\d+"}
    assert call_kwargs["input_format"] == "html"


def test_create_regex_strategy_both_patterns():
    """Test creating regex strategy with both built-in and custom (invalid)."""
    with pytest.raises(ValueError) as exc_info:
        ExtractionConfigRegex(
            built_in_patterns=["Email"],
            custom_patterns={"price": r"\$\d+"},
        )
    assert "Cannot specify both" in str(exc_info.value)


def test_create_regex_strategy_no_patterns():
    """Test creating regex strategy with no patterns (invalid)."""
    with pytest.raises(ValueError) as exc_info:
        # Validation happens in Pydantic model
        ExtractionConfigRegex(input_format="html")
    assert "Must specify either" in str(exc_info.value)


@patch("c4a_mcp.presets.extraction_factory.JsonCssExtractionStrategy")
def test_create_css_strategy(mock_css_class):
    """Test creating CSS extraction strategy."""
    from c4a_mcp.presets.extraction_factory import _create_css_strategy
    
    mock_strategy = MagicMock()
    mock_css_class.return_value = mock_strategy

    schema = {"name": "Test", "baseSelector": "div", "fields": []}
    config = ExtractionConfigCss(extraction_schema=schema)
    result = _create_css_strategy(config)

    assert result == mock_strategy
    mock_css_class.assert_called_once_with(schema=schema)


def test_create_css_strategy_missing_schema():
    """Test creating CSS strategy without schema (invalid)."""
    with pytest.raises(Exception) as exc_info:
        # Pydantic will raise ValidationError for missing required field
        ExtractionConfigCss()
    # Pydantic validation error for missing required field
    assert "extraction_schema" in str(exc_info.value) or "schema" in str(exc_info.value)


@patch("c4a_mcp.presets.extraction_factory.RegexExtractionStrategy")
def test_map_pattern_names_to_flags(mock_regex_class):
    """Test mapping pattern names to IntFlag values."""
    from c4a_mcp.presets.extraction_factory import _map_pattern_names_to_flags

    # Mock the pattern flags
    mock_regex_class.Email = 1
    mock_regex_class.PhoneUS = 2
    mock_regex_class.Url = 4
    mock_regex_class.Nothing = 0

    result = _map_pattern_names_to_flags(["Email", "PhoneUS", "Url"])
    # Result should be combination of flags (1 | 2 | 4 = 7)
    assert result == 7


@patch("c4a_mcp.presets.extraction_factory.RegexExtractionStrategy")
def test_map_pattern_names_invalid_pattern(mock_regex_class):
    """Test mapping with invalid pattern name."""
    from c4a_mcp.presets.extraction_factory import _map_pattern_names_to_flags

    mock_regex_class.Nothing = 0

    with pytest.raises(ValueError) as exc_info:
        _map_pattern_names_to_flags(["InvalidPattern"])
    assert "Unknown built-in pattern" in str(exc_info.value)

