# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Extraction Factory]
# :relates-to: uses: "crawl4ai extraction strategies", depends-on: "presets.models.ExtractionConfig"
# :rationale: "Factory for creating extraction strategy instances from validated config, isolating strategy creation logic for testability."
# :references: PRD: "F001, F004", SPEC: "SPEC-F001, SPEC-F004"
# :contract: pre: "Valid extraction_strategy type and config", post: "Returns ExtractionStrategy instance or None"
# :decision_cache: "Factory pattern chosen to isolate strategy creation, enable testing without crawl4ai imports, and centralize validation [ARCH-008]"
# LLM:END

"""
Factory for creating extraction strategy instances.

This module creates crawl4ai extraction strategies (RegexExtractionStrategy,
JsonCssExtractionStrategy, LLMExtractionStrategy) from validated configuration.
"""

import logging
from typing import Any

from crawl4ai import (
    JsonCssExtractionStrategy,
    LLMConfig,
    LLMExtractionStrategy,
    RegexExtractionStrategy,
)

logger = logging.getLogger(__name__)


def create_extraction_strategy(
    strategy_type: str | None, config: dict[str, Any] | None
) -> Any:  # Returns ExtractionStrategy | None, but avoiding import
    """
    Create an extraction strategy instance from config.

    Args:
        strategy_type: Type of extraction strategy ("regex", "css", "llm", or None)
        config: Configuration dictionary for the strategy

    Returns:
        ExtractionStrategy instance or None if strategy_type is None

    Raises:
        ValueError: If strategy_type is invalid or config is malformed
        ImportError: If crawl4ai extraction strategies cannot be imported
    """
    if strategy_type is None:
        return None

    if config is None:
        raise ValueError(f"extraction_strategy_config required when extraction_strategy='{strategy_type}'")

    strategy_type_lower = strategy_type.lower()

    logger.debug(
        "[C4A-MCP | Presets | Extraction Factory] Creating extraction strategy | "
        "data: {strategy_type: %s}",
        strategy_type_lower,
    )

    try:
        if strategy_type_lower == "regex":
            return _create_regex_strategy(config)
        elif strategy_type_lower == "css":
            return _create_css_strategy(config)
        elif strategy_type_lower == "llm":
            return _create_llm_strategy(config)
        else:
            raise ValueError(
                f"Unsupported extraction_strategy: {strategy_type}. "
                "Supported: 'regex', 'css', 'llm'"
            )
    except ImportError as e:
        logger.error(
            "[C4A-MCP | Presets | Extraction Factory] Failed to import crawl4ai strategies | "
            "data: {error: %s}",
            str(e),
        )
        raise ImportError(
            "Failed to import crawl4ai extraction strategies. "
            "Ensure crawl4ai is installed with required dependencies."
        ) from e


def _create_regex_strategy(config: dict[str, Any]) -> Any:
    """Create RegexExtractionStrategy from config."""
    built_in_patterns = config.get("built_in_patterns")
    custom_patterns = config.get("custom_patterns")
    input_format = config.get("input_format", "fit_html")

    # Validate: either built_in or custom, not both
    if built_in_patterns and custom_patterns:
        raise ValueError("Cannot specify both built_in_patterns and custom_patterns")
    if not built_in_patterns and not custom_patterns:
        raise ValueError("Must specify either built_in_patterns or custom_patterns")

    # Map string pattern names to RegexExtractionStrategy flags
    if built_in_patterns:
        pattern_flags = _map_pattern_names_to_flags(built_in_patterns)
        return RegexExtractionStrategy(pattern=pattern_flags, input_format=input_format)
    else:
        return RegexExtractionStrategy(custom=custom_patterns, input_format=input_format)


def _map_pattern_names_to_flags(pattern_names: list[str]) -> int:
    """Map pattern name strings to RegexExtractionStrategy IntFlag values."""
    flag_map = {
        "Email": RegexExtractionStrategy.Email,
        "PhoneIntl": RegexExtractionStrategy.PhoneIntl,
        "PhoneUS": RegexExtractionStrategy.PhoneUS,
        "Url": RegexExtractionStrategy.Url,
        "IPv4": RegexExtractionStrategy.IPv4,
        "IPv6": RegexExtractionStrategy.IPv6,
        "Uuid": RegexExtractionStrategy.Uuid,
        "Currency": RegexExtractionStrategy.Currency,
        "Percentage": RegexExtractionStrategy.Percentage,
        "Number": RegexExtractionStrategy.Number,
        "DateIso": RegexExtractionStrategy.DateIso,
        "DateUS": RegexExtractionStrategy.DateUS,
        "Time24h": RegexExtractionStrategy.Time24h,
        "PostalUS": RegexExtractionStrategy.PostalUS,
        "PostalUK": RegexExtractionStrategy.PostalUK,
        "HexColor": RegexExtractionStrategy.HexColor,
        "TwitterHandle": RegexExtractionStrategy.TwitterHandle,
        "Hashtag": RegexExtractionStrategy.Hashtag,
        "MacAddr": RegexExtractionStrategy.MacAddr,
        "Iban": RegexExtractionStrategy.Iban,
        "CreditCard": RegexExtractionStrategy.CreditCard,
        "All": RegexExtractionStrategy.All,
    }

    result = RegexExtractionStrategy.Nothing
    for name in pattern_names:
        if name not in flag_map:
            raise ValueError(
                f"Unknown built-in pattern: {name}. "
                f"Valid patterns: {', '.join(flag_map.keys())}"
            )
        result |= flag_map[name]

    return result


def _create_css_strategy(config: dict[str, Any]) -> Any:
    """Create JsonCssExtractionStrategy from config."""
    schema = config.get("schema")
    if not schema:
        raise ValueError("'schema' required in extraction_strategy_config for 'css' strategy")

    return JsonCssExtractionStrategy(schema=schema)


def _create_llm_strategy(config: dict[str, Any]) -> Any:
    """Create LLMExtractionStrategy from config."""
    provider = config.get("provider")
    if not provider:
        raise ValueError("'provider' required in extraction_strategy_config for 'llm' strategy")

    api_token = config.get("api_token")
    # NOTE(REVIEWER): api_token may contain sensitive data. Consider logging
    # only presence/absence, not the actual token value. Current implementation
    # passes token to LLMConfig which should handle it securely.
    schema = config.get("schema")
    instruction = config.get("instruction")
    extraction_type = config.get("extraction_type", "block")

    llm_config = LLMConfig(provider=provider, api_token=api_token)

    return LLMExtractionStrategy(
        llm_config=llm_config,
        schema=schema,
        instruction=instruction,
        extraction_type=extraction_type,
    )

