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
JsonCssExtractionStrategy) from validated configuration.
"""

import logging
from typing import Any

from crawl4ai import (
    JsonCssExtractionStrategy,
    RegexExtractionStrategy,
)

from .models import ExtractionConfig, ExtractionConfigCss, ExtractionConfigRegex

logger = logging.getLogger(__name__)


def create_extraction_strategy(
    strategy_type: str | None, config: ExtractionConfig | None
) -> Any:  # Returns ExtractionStrategy | None, but avoiding import
    """
    Create an extraction strategy instance from config.

    Args:
        strategy_type: Type of extraction strategy ("regex", "css", or None)
        config: Configuration Pydantic model for the strategy

    Returns:
        ExtractionStrategy instance or None if strategy_type is None

    Raises:
        ValueError: If strategy_type is invalid or config is malformed
        ImportError: If crawl4ai extraction strategies cannot be imported
    """
    if strategy_type is None:
        return None

    if config is None:
        raise ValueError(
            f"extraction_strategy_config required when extraction_strategy='{strategy_type}'"
        )

    strategy_type_lower = strategy_type.lower()

    logger.debug(
        "[C4A-MCP | Presets | Extraction Factory] Creating extraction strategy | "
        "data: {strategy_type: %s}",
        strategy_type_lower,
    )

    try:
        if strategy_type_lower == "regex":
            if not isinstance(config, ExtractionConfigRegex):
                raise ValueError(
                    f"extraction_strategy='regex' requires ExtractionConfigRegex, "
                    f"got {type(config).__name__}"
                )
            return _create_regex_strategy(config)
        elif strategy_type_lower == "css":
            if not isinstance(config, ExtractionConfigCss):
                raise ValueError(
                    f"extraction_strategy='css' requires ExtractionConfigCss, "
                    f"got {type(config).__name__}"
                )
            return _create_css_strategy(config)
        else:
            raise ValueError(
                f"Unsupported extraction_strategy: {strategy_type}. Supported: 'regex', 'css'"
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


def _create_regex_strategy(config: ExtractionConfig) -> Any:
    """Create RegexExtractionStrategy from config."""
    if not isinstance(config, ExtractionConfigRegex):
        raise ValueError(f"Expected ExtractionConfigRegex, got {type(config).__name__}")


    built_in_patterns = config.built_in_patterns
    custom_patterns = config.custom_patterns
    input_format = config.input_format

    # Map string pattern names to RegexExtractionStrategy flags
    if built_in_patterns:
        pattern_flags = _map_pattern_names_to_flags(built_in_patterns)
        return RegexExtractionStrategy(pattern=pattern_flags, input_format=input_format)
    else:
        return RegexExtractionStrategy(custom=custom_patterns, input_format=input_format)


def _map_pattern_names_to_flags(pattern_names: list[str]) -> int:
    """Map pattern name strings to RegexExtractionStrategy IntFlag values."""
    # Define mapping with correct case keys
    # Note: We create a case-insensitive lookup map below
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

    # Create case-insensitive lookup map (lowercase key -> flag value)
    lookup_map = {k.lower(): v for k, v in flag_map.items()}

    result = RegexExtractionStrategy.Nothing
    for name in pattern_names:
        name_lower = name.lower()
        if name_lower not in lookup_map:
            raise ValueError(
                f"Unknown built-in pattern: {name}. "
                f"Valid patterns: {', '.join(sorted(flag_map.keys()))}"
            )
        result |= lookup_map[name_lower]

    return result


def _create_css_strategy(config: ExtractionConfig) -> Any:
    """Create JsonCssExtractionStrategy from config."""
    if not isinstance(config, ExtractionConfigCss):
        raise ValueError(f"Expected ExtractionConfigCss, got {type(config).__name__}")


    return JsonCssExtractionStrategy(schema=config.extraction_schema)
