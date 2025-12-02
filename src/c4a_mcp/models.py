# LLM:METADATA
# :hierarchy: [C4A-MCP | Models]
# :relates-to: motivated_by: "PRD-F001", implements: "SPEC-F001"
# :rationale: "Defines strict types for tool inputs and outputs to ensure runtime safety and schema generation."
# :contract: invariant: "All inputs must be serializable to JSON"
# :decision_cache: "Used Pydantic for robust validation and schema generation compatible with MCP [ARCH-002]"
# LLM:END

import logging
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Valid c4a-script commands from documentation
VALID_C4A_COMMANDS = {
    "GO",
    "RELOAD",
    "BACK",
    "FORWARD",  # Navigation
    "WAIT",  # Wait
    "CLICK",
    "DOUBLE_CLICK",
    "RIGHT_CLICK",
    "SCROLL",
    "DRAG",  # Mouse
    "TYPE",
    "PRESS",
    "CLEAR",
    "SET",  # Keyboard
    "IF",
    "REPEAT",  # Control Flow
    "SETVAR",
    "EVAL",  # Variables & Advanced
}

# Valid config fields based on CrawlerRunConfig and _map_config implementation
VALID_CONFIG_FIELDS = {
    "bypass_cache",
    "timeout",
    "css_selector",
    "wait_for",
    "word_count_threshold",
    "exclude_external_links",
    "exclude_social_media_links",
    "extraction_strategy",
    "extraction_strategy_schema",
}


class RunnerInput(BaseModel):
    """
    Input parameters for the crawl runner tool.
    """

    url: str = Field(
        ..., description="The starting URL for the crawl session."
    )
    script: str | None = Field(
        None,
        description=(
            "A c4a-script DSL string defining interaction steps "
            "(GO, WAIT, CLICK, etc)."
        ),
    )
    config: dict[str, Any] | None = Field(
        None,
        description=(
            "Configuration object mapping to crawl4ai's CrawlerRunConfig "
            "(css_selector, wait_for, etc)."
        ),
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """
        Validate URL format and protocol.

        Args:
            v: URL string to validate

        Returns:
            Validated URL string

        Raises:
            ValueError: If URL format is invalid or protocol is not http/https
        """
        parsed = urlparse(v)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            scheme_str = parsed.scheme or "no scheme"
            raise ValueError(
                f"URL must use http:// or https:// protocol, got: {scheme_str}"
            )
        if not parsed.netloc:
            raise ValueError("Invalid URL format: missing netloc (domain)")
        return v

    @field_validator("script")
    @classmethod
    def validate_script(cls, v: str | None) -> str | None:
        """
        Validate c4a-script commands.

        Args:
            v: Script string to validate

        Returns:
            Validated script string or None

        Raises:
            ValueError: If script contains invalid c4a-script commands
        """
        if v is None:
            return v

        for line_num, line in enumerate(v.splitlines(), start=1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Extract first word (command)
            parts = line.split()
            if not parts:
                continue

            first_word = parts[0]
            if first_word not in VALID_C4A_COMMANDS:
                valid_cmds = ", ".join(sorted(VALID_C4A_COMMANDS))
                raise ValueError(
                    f"Invalid c4a-script command '{first_word}' at line {line_num}. "
                    f"Valid commands: {valid_cmds}"
                )

        return v

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Validate config fields against CrawlerRunConfig.

        Args:
            v: Config dictionary to validate

        Returns:
            Validated config dictionary or None

        Raises:
            ValueError: If config contains invalid values or missing required fields
        """
        if v is None:
            return v

        # Warn about unknown fields but don't fail (for compatibility)
        unknown_fields = set(v.keys()) - VALID_CONFIG_FIELDS
        if unknown_fields:
            logger.warning(
                "[C4A-MCP | Models] Unknown config fields will be ignored | "
                "data: {unknown_fields: %s}",
                sorted(unknown_fields),
            )

        # Validate extraction_strategy
        if "extraction_strategy" in v:
            strategy = v["extraction_strategy"]
            if strategy and strategy.lower() != "jsoncss":
                raise ValueError(
                    f"Only 'jsoncss' extraction_strategy is supported, got: {strategy}"
                )
            if strategy and strategy.lower() == "jsoncss" and "extraction_strategy_schema" not in v:
                raise ValueError(
                    "extraction_strategy_schema is required when extraction_strategy='jsoncss'"
                )

        # Validate types for known fields
        if "bypass_cache" in v and not isinstance(v["bypass_cache"], bool):
            got_type = type(v["bypass_cache"]).__name__
            raise ValueError(f"bypass_cache must be bool, got {got_type}")

        if "timeout" in v:
            timeout = v["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError(
                    f"timeout must be a positive number, got: {timeout}"
                )

        if "css_selector" in v:
            css_sel = v["css_selector"]
            if css_sel is not None and not isinstance(css_sel, str):
                got_type = type(css_sel).__name__
                raise ValueError(f"css_selector must be str or None, got {got_type}")

        if "wait_for" in v:
            wait_val = v["wait_for"]
            if wait_val is not None and not isinstance(wait_val, str):
                got_type = type(wait_val).__name__
                raise ValueError(f"wait_for must be str or None, got {got_type}")

        if "word_count_threshold" in v and v["word_count_threshold"] is not None:
            threshold = v["word_count_threshold"]
            if not isinstance(threshold, int) or threshold < 0:
                raise ValueError(
                    f"word_count_threshold must be a non-negative integer, "
                    f"got: {threshold}"
                )

        if "exclude_external_links" in v:
            ext_links = v["exclude_external_links"]
            if not isinstance(ext_links, bool):
                got_type = type(ext_links).__name__
                raise ValueError(
                    f"exclude_external_links must be bool, got {got_type}"
                )

        if "exclude_social_media_links" in v:
            social_links = v["exclude_social_media_links"]
            if not isinstance(social_links, bool):
                got_type = type(social_links).__name__
                raise ValueError(
                    f"exclude_social_media_links must be bool, got {got_type}"
                )

        if "extraction_strategy_schema" in v:
            schema = v["extraction_strategy_schema"]
            if not isinstance(schema, dict):
                got_type = type(schema).__name__
                raise ValueError(
                    f"extraction_strategy_schema must be dict, got {got_type}"
                )

        return v


class RunnerOutput(BaseModel):
    """
    Structured output from the crawl runner tool.
    """

    markdown: str = Field(..., description="The extracted content in Markdown format.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the crawl (title, url, timestamp)."
    )
    error: str | None = Field(None, description="Error message if the crawl failed.")
