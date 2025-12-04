# LLM:METADATA
# :hierarchy: [C4A-MCP | Models]
# :relates-to: uses: "config_models.CrawlerConfigYAML", motivated_by: "PRD-F001", implements: "SPEC-F001"
# :rationale: "Defines strict types for tool inputs and outputs to ensure runtime safety and schema generation."
# :contract: invariant: "All inputs must be serializable to JSON"
# :decision_cache: "Used Pydantic for robust validation and schema generation compatible with MCP [ARCH-002]"
# LLM:END

import logging
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from .config_models import CrawlerConfigYAML

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

# NOTE: VALID_CONFIG_FIELDS removed - validation now handled by CrawlerConfigYAML in config_models.py


class RunnerInput(BaseModel):
    """
    Input parameters for the crawl runner tool.
    """

    url: str = Field(..., description="The starting URL for the crawl session.")
    script: str | None = Field(
        None,
        description=(
            "A c4a-script DSL string defining interaction steps " "(GO, WAIT, CLICK, etc)."
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
            raise ValueError(f"URL must use http:// or https:// protocol, got: {scheme_str}")
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
        Validate config by delegating to CrawlerConfigYAML model.

        All validation logic is centralized in config_models.CrawlerConfigYAML.
        This ensures consistency between YAML configs and tool-level configs.

        If config is a serialized CrawlerRunConfig (from preset tools), skip validation
        and return as-is. It will be deserialized in CrawlRunner._build_run_config().

        Args:
            v: Config dictionary to validate

        Returns:
            Validated config dictionary or None

        Raises:
            ValueError: If config validation fails
        """
        if v is None:
            return v

        # Check if this is a serialized CrawlerRunConfig from preset tools
        # Format: {type: "CrawlerRunConfig", params: {...}}
        if isinstance(v, dict) and "type" in v and v["type"] == "CrawlerRunConfig":
            # Skip validation - will be deserialized in CrawlRunner._build_run_config()
            return v

        # Let CrawlerConfigYAML handle all validation
        try:
            validated_config = CrawlerConfigYAML.from_dict(v)
            # Return as dict for backward compatibility
            return validated_config.model_dump(exclude_none=True)
        except Exception as e:
            logger.error(
                "[C4A-MCP | Models] Config validation failed | data: {error: %s}",
                str(e),
            )
            raise ValueError(f"Invalid config: {e}") from e


class RunnerOutput(BaseModel):
    """
    Structured output from the crawl runner tool.
    """

    markdown: str = Field(..., description="The extracted content in Markdown format.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the crawl (title, url, timestamp)."
    )
    error: str | None = Field(None, description="Error message if the crawl failed.")
