# LLM:METADATA
# :hierarchy: [C4A-MCP | Configuration]
# :relates-to: uses: "pydantic.BaseModel", uses: "crawl4ai.CrawlerRunConfig", uses: "crawl4ai.BrowserConfig"
# :rationale: "Centralized configuration validation and YAML loading for browser and crawler settings."
# :contract: invariant: "All config validation happens here, not in RunnerInput"
# :decision_cache: "Pydantic models own their validation logic (SRP) [ARCH-005]"
# LLM:END

import logging
from pathlib import Path
from typing import Any

import yaml
from crawl4ai import BrowserConfig, CacheMode, CrawlerRunConfig
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class JsonCssSchema(BaseModel):
    """Schema for JsonCssExtractionStrategy."""

    baseSelector: str = Field(..., description="Base CSS selector for the list of items")
    fields: list[dict[str, Any]] = Field(..., description="List of fields to extract")


class CrawlerConfigYAML(BaseModel):
    """Subset of CrawlerRunConfig attributes for YAML config.

    All validation logic is centralized here - no duplication in RunnerInput.
    Pydantic handles type validation, custom validators handle business rules.

    No default values are defined here - they come from CrawlerRunConfig itself.
    """

    timeout: int | None = None
    bypass_cache: bool | None = None
    css_selector: str | None = None
    wait_for: str | None = None
    word_count_threshold: int | None = None
    exclude_external_links: bool | None = None
    exclude_social_media_links: bool | None = None
    extraction_strategy: str | None = None
    extraction_strategy_schema: JsonCssSchema | dict[str, Any] | None = None

    # VALIDATION: Moved from models.py RunnerInput.validate_config()
    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int | None) -> int | None:
        """Validate timeout is positive."""
        if v is not None and v <= 0:
            raise ValueError(f"timeout must be positive, got: {v}")
        return v

    @field_validator("word_count_threshold")
    @classmethod
    def validate_word_count(cls, v: int | None) -> int | None:
        """Validate word count threshold is non-negative."""
        if v is not None and v < 0:
            raise ValueError(f"word_count_threshold must be non-negative, got: {v}")
        return v

    @field_validator("extraction_strategy")
    @classmethod
    def validate_extraction_strategy(cls, v: str | None) -> str | None:
        """Validate extraction strategy is supported."""
        if v is not None and v.lower() != "jsoncss":
            raise ValueError(f"Only 'jsoncss' extraction_strategy supported, got: {v}")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Cross-field validation after model initialization."""
        if self.extraction_strategy == "jsoncss" and not self.extraction_strategy_schema:
            raise ValueError(
                "extraction_strategy_schema required when extraction_strategy='jsoncss'"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CrawlerConfigYAML":
        """Load config from YAML file with validation.

        Args:
            path: Path to YAML file

        Returns:
            Validated CrawlerConfigYAML instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {path}: expected dict, got {type(data)}")

        return cls(**data.get("crawler", {}))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrawlerConfigYAML":
        """Create from dict (e.g., from tool call config param).

        Args:
            data: Configuration dictionary

        Returns:
            Validated CrawlerConfigYAML instance
        """
        return cls(**data)

    def merge(self, overrides: "CrawlerConfigYAML | dict[str, Any]") -> "CrawlerConfigYAML":
        """Merge with another config, overrides take precedence (non-None values).

        Args:
            overrides: Config to merge (dict or CrawlerConfigYAML)

        Returns:
            New merged CrawlerConfigYAML instance
        """
        if isinstance(overrides, dict):
            overrides = CrawlerConfigYAML(**overrides)

        merged_data = self.model_dump(exclude_none=True)
        override_data = overrides.model_dump(exclude_none=True)
        merged_data.update(override_data)

        return CrawlerConfigYAML(**merged_data)

    def to_crawler_run_config_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for CrawlerRunConfig, excluding None values.

        Returns:
            Dict suitable for CrawlerRunConfig(**kwargs)
        """
        kwargs = self.model_dump(exclude_none=True)

        # Convert bypass_cache to cache_mode
        if "bypass_cache" in kwargs:
            bypass = kwargs.pop("bypass_cache")
            kwargs["cache_mode"] = CacheMode.BYPASS if bypass else CacheMode.ENABLED

        # Convert timeout from seconds to milliseconds
        if "timeout" in kwargs:
            kwargs["page_timeout"] = int(kwargs.pop("timeout") * 1000)

        # Convert schema model to dict if present
        if "extraction_strategy_schema" in kwargs and isinstance(
            kwargs["extraction_strategy_schema"], BaseModel
        ):
            kwargs["extraction_strategy_schema"] = kwargs["extraction_strategy_schema"].model_dump()

        return kwargs


class BrowserConfigYAML(BaseModel):
    """Subset of BrowserConfig for YAML config.

    No default values defined - they come from BrowserConfig itself.
    """

    headless: bool | None = None
    browser_type: str | None = None
    verbose: bool | None = None
    user_agent: str | None = None

    @field_validator("browser_type")
    @classmethod
    def validate_browser_type(cls, v: str | None) -> str | None:
        """Validate browser type is supported."""
        if v is not None and v not in ("chromium", "firefox", "webkit"):
            raise ValueError(f"Invalid browser_type: {v}, must be chromium/firefox/webkit")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BrowserConfigYAML":
        """Load from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Validated BrowserConfigYAML instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {path}: expected dict, got {type(data)}")

        return cls(**data.get("browser", {}))

    def merge(self, overrides: "BrowserConfigYAML | dict[str, Any]") -> "BrowserConfigYAML":
        """Merge with another config, overrides take precedence.

        Args:
            overrides: Config to merge (dict or BrowserConfigYAML)

        Returns:
            New merged BrowserConfigYAML instance
        """
        if isinstance(overrides, dict):
            overrides = BrowserConfigYAML(**overrides)

        merged_data = self.model_dump(exclude_none=True)
        override_data = overrides.model_dump(exclude_none=True)
        merged_data.update(override_data)

        return BrowserConfigYAML(**merged_data)

    def to_browser_config(self) -> BrowserConfig:
        """Convert to BrowserConfig instance.

        Returns:
            BrowserConfig with settings from this YAML config
        """
        kwargs = self.model_dump(exclude_none=True)
        return BrowserConfig(**kwargs)


class AppConfig(BaseModel):
    """Root config containing both browser and crawler settings."""

    browser: BrowserConfigYAML = Field(default_factory=BrowserConfigYAML)
    crawler: CrawlerConfigYAML = Field(default_factory=CrawlerConfigYAML)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load full app config from YAML.

        Args:
            path: Path to YAML file

        Returns:
            Validated AppConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {path}: expected dict, got {type(data)}")

        return cls(
            browser=BrowserConfigYAML(**data.get("browser", {})),
            crawler=CrawlerConfigYAML(**data.get("crawler", {})),
        )
