# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets | Models]
# :relates-to: extends: "config_models.CrawlerConfigYAML", uses: "pydantic.BaseModel", implements: "SPEC-F003 extension"
# :rationale: "Type-safe validation of preset tool parameters, ensuring runtime safety and schema generation for MCP."
# :contract: pre: "All inputs must be JSON-serializable", post: "Validated models ready for strategy creation"
# :decision_cache: "Extended CrawlerConfigYAML pattern for consistency, used discriminated unions for extraction configs [ARCH-007]"
# LLM:END

"""
Pydantic models for preset tool inputs.

These models validate parameters for preset tools, extending the base
CrawlerConfigYAML pattern for consistency with existing code.
"""

import logging
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config_models import CrawlerConfigYAML

logger = logging.getLogger(__name__)


class ExtractionConfigRegex(BaseModel):
    """Configuration for RegexExtractionStrategy."""

    type: Literal["regex"] = "regex"
    built_in_patterns: list[str] | None = Field(
        None, description="List of built-in pattern names (Email, PhoneUS, Url, etc.)"
    )
    custom_patterns: dict[str, str] | None = Field(
        None, description="Custom patterns as {label: regex} dict"
    )
    input_format: str = Field(
        "fit_html", description="Input format: html, markdown, text, or fit_html"
    )

    @model_validator(mode="after")
    def validate_patterns(self) -> "ExtractionConfigRegex":
        """Ensure either built_in_patterns or custom_patterns is provided, not both."""
        if self.built_in_patterns and self.custom_patterns:
            raise ValueError("Cannot specify both built_in_patterns and custom_patterns")
        if not self.built_in_patterns and not self.custom_patterns:
            raise ValueError("Must specify either built_in_patterns or custom_patterns")
        return self


class ExtractionConfigCss(BaseModel):
    """Configuration for JsonCssExtractionStrategy."""

    type: Literal["css"] = "css"
    schema: dict[str, Any] = Field(
        ..., description="CSS extraction schema with name, baseSelector, and fields"
    )


class ExtractionConfigLlm(BaseModel):
    """Configuration for LLMExtractionStrategy."""

    type: Literal["llm"] = "llm"
    provider: str = Field(..., description="LLM provider (e.g., openai/gpt-4o-mini)")
    api_token: str | None = Field(
        None, description="API token or env:VAR_NAME format"
    )
    schema: dict[str, Any] | None = Field(
        None, description="Pydantic schema or dict for structured extraction"
    )
    instruction: str | None = Field(None, description="Custom extraction instruction")
    extraction_type: str = Field(
        "block", description="Extraction type: block or schema"
    )


# Union type for extraction configs (discriminated by "type" field)
ExtractionConfig = ExtractionConfigRegex | ExtractionConfigCss | ExtractionConfigLlm


class PresetBaseConfig(CrawlerConfigYAML):
    """Base configuration shared by all preset tools.

    Extends CrawlerConfigYAML to include all common CrawlerRunConfig parameters
    plus extraction strategy configuration.
    """

    # Extraction parameters
    extraction_strategy: str | None = Field(
        None, description="Extraction strategy type: regex, css, llm, or None"
    )
    extraction_strategy_config: dict[str, Any] | None = Field(
        None, description="Configuration for the extraction strategy"
    )

    # Extended content processing parameters
    excluded_tags: list[str] | None = None
    excluded_selector: str | None = None
    only_text: bool | None = None
    remove_forms: bool | None = None
    process_iframes: bool | None = None

    # Extended page navigation parameters
    wait_until: str | None = None
    wait_for_images: bool | None = None
    delay_before_return_html: float | None = None
    check_robots_txt: bool | None = None

    # Extended page interaction parameters
    scan_full_page: bool | None = None
    scroll_delay: float | None = None
    remove_overlay_elements: bool | None = None
    simulate_user: bool | None = None
    override_navigator: bool | None = None
    magic: bool | None = None

    @field_validator("extraction_strategy")
    @classmethod
    def validate_extraction_strategy(cls, v: str | None) -> str | None:
        """Validate extraction strategy type."""
        if v is not None and v.lower() not in ("regex", "css", "llm"):
            raise ValueError(
                f"extraction_strategy must be 'regex', 'css', 'llm', or None, got: {v}"
            )
        return v.lower() if v else None

    @model_validator(mode="after")
    def validate_extraction_config(self) -> "PresetBaseConfig":
        """Validate extraction_strategy_config matches extraction_strategy."""
        if self.extraction_strategy and not self.extraction_strategy_config:
            raise ValueError(
                f"extraction_strategy_config required when extraction_strategy='{self.extraction_strategy}'"
            )
        if not self.extraction_strategy and self.extraction_strategy_config:
            raise ValueError(
                "extraction_strategy_config provided but extraction_strategy is None"
            )
        return self


class DeepCrawlPresetInput(BaseModel):
    """Input model for crawl_deep and crawl_deep_smart tools."""

    url: str = Field(..., description="Starting URL for deep crawl")
    max_depth: int = Field(2, description="Maximum crawl depth")
    max_pages: int = Field(50, description="Maximum number of pages to crawl")
    include_external: bool = Field(
        False, description="Whether to follow external links"
    )
    # All common preset parameters
    extraction_strategy: str | None = None
    extraction_strategy_config: dict[str, Any] | None = None
    timeout: int | float | None = None
    css_selector: str | None = None
    word_count_threshold: int | None = None
    wait_for: str | None = None
    excluded_tags: list[str] | None = None
    excluded_selector: str | None = None
    only_text: bool | None = None
    remove_forms: bool | None = None
    process_iframes: bool | None = None
    exclude_external_links: bool | None = None
    exclude_social_media_links: bool | None = None
    wait_until: str | None = None
    wait_for_images: bool | None = None
    delay_before_return_html: float | None = None
    check_robots_txt: bool | None = None
    scan_full_page: bool | None = None
    scroll_delay: float | None = None
    remove_overlay_elements: bool | None = None
    simulate_user: bool | None = None
    override_navigator: bool | None = None
    magic: bool | None = None
    bypass_cache: bool | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        parsed = urlparse(v)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http:// or https:// protocol, got: {v}")
        if not parsed.netloc:
            raise ValueError("Invalid URL format: missing netloc (domain)")
        return v

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, v: int) -> int:
        """Validate max_depth is positive."""
        if v <= 0:
            raise ValueError(f"max_depth must be positive, got: {v}")
        return v

    @field_validator("max_pages")
    @classmethod
    def validate_max_pages(cls, v: int) -> int:
        """Validate max_pages is positive."""
        if v <= 0:
            raise ValueError(f"max_pages must be positive, got: {v}")
        return v


class CrawlDeepSmartInput(DeepCrawlPresetInput):
    """Input model for crawl_deep_smart tool (adds keywords)."""

    keywords: list[str] = Field(..., description="Keywords for relevance scoring")

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v: list[str]) -> list[str]:
        """Validate keywords list is not empty."""
        if not v:
            raise ValueError("keywords list cannot be empty")
        return v


class ScrapePagePresetInput(BaseModel):
    """Input model for scrape_page tool."""

    url: str = Field(..., description="URL of the page to scrape")
    # All common preset parameters
    extraction_strategy: str | None = None
    extraction_strategy_config: dict[str, Any] | None = None
    timeout: int | float | None = None
    css_selector: str | None = None
    word_count_threshold: int | None = None
    wait_for: str | None = None
    excluded_tags: list[str] | None = None
    excluded_selector: str | None = None
    only_text: bool | None = None
    remove_forms: bool | None = None
    process_iframes: bool | None = None
    exclude_external_links: bool | None = None
    exclude_social_media_links: bool | None = None
    wait_until: str | None = None
    wait_for_images: bool | None = None
    delay_before_return_html: float | None = None
    check_robots_txt: bool | None = None
    scan_full_page: bool | None = None
    scroll_delay: float | None = None
    remove_overlay_elements: bool | None = None
    simulate_user: bool | None = None
    override_navigator: bool | None = None
    magic: bool | None = None
    bypass_cache: bool | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        parsed = urlparse(v)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http:// or https:// protocol, got: {v}")
        if not parsed.netloc:
            raise ValueError("Invalid URL format: missing netloc (domain)")
        return v

