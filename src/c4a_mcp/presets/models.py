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

from __future__ import annotations

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
    extraction_schema: dict[str, Any] = Field(
        ..., description="CSS extraction schema with name, baseSelector, and fields", alias="schema"
    )
    
    model_config = {"populate_by_name": True}


# Union type for extraction configs (discriminated by "type" field)
ExtractionConfig = ExtractionConfigRegex | ExtractionConfigCss


class PresetBaseConfig(CrawlerConfigYAML):
    """Base configuration shared by all preset tools.

    Extends CrawlerConfigYAML to include all common CrawlerRunConfig parameters
    plus extraction strategy configuration.
    """

    # Extraction parameters
    extraction_strategy: str | None = Field(
        None, description="Extraction strategy type: regex, css, or None"
    )
    extraction_strategy_config: ExtractionConfig | None = Field(
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
        if v is not None and v.lower() not in ("regex", "css"):
            raise ValueError(
                f"extraction_strategy must be 'regex', 'css', or None, got: {v}"
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
    extraction_strategy_config: ExtractionConfig | None = None
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
    extraction_strategy_config: ExtractionConfig | None = None
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


class AdaptiveCrawlInput(BaseModel):
    """Base input model for adaptive crawling tools.
    
    Adaptive crawling uses query-based relevance scoring to determine
    when sufficient information has been gathered. Stops automatically
    when confidence threshold is reached.
    """
    
    url: str = Field(..., description="Starting URL for adaptive crawl")
    query: str = Field(..., description="Query string for relevance-based crawling")
    confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Stop when confidence reaches this threshold (0.0-1.0)"
    )
    max_pages: int = Field(20, gt=0, description="Maximum number of pages to crawl")
    top_k_links: int = Field(3, gt=0, description="Number of links to follow per page")
    min_gain_threshold: float = Field(
        0.1, ge=0.0, description="Minimum expected information gain to continue crawling"
    )
    # Common preset parameters (subset - adaptive crawling has its own config)
    timeout: int | float | None = None
    css_selector: str | None = None
    word_count_threshold: int | None = None
    wait_for: str | None = None
    exclude_external_links: bool | None = None
    exclude_social_media_links: bool | None = None
    bypass_cache: bool | None = None
    # Optional runner-level config (strategy overrides, adaptive_config_params, timeout)
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional runner config (strategy, adaptive_config_params, timeout).",
    )
    
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
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


class AdaptiveStatisticalInput(AdaptiveCrawlInput):
    """Input model for statistical adaptive crawling.
    
    Uses term-based analysis and information theory. Fast, efficient,
    no external dependencies. Best for well-defined queries with specific terminology.
    """
    
    # No additional fields - statistical is the default strategy
    pass


class AdaptiveEmbeddingInput(AdaptiveCrawlInput):
    """Input model for embedding-based adaptive crawling.
    
    Uses semantic embeddings for deeper understanding. Captures meaning
    beyond exact term matches. Requires embedding model or LLM API.
    """
    
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier (HuggingFace model name)"
    )
    embedding_llm_config: dict[str, Any] | None = Field(
        None,
        description="LLM config for query expansion: {provider: str, api_token: str, ...}"
    )
    n_query_variations: int = Field(
        10, gt=0, description="Number of query variations to generate for expansion"
    )
    embedding_coverage_radius: float = Field(
        0.2, ge=0.0, description="Distance threshold for semantic coverage"
    )
    embedding_k_exp: float = Field(
        3.0, gt=0.0, description="Exponential decay factor for coverage (higher = stricter)"
    )
    embedding_min_relative_improvement: float = Field(
        0.1, ge=0.0, description="Minimum relative improvement to continue crawling"
    )
    embedding_validation_min_score: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum validation score threshold"
    )
    embedding_min_confidence_threshold: float = Field(
        0.1, ge=0.0, le=1.0, description="Below this confidence = irrelevant content"
    )
    embedding_overlap_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Similarity threshold for deduplication"
    )
    embedding_quality_min_confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence for quality display"
    )
    embedding_quality_max_confidence: float = Field(
        0.95, ge=0.0, le=1.0, description="Maximum confidence for quality display"
    )


