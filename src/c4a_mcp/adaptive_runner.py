# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic | Adaptive]
# :relates-to: uses: "crawl4ai.AdaptiveCrawler", uses: "crawl4ai.AsyncWebCrawler",
#              uses: "config_models.CrawlerConfigYAML", depends-on: "presets.adaptive_factory",
#              extends: "base_runner.BaseRunner"
# :rationale: "Encapsulates adaptive crawling logic using crawl4ai's AdaptiveCrawler. Uses PatchedAdaptiveCrawler
#              to fix hardcoded 5-second timeout for link preview extraction. Separate from CrawlRunner
#              due to different API (digest() vs arun())."
# :references: PRD: "F004 extension", SPEC: "SPEC-F004 extension"
# :contract: pre: "Valid AdaptiveStatisticalInput or AdaptiveEmbeddingInput",
#            post: "Returns RunnerOutput with markdown, confidence, and metrics in metadata"
# :decision_cache: "Separate class from CrawlRunner to maintain single responsibility - different API pattern [ARCH-014].
#                   PatchedAdaptiveCrawler fixes timeout issue for slow websites [ARCH-018].
#                   Refactored to use BaseRunner [ARCH-020]"
# LLM:END

"""
Adaptive crawling runner using crawl4ai's AdaptiveCrawler.

This module provides AdaptiveCrawlRunner which executes query-based intelligent
crawling that stops when sufficient information is gathered based on relevance
metrics (coverage, consistency, saturation).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from datetime import datetime

from crawl4ai import AdaptiveCrawler, AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_configs import LinkPreviewConfig
from crawl4ai.adaptive_crawler import EmbeddingStrategy, AdaptiveConfig, CrawlState
from crawl4ai.models import CrawlResult

from .base_runner import BaseRunner
from .config_models import CrawlerConfigYAML
from .models import RunnerOutput, RunnerInput
from .presets.adaptive_factory import create_adaptive_config
from .presets.models import AdaptiveEmbeddingInput, AdaptiveStatisticalInput, AdaptiveCrawlInput

logger = logging.getLogger(__name__)


class PatchedEmbeddingStrategy:
    """
    Patched EmbeddingStrategy with fixed _get_embedding_llm_config_dict method.

    This is a wrapper that patches the EmbeddingStrategy instance after creation.
    We can't easily subclass EmbeddingStrategy because it's created inside AdaptiveCrawler.

    :hierarchy: [C4A-MCP | Logic | Adaptive | Patch]
    :rationale: "Fixes crawl4ai bug in EmbeddingStrategy._get_embedding_llm_config_dict() that always returns OpenAI config"
    """

    @staticmethod
    def patch_embedding_strategy(strategy_instance):
        """
        Patch an EmbeddingStrategy instance to fix _get_embedding_llm_config_dict.

        Args:
            strategy_instance: EmbeddingStrategy instance to patch
        """

        def patched_get_embedding_llm_config_dict():
            """Patched version that returns None when no valid API key."""
            # First check if config exists
            if not hasattr(strategy_instance, "config") or not strategy_instance.config:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] No config found in EmbeddingStrategy, "
                    "using sentence-transformers"
                )
                return None

            # Check embedding_llm_config directly (this is the source of truth)
            embedding_llm_config = getattr(strategy_instance.config, "embedding_llm_config", None)
            if embedding_llm_config is None:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: embedding_llm_config is None, "
                    "using sentence-transformers"
                )
                return None

            # If embedding_llm_config exists, get the dict version via property
            try:
                config_dict = strategy_instance.config._embedding_llm_config_dict
            except AttributeError:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: _embedding_llm_config_dict "
                    "property not found, using sentence-transformers"
                )
                return None

            # If config_dict is None or empty, return None
            if not config_dict:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: config_dict is None or empty, using sentence-transformers"
                )
                return None

            # Check if config has a valid API key
            api_key = config_dict.get("api_token") or config_dict.get("api_key")
            if not api_key or (isinstance(api_key, str) and not api_key.strip()):
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: LLM config has no valid API key, "
                    "using sentence-transformers | data: {provider: %s, api_key: %s}",
                    config_dict.get("provider", "unknown"),
                    "None" if api_key is None else "empty",
                )
                return None

            # Valid config with API key found
            logger.debug(
                "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: Using LLM config with API key | "
                "data: {provider: %s, has_api_key: True}",
                config_dict.get("provider", "unknown"),
            )
            return config_dict

        # Replace the method
        strategy_instance._get_embedding_llm_config_dict = patched_get_embedding_llm_config_dict
        logger.debug(
            "[C4A-MCP | Logic | Adaptive | Patch] Patched EmbeddingStrategy._get_embedding_llm_config_dict"
        )


class PatchedAdaptiveCrawler(AdaptiveCrawler):
    """
    Patched AdaptiveCrawler with configurable timeout for link preview extraction.

    This class overrides _crawl_with_preview() to use a configurable timeout
    instead of the hardcoded 5-second timeout in the original implementation.
    This is necessary for slow websites that need more time for head extraction.

    Also patches EmbeddingStrategy to fix _get_embedding_llm_config_dict bug.

    :hierarchy: [C4A-MCP | Logic | Adaptive | Patch]
    :rationale: "Fixes hardcoded 5-second timeout in crawl4ai AdaptiveCrawler._crawl_with_preview() for slow websites and EmbeddingStrategy embedding config bug"
    :contract: pre: "timeout > 0", post: "Uses custom timeout for link preview extraction and patched EmbeddingStrategy"
    """

    _link_preview_timeout: int = 30
    _page_timeout: int = 60000

    @classmethod
    def patch_adaptive_crawler(cls, link_preview_timeout: int, page_timeout: int):
        """
        Class method to set the timeouts for all instances of PatchedAdaptiveCrawler.
        This is a workaround to configure the crawler before it's instantiated by crawl4ai's internal logic.
        """
        cls._link_preview_timeout = link_preview_timeout
        cls._page_timeout = page_timeout
        logger.debug(
            "[C4A-MCP | Logic | Adaptive | Patch] PatchedAdaptiveCrawler class-level timeouts set | "
            "data: {link_preview_timeout: %d, page_timeout: %d}",
            link_preview_timeout,
            page_timeout,
        )

    def __init__(
        self,
        crawler: Optional[AsyncWebCrawler] = None,
        config: Optional[Any] = None,  # AdaptiveConfig
        strategy: Optional[Any] = None,  # CrawlStrategy
        link_preview_timeout: Optional[int] = None,  # Deprecated, use patch_adaptive_crawler
        page_timeout: Optional[int] = None,  # Deprecated, use patch_adaptive_crawler
    ):
        """
        Initialize patched AdaptiveCrawler with custom timeout.

        Args:
            crawler: AsyncWebCrawler instance (optional)
            config: AdaptiveConfig instance (optional)
            strategy: CrawlStrategy instance (optional)
            link_preview_timeout: Timeout in seconds for link preview head extraction (default: 30)
            page_timeout: Timeout in milliseconds for page loading (default: 60000)
        """
        super().__init__(crawler, config, strategy)
        # Use class-level patched timeouts if not explicitly provided (for backward compatibility)
        self._link_preview_timeout = (
            link_preview_timeout
            if link_preview_timeout is not None
            else PatchedAdaptiveCrawler._link_preview_timeout
        )
        self._page_timeout = (
            page_timeout if page_timeout is not None else PatchedAdaptiveCrawler._page_timeout
        )

        # Patch EmbeddingStrategy if it was created
        if isinstance(self.strategy, EmbeddingStrategy):
            PatchedEmbeddingStrategy.patch_embedding_strategy(self.strategy)

        logger.debug(
            "[C4A-MCP | Logic | Adaptive | Patch] PatchedAdaptiveCrawler initialized | "
            "data: {link_preview_timeout: %d, page_timeout: %d, strategy_type: %s}",
            self._link_preview_timeout,
            self._page_timeout,
            type(self.strategy).__name__,
        )

    async def _crawl_with_preview(self, url: str, query: str) -> Optional[CrawlResult]:
        """
        Crawl a URL with link preview enabled using configurable timeout.

        Overrides the parent method to use self._link_preview_timeout instead
        of the hardcoded 5-second timeout.

        Args:
            url: URL to crawl
            query: Query string for BM25 scoring

        Returns:
            CrawlResult or None if crawl failed
        """
        config = CrawlerRunConfig(
            link_preview_config=LinkPreviewConfig(
                include_internal=True,
                include_external=False,
                query=query,  # For BM25 scoring
                concurrency=5,
                timeout=self._link_preview_timeout,  # Use configurable timeout
                max_links=50,  # Reasonable limit
                verbose=False,
            ),
            score_links=True,  # Enable intrinsic scoring
            page_timeout=self._page_timeout,  # Add page timeout
        )

        try:
            result = await self.crawler.arun(url=url, config=config)
            # Extract the actual CrawlResult from the container
            if hasattr(result, "_results") and result._results:
                result = result._results[0]

            # Filter out all links that do not have head_data
            if hasattr(result, "links") and result.links:
                result.links["internal"] = [
                    link for link in result.links["internal"] if link.get("head_data")
                ]
                # For now let's ignore external links without head_data
                # result.links['external'] = [link for link in result.links['external'] if link.get('head_data')]

            return result
        except Exception as e:
            logger.warning(
                "[C4A-MCP | Logic | Adaptive | Patch] Error crawling %s | data: {error: %s, error_type: %s}",
                url,
                str(e),
                type(e).__name__,
            )
            return None

    def _create_strategy(self, strategy_name: str):
        """
        Create strategy instance and patch EmbeddingStrategy if needed.

        Overrides parent method to patch EmbeddingStrategy after creation.
        """
        strategy = super()._create_strategy(strategy_name)

        # Patch EmbeddingStrategy to fix _get_embedding_llm_config_dict bug
        if isinstance(strategy, EmbeddingStrategy):
            PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)

        return strategy


# --- Configuration ---
@dataclass
class AdaptiveRunConfiguration:
    """Container for adaptive run configuration and timeouts."""

    adaptive_config: AdaptiveConfig
    link_preview_timeout: int
    page_timeout: int


class AdaptiveCrawlRunner(BaseRunner):
    """
    Executes adaptive crawl sessions using crawl4ai's AdaptiveCrawler.

    Uses digest() method instead of arun() - different API pattern from
    standard crawling. Stops crawling when confidence threshold is reached
    based on query relevance.
    """

    def _build_config(
        self, inputs: Union[RunnerInput, AdaptiveCrawlInput]
    ) -> AdaptiveRunConfiguration:
        """
        Builds the AdaptiveRunConfiguration from inputs.

        Args:
            inputs: The runner input containing URL and config overrides.

        Returns:
            AdaptiveRunConfiguration: The configured run context.
        """
        # 1. Start with defaults
        config_dict = self.default_crawler_config.dict(exclude_none=True)

        # 2. Apply config from inputs if provided
        if inputs.config:
            config_dict.update(inputs.config)

        # 3. Create AdaptiveConfig
        strategy = config_dict.get("strategy", "statistical")
        adaptive_params = config_dict.get("adaptive_config_params", {})
        if adaptive_params is not None and not isinstance(adaptive_params, dict):
            raise ValueError(
                f"adaptive_config_params must be a dict, got {type(adaptive_params).__name__}"
            )

        adaptive_config = create_adaptive_config(strategy, adaptive_params)

        # 4. Extract timeouts (defaulting if not present)
        link_preview_timeout = config_dict.get("timeout", 5) or 5
        # Convert total timeout to ms for page_timeout if needed, or use a default
        page_timeout = 60000
        if config_dict.get("timeout"):
            # If top level timeout is seconds, convert to ms?
            # But crawl4ai usually takes ms for page_timeout.
            # Let's assume standard behavior:
            page_timeout = int(config_dict.get("timeout") * 1000)

        return AdaptiveRunConfiguration(
            adaptive_config=adaptive_config,
            link_preview_timeout=link_preview_timeout,
            page_timeout=page_timeout,
        )

    async def _execute_core(
        self,
        crawler: AsyncWebCrawler,
        config: AdaptiveRunConfiguration,
        inputs: Union[RunnerInput, AdaptiveCrawlInput],
    ) -> CrawlState:
        """
        Executes the adaptive crawl using the built configuration.

        Args:
            crawler: The initialized AsyncWebCrawler instance.
            config: The AdaptiveRunConfiguration object.
            inputs: The runner input.

        Returns:
            CrawlState: The result of the adaptive crawl.
        """
        # Patch AdaptiveCrawler to use our timeouts and callbacks
        PatchedAdaptiveCrawler.patch_adaptive_crawler(
            link_preview_timeout=config.link_preview_timeout, page_timeout=config.page_timeout
        )

        # Patch EmbeddingStrategy if needed
        if config.adaptive_config.strategy == "embedding":
            # Note: We can't easily patch the inner strategy instance before it's created
            # inside AdaptiveCrawler, but our PatchedAdaptiveCrawler triggers PatchedEmbeddingStrategy
            pass

        # Check if browser is available
        if not getattr(crawler, "browser", None):
            # Force start if not started (BaseRunner should have started it, but purely defensive)
            await crawler.start()

        # Create AdaptiveCrawler instance
        # We pass the existing crawler instance to reuse the browser session
        adaptive_crawler = PatchedAdaptiveCrawler(crawler, config.adaptive_config)

        # Add logging callback for progress
        # Note: crawl4ai's AdaptiveCrawler doesn't natively support a progress callback arg in digest yet,
        # but our PatchedAdaptiveCrawler might support it if we extended it.
        # For now, we rely on the LoggerWriter capturing stdout/stderr.

        # Execute crawl
        logger.info(
            "[Adaptive | Crawl] Starting digest | url: %s, strategy: %s",
            inputs.url,
            config.adaptive_config.strategy,
        )

        # digest() returns CrawlState
        query = getattr(inputs, "query", "")
        if not query:
            # Fallback or error? Adaptive requires query.
            # If passed RunnerInput without query, it will fail digest validation probably or yield poor results.
            # But here we just pass it.
            logger.warning("[Adaptive] Input missing 'query' field. Using empty string.")

        crawl_state = await adaptive_crawler.digest(start_url=inputs.url, query=query)

        return crawl_state

    def _process_result(
        self, result: CrawlState, inputs: Union[RunnerInput, AdaptiveCrawlInput]
    ) -> RunnerOutput:
        """
        Process the adaptive crawl result.

        Args:
            result: The CrawlState object returned by digest().
            inputs: The runner input.

        Returns:
            RunnerOutput with aggregated markdown and metrics.
        """
        # Extract metrics
        # CrawlState.metrics is a dict
        metrics_dict = result.metrics if hasattr(result, "metrics") else {}
        confidence = metrics_dict.get("confidence", 0.0)

        metrics = {
            "coverage": metrics_dict.get("coverage", 0.0),
            "consistency": metrics_dict.get("consistency", 0.0),
            "saturation": metrics_dict.get("saturation", 0.0),
        }

        # Add strategy-specific metrics if available
        if inputs.config and inputs.config.get("strategy") == "embedding":
            metrics["validation_confidence"] = metrics_dict.get("validation_confidence", 0.0)
            metrics["avg_min_distance"] = metrics_dict.get("avg_min_distance", 0.0)

        # pages_crawled
        pages_crawled = (
            len(result.crawled_urls)
            if hasattr(result, "crawled_urls") and result.crawled_urls
            else 0
        )
        metrics["pages_crawled"] = pages_crawled

        logger.info(
            "[Adaptive | Result] Processed | url: %s, confidence: %.2f, pages: %d, metrics: %s",
            inputs.url,
            confidence,
            pages_crawled,
            metrics,
        )

        # Aggregate markdown from knowledge_base
        markdown_content = ""
        knowledge_base = getattr(result, "knowledge_base", [])
        if knowledge_base:
            markdown_parts = []
            for item in knowledge_base:
                # item should be CrawlResult
                if hasattr(item, "markdown") and item.markdown:
                    # markdown might be object or string depending on version
                    md = getattr(item.markdown, "raw_markdown", None)
                    if md is None:
                        md = str(item.markdown)
                    if md:
                        markdown_parts.append(md)

            if markdown_parts:
                markdown_content = "\n\n---\n\n".join(markdown_parts)

        return RunnerOutput(
            markdown=markdown_content,
            metadata={
                "url": inputs.url,
                "title": f"Adaptive Crawl: {inputs.query}",
                "status": 200,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence,
                "metrics": metrics,
            },
            error=None,
        )
