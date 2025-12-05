# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic | Adaptive]
# :relates-to: uses: "crawl4ai.AdaptiveCrawler", uses: "crawl4ai.AsyncWebCrawler", uses: "config_models.CrawlerConfigYAML", depends-on: "presets.adaptive_factory"
# :rationale: "Encapsulates adaptive crawling logic using crawl4ai's AdaptiveCrawler. Uses PatchedAdaptiveCrawler to fix hardcoded 5-second timeout for link preview extraction. Separate from CrawlRunner due to different API (digest() vs arun())."
# :references: PRD: "F004 extension", SPEC: "SPEC-F004 extension"
# :contract: pre: "Valid AdaptiveStatisticalInput or AdaptiveEmbeddingInput", post: "Returns RunnerOutput with markdown, confidence, and metrics in metadata"
# :decision_cache: "Separate class from CrawlRunner to maintain single responsibility - different API pattern [ARCH-014]. PatchedAdaptiveCrawler fixes timeout issue for slow websites [ARCH-018]"
# LLM:END

"""
Adaptive crawling runner using crawl4ai's AdaptiveCrawler.

This module provides AdaptiveCrawlRunner which executes query-based intelligent
crawling that stops when sufficient information is gathered based on relevance
metrics (coverage, consistency, saturation).
"""

import contextlib
import io
import logging
import traceback
from datetime import datetime
from typing import Any, Optional, Union

from crawl4ai import AdaptiveCrawler, AsyncWebCrawler, BrowserConfig
from crawl4ai.async_configs import CrawlerRunConfig, LinkPreviewConfig
from crawl4ai.adaptive_crawler import EmbeddingStrategy
from crawl4ai.models import CrawlResult

from .config_models import CrawlerConfigYAML
from .models import RunnerOutput
from .presets.adaptive_factory import create_adaptive_config
from .presets.models import AdaptiveEmbeddingInput, AdaptiveStatisticalInput

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
            if not hasattr(strategy_instance, 'config') or not strategy_instance.config:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] No config found in EmbeddingStrategy, using sentence-transformers"
                )
                return None
            
            # Check embedding_llm_config directly (this is the source of truth)
            embedding_llm_config = getattr(strategy_instance.config, 'embedding_llm_config', None)
            if embedding_llm_config is None:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: embedding_llm_config is None, using sentence-transformers"
                )
                return None
            
            # If embedding_llm_config exists, get the dict version via property
            try:
                config_dict = strategy_instance.config._embedding_llm_config_dict
            except AttributeError:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: _embedding_llm_config_dict property not found, using sentence-transformers"
                )
                return None
            
            # If config_dict is None or empty, return None
            if not config_dict:
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: config_dict is None or empty, using sentence-transformers"
                )
                return None
            
            # Check if config has a valid API key
            api_key = config_dict.get('api_token') or config_dict.get('api_key')
            if not api_key or (isinstance(api_key, str) and not api_key.strip()):
                logger.debug(
                    "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: LLM config has no valid API key, using sentence-transformers | "
                    "data: {provider: %s, api_key: %s}",
                    config_dict.get('provider', 'unknown'),
                    'None' if api_key is None else 'empty',
                )
                return None
            
            # Valid config with API key found
            logger.debug(
                "[C4A-MCP | Logic | Adaptive | Patch] EmbeddingStrategy: Using LLM config with API key | "
                "data: {provider: %s, has_api_key: True}",
                config_dict.get('provider', 'unknown'),
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
    
    def __init__(
        self,
        crawler: Optional[AsyncWebCrawler] = None,
        config: Optional[Any] = None,  # AdaptiveConfig
        strategy: Optional[Any] = None,  # CrawlStrategy
        link_preview_timeout: int = 30,
        page_timeout: int = 60000,
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
        self._link_preview_timeout = link_preview_timeout
        self._page_timeout = page_timeout
        
        # Patch EmbeddingStrategy if it was created
        if isinstance(self.strategy, EmbeddingStrategy):
            PatchedEmbeddingStrategy.patch_embedding_strategy(self.strategy)
        
        logger.debug(
            "[C4A-MCP | Logic | Adaptive | Patch] PatchedAdaptiveCrawler initialized | "
            "data: {link_preview_timeout: %d, page_timeout: %d, strategy_type: %s}",
            link_preview_timeout,
            page_timeout,
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
                verbose=False
            ),
            score_links=True,  # Enable intrinsic scoring
            page_timeout=self._page_timeout,  # Add page timeout
        )
        
        try:
            result = await self.crawler.arun(url=url, config=config)
            # Extract the actual CrawlResult from the container
            if hasattr(result, '_results') and result._results:
                result = result._results[0]

            # Filter out all links that do not have head_data
            if hasattr(result, 'links') and result.links:
                result.links['internal'] = [link for link in result.links['internal'] if link.get('head_data')]
                # For now let's ignore external links without head_data
                # result.links['external'] = [link for link in result.links['external'] if link.get('head_data')]

            return result
        except Exception as e:
            logger.warning(
                "[C4A-MCP | Logic | Adaptive | Patch] Error crawling %s | "
                "data: {error: %s, error_type: %s}",
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


class AdaptiveCrawlRunner:
    """
    Executes adaptive crawl sessions using crawl4ai's AdaptiveCrawler.
    
    Uses digest() method instead of arun() - different API pattern from
    standard crawling. Stops crawling when confidence threshold is reached
    based on query relevance.
    
    Attributes:
        default_crawler_config: Default crawler configuration from YAML
        browser_config: Pre-created browser configuration
    """
    
    def __init__(
        self,
        default_crawler_config: CrawlerConfigYAML,
        browser_config: BrowserConfig,
    ):
        """Initialize AdaptiveCrawlRunner with default configs.
        
        Args:
            default_crawler_config: Default crawler settings (from YAML + overrides)
            browser_config: Pre-created browser configuration
        """
        self.default_crawler_config = default_crawler_config
        self.browser_config = browser_config
        logger.info(
            "[C4A-MCP | Logic | Adaptive] AdaptiveCrawlRunner initialized | "
            "data: {browser_type: %s, headless: %s}",
            browser_config.browser_type,
            browser_config.headless,
        )
    
    async def run(
        self, inputs: Union[AdaptiveStatisticalInput, AdaptiveEmbeddingInput]
    ) -> RunnerOutput:
        """
        Execute adaptive crawl with query-based stopping.
        
        Args:
            inputs: AdaptiveRunnerInput with url, query, and config
        
        Returns:
            RunnerOutput containing markdown, metadata (with confidence/metrics), error
        """
        try:
            # Extract strategy and parameters from config
            # Preserve dict-like config access for backward compatibility
            config_dict = inputs.config or {}
            strategy = config_dict.get("strategy", "statistical")
            adaptive_params = config_dict.get("adaptive_config_params", {})
            
            # Validate config structure
            if strategy not in ("statistical", "embedding"):
                raise ValueError(
                    f"Invalid strategy: {strategy}. "
                    "Must be one of: 'statistical', 'embedding'"
                )
            if not isinstance(adaptive_params, dict):
                raise ValueError(
                    f"adaptive_config_params must be a dict, got: {type(adaptive_params).__name__}"
                )
            
            # Create AdaptiveConfig from parameters using factory
            adaptive_config = create_adaptive_config(strategy, adaptive_params)
            
            # Extract timeout configuration
            # Use timeout from inputs.config, fallback to default_crawler_config, then default to 30 seconds
            link_preview_timeout = 30  # Default: 30 seconds for head extraction
            page_timeout_ms = 60000  # Default: 60 seconds for page loading
            
            if config_dict.get("timeout") is not None:
                # User-provided timeout in seconds
                user_timeout = int(config_dict["timeout"])
                link_preview_timeout = user_timeout
                page_timeout_ms = user_timeout * 1000
            elif self.default_crawler_config.timeout is not None:
                # Use timeout from default config
                config_timeout = int(self.default_crawler_config.timeout)
                link_preview_timeout = config_timeout
                page_timeout_ms = config_timeout * 1000
            
            logger.debug(
                "[C4A-MCP | Logic | Adaptive] Starting adaptive crawl | "
                "data: {url: %s, query: %s, strategy: %s, link_preview_timeout: %d, page_timeout_ms: %d}",
                inputs.url,
                inputs.query[:50] if len(inputs.query) > 50 else inputs.query,
                strategy,
                link_preview_timeout,
                page_timeout_ms,
            )
            
            # CRITICAL: Redirect stdout/stderr to prevent crawl4ai progress messages
            # from breaking MCP JSON-RPC protocol (which expects only JSON on stdout)
            # But we want to log sentence-transformers download progress in real-time
            class LoggingStringIO(io.StringIO):
                """StringIO that also logs to logger in real-time."""
                def __init__(self, logger_instance, prefix):
                    super().__init__()
                    self.logger = logger_instance
                    self.prefix = prefix
                    self.buffer = ""
                
                def write(self, s):
                    if s:
                        super().write(s)
                        self.buffer += s
                        # Log complete lines immediately
                        if '\n' in s or '\r' in s:
                            lines = self.buffer.split('\n')
                            self.buffer = lines[-1]  # Keep incomplete line
                            for line in lines[:-1]:
                                if line.strip():
                                    self.logger.info(
                                        f"[C4A-MCP | Logic | Adaptive] {self.prefix}: {line.strip()}"
                                    )
                
                def getvalue(self):
                    """Override getvalue to ensure it works correctly."""
                    return super().getvalue()
            
            stdout_capture = LoggingStringIO(logger, "stdout")
            stderr_capture = LoggingStringIO(logger, "stderr")
            
            # Log before starting crawl, especially for embedding strategy
            if strategy == "embedding":
                logger.info(
                    "[C4A-MCP | Logic | Adaptive] Starting embedding crawl - "
                    "first model load may take several minutes to download from HuggingFace"
                )
            
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                async with AsyncWebCrawler(config=self.browser_config) as crawler:
                    # Create PatchedAdaptiveCrawler with configurable timeout
                    adaptive = PatchedAdaptiveCrawler(
                        crawler,
                        adaptive_config,
                        link_preview_timeout=link_preview_timeout,
                        page_timeout=page_timeout_ms,
                    )
                    
                    # Execute adaptive crawl
                    logger.debug(
                        "[C4A-MCP | Logic | Adaptive] Starting digest() | "
                        "data: {url: %s, query: %s, strategy: %s}",
                        inputs.url,
                        inputs.query[:100] if len(inputs.query) > 100 else inputs.query,
                        strategy,
                    )
                    
                    # For embedding strategy, log that model loading is about to happen
                    if strategy == "embedding":
                        logger.info(
                            "[C4A-MCP | Logic | Adaptive] Embedding strategy: "
                            "model will be loaded on first use (may take 2-5 minutes)"
                        )
                    
                    result = await adaptive.digest(
                        start_url=inputs.url,
                        query=inputs.query
                    )
                    
                    # Log result structure for debugging
                    logger.debug(
                        "[C4A-MCP | Logic | Adaptive] Digest completed | "
                        "data: {result_type: %s, has_markdown: %s, has_url: %s, has_metadata: %s, "
                        "has_crawled_urls: %s, result_attrs: %s}",
                        type(result).__name__,
                        hasattr(result, "markdown"),
                        hasattr(result, "url"),
                        hasattr(result, "metadata"),
                        hasattr(result, "crawled_urls"),
                        [attr for attr in dir(result) if not attr.startswith("_")][:10],
                    )
            
            # Log captured output for debugging
            # This includes sentence-transformers download progress
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()
            if captured_stdout:
                logger.info(
                    "[C4A-MCP | Logic | Adaptive] Captured stdout (may include model download progress) | "
                    "data: {output: %s}",
                    captured_stdout[:1000],  # Increased limit to see more of download progress
                )
            if captured_stderr:
                # stderr often contains sentence-transformers download progress
                logger.info(
                    "[C4A-MCP | Logic | Adaptive] Captured stderr (may include model download progress) | "
                    "data: {output: %s}",
                    captured_stderr[:1000],  # Increased limit to see more of download progress
                )
            
            # Extract markdown content
            # AdaptiveCrawler.digest() returns result with markdown
            # TODO(REVIEWER): The markdown extraction logic uses multiple hasattr/getattr checks. Consider
            # documenting the expected result structure or adding type hints to make the contract explicit.
            markdown_content = ""
            
            # Diagnostic logging for markdown extraction
            logger.debug(
                "[C4A-MCP | Logic | Adaptive] Extracting markdown | "
                "data: {has_markdown_attr: %s, markdown_is_none: %s, markdown_type: %s}",
                hasattr(result, "markdown"),
                not hasattr(result, "markdown") or result.markdown is None,
                type(getattr(result, "markdown", None)).__name__ if hasattr(result, "markdown") else "N/A",
            )
            
            if hasattr(result, "markdown") and result.markdown:
                if isinstance(result.markdown, str):
                    markdown_content = result.markdown
                    logger.debug(
                        "[C4A-MCP | Logic | Adaptive] Markdown is string | "
                        "data: {length: %d, preview: %s}",
                        len(markdown_content),
                        markdown_content[:200] if markdown_content else "",
                    )
                else:
                    # Try to extract from MarkdownGenerationResult
                    markdown_obj = result.markdown
                    logger.debug(
                        "[C4A-MCP | Logic | Adaptive] Markdown is object | "
                        "data: {type: %s, has_raw_markdown: %s, has_fit_markdown: %s, attrs: %s}",
                        type(markdown_obj).__name__,
                        hasattr(markdown_obj, "raw_markdown"),
                        hasattr(markdown_obj, "fit_markdown"),
                        [attr for attr in dir(markdown_obj) if not attr.startswith("_")][:10],
                    )
                    markdown_content = getattr(markdown_obj, "raw_markdown", None)
                    if markdown_content is None:
                        markdown_content = getattr(markdown_obj, "fit_markdown", None)
                    if markdown_content is None:
                        markdown_content = str(markdown_obj)
                    # Ensure markdown_content is a string
                    if markdown_content is None:
                        markdown_content = ""
                    
                    logger.debug(
                        "[C4A-MCP | Logic | Adaptive] Extracted markdown from object | "
                        "data: {length: %d, preview: %s}",
                        len(markdown_content) if markdown_content else 0,
                        markdown_content[:200] if markdown_content else "",
                    )
            else:
                logger.warning(
                    "[C4A-MCP | Logic | Adaptive] No markdown found in result | "
                    "data: {has_markdown_attr: %s, markdown_value: %s}",
                    hasattr(result, "markdown"),
                    str(getattr(result, "markdown", None))[:100] if hasattr(result, "markdown") else "N/A",
                )
                
                # Try to get markdown from knowledge_base if available
                if hasattr(adaptive, "state") and hasattr(adaptive.state, "knowledge_base"):
                    kb = adaptive.state.knowledge_base
                    logger.debug(
                        "[C4A-MCP | Logic | Adaptive] Checking knowledge_base | "
                        "data: {kb_length: %d}",
                        len(kb) if kb else 0,
                    )
                    if kb and len(kb) > 0:
                        # Try to aggregate markdown from knowledge base
                        kb_markdowns = []
                        for i, kb_item in enumerate(kb[:3]):  # Check first 3 items
                            if hasattr(kb_item, "markdown"):
                                kb_md = kb_item.markdown
                                if isinstance(kb_md, str):
                                    kb_markdowns.append(kb_md)
                                elif hasattr(kb_md, "raw_markdown"):
                                    kb_markdowns.append(kb_md.raw_markdown)
                                logger.debug(
                                    "[C4A-MCP | Logic | Adaptive] KB item %d markdown | "
                                    "data: {has_markdown: %s, type: %s, length: %d}",
                                    i,
                                    hasattr(kb_item, "markdown"),
                                    type(getattr(kb_item, "markdown", None)).__name__,
                                    len(str(getattr(kb_item, "markdown", ""))),
                                )
                        if kb_markdowns:
                            markdown_content = "\n\n---\n\n".join(kb_markdowns)
                            logger.debug(
                                "[C4A-MCP | Logic | Adaptive] Aggregated markdown from knowledge_base | "
                                "data: {items_used: %d, total_length: %d}",
                                len(kb_markdowns),
                                len(markdown_content),
                            )
            
            # Extract confidence and metrics from AdaptiveCrawler
            # NOTE(REVIEWER): Using getattr with defaults is defensive but makes it unclear what the actual
            # AdaptiveCrawler API provides. Consider checking crawl4ai documentation or adding assertions
            # to fail fast if expected attributes are missing.
            confidence = getattr(adaptive, "confidence", 0.0)
            
            # Try to extract metrics from adaptive crawler
            # Metrics may be available via adaptive.metrics or result.metrics
            # TODO(REVIEWER): The metrics extraction assumes specific keys (coverage, consistency, saturation).
            # Document expected metrics structure or make extraction more robust to handle missing keys gracefully.
            metrics = {}
            if hasattr(adaptive, "metrics"):
                metrics_dict = adaptive.metrics
                if isinstance(metrics_dict, dict):
                    metrics = {
                        "coverage": metrics_dict.get("coverage", 0.0),
                        "consistency": metrics_dict.get("consistency", 0.0),
                        "saturation": metrics_dict.get("saturation", 0.0),
                    }
            
            # Get pages crawled count
            pages_crawled = 0
            if hasattr(result, "crawled_urls"):
                pages_crawled = len(result.crawled_urls) if result.crawled_urls else 0
            metrics["pages_crawled"] = pages_crawled
            
            # Get URL and title from result
            result_url = inputs.url
            result_title = None
            if hasattr(result, "url"):
                result_url = result.url
            if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                result_title = result.metadata.get("title")
            
            logger.info(
                "[C4A-MCP | Logic | Adaptive] Adaptive crawl completed | "
                "data: {url: %s, confidence: %.2f, pages_crawled: %d}",
                result_url,
                confidence,
                pages_crawled,
            )
            
            return RunnerOutput(
                markdown=markdown_content,
                metadata={
                    "url": result_url,
                    "title": result_title,
                    "timestamp": datetime.now().isoformat(),
                    "status": getattr(result, "status_code", 200) if hasattr(result, "status_code") else 200,
                    "confidence": confidence,
                    "metrics": metrics,
                },
                error=None,
            )
        except Exception as e:
            # Catch any unexpected errors and try to categorize them
            error_message = str(e).lower()
            formatted_error = ""
            
            # Log full traceback for debugging
            logger.error(
                "[C4A-MCP | Logic | Adaptive] Error during adaptive crawl execution | "
                "data: {error_type: %s, error: %s}",
                type(e).__name__,
                str(e),
            )
            logger.debug(
                "[C4A-MCP | Logic | Adaptive] Full traceback | data: {traceback: %s}",
                traceback.format_exc(),
            )
            
            if "timeout" in error_message:
                formatted_error = f"Timeout during adaptive crawl: {e}"
            elif (
                "network" in error_message
                or "connection" in error_message
                or "http error" in error_message
            ):
                formatted_error = f"Network error: {e}"
            elif "embedding" in error_message or "llm" in error_message:
                formatted_error = f"Embedding/LLM error: {e}"
            else:
                formatted_error = f"An unexpected error occurred: {e}"
            
            # Return only sanitized error message to client
            return RunnerOutput(markdown="", error=formatted_error)

