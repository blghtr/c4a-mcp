# LLM:METADATA
# :hierarchy: [C4A-MCP | Logic | Base]
# :relates-to: implemented-by: "runner_tool.CrawlRunner", implemented-by: "adaptive_runner.AdaptiveCrawlRunner"
# :rationale: "Abstract base class encapsulating shared infrastructure logic (logging, lifecycle, error handling) to prevent duplication."
# :contract: pre: "Subclasses must implement _build_config and _execute_core", post: "Standardized execution flow"
# :decision_cache: "Extracted from runner_tool.py and adaptive_runner.py to reduce 80% duplication [ARCH-020]"
# LLM:END

import asyncio
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig

from .models import RunnerOutput
from .logging_utils import Crawl4AiLoggerAdapter
from .crawler_registry import (
    register as register_crawler,
    deregister as deregister_crawler,
    close_crawler,
    kill_child_browsers,
)

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Abstract base class for crawl execution logic.

    Implements the Template Method Pattern for the crawl lifecycle:
    1. Setup Logging
    2. Build Config (Abstract)
    3. Init/Register Crawler
    4. Execute Strategy (Abstract)
    5. process Result
    6. Error Handling & Cleanup
    """

    def __init__(
        self,
        default_crawler_config: Any,
        browser_config: BrowserConfig,
        crawler_class: Any = AsyncWebCrawler,
    ):
        """Initialize BaseRunner.

        Args:
            default_crawler_config: Default crawler settings
            browser_config: Pre-created browser configuration
            crawler_class: Class to use for crawling (default: AsyncWebCrawler)
        """
        self.default_crawler_config = default_crawler_config
        self.browser_config = browser_config
        self.crawler_class = crawler_class
        logger.info(
            "[%s] Initialized | data: {browser_type: %s, headless: %s}",
            self.__class__.__name__,
            browser_config.browser_type,
            browser_config.headless,
        )

    async def run(self, inputs: Any) -> RunnerOutput:
        """
        Executes the crawl logic with standardized lifecycle management.

        Args:
            inputs: Validated input parameters (RunnerInput or AdaptiveInput)

        Returns:
            RunnerOutput containing markdown or error info.
        """
        run_config = None
        try:
            logger.debug(
                "[%s | Run] Starting execution | data: {url: %s, type: %s}",
                self.__class__.__name__,
                inputs.url,
                type(inputs).__name__,
            )

            # 1. Setup Logger Adapter
            # Direct injection of logger into crawl4ai to avoid global
            # sys.stdout patching
            crawler_logger = Crawl4AiLoggerAdapter(logger)

            # 2. Build Config (Abstract)
            run_config = self._build_config(inputs)

            # 3. Init & Register Crawler
            crawler = self.crawler_class(config=self.browser_config, logger=crawler_logger)
            register_crawler(crawler)
            entered = False
            task: asyncio.Task | None = None

            try:
                async with crawler as active_crawler:
                    entered = True

                    # 4. Execute Strategy (Abstract)
                    # We wrap execution in a task to enable cancellation
                    task = asyncio.create_task(
                        self._execute_core(active_crawler, run_config, inputs)
                    )
                    result = await task

                    # 5. Process Result
                    return self._process_result(result, inputs)

            except asyncio.CancelledError:
                logger.warning(
                    "[%s | Run] Cancelled during crawl; aggressive cleanup",
                    self.__class__.__name__,
                )
                # Force browser closure if task is still running
                if task and not task.done():
                    task.cancel()
                    await self._force_browser_close(crawler)

                kill_child_browsers(logger, timeout=1.0)
                raise

            finally:
                # Ensure graceful cleanup on exit
                cleanup_error = None
                try:
                    if not entered:
                        await close_crawler(crawler, logger, timeout=5.0)
                except Exception as e:
                    cleanup_error = e
                    logger.warning("[%s | Run] Cleanup failed: %s", self.__class__.__name__, e)
                finally:
                    deregister_crawler(crawler)
                    if cleanup_error is not None or not entered:
                        kill_child_browsers(logger, timeout=2.0)

        except Exception as e:
            return self._handle_error(e, run_config)

    @abstractmethod
    def _build_config(self, inputs: Any) -> Any:
        """Build the specific configuration object for the crawl."""
        pass

    @abstractmethod
    async def _execute_core(self, crawler: AsyncWebCrawler, config: Any, inputs: Any) -> Any:
        """Execute the specific crawl strategy (arun or digest)."""
        pass

    def _process_result(self, result: Any, inputs: Any) -> RunnerOutput:
        """Process the raw result into RunnerOutput. Can be overridden."""
        # Default implementation for standard crawl result
        markdown_content = self._extract_markdown(result)

        return RunnerOutput(
            markdown=markdown_content,
            metadata={
                "url": inputs.url,  # Use input URL as fallback
                "title": (
                    result.metadata.get("title")
                    if hasattr(result, "metadata") and result.metadata
                    else None
                ),
                "timestamp": datetime.now().isoformat(),
                "status": getattr(result, "status_code", 200),
            },
            error=None,
        )

    def _extract_markdown(self, result: Any) -> str:
        """Shared markdown extraction logic."""
        if not result:
            return ""

        # Handle list results (deep crawl)
        if isinstance(result, list):
            parts = []
            for res in result:
                parts.append(self._extract_single_markdown(res))
            return "\n\n---\n\n".join(filter(None, parts))

        return self._extract_single_markdown(result)

    def _extract_single_markdown(self, result: Any) -> str:
        """Extract markdown from a single result object."""
        if not result or not hasattr(result, "markdown"):
            return ""

        md = result.markdown
        if md is None:
            return ""

        if isinstance(md, str):
            return md

        # Handle MarkdownGenerationResult object
        content = getattr(md, "raw_markdown", "")
        if not content:
            content = getattr(md, "fit_markdown", "")

        return content or ""

    async def _force_browser_close(self, crawler: AsyncWebCrawler):
        """Forcefully close the browser instance."""
        try:
            browser = getattr(crawler, "browser", None)
            if browser:
                await browser.close()
                logger.debug("[%s | Cleanup] Forced browser close", self.__class__.__name__)
        except Exception as e:
            logger.warning(
                "[%s | Cleanup] Failed to force close browser: %s",
                self.__class__.__name__,
                e,
            )

    def _handle_error(self, e: Exception, config: Any) -> RunnerOutput:
        """Map exceptions to user-friendly error messages."""
        error_message = str(e).lower()
        formatted_error = ""

        logger.error(
            "[%s | Error] Execution failed | data: {type: %s, error: %s}",
            self.__class__.__name__,
            type(e).__name__,
            str(e),
        )
        logger.debug("Traceback: %s", traceback.format_exc())

        if "timeout" in error_message:
            formatted_error = f"Timeout error: {e}"
        elif any(x in error_message for x in ["network", "connection", "http error"]):
            formatted_error = f"Network error: {e}"
        elif "script" in error_message or "js error" in error_message:
            formatted_error = f"Script execution failed: {e}"
        else:
            formatted_error = f"An unexpected error occurred: {e}"

        return RunnerOutput(markdown="", error=formatted_error)
