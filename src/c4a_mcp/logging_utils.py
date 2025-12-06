# LLM:METADATA
# :hierarchy: [C4A-MCP | Logging]
# :rationale: "Centralized logging config/utils. Contains output redirection."
# :contract: pre: "None", post: "Logging configured, helpers available"
# LLM:END

import logging
import os
import sys
from pathlib import Path
from typing import Any

from crawl4ai.async_logger import AsyncLoggerBase


class ConsecutiveDedupFilter(logging.Filter):
    """Filter that drops consecutive duplicate log messages."""

    def __init__(self) -> None:
        super().__init__()
        self._last: tuple[int, str] | None = None

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        key = (record.levelno, record.getMessage())
        if key == self._last:
            return False
        self._last = key
        return True


class LoggerWriter:
    """Stream writer that redirects output to a logger in real-time.

    This replaces StringIO buffering to enable live log streaming during
    long-running operations.
    """

    def __init__(self, logger_instance: logging.Logger, level: int = logging.DEBUG):
        """Initialize LoggerWriter.

        Args:
            logger_instance: Logger to write to
            level: Log level for messages (default: DEBUG)
        """
        self.logger = logger_instance
        self.level = level
        self.buffer = ""
        self._last_line: str | None = None

    def write(self, message: str) -> int:
        """Write message to logger.

        Buffers partial lines and logs complete lines immediately.

        Args:
            message: Text to write

        Returns:
            Number of characters written
        """
        if message:
            # Buffer the message
            self.buffer += message

            # If we have complete lines, log them
            if "\n" in self.buffer or "\r" in self.buffer:
                # Handle both newline types
                lines = self.buffer.replace("\r", "\n").split("\n")
                # Log all complete lines
                for line in lines[:-1]:
                    cleaned = line.strip()
                    if not cleaned:
                        continue
                    # Skip immediate duplicates to avoid double-logging
                    if cleaned == self._last_line:
                        continue
                    self._last_line = cleaned
                    self.logger.log(self.level, "[crawl4ai] %s", cleaned)
                # Keep the incomplete line in buffer
                self.buffer = lines[-1]

        return len(message)

    def flush(self) -> None:
        """Flush any remaining buffered content."""
        if self.buffer.strip():
            cleaned = self.buffer.strip()
            if cleaned != self._last_line:
                self.logger.log(self.level, "[crawl4ai] %s", cleaned)
                self._last_line = cleaned
            self.buffer = ""


class Crawl4AiLoggerAdapter(AsyncLoggerBase):
    """Adapter to route crawl4ai logs to standard python logging."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def debug(self, message: str, tag: str = "DEBUG", **kwargs: Any) -> None:
        self.logger.debug("[crawl4ai] [%s] %s", tag, message)

    def info(self, message: str, tag: str = "INFO", **kwargs: Any) -> None:
        self.logger.info("[crawl4ai] [%s] %s", tag, message)

    def success(self, message: str, tag: str = "SUCCESS", **kwargs: Any) -> None:
        self.logger.info("[crawl4ai] [%s] %s", tag, message)

    def warning(self, message: str, tag: str = "WARNING", **kwargs: Any) -> None:
        self.logger.warning("[crawl4ai] [%s] %s", tag, message)

    def error(self, message: str, tag: str = "ERROR", **kwargs: Any) -> None:
        self.logger.error("[crawl4ai] [%s] %s", tag, message)

    def url_status(
        self,
        url: str,
        success: bool,
        timing: float,
        tag: str = "FETCH",
        url_length: int = 100,
    ) -> None:
        level = logging.INFO if success else logging.ERROR
        status = "SUCCESS" if success else "FAILED"
        self.logger.log(
            level,
            "[crawl4ai] [%s] URL: %s | Status: %s | Time: %.2fs",
            tag,
            url,
            status,
            timing,
        )

    def error_status(
        self,
        url: str,
        error: str,
        tag: str = "ERROR",
        url_length: int = 100,
    ) -> None:
        self.logger.error(
            "[crawl4ai] [%s] URL: %s | Error: %s",
            tag,
            url,
            error,
        )


def setup_logging(log_file_path: Path | None = None) -> None:
    """
    Configure application-wide logging.

    Respects LOGLEVEL environment variable.
    Configures handlers for both stderr (monitoring) and file output (persistent).

    Args:
        log_file_path: Optional path to log file. If None, defaults to
            `server_debug.log` in the project root directory.

    Note:
        CRITICAL: Must use stderr to keep stdout clean for MCP JSON-RPC.
    """
    # Determine log level from environment variable
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()

    # Default log file path: project root / server_debug.log
    if log_file_path is None:
        # Assume this file is in src/c4a_mcp/, so go up 2 levels
        log_file_path = Path(__file__).parent.parent.parent / "server_debug.log"

    # Configure logging with dual handlers
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),  # Real-time monitoring
            logging.FileHandler(
                log_file_path,
                mode="a",
                encoding="utf-8",
            ),  # Persistent debugging
        ],
    )

    # Attach a dedup filter to all handlers to suppress duplicates
    dedup_filter = ConsecutiveDedupFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(dedup_filter)
