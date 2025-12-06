# LLM:METADATA
# :hierarchy: [C4A-MCP | Logging]
# :rationale: "Centralized logging configuration to ensure consistent logging setup across the application. Must use stderr to keep stdout clean for MCP JSON-RPC communication."
# :contract: pre: "None", post: "Logging is configured with handlers for stderr and file output"
# LLM:END

import logging
import os
import sys
from pathlib import Path


class ConsecutiveDedupFilter(logging.Filter):
    """Filter that drops consecutive duplicate log messages (same level+message)."""

    def __init__(self) -> None:
        super().__init__()
        self._last: tuple[int, str] | None = None

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        key = (record.levelno, record.getMessage())
        if key == self._last:
            return False
        self._last = key
        return True


def setup_logging(log_file_path: Path | None = None) -> None:
    """
    Configure application-wide logging.

    Respects LOGLEVEL environment variable (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    Configures handlers for both stderr (for real-time monitoring) and file output
    (for persistent debugging).

    Args:
        log_file_path: Optional path to log file. If None, defaults to
            `server_debug.log` in the project root directory.

    Note:
        CRITICAL: Must use stderr to keep stdout clean for MCP JSON-RPC communication.
    """
    # Determine log level from environment variable
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()

    # Default log file path: project root / server_debug.log
    if log_file_path is None:
        # Assume this file is in src/c4a_mcp/, so go up 2 levels to project root
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

    # Attach a dedup filter to all handlers to suppress immediate duplicate lines.
    dedup_filter = ConsecutiveDedupFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(dedup_filter)

