# LLM:METADATA
# :hierarchy: [C4A-MCP | Server]
# :relates-to: uses: "fastmcp.FastMCP", uses: "runner_tool", uses: "config_models.AppConfig"
# :rationale: "Entry point for the MCP server, handling tool registration and request routing."
# :references: PRD: "F001", SPEC: "SPEC-F001, SPEC-F004"
# :contract: invariant: "Server must remain responsive and handle exceptions gracefully"
# :decision_cache: "Using FastMCP standalone for richer features (auth, middleware, deployment) [ARCH-004]. Lifespan context manages CrawlRunner lifecycle [ARCH-012]"
# LLM:END

import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP, Context

from .config_models import AppConfig
from .models import RunnerInput, RunnerOutput
from .presets.preset_tools import crawl_deep, crawl_deep_smart, scrape_page
from .runner_tool import CrawlRunner

# Configure logging
# Respect LOGLEVEL environment variable (DEBUG, INFO, WARNING, ERROR)
# CRITICAL: Must use stderr to keep stdout clean for MCP JSON-RPC communication
# TODO(REVIEWER): Move to separate file
log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            Path(__file__).parent.parent.parent / "server_debug.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)

# Configure logger with hierarchy path format
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Loading (Startup)
# ============================================================================

# Load default config from package
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "defaults.yaml"

try:
    app_config = AppConfig.from_yaml(DEFAULT_CONFIG_PATH)
    logger.info(
        "[C4A-MCP | Server] Loaded default config | data: {path: %s}",
        DEFAULT_CONFIG_PATH,
    )
except Exception as e:
    logger.error(
        "[C4A-MCP | Server] Failed to load default config | data: {path: %s, error: %s}",
        DEFAULT_CONFIG_PATH,
        str(e),
    )
    raise

# Check for override config path in sys.argv (from MCP args)
# Expected format: --config=/path/to/overrides.yaml
for arg in sys.argv[1:]:
    if arg.startswith("--config="):
        override_path = Path(arg.split("=", 1)[1])
        try:
            override_config = AppConfig.from_yaml(override_path)
            # Merge: override takes precedence
            app_config.browser = app_config.browser.merge(override_config.browser)
            app_config.crawler = app_config.crawler.merge(override_config.crawler)
            logger.info(
                "[C4A-MCP | Server] Loaded config overrides | data: {path: %s}",
                override_path,
            )
        except Exception as e:
            logger.error(
                "[C4A-MCP | Server] Failed to load override config | "
                "data: {path: %s, error: %s}",
                override_path,
                str(e),
            )
            # Don't raise - continue with defaults
            # NOTE(REVIEWER): Fallback to defaults is safe, but user might miss the error if logs aren't monitored.
            # TODO(REVIEWER): Consider adding a `--strict-config` flag to exit on override failure.
        break

# ============================================================================
# Lifespan Context Manager
# ============================================================================


@asynccontextmanager
async def lifespan(mcp_server: FastMCP):
    """
    Manage server lifecycle: initialize resources on startup, cleanup on shutdown.

    Yields:
        dict: Lifespan state with crawl_runner and app_config
    """
    logger.info("[C4A-MCP | Server] Starting lifespan: initializing resources")

    # Create BrowserConfig from app config
    browser_config = app_config.browser.to_browser_config()
    logger.info(
        "[C4A-MCP | Server] Created BrowserConfig | data: {headless: %s, type: %s}",
        browser_config.headless,
        browser_config.browser_type,
    )

    # Instantiate CrawlRunner with configs
    crawl_runner = CrawlRunner(
        default_crawler_config=app_config.crawler,
        browser_config=browser_config,
    )

    logger.info("[C4A-MCP | Server] CrawlRunner initialized")

    try:
        # Yield state to make available during server runtime
        yield {"crawl_runner": crawl_runner, "app_config": app_config}
    finally:
        # Cleanup resources on shutdown
        logger.info("[C4A-MCP | Server] Shutting down: cleaning up resources")
        # NOTE: crawl4ai's AsyncWebCrawler uses context manager, so no explicit cleanup needed
        # Browser instances are created/destroyed per request in runner_tool.py
        logger.info("[C4A-MCP | Server] Shutdown complete")


# ============================================================================
# MCP Server Setup
# ============================================================================

# Initialize FastMCP server with lifespan
mcp = FastMCP(name="c4a-mcp", lifespan=lifespan)


# NOTE(REVIEWER): Tool signature matches PRD-F001 requirements.
# FastMCP will automatically generate MCP tool schema from function signature.
@mcp.tool
async def runner(
    url: str, script: str | None = None, config: dict | None = None, ctx: Context | None = None
) -> str:
    """
    Executes a web crawl session with optional interaction script and configuration.

    This tool uses crawl4ai to navigate web pages, interact with dynamic content (click buttons,
    fill forms, wait for elements), and extract content as markdown. It supports complex
    multi-step workflows via c4a-script DSL and fine-grained extraction control via config.

    Args:
        url: The starting URL (must use http:// or https:// protocol).
        script: Optional c4a-script DSL string for browser interactions. See C4A-SCRIPT REFERENCE below.
        config: Optional configuration dict for extraction behavior. See CONFIG KEYS REFERENCE below.

    Returns:
        JSON string with structure:
        {
            "markdown": str,        # Extracted content in markdown format
            "metadata": {           # Page metadata
                "url": str,
                "title": str | None,
                "timestamp": str,   # ISO format
                "status": int        # HTTP status code
            },
            "error": str | None     # Error message if crawl failed
        }

    ## C4A-SCRIPT REFERENCE

    c4a-script is a DSL for browser automation. Commands are executed line by line.
    Comments start with `#`. CSS selectors use backticks: `` `#id` `` or `` `.class` ``.

    ### Navigation

    - `GO <url>` - Navigate to URL
    - `RELOAD` - Refresh current page
    - `BACK` - Go back in history
    - `FORWARD` - Go forward in history

    ### Wait

    - `WAIT <seconds>` - Wait for time (e.g., `WAIT 3`)
    - `WAIT `<selector>` <timeout>` - Wait for element (e.g., `WAIT `#content` 10`)

    ### Mouse

    - `CLICK `<selector>`` - Click element (e.g., `CLICK `button.submit``)
    - `CLICK <x> <y>` - Click coordinates
    - `DOUBLE_CLICK `<selector>`` - Double-click element
    - `RIGHT_CLICK `<selector>`` - Right-click element
    - `SCROLL <direction> <amount>` - Scroll (e.g., `SCROLL DOWN 500`)
    - `DRAG <x1> <y1> <x2> <y2>` - Drag from point to point

    ### Keyboard

    - `TYPE "<text>"` - Type text (e.g., `TYPE "Hello World"`)
    - `TYPE $variable` - Type variable value
    - `PRESS <key>` - Press key (e.g., `PRESS Enter`, `PRESS Tab`)
    - `CLEAR `<selector>`` - Clear input field
    - `SET `<selector>` "<value>"` - Set input value directly

    ### Control Flow

    - `IF (EXISTS `<selector>`) THEN <command>` - Conditional execution
    - `REPEAT (<command>, <count>)` - Loop commands

    ### Variables

    - `SETVAR <name> = "<value>"` - Create variable
    - `$variable` - Use variable (prefix with `$`)

    ### Examples

    **Simple navigation:**
    ```
    # Simple navigation
    GO https://example.com
    WAIT `#main-content` 10
    ```

    **Form interaction:**
    ```
    # Form interaction
    CLICK `#email-input`
    TYPE "user@example.com"
    PRESS Tab
    TYPE "password123"
    CLICK `button[type="submit"]`
    ```

    **With variables:**
    ```
    # With variables
    SETVAR search_term = "crawl4ai"
    GO https://duckduckgo.com
    WAIT `input[name="q"]` 5
    TYPE $search_term
    PRESS Enter
    ```

    ## CONFIG KEYS REFERENCE

    All config keys are optional. Default timeout is 60 seconds.

    ### Content Filtering

    - `css_selector: str | None`
      - CSS selector to limit extraction scope (e.g., `".main-content"`, `"#article"`)
      - Only content matching this selector will be included in markdown.

    - `word_count_threshold: int | None`
      - Minimum word count for text blocks (default: ~200).
      - Blocks below this threshold are filtered out. Lower for shorter content.

    - `wait_for: str | None`
      - CSS selector or JS expression to wait for before extraction.
      - Examples: `"css:.loaded"`, `"js:() => window.ready === true"`

    ### Link Filtering

    - `exclude_external_links: bool` (default: `False`)
      - Exclude links to external domains from extraction.

    - `exclude_social_media_links: bool` (default: `False`)
      - Exclude social media links (Twitter, Facebook, etc.) from extraction.

    ### Caching

    - `bypass_cache: bool` (default: `True`)
      - If `True`, bypass cache and fetch fresh content.
      - If `False`, use cached content when available.

    ### Timeout

    - `timeout: int | float` (default: `60`)
      - Maximum wait time in seconds before timing out.
      - Converted internally to milliseconds for crawl4ai.

    ### Structured Extraction

    - `extraction_strategy: str | None`
      - Currently only `"jsoncss"` is supported.
      - Requires `extraction_strategy_schema` to be provided.

    - `extraction_strategy_schema: dict | None`
      - Schema dict for JSON CSS extraction (required if `extraction_strategy="jsoncss"`).
      - Example: `{"title": "h1", "content": ".article-body"}`

    ### Config Examples

    **Extract only main content:**
    ```json
    {"css_selector": ".main-article", "word_count_threshold": 50}
    ```

    **Wait for dynamic content, exclude external links:**
    ```json
    {"wait_for": "css:.dynamic-content", "exclude_external_links": True}
    ```

    **Structured extraction:**
    ```json
    {
      "extraction_strategy": "jsoncss",
      "extraction_strategy_schema": {
        "title": "h1.title",
        "author": ".author-name",
        "body": ".article-content"
      }
    }
    ```

    **Full example with script and config:**
    ```json
    {
      "timeout": 90,
      "css_selector": "#main-content",
      "word_count_threshold": 100,
      "exclude_external_links": True,
      "bypass_cache": True
    }
    ```

    ## BEST PRACTICES

    1. Always use `WAIT` before `CLICK` to ensure elements are loaded:
       ```
       WAIT `#button` 5
       CLICK `#button`
       ```

    2. Use `css_selector` in config to focus on relevant content and reduce noise.

    3. Increase timeout for slow-loading pages or complex interactions.

    4. Use variables for reusable values (URLs, credentials, etc.).

    5. Handle optional elements with `IF (EXISTS ...)` to avoid errors.

    6. For forms, use `TYPE` for text input and `PRESS` for special keys (Tab, Enter).

    ## ERROR HANDLING

    The tool returns errors in the `"error"` field of the JSON response:

    - **Network errors**: `"Network error: <details>"`
    - **Timeouts**: `"Timeout after X seconds: <details>"`
    - **Script errors**: `"Script execution failed: <details>"`
    - **Invalid input**: Validation errors are returned before execution

    Always check the `"error"` field in the response. If error is not `None`, markdown will be empty.
    """
    try:
        # Access crawl_runner from lifespan state via Context
        if ctx is None:
            raise ValueError("Context is required for runner tool")
        crawl_runner = ctx.get_state("crawl_runner")
        if crawl_runner is None:
            raise ValueError(
                "crawl_runner not found in context state. "
                "Ensure the server lifespan properly initializes crawl_runner."
            )

        # Validate inputs using RunnerInput model
        # Pydantic will handle validation and type coercion
        runner_input = RunnerInput(url=url, script=script, config=config)

        # Invoke CrawlRunner
        runner_output: RunnerOutput = await crawl_runner.run(runner_input)

        # Return result as JSON string
        return runner_output.model_dump_json()
    except Exception as e:
        # Log full error details with traceback for debugging
        logger.error(
            "[C4A-MCP | Server] Error during runner tool execution | "
            "data: {error_type: %s, error: %s}",
            type(e).__name__,
            str(e),
        )
        logger.debug(
            "[C4A-MCP | Server] Full traceback | data: {traceback: %s}",
            traceback.format_exc(),
        )
        # Return sanitized error message to client
        error_output = RunnerOutput(
            markdown="",
            error=f"Tool execution failed: {type(e).__name__} - {e}",
        )
        return error_output.model_dump_json()


# Register preset tools
# NOTE: FastMCP automatically generates MCP tool schemas from function signatures
mcp.tool(crawl_deep)
mcp.tool(crawl_deep_smart)
mcp.tool(scrape_page)


def serve():
    """Programmatic entry point for running the server."""
    mcp.run()
