# LLM:METADATA
# :hierarchy: [C4A-MCP | Server]
# :relates-to: uses: "mcp.FastMCP", uses: "runner_tool"
# :rationale: "Entry point for the MCP server, handling tool registration and request routing."
# :contract: invariant: "Server must remain responsive and handle exceptions gracefully"
# :decision_cache: "Using FastMCP for simplified decorator-based tool definition [ARCH-004]"
# LLM:END

import logging
import traceback

from mcp.server.fastmcp import FastMCP

from .models import RunnerInput, RunnerOutput
from .runner_tool import CrawlRunner

# Configure logger with hierarchy path format
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("c4a-mcp")

# Instantiate CrawlRunner once, as it's stateless and can be reused
crawl_runner = CrawlRunner()


# NOTE(REVIEWER): Tool signature matches PRD-F001 requirements.
# FastMCP will automatically generate MCP tool schema from function signature.
@mcp.tool()
async def runner(url: str, script: str | None = None, config: dict | None = None) -> str:
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


def serve():
    """Programmatic entry point for running the server."""
    mcp.run()
