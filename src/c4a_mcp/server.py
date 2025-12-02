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
async def runner(url: str, script: str = None, config: dict = None) -> str:
    """
    Executes a web crawl session with optional interaction script and configuration.

    Args:
        url: The starting URL.
        script: Optional c4a-script DSL for interactions (WAIT, CLICK, etc.).
        config: Optional configuration object (css_selector, etc.).

    Returns:
        JSON string representation of the RunnerOutput.
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
