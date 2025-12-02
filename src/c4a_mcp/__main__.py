# LLM:METADATA
# :hierarchy: [C4A-MCP | CLI]
# :relates-to: calls: "server.serve"
# :rationale: "Allows running the server module directly via `python -m c4a_mcp`."
# :contract: pre: "Environment must be set up", post: "Server starts"
# :decision_cache: "Standard Python CLI pattern [ARCH-005]"
# LLM:END

from .server import serve

if __name__ == "__main__":
    serve()
