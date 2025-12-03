#!/usr/bin/env python3
"""Use MCP tools directly via in-memory client."""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import Client
from c4a_mcp.server import mcp


async def use_tools():
    """Use tools via in-memory client."""
    url = "https://dev.plastinka.com/"
    
    print(f"Using scrape_page tool on {url}...\n")
    
    # Use in-memory transport - pass server directly
    async with Client(mcp) as client:
        # Call scrape_page tool - pass url as direct argument
        result = await client.call_tool("scrape_page", {"url": url, "kwargs": {}})
        
        # Extract text content - result is a string (JSON)
        if isinstance(result, str):
            print(f"Result length: {len(result)} chars")
            print(f"\nFirst 1000 chars:\n{result[:1000]}")
        elif hasattr(result, 'content') and result.content:
            text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            print(f"Result length: {len(text)} chars")
            print(f"\nFirst 1000 chars:\n{text[:1000]}")
        else:
            print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(use_tools())

