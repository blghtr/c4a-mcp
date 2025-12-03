# LLM:METADATA
# :hierarchy: [C4A-MCP | Presets]
# :relates-to: uses: "runner_tool.CrawlRunner", uses: "config_models.CrawlerConfigYAML", motivated_by: "current_results.md", implements: "PRD-F001 extension"
# :rationale: "Provides preset tools for common crawling patterns, reducing cognitive load on AI models by offering sensible defaults with override flexibility."
# :contract: invariant: "All preset tools must return RunnerOutput format consistent with runner tool"
# :decision_cache: "Created separate presets module to isolate preset logic from core runner, enabling independent testing and maintenance [ARCH-006]"
# LLM:END

"""
Preset tools for common crawling patterns.

This module provides high-level tools for common web crawling scenarios:
- Deep crawling (BFS and Best-First)
- Single-page scraping

All tools support extraction strategies (regex, CSS, LLM) and extensive
parameter customization while maintaining sensible defaults.
"""

from .preset_tools import create_preset_tools

__all__ = [
    "create_preset_tools",
]

