# GEMINI.md - Context & Instructions for c4a-mcp

## Project Overview

**c4a-mcp** is a Model Context Protocol (MCP) server that provides web browsing and data extraction capabilities to AI agents using `crawl4ai`. It exposes a `runner` tool that allows agents to navigate URLs, interact with dynamic content using a custom DSL (`c4a-script`), and extract structured markdown data.

## Architecture

The project follows a layered architecture designed for modularity and testability:

1.  **Server Layer (`src/c4a_mcp/server.py`)**: 
    - Uses `mcp.FastMCP` to handle the protocol details.
    - Loads configuration (defaults + CLI overrides).
    - Registers the `runner` tool.
    - Handles top-level error trapping and logging.

2.  **Logic Layer (`src/c4a_mcp/runner_tool.py`)**:
    - Contains `CrawlRunner`, which encapsulates the business logic.
    - Maps `RunnerInput` (tool arguments) to `crawl4ai`'s `CrawlerRunConfig`.
    - Manages the lifecycle of `AsyncWebCrawler`.
    - Handles specific error mapping (Timeout, Network, Script errors).

3.  **Data Layer (`src/c4a_mcp/models.py`, `config_models.py`)**:
    - Pydantic models define strict schemas for inputs, outputs, and configuration.
    - Implements a 3-layer configuration merge strategy: `Defaults` -> `File Overrides` -> `Tool Runtime Overrides`.

## Development Standards

### 1. Semantic Metadata (CRITICAL)
All major functions and classes MUST be preceded by an `LLM:METADATA` block. This provides context for both human developers and AI agents.

**Format:**
```python
# LLM:METADATA
# :hierarchy: [Domain | Component]
# :relates-to: uses: "Module.Class", implements: "SPEC-ID"
# :rationale: "Why does this code exist?"
# :contract: pre: "Conditions before execution", post: "Guarantees after execution"
# :decision_cache: "Why a specific approach was chosen [ID]"
# LLM:END
```

### 2. Structured Logging
Logging MUST follow a strict format to enable easy parsing and filtering.

**Format:**
```python
logger.info(
    "[Hierarchy | Path] Human readable message | data: {key: %s, key2: %s}",
    value1,
    value2
)
```
*   **Hierarchy:** Matches the `:hierarchy:` field in the metadata.
*   **Data:** JSON-like structure for machine-readable context.

### 3. Tooling & Quality
*   **Package Manager:** `uv` (must be used for all dependency operations).
*   **Linting/Formatting:** `ruff` and `black` (enforced via `pre-commit`).
*   **Type Hints:** Strict Python type hinting is required.
*   **Testing:** `pytest` with `pytest-asyncio`.

## Build & Run

### Setup
```bash
# Install dependencies
uv pip install --system -e ".[dev]"

# Install pre-commit hooks (Run once)
uv run pre-commit install
```

### Operations
```bash
# Run Server (Default)
uv run c4a-mcp

# Run Server (With Config Override)
uv run c4a-mcp --config=./my-config.yaml

# Run Tests
uv run pytest

# Run Tests (Verbose)
uv run pytest -v

# Run Linter/Formatter
uv run pre-commit run --all-files
```

### Docker
```bash
# Build
docker build -t c4a-mcp:local .

# Run
docker run c4a-mcp:local
```

## Key Files

*   `src/c4a_mcp/server.py`: Entry point. Initializes `FastMCP` and loads config.
*   `src/c4a_mcp/runner_tool.py`: Core logic wrapper for `crawl4ai`.
*   `src/c4a_mcp/config/defaults.yaml`: Default configuration values.
*   `src/c4a_mcp/models.py`: Input/Output models for the MCP tools.
*   `SPEC.md`: Technical specification and invariants.
*   `PRD.md`: Product requirements.

## Tool Reference: `runner`

The primary tool exposed by this server.

*   **Input (`RunnerInput`)**:
    *   `url` (str): Target URL.
    *   `script` (str, optional): `c4a-script` DSL for interactions (GO, CLICK, TYPE, WAIT).
    *   `config` (dict, optional): Overrides for extraction/crawling behavior (CSS selectors, timeout, etc.).
*   **Output (`RunnerOutput`)**:
    *   `markdown` (str): Extracted text content.
    *   `metadata` (dict): URL, title, status, timestamp.
    *   `error` (str | None): Error description if failed.
