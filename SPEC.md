# Technical Specification: Crawl4AI MCP Server

## System Invariants
- **INV-01**: The server MUST expose the core `runner` tool AND three preset tools: `crawl_deep`, `crawl_deep_smart`, and `scrape_page`.
- **INV-02**: All tool outputs MUST be valid JSON matching the `CallToolResult` schema from MCP.
- **INV-03**: The `runner` and preset tools MUST NOT persist state between independent tool calls (stateless).

## Feature Specifications

### SPEC-F001: The `runner` Tool
**Rationale**: To provide a unified, low-level interface for AI agents to execute complex web interactions.
**References**: PRD-F001

**Contract**:
- **GIVEN** a valid `url` AND an optional `script` AND optional `config`
- **WHEN** the `runner` tool is invoked
- **THEN** it SHALL initialize an `AsyncWebCrawler` context
- **AND** execute the crawl session
- **AND** return a JSON object containing `markdown` (string), `metadata` (object), and `error` (string or null).

**Input Schema**:
```python
class RunnerInput(BaseModel):
    url: str = Field(..., description="The starting URL for the crawl session.")
    script: Optional[str] = Field(None, description="A c4a-script DSL string defining interaction steps (GO, WAIT, CLICK, etc).")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration object mapping to crawl4ai's CrawlerRunConfig (css_selector, wait_for, etc).")
```

**Acceptance Predicates**:
- [ ] `runner` accepts valid URL and returns content in `markdown` field.
- [ ] `runner` accepts `script` and successfully executes interactions (e.g., clicking a button before extraction).
- [ ] `runner` returns error message in `error` field upon failure (e.g., invalid URL), ensuring the server does not crash.

### SPEC-F002: `c4a-script` Integration
**Rationale**: To enable complex user interactions via a simple DSL.
**References**: PRD-F002

**Contract**:
- **GIVEN** a non-empty `script` string
- **WHEN** processing the request
- **THEN** the `script` SHALL be passed to the `crawl4ai` engine as the interaction logic.
- **AND** the execution environment SHALL NOT allow arbitrary system command execution (limited to browser scope).

**Implementation Details**:
- Maps `script` input to the appropriate parameter in `AsyncWebCrawler.arun()`.

### SPEC-F003: Configuration Handling
**Rationale**: To allow fine-grained control over extraction and crawling behavior.
**References**: PRD-F003

**Contract**:
- **GIVEN** a `config` object
- **WHEN** initializing the crawl run
- **THEN** specific properties (`css_selector`, `word_count_threshold`, `wait_for`, `exclude_external_links`, `exclude_social_media_links`) SHALL be mapped to `CrawlerRunConfig`.
- **AND** reasonable defaults SHALL be applied for missing fields (e.g., `bypass_cache: true`).

**Acceptance Predicates**:
- [ ] Providing `css_selector` restricts output markdown to that element.
- [ ] Providing `word_count_threshold` filters extracted text.

### SPEC-F004: Preset Tools
**Rationale**: To provide simplified, purpose-built tools for common crawling patterns (Deep BFS, Keyword-based, Single-page scraping).
**References**: PRD-F004

**Contract**:
- **GIVEN** a call to `crawl_deep`, `crawl_deep_smart`, or `scrape_page`
- **WHEN** the tool is executed
- **THEN** it SHALL map inputs to a specialized `DeepCrawlStrategy` or `ExtractionStrategy`
- **AND** invoke the underlying `CrawlRunner` with a generated `CrawlerRunConfig`.

**Invariants**:
- Preset tools must reuse the same `CrawlRunner` logic as the main tool to ensure consistent error handling and browser configuration.
- `crawl_deep` must default to BFS strategy.
- `crawl_deep_smart` must default to Best-First strategy using provided keywords.

## Architecture & Dependencies
- **Language**: Python 3.10+
- **Core Library**: `crawl4ai`
- **Protocol**: `mcp` (Model Context Protocol)
- **Server Framework**: `mcp` (using `FastMCP` class).
- **Patterns**: Dependency Injection (via Factory pattern) used for preset tools to allow stateless tool creation and easier testing.

## Error Handling
- **Network Errors**: Return `{ "error": "Network error: <details>", "markdown": "" }`.
- **Timeout**: Return `{ "error": "Timeout after X seconds", "markdown": "" }`.
- **Invalid Script**: Return `{ "error": "Script execution failed: <details>", "markdown": "" }`.
