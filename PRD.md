# Product Requirements Document (PRD)

## Problem Statement
AI Agents currently lack a standardized, high-level interface to interact with dynamic web pages effectively. While simple fetching tools exist, they cannot handle complex user interactions (scrolling, clicking, waiting for AJAX) or structured data extraction without writing complex, brittle JavaScript. `crawl4ai` offers these capabilities via its `c4a-script` DSL, but this functionality is not exposed via a Model Context Protocol (MCP) server, limiting its accessibility to LLMs like Claude or Gemini.

## User Personas
1.  **AI Agents**: Need to independently navigate websites, click buttons (e.g., "Load More"), and extract specific data to answer user queries.
2.  **Agent Developers**: Need a reliable, configuration-driven tool to give their agents web browsing capabilities without building custom scraping microservices.

## User Stories
- [ ] **Story 1**: As an AI agent, I want to submit a URL and a `c4a-script` (DSL) to a tool so that I can perform multi-step interactions (e.g., search, wait, click) on a page before extracting content.
- [ ] **Story 2**: As an AI agent, I want to provide a configuration object (specifying CSS selectors or extraction strategies) alongside my script so that the returned data is structured and irrelevant noise is filtered out.
- [ ] **Story 3**: As an AI agent, I want to receive a markdown representation of the final page state so that I can easily read and process the information.
- [ ] **Story 4**: As an AI agent, I want to crawl a website deeply (BFS) to gather context from multiple linked pages without manually managing the crawl queue.
- [ ] **Story 5**: As an AI agent, I want to crawl a website prioritizing pages that match specific keywords to find relevant information faster.
- [ ] **Story 6**: As an AI agent, I want to use simplified, high-level tools for common tasks (like single-page scraping with a specific strategy) without constructing complex configuration objects.

## Functional Requirements

### F001: The `runner` Tool (P0)
A single MCP tool named `runner` that executes a crawl session.

*   **Input Parameters:**
    *   `url` (string, required): The starting URL for the session.
    *   `script` (string, optional): A `c4a-script` DSL string defining the interaction steps (e.g., `WAIT`, `CLICK`, `SCROLL`).
    *   `config` (object, optional): A JSON-serializable object (standard Python types only) mapping to `crawl4ai`'s `CrawlerRunConfig`. Key fields include:
        *   `css_selector`: To scope the extraction.
        *   `word_count_threshold`: To filter small text blocks.
        *   `extraction_strategy`: To specify structured extraction (e.g., JsonCssExtractionStrategy).
        *   `deep_crawl_strategy_params`: Parameters for deep crawling strategies (created from parameters at runtime).
        *   `extraction_strategy_params`: Parameters for extraction strategies (created from parameters at runtime).

*   **Output:**
    *   Returns a JSON object containing:
        *   `markdown`: The extracted text in markdown format.
        *   `metadata`: Basic metadata about the crawl (title, url, timestamp).
        *   `error`: Error message if the crawl failed.

### F002: `c4a-script` Integration (P0)
The server must correctly interpret the provided `script` string and pass it to the `crawl4ai` engine. It should handle standard commands like `GO`, `WAIT`, `CLICK`, `TYPE`, `SCROLL`.

### F003: Configuration Handling (P1)
The `runner` must accept and apply standard `crawl4ai` configuration options.
*   Support for `css_selector` to limit scope.
*   Support for excluding external links or tags.

### F004: Preset Tools (P1)
The server must expose high-level preset tools for common crawling patterns, abstracting complex configurations.

*   **`crawl_deep`**: Performs Breadth-First Search (BFS) crawling.
    *   Inputs: `url`, `max_depth` (default 2), `max_pages` (default 50), `include_external` (bool).
    *   Supports `extraction_strategy` ("regex", "css", "llm") and `extraction_strategy_config`.
*   **`crawl_deep_smart`**: Performs Best-First Search crawling based on keywords.
    *   Inputs: `url`, `keywords` (list[str]), `max_depth`, `max_pages`.
*   **`scrape_page`**: Single-page scraping with explicit strategy support.
    *   Inputs: `url`, `extraction_strategy`, `extraction_strategy_config`.

## Non-Functional Requirements
-   **Performance**: The tool should timeout gracefully if a script hangs (default 60s).
-   **Reliability**: Must handle network errors or invalid selectors without crashing the MCP server.
-   **Security**: The server executes web interactions; it should be run in a sandboxed environment or with explicit user permission for each domain if possible (handled by MCP client permissions).

## MVP Scope (Release 1.0)
-   Core `runner` tool for low-level control.
-   Preset tools (`crawl_deep`, `crawl_deep_smart`, `scrape_page`) for high-level workflows.
-   Support for text/markdown extraction.
-   Support for basic `c4a-script` execution.
-   No authentication/persistence in MVP.
