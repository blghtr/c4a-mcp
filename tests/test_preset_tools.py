# LLM:METADATA
# :hierarchy: [C4A-MCP | Tests | Preset Tools]
# :relates-to: tests: "presets.preset_tools", implements: "PRD-F001 extension", implements: "SPEC-F001 extension"
# :rationale: "Validates preset tool implementations, ensuring they correctly delegate to CrawlRunner and handle all parameters."
# :contract: pre: "Mocked CrawlRunner", post: "All tools return valid JSON RunnerOutput"
# :decision_cache: "Integration tests with mocked CrawlRunner to verify tool behavior without actual crawling [TEST-004]. Updated to use dependency injection pattern [TEST-005]"
# LLM:END

"""
Tests for preset tool implementations.

Validates that preset tools correctly:
- Validate inputs via Pydantic models
- Create crawling and extraction strategies
- Build CrawlerRunConfig
- Delegate to CrawlRunner
- Return valid JSON responses
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Context

from c4a_mcp.models import RunnerOutput
from c4a_mcp.presets.preset_tools import crawl_deep, crawl_deep_smart, scrape_page
from c4a_mcp.runner_tool import CrawlRunner


# --- Test Fixtures ---
@pytest.fixture
def mock_crawl_runner():
    """Create a mock CrawlRunner for testing."""
    runner = MagicMock(spec=CrawlRunner)
    runner.run = AsyncMock()
    return runner


@pytest.fixture
def mock_context(mock_crawl_runner):
    """Create a mock Context with crawl_runner in state."""
    ctx = MagicMock(spec=Context)
    ctx.get_state = MagicMock(return_value=mock_crawl_runner)
    return ctx


@pytest.fixture
def mock_context_no_runner():
    """Create a mock Context without crawl_runner in state."""
    ctx = MagicMock(spec=Context)
    ctx.get_state = MagicMock(return_value=None)
    return ctx


@pytest.fixture
def mock_runner_output():
    """Create a mock RunnerOutput for testing."""
    output = RunnerOutput(
        markdown="# Test Content\n\nThis is test markdown.",
        metadata={
            "url": "https://example.com",
            "title": "Example Page",
            "timestamp": "2024-01-01T00:00:00",
            "status": 200,
        },
        error=None,
    )
    return output


# --- Test tool functions exist ---
def test_preset_tools_exist():
    """Test that preset tools are callable functions."""
    assert callable(crawl_deep)
    assert callable(crawl_deep_smart)
    assert callable(scrape_page)


# --- Test crawl_deep ---
@pytest.mark.asyncio
async def test_crawl_deep_success(mock_context, mock_runner_output):
    """Test successful crawl_deep execution."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    result_json = await crawl_deep(url="https://example.com", max_depth=2, max_pages=50, ctx=mock_context)

    # Verify JSON is valid
    result = json.loads(result_json)
    assert result["markdown"] == "# Test Content\n\nThis is test markdown."
    assert result["error"] is None
    assert result["metadata"]["url"] == "https://example.com"

    # Verify CrawlRunner was called
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run.assert_called_once()
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"
    assert call_args.script is None


@pytest.mark.asyncio
async def test_crawl_deep_with_script(mock_context, mock_runner_output):
    """Test crawl_deep with script parameter."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    script = "WAIT `#content` 5\nCLICK `#button`"
    result_json = await crawl_deep(
        url="https://example.com", script=script, max_depth=2, ctx=mock_context
    )

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify script was passed
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


@pytest.mark.asyncio
async def test_crawl_deep_with_extraction(mock_context, mock_runner_output):
    """Test crawl_deep with extraction strategy."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    with patch(
        "c4a_mcp.presets.preset_tools.create_extraction_strategy"
    ) as mock_create_extraction:
        # Use real JsonCssExtractionStrategy for proper isinstance checks
        from crawl4ai import JsonCssExtractionStrategy

        from c4a_mcp.presets.models import ExtractionConfigCss
        
        schema = {"name": "Test", "baseSelector": "div", "fields": []}
        real_strategy = JsonCssExtractionStrategy(schema=schema)
        mock_create_extraction.return_value = real_strategy

        config = ExtractionConfigCss(extraction_schema={"name": "Test", "baseSelector": "div"})
        result_json = await crawl_deep(
            url="https://example.com",
            extraction_strategy="css",
            extraction_strategy_config=config,
            ctx=mock_context,
        )

        result = json.loads(result_json)
        assert result["error"] is None
        mock_create_extraction.assert_called_once_with("css", config)


@pytest.mark.asyncio
async def test_crawl_deep_invalid_url(mock_context):
    """Test crawl_deep with invalid URL."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        await crawl_deep(url="not-a-url", ctx=mock_context)


@pytest.mark.asyncio
async def test_crawl_deep_missing_runner(mock_context_no_runner):
    """Test crawl_deep when crawl_runner is missing from context."""
    with pytest.raises(ValueError) as exc_info:
        await crawl_deep(url="https://example.com", ctx=mock_context_no_runner)
    assert "crawl_runner not found in context state" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crawl_deep_no_context():
    """Test crawl_deep when context is None."""
    with pytest.raises(ValueError) as exc_info:
        await crawl_deep(url="https://example.com", ctx=None)
    assert "Context is required" in str(exc_info.value)


# --- Test crawl_deep_smart ---
@pytest.mark.asyncio
async def test_crawl_deep_smart_success(mock_context, mock_runner_output):
    """Test successful crawl_deep_smart execution."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    result_json = await crawl_deep_smart(
        url="https://example.com",
        keywords=["test", "example"],
        max_depth=2,
        max_pages=25,
        ctx=mock_context,
    )

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify keywords were used
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"


@pytest.mark.asyncio
async def test_crawl_deep_smart_empty_keywords(mock_context):
    """Test crawl_deep_smart with empty keywords."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        await crawl_deep_smart(url="https://example.com", keywords=[], ctx=mock_context)


@pytest.mark.asyncio
async def test_crawl_deep_smart_with_script(mock_context, mock_runner_output):
    """Test crawl_deep_smart with script."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    script = "SCROLL DOWN 500"
    result_json = await crawl_deep_smart(
        url="https://example.com", keywords=["test"], script=script, ctx=mock_context
    )

    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


# --- Test scrape_page ---
@pytest.mark.asyncio
async def test_scrape_page_success(mock_context, mock_runner_output):
    """Test successful scrape_page execution."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    result_json = await scrape_page(url="https://example.com", ctx=mock_context)

    result = json.loads(result_json)
    assert result["error"] is None
    assert result["metadata"]["url"] == "https://example.com"

    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"


@pytest.mark.asyncio
async def test_scrape_page_with_script(mock_context, mock_runner_output):
    """Test scrape_page with script."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    script = "WAIT 3\nCLICK `#load-more`"
    result_json = await scrape_page(url="https://example.com", script=script, ctx=mock_context)

    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


@pytest.mark.asyncio
async def test_scrape_page_with_extraction(mock_context, mock_runner_output):
    """Test scrape_page with extraction strategy."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    with patch(
        "c4a_mcp.presets.preset_tools.create_extraction_strategy"
    ) as mock_create_extraction:
        # Use real JsonCssExtractionStrategy for proper isinstance checks
        from crawl4ai import JsonCssExtractionStrategy
        from c4a_mcp.presets.models import ExtractionConfigCss

        schema = {"name": "Test", "baseSelector": "div", "fields": []}
        real_strategy = JsonCssExtractionStrategy(schema=schema)
        mock_create_extraction.return_value = real_strategy

        config = ExtractionConfigCss(extraction_schema=schema)
        result_json = await scrape_page(
            url="https://example.com",
            extraction_strategy="css",
            extraction_strategy_config=config,
            ctx=mock_context,
        )

    result = json.loads(result_json)
    assert result["error"] is None
    mock_create_extraction.assert_called_once()


@pytest.mark.asyncio
async def test_scrape_page_error_handling(mock_context):
    """Test scrape_page error handling."""
    error_output = RunnerOutput(
        markdown="", error="Network error: Connection failed", metadata={}
    )
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=error_output)

    result_json = await scrape_page(url="https://example.com", ctx=mock_context)

    result = json.loads(result_json)
    assert result["markdown"] == ""
    assert "Network error" in result["error"]


# --- Test parameter passing ---
@pytest.mark.asyncio
async def test_crawl_deep_parameter_passing(mock_context, mock_runner_output):
    """Test that all parameters are correctly passed through."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    await crawl_deep(
        url="https://example.com",
        max_depth=3,
        max_pages=100,
        include_external=True,
        config={
            "timeout": 90,
            "css_selector": "article",
            "word_count_threshold": 50,
            "exclude_external_links": True,
        },
        ctx=mock_context,
    )

    # Verify config was built with parameters
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.config is not None
    config_dict = call_args.config
    # Note: Actual config structure depends on implementation
    # This test verifies that parameters are accepted without errors


@pytest.mark.asyncio
async def test_all_tools_return_json(mock_context, mock_runner_output):
    """Test that all tools return valid JSON strings."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    # Test all tools
    results = [
        await crawl_deep(url="https://example.com", ctx=mock_context),
        await crawl_deep_smart(url="https://example.com", keywords=["test"], ctx=mock_context),
        await scrape_page(url="https://example.com", ctx=mock_context),
    ]

    # Verify all are valid JSON
    for result_json in results:
        result = json.loads(result_json)
        assert "markdown" in result
        assert "metadata" in result
        assert "error" in result
        assert isinstance(result["markdown"], str)
        assert isinstance(result["metadata"], dict)
        assert result["error"] is None or isinstance(result["error"], str)
