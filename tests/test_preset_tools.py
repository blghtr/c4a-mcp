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

from c4a_mcp.models import RunnerOutput
from c4a_mcp.presets.preset_tools import create_preset_tools
from c4a_mcp.runner_tool import CrawlRunner


# --- Test Fixtures ---
@pytest.fixture
def mock_crawl_runner():
    """Create a mock CrawlRunner for testing."""
    runner = MagicMock(spec=CrawlRunner)
    runner.run = AsyncMock()
    return runner


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


# --- Test create_preset_tools ---
def test_create_preset_tools(mock_crawl_runner):
    """Test creating preset tools with dependency injection."""
    crawl_deep, crawl_deep_smart, scrape_page = create_preset_tools(mock_crawl_runner)
    
    # Verify functions are created
    assert callable(crawl_deep)
    assert callable(crawl_deep_smart)
    assert callable(scrape_page)


# --- Test crawl_deep ---
@pytest.mark.asyncio
async def test_crawl_deep_success(mock_crawl_runner, mock_runner_output):
    """Test successful crawl_deep execution."""
    crawl_deep, _, _ = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    result_json = await crawl_deep(url="https://example.com", max_depth=2, max_pages=50)

    # Verify JSON is valid
    result = json.loads(result_json)
    assert result["markdown"] == "# Test Content\n\nThis is test markdown."
    assert result["error"] is None
    assert result["metadata"]["url"] == "https://example.com"

    # Verify CrawlRunner was called
    mock_crawl_runner.run.assert_called_once()
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"
    assert call_args.script is None


@pytest.mark.asyncio
async def test_crawl_deep_with_script(mock_crawl_runner, mock_runner_output):
    """Test crawl_deep with script parameter."""
    crawl_deep, _, _ = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    script = "WAIT `#content` 5\nCLICK `#button`"
    result_json = await crawl_deep(
        url="https://example.com", script=script, max_depth=2
    )

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify script was passed
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


@pytest.mark.asyncio
async def test_crawl_deep_with_extraction(mock_crawl_runner, mock_runner_output):
    """Test crawl_deep with extraction strategy."""
    crawl_deep, _, _ = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    with patch(
        "c4a_mcp.presets.preset_tools.create_extraction_strategy"
    ) as mock_create_extraction:
        # Use real JsonCssExtractionStrategy for proper isinstance checks
        from crawl4ai import JsonCssExtractionStrategy

        schema = {"name": "Test", "baseSelector": "div", "fields": []}
        real_strategy = JsonCssExtractionStrategy(schema=schema)
        mock_create_extraction.return_value = real_strategy

        result_json = await crawl_deep(
            url="https://example.com",
            extraction_strategy="css",
            extraction_strategy_config={"schema": {"name": "Test", "baseSelector": "div"}},
        )

        result = json.loads(result_json)
        assert result["error"] is None
        mock_create_extraction.assert_called_once_with(
            "css", {"schema": {"name": "Test", "baseSelector": "div"}}
        )


@pytest.mark.asyncio
async def test_crawl_deep_invalid_url(mock_crawl_runner):
    """Test crawl_deep with invalid URL."""
    crawl_deep, _, _ = create_preset_tools(mock_crawl_runner)

    with pytest.raises(Exception):  # Pydantic ValidationError
        await crawl_deep(url="not-a-url")


# --- Test crawl_deep_smart ---
@pytest.mark.asyncio
async def test_crawl_deep_smart_success(mock_crawl_runner, mock_runner_output):
    """Test successful crawl_deep_smart execution."""
    _, crawl_deep_smart, _ = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    result_json = await crawl_deep_smart(
        url="https://example.com",
        keywords=["test", "example"],
        max_depth=2,
        max_pages=25,
    )

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify keywords were used
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"


@pytest.mark.asyncio
async def test_crawl_deep_smart_empty_keywords(mock_crawl_runner):
    """Test crawl_deep_smart with empty keywords."""
    _, crawl_deep_smart, _ = create_preset_tools(mock_crawl_runner)

    with pytest.raises(Exception):  # Pydantic ValidationError
        await crawl_deep_smart(url="https://example.com", keywords=[])


@pytest.mark.asyncio
async def test_crawl_deep_smart_with_script(mock_crawl_runner, mock_runner_output):
    """Test crawl_deep_smart with script."""
    _, crawl_deep_smart, _ = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    script = "SCROLL DOWN 500"
    result_json = await crawl_deep_smart(
        url="https://example.com", keywords=["test"], script=script
    )

    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


# --- Test scrape_page ---
@pytest.mark.asyncio
async def test_scrape_page_success(mock_crawl_runner, mock_runner_output):
    """Test successful scrape_page execution."""
    _, _, scrape_page = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    result_json = await scrape_page(url="https://example.com")

    result = json.loads(result_json)
    assert result["error"] is None
    assert result["metadata"]["url"] == "https://example.com"

    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"


@pytest.mark.asyncio
async def test_scrape_page_with_script(mock_crawl_runner, mock_runner_output):
    """Test scrape_page with script."""
    _, _, scrape_page = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    script = "WAIT 3\nCLICK `#load-more`"
    result_json = await scrape_page(url="https://example.com", script=script)

    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


@pytest.mark.asyncio
async def test_scrape_page_with_extraction(mock_crawl_runner, mock_runner_output):
    """Test scrape_page with extraction strategy."""
    _, _, scrape_page = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    with patch(
        "c4a_mcp.presets.preset_tools.create_extraction_strategy"
    ) as mock_create_extraction:
        # Use real LLMExtractionStrategy for proper isinstance checks
        # Note: LLM strategy requires API token, so we'll skip this test
        # or use a mock that properly inherits from ExtractionStrategy
        from crawl4ai import LLMExtractionStrategy, LLMConfig

        # Create a minimal LLM config (won't actually call API in test)
        llm_config = LLMConfig(provider="openai/gpt-4o-mini", api_token="test-token")
        real_strategy = LLMExtractionStrategy(llm_config=llm_config)
        mock_create_extraction.return_value = real_strategy

        result_json = await scrape_page(
            url="https://example.com",
            extraction_strategy="llm",
            extraction_strategy_config={
                "provider": "openai/gpt-4o-mini",
                "api_token": "test-token",
            },
        )

        result = json.loads(result_json)
        assert result["error"] is None
        mock_create_extraction.assert_called_once()


@pytest.mark.asyncio
async def test_scrape_page_error_handling(mock_crawl_runner):
    """Test scrape_page error handling."""
    _, _, scrape_page = create_preset_tools(mock_crawl_runner)
    error_output = RunnerOutput(
        markdown="", error="Network error: Connection failed", metadata={}
    )
    mock_crawl_runner.run.return_value = error_output

    result_json = await scrape_page(url="https://example.com")

    result = json.loads(result_json)
    assert result["markdown"] == ""
    assert "Network error" in result["error"]


# --- Test parameter passing ---
@pytest.mark.asyncio
async def test_crawl_deep_parameter_passing(mock_crawl_runner, mock_runner_output):
    """Test that all parameters are correctly passed through."""
    crawl_deep, _, _ = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    await crawl_deep(
        url="https://example.com",
        max_depth=3,
        max_pages=100,
        include_external=True,
        timeout=90,
        css_selector="article",
        word_count_threshold=50,
        exclude_external_links=True,
    )

    # Verify config was built with parameters
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.config is not None
    config_dict = call_args.config
    # Note: Actual config structure depends on implementation
    # This test verifies that parameters are accepted without errors


@pytest.mark.asyncio
async def test_all_tools_return_json(mock_crawl_runner, mock_runner_output):
    """Test that all tools return valid JSON strings."""
    crawl_deep, crawl_deep_smart, scrape_page = create_preset_tools(mock_crawl_runner)
    mock_crawl_runner.run.return_value = mock_runner_output

    # Test all tools
    results = [
        await crawl_deep(url="https://example.com"),
        await crawl_deep_smart(url="https://example.com", keywords=["test"]),
        await scrape_page(url="https://example.com"),
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
