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
- Pass strategy parameters (not objects) to CrawlRunner via config
- Build config dict with strategy_params
- Delegate to CrawlRunner
- Return valid JSON responses
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context

from c4a_mcp.adaptive_runner import AdaptiveCrawlRunner
from c4a_mcp.models import RunnerOutput
from c4a_mcp.presets.models import (
    AdaptiveEmbeddingInput,
    AdaptiveStatisticalInput,
    CrawlDeepSmartInput,
    DeepCrawlPresetInput,
    ExtractionConfigCss,
    ScrapePagePresetInput,
)
from c4a_mcp.presets.preset_tools import (
    adaptive_crawl_embedding,
    adaptive_crawl_statistical,
    crawl_deep,
    crawl_deep_smart,
    scrape_page,
)
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


@pytest.fixture
def mock_adaptive_runner():
    """Create a mock AdaptiveCrawlRunner for testing."""
    runner = MagicMock(spec=AdaptiveCrawlRunner)
    runner.run = AsyncMock()
    return runner


@pytest.fixture
def mock_adaptive_context(mock_adaptive_runner):
    """Create a mock Context with adaptive_crawl_runner in state."""
    ctx = MagicMock(spec=Context)
    ctx.get_state = MagicMock(return_value=mock_adaptive_runner)
    return ctx


@pytest.fixture
def mock_adaptive_runner_output():
    """Create a mock RunnerOutput with confidence/metrics for adaptive crawling."""
    output = RunnerOutput(
        markdown="# Adaptive Content\n\nThis is adaptive crawl markdown.",
        metadata={
            "url": "https://example.com",
            "title": "Example Page",
            "timestamp": "2024-01-01T00:00:00",
            "status": 200,
            "confidence": 0.85,
            "metrics": {
                "coverage": 0.8,
                "consistency": 0.75,
                "saturation": 0.7,
                "pages_crawled": 5,
            },
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
    assert callable(adaptive_crawl_statistical)
    assert callable(adaptive_crawl_embedding)


# --- Test crawl_deep ---
@pytest.mark.asyncio
async def test_crawl_deep_success(mock_context, mock_runner_output):
    """Test successful crawl_deep execution."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    params = DeepCrawlPresetInput(url="https://example.com", max_depth=2, max_pages=50)
    result_json = await crawl_deep(params=params, ctx=mock_context)

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
    params = DeepCrawlPresetInput(url="https://example.com", max_depth=2)
    result_json = await crawl_deep(params=params, script=script, ctx=mock_context)

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

    config = ExtractionConfigCss(extraction_schema={"name": "Test", "baseSelector": "div"})
    params = DeepCrawlPresetInput(
        url="https://example.com",
        extraction_strategy="css",
        extraction_strategy_config=config,
    )
    result_json = await crawl_deep(params=params, ctx=mock_context)

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify extraction_strategy_params were passed in config
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.config is not None
    assert "extraction_strategy_params" in call_args.config
    assert call_args.config["extraction_strategy_params"]["strategy_type"] == "css"


@pytest.mark.asyncio
async def test_crawl_deep_invalid_url(mock_context):
    """Test crawl_deep with invalid URL."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        await crawl_deep(params={"url": "not-a-url"}, ctx=mock_context)


@pytest.mark.asyncio
async def test_crawl_deep_missing_runner(mock_context_no_runner):
    """Test crawl_deep when crawl_runner is missing from context."""
    with pytest.raises(ValueError) as exc_info:
        await crawl_deep(params=DeepCrawlPresetInput(url="https://example.com"), ctx=mock_context_no_runner)
    assert "crawl_runner not found in context state" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crawl_deep_no_context():
    """Test crawl_deep when context is None."""
    with pytest.raises(ValueError) as exc_info:
        await crawl_deep(params=DeepCrawlPresetInput(url="https://example.com"), ctx=None)
    assert "Context is required" in str(exc_info.value)


# --- Test crawl_deep_smart ---
@pytest.mark.asyncio
async def test_crawl_deep_smart_success(mock_context, mock_runner_output):
    """Test successful crawl_deep_smart execution."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    params = CrawlDeepSmartInput(
        url="https://example.com",
        keywords=["test", "example"],
        max_depth=2,
        max_pages=25,
    )
    result_json = await crawl_deep_smart(params=params, ctx=mock_context)

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify keywords were passed in crawling_strategy_params
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"
    assert call_args.config is not None
    assert "deep_crawl_strategy_params" in call_args.config
    assert call_args.config["deep_crawl_strategy_params"]["strategy_type"] == "best_first"
    assert call_args.config["deep_crawl_strategy_params"]["keywords"] == ["test", "example"]


@pytest.mark.asyncio
async def test_crawl_deep_smart_empty_keywords(mock_context):
    """Test crawl_deep_smart with empty keywords."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        await crawl_deep_smart(params={"url": "https://example.com", "keywords": []}, ctx=mock_context)


@pytest.mark.asyncio
async def test_crawl_deep_smart_with_script(mock_context, mock_runner_output):
    """Test crawl_deep_smart with script."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    script = "SCROLL DOWN 500"
    params = CrawlDeepSmartInput(url="https://example.com", keywords=["test"])
    await crawl_deep_smart(params=params, script=script, ctx=mock_context)

    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


# --- Test scrape_page ---
@pytest.mark.asyncio
async def test_scrape_page_success(mock_context, mock_runner_output):
    """Test successful scrape_page execution."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    params = ScrapePagePresetInput(url="https://example.com")
    result_json = await scrape_page(params=params, ctx=mock_context)

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
    params = ScrapePagePresetInput(url="https://example.com")
    await scrape_page(params=params, script=script, ctx=mock_context)

    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.script == script


@pytest.mark.asyncio
async def test_scrape_page_with_extraction(mock_context, mock_runner_output):
    """Test scrape_page with extraction strategy."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    schema = {"name": "Test", "baseSelector": "div", "fields": []}
    config = ExtractionConfigCss(extraction_schema=schema)
    params = ScrapePagePresetInput(
        url="https://example.com",
        extraction_strategy="css",
        extraction_strategy_config=config,
    )
    result_json = await scrape_page(params=params, ctx=mock_context)

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify extraction_strategy_params were passed in config
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.config is not None
    assert "extraction_strategy_params" in call_args.config
    assert call_args.config["extraction_strategy_params"]["strategy_type"] == "css"


@pytest.mark.asyncio
async def test_scrape_page_error_handling(mock_context):
    """Test scrape_page error handling."""
    error_output = RunnerOutput(
        markdown="", error="Network error: Connection failed", metadata={}
    )
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=error_output)

    params = ScrapePagePresetInput(url="https://example.com")
    result_json = await scrape_page(params=params, ctx=mock_context)

    result = json.loads(result_json)
    assert result["markdown"] == ""
    assert "Network error" in result["error"]


# --- Test parameter passing ---
@pytest.mark.asyncio
async def test_crawl_deep_parameter_passing(mock_context, mock_runner_output):
    """Test that all parameters are correctly passed through."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    params = DeepCrawlPresetInput(
        url="https://example.com",
        max_depth=3,
        max_pages=100,
        include_external=True,
        timeout=90,
        css_selector="article",
        word_count_threshold=50,
        exclude_external_links=True,
    )
    await crawl_deep(params=params, ctx=mock_context)

    # Verify config was built with parameters
    mock_crawl_runner = mock_context.get_state.return_value
    call_args = mock_crawl_runner.run.call_args[0][0]
    assert call_args.config is not None
    config_dict = call_args.config
    
    # Verify crawling strategy params were passed
    assert "deep_crawl_strategy_params" in config_dict
    assert config_dict["deep_crawl_strategy_params"]["strategy_type"] == "bfs"
    assert config_dict["deep_crawl_strategy_params"]["max_depth"] == 3
    assert config_dict["deep_crawl_strategy_params"]["max_pages"] == 100
    assert config_dict["deep_crawl_strategy_params"]["include_external"] is True


@pytest.mark.asyncio
async def test_all_tools_return_json(mock_context, mock_runner_output):
    """Test that all tools return valid JSON strings."""
    mock_crawl_runner = mock_context.get_state.return_value
    mock_crawl_runner.run = AsyncMock(return_value=mock_runner_output)

    # Test all tools
    results = [
        await crawl_deep(params=DeepCrawlPresetInput(url="https://example.com"), ctx=mock_context),
        await crawl_deep_smart(params=CrawlDeepSmartInput(url="https://example.com", keywords=["test"]), ctx=mock_context),
        await scrape_page(params=ScrapePagePresetInput(url="https://example.com"), ctx=mock_context),
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


# --- Test adaptive_crawl_statistical ---
@pytest.mark.asyncio
async def test_adaptive_crawl_statistical_success(
    mock_adaptive_context, mock_adaptive_runner_output
):
    """Test successful adaptive_crawl_statistical execution."""
    mock_adaptive_runner = mock_adaptive_context.get_state.return_value
    mock_adaptive_runner.run = AsyncMock(return_value=mock_adaptive_runner_output)

    params = AdaptiveStatisticalInput(url="https://example.com", query="test query")
    result_json = await adaptive_crawl_statistical(params=params, ctx=mock_adaptive_context)

    # Verify JSON is valid
    result = json.loads(result_json)
    assert result["markdown"] == "# Adaptive Content\n\nThis is adaptive crawl markdown."
    assert result["error"] is None
    assert result["metadata"]["url"] == "https://example.com"
    assert result["metadata"]["confidence"] == 0.85
    assert "metrics" in result["metadata"]
    assert result["metadata"]["metrics"]["pages_crawled"] == 5

    # Verify AdaptiveCrawlRunner was called
    mock_adaptive_runner.run.assert_called_once()
    call_args = mock_adaptive_runner.run.call_args[0][0]
    assert call_args.url == "https://example.com"
    assert call_args.query == "test query"
    assert call_args.config["strategy"] == "statistical"


@pytest.mark.asyncio
async def test_adaptive_crawl_statistical_no_context():
    """Test adaptive_crawl_statistical without context."""
    with pytest.raises(ValueError, match="Context is required"):
        await adaptive_crawl_statistical(
            params=AdaptiveStatisticalInput(url="https://example.com", query="test"),
            ctx=None,
        )


@pytest.mark.asyncio
async def test_adaptive_crawl_statistical_no_runner(mock_context_no_runner):
    """Test adaptive_crawl_statistical without adaptive_crawl_runner in context."""
    with pytest.raises(ValueError, match="adaptive_crawl_runner not found"):
        await adaptive_crawl_statistical(
            params=AdaptiveStatisticalInput(url="https://example.com", query="test"),
            ctx=mock_context_no_runner,
        )


@pytest.mark.asyncio
async def test_adaptive_crawl_statistical_validation_error(mock_adaptive_context):
    """Test adaptive_crawl_statistical with invalid input."""
    with pytest.raises(ValueError):
        await adaptive_crawl_statistical(
            params={"url": "not-a-url", "query": "test"},
            ctx=mock_adaptive_context,
        )


# --- Test adaptive_crawl_embedding ---
@pytest.mark.asyncio
async def test_adaptive_crawl_embedding_success(
    monkeypatch,
    mock_adaptive_context, mock_adaptive_runner_output
):
    """Test successful adaptive_crawl_embedding execution."""
    # Pretend sentence_transformers is available
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    mock_adaptive_runner = mock_adaptive_context.get_state.return_value
    mock_adaptive_runner.run = AsyncMock(return_value=mock_adaptive_runner_output)

    params = AdaptiveEmbeddingInput(
        url="https://example.com",
        query="test query",
        embedding_model="custom-model",
        n_query_variations=15,
    )
    result_json = await adaptive_crawl_embedding(
        params=params,
        ctx=mock_adaptive_context,
    )

    # Verify JSON is valid
    result = json.loads(result_json)
    assert result["error"] is None
    assert result["metadata"]["confidence"] == 0.85

    # Verify AdaptiveCrawlRunner was called with embedding strategy
    mock_adaptive_runner.run.assert_called_once()
    call_args = mock_adaptive_runner.run.call_args[0][0]
    assert call_args.config["strategy"] == "embedding"
    assert call_args.config["adaptive_config_params"]["embedding_model"] == "custom-model"
    assert call_args.config["adaptive_config_params"]["n_query_variations"] == 15


@pytest.mark.asyncio
async def test_adaptive_crawl_embedding_with_llm_config(
    mock_adaptive_context, mock_adaptive_runner_output
):
    """Test adaptive_crawl_embedding with LLM config."""
    mock_adaptive_runner = mock_adaptive_context.get_state.return_value
    mock_adaptive_runner.run = AsyncMock(return_value=mock_adaptive_runner_output)

    llm_config = {
        "provider": "openai/gpt-4",
        "api_token": "test-token",
    }

    params = AdaptiveEmbeddingInput(
        url="https://example.com",
        query="test query",
        embedding_llm_config=llm_config,
    )

    result_json = await adaptive_crawl_embedding(
        params=params,
        ctx=mock_adaptive_context,
    )

    result = json.loads(result_json)
    assert result["error"] is None

    # Verify LLM config was passed
    call_args = mock_adaptive_runner.run.call_args[0][0]
    assert (
        call_args.config["adaptive_config_params"]["embedding_llm_config"]
        == llm_config
    )


@pytest.mark.asyncio
async def test_adaptive_crawl_embedding_no_context():
    """Test adaptive_crawl_embedding without context."""
    with pytest.raises(ValueError, match="Context is required"):
        await adaptive_crawl_embedding(
            params=AdaptiveEmbeddingInput(url="https://example.com", query="test"),
            ctx=None,
        )


@pytest.mark.asyncio
async def test_adaptive_crawl_embedding_validation_error(mock_adaptive_context):
    """Test adaptive_crawl_embedding with invalid input."""
    with pytest.raises(ValueError):
        await adaptive_crawl_embedding(
            params={"url": "https://example.com", "query": ""},  # Empty query
            ctx=mock_adaptive_context,
        )
