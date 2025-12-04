# LLM:METADATA
# :hierarchy: [C4A-MCP | Tests | Adaptive Runner]
# :relates-to: tests: "adaptive_runner", implements: "SPEC-F004 extension"
# :rationale: "Validates AdaptiveCrawlRunner implementation, ensuring it correctly uses AdaptiveCrawler and extracts confidence/metrics."
# :contract: pre: "Mocked AdaptiveCrawler and AsyncWebCrawler", post: "All runner methods return valid RunnerOutput"
# :decision_cache: "Integration tests with mocked crawl4ai components to verify runner behavior without actual crawling [TEST-007]"
# LLM:END

"""
Tests for adaptive crawling runner.

Validates that AdaptiveCrawlRunner correctly:
- Creates AdaptiveConfig from parameters
- Uses AdaptiveCrawler.digest() method
- Extracts confidence and metrics from results
- Handles errors appropriately
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from c4a_mcp.adaptive_runner import AdaptiveCrawlRunner, AdaptiveRunnerInput
from c4a_mcp.models import RunnerOutput
from c4a_mcp.config_models import CrawlerConfigYAML
from crawl4ai import BrowserConfig


# --- Test Fixtures ---
@pytest.fixture
def mock_browser_config():
    """Create a mock BrowserConfig."""
    config = MagicMock(spec=BrowserConfig)
    config.browser_type = "chromium"
    config.headless = True
    return config


@pytest.fixture
def mock_crawler_config():
    """Create a mock CrawlerConfigYAML."""
    config = MagicMock(spec=CrawlerConfigYAML)
    config.timeout = None  # Add timeout attribute
    return config


@pytest.fixture
def adaptive_runner(mock_crawler_config, mock_browser_config):
    """Create an AdaptiveCrawlRunner instance for testing."""
    return AdaptiveCrawlRunner(
        default_crawler_config=mock_crawler_config,
        browser_config=mock_browser_config,
    )


@pytest.fixture
def mock_adaptive_result():
    """Create a mock result from AdaptiveCrawler.digest()."""
    result = MagicMock()
    result.markdown = "# Test Content\n\nThis is test markdown."
    result.url = "https://example.com"
    result.status_code = 200
    result.metadata = {"title": "Example Page"}
    result.crawled_urls = ["https://example.com", "https://example.com/page1"]
    return result


@pytest.fixture
def mock_adaptive_crawler(mock_adaptive_result):
    """Create a mock AdaptiveCrawler."""
    adaptive = MagicMock()
    adaptive.digest = AsyncMock(return_value=mock_adaptive_result)
    adaptive.confidence = 0.85
    adaptive.metrics = {
        "coverage": 0.8,
        "consistency": 0.75,
        "saturation": 0.7,
    }
    return adaptive


# --- Test AdaptiveCrawlRunner.run ---
@patch("c4a_mcp.adaptive_runner.AsyncWebCrawler")
@patch("c4a_mcp.adaptive_runner.PatchedAdaptiveCrawler")
@patch("c4a_mcp.presets.adaptive_factory.create_adaptive_config")
@patch("contextlib.redirect_stdout")
@patch("contextlib.redirect_stderr")
@pytest.mark.asyncio
async def test_run_success(
    mock_redirect_stderr,
    mock_redirect_stdout,
    mock_create_config,
    mock_patched_adaptive_crawler_class,
    mock_async_web_crawler_class,
    adaptive_runner,
    mock_adaptive_crawler,
    mock_adaptive_result,
):
    """Test successful adaptive crawl execution."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.strategy = "statistical"  # Add strategy attribute
    mock_create_config.return_value = mock_config
    
    mock_crawler = MagicMock()
    mock_async_web_crawler_class.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_async_web_crawler_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    # Ensure mock_adaptive_crawler has strategy attribute for patching
    mock_adaptive_crawler.strategy = MagicMock()  # Mock strategy object
    mock_patched_adaptive_crawler_class.return_value = mock_adaptive_crawler
    
    # Mock context managers for redirect_stdout/stderr
    mock_redirect_stdout.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stdout.return_value.__exit__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__exit__ = MagicMock(return_value=None)
    
    # Create input
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test query",
        config={
            "strategy": "statistical",
            "adaptive_config_params": {
                "confidence_threshold": 0.7,
                "max_pages": 20,
            },
        },
    )
    
    # Execute
    result = await adaptive_runner.run(inputs)
    
    # Verify
    assert isinstance(result, RunnerOutput)
    assert result.markdown == "# Test Content\n\nThis is test markdown."
    assert result.error is None
    assert result.metadata["url"] == "https://example.com"
    assert result.metadata["title"] == "Example Page"
    assert result.metadata["confidence"] == 0.85
    assert "metrics" in result.metadata
    assert result.metadata["metrics"]["coverage"] == 0.8
    assert result.metadata["metrics"]["pages_crawled"] == 2
    
    # Verify PatchedAdaptiveCrawler was created and digest was called
    mock_patched_adaptive_crawler_class.assert_called_once_with(
        mock_crawler, mock_config, link_preview_timeout=30, page_timeout=60000
    )
    mock_adaptive_crawler.digest.assert_called_once_with(
        start_url="https://example.com",
        query="test query"
    )


@patch("c4a_mcp.adaptive_runner.AsyncWebCrawler")
@patch("c4a_mcp.adaptive_runner.PatchedAdaptiveCrawler")
@patch("c4a_mcp.presets.adaptive_factory.create_adaptive_config")
@patch("contextlib.redirect_stdout")
@patch("contextlib.redirect_stderr")
@pytest.mark.asyncio
async def test_run_with_string_markdown(
    mock_redirect_stderr,
    mock_redirect_stdout,
    mock_create_config,
    mock_patched_adaptive_crawler_class,
    mock_async_web_crawler_class,
    adaptive_runner,
):
    """Test handling when markdown is already a string."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.strategy = "statistical"  # Add strategy attribute
    mock_create_config.return_value = mock_config
    
    mock_crawler = MagicMock()
    mock_async_web_crawler_class.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_async_web_crawler_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_result = MagicMock()
    mock_result.markdown = "Simple string markdown"
    mock_result.url = "https://example.com"
    mock_result.status_code = 200
    mock_result.metadata = {}
    mock_result.crawled_urls = []
    
    mock_adaptive = MagicMock()
    mock_adaptive.digest = AsyncMock(return_value=mock_result)
    mock_adaptive.confidence = 0.5
    mock_adaptive.metrics = {}
    mock_adaptive.strategy = MagicMock()  # Add strategy for patching
    mock_patched_adaptive_crawler_class.return_value = mock_adaptive
    
    # Mock context managers
    mock_redirect_stdout.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stdout.return_value.__exit__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__exit__ = MagicMock(return_value=None)
    
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test",
        config={"strategy": "statistical", "adaptive_config_params": {}},
    )
    
    result = await adaptive_runner.run(inputs)
    
    assert result.markdown == "Simple string markdown"
    assert result.metadata["confidence"] == 0.5


@patch("c4a_mcp.adaptive_runner.AsyncWebCrawler")
@patch("c4a_mcp.adaptive_runner.PatchedAdaptiveCrawler")
@patch("c4a_mcp.presets.adaptive_factory.create_adaptive_config")
@patch("contextlib.redirect_stdout")
@patch("contextlib.redirect_stderr")
@pytest.mark.asyncio
async def test_run_error_handling(
    mock_redirect_stderr,
    mock_redirect_stdout,
    mock_create_config,
    mock_patched_adaptive_crawler_class,
    mock_async_web_crawler_class,
    adaptive_runner,
):
    """Test error handling during adaptive crawl."""
    # Setup mocks to raise exception
    mock_config = MagicMock()
    mock_config.strategy = "statistical"  # Add strategy attribute
    mock_create_config.return_value = mock_config
    
    mock_crawler = MagicMock()
    mock_async_web_crawler_class.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_async_web_crawler_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_adaptive = MagicMock()
    mock_adaptive.digest = AsyncMock(side_effect=Exception("Network error: connection failed"))
    mock_adaptive.strategy = MagicMock()  # Add strategy for patching
    mock_patched_adaptive_crawler_class.return_value = mock_adaptive
    
    # Mock context managers
    mock_redirect_stdout.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stdout.return_value.__exit__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__exit__ = MagicMock(return_value=None)
    
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test",
        config={"strategy": "statistical", "adaptive_config_params": {}},
    )
    
    result = await adaptive_runner.run(inputs)
    
    assert isinstance(result, RunnerOutput)
    assert result.markdown == ""
    assert result.error is not None
    assert "Network error" in result.error


@patch("c4a_mcp.adaptive_runner.AsyncWebCrawler")
@patch("c4a_mcp.adaptive_runner.PatchedAdaptiveCrawler")
@patch("c4a_mcp.presets.adaptive_factory.create_adaptive_config")
@patch("contextlib.redirect_stdout")
@patch("contextlib.redirect_stderr")
@pytest.mark.asyncio
async def test_run_timeout_error(
    mock_redirect_stderr,
    mock_redirect_stdout,
    mock_create_config,
    mock_patched_adaptive_crawler_class,
    mock_async_web_crawler_class,
    adaptive_runner,
):
    """Test timeout error handling."""
    mock_config = MagicMock()
    mock_config.strategy = "statistical"  # Add strategy attribute
    mock_create_config.return_value = mock_config
    
    mock_crawler = MagicMock()
    mock_async_web_crawler_class.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_async_web_crawler_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_adaptive = MagicMock()
    mock_adaptive.digest = AsyncMock(side_effect=TimeoutError("Timeout after 60 seconds"))
    mock_adaptive.strategy = MagicMock()  # Add strategy for patching
    mock_patched_adaptive_crawler_class.return_value = mock_adaptive
    
    # Mock context managers
    mock_redirect_stdout.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stdout.return_value.__exit__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__exit__ = MagicMock(return_value=None)
    
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test",
        config={"strategy": "statistical", "adaptive_config_params": {}},
    )
    
    result = await adaptive_runner.run(inputs)
    
    assert result.error is not None
    assert "Timeout" in result.error


@patch("c4a_mcp.adaptive_runner.AsyncWebCrawler")
@patch("c4a_mcp.adaptive_runner.PatchedAdaptiveCrawler")
@patch("c4a_mcp.presets.adaptive_factory.create_adaptive_config")
@patch("contextlib.redirect_stdout")
@patch("contextlib.redirect_stderr")
@pytest.mark.asyncio
async def test_run_embedding_error(
    mock_redirect_stderr,
    mock_redirect_stdout,
    mock_create_config,
    mock_patched_adaptive_crawler_class,
    mock_async_web_crawler_class,
    adaptive_runner,
):
    """Test embedding/LLM error handling."""
    mock_config = MagicMock()
    mock_config.strategy = "embedding"  # Add strategy attribute
    mock_create_config.return_value = mock_config
    
    mock_crawler = MagicMock()
    mock_async_web_crawler_class.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_async_web_crawler_class.return_value.__aexit__ = AsyncMock(return_value=None)
    
    mock_adaptive = MagicMock()
    mock_adaptive.digest = AsyncMock(side_effect=Exception("Embedding model not found"))
    mock_adaptive.strategy = MagicMock()  # Add strategy for patching
    mock_patched_adaptive_crawler_class.return_value = mock_adaptive
    
    # Mock context managers
    mock_redirect_stdout.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stdout.return_value.__exit__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__enter__ = MagicMock(return_value=None)
    mock_redirect_stderr.return_value.__exit__ = MagicMock(return_value=None)
    
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test",
        config={"strategy": "embedding", "adaptive_config_params": {}},
    )
    
    result = await adaptive_runner.run(inputs)
    
    assert result.error is not None
    assert "Embedding/LLM error" in result.error or "unexpected error" in result.error.lower()


@pytest.mark.asyncio
async def test_run_invalid_strategy(adaptive_runner):
    """Test validation of invalid strategy."""
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test",
        config={"strategy": "invalid_strategy", "adaptive_config_params": {}},
    )
    
    result = await adaptive_runner.run(inputs)
    
    assert result.error is not None
    assert "Invalid strategy" in result.error
    assert "invalid_strategy" in result.error


@pytest.mark.asyncio
async def test_run_invalid_adaptive_params_type(adaptive_runner):
    """Test validation of invalid adaptive_params type."""
    inputs = AdaptiveRunnerInput(
        url="https://example.com",
        query="test",
        config={"strategy": "statistical", "adaptive_config_params": "not_a_dict"},
    )
    
    result = await adaptive_runner.run(inputs)
    
    assert result.error is not None
    assert "adaptive_config_params must be a dict" in result.error


# --- Test PatchedEmbeddingStrategy._get_embedding_llm_config_dict ---
def test_get_embedding_llm_config_dict_none_config():
    """Test that patched _get_embedding_llm_config_dict returns None when embedding_llm_config is None."""
    from c4a_mcp.adaptive_runner import PatchedEmbeddingStrategy
    from crawl4ai.adaptive_crawler import EmbeddingStrategy
    
    # Create an EmbeddingStrategy instance
    strategy = EmbeddingStrategy()
    
    # Create a mock config with embedding_llm_config = None
    mock_config = MagicMock()
    mock_config.embedding_llm_config = None
    mock_config._embedding_llm_config_dict = None
    strategy.config = mock_config
    
    # Patch the strategy
    PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)
    
    # Test: should return None when embedding_llm_config is None
    result = strategy._get_embedding_llm_config_dict()
    assert result is None


def test_get_embedding_llm_config_dict_no_api_key():
    """Test that patched _get_embedding_llm_config_dict returns None when config dict has no API key."""
    from c4a_mcp.adaptive_runner import PatchedEmbeddingStrategy
    from crawl4ai.adaptive_crawler import EmbeddingStrategy
    
    # Create an EmbeddingStrategy instance
    strategy = EmbeddingStrategy()
    
    # Create a mock config with config dict but no API key
    mock_config = MagicMock()
    mock_config.embedding_llm_config = MagicMock()  # Not None, but no valid API key
    mock_config._embedding_llm_config_dict = {
        'provider': 'openai/text-embedding-3-small',
        'api_token': None,  # No API key
    }
    strategy.config = mock_config
    
    # Patch the strategy
    PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)
    
    # Test: should return None when API key is missing/None
    result = strategy._get_embedding_llm_config_dict()
    assert result is None


def test_get_embedding_llm_config_dict_empty_api_key():
    """Test that patched _get_embedding_llm_config_dict returns None when API key is empty string."""
    from c4a_mcp.adaptive_runner import PatchedEmbeddingStrategy
    from crawl4ai.adaptive_crawler import EmbeddingStrategy
    
    # Create an EmbeddingStrategy instance
    strategy = EmbeddingStrategy()
    
    # Create a mock config with empty API key
    mock_config = MagicMock()
    mock_config.embedding_llm_config = MagicMock()
    mock_config._embedding_llm_config_dict = {
        'provider': 'openai/text-embedding-3-small',
        'api_token': '',  # Empty string
    }
    strategy.config = mock_config
    
    # Patch the strategy
    PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)
    
    # Test: should return None when API key is empty
    result = strategy._get_embedding_llm_config_dict()
    assert result is None


def test_get_embedding_llm_config_dict_valid_api_key():
    """Test that patched _get_embedding_llm_config_dict returns config when valid API key is present."""
    from c4a_mcp.adaptive_runner import PatchedEmbeddingStrategy
    from crawl4ai.adaptive_crawler import EmbeddingStrategy
    
    # Create an EmbeddingStrategy instance
    strategy = EmbeddingStrategy()
    
    # Create a mock config with valid API key
    mock_config = MagicMock()
    mock_config.embedding_llm_config = MagicMock()
    config_dict = {
        'provider': 'openai/text-embedding-3-small',
        'api_token': 'sk-valid-key-12345',
    }
    mock_config._embedding_llm_config_dict = config_dict
    strategy.config = mock_config
    
    # Patch the strategy
    PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)
    
    # Test: should return config dict when valid API key is present
    result = strategy._get_embedding_llm_config_dict()
    assert result == config_dict
    assert result['api_token'] == 'sk-valid-key-12345'


def test_get_embedding_llm_config_dict_api_key_variant():
    """Test that patched _get_embedding_llm_config_dict checks both api_token and api_key fields."""
    from c4a_mcp.adaptive_runner import PatchedEmbeddingStrategy
    from crawl4ai.adaptive_crawler import EmbeddingStrategy
    
    # Create an EmbeddingStrategy instance
    strategy = EmbeddingStrategy()
    
    # Test with api_key instead of api_token
    mock_config = MagicMock()
    mock_config.embedding_llm_config = MagicMock()
    config_dict = {
        'provider': 'openai/text-embedding-3-small',
        'api_key': 'sk-valid-key-67890',  # Using api_key instead of api_token
    }
    mock_config._embedding_llm_config_dict = config_dict
    strategy.config = mock_config
    
    # Patch the strategy
    PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)
    
    # Test: should return config dict when valid API key is present (using api_key field)
    result = strategy._get_embedding_llm_config_dict()
    assert result == config_dict
    assert result['api_key'] == 'sk-valid-key-67890'


def test_get_embedding_llm_config_dict_no_config():
    """Test that patched _get_embedding_llm_config_dict returns None when config is None."""
    from c4a_mcp.adaptive_runner import PatchedEmbeddingStrategy
    from crawl4ai.adaptive_crawler import EmbeddingStrategy
    
    # Create an EmbeddingStrategy instance
    strategy = EmbeddingStrategy()
    strategy.config = None
    
    # Patch the strategy
    PatchedEmbeddingStrategy.patch_embedding_strategy(strategy)
    
    # Test: should return None when config is None
    result = strategy._get_embedding_llm_config_dict()
    assert result is None

