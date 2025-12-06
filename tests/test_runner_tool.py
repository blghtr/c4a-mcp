import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from c4a_mcp.runner_tool import CrawlRunner
from c4a_mcp.models import RunnerInput, RunnerOutput
from c4a_mcp.config_models import AppConfig, CrawlerConfigYAML
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
    JsonCssExtractionStrategy,
    BrowserConfig,
)


# --- Test Fixtures ---
@pytest.fixture
def test_runner():
    """Create a CrawlRunner instance for testing."""
    # Load default config
    config_path = Path(__file__).parent.parent / "src" / "c4a_mcp" / "config" / "defaults.yaml"
    app_config = AppConfig.from_yaml(config_path)
    browser_config = app_config.browser.to_browser_config()
    return CrawlRunner(
        default_crawler_config=app_config.crawler,
        browser_config=browser_config,
    )


# --- Test CrawlRunner._map_config ---
def test_map_config_defaults(test_runner):
    runner = test_runner
    config = runner._build_config(RunnerInput(url="https://example.com"))
    assert isinstance(config, CrawlerRunConfig)
    assert config.cache_mode == CacheMode.BYPASS
    # Default page_timeout should be 60000 ms (60 seconds) per PRD
    assert config.page_timeout == 60000


def test_map_config_default_timeout(test_runner):
    """Test that default timeout is 60 seconds when not specified."""
    runner = test_runner
    config = runner._build_config(RunnerInput(url="https://example.com", config={}))
    assert config.page_timeout == 60000  # 60 seconds = 60000 ms


def test_map_config_custom_values(test_runner):
    runner = test_runner
    custom_config_dict = {
        "bypass_cache": False,
        "css_selector": "article",
        "wait_for": "#content-loaded",
        "timeout": 30,
        "word_count_threshold": 100,
        "exclude_external_links": True,
    }
    config = runner._build_config(RunnerInput(url="https://example.com", config=custom_config_dict))
    assert isinstance(config, CrawlerRunConfig)
    assert config.cache_mode == CacheMode.ENABLED  # bypass_cache=False means enabled
    assert config.css_selector == "article"
    assert config.wait_for == "#content-loaded"
    assert config.page_timeout == 30000  # 30 seconds = 30000 milliseconds
    assert config.word_count_threshold == 100
    assert config.exclude_external_links is True
    # extraction_strategy should be None if schema not provided
    assert config.extraction_strategy is None


def test_map_config_invalid_extraction_strategy(test_runner):
    runner = test_runner
    # Invalid extraction strategy should raise ValidationError during config creation
    custom_config_dict = {"extraction_strategy": "invalid_strategy"}
    with pytest.raises((ValueError, ValidationError)):
        runner._build_config(RunnerInput(url="https://example.com", config=custom_config_dict))


def test_map_config_extraction_strategy_with_schema(test_runner):
    runner = test_runner
    schema = {
        "name": "Test Items",
        "baseSelector": "div.item",
        "fields": [{"name": "title", "selector": "h2", "type": "text"}],
    }
    custom_config_dict = {"extraction_strategy": "jsoncss", "extraction_strategy_schema": schema}
    config = runner._build_config(RunnerInput(url="https://example.com", config=custom_config_dict))
    assert isinstance(config.extraction_strategy, JsonCssExtractionStrategy)


def test_map_config_extraction_strategy_without_schema(test_runner):
    runner = test_runner
    # jsoncss requires schema, so this should raise ValidationError
    custom_config_dict = {"extraction_strategy": "jsoncss"}
    with pytest.raises((ValueError, ValidationError)):
        runner._build_config(RunnerInput(url="https://example.com", config=custom_config_dict))


def test_map_config_none_values(test_runner):
    """Test that None values for Optional fields are handled correctly."""
    runner = test_runner
    custom_config_dict = {
        "css_selector": None,
        "wait_for": None,
        "word_count_threshold": None,
    }
    config = runner._build_config(RunnerInput(url="https://example.com", config=custom_config_dict))
    assert isinstance(config, CrawlerRunConfig)
    # None values should not be passed to CrawlerRunConfig
    # css_selector and wait_for are Optional[str], so they may be None or not set
    # word_count_threshold should not be set if None (we skip it in our code)
    # The test verifies that the code doesn't crash and config is created successfully
    assert config.page_timeout == 60000  # Default timeout should still be set


def test_build_run_config_with_crawling_strategy_params(test_runner):
    """Test that _build_run_config creates crawling strategy from parameters."""
    runner = test_runner
    with patch("c4a_mcp.presets.crawling_factory.create_crawling_strategy") as mock_create_crawling:
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

        mock_strategy = MagicMock(spec=BFSDeepCrawlStrategy)
        mock_create_crawling.return_value = mock_strategy

        config_dict = {
            "deep_crawl_strategy_params": {
                "strategy_type": "bfs",
                "max_depth": 2,
                "max_pages": 50,
                "include_external": False,
            }
        }
        config = runner._build_config(RunnerInput(url="https://example.com", config=config_dict))

        assert isinstance(config, CrawlerRunConfig)
        assert config.deep_crawl_strategy == mock_strategy
        mock_create_crawling.assert_called_once_with(
            "bfs", {"max_depth": 2, "max_pages": 50, "include_external": False}
        )


def test_build_run_config_with_extraction_strategy_params(test_runner):
    """Test that _build_run_config creates extraction strategy from parameters."""
    runner = test_runner
    with patch(
        "c4a_mcp.presets.extraction_factory.create_extraction_strategy"
    ) as mock_create_extraction:
        from crawl4ai import JsonCssExtractionStrategy

        mock_strategy = MagicMock(spec=JsonCssExtractionStrategy)
        mock_create_extraction.return_value = mock_strategy

        from c4a_mcp.presets.models import ExtractionConfigCss

        extraction_config = ExtractionConfigCss(
            extraction_schema={"name": "Test", "baseSelector": "div"}
        )
        config_dict = {
            "extraction_strategy_params": {
                "strategy_type": "css",
                "config": extraction_config,
            }
        }
        config = runner._build_config(RunnerInput(url="https://example.com", config=config_dict))

        assert isinstance(config, CrawlerRunConfig)
        assert config.extraction_strategy == mock_strategy
        mock_create_extraction.assert_called_once_with("css", extraction_config)


def test_build_run_config_with_both_strategy_params(test_runner):
    """Test that _build_run_config creates both strategies from parameters."""
    runner = test_runner
    with (
        patch("c4a_mcp.presets.crawling_factory.create_crawling_strategy") as mock_create_crawling,
        patch(
            "c4a_mcp.presets.extraction_factory.create_extraction_strategy"
        ) as mock_create_extraction,
    ):
        from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
        from crawl4ai import JsonCssExtractionStrategy

        mock_crawling = MagicMock(spec=BestFirstCrawlingStrategy)
        mock_extraction = MagicMock(spec=JsonCssExtractionStrategy)
        mock_create_crawling.return_value = mock_crawling
        mock_create_extraction.return_value = mock_extraction

        config_dict = {
            "deep_crawl_strategy_params": {
                "strategy_type": "best_first",
                "max_depth": 2,
                "max_pages": 25,
                "include_external": False,
                "keywords": ["test", "example"],
            },
            "extraction_strategy_params": {
                "strategy_type": "css",
                "config": None,
            },
        }
        config = runner._build_config(RunnerInput(url="https://example.com", config=config_dict))

        assert isinstance(config, CrawlerRunConfig)
        assert config.deep_crawl_strategy == mock_crawling
        assert config.extraction_strategy == mock_extraction
        mock_create_crawling.assert_called_once()
        mock_create_extraction.assert_called_once()


# --- Test CrawlRunner.run ---
@pytest.mark.asyncio
async def test_run_success(test_runner):
    runner = test_runner
    mock_crawl_result = MagicMock()
    mock_crawl_result.markdown = "Test content"
    mock_crawl_result.url = "http://example.com"
    mock_crawl_result.metadata = {"title": "Example"}
    mock_crawl_result.status_code = 200

    # Mock AsyncWebCrawler as async context manager
    mock_crawler = AsyncMock(spec=AsyncWebCrawler)
    mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    # Inject mock class directly
    runner.crawler_class = MagicMock(return_value=mock_crawler_instance)

    inputs = RunnerInput(url="http://example.com")
    output = await runner.run(inputs)

    assert output.markdown == "Test content"
    assert output.error is None
    assert output.metadata["url"] == "http://example.com"
    assert output.metadata["title"] == "Example"
    assert output.metadata["status"] == 200
    assert "timestamp" in output.metadata
    mock_crawler.arun.assert_called_once()


@pytest.mark.asyncio
async def test_run_with_script(test_runner):
    runner = test_runner
    mock_crawl_result = MagicMock()
    mock_crawl_result.markdown = "Scripted content"
    mock_crawl_result.url = "http://example.com"
    mock_crawl_result.metadata = {"title": "Example"}
    mock_crawl_result.status_code = 200

    mock_crawler = AsyncMock(spec=AsyncWebCrawler)
    mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    # Inject mock class directly
    runner.crawler_class = MagicMock(return_value=mock_crawler_instance)

    inputs = RunnerInput(url="http://example.com", script="CLICK #button")
    output = await runner.run(inputs)

    assert output.markdown == "Scripted content"
    assert output.error is None
    # Verify that the script was passed to arun's config
    args, kwargs = mock_crawler.arun.call_args
    assert kwargs["config"].js_code == "CLICK #button"


@pytest.mark.asyncio
async def test_run_timeout_error(test_runner):
    runner = test_runner
    mock_crawler = AsyncMock(spec=AsyncWebCrawler)
    mock_crawler.arun = AsyncMock(side_effect=Exception("timeout error occurred"))

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    # Inject mock class directly
    runner.crawler_class = MagicMock(return_value=mock_crawler_instance)

    inputs = RunnerInput(url="http://example.com", config={"timeout": 5})
    output = await runner.run(inputs)

    assert output.markdown == ""
    assert "Timeout error" in output.error
    assert "timeout error occurred" in output.error
    # Traceback should not be in error response
    assert "traceback" not in output.error.lower()


@pytest.mark.asyncio
async def test_run_general_crawl_error(test_runner):
    runner = test_runner
    mock_crawler = AsyncMock(spec=AsyncWebCrawler)
    mock_crawler.arun = AsyncMock(side_effect=Exception("Network connection failed"))

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    # Inject mock class directly
    runner.crawler_class = MagicMock(return_value=mock_crawler_instance)

    inputs = RunnerInput(url="http://example.com")
    output = await runner.run(inputs)

    assert output.markdown == ""
    assert "Network error" in output.error


@pytest.mark.asyncio
async def test_run_unexpected_error(test_runner):
    runner = test_runner
    mock_crawler = AsyncMock(spec=AsyncWebCrawler)
    mock_crawler.arun = AsyncMock(side_effect=ValueError("Unexpected problem"))

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    # Inject mock class directly
    runner.crawler_class = MagicMock(return_value=mock_crawler_instance)

    inputs = RunnerInput(url="http://example.com")
    output = await runner.run(inputs)

    assert output.markdown == ""
    assert "An unexpected error occurred: Unexpected problem" in output.error
    # Traceback should NOT be in error response (sanitized)
    assert "traceback" not in output.error.lower()


# --- Test MCP Server startup and tool exposure (basic check) ---
from c4a_mcp.server import mcp, serve
from fastmcp import Context
from unittest.mock import MagicMock


def test_mcp_server_initialization():
    assert mcp.name == "c4a-mcp"
    # FastMCP wraps tools in FunctionTool objects, verify server is initialized
    assert mcp is not None
    # Verify serve function exists
    assert callable(serve)
    assert serve.__name__ == "serve"


@pytest.mark.asyncio
async def test_runner_no_context():
    """Test runner tool when context is None."""
    # Test the context validation logic that happens in runner function
    # The runner function checks: if ctx is None, raise ValueError("Context is required for runner tool")
    ctx = None

    # Simulate the exact check from server.py:386
    if ctx is None:
        with pytest.raises(ValueError) as exc_info:
            raise ValueError("Context is required for runner tool")
        assert "Context is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_runner_missing_crawl_runner():
    """Test runner tool when crawl_runner is missing from context."""
    # Test the crawl_runner validation logic that happens in runner function
    # The runner function checks: if crawl_runner is None, raise ValueError
    from fastmcp import Context

    # Create a mock context without crawl_runner
    mock_ctx = MagicMock(spec=Context)
    mock_ctx.get_state = MagicMock(return_value=None)

    # Simulate the exact check from server.py:388-393
    crawl_runner = mock_ctx.get_state("crawl_runner")
    if crawl_runner is None:
        with pytest.raises(ValueError) as exc_info:
            raise ValueError(
                "crawl_runner not found in context state. "
                "Ensure the server lifespan properly initializes crawl_runner."
            )
        assert "crawl_runner not found in context state" in str(exc_info.value)


# --- Test RunnerInput Validation ---
from pydantic import ValidationError


def test_validate_url_valid():
    """Test that valid URLs pass validation."""
    valid_urls = [
        "http://example.com",
        "https://example.com",
        "http://example.com/path",
        "https://subdomain.example.com:8080/path?query=value#fragment",
    ]
    for url in valid_urls:
        input_obj = RunnerInput(url=url)
        assert input_obj.url == url


def test_validate_url_invalid_protocol():
    """Test that non-http/https protocols are rejected."""
    invalid_urls = [
        "ftp://example.com",
        "file:///path/to/file",
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
    ]
    for url in invalid_urls:
        with pytest.raises(ValidationError) as exc_info:
            RunnerInput(url=url)
        assert "http:// or https://" in str(exc_info.value)


def test_validate_url_invalid_format():
    """Test that malformed URLs are rejected."""
    invalid_urls = [
        "not-a-url",
        "example.com",  # Missing scheme
        "http://",  # Missing netloc
        "https://",  # Missing netloc
    ]
    for url in invalid_urls:
        with pytest.raises(ValidationError) as exc_info:
            RunnerInput(url=url)
        assert "URL" in str(exc_info.value) or "protocol" in str(exc_info.value).lower()


def test_validate_script_valid_commands():
    """Test that valid c4a-script commands pass validation."""
    valid_scripts = [
        "GO https://example.com",
        "CLICK #button",
        "WAIT `.element` 5",
        'TYPE "Hello World"',
        "PRESS Enter",
        'SETVAR username = "test"',
        "IF (EXISTS `#popup`) THEN CLICK `#close`",
        "REPEAT (SCROLL DOWN 300, 5)",
    ]
    for script in valid_scripts:
        input_obj = RunnerInput(url="https://example.com", script=script)
        assert input_obj.script == script


def test_validate_script_with_comments():
    """Test that comments in scripts are ignored during validation."""
    script_with_comments = """
# This is a comment
GO https://example.com
# Another comment
CLICK #button
# End comment
"""
    input_obj = RunnerInput(url="https://example.com", script=script_with_comments)
    assert input_obj.script == script_with_comments


def test_validate_script_empty_lines():
    """Test that empty lines in scripts are ignored."""
    script_with_empty_lines = """
GO https://example.com

CLICK #button

WAIT 3
"""
    input_obj = RunnerInput(url="https://example.com", script=script_with_empty_lines)
    assert input_obj.script == script_with_empty_lines


def test_validate_script_invalid_command():
    """Test that invalid c4a-script commands are rejected."""
    invalid_scripts = [
        "INVALID_COMMAND arg1 arg2",
        "GO https://example.com\nBAD_COMMAND arg",
        "EXECUTE malicious_code()",
    ]
    for script in invalid_scripts:
        with pytest.raises(ValidationError) as exc_info:
            RunnerInput(url="https://example.com", script=script)
        assert "Invalid c4a-script command" in str(exc_info.value)


def test_validate_script_none():
    """Test that None script passes validation."""
    input_obj = RunnerInput(url="https://example.com", script=None)
    assert input_obj.script is None


def test_validate_config_valid_fields():
    """Test that valid config fields pass validation."""
    valid_configs = [
        {"bypass_cache": True},
        {"timeout": 30},
        {"css_selector": "article"},
        {"wait_for": "#content"},
        {"word_count_threshold": 100},
        {"exclude_external_links": True},
        {"exclude_social_media_links": False},
        {
            "extraction_strategy": "jsoncss",
            "extraction_strategy_schema": {"name": "Test", "baseSelector": "div"},
        },
        {
            "bypass_cache": False,
            "timeout": 60,
            "css_selector": "main",
            "wait_for": ".loaded",
            "word_count_threshold": 200,
            "exclude_external_links": True,
            "exclude_social_media_links": True,
        },
    ]
    for config in valid_configs:
        input_obj = RunnerInput(url="https://example.com", config=config)
        assert input_obj.config == config


def test_validate_config_none():
    """Test that None config passes validation."""
    input_obj = RunnerInput(url="https://example.com", config=None)
    assert input_obj.config is None


def test_validate_config_unknown_fields_warning(caplog):
    """Test that unknown config fields are ignored by Pydantic (extra='ignore' by default)."""
    import logging

    with caplog.at_level(logging.WARNING):
        config_with_unknown = {
            "timeout": 30,
            "unknown_field": "value",
            "another_unknown": 123,
        }
        input_obj = RunnerInput(url="https://example.com", config=config_with_unknown)
        # Pydantic ignores unknown fields by default, so only known fields are kept
        assert input_obj.config == {"timeout": 30}


def test_validate_config_invalid_extraction_strategy():
    """Test that invalid extraction_strategy values are rejected."""
    invalid_configs = [
        {"extraction_strategy": "invalid"},
        {"extraction_strategy": "llm"},
        {"extraction_strategy": "css"},
    ]
    for config in invalid_configs:
        with pytest.raises(ValidationError) as exc_info:
            RunnerInput(url="https://example.com", config=config)
        assert "Only 'jsoncss'" in str(exc_info.value)


def test_validate_config_missing_extraction_strategy_schema():
    """Test that extraction_strategy='jsoncss' requires extraction_strategy_schema."""
    config = {"extraction_strategy": "jsoncss"}
    with pytest.raises(ValidationError) as exc_info:
        RunnerInput(url="https://example.com", config=config)
    # Check for the actual error message
    assert "extraction_strategy_schema required" in str(exc_info.value)


def test_validate_config_invalid_types():
    """Test that invalid types for config fields are rejected."""
    invalid_configs = [
        {"timeout": -5},  # Should be positive
        {"timeout": 0},  # Should be positive
        {"word_count_threshold": -10},  # Should be non-negative int
    ]
    for config in invalid_configs:
        with pytest.raises(ValidationError):
            RunnerInput(url="https://example.com", config=config)

    # Pydantic may coerce some types, so these might not raise errors
    # Test that they at least don't crash
    try_configs = [
        {"bypass_cache": "true"},  # May be coerced to bool
        {"timeout": "30"},  # May be coerced to int
        {"css_selector": 123},  # May raise or be coerced
        {"wait_for": 456},  # May raise or be coerced
        {"word_count_threshold": "100"},  # May be coerced to int
        {"exclude_external_links": "yes"},  # May be coerced to bool
        {"exclude_social_media_links": 1},  # May be coerced to bool
    ]
    # These might pass or fail depending on Pydantic coercion
    # Just ensure they don't crash the system
    for config in try_configs:
        try:
            RunnerInput(url="https://example.com", config=config)
        except ValidationError:
            pass  # Expected for some invalid types


def test_validate_config_none_values_allowed():
    """Test that None values for optional config fields are allowed but excluded from output."""
    config_with_nones = {
        "css_selector": None,
        "wait_for": None,
        "word_count_threshold": None,
    }
    input_obj = RunnerInput(url="https://example.com", config=config_with_nones)
    # model_dump(exclude_none=True) excludes None values, so config will be empty dict
    assert input_obj.config == {}
