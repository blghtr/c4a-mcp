# LLM:METADATA
# :hierarchy: [Tests | Unit | Crawler Registry]
# :rationale: "Ensure crawler lifecycle helpers close resources and clear registry"
# :contract: pre: "Fake crawler objects support awaited close paths", post: "Registry empty and correct close path taken"
# LLM:END

import asyncio
import logging

import pytest

from c4a_mcp.crawler_registry import (
    cleanup_all,
    close_crawler,
    deregister,
    register,
)


def _make_logger():
    logger = logging.getLogger("test_crawler_registry")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    return logger


class FakeAsync:
    def __init__(self):
        self.called = False
        self.to_raise = None
        self.args = None
        self.kwargs = None

    async def __call__(self, *args, **kwargs):
        self.called = True
        self.args = args
        self.kwargs = kwargs
        if self.to_raise:
            raise self.to_raise


class FakeCrawler:
    def __init__(self):
        self.__aexit__ = FakeAsync()  # type: ignore[attr-defined]
        self.browser = None
        self.playwright = None


@pytest.mark.asyncio
async def test_close_crawler_prefers_aexit_success():
    logger = _make_logger()
    crawler = FakeCrawler()
    register(crawler)

    await close_crawler(crawler, logger, timeout=1)

    assert crawler.__aexit__.called  # type: ignore[attr-defined]
    # Registry not auto-cleared here; caller decides.
    deregister(crawler)


@pytest.mark.asyncio
async def test_close_crawler_fallback_to_browser_then_playwright():
    logger = _make_logger()
    crawler = FakeCrawler()
    crawler.__aexit__.to_raise = RuntimeError("boom")  # type: ignore[attr-defined]
    crawler.browser = type("B", (), {"close": FakeAsync()})()
    crawler.playwright = type("P", (), {"stop": FakeAsync()})()

    register(crawler)
    await close_crawler(crawler, logger, timeout=1)

    assert crawler.__aexit__.called  # type: ignore[attr-defined]
    assert crawler.browser.close.called  # type: ignore[attr-defined]
    assert not crawler.playwright.stop.called  # type: ignore[attr-defined]
    deregister(crawler)


@pytest.mark.asyncio
async def test_close_crawler_uses_playwright_when_browser_fails():
    logger = _make_logger()
    crawler = FakeCrawler()
    crawler.__aexit__.to_raise = RuntimeError("aexit fails")  # type: ignore[attr-defined]
    crawler.browser = type("B", (), {"close": FakeAsync()})()
    crawler.browser.close.to_raise = RuntimeError("browser fails")  # type: ignore[attr-defined]
    crawler.playwright = type("P", (), {"stop": FakeAsync()})()

    register(crawler)
    await close_crawler(crawler, logger, timeout=1)

    assert crawler.__aexit__.called  # type: ignore[attr-defined]
    assert crawler.browser.close.called  # type: ignore[attr-defined]
    assert crawler.playwright.stop.called  # type: ignore[attr-defined]
    deregister(crawler)


@pytest.mark.asyncio
async def test_cleanup_all_closes_all_and_clears_registry():
    logger = _make_logger()
    crawler1 = FakeCrawler()
    crawler2 = FakeCrawler()
    crawler2.__aexit__.to_raise = RuntimeError("force fallback")  # type: ignore[attr-defined]
    crawler2.browser = type("B", (), {"close": FakeAsync()})()

    register(crawler1)
    register(crawler2)

    await cleanup_all(logger, timeout=1)

    assert crawler1.__aexit__.called  # type: ignore[attr-defined]
    assert crawler2.__aexit__.called  # type: ignore[attr-defined]
    assert crawler2.browser.close.called  # type: ignore[attr-defined]


