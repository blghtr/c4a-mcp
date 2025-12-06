# LLM:METADATA
# :hierarchy: [C4A-MCP | Infra | Crawler Registry]
# :rationale: "Track live AsyncWebCrawler instances to ensure cleanup on cancel or preflight"
# :contract: pre: "Crawler instances registered/deregistered around use", post: "All tracked crawlers can be closed safely"
# LLM:END

import asyncio
import logging
import weakref

import psutil

_registry: set[weakref.ref] = set()


def register(crawler) -> None:
    """Add crawler to registry via weakref to avoid leaks."""
    try:
        _registry.add(weakref.ref(crawler))
    except Exception:
        # Registry is best-effort; ignore failures
        pass


def deregister(crawler) -> None:
    """Remove crawler from registry if present."""
    dead = []
    for ref in _registry:
        obj = ref()
        if obj is None or obj is crawler:
            dead.append(ref)
    for ref in dead:
        _registry.discard(ref)


async def _await_with_timeout(coro, timeout: float) -> None:
    await asyncio.wait_for(coro, timeout=timeout)


async def close_crawler(
    crawler,
    logger: logging.Logger,
    timeout: float = 10.0,
) -> None:
    """
    Best-effort close of a crawler with fallback to browser/playwright handles.

    Args:
        crawler: AsyncWebCrawler instance
        logger: logger for diagnostics
        timeout: timeout seconds for orderly close
    """
    if crawler is None:
        return

    try:
        await _await_with_timeout(crawler.__aexit__(None, None, None), timeout)
        return
    except asyncio.TimeoutError:
        logger.warning(
            "[C4A-MCP | Infra | Crawler Registry] __aexit__ timed out (likely connection closed) | "
            "data: {timeout: %.1f}s",
            timeout,
        )
    except Exception as e:
        logger.warning(
            "[C4A-MCP | Infra | Crawler Registry] __aexit__ failed, trying fallbacks | "
            "data: {error: %s, type: %s}",
            str(e),
            type(e).__name__,
        )

    # Fallback: try browser then playwright
    browser = getattr(crawler, "browser", None)
    if browser is not None:
        try:
            await _await_with_timeout(browser.close(), timeout)
            return
        except Exception as e:
            logger.warning(
                "[C4A-MCP | Infra | Crawler Registry] browser.close failed | data: {error: %s, type: %s}",
                str(e),
                type(e).__name__,
            )

    playwright = getattr(crawler, "playwright", None)
    if playwright is not None:
        try:
            await _await_with_timeout(playwright.stop(), timeout)
        except Exception as e:
            logger.error(
                "[C4A-MCP | Infra | Crawler Registry] playwright.stop failed | data: {error: %s, type: %s}",
                str(e),
                type(e).__name__,
            )


async def cleanup_all(logger: logging.Logger, timeout: float = 10.0) -> None:
    """Close all tracked crawlers; registry is cleared as items are processed."""
    refs = list(_registry)
    for ref in refs:
        crawler = ref()
        if crawler is None:
            _registry.discard(ref)
            continue
        await close_crawler(crawler, logger, timeout=timeout)
        _registry.discard(ref)


def _is_playwright_like(proc: psutil.Process) -> bool:
    """Heuristic to detect playwright/Chrome child processes spawned by this service."""
    try:
        name = (proc.name() or "").lower()
    except psutil.Error:
        name = ""
    match_tokens = ("playwright", "chrome", "chromium", "msedge", "node")
    if any(tok in name for tok in match_tokens):
        return True
    try:
        cmd = " ".join(proc.cmdline()).lower()
        if any(tok in cmd for tok in match_tokens):
            return True
    except psutil.Error:
        pass
    return False


def kill_child_browsers(logger: logging.Logger, timeout: float = 5.0) -> None:
    """
    Kill lingering playwright/browser child processes owned by the current process.

    This is a last-resort cleanup for cases where cancellation or crashes leave
    Playwright/Chrome descendants running.
    """
    try:
        parent = psutil.Process()
    except psutil.Error:
        return

    targets = [p for p in parent.children(recursive=True) if _is_playwright_like(p)]
    if not targets:
        return

    for p in targets:
        try:
            logger.warning(
                "[C4A-MCP | Infra | Crawler Registry] Terminating stray process | data: {pid: %d, name: %s}",
                p.pid,
                p.name(),
            )
            p.terminate()
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as e:
            logger.warning(
                "[C4A-MCP | Infra | Crawler Registry] Failed to terminate process | data: {pid: %s, error: %s}",
                getattr(p, "pid", "unknown"),
                str(e),
            )

    _, alive = psutil.wait_procs(targets, timeout=timeout)
    for p in alive:
        try:
            logger.warning(
                "[C4A-MCP | Infra | Crawler Registry] Killing stubborn process | data: {pid: %d, name: %s}",
                p.pid,
                p.name(),
            )
            p.kill()
        except psutil.Error:
            continue

