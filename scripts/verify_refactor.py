import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from c4a_mcp.config_models import AppConfig
from c4a_mcp.models import RunnerInput
from c4a_mcp.presets.models import AdaptiveStatisticalInput
from c4a_mcp.runner_tool import CrawlRunner
from c4a_mcp.adaptive_runner import AdaptiveCrawlRunner

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("verify_refactor")


async def verify_standard_crawl():
    logger.info("=== Verifying Standard Crawl ===")

    # Load config
    config_path = src_path / "c4a_mcp" / "config" / "defaults.yaml"
    app_config = AppConfig.from_yaml(config_path)
    browser_config = app_config.browser.to_browser_config()

    # Initialize runner
    runner = CrawlRunner(default_crawler_config=app_config.crawler, browser_config=browser_config)

    # Run simple crawl
    inputs = RunnerInput(url="https://example.com", config={"bypass_cache": True})

    try:
        result = await runner.run(inputs)

        if result.error:
            logger.error(f"Standard crawl failed: {result.error}")
            return False

        logger.info(f"Standard crawl successful!")
        logger.info(f"URL: {result.metadata.get('url')}")
        logger.info(f"Status: {result.metadata.get('status')}")
        logger.info(f"Markdown length: {len(result.markdown)}")
        return True

    except Exception as e:
        logger.exception(f"Standard crawl exception: {e}")
        return False


async def verify_adaptive_crawl():
    logger.info("\n=== Verifying Adaptive Crawl ===")

    # Load config
    config_path = src_path / "c4a_mcp" / "config" / "defaults.yaml"
    app_config = AppConfig.from_yaml(config_path)
    browser_config = app_config.browser.to_browser_config()
    browser_config.headless = True  # Ensure headless

    # Initialize runner
    runner = AdaptiveCrawlRunner(
        default_crawler_config=app_config.crawler, browser_config=browser_config
    )

    # Run adaptive crawl
    # Note: Using a simple query on a simple site to avoid long execution times
    inputs = AdaptiveStatisticalInput(
        url="https://example.com", query="example domain", max_pages=1, confidence_threshold=0.5
    )

    try:
        result = await runner.run(inputs)

        if result.error:
            logger.error(f"Adaptive crawl failed: {result.error}")
            return False

        logger.info(f"Adaptive crawl successful!")
        logger.info(f"Confidence: {result.metadata.get('confidence')}")
        logger.info(f"Metrics: {result.metadata.get('metrics')}")
        logger.info(f"Markdown length: {len(result.markdown)}")
        return True

    except Exception as e:
        logger.exception(f"Adaptive crawl exception: {e}")
        return False


async def main():
    success_std = await verify_standard_crawl()
    success_adapt = await verify_adaptive_crawl()

    if success_std and success_adapt:
        logger.info("\n✅ ALL VERIFICATION CHECKS PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
