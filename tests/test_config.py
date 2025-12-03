"""Quick smoke test for config system."""

from pathlib import Path

from c4a_mcp.config_models import AppConfig, CrawlerConfigYAML

# Test 1: Load defaults.yaml
print("Test 1: Loading defaults.yaml...")
# Path relative to project root, not test directory
config_path = Path(__file__).parent.parent / "src" / "c4a_mcp" / "config" / "defaults.yaml"
app_config = AppConfig.from_yaml(config_path)
print(f"✓ Loaded: browser={app_config.browser.browser_type}, timeout={app_config.crawler.timeout}s")

# Test 2: Merge configs
print("\nTest 2: Testing config merge...")
override = CrawlerConfigYAML(timeout=120, bypass_cache=False)
merged = app_config.crawler.merge(override)
print(f"✓ Merged: timeout={merged.timeout}s (was 60s), bypass_cache={merged.bypass_cache}")

# Test 3: Convert to CrawlerRunConfig kwargs
print("\nTest 3: Converting to CrawlerRunConfig kwargs...")
kwargs = merged.to_crawler_run_config_kwargs()
print(f"✓ Kwargs: {kwargs}")

print("\n✅ All tests passed!")
