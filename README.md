# c4a-mcp

Model Context Protocol (MCP) server for web interaction using `crawl4ai`.

This project aims to provide AI agents with advanced web browsing and data extraction capabilities via the MCP `runner` tool.

## Quick Install (Cursor)

Install this MCP server in Cursor with one click:

<a href="cursor://anysphere.cursor-deeplink/mcp/install?name=c4a-mcp&config=eyJjb21tYW5kIjogImRvY2tlciIsICJhcmdzIjogWyJydW4iLCAiLWkiLCAiLS1ybSIsICJnaGNyLmlvL2JsZ2h0ci9jNGEtbWNwOmxhdGVzdCJdfQ=="><img src="https://cursor.com/deeplink/mcp-install-dark.png" alt="Add c4a-mcp MCP server to Cursor" style="max-height: 32px;" /></a>

**Note:** Requires Docker to be installed and running. The server will run in a container from `ghcr.io/blghtr/c4a-mcp:latest`.

## MCP Server Installation

### Option 1: One-Click Install (Cursor)

Click the button above to install automatically in Cursor, or use the deeplink:

```
cursor://anysphere.cursor-deeplink/mcp/install?name=c4a-mcp&config=eyJjb21tYW5kIjogImRvY2tlciIsICJhcmdzIjogWyJydW4iLCAiLWkiLCAiLS1ybSIsICJnaGNyLmlvL2JsZ2h0ci9jNGEtbWNwOmxhdGVzdCJdfQ==
```

### Option 2: Manual Installation

Add to your `mcp.json` file (typically located at `~/.cursor/mcp.json` or `%APPDATA%\Cursor\User\mcp.json` on Windows):

```json
{
  "mcpServers": {
    "c4a-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "ghcr.io/blghtr/c4a-mcp:latest"
      ]
    }
  }
}
```

**With environment variables** (for LLM-based extraction):

```json
{
  "mcpServers": {
    "c4a-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e", "OPENAI_API_KEY",
        "-e", "GEMINI_API_KEY",
        "ghcr.io/blghtr/c4a-mcp:latest"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

**Requirements:**
- Docker must be installed and running
- For private repositories, authenticate with GitHub Container Registry:
  ```bash
  echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
  ```

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management

### Installation

```bash
# Install dependencies
uv pip install --system -e ".[dev]"
```

### Pre-commit Hooks

This project uses pre-commit hooks to automatically format and lint code before commits.

**Initial Setup:**

```bash
# Install pre-commit hooks
uv run pre-commit install
```

**Usage:**

Pre-commit hooks will run automatically on `git commit`. They will:
- Format code with `black` and `ruff format`
- Fix linting issues with `ruff`
- Check YAML/JSON files for syntax errors
- Remove trailing whitespace and fix end-of-file issues

**Manual Run:**

```bash
# Run hooks on all files
uv run pre-commit run --all-files

# Run hooks on staged files only
uv run pre-commit run
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v
```

## CI/CD

This project uses GitHub Actions for continuous integration and deployment.

### Workflow

The CI/CD pipeline (`/.github/workflows/ci-cd.yml`) performs the following:

1. **Testing**: Runs tests on Python 3.11 and 3.12
2. **Docker Build**: Builds Docker image on push to `main` or tag creation
3. **Docker Push**: Publishes image to GitHub Container Registry (ghcr.io)

### Docker Image

Docker images are automatically built and pushed to:
```
ghcr.io/blghtr/c4a-mcp
```

**Available Tags:**
- `latest` - Latest commit on `main` branch
- `v<version>` - Semantic version tags (e.g., `v0.1.0`)

**Usage:**

```bash
# Pull the latest image
docker pull ghcr.io/blghtr/c4a-mcp:latest

# Run the container
docker run ghcr.io/blghtr/c4a-mcp:latest
```

**Note:** For private repositories, you'll need to authenticate:

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull the image
docker pull ghcr.io/blghtr/c4a-mcp:latest
```

### Local Docker Build

To build the Docker image locally:

```bash
# Build the image
docker build -t c4a-mcp:local .

# Run the container
docker run c4a-mcp:local
```

## Environment Variables

The MCP server itself does not require any environment variables. However, if you plan to use LLM-based extraction strategies with crawl4ai, you may need to set API keys for your chosen provider:

### Optional LLM Provider API Keys

- `OPENAI_API_KEY` - For OpenAI models (gpt-4o, gpt-4o-mini, o1-mini, etc.)
- `ANTHROPIC_API_KEY` - For Anthropic models (claude-3-5-sonnet, etc.)
- `GEMINI_API_KEY` - For Google Gemini models
- `GROQ_API_KEY` - For Groq models
- `DEEPSEEK_API_KEY` - For DeepSeek models

These are only needed if you use LLM-based extraction strategies in your crawl configuration. The server will work fine without them for standard crawling.

### Using .env File

The project uses `python-dotenv`, so you can create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## Troubleshooting

### Playwright Browser Installation Failures

**Issue:** `playwright install` fails or browsers are not found.

**Solutions:**
1. **Local Development:**
   ```bash
   # Run crawl4ai setup command
   uv run crawl4ai-setup
   
   # Or manually install browsers
   uv run playwright install chromium
   ```

2. **Docker:**
   - Ensure the Dockerfile includes all required system libraries (see Dockerfile for full list)
   - Verify Playwright installation step runs: `RUN playwright install --with-deps chromium`

3. **Check Installation:**
   ```bash
   uv run crawl4ai-doctor
   ```

### MCP Connection Issues

**Issue:** Cannot connect to MCP server or tools not available.

**Solutions:**
1. **Verify Server is Running:**
   ```bash
   # Start the server
   uv run c4a-mcp
   ```

2. **Check MCP Client Configuration:**
   - Ensure the server command points to: `c4a-mcp` or `python -m c4a_mcp`
   - Verify transport method (stdio, SSE, etc.) matches your client

3. **Check Logs:**
   - Enable debug logging to see detailed error messages
   - Look for connection errors in the server logs

### Docker Build Failures

**Issue:** Docker build fails with dependency or permission errors.

**Solutions:**
1. **Clear Build Cache:**
   ```bash
   docker build --no-cache -t c4a-mcp:local .
   ```

2. **Check System Dependencies:**
   - Ensure all Playwright system libraries are included in Dockerfile
   - Verify Python version matches (3.11+)

3. **Permission Issues:**
   - The Dockerfile now runs as non-root user (appuser)
   - If you need to modify files, ensure proper ownership

4. **Network Issues:**
   - Check if you can reach PyPI and GitHub Container Registry
   - Consider using build-time network settings if behind a proxy

### Test Failures

**Issue:** Tests fail in CI/CD or locally.

**Solutions:**
1. **Install Dev Dependencies:**
   ```bash
   uv pip install --system -e ".[dev]"
   ```

2. **Run Tests with Verbose Output:**
   ```bash
   uv run pytest -v
   ```

3. **Check Python Version:**
   - Ensure Python 3.11+ is installed
   - CI/CD tests on 3.11 and 3.12

### Pre-commit Hook Failures

**Issue:** Pre-commit hooks fail or skip.

**Solutions:**
1. **Update Hooks:**
   ```bash
   uv run pre-commit autoupdate
   ```

2. **Run Manually:**
   ```bash
   uv run pre-commit run --all-files
   ```

3. **Skip Hooks (not recommended):**
   ```bash
   git commit --no-verify
   ```
