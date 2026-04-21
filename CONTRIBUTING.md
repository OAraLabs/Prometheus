# Contributing to Prometheus

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/whieber1/Prometheus-.git
cd Prometheus-
pip install -e ".[dev]"
git config core.hooksPath .githooks   # enable pre-commit secret scanning
uv run pytest tests/ -v               # make sure everything passes
```

> **Note:** Git does not auto-enable hooks from cloned repos (security policy).
> The `git config core.hooksPath .githooks` step activates the pre-commit hook
> that blocks accidental commits of secrets, private IPs, and infrastructure data.

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/test_adapter.py -v

# With coverage
uv run pytest tests/ --cov=prometheus
```

All tests must pass before submitting a PR.

## Code Style

- Python 3.11+ with type hints
- Pydantic for data validation where appropriate
- Every file extracted from a donor project must include a provenance header:

```python
# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/base.py
# License: MIT
# Modified: renamed imports from openharness → prometheus
```

- New code doesn't need provenance headers, just standard docstrings

## Project Structure

- `src/prometheus/` — all production code
- `tests/` — all test files (mirror the source structure)
- `config/` — configuration files
- `scripts/` — daemon, health check, systemd service
- `benchmarks/` — benchmark test suite

## What to Work On

Check the GitHub issues or the roadmap in README.md. Good first contributions:

- Adding a new gateway adapter (follow the pattern in `gateway/telegram.py`)
- Adding a new builtin tool (follow the pattern in `tools/builtin/`)
- Improving test coverage (especially integration tests)
- Documentation improvements

## Pull Request Process

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`uv run pytest tests/ -v`)
5. Commit with a clear message
6. Push and open a PR

## Architecture Decisions

If your contribution changes the architecture (new subsystem, new provider, new gateway), open an issue first to discuss. The architecture doc (`sovereign-harness-architecture.md`) is the reference for design decisions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
