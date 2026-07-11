# Contributing to Prometheus

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/OAraLabs/Prometheus.git
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

### Web / WebSocket tests run under system `python3`, not `uv`

The `uv` environment does **not** include `fastapi` or `websockets`. Tests that
import them — the `tests/test_api_*.py` web suite, the WebSocket auth tests, and
the endpoint half of `tests/test_boot_sha_staleness.py` — call
`pytest.importorskip(...)`, so they are **silently skipped** under `uv run
pytest` (reported as skipped, never run). Run them under the system interpreter —
the same one the daemon runs on, which has those packages:

```bash
# web / WS / endpoint tests — system python3, NOT uv
python3 -m pytest tests/test_api_cron.py tests/test_boot_sha_staleness.py -v
```

So a change under `web/` (FastAPI routes) or `gateway/` WS auth is **not** fully
exercised by `uv run pytest` alone — run the relevant suite under `python3` too.

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

If your contribution changes the architecture (new subsystem, new provider, new gateway), open an issue first to discuss. The guide pages under `docs/guide/` (especially the [feature reference](docs/guide/features.md)) are the reference for the current design.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
