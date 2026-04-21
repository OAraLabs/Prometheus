---
name: modern-python
description: "Configures Python projects with modern tooling (uv, ruff, ty). Use when creating new Python projects, writing standalone scripts with PEP 723, migrating from pip/Poetry/mypy/black, or setting up pyproject.toml configuration."
version: 1.0.0
author: Trail of Bits
license: CC-BY-SA-4.0
---
<!-- Provenance: trailofbits/skills | plugins/modern-python/skills/modern-python/SKILL.md | CC-BY-SA-4.0 -->

# Modern Python

Guide for modern Python tooling and best practices, based on [trailofbits/cookiecutter-python](https://github.com/trailofbits/cookiecutter-python).

## When to Use

- Creating a new Python project or package
- Setting up `pyproject.toml` configuration
- Configuring development tools (linting, formatting, testing)
- Writing Python scripts with external dependencies
- Migrating from legacy tools (when user requests it)

## When NOT to Use

- **User wants to keep legacy tooling**: Respect existing workflows if explicitly requested
- **Python < 3.11 required**: These tools target modern Python
- **Non-Python projects**: Mixed codebases where Python isn't primary

## Anti-Patterns to Avoid

| Avoid | Use Instead |
|-------|-------------|
| `[tool.ty]` python-version | `[tool.ty.environment]` python-version |
| `uv pip install` | `uv add` and `uv sync` |
| Editing pyproject.toml manually to add deps | `uv add <pkg>` / `uv remove <pkg>` |
| `hatchling` build backend | `uv_build` (simpler, sufficient for most cases) |
| Poetry | uv (faster, simpler, better ecosystem integration) |
| requirements.txt | PEP 723 for scripts, pyproject.toml for projects |
| mypy / pyright | ty (faster, from Astral team) |
| `[project.optional-dependencies]` for dev tools | `[dependency-groups]` (PEP 735) |
| Manual virtualenv activation | `uv run <cmd>` |
| pre-commit | prek (faster, no Python runtime needed) |

**Key principles:**
- Always use `uv add` and `uv remove` to manage dependencies
- Never manually activate or manage virtual environments -- use `uv run` for all commands
- Use `[dependency-groups]` for dev/test/docs dependencies, not `[project.optional-dependencies]`

## Tool Overview

| Tool | Purpose | Replaces |
|------|---------|----------|
| **uv** | Package/dependency management | pip, virtualenv, pip-tools, pipx, pyenv |
| **ruff** | Linting AND formatting | flake8, black, isort, pyupgrade, pydocstyle |
| **ty** | Type checking | mypy, pyright (faster alternative) |
| **pytest** | Testing with coverage | unittest |
| **prek** | Pre-commit hooks | pre-commit (faster, Rust-native) |

### Security Tools

| Tool | Purpose | When It Runs |
|------|---------|--------------|
| **shellcheck** | Shell script linting | pre-commit |
| **detect-secrets** | Secret detection | pre-commit |
| **actionlint** | Workflow syntax validation | pre-commit, CI |
| **zizmor** | Workflow security audit | pre-commit, CI |
| **pip-audit** | Dependency vulnerability scanning | CI, manual |
| **Dependabot** | Automated dependency updates | scheduled |

## Decision Tree

```
What are you doing?
|
+-- Single-file script with dependencies?
|   -> Use PEP 723 inline metadata (see below)
|
+-- New multi-file project (not distributed)?
|   -> Minimal uv setup (see Quick Start)
|
+-- New reusable package/library?
|   -> Full project setup (see Full Setup)
|
+-- Migrating existing project?
    -> See Migration Guide
```

## Quick Start: Minimal Project

```bash
# Create project with uv
uv init myproject
cd myproject

# Add dependencies
uv add requests rich

# Add dev dependencies
uv add --group dev pytest ruff ty

# Run code
uv run python src/myproject/main.py

# Run tools
uv run pytest
uv run ruff check .
```

## Full Project Setup

For a bootstrapped template:
```bash
uvx cookiecutter gh:trailofbits/cookiecutter-python
```

### 1. Create Project Structure

```bash
uv init --package myproject
cd myproject
```

### 2. Configure pyproject.toml

```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = [{include-group = "lint"}, {include-group = "test"}, {include-group = "audit"}]
lint = ["ruff", "ty"]
test = ["pytest", "pytest-cov"]
audit = ["pip-audit"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["ALL"]
ignore = ["D", "COM812", "ISC001"]

[tool.pytest]
addopts = ["--cov=myproject", "--cov-fail-under=80"]

[tool.ty.terminal]
error-on-warning = true

[tool.ty.environment]
python-version = "3.11"

[tool.ty.rules]
possibly-unresolved-reference = "error"
unused-ignore-comment = "warn"
```

### 3. Install Dependencies

```bash
uv sync --all-groups
```

### 4. Add Makefile

```makefile
.PHONY: dev lint format test build

dev:
	uv sync --all-groups

lint:
	uv run ruff format --check && uv run ruff check && uv run ty check src/

format:
	uv run ruff format .

test:
	uv run pytest

build:
	uv build
```

## PEP 723: Inline Script Metadata

For standalone scripts with external dependencies:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "rich",
# ]
# ///

import requests
from rich import print

response = requests.get("https://httpbin.org/ip")
print(response.json())
```

Run with: `uv run script.py`

## Migration Guide

### From requirements.txt + pip

**For standalone scripts:** Convert to PEP 723 inline metadata

**For projects:**
```bash
uv init --bare
# Add each dependency
uv add requests rich
uv sync
```

Then delete `requirements.txt`, `requirements-dev.txt`, and old virtual environments. Add `uv.lock` to version control.

### From setup.py / setup.cfg

1. `uv init --bare`
2. `uv add` each dependency from `install_requires`
3. `uv add --group dev` for dev dependencies
4. Copy metadata to `[project]`
5. Delete `setup.py`, `setup.cfg`, `MANIFEST.in`

### From flake8 + black + isort

1. Remove old tools via `uv remove`
2. Delete `.flake8`, `[tool.black]`, `[tool.isort]` configs
3. `uv add --group dev ruff`
4. Run `uv run ruff check --fix .` then `uv run ruff format .`

### From mypy / pyright

1. Remove old tool via `uv remove`
2. Delete `mypy.ini`, `pyrightconfig.json`, or related config sections
3. `uv add --group dev ty`
4. Run `uv run ty check src/`

## uv Command Reference

| Command | Description |
|---------|-------------|
| `uv init` | Create new project |
| `uv init --package` | Create distributable package |
| `uv add <pkg>` | Add dependency |
| `uv add --group dev <pkg>` | Add to dependency group |
| `uv remove <pkg>` | Remove dependency |
| `uv sync` | Install dependencies |
| `uv sync --all-groups` | Install all dependency groups |
| `uv run <cmd>` | Run command in venv |
| `uv run --with <pkg> <cmd>` | Run with temporary dependency |
| `uv build` | Build package |
| `uv publish` | Publish to PyPI |

### Ad-hoc Dependencies with `--with`

```bash
# Run with temporary package
uv run --with requests python -c "import requests; print(requests.get('https://httpbin.org/ip').json())"

# Multiple packages
uv run --with requests --with rich python script.py

# Combine with project deps
uv run --with httpx pytest
```

**When to use `--with` vs `uv add`:**
- `uv add`: Package is a project dependency (goes in pyproject.toml/uv.lock)
- `--with`: One-off usage, testing, or scripts outside a project context

## Dependency Groups

```toml
[dependency-groups]
dev = ["ruff", "ty"]
test = ["pytest", "pytest-cov", "hypothesis"]
docs = ["sphinx", "myst-parser"]
```

Install with: `uv sync --group dev --group test`

## Best Practices Checklist

- [ ] Use `src/` layout for packages
- [ ] Set `requires-python = ">=3.11"`
- [ ] Configure ruff with `select = ["ALL"]` and explicit ignores
- [ ] Use ty for type checking
- [ ] Enforce test coverage minimum (80%+)
- [ ] Use dependency groups instead of extras for dev tools
- [ ] Add `uv.lock` to version control
- [ ] Use PEP 723 for standalone scripts
