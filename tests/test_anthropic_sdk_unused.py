"""The `anthropic` extra installed the Anthropic SDK, which nothing in Prometheus imports.

The Anthropic provider speaks HTTP itself (`providers/anthropic.py` imports httpx), so the SDK and the four
packages it alone brought (jiter, docstring-parser, sniffio, and distro) were dead weight in every install that
asked for the extra, including Prometheus.app and the Docker image. The extra's NAME stays, empty, so an
existing `pip install 'oara-prometheus[anthropic]'` or `uv sync --extra anthropic` still works (the `web`
extra is the precedent: kept defined after its packages moved to base).
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_nothing_in_the_package_imports_the_anthropic_sdk():
    pattern = re.compile(r"^\s*(?:import anthropic\b|from anthropic\b)")
    offenders = [
        str(p.relative_to(REPO))
        for p in (REPO / "src" / "prometheus").rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
        if pattern.match(line)
    ]
    assert offenders == []


def test_no_extra_installs_the_anthropic_sdk():
    extras = _pyproject()["project"]["optional-dependencies"]
    carrying = [name for name, reqs in extras.items() if any(re.match(r"anthropic\b", r) for r in reqs)]
    assert carrying == [], f"extras that still install the unused SDK: {carrying}"


def test_distro_is_declared_where_it_is_used():
    """context/environment.py imports `distro` for the Linux OS line. It arrived only through the Anthropic SDK,
    so dropping the SDK silently changed the system prompt on every Linux install (the parity replay's
    compaction token counts moved by one). Anything the code imports for real must be declared."""
    env = (REPO / "src" / "prometheus" / "context" / "environment.py").read_text(encoding="utf-8")
    assert "import distro" in env
    deps = _pyproject()["project"]["dependencies"]
    assert any(re.match(r"distro\b", d) and "linux" in d for d in deps), deps


def test_the_extra_name_still_resolves_for_old_install_commands():
    assert _pyproject()["project"]["optional-dependencies"].get("anthropic") == []


def test_the_app_and_the_image_do_not_ask_for_it():
    spec = importlib.util.spec_from_file_location("build_app", REPO / "packaging" / "macos" / "build_app.py")
    build_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_app)
    assert "anthropic" not in build_app.DEFAULT_EXTRAS
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    default = re.search(r'^ARG PROMETHEUS_EXTRAS="([^"]*)"', dockerfile, re.M).group(1).split()
    assert "anthropic" not in default


def _closure(extras: tuple[str, ...]) -> set[str]:
    """Every package uv.lock would install for oara-prometheus with these extras, following each dependency's
    own extras. Markers are ignored, which over-approximates: stricter, never looser."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"]}
    seen: set[str] = set()
    expanded: set[tuple[str, str]] = set()  # (package, extra) already followed, so a cycle cannot loop
    todo: list[tuple[str, tuple[str, ...]]] = [("oara-prometheus", extras)]
    while todo:
        name, wanted = todo.pop()
        pkg = packages.get(name)
        if pkg is None:
            continue
        deps = [] if name in seen else list(pkg.get("dependencies", []))
        seen.add(name)
        for extra in wanted:
            if (name, extra) not in expanded:
                expanded.add((name, extra))
                deps += pkg.get("optional-dependencies", {}).get(extra, [])
        for dep in deps:
            todo.append((dep["name"], tuple(dep.get("extra", []))))
    return seen


def test_what_the_app_and_the_image_install_holds_no_anthropic_sdk():
    """The developer-only `evals` extra still reaches it (arize-phoenix -> pydantic-ai-slim[anthropic]); what
    ships must not."""
    spec = importlib.util.spec_from_file_location("build_app", REPO / "packaging" / "macos" / "build_app.py")
    build_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_app)
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    image = tuple(re.search(r'^ARG PROMETHEUS_EXTRAS="([^"]*)"', dockerfile, re.M).group(1).split())
    for label, extras in (("base", ()), ("Prometheus.app", tuple(build_app.DEFAULT_EXTRAS)), ("Docker image", image),
                          ("[full]", ("full",))):
        assert "anthropic" not in _closure(extras), f"{label} still installs the Anthropic SDK"
    # The walk can see it at all: the developer-only `evals` extra does reach it.
    assert "anthropic" in _closure(("evals",))
