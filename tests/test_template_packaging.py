"""The config template ships — asserted against an INSTALLED package.

WHY IT IS SHAPED THIS WAY
-------------------------
The template did not ship for the entire life of the project.
``pyproject.toml`` packaged ``src/prometheus`` and the template lived at the
repo root, so a ``pip install`` had no copy of it, ever. A git checkout had it
by accident of the checkout — which is exactly why nothing noticed.

That means a test asserting the REPO has the file would have passed
throughout the defect. The only assertion worth making is against the artefact
a user receives: build a wheel, install it into a scratch venv, and look for
the file from the installed package's own root (§2d — assert what the consumer
gets, not the container it was put in).

That is slow-ish (a few seconds) and it is the whole point of the test. The
cheap in-repo checks below exist too, and are explicitly the weaker half.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _clean_env() -> dict:
    """Environment for probing the INSTALLED package.

    ⚠ PYTHONPATH MUST BE STRIPPED. The dev loop runs pytest with
    `PYTHONPATH=$PWD/src`, and a subprocess inherits it — so the probe
    imported the WORKTREE's prometheus, rglob'd the source tree, and reported
    on a package that was never installed. Both assertions here failed against
    the wrong artefact before this existed, which is the same class of mistake
    the test is written to catch: reading the container you have rather than
    the one the consumer gets.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def test_resolver_finds_the_template_in_this_checkout():
    """The cheap half. Proves the checkout, which was never the broken case."""
    from prometheus.config.template import get_template_path, load_template

    p = get_template_path()
    assert p.is_file(), p
    assert load_template(), "template parsed to an empty mapping"


def test_pyproject_force_includes_the_template():
    """Config-level pin, so a removed stanza fails fast and by name.

    Not sufficient on its own — this asserts the configuration, and the
    configuration was never what shipped. The installed-package test below is
    the one that proves delivery.
    """
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    fi = (data["tool"]["hatch"]["build"]["targets"]["wheel"]
          .get("force-include", {}))
    assert "config/prometheus.yaml.default" in fi, (
        "the wheel no longer force-includes the config template. Installed "
        "packages lose it silently — a pip user's answer to 'what can I "
        "configure?' goes back to 'clone the repo'."
    )
    assert fi["config/prometheus.yaml.default"].endswith(
        "prometheus/config/prometheus.yaml.default"), (
        "the template must land inside the package, where "
        "config.template.get_template_path() looks first"
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_template_is_present_in_an_INSTALLED_package(tmp_path: Path):
    """Build a wheel, install it, and find the template from the install.

    This is the assertion the defect required. Everything else in this file
    would have been green for the whole time the template did not ship.
    """
    dist = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(dist)],
        cwd=REPO, capture_output=True, text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable here: {build.stderr.strip()[:200]}")

    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"

    venv = tmp_path / "venv"
    for cmd in (["uv", "venv", str(venv), "--python", f"{sys.version_info.major}."
                 f"{sys.version_info.minor}"],
                ["uv", "pip", "install", "--python",
                 str(venv / "bin" / "python"), str(wheels[0])]):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"scratch venv unavailable: {proc.stderr.strip()[:200]}")

    # Ask the INSTALLED interpreter, from the installed package's own root.
    probe = (
        "import pathlib, prometheus;"
        "root = pathlib.Path(prometheus.__file__).parent;"
        "hits = sorted(str(p.relative_to(root)) "
        "for p in root.rglob('prometheus.yaml.default'));"
        "print('|'.join(hits))"
    )
    out = subprocess.run([str(venv / "bin" / "python"), "-c", probe],
                         capture_output=True, text=True, env=_clean_env(),
                         cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    hits = [h for h in out.stdout.strip().split("|") if h]
    assert hits, (
        "prometheus.yaml.default is NOT in the installed package. This is the "
        "original defect: the template lived at the repo root, outside "
        "`packages = [\"src/prometheus\"]`, so every pip install shipped "
        "without it while every git checkout had it by accident."
    )
    assert "config/prometheus.yaml.default" in hits, (
        f"template installed at an unexpected path: {hits}. "
        f"config.template.get_template_path() looks beside itself first."
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_installed_resolver_returns_the_PACKAGED_copy(tmp_path: Path):
    """The resolver must resolve to the packaged file when installed.

    Finding the file in the wheel and having the resolver find it are two
    claims. The second is the one every caller depends on, and a resolver that
    silently falls through to a repo-relative path would look fine on a
    developer's machine and raise on a user's.
    """
    dist = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(dist)],
        cwd=REPO, capture_output=True, text=True,
    )
    if build.returncode != 0:
        pytest.skip("wheel build unavailable here")
    wheels = list(dist.glob("*.whl"))
    venv = tmp_path / "venv"
    for cmd in (["uv", "venv", str(venv), "--python", f"{sys.version_info.major}."
                 f"{sys.version_info.minor}"],
                ["uv", "pip", "install", "--python",
                 str(venv / "bin" / "python"), str(wheels[0])]):
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            pytest.skip("scratch venv unavailable")

    probe = (
        "from prometheus.config.template import get_template_path, load_template;"
        "p = get_template_path();"
        "print(str(p));"
        "print(len(load_template()))"
    )
    out = subprocess.run([str(venv / "bin" / "python"), "-c", probe],
                         capture_output=True, text=True, env=_clean_env(),
                         cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    path_line, sections = out.stdout.strip().splitlines()
    assert "site-packages" in path_line, (
        f"the installed resolver returned {path_line!r} — not the packaged "
        f"copy. It fell through to a checkout path that will not exist on a "
        f"user's machine."
    )
    assert int(sections) > 0, "installed template parsed to an empty mapping"
