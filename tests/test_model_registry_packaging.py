"""The model registry ships, and an INSTALLED package uses it (WP-X.25).

WHAT WAS WRONG
--------------
``config/model_registry.yaml`` decides the adapter tier for a local model:
listed with native tool calling -> "light", otherwise "full". It lived at the
repo root, outside ``packages = ["src/prometheus"]``, and
``__main__._has_native_tool_calling`` looked only at ``<repo>/config/``. On
every pip and Homebrew install that path does not exist, the check returned
False without a word, and every local model ran at tier "full". Every git
checkout had the file, which is why nothing noticed.

WHY THE TESTS ARE SHAPED THIS WAY
---------------------------------
A test that the REPO has the file passed for the whole life of the defect.
So the tests that matter build the artefacts a user receives and look from the
far side:

* a wheel, installed into a scratch venv, asked for a listed model's tier;
* an sdist, from which a wheel is then built. That is what Homebrew installs,
  and the sdist already carried the file under /config while the wheel built
  from it did not.

A packaging check that could not run must not read as passed: in CI (``CI``
set) an unavailable build FAILS instead of skipping.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LISTED_LOCAL_MODEL = "gemma-4-26b-a4b-it-Q4_K_M"   # registry key gemma-4: native tool calling
UNLISTED_LOCAL_MODEL = "some-unlisted-local-model"


def _clean_env() -> dict:
    """Probe the INSTALLED package: the dev loop's PYTHONPATH=$PWD/src would
    otherwise import the worktree's source and prove nothing."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def _unavailable(what: str, detail: str):
    if os.environ.get("CI"):
        pytest.fail(f"{what} unavailable in CI, so the packaging check did not "
                    f"run: {detail[:300]}")
    pytest.skip(f"{what} unavailable here: {detail[:200]}")


# What `uv build --offline` prints when the backend is simply not cached. Any
# OTHER failure is the build itself failing, which is what these tests exist
# to catch: a wheel built from an sdist that lost the registry fails with
# "Forced include not found", and that must never read as "unavailable".
_UNAVAILABLE_MARKERS = ("network was disabled", "network connectivity is disabled")


def _uv_build(kind: str, out: Path, *source: str) -> Path:
    if shutil.which("uv") is None:
        _unavailable("uv", "not on PATH")
    proc = subprocess.run(
        ["uv", "build", f"--{kind}", "--offline", "--out-dir", str(out), *source],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if any(m in err.lower() for m in _UNAVAILABLE_MARKERS):
            _unavailable(f"{kind} build", err)
        pytest.fail(f"the {kind} build FAILED (not unavailable):\n{err[-2500:]}")
    pattern = "*.whl" if kind == "wheel" else "*.tar.gz"
    built = list(out.glob(pattern))
    assert len(built) == 1, f"expected one {kind}, got {built}"
    return built[0]


# ── the cheap half: config and the checkout ─────────────────────────────────

def test_pyproject_force_includes_the_model_registry():
    """Config-level pin, so a removed stanza fails fast and by name."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    fi = (data["tool"]["hatch"]["build"]["targets"]["wheel"]
          .get("force-include", {}))
    assert fi.get("config/model_registry.yaml") == "prometheus/config/model_registry.yaml", (
        "the wheel no longer force-includes config/model_registry.yaml. "
        "Installed packages then run every local model at adapter tier 'full'."
    )


def test_resolver_finds_the_registry_in_this_checkout():
    from prometheus.config.model_registry import get_model_registry_path

    path = get_model_registry_path()
    assert path.is_file(), path
    assert "gemma-4" in path.read_text(encoding="utf-8")


def test_a_missing_registry_is_a_warning_not_a_silent_full(monkeypatch, caplog):
    """The defect hid because the fallback was silent. A missing registry
    still means no native tool calling (tier 'full'), but it says so."""
    from prometheus import __main__ as main_mod
    from prometheus.config import model_registry

    def _missing():
        raise model_registry.ModelRegistryNotFound("model_registry.yaml not found")

    monkeypatch.setattr(model_registry, "get_model_registry_path", _missing)
    caplog.set_level(logging.WARNING)

    tier = main_mod._get_adapter_tier("llama_cpp", LISTED_LOCAL_MODEL)

    assert tier == "full"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("model registry: NOT FOUND" in m and "'full'" in m for m in warnings), warnings


# ── the far side: artefacts a user installs ─────────────────────────────────

@pytest.fixture(scope="module")
def installed_wheel_python(tmp_path_factory) -> Path:
    """A scratch venv with ONLY the built wheel (and its dependencies)."""
    tmp = tmp_path_factory.mktemp("registry-wheel")
    wheel = _uv_build("wheel", tmp / "dist")
    venv = tmp / "venv"
    for cmd in (["uv", "venv", str(venv), "--python", sys.executable],
                ["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
                 str(wheel)]):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            _unavailable("scratch venv", proc.stderr.strip())
    return venv / "bin" / "python"


def test_an_installed_wheel_resolves_a_listed_local_model_to_light(
        installed_wheel_python, tmp_path):
    """THE assertion the defect needed: from an install, a listed local model
    gets tier 'light', an unlisted one 'full', and the registry used is the
    packaged copy, not a checkout path."""
    # The tier first: it is the behavior, and it is what a build without the
    # registry gets wrong. The resolver path second, and optional in the probe
    # so a package without the resolver still reports its tiers.
    probe = (
        "from prometheus.__main__ import _get_adapter_tier;"
        f"print(_get_adapter_tier('llama_cpp', {LISTED_LOCAL_MODEL!r}));"
        f"print(_get_adapter_tier('llama_cpp', {UNLISTED_LOCAL_MODEL!r}));"
        "import importlib.util as u;"
        "m = u.find_spec('prometheus.config.model_registry');"
        "print(__import__('prometheus.config.model_registry', fromlist=['x'])"
        ".get_model_registry_path() if m else 'NO-RESOLVER')"
    )
    out = subprocess.run([str(installed_wheel_python), "-c", probe],
                         capture_output=True, text=True, env=_clean_env(),
                         cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    listed, unlisted, path = out.stdout.strip().splitlines()[-3:]
    assert listed == "light", (
        f"an installed package gave {LISTED_LOCAL_MODEL!r} tier {listed!r}: the "
        f"registry is not shipped or not found, so every local model runs 'full'")
    assert unlisted == "full"
    assert "site-packages" in path, (
        f"the installed resolver returned {path!r}, not the packaged copy")


def test_a_wheel_built_from_the_sdist_ships_the_registry(tmp_path):
    """Homebrew installs from the sdist, so pip builds a wheel from it. The
    sdist carried the file under /config all along; the wheel built from it
    did not, and that wheel is what lands on disk."""
    sdist = _uv_build("sdist", tmp_path / "sdist")
    wheel = _uv_build("wheel", tmp_path / "from-sdist", str(sdist))
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    assert "prometheus/config/model_registry.yaml" in names, (
        "a wheel built from the sdist (Homebrew's path) has no "
        "prometheus/config/model_registry.yaml, so a Homebrew install runs "
        "every local model at adapter tier 'full'")


def test_an_sdist_that_lost_the_registry_fails_the_check_not_skips_it(tmp_path):
    """The regression this file guards, made on purpose: an sdist without
    config/model_registry.yaml. Building Homebrew's wheel from it must FAIL
    the check, loudly and with the build's own error, never skip."""
    import tarfile

    sdist = _uv_build("sdist", tmp_path / "sdist")
    unpacked = tmp_path / "unpacked"
    with tarfile.open(sdist) as tf:
        tf.extractall(unpacked, filter="data")
    [root] = list(unpacked.iterdir())
    (root / "config" / "model_registry.yaml").unlink()
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    broken = broken_dir / sdist.name
    with tarfile.open(broken, "w:gz") as tf:
        tf.add(root, arcname=root.name)

    with pytest.raises(pytest.fail.Exception, match="Forced include not found"):
        _uv_build("wheel", tmp_path / "from-broken", str(broken))
