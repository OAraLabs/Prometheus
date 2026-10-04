"""D12: the ``computer`` extra pins ``cua-driver`` EXACTLY.

``>=0.28,<1`` admitted 0.28.3 and every release after it, and from 0.28.3
``GetWindowStateInput`` has a REQUIRED keyword (``max_image_dimension``) our
observe call does not pass — so every observation on such an install raises.
Only ``uv.lock`` stood between a fresh install and a driver that can never
observe; ``pip install 'oara-prometheus[computer]'``, the remedy
``computer/cua.py`` itself prints, ignores the lock and resolved 0.33.1.

An upgrade is a deliberate act: bump the pin, re-run the translation tests
against the new wheel (``tests/test_cua_adapter.py``'s real-SDK section) and
the on-box outcome check, in one change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PINNED = "0.28.2"


def test_the_computer_extra_pins_the_driver_exactly():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["optional-dependencies"]["computer"] == [
        f"cua-driver=={PINNED}"
    ]


def test_the_lockfile_agrees_with_the_pin():
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    packages = lock["package"]
    assert [p["version"] for p in packages if p["name"] == "cua-driver"] == [
        PINNED
    ]
    project = next(p for p in packages if p["name"] == "oara-prometheus")
    specifiers = [
        d.get("specifier") for d in project["metadata"]["requires-dist"]
        if d["name"] == "cua-driver"
    ]
    assert specifiers == [f"=={PINNED}"]


def test_an_installed_driver_is_the_pinned_one():
    pytest.importorskip("cua_driver")
    from importlib.metadata import version

    assert version("cua-driver") == PINNED
