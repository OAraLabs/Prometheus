"""D12: the ``computer`` extra pins ``cua-driver`` EXACTLY.

A release on the same minor line can change the SDK's types: 0.28.3 made
``GetWindowStateInput`` take a REQUIRED ``max_image_dimension`` and gave
``ActionResult`` ``summary`` and ``error``, and an adapter written for 0.28.2
raised on every observation there. ``pip install 'oara-prometheus[computer]'``,
the remedy ``computer/cua.py`` itself prints, ignores the lock, so only the
exact pin keeps a fresh install on the version the adapter was written for.
The pin moved 0.28.2 -> 0.28.3 on 2026-11-13 (0.28.2 left PyPI), with the
adapter adapted.

An upgrade is a deliberate act: bump the pin, re-run the translation tests
against the new wheel (``tests/test_cua_adapter.py``'s real-SDK section) and
the on-box outcome check, in one change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PINNED = "0.28.3"


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
