"""Locating the shipped config template — installed OR in a checkout.

WHY THIS EXISTS
---------------
``config/prometheus.yaml.default`` is the best documentation of what
Prometheus can be configured to do, and until now it did not ship:
``pyproject.toml`` packaged only ``src/prometheus`` while the template sat at
the repo root, so ``pip install oara-prometheus`` had no copy of it. A
git-checkout install had the file by accident of the checkout; every other
install shape had nothing, and "what can I configure?" was answerable only by
cloning the repo.

The wheel now force-includes it at ``prometheus/config/prometheus.yaml.default``.
This module is the single resolver, so nothing else has to know which of the
two layouts it is running in.

⚠ THE TEST THAT MATTERS IS NOT IN THIS REPO'S TREE.
``tests/test_template_packaging.py`` builds a wheel, installs it into a
scratch venv, and rglobs for the file from the INSTALLED package. Asserting
that the repo has the file proves the checkout, which was never the broken
case (§2d — assert the artefact the consumer receives, not the container it
was put in).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

TEMPLATE_NAME = "prometheus.yaml.default"


class TemplateNotFound(FileNotFoundError):
    """The shipped template is missing from an installed package."""


@lru_cache(maxsize=1)
def get_template_path() -> Path:
    """Absolute path to the shipped config template.

    Looks, in order:

    1. beside this module — where the wheel force-includes it;
    2. ``<repo>/config/`` — a source checkout or an editable install, where
       the file lives at its documented path and is not copied into the
       package tree.

    Raises :class:`TemplateNotFound` rather than returning ``None``. A caller
    that receives ``None`` here writes ``or {}`` and silently proceeds with no
    defaults, which is the failure mode this whole area exists to remove.
    """
    packaged = Path(__file__).resolve().parent / TEMPLATE_NAME
    if packaged.is_file():
        return packaged

    # src/prometheus/config/template.py -> repo root is four parents up.
    repo_root = Path(__file__).resolve().parents[3]
    checkout = repo_root / "config" / TEMPLATE_NAME
    if checkout.is_file():
        return checkout

    raise TemplateNotFound(
        f"{TEMPLATE_NAME} not found beside {packaged.parent} nor at "
        f"{checkout}. The wheel force-includes it via "
        f"[tool.hatch.build.targets.wheel.force-include]; if that stanza was "
        f"removed, installed packages lose the template silently."
    )


def read_template_text() -> str:
    """The template's raw text, comments included.

    Comments are most of its value — the keys alone are a schema, and the
    schema was never the part that was hard to find.
    """
    return get_template_path().read_text(encoding="utf-8")


def load_template() -> dict:
    """The template parsed to a dict."""
    import yaml

    data = yaml.safe_load(read_template_text())
    return data if isinstance(data, dict) else {}
