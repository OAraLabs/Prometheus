"""The startup banner card on the README must say what the code prints.

``docs/assets/shots/term-startup-banner.svg`` opens the README (and so the PyPI
page). ``scripts/render_startup_banner_svg.py`` renders it from
``prometheus.banner.render_banner``, the function ``oara daemon`` prints; this
compares the checked-in card's TEXT against a fresh render, with version
numbers aside, so a release bump alone does not fail it but any other change to
the banner does, until the card is regenerated. Same pattern as
tests/test_pairing_banner_card.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import render_startup_banner_svg as card  # noqa: E402

from prometheus.banner import _TAGLINE  # noqa: E402


def test_card_text_matches_what_the_banner_prints():
    committed = card.OUT.read_text(encoding="utf-8")
    assert card.same_card(committed, card.render()), (
        "term-startup-banner.svg is stale — regenerate with "
        "`uv run python scripts/render_startup_banner_svg.py`"
    )


def test_the_line_parser_reads_the_card():
    """Guard the guard: an empty parse would make the comparison vacuous."""
    # Rich drops leading and between-style spaces from the SVG text, so compare stripped
    # lines and words, not columns.
    lines = [line.strip() for line in card.svg_lines(card.OUT.read_text(encoding="utf-8"))]
    assert lines[0] == "$ oara daemon"
    assert _TAGLINE in lines
    assert any("daemon" in line and ":8005" in line for line in lines)
    assert any(line.startswith("OAra Labs · prometheus") for line in lines)


def test_a_version_bump_alone_does_not_make_the_card_stale():
    fresh = card.render()
    bumped = fresh.replace(card.__version__, "9.9.99")
    assert bumped != fresh
    assert card.same_card(fresh, bumped)


def test_any_other_change_does():
    fresh = card.render()
    changed = fresh.replace("oara.ai/beacon", "oara.ai/beacom")
    assert changed != fresh
    assert not card.same_card(fresh, changed)
