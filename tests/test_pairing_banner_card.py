"""The setup-mode banner card in the docs must say what the code prints.

``docs/assets/shots/term-pairing-banner.svg`` is on the README, the install
guide and the PyPI page. Rendered once by hand, it drifted on two counts
while its tests stayed green: it kept saying ``prometheus daemon`` after the
``oara`` rename, and it sent readers to the private beacon-desktop repo — a
GitHub 404 — after the code itself had been corrected.

``scripts/render_pairing_banner_svg.py`` renders the card from
``format_pairing_banner``; this compares the checked-in card's TEXT against a
fresh render. Text, not bytes, so a Rich upgrade that only changes markup
does not fail it — but any change to what the banner says does, until the
card is regenerated.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import render_pairing_banner_svg as card  # noqa: E402


def test_card_text_matches_what_the_banner_prints():
    committed = card.svg_lines(card.OUT.read_text(encoding="utf-8"))
    fresh = card.svg_lines(card.render())
    assert committed == fresh, (
        "term-pairing-banner.svg is stale — regenerate with "
        "`uv run python scripts/render_pairing_banner_svg.py`"
    )


def test_the_line_parser_reads_the_card():
    """Guard the guard: an empty parse would make the comparison vacuous."""
    lines = card.svg_lines(card.OUT.read_text(encoding="utf-8"))
    assert lines[0] == "$ oara daemon"
    assert "  Don't have Beacon yet?  https://oara.ai/beacon" in lines
    assert f"    {card.CODE}" in lines
