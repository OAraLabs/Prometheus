#!/usr/bin/env python3
"""Render ``docs/assets/shots/term-pairing-banner.svg`` from the real banner.

WHY THIS EXISTS
---------------
The README and the install guide show a terminal card of what ``oara daemon``
prints in setup mode. It was rendered once by hand and then went stale twice
over while the code moved on: it still said ``prometheus daemon`` after the
``oara`` rename, and it showed the private beacon-desktop repo as the place
to get Beacon — a GitHub 404 for every reader, on the PyPI page too.

So the card is rendered here FROM ``format_pairing_banner`` — the function
the daemon itself prints — and ``tests/test_pairing_banner_card.py`` compares
the checked-in card's text against a fresh render, so it cannot drift again
without CI saying so.

TWO INPUTS ARE FIXED, DELIBERATELY
----------------------------------
The same generic values the doc cards have used since #130: the hostname is
``your-host`` (a reader's never matched ours, and ours is not for
publishing), and the pairing code is ``528491`` (the real one is random per
boot). Everything else on the card is the function's own output.

Usage:
    uv run python scripts/render_pairing_banner_svg.py          # write the card
    uv run python scripts/render_pairing_banner_svg.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import html
import io
import re
import socket
import sys
from pathlib import Path
from unittest import mock

from rich.console import Console
from rich.text import Text

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from prometheus.web.setup_server import format_pairing_banner  # noqa: E402

OUT = REPO / "docs" / "assets" / "shots" / "term-pairing-banner.svg"
HOST = "your-host"
CODE = "528491"
PORT = 8005
# The doc cards' shared look (term-doctor.svg): 74 columns, a bold green
# prompt, the one value that matters in bold amber.
WIDTH = 74
PROMPT_STYLE = "bold #00823d"
CODE_STYLE = "bold #d08442"


def render() -> str:
    """The card as SVG: the command, then exactly what the daemon prints."""
    with mock.patch.object(socket, "gethostname", return_value=HOST):
        banner = format_pairing_banner(CODE, PORT)
    console = Console(record=True, width=WIDTH, file=io.StringIO(), force_terminal=True)
    console.print("$ oara daemon", style=PROMPT_STYLE, highlight=False)
    body = Text(banner)
    body.highlight_words([f"    {CODE}"], style=CODE_STYLE)
    # The daemon calls print(banner): the banner, then print's own newline.
    console.print(body, highlight=False)
    return console.export_svg(title="oara — pairing-banner")


def svg_lines(svg: str) -> list[str]:
    """The card's visible text, one string per terminal line.

    Rich draws each line as one or more ``<text>`` runs clipped to
    ``…-line-N``; joining the runs per N gives back the line. Comparing
    text rather than bytes keeps a Rich upgrade that only changes markup
    from failing the freshness test.
    """
    runs = re.findall(r'<text [^>]*clip-path="url\(#[\w-]+-line-(\d+)\)"[^>]*>([^<]*)</text>', svg)
    lines: dict[int, str] = {}
    for n, text in runs:
        lines[int(n)] = lines.get(int(n), "") + html.unescape(text).replace("\xa0", " ")
    return [lines[n].rstrip() for n in sorted(lines)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the card is stale")
    args = parser.parse_args()
    fresh = render()
    if args.check:
        if svg_lines(OUT.read_text(encoding="utf-8")) != svg_lines(fresh):
            print(f"{OUT.relative_to(REPO)} is stale — run {Path(__file__).relative_to(REPO)}")
            return 1
        print(f"{OUT.relative_to(REPO)} is current")
        return 0
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
