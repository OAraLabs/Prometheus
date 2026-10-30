#!/usr/bin/env python3
"""Render ``docs/assets/shots/term-startup-banner.svg`` from the real banner.

WHY THIS EXISTS
---------------
The README opens with the banner ``oara daemon`` prints on a terminal (#759). A
hand-made copy would go stale the first time the wordmark, the tagline or a row
changed, as the pairing card once did (see ``render_pairing_banner_svg.py``).
So the card is rendered here FROM ``render_banner``, the function the daemon
prints, in its own colours, and ``tests/test_startup_banner_card.py`` compares
the checked-in card's text against a fresh render.

THE INPUTS ARE FIXED, DELIBERATELY
----------------------------------
The version is ``prometheus.__version__`` from this checkout, so a regenerated
card names the release it ships in. The freshness check ignores the version
number (``same_card``): a version bump alone must not fail CI, while any other
change to what the banner says does. The port is the default, 8005. The model is a neutral example: a reader's never
matches anyone's, and the card shows the row's shape, not a machine.

Usage:
    uv run python scripts/render_startup_banner_svg.py          # write the card
    uv run python scripts/render_startup_banner_svg.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import DEFAULT_TERMINAL_THEME, TerminalTheme
from rich.text import Text

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from prometheus import __version__  # noqa: E402
from prometheus.banner import render_banner  # noqa: E402
from render_pairing_banner_svg import PROMPT_STYLE, svg_lines  # noqa: E402,F401  (svg_lines: shared parser)

OUT = REPO / "docs" / "assets" / "shots" / "term-startup-banner.svg"
PORT = 8005
MODEL = "Qwen3.8-27B"
BACKEND = "llama_cpp"
WIDTH = 48
# The NOX plate the Beacon app and oara.ai use, so the banner's sang and brass
# read as they do on a dark terminal. ANSI colours are Rich's defaults: the
# banner paints with 24-bit colour and dim only.
THEME = TerminalTheme(
    (15, 18, 25),
    (216, 207, 182),
    [tuple(DEFAULT_TERMINAL_THEME.ansi_colors[i]) for i in range(8)],
    [tuple(DEFAULT_TERMINAL_THEME.ansi_colors[i]) for i in range(8, 16)],
)
_VERSION = re.compile(r"\b\d+\.\d+\.\d+\S*")


def render() -> str:
    """The card as SVG: the command, then exactly what the daemon prints, in colour."""
    banner = render_banner(version=__version__, port=PORT, model=MODEL, backend=BACKEND, color=True)
    console = Console(record=True, width=WIDTH, file=io.StringIO(), force_terminal=True, color_system="truecolor")
    console.print("$ oara daemon", style=PROMPT_STYLE, highlight=False)
    console.print(Text.from_ansi(banner.rstrip("\n")), highlight=False)
    return console.export_svg(title="oara daemon", theme=THEME)


def same_card(a: str, b: str) -> bool:
    """Whether two cards say the same thing, version numbers aside."""
    norm = lambda svg: [_VERSION.sub("X.Y.Z", line) for line in svg_lines(svg)]  # noqa: E731
    return norm(a) == norm(b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the card is stale")
    args = parser.parse_args()
    fresh = render()
    if args.check:
        if not same_card(OUT.read_text(encoding="utf-8"), fresh):
            print(f"{OUT.relative_to(REPO)} is stale — run {Path(__file__).relative_to(REPO)}")
            return 1
        print(f"{OUT.relative_to(REPO)} is current")
        return 0
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
