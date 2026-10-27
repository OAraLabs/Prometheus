"""Characters that hide text from a person reading it, while a model still reads it.

Skills and MEMORY.md / USER.md go into the agent's prompt, and a person reviewing one sees only what renders. These
characters let a write carry text nobody reviewing it would notice: Unicode tag characters (each one an invisible
copy of an ASCII letter), the bidi embedding, override and isolate controls (which reorder what is shown), zero-width
spaces and joiners used as separators, the invisible math operators, a byte-order mark inside the text, and the
supplementary variation selectors (which can carry hidden bytes after a visible character).

Deliberately NOT here, because real writing needs them: ZWJ (U+200D, emoji sequences and Indic conjuncts), ZWNJ
(U+200C, Persian and other scripts), the LRM/RLM direction marks (U+200E/U+200F), the ordinary variation selectors
(U+FE00..U+FE0F, the emoji presentation selector among them), and a byte-order mark as the very first character.

Used by ``DangerousCodeScanner.scan_markdown_content`` (every skill writer) and the memory tool.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

#: Inclusive code-point ranges that hide text. See the module docstring for what is left out, and why.
_HIDING_RANGES: tuple[tuple[int, int], ...] = (
    (0x200B, 0x200B),     # ZERO WIDTH SPACE
    (0x202A, 0x202E),     # LRE, RLE, PDF, LRO, RLO: bidi embeddings and overrides
    (0x2060, 0x2064),     # WORD JOINER and the invisible operators
    (0x2066, 0x2069),     # LRI, RLI, FSI, PDI: bidi isolates
    (0xFEFF, 0xFEFF),     # ZERO WIDTH NO-BREAK SPACE (a byte-order mark anywhere but the start)
    (0xE0000, 0xE007F),   # TAG characters: invisible copies of ASCII
    (0xE0100, 0xE01EF),   # VARIATION SELECTORS SUPPLEMENT
)


@dataclass(frozen=True)
class InvisibleChar:
    index: int        # offset in the text
    line: int         # 1-based line number
    codepoint: int
    name: str         # the Unicode name, or U+XXXX

    def describe(self) -> str:
        return f"U+{self.codepoint:04X} {self.name} (line {self.line})"


def _hides(codepoint: int) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in _HIDING_RANGES)


def find_invisible(text: str) -> list[InvisibleChar]:
    """Every hiding character in *text*, in order. A byte-order mark as the first character is allowed."""
    found: list[InvisibleChar] = []
    line = 1
    for index, char in enumerate(text or ""):
        if char == "\n":
            line += 1
            continue
        codepoint = ord(char)
        if not _hides(codepoint) or (index == 0 and codepoint == 0xFEFF):
            continue
        found.append(InvisibleChar(index, line, codepoint, unicodedata.name(char, f"U+{codepoint:04X}")))
    return found


def describe_invisible(found: list[InvisibleChar], limit: int = 3) -> str:
    """A short, plain list for a refusal message: the first few, and how many in all."""
    shown = ", ".join(f.describe() for f in found[:limit])
    more = f", and {len(found) - limit} more" if len(found) > limit else ""
    return f"{len(found)} invisible character(s): {shown}{more}"
