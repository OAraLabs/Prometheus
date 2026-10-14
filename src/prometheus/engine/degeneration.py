"""Degenerate-output detection (#772) — pure, no I/O.

On 2026-10-13 ``qwen3.8-max`` (DashScope) broke down twice at ~241k tokens of
context. Both rounds were saved as ordinary assistant messages; one was read
back as history by the next turn, which then broke down too:

* **240,772 tokens** — visible text of digits and brackets carrying 28 C0
  control bytes (``\\x00``, ``\\x01``, ``\\x02``, ``\\x05``); the thinking block
  was just ``"Button: \\n"``.
* **244,459 tokens** — 6,416 output tokens of ``"0 00 000 …"`` in the THINKING
  block, while the visible text was the single word ``Report``.

Two tests, each applied to visible text AND thinking separately:

``control_bytes``
    Any C0 control character other than ``\\t``, ``\\n``, ``\\r``. Model prose
    never needs them; the first breakdown had 28.

``letter_ratio``
    Over at least :data:`MIN_CHARS` non-space characters, fewer than
    :data:`MAX_LETTER_SHARE` are letters (any script, Unicode ``L*``) AND one
    single character makes up at least :data:`MIN_DOMINANT_SHARE` of them.

    The second condition is not decoration. Letter share alone cannot separate
    garbage from legitimate output: the first breakdown, with its control bytes
    removed, has the SAME letter share (8%) as a markdown results table, and a
    plain number list or CSV has 0%. What the flood has and they do not is one
    character drowning out the rest (``0`` = 99.6% of the second breakdown's
    thinking; tables, number lists, CSV and hex dumps stay under 30%).
    The thresholds are pinned by tests/test_degenerate_output_guard.py against
    the real incident text and those legitimate shapes.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

#: C0 controls except tab, newline and carriage return.
_C0 = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

#: Non-space characters needed before the letter test applies at all, so a
#: short reply ("OK", "200", a lone emoji) can never trip it.
MIN_CHARS = 200
#: Below this share of letters, the text is a candidate...
MAX_LETTER_SHARE = 0.05
#: ...and it is degenerate when one character makes up at least this share.
MIN_DOMINANT_SHARE = 0.85

CONTROL_BYTES = "control_bytes"
LETTER_RATIO = "letter_ratio"


@dataclass(frozen=True)
class Degeneration:
    """Why a round's output was judged degenerate."""

    test: str    # CONTROL_BYTES | LETTER_RATIO
    where: str   # "text" | "thinking"
    detail: str  # short, human-readable, never the raw output


def check(content: str) -> tuple[str, str] | None:
    """``(test, detail)`` when *content* is degenerate, else ``None``."""
    if not content:
        return None
    control = _C0.findall(content)
    if control:
        return CONTROL_BYTES, f"{len(control)} C0 control byte(s)"
    non_space = [ch for ch in content if not ch.isspace()]
    if len(non_space) < MIN_CHARS:
        return None
    letters = sum(1 for ch in non_space if unicodedata.category(ch).startswith("L"))
    letter_share = letters / len(non_space)
    if letter_share >= MAX_LETTER_SHARE:
        return None
    dominant, count = Counter(non_space).most_common(1)[0]
    dominant_share = count / len(non_space)
    if dominant_share < MIN_DOMINANT_SHARE:
        return None
    return LETTER_RATIO, (
        f"{letter_share:.1%} letters over {len(non_space)} chars; "
        f"{dominant!r} is {dominant_share:.1%} of them"
    )


def detect(text: str, thinking: str = "") -> Degeneration | None:
    """Check visible *text* first, then *thinking*; the first finding wins."""
    for where, content in (("text", text), ("thinking", thinking)):
        found = check(content)
        if found is not None:
            return Degeneration(test=found[0], where=where, detail=found[1])
    return None
