"""The shipped template must not file a key under the wrong section.

THE DEFECT
----------
YAML indentation decides which mapping a key belongs to; a reader's EYE
decides which comment block explains it, and the two can disagree silently.
The template had four keys separated from their own section by a top-level
comment block introducing a DIFFERENT one:

* ``model.suppress_thinking`` sat inside ``compaction:``'s comment block —
  while twelve lines earlier the same key was documented in commented-out
  form as *"UNSET defaults to true"*. The file contradicted itself about one
  key, in two places, and both readings were visible at once.
* ``router.fallback`` sat under the block explaining ``slash_commands``.
* ``sentinel.synthesis_enabled`` and six siblings sat under the block
  explaining passive memory recall — a different feature entirely.
* ``tracing.service_name`` / ``phoenix_endpoint`` were wedged BETWEEN the
  wiki-root documentation and the ``wiki:`` section it describes.

None of these is a parse error, so nothing caught them. They are wrong only
to the person the file exists for: the operator reading it to learn what a
knob does. That is the whole purpose of this template — the live config
carries no comments of its own — so misfiled documentation here is the
product being wrong, not a cosmetic issue.

THE RULE
--------
A nested key (indent > 0) may not be immediately preceded — blank lines
ignored — by a comment at indent 0. A top-level comment block introduces a
top-level section; any nested key after it reads as belonging to that
section, and does not.

Deliberately narrow. It does not try to judge whether a comment is ABOUT the
key below it (undecidable), only that the file's indentation and its visual
grouping do not openly contradict each other. That was enough to find all
four.
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml.default"

_KEY = re.compile(r"^(\s*)([A-Za-z_][\w.\-]*)\s*:")


def _misplaced_keys() -> list[tuple[int, str, str]]:
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()
    out: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = _KEY.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        if len(m.group(1)) == 0:
            continue
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j < 0:
            continue
        prev = lines[j]
        if prev.lstrip().startswith("#") and len(prev) - len(prev.lstrip()) == 0:
            out.append((i + 1, line.strip(), prev.strip()))
    return out


def test_no_nested_key_sits_under_a_top_level_comment_block():
    bad = _misplaced_keys()
    assert not bad, (
        "a nested key follows a TOP-LEVEL comment block, so it reads as "
        "belonging to a section it is not in:\n"
        + "\n".join(
            f"  line {n}: {key!r}\n"
            f"      preceded by: {comment!r}"
            for n, key, comment in bad
        )
        + "\n\nEither move the key up into its own section, or move the "
          "comment block down to sit against the section it describes."
    )


def test_suppress_thinking_is_documented_exactly_once():
    """The specific contradiction that motivated the rule.

    ``suppress_thinking`` defaults to True when ABSENT
    (``providers/registry.py`` — ``config.get("suppress_thinking", True)``),
    so the template documents it in commented-out form and must not ALSO set
    it. Setting it changed no behaviour, which is precisely why the
    contradiction survived: both readings were correct about the value and
    only the file was incoherent.
    """
    text = TEMPLATE.read_text(encoding="utf-8")
    live = re.findall(r"^\s*suppress_thinking\s*:", text, re.M)
    documented = re.findall(r"^\s*#\s*suppress_thinking\s*:", text, re.M)
    assert not live, (
        "suppress_thinking is SET in the template; it should be documented "
        "commented-out, because absent already means true"
    )
    assert len(documented) == 1, (
        f"expected exactly one commented-out suppress_thinking line, "
        f"found {len(documented)}"
    )
