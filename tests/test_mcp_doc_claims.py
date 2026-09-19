"""The MCP consent claims in the user docs must agree with the code.

Same class of gate as tests/test_honest_status_notes.py — the DOCUMENT is
read and compared against the code, never the reverse — for a paragraph
that lives outside the "Honest status notes" section: the MCP bullet in
docs/guide/features.md and its README twin. Three of its claims were false
on 2026-09-18, and nothing checked them:

* "``/approve always`` remembers the answer per tool" — it never has. An
  MCP request carries no path and no command, so ``derive_grant`` (rule 4,
  approval_queue.py) has no extent to describe and mints nothing;
  ``/approve always`` approves once and the reply says so. The same
  sentence stood in ``checker.py``'s own comment on the rule.
* "a tool the server does not declare read-only requires confirmation" —
  true, and it carried the converse: a ``readOnlyHint`` tool skipped the
  prompt, which it did. Reversed 2026-09-18: every MCP call prompts,
  because the hint is a third party's self-declaration the gate cannot
  check (it never sees an MCP argument).
* "Deferred by default (``mcp_always_deferred: true``)" — the default has
  been ``false`` since #369/#376 (2026-09-01).

WHY PROHIBITIONS AND NOT STRING PINS (the router block's reasoning, reused):
the prose will be reworded, and a pin on today's wording fails on an
innocent edit and teaches the next author to weaken the test. What is
stable is each ERROR, so each detector targets the shape of the claim that
actually shipped, and ``test_the_detectors_catch_what_shipped`` runs them
against those sentences so they cannot be softened into matching nothing.
``LEGAL`` holds the corrected sentences, so they cannot be tightened until
they ban the truth either.

Each claim has a CODE half, asserted in the direction the doc now states
it: a code change that makes the old sentence true again fails here and
the prose gets rewritten rather than silently re-broken.

SCOPE: the docs a user reads as current — README.md, docs/guide/ and
docs/reference/. docs/FOUNDATION.md carries the dated 2026-09-01 status
record of the earlier rule (superseded in place by a dated line, not
rewritten) and docs/PROMETHEUS-FULL.md is a sprint chronicle; both are
history, and history is allowed to say what was true when it was written.

No ``mcp`` SDK import anywhere here, deliberately — the code halves read
the gate and the approval queue, and the advertise default is read from
``bootstrap.py``'s source, because importing ``prometheus.mcp`` pulls the
optional SDK and this file's claims do not depend on it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _user_docs() -> list[Path]:
    return [
        REPO / "README.md",
        *sorted((REPO / "docs" / "guide").glob("*.md")),
        *sorted((REPO / "docs" / "reference").glob("*.md")),
    ]


def _lines_mentioning_mcp() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for doc in _user_docs():
        for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bmcp\b", line, re.I):
                out.append((str(doc.relative_to(REPO)), i, line))
    return out


# ── claim 1: "/approve always remembers ..." ────────────────────────────
# A POSITIVE claim only. "cannot remember" / "does not remember" is the
# corrected wording and must stay legal — hence the negation guard between
# the verb and "remembers", and the third-person form.
_APPROVE_ALWAYS_REMEMBERS = re.compile(
    r"/approve\s+always`?(?:(?!\b(?:not|cannot|never|no)\b)[^.\n]){0,60}?"
    r"\bremembers\b",
    re.I,
)

# ── claim 2: a readOnlyHint tool skips confirmation ─────────────────────
_HINT_SKIPS_CONFIRMATION = (
    # "a tool the server does not declare read-only (...) requires confirmation"
    # "tools a server doesn't declare read-only require confirmation"
    re.compile(
        r"\b(?:does\s+not|doesn'?t)\s+declare\s+read-only\b[^.\n]{0,80}"
        r"\brequires?\s+confirmation\b",
        re.I,
    ),
    # "a readOnlyHint tool does not require confirmation"
    re.compile(
        r"readOnlyHint\b[^.\n]{0,40}\bdoes\s+not\s+(?:require|need)\b", re.I
    ),
)

# ── claim 3: the advertise-or-defer default ─────────────────────────────
_DEFERRED_BY_DEFAULT = re.compile(r"\bdeferred\s+by\s+default\b", re.I)
_ADVERTISED_BY_DEFAULT = re.compile(r"\badvertised\s+by\s+default\b", re.I)


def _claims_approve_always_remembers(line: str) -> bool:
    return bool(_APPROVE_ALWAYS_REMEMBERS.search(line))


def _claims_hint_skips_confirmation(line: str) -> bool:
    return any(p.search(line) for p in _HINT_SKIPS_CONFIRMATION)


def _claims_deferred_by_default(line: str) -> bool:
    return bool(_DEFERRED_BY_DEFAULT.search(line))


# The sentences that were live in this repo until 2026-09-18, verbatim.
SHIPPED = {
    "features.md Gated clause (pre-fix)":
        "**Gated:** a tool the server does not declare read-only (`readOnlyHint`) "
        "requires confirmation through the SecurityGate before it runs; "
        "`/approve always` remembers the answer per tool.",
    "README MCP bullet (pre-fix)":
        "- Scoped and gated: per-server `allowed_tools` allowlists (enforced at "
        "discovery *and* at call time), and tools a server doesn't declare "
        "read-only require confirmation before they run",
    "features.md Deferred clause (pre-fix)":
        "**Deferred by default:** MCP tools stay out of the advertised catalog "
        "(`tools.deferred_loading.mcp_always_deferred: true`) and are reached via "
        "`tool_search` or by exact name; set it `false` to advertise them, and "
        "`search_mcp: false` to hide them from fuzzy search.",
}

# The corrected sentences. Tightening a detector until one of these trips
# would ban the truth, and the only way back would be to delete it.
LEGAL = (
    "`/approve` approves the one call; `/approve always` cannot remember an MCP "
    "tool today — the request carries no path or command, so there is no extent "
    "a grant could describe, and the reply says so.",
    "every MCP tool call requires confirmation before it runs — a server's "
    "`readOnlyHint` is recorded and shown, never trusted to skip the prompt",
    "The server's `readOnlyHint` is recorded (on the `/api/mcp/servers` card, and "
    "named in the prompt) but never trusted to skip confirmation: the gate sees no "
    "MCP argument, so nothing beneath the hint could check it.",
    "**Advertised by default:** a configured server's tools are in the advertised "
    "catalog and in `tool_search`; set `tools.deferred_loading.mcp_always_deferred: "
    "true` to keep them reachable only via `tool_search` or by exact name",
)


def test_the_detectors_catch_what_shipped():
    expected = {
        "features.md Gated clause (pre-fix)": (
            _claims_approve_always_remembers, _claims_hint_skips_confirmation),
        "README MCP bullet (pre-fix)": (_claims_hint_skips_confirmation,),
        "features.md Deferred clause (pre-fix)": (_claims_deferred_by_default,),
    }
    missed = [f"{key} / {fn.__name__}" for key, fns in expected.items()
              for fn in fns if not fn(SHIPPED[key])]
    assert not missed, (
        "a detector no longer catches prose this repo actually shipped; it has "
        "been weakened past the point of proving anything:\n  "
        + "\n  ".join(missed)
    )


def test_the_detectors_leave_the_corrected_sentences_alone():
    tripped = [s[:80] for s in LEGAL
               if _claims_approve_always_remembers(s)
               or _claims_hint_skips_confirmation(s)
               or _claims_deferred_by_default(s)]
    assert not tripped, f"a detector bans the corrected wording: {tripped}"


# ── claim 1, both halves ─────────────────────────────────────────────────

def test_no_user_doc_says_approve_always_remembers_an_mcp_tool():
    bad = [f"{d}:{i}: {ln.strip()[:120]}" for d, i, ln in _lines_mentioning_mcp()
           if _claims_approve_always_remembers(ln)]
    assert not bad, (
        "a user doc says /approve always remembers an MCP approval. It approves "
        "ONCE: derive_grant mints nothing for a request with no path or command "
        "(test_approve_always_mints_no_grant_for_an_mcp_request). If that has "
        "changed, flip the code half first, then the prose:\n  "
        + "\n  ".join(bad)
    )


def test_approve_always_mints_no_grant_for_an_mcp_request():
    """The code half. What an MCP approval carries is a tool name and a
    reason — SecurityGate remembers file_path=None, command=None for it —
    and from that derive_grant (rule 4) produces no grant, so the prompt
    offers no /remember line and /approve always answers 'approved ONCE'."""
    from prometheus.permissions.approval_queue import (
        PendingAction,
        derive_grant,
        prospective_extents,
    )

    action = PendingAction(
        request_id="r1",
        tool_name="mcp__srv__thing",
        description="Third-party MCP tool mcp__srv__thing requires confirmation",
    )
    assert derive_grant(action, verb="always") is None
    assert prospective_extents(action) == {}


# ── claim 2, both halves ─────────────────────────────────────────────────

def test_no_user_doc_says_a_read_only_hint_skips_confirmation():
    bad = [f"{d}:{i}: {ln.strip()[:120]}" for d, i, ln in _lines_mentioning_mcp()
           if _claims_hint_skips_confirmation(ln)]
    assert not bad, (
        "a user doc says (or implies by its converse) that a readOnlyHint tool "
        "skips confirmation. Since 2026-09-18 every mcp__ call prompts "
        "(test_the_gate_prompts_for_every_mcp_call). If the rule changed, flip "
        "the code half first, then the prose:\n  " + "\n  ".join(bad)
    )


def test_the_gate_prompts_for_every_mcp_call():
    from prometheus.permissions.checker import SecurityGate

    for hint in (True, False):
        decision = SecurityGate().evaluate("mcp__srv__lookup", is_read_only=hint)
        assert decision.action == "APPROVE", (hint, decision)
        assert decision.requires_confirmation, (hint, decision)


# ── claim 3, both halves ─────────────────────────────────────────────────

def _shipped_advertise_default() -> bool:
    """``bootstrap.MCP_ALWAYS_DEFERRED_DEFAULT`` read from source (importing
    the module pulls the optional SDK)."""
    src = (REPO / "src" / "prometheus" / "mcp" / "bootstrap.py").read_text(
        encoding="utf-8")
    m = re.search(r"^MCP_ALWAYS_DEFERRED_DEFAULT\s*=\s*(True|False)\b", src, re.M)
    assert m, "bootstrap.py no longer declares MCP_ALWAYS_DEFERRED_DEFAULT"
    return m.group(1) == "True"


def test_the_template_ships_the_same_advertise_default_as_the_code():
    template = yaml.safe_load(
        (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    assert template["tools"]["deferred_loading"]["mcp_always_deferred"] is (
        _shipped_advertise_default()
    )


def test_the_docs_state_the_advertise_default_the_code_ships():
    deferred = _shipped_advertise_default()
    wrong, right = (
        (_ADVERTISED_BY_DEFAULT, _DEFERRED_BY_DEFAULT) if deferred
        else (_DEFERRED_BY_DEFAULT, _ADVERTISED_BY_DEFAULT)
    )
    bad = [f"{d}:{i}: {ln.strip()[:120]}" for d, i, ln in _lines_mentioning_mcp()
           if wrong.search(ln)]
    assert not bad, (
        f"MCP_ALWAYS_DEFERRED_DEFAULT is {deferred}; a user doc claims the "
        f"opposite default:\n  " + "\n  ".join(bad)
    )
    assert any(right.search(ln) for _, _, ln in _lines_mentioning_mcp()), (
        "no user doc states the MCP advertise default at all any more"
    )
