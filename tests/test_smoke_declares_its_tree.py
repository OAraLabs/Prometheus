"""The smoke script must say which codebase it scored — audit follow-up.

`scripts/smoke_test_tool_calling.py` builds its own AgentLoop, ToolRegistry
and SecurityGate in-process; it does not drive the running daemon. So the
"6/6" it prints is a statement about whatever `import prometheus` resolved
to, and it used to say nothing about which tree that was.

Measured 2026-09-10: a `_prometheus.pth` in user site-packages (present
since 2026-04-07) puts a dev checkout on every interpreter's sys.path.
With PYTHONPATH set — the systemd unit sets it — the deploy tree wins.
Without it, the checkout does. That checkout was 79 commits behind on a
feature branch with 62 dirty files, and **42 bare runs between 2026-08-30
and 2026-09-10** scored it while the report read as a statement about the
deployment.

⚠ THE DEFECT IS THE SILENCE, NOT THE PATH ENTRY. Removing the `.pth` fixes
this instance; the next one is another venv, or a wheel installed beside a
checkout. What is fixed here is a verification script that could bind to a
different codebase and still hand you a number.

The contract mirrors `context.budget.resolve_effective_limit`: return the
value AND where it came from, with "unknown" a state of its own — because a
tag that renders like agreement when nothing was checked is the same failure
in a smaller font.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture()
def smoke():
    import smoke_test_tool_calling as mod
    return mod


# ── the three states are three states ───────────────────────────────

def test_the_verdict_is_a_tag_not_a_boolean(smoke):
    assert {smoke.PROVENANCE_MATCHES, smoke.PROVENANCE_MISMATCH,
            smoke.PROVENANCE_UNKNOWN} == {"matches", "mismatch", "unknown"}


def test_unknown_and_matches_never_render_identically(smoke):
    """The rule, asserted directly.

    An UNKNOWN that prints like a MATCH is how a verification step becomes
    decoration: the reader sees a block above a green score and takes it for
    confirmation. All three renderings must be distinguishable from each
    other, not merely non-empty.
    """
    facts = {
        "loaded_package": "/x/src/prometheus", "loaded_src": "/x/src",
        "git_root": "/x", "sha": "a" * 40, "branch": "main", "dirty": 0,
        "service_src": "/y/src",
    }
    rendered = {v: smoke.render_provenance(facts, v)
                for v in (smoke.PROVENANCE_MATCHES, smoke.PROVENANCE_MISMATCH,
                          smoke.PROVENANCE_UNKNOWN)}
    assert len(set(rendered.values())) == 3, rendered
    assert "UNKNOWN" in rendered[smoke.PROVENANCE_UNKNOWN]
    assert "MISMATCH" in rendered[smoke.PROVENANCE_MISMATCH]
    # ...and the two non-confirming states must not read as confirmation.
    for v in (smoke.PROVENANCE_MISMATCH, smoke.PROVENANCE_UNKNOWN):
        assert "TREE OK" not in rendered[v]


def test_an_unknown_verdict_never_claims_a_service_tree(smoke):
    facts = {"loaded_package": "/x/src/prometheus", "loaded_src": "/x/src",
             "git_root": None, "sha": None, "branch": None, "dirty": None,
             "service_src": None}
    out = smoke.render_provenance(facts, smoke.PROVENANCE_UNKNOWN)
    assert "could not be determined" in out
    assert "same tree" not in out


def test_missing_git_facts_render_as_unknown_not_as_blank(smoke):
    """A blank SHA beside a green score reads as 'nothing to report'."""
    facts = {"loaded_package": "/x/src/prometheus", "loaded_src": "/x/src",
             "git_root": None, "sha": None, "branch": None, "dirty": None,
             "service_src": "/x/src"}
    out = smoke.render_provenance(facts, smoke.PROVENANCE_MATCHES)
    assert "unknown" in out
    assert "dirty state unknown" in out


# ── the gate ────────────────────────────────────────────────────────

def _gate(smoke, monkeypatch, capsys, verdict, *, allow=False, facts=None):
    facts = facts or {
        "loaded_package": "/dev/src/prometheus", "loaded_src": "/dev/src",
        "git_root": "/dev", "sha": "b" * 40, "branch": "diag/x", "dirty": 62,
        "service_src": "/deploy/src",
    }
    monkeypatch.setattr(smoke, "resolve_package_provenance",
                        lambda: (facts, verdict))
    rc = smoke.provenance_gate(allow)
    return rc, capsys.readouterr().out


def test_matches_proceeds(smoke, monkeypatch, capsys):
    rc, out = _gate(smoke, monkeypatch, capsys, smoke.PROVENANCE_MATCHES)
    assert rc == 0
    assert "TREE OK" in out


def test_mismatch_refuses_and_names_both_paths(smoke, monkeypatch, capsys):
    """Naming ONE path leaves the reader to guess which is wrong."""
    rc, out = _gate(smoke, monkeypatch, capsys, smoke.PROVENANCE_MISMATCH)
    assert rc != 0
    assert "/dev/src/prometheus" in out, "the tree under test is not named"
    assert "/deploy/src" in out, "the service's tree is not named"
    assert "REFUSING TO RUN" in out


def test_mismatch_prints_the_corrected_command(smoke, monkeypatch, capsys):
    _rc, out = _gate(smoke, monkeypatch, capsys, smoke.PROVENANCE_MISMATCH)
    assert "PYTHONPATH=/deploy/src" in out


def test_unknown_refuses_by_default(smoke, monkeypatch, capsys):
    rc, out = _gate(smoke, monkeypatch, capsys, smoke.PROVENANCE_UNKNOWN)
    assert rc != 0
    assert "REFUSING TO RUN" in out


def test_unknown_can_be_accepted_out_loud(smoke, monkeypatch, capsys):
    rc, out = _gate(smoke, monkeypatch, capsys, smoke.PROVENANCE_UNKNOWN,
                    allow=True)
    assert rc == 0
    assert "--allow-unverified-tree" in out, (
        "accepting an unverified tree must leave a trace in the output — "
        "otherwise the escape hatch reintroduces the silence"
    )


def test_the_escape_hatch_does_not_silence_a_mismatch(smoke, monkeypatch, capsys):
    """A flag that says 'I could not check' must not also mean 'I checked
    and it is wrong, run anyway'."""
    rc, _out = _gate(smoke, monkeypatch, capsys, smoke.PROVENANCE_MISMATCH,
                     allow=True)
    assert rc != 0


def test_the_script_wires_the_gate_before_running_anything(smoke):
    """A resolver nothing calls is the shape of the dead `expect_tools`
    parameter this same file already carries a scar from."""
    src = (REPO / "scripts" / "smoke_test_tool_calling.py").read_text(encoding="utf-8")
    body = src[src.index('if __name__ == "__main__":'):] if '__main__' in src else src
    gate = body.index("provenance_gate(")
    run = body.index("asyncio.run(main(")
    assert gate < run, "the gate must decide before any test executes"


# ── against this checkout, for real ─────────────────────────────────

def test_the_resolver_reports_real_fields_here(smoke):
    facts, verdict = smoke.resolve_package_provenance()
    assert verdict in {"matches", "mismatch", "unknown"}
    assert facts["loaded_package"].endswith("prometheus")
    if facts["sha"] is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", facts["sha"]), facts["sha"]
    # Whatever the verdict, the rendering must be one of the three shapes.
    out = smoke.render_provenance(facts, verdict)
    assert out.splitlines()[0].strip().startswith("TREE ")
