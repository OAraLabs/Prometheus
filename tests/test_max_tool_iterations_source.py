"""LONGHAUL-1b — the tool-call ceiling has ONE source of truth, and it is raised.

Two things are pinned here. First that the number is no longer written out at
eight sites (it had already drifted: live ran 50 against a template of 25, the
divergence docs/sprints/SPRINT-CONSENT.md names because the config drift guard
checks key PRESENCE and cannot see a value mismatch). Second that a long
PRODUCTIVE run now survives past the old ceiling — the outcome the raise exists
for, asserted through the real entry point rather than by reading a constant.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from prometheus.config.shipped_defaults import (  # noqa: E402
    SHIPPED_MAX_TOOL_ITERATIONS,
    SHIPPED_MAX_TOOL_ITERATIONS_CLOUD,
    resolve_max_tool_iterations,
    resolve_max_tool_iterations_cloud,
)
from tests.support.real_app import (  # noqa: E402
    BOUNDARY_DOUBLE,
    RecordingProvider,
    build_real_app,
)

REPO = Path(__file__).resolve().parents[1]
OLD_CAP = 50  # the ceiling this sprint raises


# --------------------------------------------------------------------------- #
# 1 — no site re-states the number
# --------------------------------------------------------------------------- #


def test_no_hardcoded_cap_defaults_outside_shipped_defaults():
    """Every construction site must resolve the ceiling, never restate it.

    This is the guard the drift itself asked for: a value written in eight
    places drifts, and the existing config drift guard cannot see it because it
    checks whether a KEY is present, not what it equals.
    """
    pattern = re.compile(
        r"max_tool_iterations(?:_cloud)?[\"']?\s*[:=,]\s*[\"']?(\d+)"
    )
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        if path.name == "shipped_defaults.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue          # historical comments may cite old values
            m = pattern.search(line)
            if m:
                offenders.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    # gym/runner.py pins a deliberately tiny ceiling for benchmark determinism.
    offenders = [o for o in offenders if "gym/runner.py" not in o]
    assert not offenders, (
        "the ceiling is restated instead of resolved — that is how 50-vs-25 "
        "drifted in the first place:\n  " + "\n  ".join(offenders)
    )


def test_template_matches_the_code_default():
    """SPRINT-CONSENT.md: 'The template value and the code default must match.'"""
    text = (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
    local = re.search(r"^\s*max_tool_iterations:\s*(\d+)", text, re.M)
    cloud = re.search(r"^\s*max_tool_iterations_cloud:\s*(\d+)", text, re.M)
    assert local and cloud, "template no longer declares the ceiling"
    assert int(local.group(1)) == SHIPPED_MAX_TOOL_ITERATIONS, (
        f"template says {local.group(1)}, code default is {SHIPPED_MAX_TOOL_ITERATIONS}"
    )
    assert int(cloud.group(1)) == SHIPPED_MAX_TOOL_ITERATIONS_CLOUD


def test_the_ceiling_actually_rose():
    """Both ceilings clear the old value.

    NOT asserted: cloud >= local. That held for the whole prior life of the key
    (50 vs 25) and an earlier cut of this file pinned it — but the ordering is
    now deliberately INVERTED, because cloud rounds are billed per call and
    local rounds are not. Re-adding that assertion would re-encode a rationale
    the operator has explicitly retired.
    """
    assert SHIPPED_MAX_TOOL_ITERATIONS > OLD_CAP
    assert SHIPPED_MAX_TOOL_ITERATIONS_CLOUD > OLD_CAP


def test_resolver_rejects_values_that_would_halt_every_turn():
    """A zero or negative ceiling stops a turn on its first tool batch, and a
    hand-edited YAML yields strings. Both are 'not configured', not obeyed."""
    assert resolve_max_tool_iterations({"max_tool_iterations": 7}) == 7
    assert resolve_max_tool_iterations({"max_tool_iterations": "7"}) == 7
    for bad in (0, -1, None, "", "abc", {}):
        assert resolve_max_tool_iterations({"max_tool_iterations": bad}) == (
            SHIPPED_MAX_TOOL_ITERATIONS
        ), f"{bad!r} was obeyed instead of falling back"
    assert resolve_max_tool_iterations_cloud({}) == SHIPPED_MAX_TOOL_ITERATIONS_CLOUD


# --------------------------------------------------------------------------- #
# 2 — the outcome: a long PRODUCTIVE run survives past the old ceiling
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_long_productive_run_survives_past_the_old_cap(tmp_path):
    """60 DISTINCT tool calls — well past the old ceiling of 50 — must run to
    completion instead of being cut off on round count.

    Every call is distinct and returns new data, so it is productive by the
    repeat detector's definition and must not trip it either. This is the
    outcome the raise exists for; asserting the constant alone would prove
    nothing about the loop.
    """
    rounds = OLD_CAP + 10
    script = [
        ("tool", "bash", {"command": f"echo step-{i}", "cwd": str(tmp_path)})
        for i in range(rounds)
    ] + [("text", "completed the long run")]
    rec = RecordingProvider(label="primary:local", script=script)
    h = build_real_app(
        primary=rec, tool_config={"workspace_root": str(tmp_path)}
    )

    with h.client:
        h.send_turn(SESSION := "web", "do the long job", timeout=60.0)

        assert len(rec.requests) == rounds + 1, (
            f"expected all {rounds} tool rounds + the closing text, saw "
            f"{len(rec.requests)} — the ceiling cut a productive run short"
        )
        sess = h.session_mgr.get_or_create(SESSION)
        final = next(
            (m.text for m in reversed(sess.messages)
             if m.role == "assistant" and (m.text or "").strip()), "")
        assert "completed the long run" in final, final
        assert "Tool iteration limit reached" not in final, final
        assert "Halted: no progress" not in final, (
            "distinct productive calls tripped the repeat detector"
        )
