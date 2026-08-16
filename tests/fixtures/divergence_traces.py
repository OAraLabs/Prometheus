"""Divergence fixtures — PROVENANCE IS PART OF THE FIXTURE.

Each trace below states whether it is RECORDED or SYNTHETIC, because a
calibration round that cannot tell the two apart is calibrating against its
own author. Do not add a trace here without that line.

────────────────────────────────────────────────────────────────────────────
FL4_DEPLOY_PROOF — RECORDED
    Source: the brain vault, wiki/log.md:217, [FINDING 2026-08-12 21:45].
    A deploy-proof turn that "did exactly what it was asked (six sequential
    `echo` calls, all successful)" scored **0.96** and logged
    `WARNING Divergence: … diverged`. Goal alignment scored **0.04** on a turn
    that complied exactly.
    The recorded mechanism: `_create_checkpoint` clears
    `tool_calls_since_checkpoint`, and `evaluate` runs immediately after
    `maybe_checkpoint` in the same block, so on a checkpoint-boundary step the
    tool window is empty, the failure-rate and repetition terms are skipped,
    and the average collapses to the single goal-alignment term.

    The tool NAMES, COUNT, SUCCESS and the checkpoint-boundary timing are from
    that record. The exact prose of the goal message and the echoed strings
    were NOT preserved in the vault entry, and are invented here.

    ⚠⚠ WHAT THIS FIXTURE DOES AND DOES NOT PIN — read before citing it.

    It pins the TOOL-SHAPE direction ONLY: a run of same-tool calls that are
    PRODUCTIVE (distinct, non-empty results) must not trip the repetition
    floor. That is a real property and this fixture proves it.

    It does NOT reproduce the recorded incident. The recorded 0.96 depended on
    goal alignment scoring **0.04** — near-zero lexical overlap between the
    turn's text and the request. This fixture yields alignment **1.0**
    (goal_alignment term 0.0), because the invented prose overlaps its own
    invented goal. Reproducing 0.04 would require the actual request text,
    which no longer exists.

    **So the LEXICAL direction — the mechanism of the actual recorded defect —
    is NOT PINNED BY ANYTHING in this repository.** The lone-term rule that
    addresses it is exercised by
    ``test_a_lone_lexical_term_can_never_declare_divergence``, which is a
    CONSTRUCTED case, not this trace. Do not read a green run of this file as
    evidence that the recorded 0.96 turn would now score correctly.

────────────────────────────────────────────────────────────────────────────
SYMBIOTE_STUCK — ⚠ SYNTHETIC. NOT A RECORDED TRACE.
    Will observed a SYMBIOTE documentation turn scoring as a false negative —
    seven greps, different arguments each time, the agent plainly stuck. No
    trace of it exists on disk: no fixture, no vault entry, and
    `subsystem_runs` carries no divergence rows. Searched 2026-08-16.

    So this trace is CONSTRUCTED to the description, by me. It is fit for
    asserting that the aggregation no longer dilutes a repetition signal to
    nothing. It is NOT evidence about what that turn actually scored, and it
    must never be cited as such. If the shape is ever recorded for real,
    replace this and delete the label.
"""

from __future__ import annotations

# --- FL4_DEPLOY_PROOF — RECORDED -------------------------------------------

FL4_GOAL = (
    "Prove the fl4 deploy actually landed: confirm the daemon is running the "
    "merged commit and report the running sha."
)

# Six sequential, successful `bash` calls. Deliberately low lexical overlap
# with FL4_GOAL — that is what produced alignment 0.04.
FL4_TOOL_CALLS = [
    {"step": i + 1, "tool": "bash", "args": {"command": cmd},
     "result": out, "success": True, "timestamp": 1755000000.0 + i}
    for i, (cmd, out) in enumerate([
        ("echo 24f862d", "24f862d"),
        ("echo 200", "200"),
        ("echo idle", "idle"),
        ("echo 8005", "8005"),
        ("echo ok", "ok"),
        ("echo done", "done"),
    ])
]

FL4_MESSAGES = [
    {"role": "user", "content": FL4_GOAL},
    {"role": "assistant", "content": "24f862d 200 idle 8005 ok done"},
]

# The recorded score under the OLD aggregation. Kept as the thing the new
# behaviour is measured against, not as a target to reproduce.
FL4_RECORDED_SCORE = 0.96
FL4_RECORDED_ALIGNMENT = 0.04


# --- SYMBIOTE_STUCK — ⚠ SYNTHETIC ------------------------------------------

SYMBIOTE_GOAL = (
    "Document the symbiote subsystem: describe how the coordinator, morph and "
    "backup vault fit together."
)

# Seven greps, DIFFERENT arguments each time. Hash-exact repeat detection sees
# seven distinct calls and fires on none of them; same-tool detection fires.
SYMBIOTE_TOOL_CALLS = [
    {"step": i + 1, "tool": "grep", "args": {"pattern": pat},
     "result": "", "success": True, "timestamp": 1755000100.0 + i}
    for i, pat in enumerate([
        "coordinator", "morph", "backup_vault", "set_coordinator",
        "SymbioteCoordinator", "graft", "harvest",
    ])
]

SYMBIOTE_MESSAGES = [
    {"role": "user", "content": SYMBIOTE_GOAL},
    # High lexical overlap with the goal — a doc turn echoes its own subject,
    # which is why goal_alignment stayed healthy while the agent was stuck.
    {"role": "assistant",
     "content": "The symbiote coordinator, morph and backup vault fit "
                "together: the coordinator drives morph and the backup vault."},
]
