"""The template's value and the code's live fallback must be EQUAL TO EACH OTHER.

WHY THIS SHAPE, AND NOT AN EXPECTED-VALUE TABLE
------------------------------------------------
Both sides are read programmatically — the template by parsing it, the code by
walking ``src/`` with an AST pass. Neither is compared against a hand-written
list of "expected defaults".

That is the whole design. A restated expectation reproduces its author's
reading inside the test: whoever wrote the template would write the same
numbers into the table, and the table would agree with the template while the
CODE disagreed with both. The bug survives in the assertion. Comparing two
live answers is the only form that cannot do that.

THREE CATEGORIES, KEPT APART
-----------------------------
1. read with a LITERAL fallback  -> equality enforced
2. read with NO fallback         -> template must ship the key; exempt from
                                    equality, because there is nothing to be
                                    equal to
3. in the template, no reader    -> not this file's business; that is
                                    ``test_config_drift.KNOWN_UNREAD``

Plus a fourth the survey did not anticipate:

4. CONFLICT — two call sites read the same key with DIFFERENT literal
   defaults. There is no single "the code's fallback" to compare against, so
   the guard FAILS and names the disagreeing sites. Exclusions are BY SITE,
   never by key: excluding a key would have hidden ``gateway.telegram_enabled``,
   where one site said True (behaviour, public gateway) and another False
   (display) — a live security defect fixed in #219.

WHAT THE EXTRACTOR PROVES, SAID OUT LOUD (§2b)
-----------------------------------------------
``tests/support/config_defaults`` finds dict-key reads whose receiver it has
already tied to a config expression. It under-reports (attribute-held configs,
helper indirection) and it can mis-resolve a nested section to a bare name. So
this guard is a RATCHET over what it can see — never a claim of completeness.
The debt register below may only shrink.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.config.template import load_template
from tests.support.config_defaults import (
    NO_DEFAULT, equivalent, extract_config_reads, flatten, open_maps)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# DEBT REGISTER — a shrinking list, enforced in BOTH directions.
#
# Every entry is a real disagreement between what the template documents and
# what the code does when the key is absent. None is fixed here: changing 19
# runtime defaults is a behaviour decision, not a documentation one, and
# burying it inside a 364-key template PR is how a default change ships
# unread.
#
# Each entry states WHICH SIDE IS SUSPECTED WRONG, because "there is a
# disagreement" is not actionable.
# ---------------------------------------------------------------------------
KNOWN_DEFAULT_MISMATCHES: dict[str, str] = {
    # ── absence flips a subsystem relative to what the template documents.
    #    Same shape as gateway.telegram_enabled (#219), on local subsystems
    #    rather than a public surface, so none is a security defect.
    "sentinel.enabled": "CODE-SUSPECT — template false, daemon.py:1220 falls "
                        "back True: absence STARTS the proactive subsystem",
    "compaction.enabled": "TEMPLATE-SUSPECT — template true, compactor.py:226 "
                          "falls back False: absence disables compaction while "
                          "the template says it is on",
    "web.enabled": "CODE-SUSPECT — template true, doctor.py:185 falls back "
                   "False, so doctor skips the port check when the key is absent",
    # (coding.max_iterations / coding.max_task_duration_minutes left the
    # register 2026-08-28: PR-B shape-C ruling aligned the fallbacks to the
    # shipped template — the register only shrinks. The pair is now pinned
    # both-ways by tests/test_config_absence_rulings.py.)
    # ── RESOLVER: the read the pass sees is INSIDE a resolver that supplies
    #    the shipped value. Behaviour is correct; the extractor sees the inner
    #    `.get(k)` and not the function wrapping it. Registered so the blind
    #    spot is visible rather than silently excluded.
    "security.workspace_root": "RESOLVER — shipped_defaults.resolve_workspace_root "
                               "supplies SHIPPED_WORKSPACE_ROOT",
    "security.denied_paths": "RESOLVER — was ABSENCE-HOSTILE and is now fixed: "
                             "an absent key yielded [] and the entire file "
                             "boundary vanished, which every path gate this "
                             "week resolved against. resolve_denied_paths now "
                             "supplies SHIPPED_DENIED_PATHS and "
                             "checker._ALWAYS_DENIED_PATHS is a structural "
                             "floor beneath any config.",
    "gateway.telegram_enabled": "RESOLVER — shipped_defaults.resolve_telegram_enabled "
                                "supplies SHIPPED_TELEGRAM_ENABLED (#219)",
    "model.max_tool_iterations": "RESOLVER — shipped_defaults.resolve_max_tool_iterations "
                                 "supplies SHIPPED_MAX_TOOL_ITERATIONS (LONGHAUL-1b). The "
                                 "literal used to sit at EIGHT sites and had already drifted "
                                 "(live 50 vs template 25) — the very divergence this file "
                                 "exists to catch, which it could not see because the value "
                                 "was restated rather than resolved.",
    "model.max_tool_iterations_cloud": "RESOLVER — shipped_defaults."
                                       "resolve_max_tool_iterations_cloud supplies "
                                       "SHIPPED_MAX_TOOL_ITERATIONS_CLOUD (LONGHAUL-1b).",
    # ── template ships a real value; the reader has NO literal fallback, so
    #    absence yields None and the documented value is never the effective
    #    one. Category 2 with a non-empty template value.
    "coding.docker_image": "WIRE — reader has no fallback; absence yields None",
    "evals.judge_base_url": "WIRE — reader has no fallback; absence yields None",
    "gateway.media.cache_dir": "WIRE — reader has no fallback; absence yields None",
    "learning.curator_interval_seconds": "WIRE — reader has no fallback",
    # ⚠ BOTH DISPOSITIONS CORRECTED. They were registered together as one
    #    WIRE class, and they are not the same class at all — verified by
    #    outcome, three config shapes, not by reading the code.
    "security.denied_commands": "RESOLVER — absence is SAFE and always was. "
                                "checker._ALWAYS_BLOCKED_PATTERNS is a "
                                "hardcoded floor applied BEFORE the config "
                                "list, so `rm -rf /` is refused whether or not "
                                "the key exists; the config list is additive. "
                                "Never a WIRE defect.",
    "symbiote.backup.exempt_from_retention": "WIRE — reader has no fallback",
    "symbiote.harvest_model": "WIRE — reader has no fallback",
    "symbiote.scout_model": "WIRE — reader has no fallback",
    "tasks.default_timeout_seconds": "WIRE — reader has no fallback",
    "tasks.poll_initial_interval_seconds": "WIRE — reader has no fallback",
    "tasks.poll_max_interval_seconds": "WIRE — reader has no fallback",
    "whisper.enabled": "WIRE — reader has no fallback; absence yields None",
}

# Sites excluded from the CONFLICT rule — BY SITE, never by key.
#
# A key-level exclusion would have hidden gateway.telegram_enabled, whose two
# sites disagreed because one was behaviour (daemon, True) and one display
# (wizard, False) — a public gateway starting on absence (#219). Excluding the
# display SITE keeps the behaviour site under the guard. Applying this
# exclusion resolved five apparent conflicts that were wizard placeholders.
CONFLICT_SITE_EXCLUSIONS: dict[str, str] = {
    "src/prometheus/setup_wizard.py": "interactive wizard — its '?' and '' "
                                      "defaults are DISPLAY placeholders for "
                                      "'not detected yet', not config fallbacks",
}

KNOWN_CONFLICTS: dict[str, str] = {
    "model.model": "DECIDE — 'qwen3.5-32b' (daemon.py:290, __main__.py:105) vs "
                   "'unknown'/'default' at report sites",
    "model.provider": "DECIDE — '' (daemon.py:334) vs 'llama_cpp' elsewhere",
    "trajectory_export.enabled": "CODE-SUSPECT — daemon.py:1369 falls back False "
                                 "(behaviour), doctor.py:594 falls back True "
                                 "(report): doctor reports an exporter that is "
                                 "not running",
}

# Roots the extractor could not bind to their real parent. Each is a NESTED
# section surfacing as a bare top-level name because the pass never tied the
# receiver to a config expression. Documenting them as top-level keys would
# invent sections that do not exist — the inert-config class this work removes
# — so they are registered as a known blind spot instead, each naming its real
# path so the next reader does not re-derive it.
MIS_RESOLVED_ROOTS: dict[str, str] = {
    "voice": "MIS-RESOLVED — really gateway.voice.*, documented there",
    "comfyui": "MIS-RESOLVED — really image_generation.comfyui.*",
    "dashscope": "MIS-RESOLVED — really image_generation.dashscope.*",
    "ds": "MIS-RESOLVED — a LOCAL ALIAS (`ds = cfg.get(\"dashscope\")`); the "
          "name appears in no config anywhere",
    "kling": "MIS-RESOLVED — really video_generation.kling.*",
    "vision_model": "MIS-RESOLVED — really learning.video_ingest.vision_model.*",
    "checks": "MIS-RESOLVED — the live-recorder quality gate's sub-config, "
              "passed in by its caller rather than read from the root",
}


def _analysis():
    facts = extract_config_reads(REPO / "src", REPO)
    tmpl = load_template()
    tv, opens = flatten(tmpl), open_maps(tmpl)

    def under_open(k: str) -> bool:
        return any(k.startswith(m + ".") for m in opens)

    def is_section(k: str) -> bool:
        return isinstance(tv.get(k), dict) or any(
            o != k and o.startswith(k + ".") for o in facts)

    mismatches: dict[str, tuple] = {}
    conflicts: dict[str, list[str]] = {}
    absent: list[str] = []
    agreed = 0

    for key, f in sorted(facts.items()):
        if under_open(key) or is_section(key) or "." not in key:
            continue
        lits = [r for r in f.literal_defaults
                if not any(r.file == s for s in CONFLICT_SITE_EXCLUSIONS)]
        vals: list[object] = []
        for r in lits:
            if not any(equivalent(v, r.default) for v in vals):
                vals.append(r.default)
        if len(vals) > 1:
            conflicts[key] = [r.site for r in lits]
            continue
        if key not in tv:
            if key.split(".")[0] not in MIS_RESOLVED_ROOTS:
                absent.append(key)
            continue
        code_default = vals[0] if vals else NO_DEFAULT
        if equivalent(tv[key], code_default):
            agreed += 1
        else:
            mismatches[key] = (tv[key], code_default,
                               [r.site for r in (lits or f.reads)])
    return mismatches, conflicts, absent, agreed


def test_template_values_equal_the_codes_live_fallbacks():
    """Category 1. Both sides read programmatically; neither restated."""
    mismatches, _, _, agreed = _analysis()
    assert agreed > 100, f"only {agreed} keys compared — the extractor went blind"
    new = {k: v for k, v in mismatches.items() if k not in KNOWN_DEFAULT_MISMATCHES}
    assert not new, (
        f"{len(new)} config key(s) where the template documents one value and "
        f"the code falls back to another. An operator reading the template "
        f"believes something the daemon does not do.\n\n"
        + "\n".join(f"  {k}\n    template={t!r}  code={d!r}  ({s[0]})"
                    for k, (t, d, s) in new.items())
    )


def test_no_new_conflicting_defaults():
    """Category 4. Two sites, two answers — there is no 'the' default."""
    _, conflicts, _, _ = _analysis()
    new = {k: v for k, v in conflicts.items() if k not in KNOWN_CONFLICTS}
    assert not new, (
        f"{len(new)} key(s) are read with DIFFERENT literal defaults at "
        f"different call sites. gateway.telegram_enabled was exactly this — "
        f"True at the behaviour site, False at the display site, and a public "
        f"gateway started on absence (#219).\n\n"
        + "\n".join(f"  {k}: {sites}" for k, sites in new.items())
    )


def test_every_key_the_code_reads_is_in_the_template():
    """Category 2. Absence of documentation is the FL-2 class."""
    _, _, absent, _ = _analysis()
    assert not absent, (
        f"{len(absent)} key(s) are read by src/ and documented nowhere in the "
        f"template. A reader cannot learn the setting exists.\n\n  "
        + "\n  ".join(sorted(absent))
    )


def test_debt_registers_are_not_stale():
    """The ratchet. A fixed entry must leave the register, so it only shrinks.

    Without this the registers are allowlists — write once, hide forever.
    """
    mismatches, conflicts, _, _ = _analysis()
    facts = extract_config_reads(REPO / "src", REPO)
    live_roots = {k.split(".")[0] for k in facts}
    stale_m = sorted(set(KNOWN_DEFAULT_MISMATCHES) - set(mismatches))
    stale_c = sorted(set(KNOWN_CONFLICTS) - set(conflicts))
    stale_r = sorted(set(MIS_RESOLVED_ROOTS) - live_roots)
    assert not stale_r, (
        f"the extractor no longer mis-resolves {stale_r} — remove them from "
        f"MIS_RESOLVED_ROOTS; the blind-spot register only shrinks.")
    assert not stale_m and not stale_c, (
        "registered entries no longer disagree — remove them. These are "
        "shrinking debt lists, not allowlists.\n"
        f"  fixed mismatches: {stale_m}\n  fixed conflicts:  {stale_c}"
    )


@pytest.mark.parametrize(
    "register",
    [KNOWN_DEFAULT_MISMATCHES, KNOWN_CONFLICTS, MIS_RESOLVED_ROOTS])
def test_every_registered_entry_carries_a_disposition(register):
    """No silent entries. An unexplained entry is the next hiding place."""
    valid = ("WIRE", "DECIDE", "DELETE", "CODE-SUSPECT", "TEMPLATE-SUSPECT",
             "MIS-RESOLVED", "RESOLVER")
    bad = sorted(k for k, v in register.items()
                 if not v.strip().startswith(valid))
    assert not bad, f"entries must start with one of {valid}:\n  " + "\n  ".join(bad)


def test_conflict_exclusions_are_by_site_not_by_key():
    """The rule that keeps a telegram_enabled from hiding again."""
    for site in CONFLICT_SITE_EXCLUSIONS:
        assert site.endswith(".py") and "/" in site, (
            f"{site!r} is not a file path. Exclusions must name a SITE; a "
            f"key-level exclusion would have hidden gateway.telegram_enabled, "
            f"whose behaviour site and display site disagreed."
        )
