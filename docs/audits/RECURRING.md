# Recurring Audits

Checks worth re-running on a cadence (each was born from a real incident —
the incident is cited so the check's point survives staff/context turnover).

## 1. Orphan-tool grep

**Question:** does every tool class under `tools/builtin/` actually get
registered into the live registry (`__main__.create_tool_registry` /
`daemon.py`)?

**Born from:** HERMES verification audit B3 — `AnatomyTool` and
`WhisperSTTTool` existed, were imported nowhere, and silently never appeared
to the model (see `docs/audits/HERMES-VS-PROMETHEUS-VERIFICATION.md`).

**How:** for each `class *Tool(BaseTool)` in `tools/builtin/`, grep for its
registration; flag classes with zero registration sites. Tools intentionally
unregistered must say so in their module docstring.

## 2. Reachability audit (defenses live under production config?)

**Question:** for every defense — validator check, guard clause, repair path,
breaker, sanitizer — is it actually *reachable* under the production
configuration, or has a config knob (tier, strictness, feature flag, default)
quietly amputated it?

**Born from:** the 2026-06 tool-calling diagnostics (D1):
`ToolCallValidator.validate()` contained an empty-tool-name check written for
exactly the failure that then occurred 232 times — but production runs tier
"light" → strictness NONE, and the NONE short-circuit sat *above* the check,
so it was dead code in the only configuration that needed it. Fixed by the
invariants-vs-policy split (invariants run at every strictness); the general
lesson recurs.

**How:** enumerate guard/validation sites (grep for `return ValidationResult`,
`raise`, `is_error=True`, breaker `record_error`, permission checks); for each,
trace the config path that reaches it and confirm the production values
(config/prometheus.yaml + env) don't gate it off. Telemetry cross-check: a
defense that has *never once fired* in telemetry history (e.g. `repairs > 0`
count == 0 rows ever, as of this audit) is either unreachable or untested —
both worth knowing. Pairs well with the per-defense counters in
`tool_calls.error_type`.
