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

**Question:** two of them, and the second is the one this section kept missing.

1. For every defense — validator check, guard clause, repair path, breaker,
   sanitizer — is it actually *reachable* under the production configuration,
   or has a config knob (tier, strictness, feature flag, default) quietly
   amputated it?
2. **Did the configuration LOAD?** A defense reached with a config that was
   never read is not configured — it is running on whatever the code
   substituted, and the substitution is usually indistinguishable from a
   deliberate setting.

Question 1 assumes the config in memory is the config on disk. Ask question 2
first, because when it fails, question 1's answer is fiction.

**Born from:** the 2026-06 tool-calling diagnostics (D1):
`ToolCallValidator.validate()` contained an empty-tool-name check written for
exactly the failure that then occurred 232 times — but production runs tier
"light" → strictness NONE, and the NONE short-circuit sat *above* the check,
so it was dead code in the only configuration that needed it. Fixed by the
invariants-vs-policy split (invariants run at every strictness); the general
lesson recurs.

**Also born from:** the 2026-08-31 config-silence audit. `DEFAULTS_PATH` used
five `.parent` hops where four reach the repo root, so it had never resolved to
a real file on any checkout. Eight subsystems read it, every one behind
`except (OSError, Exception): section = {}` — a handler that converts "I could
not read your configuration" into "you did not configure anything." The defect
survived two years and ~6000 tests because nothing ever asked whether the read
SUCCEEDED. This section was already the right protocol and would have caught
it, except that it only ever asked whether config *gated a defense off* — never
whether the config was there to gate with. `SecurityGate.from_config()` was
reachable, ran, and enforced an empty deny list.

**How:** enumerate guard/validation sites (grep for `return ValidationResult`,
`raise`, `is_error=True`, breaker `record_error`, permission checks); for each,
trace the config path that reaches it and confirm the production values
(config/prometheus.yaml + env) don't gate it off.

Then, for the same sites, trace the config LOAD: which file is opened, whether
that path resolves on this install layout, and what the code does when it does
not. A read whose failure branch substitutes a default is only honest if it
says so — see `prometheus.config.load`, which sorts every read into LOADED /
ABSENT / UNREADABLE / MALFORMED and records the last two to the
`silent_failures` ledger. `tests/test_config_read_honesty_invariant.py` fails
the build on a broad catch that swallows a config read; the standing manual
check is the layout question the guard cannot answer — *does this path exist on
a checkout, on the deploy clone, and under site-packages?*

Telemetry cross-check: a defense that has *never once fired* in telemetry
history (e.g. `repairs > 0` count == 0 rows ever, as of this audit) is either
unreachable or untested — both worth knowing. Pairs well with the per-defense
counters in `tool_calls.error_type`. The config-load equivalent: a
`silent_failures` row with subsystem `security_gate` / `token_budget` and state
`unreadable` means a subsystem is live on defaults right now.
