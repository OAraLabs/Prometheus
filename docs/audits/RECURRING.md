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

## 3. A command lied about its own outcome

**Question:** for every command whose exit status or output decides something —
*did the suite pass, is the daemon running, did the push land* — is the status
you read the status of the thing you care about?

**Born from:** three occurrences, recorded together because none of them was
recorded at the time. `git grep -niE "PIPESTATUS|pipefail|exit code.*pipe"`
across tracked `*.md` returned **zero** before this section existed, so there
was no ledger to append a third item to. This is not a backfilled history — it
is the first entry, and it happens to have three instances.

1. **2026-08-30, Beacon parity.** A full `pytest` run read through `| tail`.
   The status belonged to `tail`, which always succeeds. A failing suite
   reported success.
2. **2026-09-07, P4 security sprint.** An `EXIT=0` printed by a smoke script,
   from the same shape. The number was real and meant nothing.
3. **2026-09-07/08, same sprint.** `echo "exit: $?"` after a pipeline, twice in
   one session.

**Not a defect in tracked code, and the distinction matters.** All five tracked
`*.sh` files already `set -o pipefail`, and no CI step reads the status of a
pipeline — both verified when this section was written. This is an **ad-hoc
command** trap. It bites in interactive sessions and agent tool calls, where the
pipeline is typed once, the number is believed, and nothing reviews it.

**How:** `set -o pipefail`, or read `${PIPESTATUS[0]}` explicitly. Never accept
a status printed after a pipe without one of the two. When the result decides
whether to ship, print the status *and* the evidence: a bare `EXIT=0` is not
evidence that the thing you ran succeeded.

### Related shapes — the report is not about what you think

Same family, different mechanism. Each was measured, not recalled.

* **`pgrep -f <pattern>` finds the shell that is running the search.** Measured:
  searching for a marker string that matches **no running program at all**
  returned exit 0 and three matches. `pgrep -f` compares full command lines, and
  the pattern is sitting in the command line of the shell invoking it — pgrep
  excludes its own PID, but not its parent. So a restart check written this way
  reports "still running" for a process that never existed, and the operator
  concludes the stop failed.

  Remedies: match on the pid file or `systemctl --user is-active` instead; or
  keep the pattern out of the searching command line (`pgrep -f "[p]attern"` is
  the old trick); or compare against a known PID rather than a string.

* **`git push` during a conflicted rebase succeeds, and pushes the wrong
  commit.** Measured on git 2.43 with a deliberately conflicted rebase:

  ```
  HEAD state                      : HEAD (no branch), detached at the partial replay
  refs/heads/feature still points : the PRE-rebase tip
  git push origin feature         : exit 0 — "* [new branch]"
  ```

  The push sends the **branch ref**, which the rebase has not moved yet. So it
  reports success while publishing the state you were rebasing *away from*, and
  the work you are standing in is not in it. Finish or abort the rebase first;
  `git status` saying "rebase in progress" is the tell.

