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


---

## 4. A control that names the wrong thing

**Question:** for every rule, check or gate — does the thing it names
actually govern the thing that goes wrong? A control is only a control if
the named mechanism, disabled, would have prevented the failure. Otherwise
it is a memorial that reads like a control, and it occupies the slot where
the real one belongs.

**Born from three, recorded together on 2026-09-10** because they were only
recognisable as one shape once they were side by side.

### 4a. The rule named the wrong mechanism

**PR #4** was merged and its branch deleted. That branch was the base of
**#7, #8 and #9**, which GitHub closed permanently — a closed PR cannot have
its base changed, so they could not be reopened. A standing rule was
written: **disable auto-delete-on-merge.**

On **2026-09-10, PR #417** was merged and its branch deleted. That branch
was the base of **#418**, which closed permanently. Same shape, one rule
later.

Measured that day, three sources:

```
repos/OAraLabs/Prometheus  .delete_branch_on_merge : false
branches/main/protection                           : HTTP 404 (not protected)
ruleset "protect-main" rules                       : deletion, non_fast_forward,
                                                     pull_request,
                                                     required_linear_history
```

**Auto-delete was already off, and had been.** Nothing in the settings,
branch protection or the ruleset deletes a head branch here. The only thing
that does is `--delete-branch` on the merge invocation. Following the rule
perfectly would have changed nothing.

**How:** omit `--delete-branch`; branches accumulate, which is cheap beside
a permanently closed PR. Before merging, check whether the head is another
PR's base (`gh pr list --json number,baseRefName`) and merge such a stack in
immediate succession.

### 4b. The check named the wrong dependency

Sequencing five PRs the same day, the merge order was computed from
**pairwise file overlap** — the standard check. #427 (generates a route,
command and config reference from source) and #430 (adds an `mcp:` section
to the config template) share **no file**: #427 touches nothing under
`config/`, #430 nothing under `docs/reference/`. Independent, by that check.

Merged #427 then #430 and `main` went red — three of #427's own drift guards,
because the generated table was two rows short of the template it describes.

**A generated artifact depends on every input to its generator, whatever
files the other change happens to touch.** File overlap cannot see that.
`routes.md` survived only because #430 guarded existing routes instead of
adding one.

**How:** land a generated-reference PR **last** among any batch touching
generator inputs, or regenerate afterwards. Repaired in #431.

### 4c. The gate named the wrong tree

`scripts/smoke_test_tool_calling.py` builds its own AgentLoop and
SecurityGate in-process; it does not drive the running daemon. Its score is
about whatever `import prometheus` resolved to, and it said nothing about
which tree that was.

A `_prometheus.pth` in user site-packages, present since **2026-04-07**,
puts a dev checkout on every interpreter's `sys.path`. With `PYTHONPATH`
set — the systemd unit sets it — the deploy tree wins. Without it the
checkout wins, and the bare `python3 scripts/smoke_test_tool_calling.py` is
the invocation in use.

**THE RETRACTION, measured from transcripts and git:**

```
dev checkout            : diag/355-llama-tokenize @ b107c29 (2026-08-29)
                          62 dirty files, 79 commits behind main
main diverged from it   : 2026-08-30, 10968cf
smoke invocations seen  : 117 bare, 72 correctly pathed (2026-08-02 .. 2026-09-10)
BARE runs on/after the divergence : 42
```

Every one of those 42 scored a tree that was not deployed, in a report read
as a statement about the deployment. Before 2026-08-30 the checkout still
tracked main closely, which is why five months of this was invisible: the
answer happened to be right, so nobody asked how it was obtained.

The count is a **lower bound**. Transcripts are a rolling window, so runs
before ~2026-08-11 are not covered, and runs outside an agent session are
not counted at all.

**How:** the script now resolves `(facts, verdict)` with `verdict` in
`matches | mismatch | unknown`, prints the loaded package path, SHA, branch
and dirty count beside the tree the systemd unit loads, and refuses on a
mismatch naming both paths. The three renderings are asserted pairwise
distinct — see 2 in this file, and
`context.budget.resolve_effective_limit`, whose `(value, source)` shape this
copies. **Unknown and matches must never render identically.**

⚠ The `.pth` itself is deliberately NOT removed. Something may depend on it,
the checkout it points at has 62 dirty files, and pulling a path entry out
from under that is how this gets a second instance instead of a fix. The
reporting was fixed first; the removal is a recommendation.

### The rule the three share

**Verify that the named mechanism was actually operating.** In 4a the
setting was one API call away and nobody asked, so a guess became a standing
rule — wrong in the direction that reads as diligence. In 4b the check ran
correctly and answered a question adjacent to the one that mattered. In 4c
the gate was never asked what it was measuring.

A control you have not seen fail is not yet known to be a control. Prefer
the version that can produce a *distinguishable* wrong answer — a tag with
an `unknown` state, a check whose subject is named — over one whose only
output is silence.

**Not found in this repository.** The 4a rule is not in `docs/`,
`PROMETHEUS.md` or any tracked `*.md` — searched when this section was
written. Wherever it lives, it should be corrected or removed rather than
left to be followed.
