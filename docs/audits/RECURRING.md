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

**How:** `set -o pipefail`, then read `$?`. Never accept a status printed after
a pipe without it. When the result decides whether to ship, print the status
*and* the evidence: a bare `EXIT=0` is not evidence that the thing you ran
succeeded.

> **This rule used to read "or read `${PIPESTATUS[0]}` explicitly."** That half
> was wrong in both shells it was aimed at — corrected 2026-09-10, see **§4d**.
> `set -o pipefail` + `$?` is the portable read, and the only one recommended
> here.

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

* **A secret scanner reported "All clean" for a scan it never performed.**
  `.githooks/pre-commit` ran every pattern through `grep -nP`, and the call was
  `... | grep -nP "$pattern" || true`. BSD grep — the grep git invokes on macOS —
  has no `-P`:

  ```
  /usr/bin/grep -qP abc  : grep: invalid option -- P
                           exit 2
  ```

  `|| true` collapses grep's **three** outcomes into two: exit 1 *no matches*
  and exit 2 *the scanner did not run* both produced an empty result. All nine
  patterns evaluated nothing, and the hook printed `All clean. No sensitive
  data found in staged files.` and exited 0 with a real match staged. Measured
  by live probe before the fix: **9 invalid-option errors, 0 blocks, exit 0.**

  This is the §3 shape with the stakes inverted — not a test suite whose pass
  you over-trust, but the repo's only secret scanner, and the mechanism behind
  the standing no-infrastructure-identifiers rule. It reported *clean*, which
  is stronger than reporting *success*: clean is a claim about the data.

  **The flag was not the defect; the failure mode was.** Swapping `-P` for `-E`
  works today and leaves the next missing flag to read as "no matches" again.
  Fixed in both directions: patterns are POSIX ERE (both greps have it), and
  grep's exit code is now a three-way contract where `>=2` **refuses the
  commit**, naming the pattern that could not be evaluated. A scanner that
  cannot run must fail closed — `unknown` and `zero` must never render the
  same. Guarded by `tests/test_precommit_fails_closed.py`, which replays a
  grep that rejects the flag, a grep that is absent, and — the one an
  exit-code check alone cannot catch — a grep that truthfully-looking returns
  exit 1 for everything. That last one is why the hook now self-tests its
  engine against a positive control before trusting any clean.

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
recognisable as one shape once they were side by side. **4d was added the same
day**, from the fourth: a rule in *this file* that named a mechanism the shell
it was written for does not have.

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

### 4d. The rule named a mechanism the shell does not have

**§3's own remedy** said: "`set -o pipefail`, **or read `${PIPESTATUS[0]}`
explicitly**." That rule was written for, and applied in, sessions whose shell
is **zsh**. `PIPESTATUS` is a bash-ism. zsh spells it `$pipestatus` and indexes
from 1, so in zsh `${PIPESTATUS[0]}` is not wrong — it is **empty**.

Measured 2026-09-10, `(exit 7) | cat` under `set -o pipefail`:

```
zsh   $?                 : 7
zsh   ${PIPESTATUS[0]}   : (empty)
zsh   ${pipestatus[1]}   : 7
bash  $?                 : 7
bash  ${PIPESTATUS[0]}   : 7    -- but only when read IMMEDIATELY
bash  ${PIPESTATUS[0]}   : 0    -- after ONE intervening command
```

Two failures, and the second is worse than the first.

**In zsh the recommended read expands to nothing.** `echo "rc=${PIPESTATUS[0]}"`
prints `rc=` — not a wrong number, no number. Piped into a comparison it is an
empty string, and `[ "" -eq 0 ]` is a syntax error, not a verdict. Followed,
the rule produces a verdict-shaped blank.

**In bash it is right only for one command.** `PIPESTATUS` is rebuilt by the
*next* command — including a bare assignment. `q=$?; echo "${PIPESTATUS[0]}"`
reports **0** for a pipeline that failed with 7, because the assignment
overwrote it. The natural way to write it is the broken way.

So the rule against trusting a status after a pipe named, as its alternative,
a mechanism that is absent in one shell and silently stale in the other. It is
§4a exactly: following it perfectly would not have helped, and it occupied the
slot where the working remedy belongs.

**How:** `set -o pipefail` and read `$?`. It is correct in both shells, it
survives an intervening command, and it is what §3 now says. If you genuinely
need a *per-stage* status, spell the shell's own name for it (`$pipestatus` in
zsh, `$PIPESTATUS` in bash) and read it on the very next line — but prefer
restructuring so you never need it: capture into a variable, then test.

### 4e. The revert that did not revert

A mutation test has two halves. Everyone checks the first — *does the test fail
when the fix is removed?* — and nobody checks the second: **was the fix actually
removed?** If the revert silently no-ops, the suite runs against the unchanged
code, passes, and reports that as the mutation's result. It is a green light
with nothing behind it, and it reads exactly like a guard that works.

Three instances, all measured, all in one session:

**1. The shell ate the revert.** A Python mutation script was written inside
double quotes: `python3 -c "... s.replace('`dropped`', ...) ..."`. Backticks
inside double quotes are command substitution, so the pattern was replaced by
the shell before Python ever saw it. The edit never happened. Both the "fails
without the fix" and "passes with it" runs were the same run.

**2. `git stash push -- <path>` on a clean path.** The file was committed, so
there was nothing to stash: *"No local changes to save"*, **exit 0**. The `||`
fallback never fired. Fifteen tests then "passed on main" — main still had the
fix, because it had never been taken out.

```
=== the same tests against main (fix stashed out) ===
15 passed in 3.69s      <- the fix was never removed
```

**3. `git checkout HEAD -- <path>` with the fix uncommitted.** This restores
from the index, which held *main's* version — so it did not revert the mutation,
it deleted the work. Sixty-one lines of an in-progress fix, gone, and the next
mutation's result was then measuring main's code while claiming to measure the
fix's.

The three share one shape and it is §4a's: **the named mechanism was never
verified to have operated.** In 1 the tool ran and edited nothing. In 2 the
tool declined and said so in a message nobody read, while exiting 0. In 3 the
tool did exactly what it is documented to do, against a different baseline than
the one intended.

**How:**

- Make the harness **assert the mutation landed** before running anything —
  anchor matched exactly once, new text present, old text absent — and abort
  loudly when it did not. An aborted mutation must not be reportable as a
  result.
- **Assert the revert landed too**, by a property of the file rather than by the
  command's exit code (`grep -c` for a symbol the fix introduces).
- **Commit before mutating.** Then `git checkout HEAD -- <path>` restores the
  fix rather than removing it, and a botched mutation costs nothing.
- Never revert with a command whose no-op case exits 0. `git stash push` on a
  clean path, `sed -i` with a pattern that does not match, and `patch -R` on an
  already-reverted file all succeed at doing nothing.

A harness that printed one line per mutation caught all three:

```
  [mutation verified present in tracker.py]
  ABORT: anchor matched 0 times in tracker.py — mutation NOT applied
  [restored; guard present again: 1]
```

The middle line is the one that matters. It fired on a real mutation whose
indentation was wrong, and the `15 passed` that followed was correctly read as
meaningless rather than as a surviving mutant.

### 4f. Two runs on one commit, disagreeing

A required check reported **`quality=FAILURE` and `quality=SUCCESS` at the same
time, on the same head SHA.** Two workflow runs had been created one second
apart, and the rollup listed both. The failing one was correct: the branch was
based on a main that still carried an `F821`, and its gate was genuinely red.

Merging on "a green check exists" would have merged it.

The obvious patch — *verify the check ran against the current head, not an
earlier one* — does not catch this. Both runs were on the current head. And
the artifacts cannot settle which one a gate would honour:

```
failure : started 20:40:53   completed 20:41:02
success : started 20:40:50   completed 20:41:25
```

The success **started earlier but completed later**, so "most recent" gives
opposite answers depending on which timestamp is read.

**GitHub does not document the resolution.** Its guidance is to *prevent* the
situation — *"make sure that job names are unique across all workflows. Using
the same job name in multiple workflows can cause ambiguous status check
results"* — not to explain which wins. So branch protection cannot see the
disagreement, and on the `completed_at` reading the spurious SUCCESS would have
satisfied a required check and merged the branch red anyway.

**The rule is not "the failure wins."** A genuine flake would make that wrong
in the other direction, and a real failure and a flaky one are identical from
the rollup. Disagreement means **unmeasured**.

**How:**

- Before merging, check that the runs on the head SHA **agree**, not merely
  that a green one exists. `gh api repos/O/R/commits/<sha>/check-runs` and group
  by name; more than one entry for a name is a stop, not a tiebreak.
- Resolve a disagreement by **reading the failing run's log** and deciding which
  is right. Never by preferring an outcome.
- Branch protection is the floor under this rule, not a substitute for it.

**And the same shape bit the tool written to enforce it.** A poll waiting for
CI used `until [ <unfinished runs> = 0 ]`. Before any run had been created that
count is zero, so it reported two PRs "COMPLETE" whose CI had not started.
**Zero runs is not zero unfinished runs** — absent read as passing, ten minutes
after the rule against it was written. The condition must require *at least one*
run AND none unfinished.

### 4g. Verifying the file a claim names is not verifying the claim

A report arrived as: *"`benchmarks/runner.py` executes a full benchmark at
import time — a blanket import probe triggered 26 network-bound task runs."*

It was checked properly. `runner.py` parsed and scanned: a correct
`if __name__ == "__main__":` guard at the bottom, no module-level execution.
Same for `suite.py` and `__init__.py`. A direct import measured at 0.36s
running nothing. Reported: **does not reproduce**, with the method shown.

The behaviour was real. `src/prometheus/benchmarks/__main__.py` called `main()`
at module level with no guard — and `__main__.py` was never opened, because the
claim named `runner.py` and `runner.py` was where the checking went.

The original report was **right about the behaviour and wrong about the file**.
Checking the named file and finding it clean then confirmed a negative that was
false — and did it with evidence attached, which made it more convincing than
an unchecked assertion would have been.

The question to ask was not *"is `runner.py` clean"* but **"what would a blanket
import probe actually import"** — which is `__main__`.

**How:**

- Reproduce the **reported behaviour**, not the reported location. Run the probe
  that produced the claim before reasoning about the file it accuses.
- When a check comes back clean, ask whether the check was **capable** of seeing
  the reported symptom. A scan of three modules cannot detect a fourth.
- A negative needs its scope stated: *"`runner.py`, `suite.py` and `__init__.py`
  are clean"* is true and survives; *"does not reproduce"* is a claim about the
  whole phenomenon and did not.

This one degraded through three hands, and each hand was doing something
defensible: the first session reported a real behaviour under the wrong
filename, the second verified that filename and reported a false negative, and
the relay between them passed each finding on without checking either. No step
was careless. The claim still ended up inverted, which is the argument for
re-deriving a finding from the symptom rather than inheriting it.

---

### 4h. A fix verified through one reader of the changed column

**#307 shipped its write-side fix and the surface that motivated the issue
kept lying.** `latency_ms` became nullable in schema v2, `record()` defaulted
to `None`, the aggregate path (`report()`) got source-tagged averages — all
tested, all merged, all deployed. And `/api/tools/recent`, the route that
hydrates Beacon's Tool Feed (the exact client bug #307 was filed from, beacon
#84), collapsed NULL straight back with `float(row[6] or 0.0)`. The store held
NULL; the served payload said 0.0; Beacon rendered `0ms` for calls that never
ran. One writer survived too: the loop's `validation_failed` path passed an
explicit `latency_ms=0.0` past the fixed default — live data showed the only
two post-boundary 0.0 rows in 22k were both from it.

The verification was real and stopped at the layer that changed. Tests proved
`record()` stores NULL and `report()` tags provenance. Nothing enumerated the
*other* readers of the column, so the defect class the fix existed to kill
consumed the fix one layer up — this month's whole audit theme ("a surface
reporting a value the system does not hold") reproduced inside its own
remediation.

**#284 got this right by accident**: its check 4 ran at `/api/usage`, a read
surface, so the billing fix was verified where the value is consumed. Same
discipline, and the difference is only which layer the test happened to sit
at.

**The rule:** when a stored representation changes — nullability, units, a
sentinel's meaning, an encoding — enumerate every reader of that column
before claiming the fix, and verify through at least the surface that
motivated the finding. Concretely:

- `git grep` the column name across `src/`, not just the module you changed;
  readers hide in dashboards, livestream tails, API routes and export scripts.
- Grep the readers for the collapse idioms: `or 0`, `or 0.0`, `or ""`,
  `COALESCE(x, 0)`, `float(x or ...)`. Each one turns "absent" back into a
  number and undoes a write-side fix silently.
- The regression test belongs at the PAYLOAD level — assert what the served
  route returns, not what the function returns. A function-boundary test
  re-commits the error: it passes while the surface still lies.
- Pin both directions: unmeasured → NULL on the wire, and a genuine measured
  zero → 0.0 on the wire. Mapping both to one value is a different collapse
  with the same shape.

**How it was caught:** not by the tests that shipped with the fix, but by a
later session quoting the issue's own stated harms back and checking each
against `main` and the live DB before closing it. "The fix is merged and
deployed" was true; "the harm is gone" was not. That gap is the rule.

**When the motivating surface lives in another repo.** The obligation does not
become unfollowable just because the surface that motivated the finding is not
in this tree — it becomes a *linked pin filed there at the same time*, not a
skipped step. "I can't test Beacon from Prometheus" is true and is not a
discharge of the rule; it is the cue to open the companion issue/PR in the
other repo before this one merges. The cross-repo contract has two ends and
each end pins its own side:

- the side that *changed the representation* pins the wire value at its own
  payload boundary (here: `/api/tools/recent` returns `latency_ms: null`);
- the side that *consumes* it pins that the value survives its own reader to
  the surface the user sees (here: `toolCallsFromApi(null latency)` → `—`).

Neither end can see the other's collapse. A consumer-side guard that turns
`null` back into `0` is invisible to the producer's tests, exactly as a
producer collapsing NULL was invisible to the consumer's. So file both, link
them, and check the consumer's *existing* coverage before filing — #85 here
had a runtime guard and a `fmtLatency(undefined)` assertion but **no smoke
driving `latency_ms: null` through `toolCallsFromApi`**, so the end-to-end path
was true-but-unpinned and a `?? 0` refactor could silently reintroduce `0ms`
with the producer green. That is the gap the companion pin closes.

A type that declares the value cannot arrive when the wire now sends it is the
same defect class one level up — `latency_ms?: number` (not `| null`) tells the
next contributor that `r.latency_ms ?? 0` is correct per the signature and
wrong on the data. Widening the type is part of the consumer-side pin, not a
separate nicety.

---

---

---

### The rule the eight share

**Verify that the named mechanism was actually operating.** In 4a the
setting was one API call away and nobody asked, so a guess became a standing
rule — wrong in the direction that reads as diligence. In 4b the check ran
correctly and answered a question adjacent to the one that mattered. In 4c
the gate was never asked what it was measuring. In 4d the mechanism was
never run in the shell it was prescribed for — one `(exit 7) | cat` would
have shown it expanding to nothing. In 4e the revert command was trusted on
its exit code, which reported that the command had run, not that it had done
anything. In 4f a green check was trusted without asking whether the other
runs on that commit agreed with it. In 4g a file was verified in place of the
claim that named it, so a scan that could not have seen the defect was read as
evidence of its absence. In 4h the changed layer was verified in place of the
consumed one, so a fix was trusted to have reached a surface no test had ever
asked about.

A control you have not seen fail is not yet known to be a control. Prefer
the version that can produce a *distinguishable* wrong answer — a tag with
an `unknown` state, a check whose subject is named — over one whose only
output is silence.

**4d is the sharpest form of it, because the rule was in this file.** A
document that collects controls which named the wrong thing contained one.
Nothing about writing the shape down inoculates the next rule against it;
the only thing that does is running the named mechanism once and reading
what it returns. §3's remedy is now one mechanism, not two, because the
second was never measured.

**Not found in this repository.** The 4a rule is not in `docs/`,
`PROMETHEUS.md` or any tracked `*.md` — searched when this section was
written. Wherever it lives, it should be corrected or removed rather than
left to be followed.
