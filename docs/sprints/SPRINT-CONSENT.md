# SPRINT: CONSENT

**Priority:** Outranks LONGHAUL. This sprint moves two phases out of LONGHAUL (audit resolution rows, approval timeout) because they are permission-system work, not loop work. LONGHAUL becomes purely the iteration cap, status push, and telemetry sprint.

**Most of the audit is already done.** The Phase 0 findings below were established by outcome on 2026-08-16 at `84142f9` and are marked **[ANSWERED]**. Do not re-derive them. Phase 0 here covers only what remains unknown.

---

## Goal

The permission system currently obtains consent under a false description, cannot revoke what it grants, and keeps no record of what was decided. Fix all three. The unoverridable `SHIPPED_DENIED_PATHS` floor is the only thing bounding this today, and it is four months younger than the defect it bounds — it is load-bearing by accident, and this sprint should make it load-bearing by design.

## The three properties that compose

Each is survivable alone. Together they are the finding of the night.

1. **The prompt describes one file; the grant covers a directory.** `derive_grant` rule 2 grants `target.resolve().parent`. A single approval of one file in `$HOME` produced a permanent `write_file` grant over all of `$HOME`.
2. **The widest grant comes from the case carrying the least information.** The rule-4 fallback produces `kind="tool", value=""`, which `matches()` evaluates without looking at the path at all. One approval grants the whole tool forever.
3. **Nothing can be revoked.** `SecurityGate` exposes `add_grant`, `list_grants`, `persist_grant` — no `remove_grant`, no `revoke`, no `clear`. No DELETE on REST; `/grants` is list-only. The sole removal path is hand-editing config plus a restart.

**The operator sees none of this at consent time.** The prompt states *duration* ("remember permanently") and never *extent*. The true scope appears only afterward, in the response: `"Approved and remembered permanently (saved to config): path_prefix /home/will"`. By then the grant exists and cannot be undone.

That is consent obtained under a false description. It is worse than a plain UI gap, because the description is *accurate about the request* and *silent about the consequence*.

---

## Phase 0 — what is already established

**[ANSWERED] `derive_grant` — four rules, first match wins.** `approval_queue.py:46`.

| # | line | condition | produces |
|---|---|---|---|
| 1 | `:59-64` | `root` given | `path_prefix` = resolved `root`. **Unreachable from the chat path** — `cmd_approve` calls `derive_grant(action)` with no root |
| 2 | `:65-67` | `action.grant_file_path` | **`path_prefix` = the target's PARENT DIRECTORY**. Docstring calls this "CC's directory-grant semantic" — deliberate |
| 3 | `:68-69` | `action.grant_command` | `command_prefix` = the command, `tool_name="bash"` |
| 4 | `:70` | fallback | **`kind="tool"`, `value=""`** — the tool itself, any target |

**[ANSWERED] Narrowest and widest per kind.**

| kind | narrowest from one approval | widest from one approval |
|---|---|---|
| `path_prefix` | deepest containing directory | **`/`** — approve any top-level file and the parent is the filesystem root |
| `command_prefix` | the exact command string | **a prefix** — `checker.py:265` is `command.startswith(self.value)`, so approving `git push` also permits `git push --force`. Bounded only by `_TRUSTED_CMD_FORBIDDEN`, which blocks `;` `\|` `&` `` ` `` `$()` chaining |
| `tool` | — | **unbounded** — `matches()` at `checker.py:249-250` returns `tool_name == self.tool_name` and never looks at the path |

**[ANSWERED] The floor holds above every grant kind.** Established by outcome, not by reading the ordering:

```
                                      .ssh    .gnupg   .config/*/*env   ordinary file
baseline, NO grants                   DENY     DENY        DENY           APPROVE
path_prefix /home/will                DENY     DENY        DENY           ALLOW
path_prefix /           (widest path) DENY     DENY        DENY           ALLOW
kind=tool, value=''     (widest ever) DENY     DENY        DENY           ALLOW
```

The ordinary-file column flips `APPROVE → ALLOW`, so this is not a uniform-DENY artifact. Mechanism: `_check_denied_path` at `checker.py:590` returns DENY *before* grants are consulted at `:599`, with the design stated in the comment — *"a grant can silence a prompt, never resurrect a block."*

**This ordering is the sprint's single most important invariant. Any refactor must preserve it, and must test it by outcome in both directions.**

**[ANSWERED] The prompt.** `approval_queue.py:134-141`. `{description}` is the `reason` from `checker.py:768` — a single path. Nothing in the prompt describes extent. Beacon inherits it unchanged: `server.py:1979` forwards `scope` into the same `cmd_approve`.

**[ANSWERED] No revocation path.** `checker.py:717` (`add_grant`), `:726` (`list_grants`), `:729` (`persist_grant`). Nothing else. Grants are write-only. The in-memory grant from the survey probe survived until a daemon restart cleared it.

**[ANSWERED] Approval resolution is never audited.** `approve()` at `approval_queue.py:158-164` sets `action._result` and signals an event. `deny()` at `:167-173` is identical. Neither writes an audit row. The request half *does* write — `checker.py:619`, `:624`, `:646` all call `_audit_log(..., CONFIRM_PENDING, ...)`. `CONFIRM_APPROVED` and `CONFIRM_REJECTED` are defined at `audit.py:35-36` and **referenced nowhere else in `src/`**. The queue holds no `AuditLogger` reference today, so wiring one in is part of the fix.

Result: 24,048 rows across four months, `allow` 23,858, `confirm_pending` 79, `deny` 111. **Zero resolutions recorded against at least six demonstrated approvals.**

**[ANSWERED] The approval timeout expires silently.** `security.approval_queue.timeout_seconds: 300`. `approval_queue.py:150-155` does `asyncio.wait_for`, sets `ApprovalResult.TIMEOUT`, and pops the request **in a `finally`**. The field incident was exactly this: request raised 15:52, popped at 15:57, the `/approve always` arrived after the window closed. **No user-facing expiry message** — the only trace is a later, confusing "No pending approval requests."

**[ANSWERED] `test_approval_grants.py` observes nothing.** 20 tests, none touching a config file. `test_always_scope_records_persistent_grant` asserts on `gate.list_grants()` (in-memory) and on `grants[0].scope == "persistent"` — a label on an object. **It would pass identically with `persist_grant` deleted.**

### Still to establish — Phase 0 proper

**0a. Who else constructs grants?** Run the count-diff instrument, not a keyword sweep. Enumerate every call site of `add_grant` and `persist_grant` across `src/`. For each, cite the caller and whether it goes through `derive_grant` or builds a grant directly. A second construction path that skips `derive_grant` would make every fix below partial.

**0b. Grant provenance.** Does a stored grant record which request created it, when, or by whom? Cite the grant dataclass. If not, revocation has no handle to revoke *by* and the audit trail has no join key.

**0c. Config write safety.** `persist_grant` writes to `prometheus.yaml` — the same file the daemon reads and the config drift guard checks. What happens on a concurrent write, a partial write, or a write while another session holds the file? Cite the write path. The probe showed a clean before/after hash, but that was a single serialized write.

**0d. Session-scope grants.** `/approve session` exists in the prompt text. Where does a session-scoped grant live, how is it distinguished from persistent, and is it cleared at session end by outcome? If session and persistent share a store with only a label distinguishing them, that is the same shape as the `scope == "persistent"` test that observes nothing.

**0e. What does Beacon show?** The prompt text at `approval_queue.py:134-141` is the Telegram rendering. Establish what the Beacon approval UI displays — the same reason string, or something else. Cite it. A fix to the Telegram prompt that leaves Beacon describing one file is half a fix.

**CHECKPOINT-HALT after Phase 0.** Report and stop.

---

## Phase 1 — Consent must describe extent

**The rule: the prompt must state the grant it is about to create, before the operator consents. Not the request. The grant.**

- Compute the prospective grant at prompt time, using the same `derive_grant` the approval path will use. Not a re-derivation — the same call, so the two cannot drift. (See Standing Principle §17, under-population at a shared construction site.)
- Render its extent in the prompt in operator terms: which tool, which paths or commands, and for how long. `/approve always` on a file in `$HOME` must say it will grant `write_file` across all of `/home/will`, permanently.
- **Default narrows to the exact target.** The directory-grant semantic becomes an explicit opt-in (`/approve always here` or equivalent), not the default. Rule 2's widening is deliberate and useful; it is not defensible as a silent default when the prompt shows one path.
- Note the ergonomic cost honestly in the PR: narrower defaults mean more prompts. The alternative is grants nobody intended and cannot remove.

**Rule 4 must not produce a grant at all.** If extent cannot be determined, it cannot be described, so informed consent cannot be obtained. The fallback should either refuse to create a persistent grant (approve once, no memory) or fail loud. Do not silently produce the widest grant in the system from the case with the least information.

**`command_prefix` needs the same treatment.** `startswith` matching means approving `git push` permits `git push --force`. Either require exact match by default with prefix as opt-in, or state the prefix semantic in the prompt. Decide and say which in the PR.

---

## Phase 2 — Revocation must exist

Grants are currently write-only. Add the missing half:

- `SecurityGate.remove_grant()`, and a `clear_grants()` for the whole set.
- Removal must clear both the in-memory list **and** the persisted config entry. A revoke that leaves the disk copy reappears on restart; a revoke that leaves the memory copy is live until restart. Both halves, tested separately.
- Gateway command (`/revoke <id>` or `/grants revoke`), and a DELETE on the REST surface so Beacon can reach it.
- Revocation needs a handle. If 0b found no provenance on the grant record, add a stable id at creation time.
- **Write an audit row on revocation.** Revoking a permission is a security decision and belongs in the same record as granting one.

---

## Phase 3 — Audit resolution rows

*(Moved here from LONGHAUL Phase 3d.)*

The accountability record captures every request and not one decision. Fix the missing write:

- `_audit_log` calls inside `ApprovalQueue.approve()` and `.deny()`, writing `CONFIRM_APPROVED` and `CONFIRM_REJECTED`.
- The queue holds no `AuditLogger` reference today — wire one in.
- Each row records: request id, tool, resolved target, **scope** (`once` / `session` / `always`), the derived grant if one was created, and the actor.
- **The scope field closes a real ambiguity permanently.** An empty grants store currently cannot distinguish "`always` was never invoked" from "`always` was invoked and dropped." With scope recorded, an `always` that writes no grant is visible in the audit log instead of indistinguishable from silence. That ambiguity cost a live probe to resolve.
- Also write a row on **timeout** (Phase 4), so an expired request is a recorded outcome rather than an absence.

Minor, same area: `tool_input_summary` was empty on the permission audit rows examined. The audit records the reason but not the arguments. Establish whether that is universal or specific to that path, and fix it if cheap.

---

## Phase 4 — Approval timeout

*(Moved here from LONGHAUL Phase 3c, whose original premise — that the iteration halt discards the queue — was disproven. The pop is the request's own timeout, independent of the halt.)*

- **Expiry must notify.** When a request times out and is popped, tell the operator: which request, which tool, which target, and that it expired unapproved. Silence here produced a field incident that read as a broken `/approve always` and cost hours to diagnose correctly.
- **300 seconds is shorter than a human's response latency to a phone notification.** Raise the default and make the value visible in the prompt ("expires in N minutes"). Choose a number and defend it in the PR.
- The template value and the code default must match. The live-vs-template drift on `max_tool_iterations` (50 vs 25) is the failure this prevents; the drift guard checks key *presence* and cannot see a value divergence.
- Write the timeout audit row from Phase 3.

---

## Phase 5 — Tests

Real instances, not mocks. Assert side effects, not calls.

1. **Prompt states extent.** Trigger an out-of-workspace write, assert the prompt text names the prospective grant's extent — not just the requested path. Assert byte-identity between the extent shown and the grant subsequently created.
2. **Default narrows.** `/approve always` on a file produces a grant covering that file, not its parent. The opt-in form produces the directory grant.
3. **Rule 4 produces no persistent grant.** Assert by reading the store, not by inspecting the return value.
4. **Revocation, both halves.** Revoke, then assert the grant is absent from `list_grants()` **and** absent from the config file on disk. Then restart-equivalent: reload config and assert it stays gone.
5. **Audit resolution rows, both directions.** An approval writes `confirm_approved`; a rejection writes `confirm_rejected`; a timeout writes its row. Assert on rows read back from the store, including the scope field.
6. **Timeout notifies.** Let a request expire, assert the operator-facing message is emitted and names the request.
7. **The floor still holds above every grant kind.** Re-run the four-kind matrix from Phase 0 as a permanent test, including `kind="tool", value=""` and `path_prefix /`. Assert the admission direction too — an ordinary file must flip `APPROVE → ALLOW` — so a uniform-DENY regression cannot pass as success.
8. **`persist_grant` deletion must fail the suite.** The existing `test_approval_grants.py` would pass with `persist_grant` deleted. Add a test that reads the config file from disk. Then prove it: delete `persist_grant` as a mutation and confirm the suite goes red.

Run with:

```bash
PYTHONPATH=$PWD/src python3 -m pytest
```

Never bare `pytest` in a worktree. **Name the environment when citing counts** — a worktree venv reported 4821 where CI's baseline is 4614.

**Mutation-test every phase.** Tests 1, 4, 7 and 8 are the load-bearing ones; break each deliberately and confirm the matching test goes red. A mutation that does not mutate is a bad mutation, not a survivor — withdraw it rather than reporting it.

Consider `@pytest.mark.acceptance` via `tests/support/real_app.py` for tests 1, 4 and 5. A conftest double would make them green while the real path stays broken.

---

## Constraints

- Additive where possible. Where a default narrows (Phase 1), say so plainly in the PR — that is a deliberate behaviour change, not a bug fix.
- **Preserve the floor-before-grants ordering at `checker.py:590` / `:599`.** Test it by outcome, both directions. It is the only thing bounding this defect class today.
- Fail loud over silent failure. No `except` blocks that swallow. A swallowed programming error must not read as a failed operation — that shape shipped #230 with two defects in eleven lines.
- Diagnose before implementing. If Phase 0 contradicts anything marked [ANSWERED] above, halt and report rather than implementing around it.
- **Verify by outcome.** Every real defect this week was found by an outcome check and missed by everything else. When something is green, ask what it actually observed.
- **Name the file you queried and the environment you ran in.** Two decoy databases surfaced in one night.
- Establish capability by the production path, never by a hand-rolled equivalent or the absence of a config key. Both directions — a false negative is a wrong answer in a safe-sounding register.
- No hardcoded hostnames or Tailscale IPs in committed non-markdown files. Never `--no-verify`. Stage explicit paths, never `git add -A`.
- **No self-merge.** CHECKPOINT-HALT before merge. Squash-merge only on explicit per-session authorization, from the main checkout — never a worktree. Check `gh pr list --base` before `--delete-branch`, and remove the worktree after.

---

## Report format

1. Current SHA from the audit preamble.
2. Phase 0 findings with `file:line` citations — especially 0a (second grant construction path) and 0b (provenance), since both change the shape of Phases 1 and 2.
3. What changed, per phase, with the Phase 1 default-narrowing called out as a deliberate behaviour change.
4. The four-kind floor matrix, re-run, as a table.
5. Test results with counts before and after, environment named.
6. Mutation matrix: what was broken, what went red, survivors with reasons, and any mutations withdrawn as inert.
7. Anything found along the way that was not in scope, listed but not fixed.
