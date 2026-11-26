# Blackboard protocol, v0 (DRAFT)

**Contract id:** `blackboard/0`
**Status:** DRAFT — not frozen. Will's reviewer reviews this before any pillar code builds on it. Nothing in it is built yet; section 12 says which pieces are open (Prometheus) and which are pillar.
**Part of:** Contracts v1, alongside the daemon API, the hook contract (`hooks.md`) and the memory schema.
**Scope:** how a coordination pillar (Blackboard) and Prometheus agree on board state, roadmap items, task hand-off, merges, acceptance results and run-level events. The pillar decides; open code enforces. This document contains no pillar internals — the coordinator's prompt, policy and classifier are the pillar's own, closed.
**References:** the Blackboard design doc (Will, 2026-09-09 / 2026-11-25; internal, not in this repo — the decisions this draft leans on are inlined where used: D3 Prometheus-without-Blackboard-unchanged, D4 open/closed line, D9 orphan board, D10 stateless coordinator, D11 coordinator-decides-merges + two acceptance gates, D13 ACP approval rule, and **D18: a pillar's actions go through the daemon API under the gate, never through hook responses**); `docs/contracts/hooks.md` (the event contract this extends); `docs/PAIRING-APPROVAL-API.md` and `docs/contracts/device-scoping.md` (the credential model a worker or client joins through). File paths are relative to the repo root; names, not line numbers.

---

## 0. The contract in brief

1. **Prometheus without Blackboard works exactly as it does today** (D3). Every open piece ships off by default; with the feature off, runs stay in place, no board is read, no event is emitted to a pillar.
2. **Board state lives on an orphan branch** (`refs/blackboard/*`), never on `main` (D9). Coordination writes never trigger CI. `git log blackboard` is the decision timeline.
3. **The coordinator decides merges; workers never merge** (D11). The open merge tool enforces the intent block and refuses conflicts instead of resolving them.
4. **Closed advises, open decides** (D4). Anything that can block or change an action, and any "is this safe" or "is this done" decision, is open code in Prometheus. The pillar's intelligence is advisory; its actions go through the daemon API under the gate (D18).
5. **Two acceptance gates, stored separately** (D11): branch-local (the worker's own check) and post-merge (the authoritative check on the integrated tree). Green plus green does not equal green.
6. **Task MDs are built from the roadmap plus board state, never from the previous task MD.** The roadmap is pinned and identical for every worker; everything else is narrow and fetched on demand.
7. **Run-level events** (`hooks/1.1`, section 8) wake the coordinator. They are observe-only for pillars; a pillar never starts a run or requests a merge through a hook response.
8. **One handoff is one ACP session** (section 9). A mission spans many sessions; ACP has no concept of a mission, and this protocol does not pretend otherwise.

---

## 1. Terms

| Term | Meaning |
|---|---|
| **Mission** | One roadmap: an immutable goal text plus a set of items. The mission text is a constitution; it is never edited by the coordinator. |
| **Item** | One unit of work on the roadmap: stable id, dependency edges, machine-checkable acceptance criteria. |
| **Board** | The orphan branch holding mission text, roadmap, interpretations log, task MDs and results. |
| **Run** | One worker execution against one item: a Loop Manager coding run today, an ACP-driven session later. |
| **Worker** | The agent executing a run. Native (a Prometheus coding run / subagent) or outside (an ACP agent Prometheus drives). |
| **Coordinator** | The pillar's diff engine: no edit tools, no loop, stateless (D10). Woken by events, makes one decision, stops. |
| **Interpretation** | A resolved underspecification, appended to the log. Case law: settled once, never argued again. |
| **Amendment** | A change to the roadmap itself. Only the human makes one; the run drops to Supervised. |
| **Intent block** | The worker's brief statement of what it changed and why, required as an argument to the merge tool. |
| **Acceptance result** | The outcome of running an item's acceptance command: exit code, output reference, tree it ran on, when. |

---

## 2. Board layout (orphan branch)

The board is an orphan branch in the repo the mission works on. Default ref: `refs/blackboard/<mission-id>`; the working name `blackboard` is an alias for the single-mission case. Never merged into `main`, never checked out by a worker.

**The ref namespace is custom, and that has consequences this contract states rather than hides.** `refs/blackboard/*` is outside `refs/heads/` and `refs/remotes/`, so:

- `git branch` does not list it; `git for-each-ref refs/blackboard/` does.
- A default `git fetch` / `git push` does not move it. The explicit refspecs are:
  - fetch: `git fetch <remote> '+refs/blackboard/*:refs/blackboard/*'`
  - push (optional, the backup): `git push <remote> 'refs/blackboard/*:refs/blackboard/*'`
- The `blackboard` alias resolves to `refs/blackboard/<mission-id>` for the repo's single mission; a repo with more than one mission has no alias and every reader names the full ref. The alias is a convention of the open reader code, not a git symbolic-ref (a symref would make `git checkout blackboard` possible, which this contract forbids for workers).
- Reads use `git show <ref>:<path>` and `git ls-tree <ref> -- <path>`; neither checks anything out, and both work on a ref a default fetch never moved (the reader fetches the namespace explicitly first).

```
refs/blackboard/<mission-id>
├── mission.md            # immutable goal text (the constitution)
├── roadmap.md            # the item table (section 3); human-readable, machine-parsed
├── roadmap.json          # the SAME items, machine form; generated from roadmap.md, never hand-edited
├── interpretations.md    # append-only log (section 4)
├── tasks/
│   └── <item-id>.md      # the current task MD for an item (section 6)
├── results/
│   ├── branch/<item-id>/<run-id>.json    # branch-local acceptance (section 7)
│   └── merged/<item-id>/<merge-id>.json  # post-merge acceptance (section 7)
└── journal/
    └── <run-id>.md       # the run's journal reference (worker-authored, brief)
```

Rules:

- **Reads never check out.** A worker reads `git show blackboard:roadmap.md` (or the ref's full name). The board is fetched on demand, narrow: the roadmap is the only thing identical for every agent.
- **Writes are commits on the orphan branch**, one logical change per commit, message carries the item id and the actor. Coordination writes never trigger CI (the branch is not `main`, no PR is opened for it; a forge adapter, if ever wanted, is a view for humans, not part of coordination).
- **Pushing the board is optional.** Local-first; a remote board ref is a backup, not a requirement.
- **Append-only where it matters:** `interpretations.md` and the results files are never rewritten; a correction is a new entry that references the old. `roadmap.md` changes only by amendment (section 4).
- **Retention is open** (design doc open question): when a mission's board is archived and what survives in the long-term wiki is undecided; nothing in v0 deletes board history.

---

## 3. Roadmap items

Each item carries, in both `roadmap.md` (table) and `roadmap.json` (generated), the fields below **except `status`, which is computed and stored nowhere on the roadmap** (see its row):

| Field | Rule |
|---|---|
| `id` | **Stable.** Assigned once, never reused, never renumbered. The branch name, task MD, results and events all key on it. Format: `<mission-id>.<n>` (e.g. `auth-rework.4`). |
| `title` | One line, human-readable. |
| `depends_on` | List of item ids. Edges are the merge order: an item is mergeable only when every dependency's post-merge acceptance is green. Cycles are a roadmap error, refused at parse. |
| `acceptance` | **Machine-checkable:** a command (argv or shell line, run on the integrated tree for the post-merge gate) plus an `expected_exit_code` (default 0). Prose-only criteria are not acceptance; they belong in the title or the mission text. The expected exit code is carried into every result record (§7) so "green" is derivable from the record alone. |
| `status` | **Not a stored field.** Computed by open code from board state + run events: `pending` / `running` / `branch-green` / `merged` / `blocked` / `amended`. It appears in the computed overlay (a VIEW of open-computed facts — the §7 records — never a pillar decision; D4 reserves "is this done" for open code). Neither `roadmap.md` nor `roadmap.json` carries a status column: status drifts, the computation is the truth. The pillar and Beacon render the overlay; they do not decide it. |
| `owner_hint` | Optional: the agent kind best suited (coder, critic, …). Advisory; the coordinator decides. |

**The mission text is immutable.** The coordinator can never edit `mission.md`. A roadmap change is an amendment (section 4), made by the human through Beacon; `mission.amended` (section 8) tells the pillar.

**Parsing is open code.** `roadmap.json` is generated from `roadmap.md` by an open generator (one source of truth, the human-readable file); a worker or pillar that wants items reads the JSON. A malformed roadmap is a named error, never a partial parse.

---

## 4. Interpretations and amendments

Workers report back with **decision requests, not status updates**: "Item 4 assumes X; X isn't true; the options are A or B; I recommend B because Y." The coordinator sorts each into:

| Kind | Meaning | What happens |
|---|---|---|
| **Interpretation** | The roadmap was underspecified | The coordinator resolves it, appends to `interpretations.md`, unblocks the worker. |
| **Amendment** | The roadmap itself is wrong | The coordinator stops and escalates to the human with the evidence; the run drops to Supervised. |

`interpretations.md` entries are append-only:

```
## <iso-timestamp> — item <item-id> — <short title>
Question: <the underspecification, as the worker met it>
Resolution: <what was decided>
Rationale: <brief>
Decided-by: coordinator (<pillar name@version>)
Supersedes: <entry ref or none>
```

An interpretation stays resolved once recorded. A later conflict with an earlier interpretation is a NEW entry that references the old — never an edit. The classifier itself (interpretation vs amendment) is pillar code, closed; the LOG FORMAT and the append-only rule are this contract, open.

An amendment changes `roadmap.md` (and regenerates `roadmap.json`) by the human's hand through Beacon; the daemon emits `mission.amended` with the diff reference. The coordinator never writes the roadmap.

---

## 5. Branch naming and worktrees

- **Branch names carry provenance:** `bb/<mission>/<item-id>/<agent>`. The coordination graph can be rebuilt from `git branch` and `git log` alone. `<agent>` is the acting agent's stable identity (section 10), not a model name.
- **Worktree per run** (open piece, off by default): each run gets its own worktree and branch, removed when the run completes. With the feature off, runs stay in place as today (the Loop Manager's single-worker behavior is unchanged). The coding sandbox's full-clone isolation is a SEPARATE mechanism and stays as it is; the worktree piece is for Loop Manager runs in the operator's own repo, where two workers in one checkout break.
- **Conflict prediction is pillar work:** on a timer, diff each live worktree against base (`--name-only`). Two workers on the same file is caught while both run. The open side only exposes the worktree list and the base ref; the diffing and the warning are the pillar's.
- **Merging:** the coordinator decides; the open merge tool (section 7) enforces. Merge order follows `depends_on`. Conflicts escalate through the amendment/interpretation classifier; **never resolved by a merge strategy flag** — the tool returns the conflict as a result.

---

## 6. Task MDs

A task MD is the hand-off to one worker for one item. **Built from the roadmap plus board state, never from the previous task MD** — chaining task MDs is how drift compounds.

The generator (open code) composes:

1. the item's row from `roadmap.json` (id, title, acceptance, dependencies);
2. the mission text (pinned, identical for everyone);
3. the interpretations that touch this item (narrow fetch, by item id);
4. the current board state relevant to the item (dependency results, live-branch list);
5. the acceptance contract verbatim — the command the post-merge gate will run, so the worker knows exactly what "done" means.

The task MD lands at `tasks/<item-id>.md` on the board and is handed to the worker as the run's description (Loop Manager Step 1) with the acceptance as Step 2. For an ACP worker the same content is the `session/prompt` text (section 9).

A task MD is regenerated when the board state it draws from changes (a dependency merges, an interpretation lands). The worker's own journal is separate (`journal/<run-id>.md`), worker-authored, brief — the "why", while the harness records the "what" (section 10).

---

## 7. Acceptance: two gates, two results, one merge tool

### The integration branch

The mission's **integration branch** is `bb/<mission>/integrate`, created by open code from `main` (or the repo's default branch) when the mission starts. It is where worker branches merge, in dependency order; it is NOT `main`. The post-merge acceptance check runs on a fresh checkout of the integration branch's tip AFTER the merge commit. Promoting the integration branch to `main` is a separate, human-gated act (a PR or the operator's own merge) — this protocol never pushes to `main`, exactly as the repo's standing rules require. Who creates it: the mission-start piece (open code, §12); the merge tool refuses to merge into a ref that does not exist.

### The merge tool (open, off by default)

A daemon tool (`bb_merge`, name provisional) that:

- **requires the intent block as an argument** — the worker's brief statement of what changed and why. The gate rejects the call without it; enforcement is in the tool, not in instructions.
- checks the item's dependencies: every `depends_on` item's post-merge result is green, else refuse with the missing ids named.
- attempts the merge of `bb/<mission>/<item>/<agent>` into the mission's integration branch (`bb/<mission>/integrate`, defined above). **A conflict is returned as a result** (the conflicted paths, the two sides' refs), never resolved with `-X ours/theirs` or any strategy flag. The coordinator escalates it.
- on a clean merge, runs the **post-merge acceptance check**: the item's acceptance command on the INTEGRATED tree (a fresh checkout of the integration branch's tip after the merge commit, not the worker's worktree).
- stores the result at `results/merged/<item-id>/<merge-id>.json`, separately from the branch-local result. Emits `merge.result` (section 8).

### Who writes the branch-local result

The **run harness** writes `results/branch/<item-id>/<run-id>.json` at run end — for a native coding run, the same place the run's acceptance command already executes (`coding/session.py`); the new piece is the RECORD on the board, not a second acceptance run. An outside ACP worker reports its own acceptance outcome in its final message; the harness records that report as the branch-local result with `source: "agent-report"` (lower trust, named), never as a daemon-run check. `run.finished` (§8) carries this record; a run that ends without one carries `branch_acceptance: null` and the coordinator treats the item as not branch-green.

### The two results

```json
{
  "item_id": "auth-rework.4",
  "kind": "branch" | "merged",
  "run_id": "c3f1a9e0",            // kind=branch: the run id
  "merge_id": "m7b2e1",            // kind=merged: the merge id (absent on branch results)
  "tree": "<commit sha the command ran on>",
  "command": "<the acceptance command, verbatim from the roadmap>",
  "expected_exit_code": 0,          // from the item's acceptance; green = exit_code == expected_exit_code
  "exit_code": 0,
  "passed": true,                   // daemon-computed: exit_code == expected_exit_code
  "source": "daemon-run" | "agent-report",   // agent-report: an outside worker's own account, lower trust
  "output_ref": "<path or blob ref to the captured output>",
  "started_at": 0.0, "ended_at": 0.0,
  "intent": { "what": "...", "why": "..." }   // merged only: the block the merge tool required
}
```

**Green plus green does not equal green.** The branch-local result says the worker's own tree passed; the post-merge result says the integrated tree passed. They are different trees, different runs, different files. A consumer (the coordinator, Beacon's overlay) reads both; neither overwrites the other. The post-merge result is the authoritative one for "is this item done".

**"Is this done" is open code's decision** (D4): the acceptance command's exit code on the integrated tree, not a model's confidence. The pillar may advise; it never marks an item merged.

---

## 8. Run-level events (`hooks/1.1`)

An additive minor version of the hook contract (`hooks.md` section 12): new events, observe-only for pillars, no new capability on any existing event. The daemon declares `hooks/1.0–1.1`; a pillar that needs run events declares `requires: hooks/1.1`.

| Event | Emitted by | `facts` (daemon-computed) | `content` |
|---|---|---|---|
| `run.started` | Prometheus (run launch) | `mission`, `item_id`, `worker` (acting identity), `branch`, `worktree`, `run_id` | `task` (the task MD text, `external` — it quotes the roadmap) |
| `run.finished` | Prometheus (run end, success path) | `run_id`, `item_id`, `branch_acceptance` (the result record, section 7; null when the run ended without one), `journal_ref` | `intent` (the worker's intent block, `model`) |
| `run.failed` | Prometheus (run end, ANY non-success exit: failure, stop, cancel, timeout) | `run_id`, `item_id`, `stop_reason` (`failed` \| `stopped` \| `cancelled` \| `timeout`), `last_journal_ref` | none |
| `worker.escalated` | Prometheus (the worker's decision request reaches the daemon) | `run_id`, `item_id` (the daemon supplies no classification hint — interpretation vs amendment is the pillar's) | `request` (the decision request text, `model`) |
| `worker.context_pressure` | Prometheus (the loop's context pre-flight crosses a threshold) | `run_id`, `session_id`, `used`, `size`, `threshold` | none |
| `merge.result` | Prometheus (the merge tool) | `item_id`, `merge_id`, `merged_acceptance` (the result record), `conflict` (null or the conflicted paths) | `intent` (`model`) |
| `mission.amended` | Prometheus (from Beacon, the human's roadmap change) | `mission`, `diff_ref`, `amended_by` (the operator's credential id) | none |

**Every exit path emits.** Like `turn_end` in hooks/1.0, a run that is stopped, cancelled or times out emits `run.failed` with the matching `stop_reason` — a coordinator waiting on a run is never left without an event. `run.finished` is the success path only (the branch-local acceptance ran and the run exited 0); a green acceptance on a non-zero run exit is still `run.failed` (the run did not finish cleanly).

**Naming.** The seven names use dots (`run.started`) where hooks/1.0 uses snake_case (`turn_start`). This is deliberate: the dot separates the run-level group from the turn-level group at a glance, and the registry keys on the exact string either way. A minor version may add names; it never renames, so the two conventions coexist without ambiguity. (If Will's reviewer prefers one convention, the rename happens HERE, before anything subscribes — not after.)

Rules carried over from `hooks/1.0` unchanged:

- **Observe-only.** Every run event's capability set is `[observe]`. A subscription that asks for anything else fails the manifest load (all-or-nothing, `hooks.md` section 12).
- **Delivered over the pillar transport** (`host` socketpair or `unix` socket), never a network address; the daemon authenticates every connection; the API token is never sent to a pillar.
- **Never block the daemon.** Run events are dispatched asynchronously; a missed delivery is a telemetry row with the real outcome, never a stall. A run in progress finishes as it does today when the pillar is missing or crashed; no new tasks are issued; nothing merges (merging requires the coordinator AND the open gate). Beacon shows the mission as paused.
- **A pillar that keeps crashing must not respawn forever** — the backoff and the 5-crashes-in-10-minutes stay-down rule (`hooks.md` section 11) apply.
- **Trust labels** (`hooks.md` 4.2): facts are daemon-computed; the intent block and the decision request are model-authored (`content`, untrusted); the task MD quotes the roadmap (`external`).
- **Ephemeral sessions:** `content` is sent empty when the run's session is ephemeral, same rule as `hooks/1.0`.

**A pillar's actions go through the daemon API under the gate (D18), never through hook responses.** The hook channel carries events OUT; the API carries actions IN. There is no "modify" or "veto" on a run event.

**The action path, and its credential tier — stated loudly because it is a permission decision.** Tools are not directly callable by API clients today; a pillar that wants to start a run or request a merge needs a ROUTE, and the route's tier is Will's to decide:

- **Starting a run:** `POST /api/code` is operator-only today (`web/route_access.py`); a scoped token cannot start runs (device-scoping default-deny). The coordinator pillar therefore needs either (a) an **owner-device credential** the operator grants it explicitly (the same tier a Beacon on another machine holds via an owner code, `device-scoping.md` §6.5), or (b) a new route under `SCOPED_ALLOWED` with its own limits. **This draft proposes (a)** — an owner-device credential, granted by the person at the machine, revocable like any device — and flags it for Will's reviewer as the one place the protocol assumes a permission grant. Nothing is widened silently: until the operator grants the credential, the pillar can observe events and advise, and every action route answers 403.
- **Requesting a merge:** the merge tool is a daemon tool; the pillar's path is a thin operator-only route (`POST /api/blackboard/merge`, name provisional) that validates the intent block and calls the same tool code under the gate. Same tier as (a). The route does not exist yet (§12); the draft names it so the tier decision is visible before it is built.
- The pillar's credential is **never the daemon's API token** (hooks.md §11's rule, verbatim). It is a device credential with its own id, so every action's audit row names the pillar that took it.

**Liveness** (candidate, to be checked once the seam exists): for native workers, the daemon's own run state; for outside workers, ACP session state (the background list and idle state). Never stored in git, and fine to lose.

---

## 9. The ACP mapping

One handoff is one ACP session. The protocol core is Prometheus's (`src/prometheus/acp/`, pinned `schema-v2.0.0-beta.4`); this table is the coordination layer's use of it.

| Blackboard | ACP v2 |
|---|---|
| Task MD handed to a worker | `session/new` + `session/prompt` (the task MD is the prompt text) |
| The worker's decision request | The final message of the turn, plus the `stopReason` when the session goes idle |
| Gate escalation | `session/request_permission` (D13: through the same gate; critical never approvable over ACP) |
| Resume after a worker dies | `session/resume` with `replayFrom`; `replay_complete` marks the end of history; live updates held until the marker is out |
| Swarm members shown in any ACP client | Subagent child sessions (a later slice; the mapping is named, not built) |
| Who is working right now | The `background` list and the idle state |
| The coordinator has no edit tools | v2 removed the client's file and terminal access |
| Context hand-off before a worker fills | `usage_update` (`used`/`size`) — the pin carries it; the hand-off keys on it (B8, open) |
| Plan updates | The pin carries `plan`; the loop emits no plan StreamEvent today — **a named gap**, like child sessions and retention, not a v0 commitment |
| One handoff | One session. A mission spans many sessions; ACP has no concept of a mission. |

**Native workers** (Prometheus coding runs, subagents) are recorded by the harness — every tool call, gate decision, cost. **Outside workers** (ACP agents Prometheus drives) report their own account at tool-call level; their permission requests go through the same gate (D13), and anything the gate escalates goes to Beacon. Trust is lower; the record is the agent's own. The harness's own record is what nobody else has.

**Internal subagents stay in-process.** Show them to clients AS ACP child sessions; don't run them THROUGH ACP (D6).

---

## 10. Accountability and the delegation chain

- **What happened** is recorded by the harness, never narrated by the agent: every tool call, file write, gate decision, spawned agent and cost.
- **Why it happened** is written by the agent, briefly (the journal, the intent block).
- **Every tool call carries the acting agent's identity and the delegation chain back to whoever granted it** (open piece, off by default): two nullable telemetry columns on `tool_calls` — `acting_agent` (the identity of the agent whose turn made the call) and `delegated_by` (the identity that granted it: the owner, a parent agent, a coordinator credential). "Who decided what, with whose authority" becomes a telemetry query. NULL on rows from before the feature; never backfilled, never invented.
- **Enforcement is in the tool, not in instructions.** The merge tool requires the intent block as an argument; the gate rejects the call without it. A worker that "forgets" the block cannot merge.
- **Workers pull what they need; nothing is broadcast.** The roadmap is pinned and identical; everything else is narrow and fetched on demand (a tool for "is anyone else touching `auth.py`", "what was decided about retries" — pillar-side queries over board state the open side exposes).

---

## 11. What is open vs. pillar

| Piece | Where | Status |
|---|---|---|
| Board layout, ref convention, read/write rules | **Protocol** (this doc) | v0 draft |
| Roadmap item schema, the generator, the parse | **Open** (Prometheus) | not built |
| Interpretations log format, append-only rule | **Protocol** | v0 draft |
| The interpretation/amendment classifier | **Pillar** (closed) | Sprint 2+ |
| Intent block schema, the merge tool's enforcement | **Open** | not built |
| Merge DECISION (which item, when) | **Pillar** | Sprint 2+ |
| Conflict prediction (the diff timer, the warning) | **Pillar** | Sprint 2+ |
| Worktree per run, removed on completion | **Open**, off by default | not built |
| Branch-local acceptance capture (the RECORD on the board; the run already executes acceptance) | **Open** | not built |
| Post-merge acceptance check, separate result | **Open**, off by default | not built |
| Run-level events (`hooks/1.1`) | **Open** | not built |
| Delegation-chain telemetry columns | **Open**, off by default | not built |
| Task MD generation (the composer) | **Open**; the coordinator fills Loop Manager Steps 1–2 | not built |
| Coordinator prompt, policy, diff engine | **Pillar** (closed) | Sprint 2+ |
| Computed status overlay — a VIEW of open-computed facts (§3 status, §7 records); Beacon renders it through the daemon API | **Open** computes; **pillar + beacon-desktop** render | Sprint 2+ |
| Hand-off before context fills (`agent_loop.py`) | **Open** (B8) | later |
| Outside workers over ACP | **Both**; needs the ACP client side | later |
| Board retention / archival, mission-end rule, board-history ↔ LCM-DAG question | **Open questions** (design doc) | undecided |

**The open/closed line (D4):** anything that can block or change an action, and any "is this safe" or "is this done" decision, is open code in Prometheus. Advisory intelligence may be closed. The merge tool, the acceptance gates, the gate, the event emission, the telemetry — open. The coordinator's judgment — pillar.

---

## 12. Build order (open pieces, one PR each, off by default)

1. **This spec** (docs PR) — Will's reviewer reviews it before any pillar code builds on it.
2. **Run-level events as `hooks/1.1`** — the seven events, observe-only, emission sites, the manifest version range. No pillar transport exists yet; the events fire into the existing hook registry (operator hooks may subscribe; a pillar subscription loads when the seam exists).
3. **Roadmap schema + generator + parse** — `roadmap.md` → `roadmap.json`, the item fields (§3), cycle detection, named parse errors. The mission-start piece creates the integration branch (§7) here.
4. **Task MD generation** — the composer (§6): roadmap + board state → `tasks/<item-id>.md`, regenerated when its inputs change.
5. **Branch-local acceptance capture** — the run harness writes `results/branch/<item-id>/<run-id>.json` at run end (§7); `run.finished` carries it.
6. **Worktree per run** for Loop Manager runs, removed on completion; off → runs in place.
7. **Merge tool** — intent block required, conflict returned as a result, dependency check. Two reviews (it changes what can merge).
8. **Post-merge acceptance check** — re-runs the item's acceptance on the integrated tree, stores the result separately. Two reviews ("is this done").
9. **Delegation chain** — `acting_agent` / `delegated_by` telemetry columns, populated at the call site.

Each: suite failure set unchanged vs baseline, parity replay, overhead with the feature off. The pillar repo and pillar code are Sprint 2, after this spec is reviewed (P6 is Will's action). The pillar's action routes (§8) are built only after Will's reviewer decides the credential tier.

---

## 13. Decisions this draft makes (for the reviewer)

- **`roadmap.json` generated from `roadmap.md`**, not a second hand-edited source: one truth, the human-readable file; the machine form is a build artifact.
- **Status is computed, never stored on the roadmap**: a status column drifts; the overlay derives from board state + events. The overlay is a VIEW of open-computed facts, never a pillar decision (D4).
- **The integration branch is `bb/<mission>/integrate`**, created from `main` at mission start; promotion to `main` stays a human-gated act. The post-merge gate runs on its tip.
- **The pillar's action credential is an owner-device credential** (§8), granted by the person at the machine, revocable — flagged for Will's reviewer as the protocol's one permission assumption. Never the API token.
- **Critical requests over ACP are reject-only** (D13 as decided): the merge tool and the acceptance gates are daemon-side, so a coordinator driving an outside worker never needs the worker to approve anything critical.
- **The task MD generator is open code**, the coordinator's choice of WHICH item is pillar: the composer is mechanical (roadmap + board state), the judgment is advisory.
- **Run events carry the result RECORDS, not summaries**: `branch_acceptance` / `merged_acceptance` are the section 7 JSON with `expected_exit_code` and daemon-computed `passed`, so a consumer never re-derives "done" from prose.
- **Every run exit path emits** (`run.failed` with a `stop_reason` for stop/cancel/timeout), like `turn_end`.
- **No mission concept in ACP**: the protocol maps handoffs to sessions and keeps the mission on the board. Two DAG-over-history systems (board history and LCM's DAG) are an open question, not a v0 commitment.