# Blackboard protocol, v0 (DRAFT)

**Contract id:** `blackboard/0`
**Status:** DRAFT — not frozen. Will's reviewer reviews this before any pillar code builds on it. Nothing in it is built yet; section 12 says which pieces are open (Prometheus) and which are pillar.
**Part of:** Contracts v1, alongside the daemon API, the hook contract (`hooks.md`) and the memory schema.
**Scope:** how a coordination pillar (Blackboard) and Prometheus agree on board state, roadmap items, task hand-off, merges, acceptance results and run-level events. The pillar decides; open code enforces. This document contains no pillar internals — the coordinator's prompt, policy and classifier are the pillar's own, closed.
**References:** the Blackboard design doc (Decisions D1–D16, Will 2026-09-09 / 2026-11-25); `hooks.md` (the event contract this extends); `PAIRING-APPROVAL-API.md` and `device-scoping.md` (the credential model a worker or client joins through). File paths are relative to the repo root; names, not line numbers.

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

Each item carries, in both `roadmap.md` (table) and `roadmap.json` (generated):

| Field | Rule |
|---|---|
| `id` | **Stable.** Assigned once, never reused, never renumbered. The branch name, task MD, results and events all key on it. Format: `<mission-id>.<n>` (e.g. `auth-rework.4`). |
| `title` | One line, human-readable. |
| `depends_on` | List of item ids. Edges are the merge order: an item is mergeable only when every dependency's post-merge acceptance is green. Cycles are a roadmap error, refused at parse. |
| `acceptance` | **Machine-checkable:** a command (argv or shell line, run on the integrated tree for the post-merge gate) plus an expected exit code (default 0). Prose-only criteria are not acceptance; they belong in the title or the mission text. |
| `status` | Computed, never hand-set: `pending` / `running` / `branch-green` / `merged` / `blocked` / `amended`. The overlay is derived from board state + run events; the roadmap file itself carries no status column (status drifts; the computation is the truth). |
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

### The merge tool (open, off by default)

A daemon tool (`bb_merge`, name provisional) that:

- **requires the intent block as an argument** — the worker's brief statement of what changed and why. The gate rejects the call without it; enforcement is in the tool, not in instructions.
- checks the item's dependencies: every `depends_on` item's post-merge result is green, else refuse with the missing ids named.
- attempts the merge of `bb/<mission>/<item>/<agent>` into the mission's integration branch. **A conflict is returned as a result** (the conflicted paths, the two sides' refs), never resolved with `-X ours/theirs` or any strategy flag. The coordinator escalates it.
- on a clean merge, runs the **post-merge acceptance check**: the item's acceptance command on the INTEGRATED tree (a fresh checkout of the merge result, not the worker's worktree).
- stores the result at `results/merged/<item-id>/<merge-id>.json`, separately from the branch-local result. Emits `merge.result` (section 8).

### The two results

```json
{
  "item_id": "auth-rework.4",
  "run_id": "c3f1a9e0",            // branch-local: the run; post-merge: the merge id
  "kind": "branch" | "merged",
  "tree": "<commit sha the command ran on>",
  "command": "<the acceptance command, verbatim from the roadmap>",
  "exit_code": 0,
  "output_ref": "<path or blob ref to the captured output>",
  "started_at": 0.0, "ended_at": 0.0,
  "intent": { "what": "...", "why": "..." }   // post-merge only: the block the merge tool required
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
| `run.finished` | Prometheus (run end, success path) | `run_id`, `item_id`, `branch_acceptance` (the result record, section 7), `journal_ref` | `intent` (the worker's intent block, `model`) |
| `run.failed` | Prometheus (run end, failure path) | `run_id`, `item_id`, `stop_reason`, `last_journal_ref` | none |
| `worker.escalated` | Prometheus (the worker's decision request reaches the daemon) | `run_id`, `item_id`, `kind_hint` (none — classification is the pillar's) | `request` (the decision request text, `model`) |
| `worker.context_pressure` | Prometheus (the loop's context pre-flight crosses a threshold) | `run_id`, `session_id`, `used`, `size`, `threshold` | none |
| `merge.result` | Prometheus (the merge tool) | `item_id`, `merge_id`, `merged_acceptance` (the result record), `conflict` (null or the conflicted paths) | `intent` (`model`) |
| `mission.amended` | Prometheus (from Beacon, the human's roadmap change) | `mission`, `diff_ref`, `amended_by` (the operator's credential id) | none |

Rules carried over from `hooks/1.0` unchanged:

- **Observe-only.** Every run event's capability set is `[observe]`. A subscription that asks for anything else fails the manifest load (all-or-nothing, `hooks.md` section 12).
- **Delivered over the pillar transport** (`host` socketpair or `unix` socket), never a network address; the daemon authenticates every connection; the API token is never sent to a pillar.
- **Never block the daemon.** Run events are dispatched asynchronously; a missed delivery is a telemetry row with the real outcome, never a stall. A run in progress finishes as it does today when the pillar is missing or crashed; no new tasks are issued; nothing merges (merging requires the coordinator AND the open gate). Beacon shows the mission as paused.
- **A pillar that keeps crashing must not respawn forever** — the backoff and the 5-crashes-in-10-minutes stay-down rule (`hooks.md` section 11) apply; the Instinct seam's unreachable crash-loop guard is the bug this avoids.
- **Trust labels** (`hooks.md` 4.2): facts are daemon-computed; the intent block and the decision request are model-authored (`content`, untrusted); the task MD quotes the roadmap (`external`).
- **Ephemeral sessions:** `content` is sent empty when the run's session is ephemeral, same rule as `hooks/1.0`.

**A pillar's actions go through the daemon API under the gate (D18), never through hook responses.** Starting a run is `POST /api/code` (or its successor); requesting a merge is the merge tool call, authenticated with the pillar's own operator-granted credential, under the gate like any tool call. The hook channel carries events OUT; the API carries actions IN. There is no "modify" or "veto" on a run event.

**Liveness** (candidate, to be checked once the seam exists): for native workers, the daemon's own run state; for outside workers, ACP session state (the background list and idle state). Never stored in git, and fine to lose.

---

## 9. The ACP mapping

One handoff is one ACP session. The protocol core is Prometheus's (`src/prometheus/acp/`, pinned `schema-v2.0.0-beta.4`); this table is the coordination layer's use of it.

| Blackboard | ACP v2 |
|---|---|
| Task MD handed to a worker | `session/new` + `session/prompt` (the task MD is the prompt text) |
| The worker's decision request | The final message of the turn, plus the `stopReason` when the session goes idle |
| Gate escalation | `session/request_permission` (D13: through the same gate; critical never approvable over ACP) |
| Resume after a worker dies (kill-at-60%) | `session/resume` with `replayFrom`; `replay_complete` marks the end of history; live updates held until the marker is out |
| Swarm members shown in any ACP client | Subagent child sessions (a later slice; the mapping is named, not built) |
| Who is working right now | The `background` list and the idle state |
| The coordinator has no edit tools | v2 removed the client's file and terminal access |
| Context hand-off before a worker fills | `usage_update` (`used`/`size`) — the pin carries it; the hand-off keys on it (B8, open) |
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
| Branch-local acceptance capture | **Open** (the coding run already runs acceptance; the result RECORD is the new piece) | not built |
| Post-merge acceptance check, separate result | **Open**, off by default | not built |
| Run-level events (`hooks/1.1`) | **Open** | not built |
| Delegation-chain telemetry columns | **Open**, off by default | not built |
| Task MD generation | **Open** (the composer); the coordinator fills Loop Manager Steps 1–2 | not built |
| Coordinator prompt, policy, diff engine | **Pillar** (closed) | Sprint 2+ |
| Roadmap structure + computed status overlay (Beacon reads through the daemon API) | **Pillar** + beacon-desktop | Sprint 2+ |
| Hand-off before context fills (`agent_loop.py`) | **Open** (B8) | later |
| Outside workers over ACP | **Both**; needs the ACP client side | later |
| Board retention / archival, mission-end rule, board-history ↔ LCM-DAG question | **Open questions** (design doc) | undecided |

**The open/closed line (D4):** anything that can block or change an action, and any "is this safe" or "is this done" decision, is open code in Prometheus. Advisory intelligence may be closed. The merge tool, the acceptance gates, the gate, the event emission, the telemetry — open. The coordinator's judgment — pillar.

---

## 12. Build order (open pieces, one PR each, off by default)

1. **This spec** (docs PR) — Will's reviewer reviews it before any pillar code builds on it.
2. **Run-level events as `hooks/1.1`** — the seven events, observe-only, emission sites, the manifest version range. No pillar transport exists yet; the events fire into the existing hook registry (operator hooks may subscribe; a pillar subscription loads when the seam exists).
3. **Worktree per run** for Loop Manager runs, removed on completion; off → runs in place.
4. **Merge tool** — intent block required, conflict returned as a result, dependency check. Two reviews (it changes what can merge).
5. **Post-merge acceptance check** — re-runs the item's acceptance on the integrated tree, stores the result separately. Two reviews ("is this done").
6. **Delegation chain** — `acting_agent` / `delegated_by` telemetry columns, populated at the call site.

Each: suite failure set unchanged vs baseline, parity replay, overhead with the feature off. The pillar repo and pillar code are Sprint 2, after this spec is reviewed (P6 is Will's action).

---

## 13. Decisions this draft makes (for the reviewer)

- **`roadmap.json` generated from `roadmap.md`**, not a second hand-edited source: one truth, the human-readable file; the machine form is a build artifact.
- **Status is computed, never stored on the roadmap**: a status column drifts; the overlay derives from board state + events.
- **Critical requests over ACP are reject-only** (D13 as decided): the merge tool and the acceptance gates are daemon-side, so a coordinator driving an outside worker never needs the worker to approve anything critical.
- **The task MD generator is open code**, the coordinator's choice of WHICH item is pillar: the composer is mechanical (roadmap + board state), the judgment is advisory.
- **Run events carry the result RECORDS, not summaries**: `branch_acceptance` / `merged_acceptance` are the section 7 JSON, so a consumer never re-derives "done" from prose.
- **No mission concept in ACP**: the protocol maps handoffs to sessions and keeps the mission on the board. Two DAG-over-history systems (board history and LCM's DAG) are an open question, not a v0 commitment.