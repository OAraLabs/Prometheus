# Hook contract, v1

**Status:** v1, frozen 2026-09-24. Drafted, then reviewed; the review decisions are recorded in [section 17](#17-decisions-from-review). From here on, a change follows section 12: an additive change is a minor version, and anything else is a major version.
**Contract id:** `hooks/1.0`
**Part of:** Contracts v1, alongside the daemon API, the blackboard protocol and the memory schema.
**Scope:** How local pillars plug into a turn. Instinct makes fast, bounded decisions such as routing and tool choice. Cognition does open-ended work such as verification and planning. This document contains no code; the typed payload models come with the implementation.
**References:** File paths are relative to `src/prometheus/`, and line numbers are pinned to `origin/main` at `5d62383` (2026-09-24). A bare `:N` means the file named most recently in the same row or paragraph; with no file named, it means `engine/agent_loop.py`. The lines will drift; the names won't.

---

## 0. The contract in brief

1. With no pillar loaded, a turn runs exactly as it does today. Section 16 lists everything that runs and explains why nothing changes.
2. Built-in behavior (the default hooks, [Appendix A](#appendix-a-inventory-of-built-in-behavior)) always runs first, in a fixed order. No pillar replaces the security gate or the adapter.
3. A pillar can only add restrictions: escalate a tool call to the human, refuse it, or stop the run. It can never let through anything the gate blocked or sent for approval.
4. A pillar changes a value in one way only: by answering a **decision point** with an option id from a table that open code built. It may do that only where the default abstains, and only when the operator names that pillar for that point in config. Every decision flag is off by default. A pillar's route choice is limited to the primary and local routes unless the operator allows cloud routes. A local route has a local provider type and an endpoint at a loopback, private-range or tailnet address (section 6.3). The `route` and `tool_choice` flags stay locked until their preconditions are met (section 6.3).
5. Every bounded decision uses one shape: the `Chooser` shape from `computer/chooser.py`. A request carries options and context. The answer is an option id or `abstain`, plus a confidence and the backend's name and version.
6. Hooks never run on the daemon's event loop. The daemon enforces deadlines by stopping its wait, not by trying to cancel work. One session's hook can never delay another session's turn.
7. A hook that times out, errors or returns something invalid leaves the turn on today's behavior. The daemon logs a WARNING and writes a telemetry row with the real outcome. Nothing ever reads as if the hook ran.
8. Pillars run on this machine, in their own processes, never inside the daemon. They talk to the daemon over a socketpair or a unix socket, never a network address. The daemon authenticates every connection before it sends a payload.
9. Installing a pillar means trusting local code with your conversation content. Section 14 says this plainly.

---

## 1. Terms

| Term | Meaning |
|---|---|
| **Turn** | One `run_loop` call (`engine/agent_loop.py:878`): one user message in, one or more rounds, ending when the model answers with no tool calls or the loop halts. |
| **Round** | One model call, plus the tool calls it requests. |
| **Default hook** | Built-in behavior at a stage of the turn. Appendix A lists all of them. Pillars cannot replace, skip or reorder them. |
| **Pillar** | A locally installed package that subscribes hooks. There are two kinds, Instinct and Cognition, and they differ in budget (section 9). |
| **Operator hook** | Today's config-defined hooks: `command`, `prompt`, `http` and `agent` (`hooks/schemas.py`). See section 15. |
| **Event** | A named point where hooks run. |
| **Pipeline point** | An event outside the turn's round structure (memory, learning). |
| **Decision point** | A bounded choice inside an event, answered with the decision shape (section 6). |
| **Capability** | What a subscription may do at an event: observe, annotate, modify one named field, or veto (section 5). |

---

## 2. Principles this contract implements

- **The seams cost nothing.** Every insertion point is a single check against a registry that is frozen at boot. With no subscriber, that check builds no payload and adds no `await` (section 16). Parity is proven by the unchanged suite, identical golden-trace replay and the overhead noise band, not by this document.
- **Prometheus works without the pillars.** Every event has a defined outcome for "no hook", and it is today's behavior.
- **Closed advises, open decides.** Anything that can block or change an action stays in open code. A pillar picks an id; open code built the options, validates the id and runs every downstream default (the adapter's validator, the gate, the approval prompt) on the result.
- **Fail visibly.** Section 9.
- **Pillars run locally.** Sections 11 and 14.
- **Claim only what's measured.** Every latency figure here is a target or a provisional deadline. None of them has been measured.

---

## 3. Events

Four events already exist in `hooks/events.py:16-19`. Only `pre_tool_use` and `post_tool_use` fire today (`engine/agent_loop.py:3902`, `:4633`). `session_start` and `session_end` are defined but nothing in `src/` fires them. This contract keeps all four names and adds six turn events and two pipeline points. New names use `before_`/`after_`; the existing `pre_`/`post_` names stay as they are for compatibility.

| Event | Stage | Class | observe | annotate | modify (via decision point) | veto strengths | Fires |
|---|---|---|---|---|---|---|---|
| `session_start` | session | Cognition | ✓ | — | — | — | Once per new or resumed session |
| `turn_start` | turn | Instinct | ✓ | — | `route` | `halt` | After turn setup, before the first round |
| `before_model_call` | round | Instinct | ✓ | — | `tool_choice` | — | Each round, just before the request is sent |
| `after_model_response` | round | Instinct | ✓ | — | — | `halt` | Each round, after the response is parsed, before it is committed |
| `pre_tool_use` | tool call | Instinct | ✓ | — | — | `escalate`, `veto`, `halt` | Each call the gate allowed or sent for approval, before the approval prompt |
| `post_tool_use` | tool call | Instinct | ✓ | — | — | — | Each executed call, after the built-in post-result steps |
| `after_tool_results` | round | Cognition | ✓ | next request only | — | `halt` | Each round with tool calls, after its results are in history |
| `before_delivery` | turn | Cognition | ✓ | note after the answer | — | — | Once, on the model's final answer, before it is committed |
| `turn_end` | turn | Cognition | ✓ | — | — | — | Once per turn on every exit path, including cancellation |
| `session_end` | session | Cognition | ✓ | — | — | — | When a session is removed, reset or shut down |
| `compaction_candidates` | pipeline (memory) | see below | ✓ | — | `compaction_span` (unreachable in v1) | — | When a compactor has chosen what to compact |
| `skill_draft_score` | pipeline (learning) | Cognition | ✓ | — | `skill_draft` (unreachable in v1) | `veto` (discard the draft) | When a skill draft exists, before it is written |

Class sets the budget (section 9). A subscription that only observes is dispatched asynchronously and never delays the turn, whatever its class.

### 3.1 Where each event fires

"Defaults before" refers to the IDs in Appendix A.

**`session_start`.** Fires when `SessionManager.get_or_create` builds a new `ChatSession` (`engine/session.py:731`), and when `rehydrate_if_cold` (`engine/session.py:739`) resumes one after a restart (`resumed: true`). The CLI and the OpenAI-compatible route mint their own ids (`__main__.py:1711`, `web/openai_api.py:191`); those sites fire it too. Observe only.

**`turn_start`.** Fires in `_run_loop` after the turn-setup defaults T1–T12, which run from `engine/agent_loop.py:925` to `:1473`, and before the round loop at `:1475`. By then the route, the advertised tool catalog, the workspace and the checkpoint are all settled, so the payload describes the turn that will actually run. The `route` decision point is not consulted here. It is consulted inline during T6, at the moment the router abstains (section 6.3).

**`before_model_call`.** Fires once per round, after defaults R1–R9 (microcompaction through the context pre-flight at `:1743-1786`) and before `stream_round_with_fallback` at `:1788`. The `tool_choice` decision point is consulted inline at R6 (`:1559`).

**`after_model_response`.** Fires once per round, after M1–M8: the stream, text tool-call extraction, markup hygiene, the strip-disagreement and empty-response guards, and the forced-tool check that ends at `:2040`. It fires before the commit at `:2059`. It does not fire on rounds those guards already retried or ended.

**`pre_tool_use`.** Fires once per tool call inside `_execute_tool_call`, after C1–C10: operator hooks, adapter repair, markup guard, validation and the security gate, which ends at `:4375`. It fires before the approval prompt at `:4381`, so a pillar veto means the human is never asked. It does not fire for calls the gate denied (there is nothing left to restrict), for calls the repeat guard refused (B3), or for calls rejected earlier in C1–C8. v1 has no decision point here. When adapter repair fails (C3–C4), the model is told and retries, as today (decision 9).

**`post_tool_use`.** Fires once per executed call, after C12–C19, which includes the LSP diagnostics hook (`:4646-4652`). It is the last step before `_execute_tool_call` returns. Observe only.

**`after_tool_results`.** Fires once per round that dispatched tools, after A1–A8, which includes the repeat and divergence halts (`:2477-2620`). If a default already halted the turn, it does not fire. It sits at the bottom of the round-loop body.

**`before_delivery`.** Fires once, when a round ends with non-empty text and no tool calls, which is the model's final answer. It fires at `:2059`, before the answer is appended and yielded at `:2084-2085`. It does not fire for synthetic messages written by Prometheus: breaker trips, the pre-flight refusal, the iteration cap, or halts. Those go straight to `turn_end`.

- **Streaming surfaces.** On web and Beacon the answer has already streamed to the user as deltas (`web/ws_server.py:1253-1262`) by the time this fires. That is why the event cannot withhold the answer, only add a note after it.
- **Telegram.** The honesty correction and teacher escalation run after `run_async` returns (`gateway/telegram.py:2048-2064`). So on Telegram, `before_delivery` is not the last thing that can change the reply. This is a known limit of v1 (decision 16).

**`turn_end`.** Fires from `run_loop`'s `finally` block (`:990`), so it runs on every exit: a normal return, a halt, an error, and the Stop button closing the generator. It is dispatched without awaiting. On a normal finish, the in-loop tail defaults E1–E2 (the boundary-escape check and the file-change summary, `:2160-2200`) have already run by then. On a halt they don't run at all, as today.

**`session_end`.** Fires on `SessionManager.remove` (`engine/session.py:896`, reached from `DELETE /api/sessions/{id}` at `web/server.py:1427` and from purge) with `reason: removed`. It also fires on `SessionManager.clear` (`engine/session.py:891`, from `/reset` and `/clear` on every surface) with `reason: reset`, and on daemon shutdown with `reason: shutdown`, best effort within a 2 s total budget. **There is no idle expiry anywhere.** A session that is simply abandoned never gets `session_end`, and this contract does not invent one.

**`compaction_candidates`.** Fires from two systems, and the `system` field says which:
- **`context_compactor`:** after `_select_span_end` (`context/compactor.py:504`) chooses the span that `apply` (`:631`) will summarize for this request's render view. This happens inside the turn, before the model call.
- **`lcm`:** after `LCMCompactor.compact` (`memory/lcm_compaction.py:62`) has chosen its batches, meaning the uncompacted rows minus the newest 32, in batches of 10 (`:90-99`). This is a background task.

Observe only in v1, so it adds no latency at the in-turn site.

**`skill_draft_score`.** Fires when a skill draft exists, before it is written:
- **`kind: create`:** in `SkillCreator.maybe_create`, after Stage 1 has produced a SKILL.md and before `persist_skill_content` (`learning/skill_creator.py:291`, defined at `:294`).
- **`kind: refine`:** in `SkillRefiner`, before the backup and overwrite (`learning/skill_refiner.py:256-260`).

Both are post-task hooks. They run only on surfaces that go through `AgentLoop.run_async`: Telegram, Slack, Discord, REST `POST /api/chat` and subagents. They do not run on web/Beacon, which calls `run_loop` directly (`web/ws_server.py:1242`). This event inherits that coverage; it does not fix it.

---

## 4. Payloads

### 4.1 Envelope

Every call carries the same envelope, on every transport:

```json
{
  "contract": "hooks/1.0",
  "event": "pre_tool_use",
  "call_id": "8f0c2e6a-…",
  "sent_at": 1790000000.123,
  "deadline_ms": 100,
  "session": { "id": "web:3f2a…", "origin": "user", "ephemeral": false },
  "turn":    { "id": "t-91c…", "round": 2 },
  "facts":   { },
  "content": { },
  "decision_request": null
}
```

- `session.id` is the turn's effective session id. It is for description only: a pillar must never derive authority from it. `session.origin` is computed by the daemon (`permissions/checker.py:244`) and is the only origin a pillar should use.
- `facts` holds values Prometheus computed from its own state or config.
- `content` holds conversation content: anything typed by the person, produced by the model, or read from outside.
- `decision_request` is present only when the call is a decision-point consultation (section 6).

### 4.2 Trust labels

Every field in the per-event schemas carries one source label. The typed payload models carry it as field metadata, and the wire format enforces it by placement.

| Label | Meaning | Where it lives |
|---|---|---|
| `daemon` | Computed by Prometheus from its own state | `facts` |
| `operator` | From the operator's config | `facts` |
| `user` | Typed by the person. Not model-controlled, but may carry pasted third-party text | `content` |
| `model` | Produced by the model. **Untrusted.** | `content` |
| `external` | Tool output, file contents, fetched pages, anything from outside. **Untrusted.** | `content` |

The rule is: when in doubt, it goes in `content`. For example, the gate's `reason` string is daemon-authored but can quote a model-chosen path or command, so it goes in `content`. Content fields are size-capped per event and marked `"truncated": true` when cut.

### 4.3 Per-event fields (v1)

| Event | `facts` | `content` |
|---|---|---|
| `session_start` | `resumed`, `rehydrated_messages`, `workspace` | none |
| `turn_start` | `route {provider, model, reason, backend, decided_by}`, `tool_choice`, `advertised_tools` (names), `profile`, `workspace`, `checkpoint_written` | `user_message` (`user`): the latest human message (`_human_message_from`, `:1025`) |
| `before_model_call` | `round`, `model`, `provider`, `adapter_tier`, `estimated_tokens`, `window`, `window_measured`, `compaction {applied, span_messages}`, `tool_choice`, `steer_pending`, `nudges` | none |
| `after_model_response` | `round`, `served_model`, `usage`, `dropped_malformed`, `stripped_to_empty`, `tool_call_count`, `degraded` | `text` (`model`), `tool_calls [{id, name, input}]` (`model`) |
| `pre_tool_use` | `tool {name, read_only, advertised}`, `repair {name_changed, repairs}`, `gate {action, trust_level}`, `path_is_write`, `workspace_roots` | `tool_input` (`model`, after repair), `raw_call {name, input}` (`model`, as emitted), `gate_reason`, `gate_path` (daemon text that may quote model input) |
| `post_tool_use` | `tool`, `is_error`, `error_type`, `latency_ms`, `truncated`, `repairs` | `tool_input` (`model`), `tool_output` (`external`) |
| `after_tool_results` | `round`, `results [{tool_use_id, name, is_error, blocked_by_repeat_guard}]`, `breaker {all_errors, recovered}`, `tool_iteration` | `results[].input` (`model`), `results[].output` (`external`) |
| `before_delivery` | `rounds`, `tool_iteration`, `served_model`, `degraded` | `final_text` (`model`), `user_message` (`user`) |
| `turn_end` | `outcome` (`completed` \| `halted` \| `error` \| `cancelled`), `reason` (an `_IterationReason` value or `pillar`), `rounds`, `tool_iteration`, `usage`, `served_model`, `applied_calls` (call_ids whose results were applied) | none |
| `session_end` | `reason` (`removed` \| `reset` \| `shutdown`) | none |
| `compaction_candidates` | `system`, `candidate {start, end, messages, est_tokens}`, `threshold`, `reused_anchor` | `span` (mixed `user`/`model`/`external`, each item labeled) |
| `skill_draft_score` | `kind` (`create` \| `refine`), `name`, `default {verdict, reason_code}`, `trace_stats {calls, errors}`, `target` | `draft` (`model`), `task` (`user`), `final_text` (`model`), `trace` (`external`) |

**Ephemeral sessions.** When `session.ephemeral` is true, `content` is sent empty. Decision requests then carry option ids and descriptions but no goal text. The reason: "Prometheus won't remember this" should not depend on a pillar keeping a promise the daemon cannot check (decision 3).

### 4.4 Response

```json
{
  "contract": "hooks/1.0",
  "call_id": "8f0c2e6a-…",
  "hook": "instinct/tool_guard",
  "verdict": "none",
  "reason": null,
  "annotation": null,
  "decision": null
}
```

- `verdict` is one of `none`, `escalate`, `veto` or `halt`, and only a strength the event allows.
- `reason` is required when the verdict is not `none`. It is at most 500 characters and is shown to the person and the model with the pillar's name.
- `annotation` is `{"text": "…"}`, and only where the event allows annotate.
- `decision` is only allowed on a decision-point consultation.

The daemon treats every response as untrusted input. Any of the following makes the whole response **invalid**:
- an unknown top-level key;
- a capability the event doesn't allow;
- a `call_id` that doesn't match;
- a malformed decision.

An invalid response is never partly applied. The daemon discards it, the turn continues on the default, and it logs a WARNING and writes a telemetry row with outcome `invalid`.

---

## 5. Capabilities

| Capability | What it allows | Constraints |
|---|---|---|
| **observe** | Receive the payload. | Cannot affect anything. Observe-only subscriptions are dispatched asynchronously after the event's outcome is fixed, and are never awaited by the turn. |
| **annotate** | Return one text note, which open code places in the event's fixed slot. | Capped at 2,000 characters. Labeled with the pillar's name. Rendered to the model with the untrusted banner (`engine/messages.py:369-406`). Never merged into tool arguments or into trusted prompt sections. The slot per event is fixed (below). |
| **modify(field)** | Change exactly one named field. | Only by answering that field's decision point (section 6). There is no other kind of modification in `hooks/1`. |
| **veto** | Return a restriction verdict. | Strengths: `escalate` (require the human's approval; tool calls only), `veto` (refuse this action), `halt` (stop the run). Each event lists which strengths it allows. |

**Annotation slots**
- `after_tool_results`: appended to the next model call's request-only system addendum. This is the same channel steers, the empty-retry nudge and memory recall already use (`:1504-1541`, `:1410-1427`). It is fenced as untrusted and never persisted.
- `before_delivery`: a separate note committed after the final answer. It is written as an injected message, the way the file-mutation verifier's summary is (`:2187-2191`), not inside the model's own message. The model therefore never reads a pillar's words as its own. The note carries a new provenance value, `pillar`, which is untrusted by default. `pillar` is added to the closed `Provenance` set (`engine/messages.py:32-36`) through the memory schema contract (decision 10).

**Veto semantics.** A veto before an action means the action doesn't happen. A veto after an action stops the run going forward; nothing is undone.
- **`halt` at `turn_start`.** The turn ends before any model call, with the message "Stopped by *pillar*: *reason*. Nothing was sent to the model."
- **`halt` at `after_model_response`.** The response is committed (it is what the model said, and it may already have streamed). Every tool call in it is answered with a `NOT EXECUTED` result, as the iteration cap does (`_unanswered_tool_results`, `:2705`). Then the turn ends.
- **`veto` at `pre_tool_use`.** The call returns an error result naming the pillar, and the round continues.
- **`halt` at `pre_tool_use`.** The call is refused and the turn ends once the round's results are in history.
- **`halt` at `after_tool_results`.** The turn ends going forward, like the repeat-detector halt (`:2477-2517`).
- **Wording.** A halt message must never claim that something which already happened was prevented. This follows the boundary-escape wording (`_boundary_escape_text`, `:2673`).
- **`escalate` takes the gate's approval path.** A pillar's `escalate` is handled exactly like a gate `APPROVE` today, for any origin, system origin included. It uses the loop context's `permission_prompt` (`engine/agent_loop.py:660`; no surface sets it today), else the gate's `request_approval` (`:4381-4384`). That:
  - asks the operator through the approval queue when `security.approval_queue` is enabled and the Telegram adapter is running (`permissions/checker.py:1579`, wired at `daemon.py:1375-1405`);
  - is refused where no approval path exists, the same refusal an `APPROVE` gets today. With no queue, `request_approval` answers no (`permissions/checker.py:1605-1607`, refused at `engine/agent_loop.py:4409-4423`); the shipped config leaves the queue off. With no prompt at all, the call is refused (`engine/agent_loop.py:4424-4438`).

  Only there is an `escalate` refused without the operator being asked. Otherwise the operator's answer decides: a denial, a queue timeout or a queue error refuses it, as it would an `APPROVE` (`permissions/checker.py:1644-1650`). This contract adds no stricter rule for any origin.

**A pillar veto is best-effort.** It applies only if it arrives as a valid answer in time; the next paragraph lists every way it can miss. The security gate is the boundary, and nothing that must be blocked may depend on a pillar.

**A veto that isn't delivered is recorded as not applied.** If a hook's veto doesn't arrive as a valid answer in time:
- **The turn goes on without it.** That hook's verdict counts as `none`, and the turn goes ahead exactly as if that hook weren't installed (section 9.2). For a tool call, the outcome is today's decision (operator hooks, then the gate), raised only by verdicts that other hooks delivered in time (section 7).
- **The row records the miss.** Its `hook_calls` row keeps the real outcome, with no verdict and `applied = 0` (section 13). The outcome is `timeout`, `error`, `invalid`, `busy`, `tripped`, `crashed` or `budget_exhausted`, and the row is never `ok`.
- **A late answer changes nothing.** An answer that arrives after the deadline is discarded and recorded as `late` (section 9.2). It is never applied.

A missed veto is never recorded, displayed or reported as a check that ran and let the action through. A pass and a miss look different in the row: a pass is `ok` with verdict `none`, and a miss is any other outcome with no verdict.

---

## 6. Decisions

### 6.1 One shape

Every bounded decision in `hooks/1` has the shape already used by the decision seam in `computer/chooser.py` and `computer/types.py`. This section defines it once. Every decision point uses it.

**DecisionRequest** (the `decision_request` field of the envelope):

```json
{
  "point": "route",
  "table_id": "rt-7c1…",
  "goal": "…",
  "options": [
    { "id": "primary", "description": "Local model on the primary backend" },
    { "id": "coder",   "description": "Local coding model on the `coder` backend" }
  ],
  "history": [],
  "default": { "status": "abstained", "reason": "no override, escalation, smart-routing or task rule matched" }
}
```

**Decision** (the `decision` field of the response):

```json
{
  "point": "route",
  "table_id": "rt-7c1…",
  "choice": "coder",
  "confidence": 0.82,
  "backend": { "name": "instinct-router", "version": "0.3.1" }
}
```

How it maps onto today's types:

| `hooks/1` | `computer/types.py` today | Note |
|---|---|---|
| `goal` | `ChoiceRequest.goal` | `content`, with the source labeled |
| `table_id` | `ChoiceRequest.snapshot_id` | Identifies the option table. A decision for an old table is **stale**. |
| `options [{id, description}]` | `ChoiceRequest.candidates`, built by `Candidate.chooser_view()` | IDs and descriptions only. Never the action's arguments or anything that could become an action (`computer/types.py`, `chooser_view`). |
| `history` | `ChoiceRequest.history` | Optional |
| `choice` | `Choice.candidate_id` | An option id, or the reserved `abstain` (`CANDIDATE_ABSTAIN`, same spelling). `reobserve` stays specific to computer use. |
| `confidence` | `Choice.confidence` | A number from 0 to 1, or null |
| `backend.name` | `Choice.source` | |
| `backend.version` | none | **New.** The implementation adds it to `Choice`, and `RuleChooser` reports its own version. |

Validation is `validate_choice` (`computer/candidates.py:156`), generalized:
- an empty choice is **invalid**;
- `abstain` means use the default's fallback;
- an id that isn't in the table is **invalid**, a protocol violation;
- a `table_id` that doesn't match is **stale**.

Invalid and stale decisions are never applied. They are logged at WARNING and recorded. The daemon maps a valid id to the option it built itself. Nothing in the decision is merged into an action.

The implementation should move `ChoiceRequest` and `Choice` into one shared module, with computer use as its first client. There should be one type, not two that drift apart.

### 6.2 Authority regions

This follows the 2026-09-20 ruling:

- **The default answers first and wins wherever it answers.** In that region no pillar is consulted for the decision. A pillar may still observe the event, and may veto where the event allows it.
- **A pillar decides only where the default abstains**, and only when `pillars.decide.<point>` names that pillar. Every point is off by default, and at most one pillar may be named per point. `tool_choice` is also locked off until the parity harness has measured its cost, and `route` until pillar-picked connections are pinned (section 6.3).
- If the pillar abstains, times out, errors, answers invalid or stale, is busy or is tripped, the result is **the default's own fallback, which is today's behavior** in that region.
- **A timeout limits delay. It is not what keeps things safe**, because a fast wrong answer passes a timeout. What keeps a pillar's decision safe:
  1. open code builds the option table;
  2. the returned id is validated against that table;
  3. every downstream default still runs: each tool call the model makes afterwards still goes through the adapter, the security gate and the approval prompt;
  4. the operator turned the point on;
  5. for `route`, the table holds only the primary and local routes (section 6.3) unless the operator allowed cloud routes (decision 17).

### 6.3 Decision points in v1

| Point | Event | Options (built by) | Default | Default abstains when | Fallback |
|---|---|---|---|---|---|
| `route` | `turn_start` | The primary, plus **local routes** as defined below. Any other route is included only with `pillars.decide.route_allow_cloud: true` (router) | `ModelRouter.route` (`router/model_router.py:747`) | The router reaches its primary branch (`:791-792`) because no per-session override, escalation, smart-routing or task rule answered | The primary, as today |
| `tool_choice` | `before_model_call` | `auto`, `none`, `required`, and `tool:<name>` for each advertised tool (loop) | The caller's `tool_choice`, or the one resolved from `mode` (`engine/agent_loop.py:1076`), with first-round forcing (`:1086-1091`, `:1559-1561`) | The round's directive is `auto` and came from `mode`, not from the caller | `auto`, as today |
| `compaction_span` | `compaction_candidates` | Cut points the compactor considered | `_select_span_end` (`context/compactor.py:504`) | **Never, today** | — |
| `skill_draft` | `skill_draft_score` | `keep`, `discard` | SkillCreator Stage 0/1 plus the name-collision check (`learning/skill_creator.py:214-291`) | **Never, today** | — |

Constraints that apply to specific points:

- **`route`: the person's choice wins.** A per-session user override (`/claude`, `/local`, …) always answers, so a pillar can never overrule the person's own choice of model.
- **`route`: local unless the operator opts in** (decision 17). A pillar's choice must never send conversation content to a hosted API, or spend money, without the operator opting in. So the option table holds:
  - the primary, which is where the turn goes anyway when the pillar abstains. If the primary is a hosted API, picking it changes nothing: the operator already sends every unrouted turn there;
  - every route that is **local**, as defined below;
  - any other route, **only** when `pillars.decide.route_allow_cloud` is `true`. The default is `false`.

**What counts as a local route** (decision 17, amended in review). A route is local only when **both** of these hold:

1. **Its provider type is local.** Types today: `llama_cpp`, `ollama`, `lm_studio` and `vllm` (`providers/registry.py:140`, `:167`).
2. **Its endpoint resolves only to local addresses.** "Endpoint" means the host of the base URL that the serving provider will actually use, resolved and reached directly (details below). It is not the text in the route's config.

   | Kind | Ranges |
   |---|---|
   | Loopback | `127.0.0.0/8`, `::1` |
   | Private range (RFC 1918) | `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` |
   | Link-local | `169.254.0.0/16`, `fe80::/10` |
   | IPv6 unique local (ULA) | `fc00::/7` |
   | Tailnet: RFC 6598 shared address space, which Tailscale assigns from | `100.64.0.0/10`, `fd7a:115c:a1e0::/48` (the second is inside ULA already) |

   An address in `100.64.0.0/10` does not prove a tailnet peer: carrier-grade NAT uses the same block.

Which address gets classified:
- **The base URL the serving provider will use.** For `lm_studio` and `vllm`, and any route with `base_url_env`, that comes from `_resolve_base_url` (`providers/registry.py:183-219`, used at `:392`): the config `base_url`, then the environment (`base_url_env`, `VLLM_BASE_URL`, `LM_STUDIO_BASE_URL`), then the default. A classifier that read only the config would call a `vllm` route with `VLLM_BASE_URL` set to an internet host local.
- **The provider instance that will actually serve the route.** Today the router caches task-rule providers by `provider:model` alone (`router/model_router.py:1054`). So two rules that differ only in `base_url` share whichever instance was built first. Before the implementation offers such a route, it must key that cache by endpoint or build its own instances.
- **Directly, never through a proxy.**
  - Provider and probe clients are httpx clients with the default `trust_env=True` (`providers/llama_cpp.py:812`, `providers/ollama.py:134`, `providers/openai_compat.py:287`, `providers/backends.py:543`). So today, an environment proxy that applies to a route's URL carries even a loopback request, and the proxy resolves the name itself. Environment proxies here means `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` or the macOS system proxy.
  - Classification and pillar-picked requests therefore connect directly to the resolved address and ignore environment proxies. That is part of what pinning (below) means.
  - Requests the default routes still go through the proxy, as today.

How addresses are judged:
- An IPv4-mapped IPv6 address (`::ffff:a.b.c.d`) is judged by the IPv4 address inside it, the unwrapping `security/url_guard.py` already does (`:125-127`).
- A hostname is resolved, and **every** address it returns must be in the table. One address outside it makes the route cloud.
- Anything else is **cloud**. That includes an address that can't be resolved, a resolution that fails or times out, and a provider type that isn't in the local list.
- These ranges are an explicit list, **not** "any address that isn't publicly routable". `url_guard.is_blocked_address` (`security/url_guard.py:107`) answers that wider question. It counts reserved, multicast, unspecified and the rest of Python's `is_private`, and it counts the tailnet only when `block_tailnet` is set, which is off by default (`:104`, `:139-140`). It is the precedent for how to check an address, not the definition of local.

**Ollama cloud models.** Ollama can serve cloud-hosted models through a local `ollama` server: the request goes to the local address, and Ollama forwards it to ollama.com. Checked against Ollama's own source, it identifies these models in two ways:
- **By name** (Ollama 0.18 and later, `internal/modelref/modelref.go`).
  - A model name whose last `:`-separated part is `cloud` or ends in `-cloud`, compared case-insensitively, is a cloud model. Examples: `gemma4:cloud`, `gpt-oss:120b-cloud`, `gpt-oss:20b:cloud`.
  - For such a name, the local server answers `/api/show` and chat calls by proxying them to ollama.com, with no marker in the response.
- **By manifest** (Ollama 0.12 and later, `api/types.go`).
  - A model whose local manifest records a remote host, such as a pulled cloud model or a copy of one under another name, carries non-empty `remote_host` and `remote_model` in `/api/tags` and `/api/show` (`ListModelResponse` and `ShowResponse`, both `omitempty`).
  - Native `/api/chat` and `/api/generate` responses carry them too. `/v1/*` responses and `/api/ps` never do.

So an `ollama` route is **cloud** when, for the model it will send, any of these is true:
- **It has a cloud name.** The name is checked first, and a cloud-named model is cloud without a call to `/api/show`, so re-classification never reaches ollama.com.
- **Ollama reports it as remote.** `/api/show` or `/api/tags` reports a non-empty `remote_host` or `remote_model`.
- **It can't be checked,** because `/api/show` fails or no model is known.

Which model is checked, the one the route will actually send:
- **For a named backend,** the configured `model`, else the probe's pick: loaded first, then first pulled (`providers/backends.py:594-597`). The router copies that pick into the route (`router/model_router.py:538-539`).
- **For an `ollama` primary,** `model.model` as sent (`daemon.py:699`). The router doesn't copy the probe's pick for the primary (`router/model_router.py:513-515`), so the classification checks the name that is sent.

A pinned pillar-picked request sends that model and no other. Today's probe calls `/api/tags`, `/api/ps` and `/api/show` (`providers/backends.py:582-617`) but reads neither field; the implementation adds the check.

What can't be detected:
- **Anything Ollama doesn't report.** A modified Ollama, or another server that answers the Ollama API, can forward upstream without these markers and so looks local. Ollamas older than 0.12 have neither cloud models nor these fields.
- **A change between classifications.** The daemon's Ollama provider uses `/v1/chat/completions` (`providers/ollama.py:123`), whose responses carry no remote marker. A model that becomes remote between classifications isn't seen until the next re-classification. One example is `ollama cp` of a cloud model onto a local name.
- **The server's cloud setting, on older servers.** On Ollama 0.17 and later, `GET /api/status` reports whether cloud features are disabled (`cloud.disabled`, `cloud.source`). The endpoint is marked experimental. Doctor shows it where present, but it doesn't replace the per-model checks. An older server, or one that isn't Ollama, can't say.

Operator levers at the source:
- Setting `OLLAMA_NO_CLOUD=1`, or `disable_ollama_cloud` in `~/.ollama/server.json`, and restarting Ollama rules out its cloud models.
- On Ollama 0.18 and later, a model name ending in `:local` makes the server refuse a model that would be served remotely.

**When a route is classified.** Only while `pillars.decide.route` is on, meaning it names a loaded pillar and its lock is lifted.
- **With the flag off, or locked as it is in v1, nothing is classified.** There is no resolution, no probe and no row, and doctor shows "not classified: route flag off" (or locked).
- **While the flag is on, when it takes effect at load,** every candidate route is classified. A candidate is any route with a local provider type.
- **In the background, once per TTL** (60 s by default, `providers/backends.py:108`), every candidate route is re-classified, whatever its last verdict.
  - A route marked cloud by a transient failure or by staleness therefore comes back on the next tick.
  - Registry backends are re-probed, and other routes are re-resolved. For `ollama` routes this includes the model check above.
- **The background work keeps its own state.** It doesn't write the backend registry's cached status. The default routing, the compactor's per-backend windows (`context/compactor.py:339`, `:433`), and the model and vision fill-in for backend overrides (`router/model_router.py:536-541`) therefore see exactly what they see today.
- **Today's on-demand probes are unchanged and don't classify.** Those are boot, the `/backends` command, the web Backends view, Anatomy, and switching to a backend with `/<name>` (`providers/backends.py:357-361`, `:392-398`, `gateway/commands.py:590`).
- **A classification older than twice the TTL counts as cloud.** The route leaves the option table until a fresh classification brings it back. A stalled background task therefore fails closed.
- **Never per turn, and never on the turn path.**
- **Each classification writes a `subsystem_runs` row** (`subsystem: pillars`, `operation: classify_route`), which doctor reads (section 13). The row holds:
  - the route, the verdict and a reason code;
  - the addresses it judged and pinned;
  - whether an environment proxy applies to the route's URL in the daemon's environment;
  - for `ollama` routes, the server's `/api/status` cloud setting, where reported;
  - the time.

**Pinning.** Pinning applies to routes offered because they are local.
- **What happens.** A pillar-picked request to such a route connects directly to the address that was classified, never to a fresh DNS answer and never through an environment proxy. It keeps the hostname for the `Host` header and the TLS server name.
- **Why.** Today every model request opens a new client and resolves the name again (`providers/llama_cpp.py:812`, `providers/ollama.py:134`, `providers/openai_compat.py:287`). Without a pin, a hostname whose answers vary (round-robin, split-horizon, a short TTL) could reach an address no classification saw.
- **Names are fine.** Tailnet names are normal and allowed; no IP literal is required.
- **When the answer changes.** If the background re-classification sees a new answer, the pin moves to it when it is local, and the route leaves the table when it isn't.
- **What isn't pinned.** Picking the primary runs exactly as the default would, unpinned. A route offered only under `route_allow_cloud` connects as it does today. Requests that the default routes are unchanged.

**The route flag is locked until pinning exists.** Like `tool_choice`, `pillars.decide.route` can't be turned on anywhere until pinning is implemented. The implementation ships it locked:
- a config that sets it loads with the flag off and logs one WARNING;
- doctor shows the flag's state as "locked: connection pinning not implemented" whenever the lock is in force, and ✗ when the config also sets it.

Lifting the lock is an edit to this section that records that pinning is in place.

**What the check can't see.** It classifies the address the daemon connects to. It does not see where the content goes after that, so these pass as local:
- a local port that forwards somewhere else: an SSH tunnel, `socat`, a port-forward;
- a local gateway, such as LiteLLM, in front of a hosted API;
- a local Ollama forwarding a model upstream that Ollama doesn't mark (above);
- a machine shared into your tailnet from someone else's;
- a carrier-NAT address in `100.64.0.0/10`.

Knowing what listens behind a local route's address is the operator's job.

**Recovery from a failed pick.** When a pillar-picked route fails, the turn goes first to the route it would have taken without the pillar: the primary. Today's recovery then applies from there, unchanged.

"Fails" means any of these:
- **The round raises before any output has streamed,** whatever the error kind. That includes connection failures and timeouts that today end the turn: today only `auth` and `billing` count as terminal (`engine/fallback.py:37`).
- **The context pre-flight refuses** the picked model's window (`engine/agent_loop.py:1743-1786`).
- **The circuit breaker trips,** for any reason (`:2343-2447`). That includes trips that today's loop would end with a diagnostic.
- **A tool call would be escalated** to `router.escalation` (`:4037`). Instead, the call gets today's retry prompt on the primary.
- **On Telegram, the reply trips the teacher's failure detector.** Wherever today's teacher escalation would run the detector for the primary, it runs on the picked turn, with the same inputs today's call builds: the turn's trace and its reply text (`gateway/telegram.py:2042`, `:2152-2153`; `escalation/detector.py:154`). "Wherever" means a Telegram adapter with tools registered, a primary that `is_cloud()` doesn't name as cloud, and `escalation.teacher_model` set (`gateway/telegram.py:2154`, `escalation/teacher.py:500-522`). A trip is a failed pick.

A pick of the primary is never a failed pick. That turn runs, and is judged, as a default turn.

The move to the primary:
- **Fresh recovery state.** The primary gets a fresh circuit breaker, so today's model switch and one-shot diagnose-and-recover (`:1429`, `:2388`) apply to it in full.
- **Prompt and catalog.** The identity line is rewritten for the primary, as today's fallback does. After a first-round failure nothing has been sent yet, so the tool catalog is re-resolved for the primary. After a later failure, the run keeps its frozen catalog, as today's circuit-breaker switch does (`:2375-2377`).
- **The rest of the turn stays on the primary.**
- **After a teacher-detector trip,** the primary re-answers the turn, carrying the turn so far. Telegram's post-turn steps then run on the primary's answer exactly as they would for a primary-served turn: the honesty check, then teacher escalation, with its gate judging the primary (`gateway/telegram.py:2048-2070`). The picked reply is never delivered: Telegram sends only after `_run_agent_turn` returns (`gateway/telegram.py:2085`), for a user message (`:2316`) and for an injected turn (`:2223`). Section 18.3 lists what is still open about this re-answer.

What this guarantees, and what it doesn't:
- **Recovery adds no destination.** After a failed pick, the turn goes only to the primary and, from there, to the operator's configured targets: `model.fallback` (`engine/fallback.py:201`, applied at `engine/agent_loop.py:1788`), `router.fallback`, `router.escalation`, and on Telegram the teacher. Today's routing and recovery already use those destinations.
- **The worst case before any output has streamed** is today's path plus one failed attempt.
- **A later failure carries the turn so far.** If the pick fails after its first round, what moves to the primary, and on to those targets, includes the picked model's tool calls and their results. Their side effects aren't undone.
- **Once output has streamed, a round isn't moved** (`engine/fallback.py:68`, `decide`), the same rule as today's fallback. A pillar-picked route that fails mid-reply ends the turn as a failed turn, where the primary might have succeeded.
- **The failed attempt itself** is bounded only by the first-hop check above.

**Why the endpoint test is needed: today's code decides "local" by provider type or name, never by where the endpoint is.** A `llama_cpp` provider pointed at someone else's server passes as local at every site below, and a `vllm` one at every site whose list includes it:

| Site | What it decides, by name only |
|---|---|
| `providers/registry.py:140`, `:167` | `_LOCAL_OPENAI_COMPAT_PROVIDERS` and `_LOCAL_PROVIDERS` ("providers that serve from a box you own"). The base URL is whatever config or the environment says (`:147-156`, `:183-219`). |
| `providers/registry.py:467-469` | `is_cloud()`, "costs money" |
| `providers/backends.py:95`, `:233`, `:307` | A named backend, and a primary that enters the registry, must be `llama_cpp` or `ollama`. A backend's `base_url` is checked only for an `http(s)` scheme (`:239-240`), so any host passes. |
| `router/model_router.py:447-450`, `:486-494` | Whether an override is a cloud preset or a local backend depends on where it came from. `slash_commands.<name>` can replace a preset's provider and `base_url` (`:470-471`, `:481-482`). |
| `router/model_router.py:1181-1195`, `__main__.py:440` | The adapter tier. Tier `off` is the runtime "is cloud" signal (`engine/agent_loop.py:857`, `:3272`, `context/dynamic_tools.py:235`). |
| `engine/fallback.py:242` | The fallback's `is_local_backend` is `provider_name in ("llama_cpp", "ollama")` |
| `telemetry/tracker.py:50` | `_CLOUD_PROVIDERS` labels telemetry and training data (read at `:897` and `engine/agent_loop.py:2984`). `telemetry/tracker.py:65` (`_LOCAL_PROVIDERS`) is a name list too, read only by drift-guard tests. |

This contract changes none of those. It defines local for the `route` decision point only, and the implementation provides that classification.
- **`tool_choice`: locked until measured.** This flag can't be turned on anywhere, on any install, until the parity harness has measured the llama.cpp prefix-cache cost of a forced round. A forced round withholds native tools so the grammar path fires (`:1572-1579`), and that changes the cached prompt prefix. The implementation ships the flag locked: a config that sets it loads with the flag off, logs one WARNING, and doctor shows ✗ naming the missing measurement. Lifting the lock is an edit to this section that records the measurement.
- **`tool_choice`: one round only.** A pillar-set directive binds one round, and a pillar may set at most one round per turn. This mirrors first-round forcing.
- **`tool_choice`: forcing must never raise.** Today a forced `{tool: X}` that the provider doesn't honor raises (`:2032-2040`). A round forced by a pillar must instead record the outcome as `not_honored` and continue.
- **No `tool_name` point in v1** (decision 9). When adapter repair fails today, the model is told and retries (`:3988-4048`). That is not a dead end. So a pillar-chosen tool name would save at most one round, at the cost of running a tool the model didn't name. Section 18 says what evidence would reopen it.
- **`compaction_span` and `skill_draft` are defined but unreachable.** Their defaults always answer today. They become reachable only if a default gains an abstain band. That is a change to default behavior and needs a separate change of its own (decision 12). In v1, `skill_draft_score` offers `veto` (discard) instead, which is a restriction and needs no decision.

---

## 7. Authority

**The order of restrictions:**

```
none  <  escalate (require approval)  <  veto (refuse this action)  <  halt (stop the run)
```

For a tool call, the effective outcome is the **most restrictive** of the gate's decision and every pillar verdict:
- If the gate says `DENY`, the call is refused and `pre_tool_use` doesn't fire.
- If the gate says `APPROVE`, pillars may raise it to `veto` or `halt`. They cannot lower it to "allow".
- If the gate says `ALLOW`, pillars may raise it to `escalate`, `veto` or `halt`.

Nothing in the protocol can express a decrease.

**Defaults no pillar replaces.** The security gate (`permissions/checker.py:697`, called at `engine/agent_loop.py:4302`) and the adapter (`adapter/__init__.py:48`, applied at `engine/agent_loop.py:3948`, `:1856`, `:1399`) always run, in their positions, on every call. The dispatcher has no API that disables, skips, reorders or wraps a default hook. After a pillar decision, every tool call the model makes still goes through the adapter and the gate.

**What a pillar can never do in `hooks/1`:**
- allow something the gate denied, or skip the approval prompt;
- change a tool call's arguments, or which tool runs;
- pick any route but the primary or a **local route**, unless the operator set `pillars.decide.route_allow_cloud`. A local route has a local provider type **and** an endpoint that resolved only to addresses in the section 6.3 table (loopback, private-range (RFC 1918, link-local, IPv6 ULA) or tailnet). It was checked directly, not through a proxy, in a classification no older than twice the TTL. A local provider type pointed at someone else's server is not local. Neither is an Ollama model that is cloud-named, reported as served upstream, or can't be checked. A pillar-picked request to a local route connects directly to the pinned, classified address. If the picked route fails, the turn goes to the primary first, then today's recovery (section 6.3);
- add a tool to the advertised catalog;
- widen the workspace or write boundary;
- change `origin`;
- write to history, except through the labeled annotation slots;
- write to memory, LCM or the skills directory directly. Those belong to the memory schema and blackboard contracts;
- call the daemon's model provider. Pillars bring their own compute.

**A pillar can stop a run or escalate** (section 5). Both are restrictions.

---

## 8. Ordering

At each event:

1. **Default hooks for the stage run first, in their declared order** (Appendix A, top to bottom within each stage). A decision point is consulted inline, at the moment its default abstains.
2. **Operator hooks run at their current positions.** In v1 these are unchanged: `pre_tool_use` operator hooks run first in `_execute_tool_call` (C1, before the adapter and the gate), and `post_tool_use` operator hooks run at C18. See section 15.
3. **Pillar subscriptions that can affect the outcome** (veto, annotate) are dispatched together and share the event's deadline. They run concurrently, so waiting costs the slowest hook, not the sum of all of them. Their results combine in **declared order**:
   - pillars in the order of `pillars.load`, then subscriptions in manifest order;
   - the verdict is the most restrictive one;
   - the reason shown is the first in declared order among the hooks that returned the winning verdict;
   - annotations are concatenated in declared order, within the slot's cap.
4. **Observe-only subscriptions** are enqueued after the event's outcome is fixed. They are never awaited.

Declared order is fixed at boot. Nothing reorders it at runtime.

---

## 9. Latency, deadlines and failure

### 9.1 Budgets (provisional, not measured)

These numbers are adopted as provisional (decision 2). The implementation re-sets them from measurements.

| Class | Target | Enforced deadline | Other bound |
|---|---|---|---|
| Instinct | p95 ≤ 50 ms per hook call, measured by the daemon from dispatch to result, transport included | 100 ms default, configurable up to 250 ms | — |
| Cognition | — | 5 s per call default, configurable up to 30 s | 15 s per turn: the total a turn waits on Cognition. Once it is spent, the rest of that turn's Cognition calls are skipped (`budget_exhausted`). |
| Pipeline (`skill_draft_score`, `compaction_candidates` at `lcm`) | — | 60 s per call | Meant to stay off the person's path, but **today `skill_draft_score` is not off it on every surface.** Post-task hooks run before `run_async` returns (Appendix B.8), so on Telegram, Slack, Discord and REST a veto subscription there would add up to this deadline to the reply. This is a known limit of v1 (decision 16). |

A hook over its target is reported by `oara doctor` as slow (a warning). It is not an error. The deadline is what the daemon enforces.

### 9.2 When a hook doesn't produce a usable answer

If the outcome is `timeout`, `error`, `invalid`, `stale`, `busy`, `tripped`, `crashed` or `budget_exhausted`:

1. The turn continues with **today's behavior** for that event, exactly as if the hook weren't installed.
2. The daemon logs **one WARNING**, with no conversation content: `hook <pillar>/<hook> event=<event> outcome=<outcome> after <n> ms — continuing with default behavior`
3. The daemon writes **one telemetry row** with that outcome (section 13). `abstain` is a valid answer: it writes a row but no WARNING.

It must never read as if the hook ran:
- no row with outcome `ok` for a call that didn't complete;
- no verdict, annotation or decision applied from such a call;
- no text shown to the person or the model implying the check happened.

A surface that shows check status (for example "verified") must read it from the row, and any outcome other than `ok` must display as "not checked". A result that arrives after its deadline is discarded and recorded as `late`. It is never applied to a later event.

---

## 10. Execution: never block the daemon

- **No pillar code runs in the daemon process.** Every pillar in v1 runs in its own process: a `host` subprocess the daemon supervises, or a `unix` service (section 11).
- **The event loop never waits on a pillar synchronously.** It sends the call over the pillar's transport and awaits the reply with a deadline.
- **Deadlines are enforced by abandoning the wait, not by cancelling the work.** `asyncio.wait_for` cancels the awaiting coroutine; it can't stop a forward pass running in another process. So:
  - a timed-out call keeps counting against the pillar's in-flight caps (below) until its late reply arrives or the connection is reset;
  - a `host` process can be stopped: after 3 consecutive timeouts the daemon restarts it (SIGTERM, then SIGKILL);
  - a `unix` service isn't the daemon's to restart. Its timeouts trip the circuit breaker instead.
- **One session never stalls another:**
  - No pillar call holds or waits on any session's turn lock (`web/ws_server.py:1182`, `gateway/telegram.py:1975`).
  - Each session has its own in-flight cap per pillar (default 2), under a global cap per pillar. A call over either cap is **not queued**; it is skipped at once with outcome `busy`.
  - Calls are admitted or refused at dispatch; there is no queue shared across sessions. A slow hook can cost its own session the pillar's help, but it cannot delay anyone else's turn.
  - Observer queues are bounded, one per pillar. On overflow, events are dropped. Each drop is counted in telemetry (`dropped`), and the WARNING for drops is rate-limited.
- **Circuit breaker per pillar.** After 5 non-`ok`, non-`abstain` outcomes within 60 s, the pillar is **tripped**. Its calls are skipped (`tripped`) for a cooldown of 5 minutes. The daemon logs one WARNING when it trips and one when it resets, and doctor shows it.
- **No shared interpreter.** Because pillar code never runs in the daemon's process, a CPU-bound pillar competes with the daemon only for CPU, which the caps bound where they are enforced. It never competes for the daemon's GIL.

---

## 11. Transports and isolation

v1 has exactly two transports. Both use one payload schema (sections 4 and 6), as UTF-8 JSON; only the framing differs. Either kind of pillar may use either transport. `kind` only sets the default.

| Transport | For | How it runs | Isolation |
|---|---|---|---|
| `host` (**default** for Instinct pillars) | A Python package exposing an entry point in the `prometheus.pillars` group | The daemon starts one supervised host subprocess per loaded pillar and loads the entry point **there**, never in the daemon. They talk over a socketpair with length-prefixed JSON. | A separate process. Caps apply (below). A crash cannot take the daemon down. |
| `unix` (**default** for Cognition pillars) | A pillar that runs as its own local service | HTTP/1.1 framing over a unix socket at `~/.prometheus/run/pillars/<name>.sock`, as `POST /hooks/1/<event>`. There is no TCP listener. | A separate process that someone else launches |

**Not in v1** (decisions 1 and 13):
- **`inproc`.** Loading a pillar into the daemon's own process is removed from v1 entirely: it can't meet "a pillar crash never takes the daemon down". Entry points remain the way Instinct pillars are packaged and discovered, and `host` is how they run. `inproc` may return in a later minor version only if the measured `host` overhead misses the Instinct budget (section 18).
- **`http` on loopback.** Not in v1. It gets added only when a real pillar can't use a unix socket (section 18).

**Authentication.** The daemon authenticates every connection before it sends any payload.
- **`host`:** the daemon creates the socketpair before it starts the host, and the host inherits one end. There is no listening address, so there is nothing for another process to connect to or squat.
- **`unix`:** the socket directory is mode 0700 and the socket 0600, owned by the daemon's user. The daemon checks that the peer's uid matches its own (`SO_PEERCRED` on Linux, `getpeereid` on macOS).
- **The daemon's API token is never sent to a pillar** (decision 14). Sending it over any local connection would hand the daemon's master credential to whatever is on the other end.
- **Pillar-to-daemon calls.** A pillar that calls the daemon (for example to post an asynchronous Cognition result) is an ordinary daemon API client. It authenticates with a token the operator gives it, under the daemon API contract.
- **The plan for when `http` is added.** It will be loopback only: `127.0.0.1` or `::1`, refusing any address that resolves elsewhere. A per-pillar secret will live at `~/.prometheus/pillars/<name>.secret` (0600), created when the pillar is enabled. On each connection, both sides answer the other's nonce with HMAC-SHA256 over that secret before any payload moves. The secret never crosses the wire, so a process squatting on the port while the pillar is down learns nothing. That minor version adopts this design, or states why it doesn't.

**Resource caps, for daemon-launched processes.**
- **Linux with a systemd user manager:** the host is launched in a transient user scope with `CPUQuota=`, `MemoryMax=` and `IPAddressDeny=any` / `IPAddressAllow=localhost`.
- **Linux without systemd:** `setrlimit(RLIMIT_AS)` and a nice value. There is no CPU quota.
- **macOS:** no per-process CPU quota exists, and `RLIMIT_AS` is not reliably enforced. Caps are **not enforced** on macOS, and doctor says exactly that.
- **Pillars running as their own services** (`unix`) are capped by whoever launches them, and doctor reports "caps: set outside Prometheus".
- A cap that is configured but not enforced is always shown as **not enforced**, never as "capped".

**Crashes.** When a host process exits:
- its in-flight calls resolve as `crashed`, falling back to the default with a WARNING and a row;
- the daemon restarts it with backoff (1 s, 2 s, 4 s … up to 60 s);
- after 5 crashes in 10 minutes it stays down until the daemon restarts, and doctor shows it.

The daemon never imports a pillar's code, so no pillar crash can take the daemon down. A `unix` service that dies has its calls resolve as `crashed` in the same way. The daemon reconnects with the same backoff, but restarting the service is up to whoever launched it.

**Host environment.** A host process gets a minimal environment. It does **not** inherit the daemon's environment, which holds provider API keys and the daemon token. Pillars must not. Operator command hooks inherited it until #571 (Appendix B.5); since #571 the daemon hands them `PATH`, `HOME`, `USER`, `LANG`, `LC_*`, `TMPDIR`, `SHELL`, their payload variables and the names in their own `env_allowlist`, nothing else (`hooks/executor.py`, `_command_environment`). Their shell is still `bash -l`, so the operator's own login files run and can export more.

---

## 12. Versioning and the manifest

- The contract version is `hooks/MAJOR.MINOR`. This document is `hooks/1.0`.
- **A minor version** may add optional payload fields, events, decision points, outcome codes, or transports. It never removes, renames or retypes anything, and never widens an existing event's capabilities. A pillar only ever receives events it subscribed to, and new optional fields must be safe to ignore. Adding a decision point to an existing event is a minor change: the point is off by default and reachable only by a subscription that declares `decide:<point>`, so no existing subscription gains a capability.
- **A major version** is anything else.
- The daemon declares the range it supports: one major version and a minor range, for example `hooks/1.0–1.2`.
- A pillar declares the version it needs, for example `requires: hooks/1.1`. It loads if and only if the majors match and its minor is within the daemon's range.

**Manifest** (shipped with the pillar; YAML shown here for readability, JSON on the wire):

```yaml
name: instinct
version: 0.3.1
requires: hooks/1.0
kind: instinct              # instinct | cognition
transport: host             # host | unix
subscriptions:
  - event: turn_start
    hook: router
    capabilities: [decide:route]
  - event: pre_tool_use
    hook: tool_guard
    capabilities: [veto, escalate]
  - event: turn_end
    hook: learner
    capabilities: [observe]
resources:
  memory_max: 2G
  cpu_quota: 50%
```

A pillar's `kind` sets its default transport and how doctor groups it. It does not set budgets: every call gets the budget of its **event's** class (section 9). An Instinct pillar that subscribes to `after_tool_results` gets the Cognition deadline there.

**Loading is all or nothing.** A pillar does not load if any of these is true:
- its version is out of range;
- it subscribes to an unknown event;
- it asks for a capability or veto strength the event doesn't allow;
- it names an unknown decision point;
- it names a transport v1 doesn't have (`inproc`, `http`).

Then:
- the daemon logs one WARNING at boot;
- it writes a `subsystem_runs` row (`subsystem: pillars`, `operation: load`, `outcome: skipped`, with the reason);
- `oara doctor` shows ✗ with the reason. For example: *"cognition 0.4.0 requires hooks/1.3; this daemon supports hooks/1.0–1.2 — upgrade Prometheus or install cognition 0.3.x."*

A pillar never runs with some of its subscriptions silently dropped. A `unix` pillar can be upgraded while the daemon runs, so the daemon repeats the version check at each connect (`GET /hooks/1/manifest`).

**Config** (all keys are new; they land with the implementation):

```yaml
pillars:
  load: []                  # pillars to load, in declared order. Empty or absent = none.
  decide:                   # at most one pillar per point; false = off (the default)
    route: false              # locked off until pillar-picked connections are pinned (section 6.3)
    route_allow_cloud: false  # true lets the route table include routes that aren't local (section 6.3)
    tool_choice: false        # locked off until the parity harness measures a forced round's cache cost (section 6.3)
  budgets:
    instinct_deadline_ms: 100
    cognition_deadline_ms: 5000
    cognition_turn_budget_ms: 15000
    pipeline_deadline_ms: 60000
```

Installing a package is not enough for a pillar to load: it must also be listed in `pillars.load`. An installed but unlisted pillar is inert. Doctor lists it as "installed, not enabled".

---

## 13. Observability

**Every pillar hook call and every decision-point consultation writes one row**, in a new `hook_calls` table in `telemetry.db` (decision 15). Its DDL comes with the implementation. Columns:

| Column | Notes |
|---|---|
| `id`, `timestamp`, `contract` | |
| `event`, `pillar`, `pillar_version`, `hook`, `transport` | |
| `session_id` | For description only. NULL for ephemeral sessions, the same rule `tool_calls` follows (`engine/agent_loop.py:4588`). |
| `turn_id`, `round_index` | |
| `deadline_ms`, `duration_ms` | `duration_ms` is NULL when nothing was measured (`busy`, `tripped`), following the schema-v2 rule for `tool_calls.latency_ms`. |
| `outcome` | `ok`, `abstain`, `timeout`, `error`, `invalid`, `stale`, `late`, `busy`, `tripped`, `crashed`, `dropped`, `budget_exhausted`, `not_honored`. A reply that arrives after its deadline updates that call's row from `timeout` to `late`. `verdict` stays NULL and `applied` stays 0. |
| `verdict`, `applied` | `verdict` is NULL whenever the outcome isn't `ok`. `applied` is 1 when the daemon used the answer, including an `ok` answer of `none`, which it combined and which let the action through. It is never 1 when the outcome isn't `ok`. So a pass (`ok`, `none`, 1) and a missed veto (another outcome, NULL, 0) can't be confused. |
| `decision_point`, `choice`, `confidence`, `backend`, `backend_version` | |
| `reason_code` | A short code, not free text |

**No content is stored in the row:** no payload, no annotation text, no reason text. A veto's reason text lives in the conversation, where the person saw it.

**Pillar lifecycle events** (load, skip, trip, reset, crash, restart) write `subsystem_runs` rows with `subsystem: pillars`.

**Logs**
- one INFO line per pillar at load, with name, version, transport, subscriptions and the decision points it holds;
- one WARNING per non-`ok`, non-`abstain` outcome (section 9.2);
- one WARNING at boot if any operator hook is configured on `session_start` or `session_end`, naming the events. Those hooks never fire in v1 (decision 4, section 15);
- no conversation content, ever.

**`oara doctor`** (`cli/doctor.py`) gains a **Pillars and hooks** section, built from config, manifests and `telemetry.db`, so it works with the daemon stopped:
- the contract range this daemon supports;
- each installed pillar: loaded, not loaded (✗ with the reason), or installed but not enabled;
- each pillar's transport and isolation, and **which caps are actually enforced**;
- each pillar's subscriptions (event, hook, capabilities), each decision point with its flag state, and `route_allow_cloud`. A flag's state reads "locked: …" whenever its lock is in force (`tool_choice`: the parity harness's measurement; `route`: connection pinning), whether or not the config sets it;
- each route's classification, local or cloud, read from its `classify_route` row, or "not classified: route flag off" (or locked) when there are no rows. The row gives:
  - the reason: the provider type; a resolution that failed or timed out; an Ollama model that is cloud-named, reported as served upstream, or couldn't be checked; or which resolved address was outside the local ranges;
  - the addresses judged and pinned, and whether an environment proxy applies to the route's URL, which pinned requests bypass;
  - when the classification was made, and whether it is older than twice the TTL and so counts as cloud;
  - for `ollama` routes, the server's `/api/status` cloud setting, where the server reports one (section 6.3);
- health over the last 24 h: number of calls, p50/p95 duration, and counts of timeouts, errors, busy, crashes and drops, plus whether the pillar is tripped;
- operator hooks from `hooks:`: event, kind, matcher, `block_on_failure`, timeout.

Doctor's exit code follows its existing rule (non-zero on any ✗, `cli/doctor.py:1050`):
- a configured pillar that fails to load is ✗;
- an operator hook configured on `session_start` or `session_end` is ✗ **"configured, never fires"** (decision 4);
- `pillars.decide.tool_choice` set while it is still locked is ✗, naming the missing parity-harness measurement (section 6.3);
- `pillars.decide.route` set while it is still locked is ✗ "locked: connection pinning not implemented" (section 6.3);
- a tripped pillar, or a p95 over target, is a warning.

The decision-4 WARNING and ✗ are the v1 behavior. Their code comes with the implementation.

---

## 14. Trust

**Installing a pillar means trusting local code with your conversation.** A pillar can receive:
- what you type;
- what the model writes;
- the arguments of every tool call;
- tool output, which includes file contents, command output and pages the agent fetched.

The daemon cannot see what a pillar does with that. The same is true of any program you install that can read your files.

What the contract does and does not guarantee:

- **Content is labeled** (section 4.2), so a well-behaved pillar can avoid being steered by text it received. The label protects the pillar from the payload. It does not protect you from the pillar.
- **"Pillars run locally, no outside hosts"** is a rule for pillars, and OAra's own pillars must follow it. The daemon enforces it only where the OS lets it: a daemon-launched host on Linux under systemd gets `IPAddressDeny`. Everywhere else it is a promise, not a guarantee, and doctor reports which of the two applies.
- **Ephemeral sessions** send pillars no content (section 4.3, decision 3).
- **A pillar's route pick stays local unless you opt in.** A pillar's route choice is limited to the primary and to local routes, unless you set `pillars.decide.route_allow_cloud` (section 6.3, decision 17). Your own per-session model choice always wins. Precisely:
  - **Local, by type and address.** A local route has a local provider type **and** an endpoint that resolved only to loopback, private-range (RFC 1918, link-local, IPv6 ULA) or tailnet addresses. The endpoint judged is the host of the resolved base URL, reached directly and never through an environment proxy.
  - **The type alone doesn't count.** A `llama_cpp` or `vllm` provider pointed at a server on the internet is **not** local, whatever its type says. Neither is an Ollama model that Ollama names or marks as cloud, or one that can't be checked.
  - **Pinned.** A pillar-picked request to a local route connects directly to the address that was classified, never to a fresh DNS answer or through a proxy. Tailnet names are fine. The route flag stays locked until pinning is implemented.
  - **Kept fresh.** While the route flag is on, every route with a local provider type is re-classified in the background every TTL (60 s by default), and a classification older than twice the TTL counts as cloud. With the flag off, nothing is classified and nothing new runs.
  - **Recovery adds no destination.** If the picked route fails, the turn goes to the primary first, and today's recovery applies from there, to targets today's routing already uses. The worst case before any output has streamed is today's path plus one failed attempt. A failure after the first round carries the turn so far, including what the picked model read, to the primary and its fallbacks. A picked route that fails after its reply has started streaming isn't moved; the turn ends as a failed turn.
  - **Telegram's teacher goes through the primary too.** If a picked turn's reply trips the teacher's failure detector, the primary re-answers first. The teacher logic then judges the primary's answer, as it does today (section 6.3).
  - **Only the first hop.** The check sees the address the daemon connects to, not where the content goes from there. A local port that forwards elsewhere passes as local: an SSH tunnel, a local gateway in front of a hosted API, or an Ollama that forwards a model upstream without saying so. Only you know what listens there. Setting `OLLAMA_NO_CLOUD=1` on an Ollama server rules out its cloud models at the source.
- **What a compromised or buggy pillar can do:**
  - see content;
  - delay a turn by up to its deadlines;
  - stop turns (a denial of service);
  - put labeled notes in front of the model, which can steer it;
  - decide within the option tables of the decision points the operator turned on.
- **What it cannot do:** execute a tool, change a tool call's arguments or which tool runs, get past the gate or the approval prompt, route to a non-local endpoint you didn't allow, or write memory.

---

## 15. Today's operator hooks and pillar hooks

**What exists today** (the full detail is in Appendix B):
- Operator hooks are defined under `hooks:` in `prometheus.yaml`, loaded by `hooks/loader.py:32` and run by `HookExecutor.execute` (`hooks/executor.py:59`).
- They run one after another, awaited on the event loop.
- There are four kinds: `command`, `prompt`, `http`, `agent`.
- Only `pre_tool_use` and `post_tool_use` fire.
- A hook can block a call (`block_on_failure`) or observe. It cannot annotate or modify.
- `prompt` and `agent` hooks send the payload to the daemon's configured model provider, which may be a hosted API. `http` hooks may target any URL.
- So operator hooks do not meet the pillar rules: they are not local-only, and they do not fall back to the default on failure (they honor `block_on_failure` instead).
- Since #571 a `prompt`/`agent` hook that times out or raises, or a `command` hook that cannot start, becomes a failed result that honors `block_on_failure`, names the hook, and logs one WARNING, instead of escaping as the tool's failure. Command hooks get a minimal environment plus a per-hook `env_allowlist`, not the daemon's (Appendix B.4 and B.5, fixed in #571).
- Since WP-X.26 every kind's `timeout_seconds` is a total deadline. A `command` hook runs in its own process group and a timeout kills the group, so what `bash` started stops too (before, `sleep 4; echo …` with a 1 s timeout returned after 4 s). An `http` hook's timeout wraps the whole request (before, httpx's per-phase timeouts let a server trickling a byte every 0.5 s hold a 1 s hook for as long as it liked, and the hook reported success). A timeout of either kind fails like the others: one WARNING, a result that honors `block_on_failure` and names the hook. The one exception is a process the hook started that leaves the group itself (`setsid`); the executor stops waiting for it after 2 s. Cancelling a turn (Ctrl-C, shutdown) also kills the group, with SIGKILL, so the hook's own `trap` cleanup does not run. Before, what bash had started kept running; only in a terminal did Ctrl-C reach the hook itself, as a SIGINT it could trap. Running in its own session, a command hook has no controlling terminal, so it cannot prompt on `/dev/tty` (a daemon under systemd never had one); its own command runs below a constant session leader, so it can still call `setsid()`. That leader is a `bash --norc -c` that discards its own stderr once its script runs and does not read `BASH_ENV` (it hands an allowlisted one to the command's login shell unchanged). What the command sees still differs from before in five ways: its `$PPID` is the leader, not the daemon; `SHLVL` is one higher; a command killed by a signal reports exit code 128+n, as a shell reports it, where a single simple command used to report -n; a warning bash prints while starting up (on bash 5.2, an `LC_ALL` naming a locale that is not installed) appears twice; and if the leader itself cannot fork (a process limit reached), the hook fails with a bare exit code and no fork error.

**The options**
- **A. One registry.** Operator hooks and pillar hooks are two kinds of entry in one registry and one dispatcher. They share the event catalogue, ordering, telemetry and doctor listing, and keep separate rules.
- **B. Separate registries.** The `HookExecutor` stays as it is, and a new pillar host sits beside it.

**Decided: A, one registry with two kinds** (decision 5). Operator hooks keep their current semantics, positions and payloads in v1. The reasons:

1. **One answer to "what runs here, and in what order".** Two registries at the same event means two orderings, and doctor and telemetry would have to reconcile them. Two parallel copies of machinery that drift apart is the defect the loop's own comments call "the two-loop defect" (`engine/agent_loop.py:955-957`).
2. **One authority rule.** Restriction-only verdicts, with the gate as the floor, apply to both kinds. Operator hooks can only block today, so they already fit.
3. **One vocabulary.** An operator can later move a check from a command hook to a pillar without renaming events.

**The two kinds keep different rules**, because they are different promises:

| | Operator hook | Pillar hook |
|---|---|---|
| Configured by | `hooks:` in `prometheus.yaml` | `pillars:` and the pillar's manifest |
| On failure | Honors `block_on_failure`: fails closed when the operator asked for that | Always falls back to the default |
| May leave the machine | Yes, if the operator points it there (an `http` URL, or a `prompt`/`agent` hook on a cloud provider) | No |
| Payload in v1 | Today's flat payload, unchanged: `{tool_name, tool_input, event}` plus `tool_output`/`tool_is_error` after the call, delivered to command hooks as `PROMETHEUS_HOOK_PAYLOAD` / `ARGUMENTS` | `hooks/1` envelope |
| Capabilities in v1 | Today's: block at `pre_tool_use`, observe at `post_tool_use` | Section 3 |
| Position in v1 | Today's: `pre_tool_use` before the adapter and gate (C1), `post_tool_use` at C18 | Section 8 |
| `session_start` / `session_end` in v1 | Never fire, as today. If any are configured, boot logs one WARNING and doctor shows ✗ "configured, never fires" (decision 4) | Fire (section 3.1) |

**Operator session hooks (decision 4).** These can be configured today, and nothing fires them (Appendix B.1). In v1 they still don't fire, because firing them would change behavior for anyone who configured them. But they are no longer silent: the boot WARNING and the doctor ✗ tell the operator their hook does nothing. The code for both comes with the implementation.

**Why B is weaker.** B ships sooner, but it leaves two hook systems at the same events, with separate ordering and observability, and nothing forcing them to agree.

**Constraints on the migration**
- Unifying the registry is a refactor of `hooks/`. It has to pass the full suite unchanged and replay the golden traces identically, like any other seam.
- Changing where operator hooks run, or what they receive, or firing their session events, is a later minor version with its own decision (decisions 4 and 6, section 18).

---

## 16. No pillar installed

### What runs

- **Every default hook in Appendix A**, at its current position, in its current order, with its current failure behavior. That includes the defaults that fail at DEBUG level today (Appendix B.10). Changing those is not part of this contract.
- **Operator hooks, exactly as today:**
  - `pre_tool_use` at `engine/agent_loop.py:3902` and `post_tool_use` at `:4633`;
  - the same payloads and the same `block_on_failure`;
  - their own timeouts: since #571 a `prompt` or `agent` hook is bounded by its `timeout_seconds` (before #571 it had none, Appendix B.4), and a hook that times out, raises, or (for `command`) cannot start fails with one WARNING and honors `block_on_failure`;
  - `session_start` and `session_end` still never fire for operator hooks.
- **One addition, deliberately (decision 4).** If an operator hook is configured on `session_start` or `session_end`, boot logs one WARNING and doctor shows ✗ "configured, never fires". This happens at boot and in doctor only. No turn runs differently, and with no such hook configured, nothing is added at all.
- **Nothing else.**
  - no host processes, worker threads or sockets;
  - no `hook_calls` rows, and no new log lines beyond the decision-4 WARNING;
  - no config keys required: an absent `pillars:` section means no pillars.

A pillar that is installed but not listed in `pillars.load` counts as not installed for all of the above.

### Why turn behavior is identical

1. **The pillar registry is built once at boot** from `pillars.load`. With nothing listed it is empty, and it stays unchanged for the life of the daemon.
2. **Every insertion point is a single `has_subscribers(event)` check** against that frozen registry: a dictionary lookup. With no subscribers it builds no payload, takes no lock and adds **no `await`**. The last part matters: an extra await point on the hot path changes how concurrent sessions interleave, even if it returns immediately.
3. **Every decision point needs a flag that names a loaded pillar.** With none loaded, every flag resolves to off, and the default's own fallback runs. That is the code path that runs today. Route classification, when the flag takes effect and in the background, runs only while `pillars.decide.route` is on, so it never runs without a pillar.
4. **Nothing is written during a turn.** There are no telemetry rows, files or sockets.

**What proves it is not this document.** The proof is the full existing suite passing unchanged, identical golden-trace replay, and overhead inside the noise band, all run on the implementation. Until then, "identical" is a design requirement, not a measured result.

---

## 17. Decisions from review

These were decided in review on 2026-09-24. Each entry says whether it was accepted as recommended, changed in review, or added in review, and where this document applies it.

1. **How Instinct runs.** *Changed in review.*
   - `host` is the default for Instinct pillars.
   - `inproc` is removed from v1 entirely, because it can't meet "a pillar crash never takes the daemon down".
   - It may return in a later minor version, but only if the measured `host` overhead misses the Instinct budget.

   Applied in sections 10, 11 and 18.
2. **Budget numbers.** *Accepted.* The deadlines, targets and thresholds are provisional, and the implementation re-sets them from measurements (section 9.1).
3. **Ephemeral sessions.** *Accepted.* Pillars get no `content` for ephemeral sessions (section 4.3).
4. **Operator hooks on `session_start` and `session_end`.** *Changed in review.*
   - They don't fire in v1.
   - They aren't left silent either: if any are configured, boot logs one WARNING and `oara doctor` shows ✗ "configured, never fires".
   - This is the v1 behavior, and its code comes with the implementation.

   Applied in sections 13, 15 and 16.
5. **Registry.** *Accepted.* One registry with two kinds (section 15).
6. **Where operator `pre_tool_use` hooks run.** *Accepted.* Unchanged in v1. Moving them after the gate is a later minor version (section 18).
7. **What "escalate" means.** *Accepted.* It means "require the human's approval". A pillar can't ask for a stronger model in v1. *Amended in the last review round:* it takes the gate's approval path for every origin, the approval queue included, and is refused without asking only where no approval path exists (sections 5 and 18.1).
8. **The `tool_choice` decision.** *Accepted, with an addition.*
   - Forcing is allowed behind the flag, for at most one round per turn, and a pillar-forced round never raises.
   - **Addition:** the flag can't be turned on anywhere until the parity harness has measured the llama.cpp prefix-cache cost of a forced round.

   Applied in section 6.3.
9. **The `tool_name` decision.** *Changed in review: removed from v1.*
   - When adapter repair fails today, the model is told and retries. That is not a dead end.
   - So the upside is at most one saved round, and the downside is running a tool the model didn't name.

   Applied in sections 3 and 6.3. Section 18 lists the evidence needed before it is reconsidered.
10. **Provenance for delivery notes.** *Accepted.* A new value, `pillar`, untrusted by default, added to the `Provenance` set through the memory schema contract (section 5).
11. **Pillar-written compaction summaries.** *Accepted.* Not in v1.
12. **The unreachable decision points** (`compaction_span`, `skill_draft`). *Accepted.* They stay inert in v1. Making either reachable is a separate change to default behavior (section 6.3).
13. **HTTP transport.** *Accepted.* Unix sockets only in v1. The loopback, mutual-HMAC design is kept as the plan for when `http` is added (sections 11 and 18).
14. **The daemon token.** *Accepted.* It is never sent to a pillar, and is used only for pillar-to-daemon calls (section 11).
15. **Where the telemetry goes.** *Accepted.* A new `hook_calls` table (section 13).
16. **Post-processing after the loop, per surface.** *Accepted.* This is a known limit of v1. Two things are separate decisions: whether the Telegram-only steps move into the loop, and whether post-task hooks move off the delivery path (sections 3.1 and 9.1).
17. **Cloud routes.** *Added in review.*
    - The `route` option table includes only the primary and local routes (section 6.3) unless the operator allows otherwise, with `pillars.decide.route_allow_cloud` (default `false`).
    - *Amended in review:* a local route has a local provider type **and** an endpoint that resolves only to loopback, private-range (RFC 1918, link-local, IPv6 ULA) or tailnet addresses. The route is classified only while the route flag is on, when the flag takes effect and then in the background at the TTL, never per turn. Anything else, including an address that can't be resolved, is cloud. Today's code decides local by type alone (section 6.3).
    - A pillar's choice must never send conversation content to a hosted API, or spend money, without the operator opting in.
    - The per-session user override still always wins.
    - *Rulings after the amendment:*
      - **Recovery.** A failed pillar pick goes to the primary first, and today's recovery applies from there, unchanged. The operator's fallback targets are not skipped. Recovery adds no destination. A teacher-detector trip on Telegram counts as a failed pick. Section 6.3 states the limits: a failure after the first round carries the turn so far, and a mid-reply failure ends the turn.
      - **Freshness.** While `pillars.decide.route` is on, every route with a local provider type is re-probed or re-resolved in the background at the TTL, and a classification older than twice the TTL counts as cloud. With the flag off, nothing is classified and nothing new runs.
      - **DNS.** No IP literals are required. A pillar-picked request connects to the classified address (pinned), and the route flag stays locked until pinning is implemented. Pinned requests bypass environment proxies, and doctor shows whether a proxy applies.
      - **Ollama.** A model that Ollama names as cloud (case-insensitively), reports as served upstream (`remote_host` or `remote_model`), or that can't be checked, is cloud.

    Applied in sections 0, 6.2, 6.3, 7, 12, 13, 14, 16 and 18. Section 6.3 also states what the rule can't see: only the first hop, and what Ollama doesn't report.

---

## 18. Possible later minor versions

None of the candidates below (the table, 18.1 and 18.2) is in v1. Each would need its own decision and a minor version bump (section 12). Section 18.3 is different: it lists open items for implementing the route decision point, not candidate changes to this contract. Each is listed with what has to be true before it is considered. Every condition is stated here in full, so it can be checked without any other document.

| Candidate | Before it is considered |
|---|---|
| `inproc` transport (decision 1) | The measured `host` transport overhead misses the Instinct budget (section 9.1). |
| `tool_name` decision point (decision 9) | Evidence of two things. First, how often adapter repair fails: `tool_calls` rows with `error_type = 'validation_failed'` (`engine/agent_loop.py:4024`). Second, what the model does next: whether its retry succeeds and in how many rounds. `retry_success` repair pairs in `training.db` record the recoveries (`:3996-4003`). |
| `http` transport on loopback (decision 13) | A real pillar that can't use a unix socket. Section 11 has the design. |
| Operator `pre_tool_use` hooks after the gate (decision 6) | A decision to change what existing operator hooks see. After the move, they would see the repaired call that will actually run. |
| Firing operator `session_start` / `session_end` hooks (decision 4) | A decision to change behavior for operators who configured them. Until then, the boot WARNING and doctor ✗ tell them the hooks never fire. |
| `tool_subset` decision point at `before_model_call`: narrow the advertised tools for one round | Evidence of three things:<br>1. **Narrowing doesn't hide the right tool.** Measured on real turns, the tool the model needed is in the narrowed set.<br>2. **The model can always ask for the full list,** so a wrong narrowing costs a round, not the task. `tool_search` is the precedent: it is in the shipped always-loaded set (`config/shipped_defaults.py:37`), an empty query returns every tool and skill name (`tools/tool_search.py:149-155`), and what it finds arrives as a tool result (`context/dynamic_tools.py:239-248`). A narrowed round must always keep `tool_search`.<br>3. **The prefix-cache cost is measured.** The catalog is frozen for the run today because changing the tools block invalidates the provider's cached prefix (`engine/agent_loop.py:1207-1211`, the #120 bug class). |
| Computer-use chooser decision point: the `Chooser` seam in `computer/` (`computer/chooser.py:39`), consulted where `RuleChooser` abstains because nothing matched (`:81-84`). Its empty-table abstain (`:71-72`) is never reached, because the loop returns before calling the chooser when there are no candidates (`computer/loop.py:129-133`). | Two things:<br>1. **Computer use is past milestone 1 and in real use.** Milestone 1 is the candidate table, the gate rule and the validation path, proven against recorded fixtures, with the tools not registered (`computer/__init__.py:3-11`). "In real use" means the computer-use tools are registered and running, so `RuleChooser`'s abstains are measured on real sessions, not fixtures. Registration has its own preconditions ([docs/audits/COMPUTER-USE-REGISTRATION.md](../audits/COMPUTER-USE-REGISTRATION.md)).<br>2. **A decision corpus exists that meets two rules** (ruled 2026-09-20):<br>- every item carries an abstain label, because "none of these" is always a possible right answer;<br>- results are reported separately for items where none of the candidates is correct, so a chooser can't score well by never abstaining.<br><br>The seam already has most of this shape. A live chooser belongs behind the same Protocol, and its timeout returns `abstain` (`computer/chooser.py:20-22`). Its answer is validated against the table the client built (`computer/loop.py:141`, `computer/candidates.py:156`).<br><br>The minor version must still settle three things:<br>- **Sync versus async.** `choose` is synchronous and is called on the event loop (`computer/chooser.py:44`, `computer/loop.py:137`). A pillar-backed chooser needs an async call to meet section 10.<br>- **Order.** Section 6.2 puts `RuleChooser` first, with a pillar only where it abstains. That is the reverse of the degradation order in `computer/chooser.py:16-18`.<br>- **Where it fires.** It must name the event or pipeline point the decision fires at. |
| Fail-closed veto subscriptions: an operator opt-in under which a **missed** veto at `pre_tool_use` becomes `escalate` (ask the human) instead of not applied | 1. A pillar veto operators use in practice.<br>2. Measured miss rates from `hook_calls`, showing no flood of approval prompts.<br><br>Details, costs and the recording rule are in section 18.1. |
| Coding-run verdict point: a Cognition verifier sees a coding run's acceptance result and diff and can ask for **one more episode**, with open code deciding | Cognition's verdicts agree with the acceptance commands, measured in shadow.<br><br>Details are in section 18.2. |

### 18.1 Fail-closed veto subscriptions

**What it is.** A pillar veto that misses at `pre_tool_use` normally isn't applied (section 5), and that stays right for every pillar that isn't opted in.
- **The opt-in** is the operator's, in config (for example `pillars.fail_closed: [<pillar>]`, next to `pillars.decide.*`). A manifest can never set it.
- **For an opted-in pillar**, a missed veto at `pre_tool_use` becomes `escalate`.
- **"Missed"** means the outcome is `timeout`, `error`, `invalid`, `busy`, `tripped` or `crashed`, decided at the deadline. A `late` answer that follows doesn't undo the escalation.

**Before it is considered.**
1. **A pillar veto operators use in practice,** where a missed check should wait for a human rather than go ahead. This doesn't make the pillar a boundary. The gate still is (section 5), and a pillar that answers `none` in time lets the call through whatever this option says (section 6.2).
2. **Measured rates of every missed outcome,** plus time spent tripped, taken from `hook_calls`. They must show the opt-in can't flood approval prompts. A trip alone would turn every call into `escalate` for the 5-minute cooldown (section 10).

**Costs the minor version must answer.**
- **Background work and approval prompts.** A missed veto becomes `escalate`, which takes the gate's approval path (section 5). A dead or tripped fail-closed pillar would therefore affect every agent-loop tool call it subscribes to, system-origin calls included, such as subagents, `local_agent` tasks and evals:
  - where an approval queue is enabled, every such call goes to the operator's queue and waits up to the queue's timeout, 1800 s by default (`daemon.py:1398`). That is the flood the measured miss rates must rule out;
  - where no approval path exists (the shipped default), every such call is refused.
  - Cron commands aren't agent-loop tool calls. They are vetted by the gate directly (`gateway/cron_scheduler.py:317-325`).
  - Coding runs set no gate and no hook executor today (`coding/session.py:255-270`), so `pre_tool_use` doesn't fire for them.
- **Other sessions.** The circuit breaker and the global in-flight cap are per pillar (section 10). One session's misses could send every session's tool calls to an approval prompt, or to a refusal where no approval path exists. The minor version must say how "one session never stalls another" still holds.

**How a miss is recorded.**
- The row keeps the real outcome, with no verdict, `applied = 0` and `reason_code = fail_closed` (section 13).
- The approval prompt says the pillar didn't answer (for example "tool_guard didn't answer in time"). It never gives a reason the pillar didn't give (section 9.2).

**Minor or major.** It adds a config key and no capability: `pre_tool_use` already allows `escalate`, and the daemon only raises `none` to `escalate`. It amends the "always falls back to the default" rule (sections 0, 9.2 and 15), for the named pillars only. Section 12's list of minor changes doesn't include an operator option like this. So that minor version must also extend section 12's list (an operator option, off by default, that changes no payload or capability), or the change is a major version.

**The operator-hook precedent.**
- `command` and `http` operator hooks fail closed on a timeout or error when `block_on_failure` is set (`hooks/schemas.py:22`, `:44`, default false; `hooks/executor.py:104-112`, `:153-159`). Since #571 that includes a `command` hook that cannot start.
- `prompt` and `agent` hooks default it to true (`hooks/schemas.py:33`, `:55`). Until #571 it governed only a negative answer, not a missed one, because they applied no timeout and caught no exceptions (Appendix B.4). Since #571 a timeout or an error fails closed too.
- So all four kinds can now fail closed on a missed answer, but fail closed means something different there: an operator hook that misses with `block_on_failure` set REFUSES the call (`engine/agent_loop.py:3908-3922`, no approval path), where this option would turn a missed veto into `escalate` and take the gate's approval path.

### 18.2 Coding-run verdict point

**What it is.** A pipeline point in `CodingSession.run` (`coding/session.py:251`). A Cognition verifier sees the run's acceptance result and diff, and can ask for **one more episode** before the run is recorded as a success. An episode is another `run_loop` call, which section 1 calls a turn. Open code decides.

**How it fits the contract.** It is a decision point with the options `accept` and `another_episode`:
- **Open code builds the table.** It offers `another_episode` only while the caps are clear.
- **The default must abstain first.** At this seam today, the default has already answered (acceptance exited 0, so accept). Section 6.2 lets a pillar decide only where the default abstains, so the default needs an abstain band here: a green run with caps to spare. That is a change to default behavior, and it needs a separate change of its own (decision 12).
- **Every miss means accept.** The fallback, and the result of any missed outcome, is `accept`: the green result stands.

**Before it is considered.** Cognition's verdicts agree with the acceptance commands, measured in shadow: the verifier runs, its verdicts are recorded, and nothing acts on them. The seam below fires only on green runs, so the shadow run must also record verdicts on ground-truth failures (`coding/session.py:375-380`). Otherwise agreement can't be measured on failing runs.

**Where it fires.** `CodingSession.run` has three success returns:
- **`coding/session.py:371-374`: acceptance, run by the session itself, passed within the caps.** Only this one can offer another episode.
- **`coding/session.py:292-296`: green after a pause ran past the wall cap.** No budget is left, so no grant is possible.
- **`coding/session.py:349-352`: green at a cap, or after a stalled episode** (`coding/session.py:339`). A stall isn't a cap, but a model that made no progress gets no extra episode.

**What open code keeps.**
- **Fresh cap checks.** The cap check at `coding/session.py:335-340` runs before the acceptance run (`coding/session.py:367`), which can take up to 240 s. So the verdict point runs a fresh `over_cap(time.monotonic() - started)` (`coding/policy.py:115`) before it consults the verifier and again before any grant. The verifier's deadline is capped at the wall time left.
- **Bounded episodes.** A granted episode is bounded by the remaining round budget (`context.max_turns`, `coding/session.py:300-304`). Like any episode today, it is checked against the wall cap only when it ends. The task manager's watchdog kills the process at `max_wall_seconds + 300` (`coding/managed.py:115`).
- **No lost success.** A grant must never turn a success into a failure. The green tree is committed or saved before the extra episode runs. If the extra episode ends red, the saved green result is what is reported.
- **Untrusted verifier text.** A decision carries no text (section 4.4). For the verifier to say why, the point must also allow `annotate`, with a fixed slot: a request-only note on the extra episode's first request, under section 5's rules (labeled, untrusted, provenance `pillar`). It never goes in as the trusted `orchestrator` message the seam uses today (`coding/session.py:359-363`, `:376-380`). Without `annotate`, no verifier text reaches the model.

**Missing input.** Today the diff stat is produced only by `_finish` (`coding/session.py:449`, `_commit_artifact` at `:195-205`), after the verdict. It is `HEAD~1..HEAD`, which misses commits the model made itself. The verifier needs a diff against the commit the branch was cut from (record it at `_prepare_branch`, `:192-193`), including untracked files, taken before `_finish`.

**Which process.** Coding runs are a `prometheus code` child process (`coding/managed.py:91-115`, `__main__.py:793`), not the daemon. The minor version must say:
- how that process reaches the pillar (a `unix` transport, or a relay through the daemon);
- whose caps and circuit breaker apply;
- which budget class the point uses.

### 18.3 Open items for the route decision point's implementation

These items aren't changes to this contract. They are what implementing the `route` decision point involves, collected from section 6.3, which states each rule in full. The lock itself lifts when pinning is in place (item 1, section 6.3). Anything further about the route decision point goes here, or in the table above, not into another amendment.

1. **Pinning.** A pillar-picked request to a local route connects directly to the classified address. It keeps the hostname for the `Host` header and TLS, and ignores environment proxies. Classification requests also connect directly. When a re-classification sees a new answer, the pin moves to it if it is local, and the route leaves the table if it isn't.
2. **Judging addresses.** Every resolved address must be in section 6.3's table. IPv4-mapped addresses are unwrapped first. The table is an explicit list, not `url_guard.is_blocked_address`. A resolution that fails or times out counts as cloud.
3. **Classification state.**
   - Classify only while the flag is on: when it takes effect, then every route with a local provider type once per TTL, in the background.
   - Never classify on the turn path, and leave today's on-demand probes unchanged. They don't classify.
   - Keep the results in state of their own, never in the backend registry's cached status.
   - Treat anything older than twice the TTL as cloud.
   - Write a `classify_route` row for each classification.
4. **The endpoint that is actually used.** Resolve the base URL through `_resolve_base_url` (`providers/registry.py:183-219`). Key the router's task-rule provider cache by endpoint (`router/model_router.py:1054`), or build separate instances.
5. **The Ollama check.**
   - Check the cloud name first, then `remote_host` and `remote_model` from `/api/show` and `/api/tags`. An unchecked model counts as cloud.
   - For an `ollama` primary, check the name that is actually sent.
   - A pinned request sends only the model that was checked.
6. **Recovery.**
   - These count as a failed pick: any error before output streams, a context pre-flight refusal, any circuit-breaker trip, a would-be tool escalation, and on Telegram a failure-detector trip. A pick of the primary is never a failed pick.
   - Move the turn to the primary with a fresh circuit breaker and a rewritten identity line. Re-resolve the catalog after a first-round failure, and keep the frozen catalog after a later one. Keep the rest of the turn on the primary.
   - Turn a would-be tool escalation into the primary's retry prompt.
   - After a detector trip, the primary re-answers. The honesty check and then the teacher run on its answer, and the picked reply is never delivered.
7. **Open questions about the teacher re-answer.** These are not decided by this contract; the implementation must settle them and record the answers here:
   - **The picked reply.** Is it kept in session history and LCM, and does the primary's re-answer see it? Today the loop's messages, final reply included, are added to the session before the post-turn steps run (`gateway/telegram.py:2014`).
   - **The span judged.** Do the post-turn steps on the re-answer judge the whole turn (the picked model's tool activity plus the primary's) or only the primary's run? On a combined trace, the detector can trip on the picked model's activity (`escalation/detector.py`).
   - **Continue or restart.** Does the re-answer continue the picked run, keeping its frozen catalog, or start a new run?
8. **Doctor.** Show:
   - each route's classification and reason;
   - the pinned addresses, and whether a proxy is bypassed;
   - the state of the locks;
   - each Ollama server's `/api/status` setting, where reported.

---

## Appendix A. Inventory of built-in behavior

Every place built-in behavior runs during a turn, in the order it runs. These are the **default hooks**. Each stage's defaults run before that stage's event:

| Stage | Defaults | Event |
|---|---|---|
| Turn setup | A.2 | `turn_start` |
| Each round, before the model call | A.3 | `before_model_call` |
| Model call and response | A.4 | `after_model_response`; `before_delivery` for the final answer (after M8's empty-turn check, before its commit) |
| Tool batch and each call | A.5, and A.6 through C10. The event fires before the approval prompt, C11. | `pre_tool_use` |
| Each call, after execution | C12–C19 | `post_tool_use` |
| After the batch | A.7 | `after_tool_results` |
| Turn end | A.8 | `turn_end` |
| Outside the turn | A.10 | `compaction_candidates` (L1, and R8 in-turn), `skill_draft_score` (L3, L4) |

A.1 and A.9 are surface code, outside `run_loop`. No event covers them in v1.

Columns:
- **Blocks?** Can it stop an action or the turn?
- **Changes?** Can it change what the model sees, what runs, or what the person receives?
- **On failure today:** how a failure of the behavior itself is surfaced now.

### A.1 Before the loop (surface code, differs by surface)

| ID | Default | Where | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| S1 | Surface authorization | `gateway/telegram.py:391`, `:848` | Drops updates from chats that aren't allowed | Yes | — | Silent drop, by design |
| S2 | Slash-command routing | `web/ws_server.py:883` | A command never becomes a turn | Yes | — | — |
| S3 | Rehydrate a cold session from LCM | `engine/session.py:739` (called at `web/ws_server.py:931`, `gateway/telegram.py:2279`) | Rebuilds history after a restart | — | Yes, the history | — |
| S4 | Per-session turn lock | `web/ws_server.py:1182`, `gateway/telegram.py:1975` | Serializes turns within a session. Web and Telegram share it. | Delays | — | — |
| S5 | LCM ingest of the user message | `ChatSession.add_user_message`, `engine/session.py:176` → `_persist_to_lcm` `:229` | Durable write | — | — | Best effort, never raises |
| S6 | Image reference rewrite | `web/ws_server.py:909` | Replaces image references with descriptions | — | Yes | — |
| S7 | Client system messages (OpenAI-compatible route) | `web/openai_api.py:196` | Appends the client's system text to the system prompt | — | Yes | — |
| S8 | Boot system prompt | `context/prompt_assembler.py:159`: SOUL, AGENTS, ANATOMY, base, environment, skills, project instructions via `project_files_section` `:126`/`:349`, memory | Assembled at boot, not per turn. Instruction files are `PROMETHEUS.md`, `CLAUDE.md`, `AGENTS.md` and others (`context/prometheusmd.py:36-46`). | — | Yes | — |

### A.2 Turn setup (`run_loop` / `_run_loop`, before the first round)

| ID | Default | Where (`engine/agent_loop.py`) | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| T1 | Per-run context copy | `:925` (`LoopContext.for_run`, `:815`) | Isolates per-run fields from the shared context | — | — | — |
| T2 | File-change turn key | `:930-933` | Scopes the shared file-mutation verifier to this turn | — | — | — |
| T3 | Ephemeral flag | `:938` | Nulls telemetry content and skips pair capture and post-task hooks | — | Yes (what is recorded) | — |
| T4 | Divergence task start | `:960-970` | Scopes the divergence detector to this task | — | — | DEBUG, fail-open |
| T5 | Tool-choice directive | `:1075-1091` | Resolves auto/none/required/`{tool}` from the caller or the mode; first-round forcing | — | Yes | — |
| T6 | **Model routing** | `:1101-1205` (`router/model_router.py:747`); identity-line rewrite `engine/agent_loop.py:1182` (`context/system_prompt.py:145`) | Picks provider, model, adapter and backend for the turn, and rewrites the `- Model:` line | — | Yes | WARNING; falls back to primary |
| T7 | Tool advertisement | `:1230-1245`; profile filter `:1252-1292`; telemetry row `:1297-1331` | Freezes the advertised catalog for the run (deferred tools, profile) | — | Yes | WARNING (profile), ERROR (empty filter), DEBUG (telemetry) |
| T8 | **Workspace and instruction-file injection** | `:1339-1380` | Resolves the session workspace (cwd, write boundary) and swaps in that workspace's `# Project Instructions` | — | Yes | WARNING |
| T9 | **Checkpoint** | `:1387-1397` (in a thread) | Snapshots the workspace before any tool runs | — | — | ERROR; the turn proceeds without an undo point |
| T10 | Adapter request formatting | `:1398-1402` (`adapter/__init__.py:124`) | Per-tier prompt and tool formatting | — | Yes | — |
| T11 | **Memory recall** | `:1410-1427` (`memory/recall.py:109`) | Adds a request-only `# Recalled memory` section | — | Yes (this run's requests) | WARNING, fail-open |
| T12 | Per-round usage envelope | `:1472-1473` (`learning/llm_envelope.py:154`) | Wraps model calls for usage and silent-failure telemetry | — | — | Records its own failures |

### A.3 Each round, before the model call

| ID | Default | Where (`engine/agent_loop.py`) | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| R1 | **Microcompaction** | `:1477-1478` (`_microcompact_old_results`, `:3244`) | Shortens old tool results **in place** (local tiers) | — | Yes (history) | Telemetry row; WARNING if the write fails |
| R2 | Steer drain | `:1504-1519` | Mid-turn user steers as a request-only addendum | — | Yes | DEBUG |
| R3 | Empty-response retry nudge | `:1524-1530` | A request-only nudge after an empty reply | — | Yes | — |
| R4 | Periodic nudge | `:1537-1541` (`:2739`) | A request-only self-reflection prompt every N rounds | — | Yes | DEBUG |
| R5 | Tier-full tool withholding | `:1550-1555` | Sends no tools in the payload, so the grammar holds | — | Yes | — |
| R6 | Round directive and grammar forcing | `:1559-1579` | Applies or relaxes forcing; withholds native tools on a forced local round | — | Yes | — |
| R7 | Visible-stream markup filter | `:1588-1594` | Hides tool-call markup from the live stream | — | Yes (what streams) | — |
| R8 | **Context compaction** (render view) | `:1601-1631` (`context/compactor.py:631`, span choice `:504`) | Replaces the oldest span with a cached summary, for this request only | — | Yes | ERROR (`log.exception`); sends uncompacted. Its own failures also go to `silent_failures`. |
| R9 | Context pre-flight | `:1743-1786` | Refuses to send a prompt over the measured window | Yes (ends the turn) | — | ERROR plus a telemetry row |

### A.4 Model call and response

| ID | Default | Where (`engine/agent_loop.py`) | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| M1 | **Stream with provider fallback** | `:1788-1812` (`engine/fallback.py:255`, `decide` `:68`); identity rewrite on degrade `engine/agent_loop.py:1674-1721` | Degrades to the fallback on a terminal auth or billing failure | — | Yes (serving model, notice) | WARNING |
| M2 | Raw output capture | `:1848` | Keeps the model's raw text for golden traces | — | — | — |
| M3 | **Text tool-call extraction** (adapter) | `:1851-1871` (`adapter/__init__.py:221`) | Turns `<tool_call>` text into structured calls | — | Yes | — |
| M4 | Final-text markup hygiene | `:1881-1932` | Strips grammar tags from delivered text and counts blocks stripped to nothing | — | Yes | WARNING |
| M5 | Strip-disagreement guard | `:1951-1988` | Retries with the stripped text fed back, bounded by the breaker | Yes | Yes | Ends the turn with an explanation |
| M6 | Empty-response guard | `:1999-2020` | One nudged retry, then an error turn | Yes | — | Error turn |
| M7 | Forced-tool check | `:2025-2040` | Raises if a forced `{tool: X}` came back as a different tool | Yes (raises) | — | Raises |
| M8 | Commit and degrade banner | `:2055-2091` | Commits non-empty turns only; puts the degrade notice in history; schedules the periodic nudge | — | Yes | — |
| M9 | Malformed-call retry | `:2100-2144` | Structured feedback when the provider dropped every call, bounded by the breaker | Yes | Yes | Breaker message |

### A.5 Tool batch, before dispatch

| ID | Default | Where (`engine/agent_loop.py`) | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| B1 | Iteration cap | `:2203-2221` | Ends the turn past `max_tool_iterations`; answers every pending call with `NOT EXECUTED` | Yes | — | — |
| B2 | **File-change snapshot** (before) | `:2226-2235` | Snapshots every path the batch may touch | — | — | DEBUG |
| B3 | Repeat-failure guard | `:2242-2262` | Refuses, without running it, a call that already failed twice this turn | Yes (per call) | — | WARNING |
| B4 | Dispatch and failure isolation | `:3692-3779`; `_safe_execute` `:3366-3419` | Read-only calls in parallel, mutating calls in sequence; a raising tool becomes an error result | — | Yes (the result, on exception) | ERROR plus a telemetry row |

### A.6 Each tool call (`_execute_tool_call`, `engine/agent_loop.py:3862`)

| ID | Default | Where | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| C1 | Operator `pre_tool_use` hooks | `:3901-3922` | Runs configured hooks on the **raw** call | Yes | — | See B.4 |
| C2 | Registry present | `:3924-3938` | — | Yes | — | Telemetry row |
| C3 | **Tool-call validation and repair** (adapter) | `:3947-3987` (`adapter/__init__.py:148`) | Validates the call; repairs the name (fuzzy match) or the arguments | — | Yes (name, arguments) | WARNING when the name changes; ERROR if pair capture fails |
| C4 | Adapter retry and escalation | `:3988-4048` (`_try_escalate_tool_call` `:3017`) | Returns a retry prompt, or has a subagent on the escalation provider run the call | Yes | Yes | WARNING; escalation is best effort |
| C5 | Unknown tool | `:4050-4065` | — | Yes | — | Telemetry row |
| C6 | Deferred-tool ("lucky guess") telemetry | `:4074-4090` | Records calls to tools that weren't advertised | — | — | — |
| C7 | Template-markup guard | `:4099-4138` (`adapter/markup_guard.py`) | Rejects arguments carrying chat-template markup | Yes | — | WARNING plus a telemetry row |
| C8 | Input validation and unwrap | `:4140-4211` | Pydantic validation; optional unwrapping of dict-wrapped arguments | Yes | Yes (unwrap) | Telemetry row |
| C9 | Gym call observer | `:4217-4225` | Scoring seam; None in production | — | — | DEBUG |
| C10 | **Security gate** | `:4228-4375` (`permissions/checker.py:920`) | ALLOW, APPROVE or DENY. An unresolved path or computer extent forces APPROVE (`engine/agent_loop.py:4358-4374`) | Yes | — | WARNING (origin, schema), then fails to the stricter side |
| C11 | Approval prompt | `:4376-4438` | Asks the human, showing the arguments; with no prompt available, denies | Yes | — | Telemetry row |
| C12 | Execution with timeout | `:4440-4494` | `asyncio.wait_for`, 300 s by default or the tool's own override | Yes (timeout) | — | ERROR plus a telemetry row |
| C13 | Printing Press suggestion | `:4503-4523` | Appends an install suggestion to a "command not found" bash failure | — | Yes | DEBUG |
| C14 | Per-result truncation | `:4530-4535` | Tool-aware cap on each result | — | Yes | — |
| C15 | **Telemetry** (`tool_calls`) | `:4543-4593` | One row per call; content columns nulled when ephemeral | — | — | — |
| C16 | Repair-pair capture | `:4601-4614` | Training pairs to `training.db` | — | — | ERROR (in `pair_capture`) |
| C17 | Divergence record | `:4620-4630` | Feeds the divergence detector | — | — | DEBUG |
| C18 | Operator `post_tool_use` hooks | `:4632-4644` | Runs configured hooks; the result is ignored | — | — | See B.4 |
| C19 | **LSP diagnostics** | `:4646-4652` (`hooks/lsp_diagnostics.py:35`) | Appends LSP errors to `write_file`/`edit_file` results | — | Yes | DEBUG |

### A.7 After the batch (each round)

| ID | Default | Where (`engine/agent_loop.py`) | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| A1 | Reassembly and failure tally | `:2279-2292` | Restores call order; counts failures per signature | — | — | — |
| A2 | Progress repeat detector | `:2299-2325` (`:3497`) | Detects identical calls that make no progress | — | — | — |
| A3 | **File-change diff** (after) | `:2328-2340` | Diffs the snapshots after each result | — | — | DEBUG |
| A4 | Circuit breaker, model fallback, diagnose-and-recover | `:2343-2447` (`_try_model_fallback` `:3113`) | Trips on repeated errors; may swap the model or raise the adapter tier once | Yes | Yes | WARNING (fallback), `circuit_breaker_diagnostics` rows |
| A5 | Cross-result budget | `:2450-2451` (`:3143`) | Truncates results in proportion to fit the per-turn budget | — | Yes | — |
| A6 | Result events and append | `:2453-2462` | Streams tool events; appends results to history | — | — | — |
| A7 | Unproductive-repeat halt | `:2477-2517` | Halts a turn that keeps getting the same result | Yes | — | WARNING plus a telemetry row |
| A8 | Divergence checkpoint, evaluation and halt | `:2533-2620` | Halts after 3 consecutive repetition-floor evaluations (`_DIVERGENCE_HALT_AFTER`, `:3461`) | Yes | — | WARNING plus a telemetry row; evaluation errors at DEBUG |

### A.8 Turn end

| ID | Default | Where | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| E1 | **Boundary-escape check** | `engine/agent_loop.py:2160-2177`, `:2192-2200` (`_boundary_escapes` `:2629`) | Asks the gate about every path that changed on disk. An escape ends the turn with "TURN ENDED". It detects; it does not prevent. | Yes (ends the turn) | Yes | WARNING per path it can't classify; DEBUG if the check raises |
| E2 | File-change summary | `engine/agent_loop.py:2178-2191` | Injects the claimed-versus-actual summary for the next turn | — | Yes | DEBUG |
| E3 | Turn teardown | `engine/agent_loop.py:990-1007` | Resets run paths; discards the verifier's turn record; ends the divergence task | — | — | DEBUG |
| E4 | Post-task learning hooks | `engine/agent_loop.py:4923-4939` (SkillCreator `daemon.py:305`, SkillRefiner `daemon.py:1927`) | Skill creation and refinement. **`run_async` surfaces only; awaited before `run_async` returns.** | Delays delivery | Yes (future prompts, via `skills/auto/`) | DEBUG |

### A.9 After the loop (surface code)

| ID | Default | Where | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| P1 | LCM persist and compaction trigger | `engine/session.py:232`, `:337` → `_persist_to_lcm` `:419`; compaction scheduled at `:286` | Durable write; background compaction | — | — | Best effort |
| P2 | Honesty correction (Telegram only) | `gateway/telegram.py:2048-2060` | Appends a correction to the reply | — | Yes | — |
| P3 | Teacher escalation (Telegram only) | `gateway/telegram.py:2064` (`:2126`) | Can replace the reply | — | Yes | — |
| P4 | Delivery | Telegram `gateway/telegram.py:2316`; web deltas `web/ws_server.py:1253-1262`; REST `web/server.py:4505`; Slack `gateway/slack.py:749`; Discord `gateway/discord.py:624` | Sends the reply | — | — | — |
| P5 | Turn-completed signal (push) | `web/ws_server.py:1342`, `gateway/telegram.py:2078` | Notification | — | — | — |

### A.10 Pipelines outside the turn

| ID | Default | Where | What it does | Blocks? | Changes? | On failure today |
|---|---|---|---|---|---|---|
| L1 | LCM compaction candidates | `memory/lcm_compaction.py:62` (`should_compact` `:147`; selection `:90-99`) | Oldest uncompacted rows minus the newest 32, in batches of 10; durable summaries | — | Only indirectly (`lcm_*` tools) | Breaker after 3 failures |
| L2 | Memory extraction | `memory/extractor.py:354`, started at `daemon.py:1774`; also LCM's pre-compaction flush | Every 1800 s; open-ended fact extraction into `memory.db` | — | Yes, through recall (T11) | ERROR (`log.exception`); model-call failures go to `silent_failures` |
| L3 | Skill draft (create) | `learning/skill_creator.py:214-291` | Stage 0 checks; Stage 1 model call returns SKIP or a SKILL.md; written to `skills/auto/` | — | Yes (future prompts) | DEBUG (E4) |
| L4 | Skill refine | `learning/skill_refiner.py:147-182`, write at `:256-260` | Rewrites the newest auto skill, after a backup | — | Yes | DEBUG (E4) |

---

## Appendix B. Today's hook machinery, as found

These are observations, not changes. This PR fixes none of them. Items 4, 5, 8 and 10 are reported as adjacent work.

1. **Events** (`hooks/events.py:13-19`). There are four: `session_start`, `session_end`, `pre_tool_use` and `post_tool_use`. A search of `src/` finds no site that fires `SESSION_START` or `SESSION_END`; they exist only as definitions.
2. **Kinds** (`hooks/schemas.py:15-63`).

   | Kind | Default timeout | Maximum timeout | `block_on_failure` default |
   |---|---|---|---|
   | `command` | 30 s | 600 s | false |
   | `prompt` | 30 s | 600 s | true |
   | `http` | 30 s | 600 s | false |
   | `agent` | 60 s | 1200 s | true |

   All four take an optional fnmatch `matcher`, which is compared against the tool name (`hooks/executor.py:207-212`).
3. **Execution** (`hooks/executor.py:59-73`). Hooks run one after another and are awaited on the event loop.
   - **`command`:** runs `/bin/bash -lc` with the payload as an environment variable. It is never spliced into the command text (`hooks/executor.py:215-279`). The timeout is enforced by killing the process (`hooks/executor.py:99-112`).
   - **`http`:** posts `{event, payload}` using httpx's timeout. Exceptions are caught.
   - **`prompt` and `agent`:** stream a request to the daemon's own provider.
4. **`prompt` and `agent` hooks have no working timeout.** *(Fixed in #571.)* `timeout_seconds` is declared (`hooks/schemas.py:31`, `:53`) but `_run_prompt_like_hook` (`hooks/executor.py:161-204`) never uses it, so a stalled provider stalls the tool call. These hooks also don't catch exceptions. A raise propagates out of `HookExecutor.execute` into `_execute_tool_call`, where `_safe_execute` (`engine/agent_loop.py:3387-3419`) turns it into "Tool X raised an exception":
   - at `pre_tool_use`, the call is refused and misreported as a tool exception;
   - at `post_tool_use` (`engine/agent_loop.py:4635`), **the tool has already run and its side effects landed, but the model is told it raised.**
5. **Command hooks inherit the daemon's whole environment** *(fixed in #571)* (`hooks/executor.py:93-96`), including provider keys and the API token. The docstring there says so. The payload itself is handled safely.
6. **Operator `pre_tool_use` sees the raw call** (`engine/agent_loop.py:3906`), before adapter repair, the markup guard, validation and the gate. `post_tool_use` sees the repaired call (`:4638-4642`).
7. **Hook results beyond "blocked" are ignored.** The loop reads only `pre.blocked` and `pre.reason` (`engine/agent_loop.py:3908-3922`). A hook's `output` never reaches the model, and `post_tool_use` results are discarded.
8. **Post-task hooks delay replies, and web/Beacon never runs them.** SkillCreator's Stage 1 model call (`learning/skill_creator.py:277`, awaited) runs inside `run_async` before it returns (`engine/agent_loop.py:4923-4939`). On Telegram, Slack, Discord and REST, the reply is therefore sent only after that call finishes, whenever Stage 0 passes (3–50 tool calls, no errors). Web/Beacon calls `run_loop` directly (`web/ws_server.py:1242`) and never runs these hooks.
9. **LSP diagnostics run after the telemetry row.** They are appended after the `tool_calls` row is written (C15 before C19), so the row never contains them.
10. **Several defaults fail at DEBUG,** below the WARNING-plus-row rule this contract sets for hooks:
    - divergence: `engine/agent_loop.py:969`, `:2571`, `:4630`, `:1007`;
    - the file-mutation verifier: `:2177`, `:2182`, `:2232`, `:2338`, `:997`;
    - steer drain: `:1511`;
    - periodic nudge: `:2773`;
    - LSP: `:4652` and `hooks/lsp_diagnostics.py:86`;
    - post-task hooks: `:4934`.

    Raising them to WARNING would change only the log level and telemetry, not the turn. It is still a change to default behavior, so it belongs in a separate change, not in this contract. Freezing v1 doesn't depend on it.
