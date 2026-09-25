# Hook contract, v1

**Status:** v1, frozen 2026-09-24. Drafted in WP-1.1; the review decisions (WP-1.3) are recorded in [section 17](#17-decisions-wp-13). From here on, a change follows section 12: an additive change is a minor version, and anything else is a major version.
**Contract id:** `hooks/1.0`
**Part of:** Contracts v1, alongside the daemon API, the blackboard protocol and the memory schema.
**Scope:** How local pillars plug into a turn. Instinct makes fast, bounded decisions such as routing and tool choice. Cognition does open-ended work such as verification and planning. This document contains no code; the typed payload models land in WP-4.2.
**References:** File paths are relative to `src/prometheus/`, and line numbers are pinned to `origin/main` at `5d62383` (2026-09-24). A bare `:N` means the file named most recently in the same row or paragraph; with no file named, it means `engine/agent_loop.py`. The lines will drift; the names won't.

---

## 0. The contract in brief

1. With no pillar loaded, a turn runs exactly as it does today. Section 16 lists everything that runs and explains why nothing changes.
2. Built-in behavior (the default hooks, [Appendix A](#appendix-a-inventory-of-built-in-behavior)) always runs first, in a fixed order. No pillar replaces the security gate or the adapter.
3. A pillar can only add restrictions: escalate a tool call to the human, refuse it, or stop the run. It can never let through anything the gate blocked or sent for approval.
4. A pillar changes a value in one way only: by answering a **decision point** with an option id from a table that open code built. It may do that only where the default abstains, and only when the operator names that pillar for that point in config. Every decision flag is off by default. A pillar's route choice never sends conversation content to a hosted API unless the operator allows cloud routes.
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

Every field in the per-event schemas carries one source label. The typed models (WP-4.2) carry it as field metadata, and the wire format enforces it by placement.

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
- **`escalate` when no one can approve.** On a system-origin call there is nobody to ask, so `escalate` becomes `veto`. This matches how the loop treats an approval with no prompt available (`:4424-4438`).

**A pillar veto is best-effort.** It does not apply if the pillar times out, errors or is tripped. The security gate is the boundary. Nothing that must be blocked may depend on a pillar.

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
| `backend.version` | none | **New.** WP-4.2 adds it to `Choice`, and `RuleChooser` reports its own version. |

Validation is `validate_choice` (`computer/candidates.py:156`), generalized:
- an empty choice is **invalid**;
- `abstain` means use the default's fallback;
- an id that isn't in the table is **invalid**, a protocol violation;
- a `table_id` that doesn't match is **stale**.

Invalid and stale decisions are never applied. They are logged at WARNING and recorded. The daemon maps a valid id to the option it built itself. Nothing in the decision is merged into an action.

WP-4.2 should move `ChoiceRequest` and `Choice` into one shared module, with computer use as its first client. There should be one type, not two that drift apart.

### 6.2 Authority regions

This follows the 2026-09-20 ruling:

- **The default answers first and wins wherever it answers.** In that region no pillar is consulted for the decision. A pillar may still observe the event, and may veto where the event allows it.
- **A pillar decides only where the default abstains**, and only when `pillars.decide.<point>` names that pillar. Every point is off by default, and at most one pillar may be named per point. `tool_choice` is also locked off until WP-1.2 has measured its cost (section 6.3).
- If the pillar abstains, times out, errors, answers invalid or stale, is busy or is tripped, the result is **the default's own fallback, which is today's behavior** in that region.
- **A timeout limits delay. It is not what keeps things safe**, because a fast wrong answer passes a timeout. What keeps a pillar's decision safe:
  1. open code builds the option table;
  2. the returned id is validated against that table;
  3. every downstream default still runs: each tool call the model makes afterwards still goes through the adapter, the security gate and the approval prompt;
  4. the operator turned the point on;
  5. for `route`, the table holds no hosted API unless the operator allowed cloud routes (decision 17).

### 6.3 Decision points in v1

| Point | Event | Options (built by) | Default | Default abstains when | Fallback |
|---|---|---|---|---|---|
| `route` | `turn_start` | The primary, plus routes served by a box the operator runs. Cloud routes are included only with `pillars.decide.route_allow_cloud: true` (router) | `ModelRouter.route` (`router/model_router.py:747`) | The router reaches its primary branch (`:791-792`) because no per-session override, escalation, smart-routing or task rule answered | The primary, as today |
| `tool_choice` | `before_model_call` | `auto`, `none`, `required`, and `tool:<name>` for each advertised tool (loop) | The caller's `tool_choice`, or the one resolved from `mode` (`engine/agent_loop.py:1076`), with first-round forcing (`:1086-1091`, `:1559-1561`) | The round's directive is `auto` and came from `mode`, not from the caller | `auto`, as today |
| `compaction_span` | `compaction_candidates` | Cut points the compactor considered | `_select_span_end` (`context/compactor.py:504`) | **Never, today** | — |
| `skill_draft` | `skill_draft_score` | `keep`, `discard` | SkillCreator Stage 0/1 plus the name-collision check (`learning/skill_creator.py:214-291`) | **Never, today** | — |

Constraints that apply to specific points:

- **`route`: the person's choice wins.** A per-session user override (`/claude`, `/local`, …) always answers, so a pillar can never overrule the person's own choice of model.
- **`route`: local unless the operator opts in** (decision 17). A pillar's choice must never send conversation content to a hosted API, or spend money, without the operator opting in. So the option table holds:
  - the primary, which is where the turn goes anyway when the pillar abstains. If the primary is a hosted API, picking it changes nothing: the operator already sends every unrouted turn there;
  - routes served by a box the operator runs: the providers in `_LOCAL_PROVIDERS` (`providers/registry.py:167`), and named backends, which may only use local providers (`providers/backends.py:95`);
  - any other route, including every hosted API, **only** when `pillars.decide.route_allow_cloud` is `true`. The default is `false`.

  A route whose provider can't be classified is treated as cloud and left out.
- **`tool_choice`: locked until measured.** This flag can't be turned on anywhere, on any install, until the parity harness (WP-1.2) has measured the llama.cpp prefix-cache cost of a forced round. A forced round withholds native tools so the grammar path fires (`:1572-1579`), and that changes the cached prompt prefix. The implementation (WP-4.2) ships the flag locked: a config that sets it loads with the flag off, logs one WARNING, and doctor shows ✗ naming the missing measurement. Lifting the lock is an edit to this section that records the measurement.
- **`tool_choice`: one round only.** A pillar-set directive binds one round, and a pillar may set at most one round per turn. This mirrors first-round forcing.
- **`tool_choice`: forcing must never raise.** Today a forced `{tool: X}` that the provider doesn't honor raises (`:2032-2040`). A round forced by a pillar must instead record the outcome as `not_honored` and continue.
- **No `tool_name` point in v1** (decision 9). When adapter repair fails today, the model is told and retries (`:3988-4048`). That is not a dead end. So a pillar-chosen tool name would save at most one round, at the cost of running a tool the model didn't name. Section 18 says what evidence would reopen it.
- **`compaction_span` and `skill_draft` are defined but unreachable.** Their defaults always answer today. They become reachable only if a default gains an abstain band. That is a change to default behavior and needs its own WP (decision 12). In v1, `skill_draft_score` offers `veto` (discard) instead, which is a restriction and needs no decision.

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
- route a turn to a hosted API, or to anything that costs money, unless the operator set `pillars.decide.route_allow_cloud`;
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

These numbers are adopted as provisional (decision 2). WP-4.x re-sets them from measurements.

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

A surface that shows check status (for example "verified") must read it from the row, and a `timeout` must display as "not checked". A result that arrives after its deadline is discarded and recorded as `late`. It is never applied to a later event.

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

**Host environment.** A host process gets a minimal environment. It does **not** inherit the daemon's environment, which holds provider API keys and the daemon token. Today's command hooks do inherit it (`hooks/executor.py:93-96`; see Appendix B). Pillars must not.

---

## 12. Versioning and the manifest

- The contract version is `hooks/MAJOR.MINOR`. This document is `hooks/1.0`.
- **A minor version** may add optional payload fields, events, decision points, outcome codes, or transports. It never removes, renames or retypes anything, and never widens an existing event's capabilities. A pillar only ever receives events it subscribed to, and new optional fields must be safe to ignore.
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

**Config** (all keys are new; they land with WP-4.2):

```yaml
pillars:
  load: []                  # pillars to load, in declared order. Empty or absent = none.
  decide:                   # at most one pillar per point; false = off (the default)
    route: false
    route_allow_cloud: false  # true lets the route table include hosted APIs (section 6.3)
    tool_choice: false        # locked off until WP-1.2 measures a forced round's cache cost (section 6.3)
  budgets:
    instinct_deadline_ms: 100
    cognition_deadline_ms: 5000
    cognition_turn_budget_ms: 15000
    pipeline_deadline_ms: 60000
```

Installing a package is not enough for a pillar to load: it must also be listed in `pillars.load`. An installed but unlisted pillar is inert. Doctor lists it as "installed, not enabled".

---

## 13. Observability

**Every pillar hook call and every decision-point consultation writes one row**, in a new `hook_calls` table in `telemetry.db` (decision 15). Its DDL lands with WP-4.x. Columns:

| Column | Notes |
|---|---|
| `id`, `timestamp`, `contract` | |
| `event`, `pillar`, `pillar_version`, `hook`, `transport` | |
| `session_id` | For description only. NULL for ephemeral sessions, the same rule `tool_calls` follows (`engine/agent_loop.py:4588`). |
| `turn_id`, `round_index` | |
| `deadline_ms`, `duration_ms` | `duration_ms` is NULL when nothing was measured (`busy`, `tripped`), following the schema-v2 rule for `tool_calls.latency_ms`. |
| `outcome` | `ok`, `abstain`, `timeout`, `error`, `invalid`, `stale`, `late`, `busy`, `tripped`, `crashed`, `dropped`, `budget_exhausted`, `not_honored` |
| `verdict`, `applied` | `applied` is 1 only when the daemon acted on the result. It is never 1 when the outcome isn't `ok`. |
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
- each pillar's subscriptions (event, hook, capabilities), each decision point with its flag state, and `route_allow_cloud`;
- health over the last 24 h: number of calls, p50/p95 duration, and counts of timeouts, errors, busy, crashes and drops, plus whether the pillar is tripped;
- operator hooks from `hooks:`: event, kind, matcher, `block_on_failure`, timeout.

Doctor's exit code follows its existing rule (non-zero on any ✗, `cli/doctor.py:1050`):
- a configured pillar that fails to load is ✗;
- an operator hook configured on `session_start` or `session_end` is ✗ **"configured, never fires"** (decision 4);
- `pillars.decide.tool_choice` set while it is still locked is ✗, naming the missing WP-1.2 measurement (section 6.3);
- a tripped pillar, or a p95 over target, is a warning.

The decision-4 WARNING and ✗ are the v1 behavior. Their code lands with WP-4.2.

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
- **Routing stays local unless you opt in.** A pillar's route choice can't send your conversation to a hosted API, or spend money, unless you set `pillars.decide.route_allow_cloud` (section 6.3, decision 17). Your own per-session model choice always wins.
- **What a compromised or buggy pillar can do:**
  - see content;
  - delay a turn by up to its deadlines;
  - stop turns (a denial of service);
  - put labeled notes in front of the model, which can steer it;
  - decide within the option tables of the decision points the operator turned on.
- **What it cannot do:** execute a tool, change a tool call's arguments or which tool runs, get past the gate or the approval prompt, route to a hosted API you didn't allow, or write memory.

---

## 15. Today's operator hooks and pillar hooks

**What exists today** (the full detail is in Appendix B):
- Operator hooks are defined under `hooks:` in `prometheus.yaml`, loaded by `hooks/loader.py:32` and run by `HookExecutor.execute` (`hooks/executor.py:59`).
- They run one after another, awaited on the event loop.
- There are four kinds: `command`, `prompt`, `http`, `agent`.
- Only `pre_tool_use` and `post_tool_use` fire.
- A hook can block a call (`block_on_failure`) or observe. It cannot annotate or modify.
- `prompt` and `agent` hooks send the payload to the daemon's configured model provider, which may be a hosted API. `http` hooks may target any URL.
- So operator hooks do not meet the pillar rules: they are not local-only, they do not fall back to the default on failure, and prompt and agent hooks have no working deadline.

**The options**
- **A. One registry.** Operator hooks and pillar hooks are two kinds of entry in one registry and one dispatcher. They share the event catalogue, ordering, telemetry and doctor listing, and keep separate rules.
- **B. Separate registries.** The `HookExecutor` stays as it is, and a new pillar host sits beside it.

**Decided: A, one registry with two kinds** (decision 5). Operator hooks keep their current semantics, positions and payloads in v1. The reasons:

1. **One answer to "what runs here, and in what order".** Two registries at the same event means two orderings, and doctor and telemetry would have to reconcile them. Two parallel copies of machinery that drift apart is the defect the loop already names "the two-loop defect" (CROSS-CUTTING §2, `engine/agent_loop.py:957`).
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

**Operator session hooks (decision 4).** These can be configured today, and nothing fires them (Appendix B.1). In v1 they still don't fire, because firing them would change behavior for anyone who configured them. But they are no longer silent: the boot WARNING and the doctor ✗ tell the operator their hook does nothing. The code for both lands with WP-4.2.

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
  - the same missing timeout for `prompt` and `agent` hooks;
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
3. **Every decision point needs a flag that names a loaded pillar.** With none loaded, every flag resolves to off, and the default's own fallback runs. That is the code path that runs today.
4. **Nothing is written during a turn.** There are no telemetry rows, files or sockets.

**What proves it is not this document.** The proof is the full existing suite passing unchanged, identical golden-trace replay, and overhead inside the noise band, all run on the implementation (WP-4.x). Until then, "identical" is a design requirement, not a measured result.

---

## 17. Decisions (WP-1.3)

These were decided in review on 2026-09-24, and they close WP-1.1 and WP-1.3. Each entry says whether it was accepted as recommended, changed in review, or added in review, and where this document applies it.

1. **How Instinct runs.** *Changed in review.*
   - `host` is the default for Instinct pillars.
   - `inproc` is removed from v1 entirely, because it can't meet "a pillar crash never takes the daemon down".
   - It may return in a later minor version, but only if the measured `host` overhead misses the Instinct budget.

   Applied in sections 10, 11 and 18.
2. **Budget numbers.** *Accepted.* The deadlines, targets and thresholds are provisional, and WP-4.x re-sets them from measurements (section 9.1).
3. **Ephemeral sessions.** *Accepted.* Pillars get no `content` for ephemeral sessions (section 4.3).
4. **Operator hooks on `session_start` and `session_end`.** *Changed in review.*
   - They don't fire in v1.
   - They aren't left silent either: if any are configured, boot logs one WARNING and `oara doctor` shows ✗ "configured, never fires".
   - This is the v1 behavior, and its code lands with WP-4.2.

   Applied in sections 13, 15 and 16.
5. **Registry.** *Accepted.* One registry with two kinds (section 15).
6. **Where operator `pre_tool_use` hooks run.** *Accepted.* Unchanged in v1. Moving them after the gate is a later minor version (section 18).
7. **What "escalate" means.** *Accepted.* It means "require the human's approval". A pillar can't ask for a stronger model in v1 (section 5).
8. **The `tool_choice` decision.** *Accepted, with an addition.*
   - Forcing is allowed behind the flag, for at most one round per turn, and a pillar-forced round never raises.
   - **Addition:** the flag can't be turned on anywhere until the parity harness (WP-1.2) has measured the llama.cpp prefix-cache cost of a forced round.

   Applied in section 6.3.
9. **The `tool_name` decision.** *Changed in review: removed from v1.*
   - When adapter repair fails today, the model is told and retries. That is not a dead end.
   - So the upside is at most one saved round, and the downside is running a tool the model didn't name.

   Applied in sections 3 and 6.3. Section 18 lists the evidence needed before it is reconsidered.
10. **Provenance for delivery notes.** *Accepted.* A new value, `pillar`, untrusted by default, added to the `Provenance` set through the memory schema contract (section 5).
11. **Pillar-written compaction summaries.** *Accepted.* Not in v1.
12. **The unreachable decision points** (`compaction_span`, `skill_draft`). *Accepted.* They stay inert in v1. Making either reachable is a change to default behavior with its own WP (section 6.3).
13. **HTTP transport.** *Accepted.* Unix sockets only in v1. The loopback, mutual-HMAC design is kept as the plan for when `http` is added (sections 11 and 18).
14. **The daemon token.** *Accepted.* It is never sent to a pillar, and is used only for pillar-to-daemon calls (section 11).
15. **Where the telemetry goes.** *Accepted.* A new `hook_calls` table (section 13).
16. **Post-processing after the loop, per surface.** *Accepted.* This is a known limit of v1. Two things are separate decisions: whether the Telegram-only steps move into the loop, and whether post-task hooks move off the delivery path (sections 3.1 and 9.1).
17. **Cloud routes.** *Added in review.*
    - The `route` option table includes only local providers unless the operator allows otherwise, with `pillars.decide.route_allow_cloud` (default `false`).
    - A pillar's choice must never send conversation content to a hosted API, or spend money, without the operator opting in.
    - The per-session user override still always wins.

    Applied in sections 6.3, 7, 12 and 14.

---

## 18. Possible later minor versions

None of these is in v1. Each would need its own decision and a minor version bump (section 12). Each is listed with what has to be true before it is considered.

| Candidate | Before it is considered |
|---|---|
| `inproc` transport (decision 1) | The measured `host` transport overhead misses the Instinct budget (section 9.1). |
| `tool_name` decision point (decision 9) | Evidence of two things. First, how often adapter repair fails: `tool_calls` rows with `error_type = 'validation_failed'` (`engine/agent_loop.py:4024`). Second, what the model does next: whether its retry succeeds and in how many rounds. `retry_success` repair pairs in `training.db` record the recoveries (`:3996-4003`). |
| `http` transport on loopback (decision 13) | A real pillar that can't use a unix socket. Section 11 has the design. |
| Operator `pre_tool_use` hooks after the gate (decision 6) | A decision to change what existing operator hooks see. After the move, they would see the repaired call that will actually run. |
| Firing operator `session_start` / `session_end` hooks (decision 4) | A decision to change behavior for operators who configured them. Until then, the boot WARNING and doctor ✗ tell them the hooks never fire. |

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
4. **`prompt` and `agent` hooks have no working timeout.** `timeout_seconds` is declared (`hooks/schemas.py:31`, `:53`) but `_run_prompt_like_hook` (`hooks/executor.py:161-204`) never uses it, so a stalled provider stalls the tool call. These hooks also don't catch exceptions. A raise propagates out of `HookExecutor.execute` into `_execute_tool_call`, where `_safe_execute` (`engine/agent_loop.py:3387-3419`) turns it into "Tool X raised an exception":
   - at `pre_tool_use`, the call is refused and misreported as a tool exception;
   - at `post_tool_use` (`engine/agent_loop.py:4635`), **the tool has already run and its side effects landed, but the model is told it raised.**
5. **Command hooks inherit the daemon's whole environment** (`hooks/executor.py:93-96`), including provider keys and the API token. The docstring there says so. The payload itself is handled safely.
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

    Raising them to WARNING would change only the log level and telemetry, not the turn. It is still a change to default behavior, so it belongs in its own WP, not in this contract. Freezing v1 doesn't depend on it.
