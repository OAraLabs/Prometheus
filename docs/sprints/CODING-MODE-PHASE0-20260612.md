# Coding Mode v2 — Phase 0 Survey Report (HALTED AT GATE)

**Date:** 2026-06-12 · **Branch:** `feat/coding-mode` @ `7493044` (post-F1 main; clean, 0/0 vs origin)
**Spec:** `docs/sprints/SPRINT-coding-mode-v2.md` · **Prerequisite F1:** MERGED (#31 `aae3742` + #32 `7493044`), daemon live, usage rows verified with real tokens.
**Outcome:** three of four gates PASS; the reasoning-budget gate FAILS → sprint halted per spec
("Verify bounded --reasoning-budget on the server; HALT if absent"). No Phase 1+ code written.

---

## Gate 1 — Managed-task registration path: PASS

`BackgroundTaskManager` (`src/prometheus/tasks/manager.py:55`):
- `create_shell_task` (:88) — SecurityGate vetting at system trust BEFORE spawn (:107),
  durable `TaskStore` persist (:124/:132), output file, timeout enforcement.
- `create_agent_task` (:136) — **accepts a `command=` override (:145, :153)** so no
  ANTHROPIC_API_KEY is needed; prompt delivered via task stdin (`write_to_task`, :180).
- `task_completed`/`task_failed` SignalBus emission on resolution (:466-472) → Telegram/Beacon
  notification path already live.

**Plan consequence:** a coding run = `create_agent_task(command="… -m prometheus code …", …)`.
No new TaskType literal needed; durability, stop, reap, notify all inherited.

## Gate 2 — Per-call thinking kwarg path: PASS

- `ApiMessageRequest.suppress_thinking: bool | None` exists (`providers/base.py:30`, shipped 1c1ba6c).
- `LlamaCppProvider` resolves request-override ?? provider-default (`llama_cpp.py:177-179`) and
  sends both `thinking`/`enable_thinking` template kwargs.
- The loop builds requests WITHOUT setting it (`agent_loop.py` request construction at the
  envelope-wrapped call site) → Phase 2 = one `LoopContext` field threaded into the request.
- **F1's `thinking` column records the effective flag per call** — the test assertion surface
  the spec asks for is already live (verified: live rows show `thinking=0` under the suppressed
  default).

## Gate 3 — Can the loop host a 30-round bounded session: PASS (no structural surgery)

- `max_turns`: per-context (`LoopContext.max_turns`; AgentLoop default 200). 30 rounds fits.
- `max_tool_iterations`: per-context (default 25 via AgentLoop; `_effective_max_tool_iterations`
  resolves per provider tier). A 30-round coding session sets its own (e.g. 120). Knob, not surgery.
- Wall-clock: no in-loop cap exists; the coding policy layer owns it, and managed-task
  `timeout_seconds` backstops at process level.
- **Circuit breaker** (`agent_loop.py:744`, hardcoded `max_identical=3, max_any=5`): trips only
  when EVERY result in a dispatch is `is_error=True`. Design consequence (decided): `code_run`
  returns failing test runs as `is_error=False` with the exit code in the output — command-ran-
  and-reported is a tool SUCCESS; `is_error` is reserved for sandbox/timeout/launch failures.
  Iterate-to-green therefore never feeds the breaker. No breaker change needed.
- Empirical: the bakeoff drove this loop 25 rounds externally capped; thinking-on mean 8.2 rounds.

## Gate 4 — Bounded `--reasoning-budget` on the server: **FAIL → HALT**

Live process (verified over SSH, `llama-server.service` on the GPU box, PID 2676768,
started 2026-06-11 16:42:32):

```
llama-server -m google_gemma-4-26B-A4B-it-Q4_K_M.gguf --mmproj mmproj-BF16.gguf
  --ubatch-size 2048 --batch-size 2048 --port 8080 --host 0.0.0.0
  -ngl 99 -c 81920 --parallel 1 --flash-attn on --jinja
```

**No `--reasoning-budget` flag.** b8660's default is `-1` = unrestricted (verified via
`llama-server --help` on the box: `N>0 for token budget` IS supported in this build, alongside
`--reasoning-budget-message`). `/props` does not expose the setting; the process args are
authoritative.

### Erratum this surfaces (recorded in the addendum file)

`BAKEOFF-ADDENDUM-thinking-on-20260611.md` disclosed its thinking-on arm as "bounded server-side
--reasoning-budget". The server it ran against is THIS process (started 16:42, addendum runs that
evening) — **the addendum's 30 thinking-on runs were unbounded.** Direction unaffected (thinking
helped even unbounded; mean wall 36s, zero runaway, 0 capped), but the cost-rail claim was wrong,
and those 30 runs are now the empirical base rate for choosing a budget value.

### The one-line fix (owner action, GPU box)

Edit `llama-server.service` ExecStart to add e.g. `--reasoning-budget 2048`
(optionally `--reasoning-budget-message "…wrap up reasoning…"`), then
`sudo systemctl daemon-reload && sudo systemctl restart llama-server`.
Verify: `ssh oara-4090@<gpu-box> 'ps aux | grep reasoning-budget'` (≈45 s model reload;
all daemon traffic pauses during the restart).

Budget-value inputs: addendum thinking-on runs averaged 8.2 rounds / 36 s wall unbounded on this
model; 2048 thinking tokens per call is a generous start (tune later — it's a flag).

## Resume instructions

1. Owner sets the budget flag (above) — or explicitly waives the gate in writing.
2. Re-verify: process args show the flag.
3. Phase 1 begins on this branch: editor+execution tools (`code_view/str_replace/create/grep/
   glob/run`, strict schemas, no extra GBNF layer) + ProcessSandbox (FULL CLONE jail, decided).
4. Phase 2: `LoopContext.suppress_thinking` → request (one field); assert via F1 envelope rows.
5. Phase 3: iterate-to-green policy (done-is-a-verdict; failure-signature step-back; native
   prompt text — no OpenHands text adaptation planned, keeping the provenance gate unexercised).
6. Phase 4: managed-task wiring + t11–t15 acceptance via the bakeoff runner
   (`~/bakeoff-harness/`, fixture marshmallow @ 27bfa77), ≥3/5 to merge.
