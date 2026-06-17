# feat(coding): mid-run supervision — pause / inject / resume (Loop Manager Sprint 2, Phase 1)

Adds a **daemon control channel** for human supervision of a running coding task: pause the
loop at an episode boundary, inject a correction, and resume — without killing and restarting.
This is the daemon half (Phase 1). The Beacon supervise UI is **Phase 2, deferred** (it stacks
on the unmerged Loop-Manager-Sprint-1 Beacon PR #9). Spec:
`docs/sprints/SPRINT-LOOP-MANAGER-SPRINT2.md`.

## The non-negotiable invariant
**With no control issued, a coding run behaves byte-identically to today.** The channel is
dormant unless a `--control-dir` is set and a pause/inject is written. Proven: the existing
behavioral suite is **76 passed unchanged vs 76 with these changes** (stash-compared in one
consistent env). `engine/agent_loop.py` (`run_loop`) is **untouched** — "knobs, not surgery."

## Design (Phase-0 approved)
- **Seam = the EPISODE boundary** (`coding/session.py`, top of the `while True` loop), where
  the orchestrator already injects trust-tagged messages and the conversation is well-formed.
  Mid-episode is unsafe (it would corrupt `run_loop`'s tool-call↔result pairing) and stays
  **out of scope** (per-round-mid-episode = HARD, named, deferred).
- **Fork 1 — a polled control file** (`coding/control.py`). The run is a separate subprocess,
  so a file the subprocess polls at the seam is the minimal fail-safe cross-process channel.
  Daemon = sole writer; the run tracks applied injections in memory (no write contention).
- **Inject = `from_injected(text, provenance="supervisor", is_trusted=True)`** — actionable
  guidance (so the model follows it; NOT untrusted-DATA-bannered), yet **excluded from memory
  fact-mining by provenance** (`memory/extractor.py:177-180`). Two independent properties:
  trusted-so-the-model-obeys, provenance-excluded-so-it-doesn't-poison-the-fact-store.
- **Wall clock**: pause does NOT stop it (monotonic). A forgotten pause auto-abandons on the
  wall, detected from *inside* the pause loop — never orphaned.

## Components
- `coding/control.py` — fail-safe control protocol (parse/serialize, pure state transitions,
  reader/writer sides).
- `coding/session.py` — `RunControl` + the episode-seam check (pause-poll respecting the wall
  cap → abandon-from-pause; inject; `coding_control` telemetry per pause/inject/resume).
- `engine/messages.py` — `"supervisor"` added to the `Provenance` literal.
- `coding/managed.py` + `__main__.py` — `--control-dir` (every daemon run gets a per-run
  control dir, outside the sandbox; dormant unless used).
- `web/server.py` — `POST /api/code/{id}/pause|inject|resume`. Run-id resolution mirrors
  `/stop`; idempotent + non-destructive on already-in-state; **409** when terminal, **404**
  unknown, **400** blank inject. (No SecurityGate — no shell command is run; bearer auth is
  the boundary, the steer is trusted guidance.)

## Tests — 122 passed, adversarial
- **Default-unchanged**: 76 == 76 (byte-identical floor).
- **Fail-safe (the merge gate), written adversarially**: empty / garbage-bytes / truncated-json
  / truncated-mid-injection / **file-vanishes-mid-pause** → the run continues + completes; none
  wedge or hang. (Surfaced a real fix: `read()` now catches `UnicodeDecodeError`, not just
  `OSError` — garbage bytes raise a decode error.)
- **Paused-past-cap**: abandons on the wall **from inside the pause loop** (`model.calls == 0`).
- **Inject**: (a) reaches the next episode trust-tagged; (b) the **real** `MemoryExtractor`
  excludes the supervisor steer — it doesn't poison the fact store.
- **Pause/resume**: pause holds before the next episode; resume completes.
- **Endpoints**: round-trip + control-file side-effects, idempotent, 404/409/400.

## Live acceptance (real Gemma)
- **A · supervised** — held at the initial episode seam → injected an observable steer →
  resumed → the model **applied it** (`# STEERED-BY-SUPERVISOR` in the artifact) AND converged
  (acceptance exit 0); pause/inject/resume telemetry rows present.
- **B · no-control** — converged (exit 0), no steer marker (diff = the fix only). Byte-identical.
- **C · corrupt control file** — read fail-safe as empty → converged normally (exit 0).

## Scope / follow-ups
- **Beacon supervise UI** — Phase 2, deferred (stacks on PR #9): Pause/Resume/Inject in
  `LoopRunView` (not CodingPanel).
- Inject: structured-vs-freeform guidance (v1 freeform).
- Per-round-mid-episode injection remains **HARD / out** (would touch the shared `run_loop`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
