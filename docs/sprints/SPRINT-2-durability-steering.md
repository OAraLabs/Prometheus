# SPRINT — Durability & Steering

**Type:** Implementation sprint
**Branch:** `feat/durability-steering` (off main after Sprint 1 merges)
**Estimated time:** 4–6 focused days
**Prerequisites:** Sprint 1 (Visible Memory & Skills) merged to main
**Trust level:** Internal code changes. Touches the agent loop — careful regression testing required.

---

## Why this sprint exists

Two of the biggest "Hermes feels reliable" experiences come from features Prometheus doesn't have:

1. **Mid-turn redirection.** When the agent goes off-track during a long autonomous run, Hermes users can `/steer "actually focus on X"` and the redirect arrives at the agent after the next tool call — no kill, no restart, no lost context. Prometheus today has only "kill the daemon and start over."

2. **Catching silent failure.** When Gemma claims it wrote a file but didn't, when a `bash` command exited 0 but produced no output, when a tool result said "success" but the side effect didn't happen — these are silent failures the Adapter Layer doesn't catch because the *response shape* was fine. Hermes ships a per-turn file-mutation verifier that surfaces these directly.

Both items are on the existing backlog (#5 and #7 in the post-verification disposition). This sprint ships them together because they share the agent-loop integration surface.

---

## Read These Files First

### Agent loop
- `src/prometheus/engine/agent_loop.py` — the core turn cycle
- `src/prometheus/engine/messages.py` — message types and append helpers
- `src/prometheus/engine/query.py` or `query_engine.py` — turn execution

### Hooks
- `src/prometheus/hooks/executor.py` — Pre/PostToolUse pipeline (do not change semantics, only extend)
- `src/prometheus/hooks/events.py` — hook event types
- `src/prometheus/hooks/registry.py` — hook registration

### Tools
- `src/prometheus/tools/builtin/file_write.py`
- `src/prometheus/tools/builtin/file_edit.py`
- `src/prometheus/tools/builtin/notebook_edit.py`
- `src/prometheus/tools/builtin/bash.py` — for detecting mv/rm/cp side effects

### Gateway
- `src/prometheus/gateway/telegram.py` — for adding `/steer` and `/queue` command handlers
- `src/prometheus/gateway/queue.py` (if it exists) — message queue patterns
- Session state — find where per-session state is held during agent loop execution

---

## Work Stream 1: Mid-turn `/steer` and `/queue`

### Goal

A user mid-conversation can inject text that the agent receives **after the next tool call completes but before the next model call** — no interrupt, no broken turn, no new user-message-flagged input.

### What to build

#### Session state additions

In whatever module holds per-session agent state, add two queues:

```python
@dataclass
class SessionState:
    # ... existing fields ...
    queued_steers: list[str] = field(default_factory=list)
    queued_prompts: list[str] = field(default_factory=list)
```

- `queued_steers` — drained on every loop iteration, appended as a system-message addendum *to the upcoming model call*
- `queued_prompts` — drained at end of turn, queued as the next user turn (or fired immediately if the agent has stopped tool-calling)

#### Agent loop integration

In `src/prometheus/engine/agent_loop.py`, after each tool result is appended and before the next model call:

```python
# Existing: append tool results to messages
# Existing: prepare next model call

if session_state.queued_steers:
    steer_text = "\n\n".join(session_state.queued_steers)
    session_state.queued_steers.clear()
    # Append as a system-message addendum, NOT as a user message
    # This preserves the tool-calling cycle without flagging it
    # as a fresh user turn
    messages.append({
        "role": "system",
        "content": f"[STEER FROM USER, mid-turn]: {steer_text}"
    })

# Then proceed with next model call as normal
```

After the turn ends (model produces a non-tool-use response, or `/goal`-style condition completes):

```python
if session_state.queued_prompts:
    next_prompt = session_state.queued_prompts.pop(0)
    # Treat as a fresh user turn
    return await self.run_turn(next_prompt, session_state=session_state)
```

#### Telegram command handlers

In `src/prometheus/gateway/telegram.py`:

```python
@command("/steer")
async def cmd_steer(ctx, text: str):
    """Inject text into the current session, arrives after next tool call."""
    session = get_active_session(ctx.user_id)
    if not session:
        return "No active session. Use /steer during a running task."
    session.queued_steers.append(text)
    return f"📍 Steered: {text[:80]}{'...' if len(text) > 80 else ''}\n   Will arrive after next tool call."

@command("/queue")
async def cmd_queue(ctx, text: str):
    """Queue text to fire as a new turn when the current one ends."""
    session = get_active_session(ctx.user_id)
    if not session:
        return "No active session."
    session.queued_prompts.append(text)
    return f"📥 Queued: {text[:80]}{'...' if len(text) > 80 else ''}\n   Will fire when current turn ends. Position: {len(session.queued_prompts)}."

@command("/status")
async def cmd_status_extension(ctx):
    """Extended to show queued items."""
    # ... existing status content ...
    if session.queued_steers:
        out += f"\n📍 Queued steers: {len(session.queued_steers)}"
    if session.queued_prompts:
        out += f"\n📥 Queued prompts: {len(session.queued_prompts)}"
    # Show preview of each
```

Also add `/unqueue` and `/clear-steers` for cancellation.

### Concurrency safety

The queue drain happens on the agent-loop thread/task; the queue append happens on the Telegram-gateway thread/task. Use `asyncio.Lock` or a thread-safe queue (`queue.Queue` if the daemon is sync-with-threads, `asyncio.Queue` if it's async). Confirm the pattern matches the rest of the daemon's concurrency model.

### Tests

- `test_steer_appends_to_next_model_call` — `/steer` arrives in the system message of the next model call
- `test_steer_does_not_break_tool_cycle` — agent continues its tool-calling loop without restart
- `test_queue_fires_as_next_turn` — `/queue` text becomes the next user turn after current turn ends
- `test_multiple_steers_concatenate` — multiple `/steer` calls before next tool result all appear
- `test_status_shows_queued` — `/status` includes queued counts and previews
- `test_unqueue_clears` — `/unqueue` empties the prompt queue

---

## Work Stream 2: Per-Turn File-Mutation Verifier

### Goal

At the end of each turn, the agent receives a system-message addendum summarizing every filesystem mutation that happened during the turn — what the code *says* it did vs what actually landed on disk. This catches the high-frequency Gemma failure where the model claims it wrote a file but the write silently failed or went to the wrong path.

### What to build

#### New hook stage

Extend the hook system with a `PostTurn` stage (if it doesn't exist) — fires once at the end of a turn, not after each tool. Place this between turn completion and next model call.

#### File mutation tracker

Create `src/prometheus/hooks/file_mutation_verifier.py`:

```python
class FileMutationVerifier:
    """Tracks claimed vs actual filesystem changes per turn.

    Intercepts tool calls that touch the filesystem:
    - file_write
    - file_edit
    - notebook_edit
    - bash (when command contains mv, rm, cp, touch, mkdir, > redirect, >> redirect)

    For each, records:
    - claimed_path: what the tool said it touched
    - claimed_action: write / edit / delete / create
    - actual_state: os.stat() result before and after

    At PostTurn, emits a summary:

    📁 Files touched this turn:
       ✓ src/prometheus/foo.py — modified (+47 lines)
       ✓ /tmp/scratch.txt — created (823 bytes)
       ⚠ docs/notes.md — CLAIMED modified, NO CHANGE ON DISK
       ✗ /etc/hosts — CLAIMED write, FAILED (permission denied)
    """
```

#### Integration with the hook system

Register as PreToolUse (to snapshot) and PostToolUse (to verify) for the relevant tools. Then aggregate per-turn for PostTurn emission.

```python
def pre_tool_use(self, tool_name, tool_input, context):
    if tool_name in FS_TOOLS:
        path = self._extract_path(tool_name, tool_input)
        context["fmv_snapshot"] = self._snapshot(path)

def post_tool_use(self, tool_name, tool_input, tool_output, context):
    if tool_name in FS_TOOLS:
        path = self._extract_path(tool_name, tool_input)
        post_state = self._snapshot(path)
        self.turn_mutations.append({
            "tool": tool_name,
            "path": path,
            "claimed": self._claim_from_output(tool_output),
            "before": context.get("fmv_snapshot"),
            "after": post_state,
        })

def post_turn(self, messages, context):
    if not self.turn_mutations:
        return  # Nothing to report
    summary = self._format_summary()
    messages.append({
        "role": "system",
        "content": summary
    })
    self.turn_mutations.clear()
```

#### Bash command parsing

For bash commands, parse the command line for filesystem operations:

```python
FS_BASH_PATTERNS = [
    (r'\bmv\s+(\S+)\s+(\S+)', 'move'),
    (r'\brm\s+(?:-\w+\s+)?(\S+)', 'delete'),
    (r'\bcp\s+(\S+)\s+(\S+)', 'copy'),
    (r'\btouch\s+(\S+)', 'create'),
    (r'\bmkdir\s+(?:-\w+\s+)?(\S+)', 'mkdir'),
    (r'>\s*(\S+)$', 'redirect_write'),
    (r'>>\s*(\S+)$', 'redirect_append'),
]
```

Track each match and snapshot the affected path. This is heuristic; it's fine if bash patterns it can't parse just don't get tracked (better than false positives).

#### Configuration

```yaml
hooks:
  file_mutation_verifier:
    enabled: true
    show_in_telegram: false  # quiet by default — only fed to model
    truncate_after_n_mutations: 20
```

### Tests

- `test_fmv_detects_file_write` — file_write tool call shows in next-turn summary
- `test_fmv_detects_bash_redirect` — `echo foo > /tmp/x` shows in summary
- `test_fmv_detects_silent_failure` — file_write claims success but file unchanged → flagged with ⚠
- `test_fmv_detects_permission_denied` — file_write fails with EACCES → flagged with ✗
- `test_fmv_empty_turn` — no mutations → no summary appended
- `test_fmv_truncation` — >20 mutations get summarized to top 20 + count

---

## Acceptance criteria for the whole sprint

A user on Telegram should be able to:

1. Send a long autonomous task (e.g. "research how to deploy Prometheus on a VPS, then write a setup guide")
2. Mid-task, send `/steer focus on Ubuntu-only, skip the Mac instructions`
3. See the agent acknowledge the steer in its next tool result or response **without restarting the turn**
4. Send `/queue and after that, write a follow-up about Tailscale integration` while the first task is still running
5. Have the second task fire automatically when the first completes

For the verifier:

6. Ask the agent to write three files. Watch the next-turn system message confirm all three landed.
7. Force a silent failure (e.g. `chmod 444` on the target dir, then ask agent to write there). The verifier should flag the failure even if the tool result returned "success."

---

## Constraints

- **Branch `feat/durability-steering`.** Off main after Sprint 1 merges.
- **No commits to main.** Will squash-merges.
- **Do not change Adapter Layer behavior.** This sprint adds hooks and queues; the validator/repair/retry logic stays untouched.
- **Steer is a system-message addendum, not a user turn.** Critical distinction — it preserves the tool-calling cycle.
- **Verifier is opt-out via config but on by default.** Failure-catching is the whole point.
- **All new tests in the existing test layout.** If `tests/test_wiring.py` is being split (Sprint 3 candidate), add new tests under the existing structure; the migration can renumber them later.
- **Test the silent-failure case specifically.** Don't ship the verifier if it can't catch the case that justified building it.

---

## Reporting back

1. Branch + commit SHAs
2. Confirmation that `/steer` arrives mid-turn without breaking tool-calling
3. Confirmation that the verifier catches at least one synthetic silent-failure case
4. Any session-state concurrency surprises you hit (this is the trickiest integration point)
5. Drive-by findings — note, don't fix
