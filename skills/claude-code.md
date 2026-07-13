---
name: claude-code
description: Delegate coding tasks to the Claude Code CLI (`claude`). Covers when to delegate to Claude Code vs handling in-Prometheus, prompt structure, one-shot vs interactive invocation, working directory handling, and result aggregation.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/claude-code.md | MIT -->

# Delegating to Claude Code CLI

The `claude` CLI (Anthropic's Claude Code) is installed on this host (`/usr/bin/claude`). Prometheus can delegate coding work to it via the `bash` tool, treating Claude Code as a specialized subagent for repo-level changes.

## When to Use

✅ **DELEGATE to Claude Code when:**

- Multi-file refactors with cross-file consistency requirements
- Full PR workflows: branch → implement → test → commit → PR
- Repo-level reasoning that benefits from Claude Code's own tool stack (Edit, Read, Bash, Grep with its native indexing)
- Work that would crowd Prometheus's context if done inline (>500-line diffs, codebase exploration)
- The user explicitly asks for "Claude Code" or "claude" by name

## When NOT to Use

❌ **DO IT IN PROMETHEUS when:**

- The change is a single-file edit Prometheus can do with `file_edit`
- The task requires Prometheus's own subsystems (SYMBIOTE harvest, LCM memory, SENTINEL)
- It's a quick lookup or read — `grep`/`file_read`/`web_fetch` are faster
- The work needs results back into Prometheus's context for further reasoning (Claude Code returns plain stdout — heavy synthesis stays in the parent agent)

## Setup

Already installed:
```bash
command -v claude    # → /usr/bin/claude
claude --version
```

Auth uses Anthropic API key from the standard locations Claude Code expects (`ANTHROPIC_API_KEY` env or `~/.claude/` config). Prometheus shares the same Anthropic key.

## Invocation Patterns

### One-shot, non-interactive (preferred for delegation)

```bash
cd /path/to/project
claude -p "Refactor src/foo.py to extract the validation logic into a separate ValidationService class. Keep public API unchanged. Run pytest after."
```

The `-p` (or `--print`) flag runs Claude Code headlessly and prints the final response to stdout — exactly what you want when delegating from Prometheus.

### With permission bypass for autonomous work

```bash
claude -p --dangerously-skip-permissions "<task>"
```

Only use when the user has authorized autonomous execution. The flag name is intentionally loud.

### Resume a previous session

```bash
claude -p --resume <session-id> "<follow-up message>"
```

Useful when iterating on the same task — Claude Code keeps its own context.

### From a specific directory

Claude Code uses CWD as its workspace root. Always `cd` first or pass `--cwd`:

```bash
(cd ~/projects/foo && claude -p "<task>")
```

## Delegation Prompt Structure

Treat the Claude Code invocation like a delegated agent call. Use the same shape as `agent-delegation.md`:

```
Goal: <one-sentence objective>
Scope: <files/dirs in scope; explicitly what's out>
Output format: <what should come back — diff, summary, PR URL>
Completion criteria: <how to know it's done — tests pass, lint clean, PR opened>
Constraints: <do not touch X, do not push without confirming, etc.>
```

Example:

```bash
claude -p "Goal: Add rate-limit middleware to the FastAPI app.
Scope: src/api/middleware/ (new file), src/api/main.py (register middleware). Do NOT touch src/db/ or tests for unrelated modules.
Output: Report files changed and pytest summary.
Completion: pytest passes, ruff check passes, no new TODOs."
```

## Combining with sessions_spawn

For parallel work across multiple repos or independent subtasks, use Prometheus's `sessions_spawn` to fan out:

```python
# Pattern (pseudocode of what the agent loop does)
sessions_spawn(task="Run claude -p '<task A>' in /repo/a")
sessions_spawn(task="Run claude -p '<task B>' in /repo/b")
# Then sessions_list / sessions_send to monitor
```

Each spawned session can drive its own Claude Code invocation. Prometheus stays the orchestrator.

## Parsing Claude Code's Output

`claude -p` prints:
1. Tool calls (truncated by default)
2. Final assistant message

For programmatic consumption, use `--output-format json`:

```bash
claude -p --output-format json "<task>" | jq -r '.result'
```

This gives a single JSON object with `result`, `cost_usd`, `duration_ms`, `session_id`. Save the `session_id` if you want to resume.

## Anti-Patterns

- ❌ `claude` (interactive mode) for delegation — leaves an attached TUI. Always use `-p`.
- ❌ Running Claude Code without `cd`-ing first — it inherits Prometheus's CWD, often wrong
- ❌ Delegating tiny tasks (`claude -p "add a print statement"`) — context startup cost dwarfs the work
- ❌ Forwarding the user's raw chat message — strip the conversational framing, hand Claude Code an *engineering brief*
- ❌ Pushing or merging on Claude Code's behalf without showing the user the diff first

## Cost Awareness

Each `claude -p` call is a fresh agent session with its own Anthropic API spend. `--output-format json` reports `cost_usd` — log it. For experiments, prefer `--model claude-haiku-4-5` (cheaper) over Sonnet/Opus when the task doesn't need deep reasoning.

## See Also

- `agent-delegation.md` — generic delegation patterns
- `subagent-driven-development.md` — multi-stage delegation with review
- `aider.md`, `codex.md`, `opencode.md` — alternative coding-agent CLIs
