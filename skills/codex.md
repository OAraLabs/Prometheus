---
name: codex
description: Delegate coding tasks to OpenAI's Codex CLI. Covers install, one-shot non-interactive invocation via `codex exec`, working-directory and sandbox flags, and when to pick Codex over Claude Code or Aider.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/codex.md | MIT -->

# Delegating to OpenAI Codex CLI

Codex CLI (Apache 2.0, openai/codex) is OpenAI's open-source coding agent. Runs locally, drives `gpt-5`-class models. Ships with its own sandboxed exec model. Good choice when you want a non-Anthropic alternative or when the user has OpenAI credits.

## When to Use

✅ **DELEGATE to Codex when:**

- User explicitly asks for "Codex" or "OpenAI" or "GPT" to do the work
- You want a cross-vendor second opinion on a tricky change
- The work suits GPT's strengths (heavy reasoning, formal/proof-style tasks)
- You're load-balancing across multiple coding agents for parallel work

## When NOT to Use

❌ **DON'T use when:**

- No `OPENAI_API_KEY` available
- The task is well-suited to Aider's git workflow → use Aider
- The task needs Anthropic-specific tools (prompt caching, vision-heavy) → use Claude Code
- Trivial single-line edit → use Prometheus's `file_edit` directly

## Install

Not currently on this host.

```bash
npm i -g @openai/codex
# verify
codex --version
```

Requires Node.js 18+. If `npm` isn't installed, prefer `corepack` or `nvm` — global npm installs without a node manager get messy.

## Auth

```bash
export OPENAI_API_KEY=sk-...
```

Or sign in interactively once (stores token under `~/.codex/`):

```bash
codex login
```

## Invocation Patterns

### One-shot non-interactive (preferred for delegation)

```bash
cd /path/to/repo
codex exec "Refactor the validation in src/forms.py to use Pydantic v2 model_validator. Update tests in tests/test_forms.py."
```

`codex exec` is the headless mode — runs the task, prints final output, exits. The interactive `codex` (no subcommand) is a TUI; don't use it for delegation.

### Sandboxing

Codex defaults to a sandboxed exec environment. To allow network/full FS access:

```bash
codex exec --sandbox danger-full-access "<task>"      # full host access
codex exec --sandbox workspace-write "<task>"          # write only inside CWD (default)
codex exec --sandbox read-only "<task>"                # read-only
```

For Prometheus delegation, `workspace-write` (default) is usually right. Only escalate when the task genuinely needs broader access.

### Specifying a model

```bash
codex exec --model gpt-5 "<task>"
codex exec --model gpt-5-mini "<task>"      # cheaper
```

### Specifying a working directory

```bash
codex exec --cd /path/to/repo "<task>"
```

Equivalent to `cd` + `codex exec`, but more explicit. Prefer this for clarity.

### JSON output for programmatic capture

```bash
codex exec --json "<task>" | jq
```

Returns structured events (similar to Claude Code's `--output-format json`). Easier to parse than raw stdout.

## Delegation Prompt Structure

Same shape as Claude Code / Aider — explicit goal, scope, output, completion:

```bash
codex exec "Goal: Migrate the user-auth tests from unittest to pytest.
Scope: tests/auth/ only. Don't touch src/ or other test directories.
Output: Report test count before/after and any tests that needed rewriting.
Completion: uv run pytest tests/auth/ -v passes."
```

## Combining with sessions_spawn

```python
sessions_spawn(task="codex exec --cd /repo/a 'task A'")
sessions_spawn(task="codex exec --cd /repo/b 'task B'")
```

## Multi-Agent Comparison Pattern

When the work is high-stakes or you want second opinions, fan the same task to multiple agents in parallel:

```python
sessions_spawn(task="claude -p '<task>'")          # Anthropic
sessions_spawn(task="codex exec '<task>'")          # OpenAI
sessions_spawn(task="aider --message '<task>' --yes <files>")   # surgical
# Then compare diffs, pick the best one
```

Useful for tricky refactors where you want to see multiple approaches before merging.

## Codex vs Claude Code vs Aider

| Trait | Codex | Claude Code | Aider |
|---|---|---|---|
| Provider | OpenAI | Anthropic | Provider-agnostic |
| Default sandbox | Yes (workspace-write) | No | No (git as safety net) |
| TUI default? | Yes (use `exec` subcmd) | No (`-p` is one-shot) | Yes (use `--message`) |
| Auto-commit | No | No | Yes |
| Output format | `--json` | `--output-format json` | Plain text |
| Best at | GPT reasoning, OpenAI-stack | Broad tool surface, Anthropic-stack | Surgical diffs + git |

## Anti-Patterns

- ❌ Interactive `codex` (no `exec`) for delegation — opens TUI, blocks the agent loop
- ❌ Defaulting to `--sandbox danger-full-access` "to avoid errors" — defeats the safety model. Read the error first.
- ❌ Forwarding the user's chat message verbatim — strip conversational fluff, hand it an engineering brief
- ❌ Running Codex without `OPENAI_API_KEY` set — it will exit immediately with an auth error

## See Also

- `claude-code.md` — Anthropic's CLI
- `aider.md` — surgical edits with git
- `opencode.md` — OpenCode TUI
- `agent-delegation.md` — generic delegation patterns
