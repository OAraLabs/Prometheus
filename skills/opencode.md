---
name: opencode
description: Delegate coding tasks to OpenCode (sst/opencode), an open-source multi-provider TUI coding agent. Covers install, headless `run` mode, provider/model selection, and when to pick OpenCode over Claude Code, Aider, or Codex.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/opencode.md | MIT -->

# Delegating to OpenCode

OpenCode (MIT, sst/opencode) is an open-source TUI coding agent built by the SST team. Provider-agnostic — works with Anthropic, OpenAI, local models via Ollama, and others. Good fallback when you want a fully open client and have flexibility on the model provider.

## When to Use

✅ **DELEGATE to OpenCode when:**

- You want an open-source client (no vendor TUI dependency)
- Routing through a local Ollama model — OpenCode handles this cleanly
- User explicitly asks for OpenCode by name
- You want to test the same prompt against a different agent harness

## When NOT to Use

❌ **DON'T use when:**

- The task is well-suited to Claude Code's mature tool stack
- Aider's git-integrated diffing is a better fit
- You haven't installed it yet and the task is one-off — install cost not worth it

## Install

Not currently on this host. Multiple install paths:

```bash
# Recommended (official install script)
curl -fsSL https://opencode.ai/install | bash

# Or via npm
npm i -g opencode-ai

# Or via Homebrew on macOS
brew install sst/tap/opencode
```

After install:

```bash
opencode --version
```

## Auth

OpenCode reads provider keys from env vars or `~/.config/opencode/auth.json`:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
# Local Ollama needs no key:
export OPENCODE_PROVIDER=ollama
```

Or run `opencode auth login` for interactive setup.

## Invocation Patterns

### Headless one-shot (preferred for delegation)

```bash
cd /path/to/repo
opencode run "Refactor the config loader to use pydantic-settings instead of manual env parsing."
```

`opencode run` is the headless mode — equivalent to Claude Code's `-p` or Codex's `exec`.

### Selecting a model

```bash
opencode run -m anthropic/claude-sonnet-4-5 "<task>"
opencode run -m openai/gpt-5 "<task>"
opencode run -m ollama/qwen2.5-coder:32b "<task>"      # local, free
```

### Continuing a session

```bash
opencode run --continue "<follow-up>"      # resume last session
opencode run --session <id> "<follow-up>"  # resume specific session
```

### Specifying working directory

```bash
opencode run --cwd /path/to/repo "<task>"
```

## Local Model Routing (Ollama)

OpenCode's killer feature for Prometheus is clean local-model support — Prometheus already has Ollama installed (`~/.local/bin/ollama`):

```bash
# List local Ollama models
ollama list

# Delegate to a local coding model (no API cost)
opencode run -m ollama/qwen2.5-coder:32b "Add type hints to src/utils/dates.py"
```

This is the open-source / no-subscription path the user has explicitly asked for. Local models won't match Sonnet quality but handle straightforward edits well.

## Delegation Prompt Structure

Same shape as Claude Code / Codex / Aider:

```bash
opencode run "Goal: <objective>
Scope: <files/dirs>
Output: <expected return>
Completion: <success criteria>"
```

## Combining with sessions_spawn

```python
# Mix providers across parallel sessions
sessions_spawn(task="opencode run -m anthropic/sonnet '<task A>'")
sessions_spawn(task="opencode run -m ollama/qwen2.5-coder '<task B>'")   # free
```

The Ollama variant runs locally — zero API cost, slower but private.

## OpenCode vs the Others

| Trait | OpenCode | Claude Code | Codex | Aider |
|---|---|---|---|---|
| Provider agnostic | ✅ Multiple | ❌ Anthropic only | ❌ OpenAI only | ✅ Multiple |
| Local model support | ✅ Strong (Ollama) | ❌ | ❌ | ✅ |
| Open source client | ✅ MIT | ❌ Anthropic | ✅ Apache 2.0 | ✅ Apache 2.0 |
| TUI default | ✅ | ❌ | ✅ | ✅ |
| Headless flag | `run` | `-p` | `exec` | `--message` |
| Git auto-commit | No | No | No | Yes |
| Sandbox | No (relies on host) | No | Yes (workspace-write) | No |

## Anti-Patterns

- ❌ Interactive `opencode` (no `run` subcommand) for delegation — opens TUI, blocks agent loop
- ❌ Using Ollama models for tasks they're too small to handle — quality cliff is real. Stick to ≥7B for code edits.
- ❌ Running multiple parallel `opencode run` with Ollama on a single GPU — they'll queue and starve each other. Serialize Ollama calls.
- ❌ Forgetting `--cwd` when delegating from sessions_spawn — inherits Prometheus's CWD

## Routing Heuristic

Quick decision tree for "which delegation CLI?":

1. Need git-clean surgical edits with auto-commits? → **Aider**
2. Need broad tool surface (web, bash, full repo work) with Anthropic? → **Claude Code**
3. Need OpenAI's GPT specifically, or sandboxed exec? → **Codex**
4. Need a local/free model or vendor-agnostic client? → **OpenCode** (with Ollama)
5. Don't know? → Default to **Claude Code** (broadest, lowest surprise)

## See Also

- `claude-code.md`, `codex.md`, `aider.md` — the other delegation targets
- `agent-delegation.md` — generic delegation patterns
- `docker.md` — for managing the underlying infra these agents may touch
