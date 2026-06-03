---
name: delegate-to-agent
description: Runtime decision tree for picking among Claude Code, Aider, Codex, and OpenCode when delegating a coding task. Routes by task shape (surgical edit vs broad work, vendor lock-in tolerance, cost ceiling, local-model preference). Use BEFORE invoking any specific delegate-to-X skill.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/delegate-to-agent.md | MIT -->

# Delegate to Agent — Routing Meta-Skill

Prometheus has skill files for four coding-agent CLIs:
- `claude-code.md` (Anthropic, installed)
- `aider.md` (Apache 2.0, vendor-agnostic)
- `codex.md` (OpenAI, Apache 2.0)
- `opencode.md` (MIT, vendor-agnostic, Ollama-friendly)

**This skill decides which one to use** before the agent loads the specific delegation skill. Treat it as a router: read the task, pick a CLI, then load that CLI's skill for the invocation details.

## When to Use

✅ Use this skill at the moment you decide a task should be delegated to a coding CLI rather than handled in-Prometheus. Run the routing decision FIRST, then load the target skill.

❌ Skip when:
- The user named a CLI explicitly ("use Claude Code", "have aider do it") — just go.
- The task is small enough to do with `file_edit` directly.
- The work needs Prometheus-specific subsystems (SYMBIOTE, SENTINEL, LCM).

## The Decision Matrix

Read the task. Score it against each row. The row that scores highest picks the CLI.

| Signal | Score for → | Claude Code | Aider | Codex | OpenCode |
|---|---|---|---|---|---|
| "Refactor file X" / "extract Y into Z" | surgical edit | 0 | **+3** | 0 | 0 |
| "Run tests, fix failures, iterate" | tool-heavy loop | **+3** | +1 | +1 | 0 |
| "Browse docs and apply patch" | needs WebFetch | **+3** | 0 | 0 | 0 |
| "Open a PR with N commits" | git-clean history | 0 | **+3** | 0 | 0 |
| "Use a local/free model" | no API spend | 0 | +1 | 0 | **+3** |
| "Sandbox the exec, untrusted code" | restricted FS | 0 | 0 | **+3** | 0 |
| "Second opinion from a different vendor" | cross-vendor | +1 | +1 | **+2** | +1 |
| User explicitly named the CLI | override | matches | matches | matches | matches |
| Default (no strong signal) | broad surface | **+1** | 0 | 0 | 0 |

Tally the scores, pick the leader. Ties → Claude Code (broadest tool surface, lowest surprise).

## Cost Awareness

Token spend per turn varies wildly:

| CLI | Default model | Approx cost per non-trivial task |
|---|---|---|
| Claude Code | claude-sonnet-4-5 | $$ |
| Aider | claude-sonnet-4-5 (same key) | $ — diff-based, token-efficient |
| Aider --architect (sonnet + haiku) | mixed | $ — best $/quality |
| Codex | gpt-5 | $$ |
| OpenCode + Ollama (qwen2.5-coder:32b) | local | $0 — slower but free |

**For loops or batched delegation**, prefer Aider with `--architect` or OpenCode-via-Ollama to keep costs bounded.

## Installation State on This Box

| CLI | Status | Activate with |
|---|---|---|
| `claude` | ✅ installed (`/usr/bin/claude`) | ready |
| `aider` | ✅ installed (uv tool, `~/.local/bin/aider`) | ready |
| `codex` | ❌ not installed | `npm i -g @openai/codex` |
| `opencode` | ❌ not installed | `curl -fsSL https://opencode.ai/install \| bash` |

If routing picks a CLI that isn't installed, **prefer the installed alternative** rather than asking the user to install. Document the alternative path in the response.

## Routing Examples

### Example 1 — "Refactor extract_user_id() out of src/auth.py into src/utils/identifiers.py"
- Signals: surgical edit (+3 Aider), git-clean (+3 Aider)
- Winner: **Aider** (6 points)
- Loaded skill: `aider.md`
- Invocation:
  ```bash
  cd ~/projects/foo
  aider --message "Extract extract_user_id() from src/auth.py to src/utils/identifiers.py. Update all call sites." --yes src/auth.py src/utils/identifiers.py
  ```

### Example 2 — "Look at the React 19 release notes, then port our app's hooks to the new patterns"
- Signals: needs WebFetch (+3 Claude Code), tool-heavy loop (+3 Claude Code)
- Winner: **Claude Code** (6 points)
- Loaded skill: `claude-code.md`
- Invocation:
  ```bash
  cd ~/projects/foo
  claude -p "Read React 19 release notes, then update src/hooks/*.ts to use the new patterns. Run pnpm test after."
  ```

### Example 3 — "Add type hints to all the files in src/utils/, keep it cheap"
- Signals: surgical edit (+3 Aider), local-model preference (+3 OpenCode if available)
- Aider wins on installed-state (OpenCode not installed): **Aider --architect**
- Invocation:
  ```bash
  cd ~/projects/foo
  aider --architect --model sonnet --editor-model haiku --message "Add type hints to all files under src/utils/" --yes src/utils/*.py
  ```

### Example 4 — "Have a non-Anthropic agent try this — Claude already failed twice"
- Signals: cross-vendor (+2 Codex)
- Winner: **Codex**, but it's not installed
- Fallback: ask user to install Codex OR try Aider with `--model gpt-5` (Aider is vendor-agnostic)
- Document the fallback in the response.

## Fallback Chain

When the chosen CLI is unavailable or fails:

```
Claude Code  →  Aider --model sonnet      (Aider can use the same Anthropic key)
Aider        →  Claude Code                (broader surface)
Codex        →  Aider --model gpt-5        (Aider speaks OpenAI too)
OpenCode     →  Aider --model ollama/...   (Aider speaks Ollama)
```

The fallback is **Aider for two of four chains** — it's the most polyglot client. When in doubt and need a fallback, reach for Aider.

## Multi-Agent Patterns

### Parallel for diverse opinions
For high-stakes refactors, fan out the same task to multiple CLIs in parallel via `sessions_spawn`:

```python
sessions_spawn(task="claude -p '<task>'")
sessions_spawn(task="aider --message '<task>' --yes <files>")
# Compare diffs, merge the best parts manually
```

Cost: 2× spend. Worth it for high-impact changes; skip for routine.

### Pipeline for plan-then-execute
For complex multi-step work, separate planning from execution:

```python
# Step 1: planner (high-context, expensive)
plan = claude -p "Plan: how to migrate <X>. Write to /tmp/plan.md, do not implement."

# Step 2: executor (cheap, focused)
aider --message "Implement /tmp/plan.md verbatim" --yes <files>
```

Best $/quality split — planning is where reasoning matters; execution is mostly mechanical.

## Anti-Patterns

- ❌ Routing on vibes — "feels like a Claude Code task" without a signal mapping
- ❌ Picking an uninstalled CLI without checking the install state row
- ❌ Always defaulting to Claude Code — wastes Anthropic spend on tasks Aider does cheaper
- ❌ Routing the same task to all four CLIs "to see who's best" — burns 4× tokens; use the parallel pattern only when warranted
- ❌ Ignoring the user's explicit CLI choice — if they said "use Codex", route to Codex

## See Also

- `claude-code.md`, `aider.md`, `codex.md`, `opencode.md` — per-CLI invocation details
- `agent-delegation.md` — generic delegation patterns
- `subagent-driven-development.md` — multi-stage delegation with review
- `sessions_spawn` tool — Prometheus's parallel-session primitive
