---
name: coding-agent-delegation
description: Delegate coding tasks to external agents (Codex, Claude Code, Pi, OpenCode) via bash. Use when building new features, reviewing PRs, refactoring codebases, or running iterative coding tasks. Not for simple one-liner fixes or reading code.
version: 1.0.0
tags: [Coding, Delegation, Agents, Automation, Multi-Agent]
---

# Coding Agent Delegation

Delegate coding tasks to external coding agents via `bash`. Use when you need to farm out substantial coding work to a specialized agent.

## When to Use

- Building/creating new features or apps
- Reviewing PRs (clone to temp dir first)
- Refactoring large codebases
- Iterative coding that needs file exploration

## When NOT to Use

- Simple one-liner fixes (just use `file_edit`)
- Reading code (use `file_read` or `grep`)
- Work in the Prometheus workspace directory (never spawn agents there)

## Agent Execution Modes

### Claude Code (no PTY needed)

```bash
# Foreground
cd /path/to/project && claude --permission-mode bypassPermissions --print 'Your task'

# Background
cd /path/to/project && claude --permission-mode bypassPermissions --print 'Your task' &
```

### Codex

```bash
# Quick one-shot (needs a git repo)
SCRATCH=$(mktemp -d) && cd $SCRATCH && git init && codex exec "Your prompt here"

# In a real project
cd ~/Projects/myproject && codex exec --full-auto 'Add error handling to the API calls'
```

**Note:** Codex refuses to run outside a trusted git directory. Use `mktemp -d && git init` for scratch work.

### Codex Flags

| Flag | Effect |
|------|--------|
| `exec "prompt"` | One-shot execution, exits when done |
| `--full-auto` | Sandboxed but auto-approves in workspace |
| `--yolo` | No sandbox, no approvals (fastest, most dangerous) |

### Pi Coding Agent

```bash
# Standard usage
cd ~/project && pi 'Your task'

# Non-interactive
pi -p 'Summarize src/'

# Different provider/model
pi --provider openai --model gpt-4o-mini -p 'Your task'
```

### OpenCode

```bash
cd ~/project && opencode run 'Your task'
```

## PR Review Pattern

Never review PRs in your live project folder. Clone to a temp directory:

```bash
REVIEW_DIR=$(mktemp -d)
git clone https://github.com/user/repo.git $REVIEW_DIR
cd $REVIEW_DIR && git fetch origin pull/130/head:pr-130 && git checkout pr-130

# Run review with your preferred agent
cd $REVIEW_DIR && codex exec "Review this PR. git diff origin/main...HEAD"

# Clean up after
rm -rf $REVIEW_DIR
```

Or use git worktrees:

```bash
git worktree add /tmp/pr-130-review pr-130-branch
cd /tmp/pr-130-review && codex exec "Review this PR against main"
git worktree remove /tmp/pr-130-review
```

## Parallel Issue Fixing with Git Worktrees

```bash
# 1. Create worktrees for each issue
git worktree add -b fix/issue-78 /tmp/issue-78 main
git worktree add -b fix/issue-99 /tmp/issue-99 main

# 2. Launch agents in each (background)
cd /tmp/issue-78 && codex --yolo exec 'Fix issue #78: <description>. Commit and push.' &
cd /tmp/issue-99 && codex --yolo exec 'Fix issue #99: <description>. Commit and push.' &

# 3. Wait and check results
wait
cd /tmp/issue-78 && git log --oneline -5
cd /tmp/issue-99 && git log --oneline -5

# 4. Create PRs
cd /tmp/issue-78 && git push -u origin fix/issue-78
cd /tmp/issue-99 && git push -u origin fix/issue-99

# 5. Cleanup
git worktree remove /tmp/issue-78
git worktree remove /tmp/issue-99
```

## Rules

1. **Use the right execution mode per agent**: Claude Code uses `--print --permission-mode bypassPermissions`; Codex/Pi/OpenCode may need interactive terminal.
2. **Respect tool choice** -- if the user asks for Codex, use Codex. Do not silently take over if an agent fails; report and ask.
3. **Be patient** -- do not kill sessions because they seem slow.
4. **Never spawn agents inside the Prometheus state directory** (`~/.prometheus/`).
5. **Never checkout branches in a live project directory** -- use temp dirs or worktrees.

## Progress Updates

When spawning coding agents in the background:

- Send 1 short message when starting (what is running + where)
- Update when something changes: milestone completes, agent asks a question, error occurs, or agent finishes
- If you kill a session, immediately say so and why
- Include what changed and where when reporting completion
