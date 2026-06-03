---
name: aider
description: Delegate surgical multi-file code edits to Aider, an open-source AI pair programmer with native git integration. Covers install, model selection, one-shot vs interactive invocation, and when to pick Aider over Claude Code.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/aider.md | MIT -->

# Delegating to Aider

Aider (Apache 2.0, paul-gauthier/aider) is the most mature open-source AI pair programmer. It does diff-based edits with native git integration — every change becomes a commit. Excellent for surgical, well-scoped refactors where you want a clean git history.

## When to Use

✅ **DELEGATE to Aider when:**

- The change is well-scoped: specific files, specific edit
- You want every edit committed automatically with a sensible message
- Need diff-based editing (Aider produces minimal diffs, not whole-file rewrites)
- Working in a Python/JS/Go/Rust/etc. repo where Aider's repo-map shines
- Token-efficient editing matters (Aider sends only relevant files)

## When NOT to Use

❌ **DON'T delegate to Aider when:**

- Task requires running tests, browsing the web, or other tools beyond editing (use Claude Code)
- You want exploration/research rather than edits (use Prometheus's `grep`/`web_search`)
- The repo isn't a git repo — Aider's git integration is core to its workflow
- Task is single-line trivial — `file_edit` is faster

## Install

Not currently on this host. Recommended install via `uv` (matches Prometheus's Python tooling preference):

```bash
uv tool install aider-chat
# verify
aider --version
```

Alternatives:
```bash
pip install aider-chat                    # if not using uv
pipx install aider-chat                   # isolated venv
```

## Auth

Aider reads from env vars:
```bash
export ANTHROPIC_API_KEY=...    # for Claude models (default to Sonnet)
export OPENAI_API_KEY=...       # for GPT models
```

Prometheus already has these — Aider inherits them.

## Invocation Patterns

### One-shot non-interactive (preferred for delegation)

```bash
cd /path/to/repo
aider --message "Extract the email validation logic from src/auth.py into a new src/validators/email.py module" \
      --yes \
      src/auth.py src/validators/email.py
```

Key flags:
- `--message` / `-m`: single instruction, exits when done
- `--yes`: auto-accept Aider's proposed edits (autonomous mode)
- Trailing args: files to include in the edit context

### With explicit model

```bash
aider --model sonnet --message "<task>" file1.py file2.py
aider --model haiku --message "<task>" file1.py file2.py      # cheaper
aider --model gpt-4o --message "<task>" file1.py file2.py     # OpenAI
```

### Architect mode (think-then-edit, higher quality)

```bash
aider --architect --model sonnet --editor-model haiku --message "<task>" <files>
```

Sonnet plans, Haiku executes — cheaper for the editing step. Good for medium-complexity refactors.

### No git commits (preview mode)

```bash
aider --no-auto-commits --message "<task>" <files>
```

Useful when Prometheus wants to inspect the diff before committing.

## Delegation Prompt Structure

Aider works best with *specific*, *file-scoped* instructions:

```
Refactor extract_user_id() out of src/handlers/auth.py into src/utils/identifiers.py.
Update all call sites. Keep the existing function signature.
```

Bad: "improve auth code" (too vague).
Good: "Replace the manual JWT decoding in `verify_token()` with PyJWT's `jwt.decode()` call. Keep the same error semantics."

## File Inclusion Rules

Files listed as trailing args are *editable*. Other files Aider discovers via repo-map are *read-only context*. Best practice:

- List the files you expect Aider to modify
- Don't list 50 files — Aider will pull what it needs via repo-map
- For "this whole module", list the module's files explicitly

## Output Parsing

Aider prints:
- Proposed edits (as diffs)
- Commit hash + message
- Any test/lint output if you configured it

For programmatic capture, redirect:

```bash
aider --message "<task>" --yes <files> 2>&1 | tee /tmp/aider-out.log
```

Then read `/tmp/aider-out.log` with `file_read`.

## Combining with Tests

Aider has built-in test integration:

```bash
aider --message "<task>" \
      --test-cmd "uv run pytest tests/test_auth.py -x" \
      --auto-test \
      --yes \
      <files>
```

`--auto-test` re-runs the test command after each edit; Aider iterates until tests pass or it gives up.

## Combining with sessions_spawn

For parallel multi-repo delegation:

```python
sessions_spawn(task="In ~/projects/foo, run aider --message '<task A>' --yes file.py")
sessions_spawn(task="In ~/projects/bar, run aider --message '<task B>' --yes file.py")
```

## Aider vs Claude Code

| Trait | Aider | Claude Code |
|---|---|---|
| Tool surface | Edit + git only | Full (Bash, Web, Edit, Read, Grep, Glob) |
| Best at | Surgical multi-file diffs | Open-ended tasks needing many tool types |
| Git integration | Auto-commit each edit | Manual via Bash |
| Repo-map | Built-in, token-efficient | Built-in |
| Cost | Lower (token-efficient) | Higher (more tool calls) |
| Streaming output | Yes | Yes |
| Web access | No | Yes (WebFetch) |
| Run tests | Yes (`--test-cmd`) | Yes (via Bash) |

**Rule of thumb**: well-scoped edit → Aider. Anything broader → Claude Code.

## Anti-Patterns

- ❌ Interactive `aider` (no `--message`) for delegation — same TUI-blocking problem as `claude`
- ❌ Skipping `--yes` then walking away — Aider hangs at the "Apply edit?" prompt
- ❌ Listing every file in the repo as a trailing arg — wastes tokens, confuses scope
- ❌ Using Aider without git — its workflow assumes commits; you lose auditability without it
- ❌ Mixing `aider --no-auto-commits` with subsequent `git commit -a` calls — Aider may stage things you didn't expect

## See Also

- `claude-code.md` — delegate to Claude Code (broader tool surface)
- `codex.md` — delegate to OpenAI Codex CLI
- `opencode.md` — delegate to OpenCode TUI
- `agent-delegation.md` — generic delegation patterns
