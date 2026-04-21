---
name: commit
description: Stage, write, and push a well-formed git commit with conventional message and pre-commit checks.
---

# commit

Create a git commit following conventional commit format.

## Steps

1. Run `git status` to see what has changed.
2. Run `git diff --staged` and `git diff` to review all changes.
3. Run `git log --oneline -5` to match the repo's commit style.
4. Stage relevant files with `git add <files>` — avoid `git add .` or `-A` unless every change is intentional.
5. Draft a commit message:
   - Format: `<type>(<scope>): <summary>` where type is feat/fix/refactor/test/docs/chore
   - Keep the summary under 72 characters
   - Add a body if the change is non-obvious
6. Run `git commit -m "..."` — never use `--no-verify`.
7. If a pre-commit hook fails, fix the issue, re-stage, and create a **new** commit — never amend.
8. Push only if the user explicitly asked for it.

## Rules

- Never force-push without explicit user approval.
- Never commit `.env`, credentials, or secrets.
- Never skip hooks (`--no-verify`).
- Prefer specific file staging over blanket `git add .`.
