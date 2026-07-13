---
name: github-code-review
description: Review code changes by analyzing git diffs, leaving inline comments on PRs, and performing thorough pre-push review. Works with gh CLI or falls back to git + GitHub REST API via curl.
version: 1.1.0
license: MIT
---

# GitHub Code Review

Perform code reviews on local changes before pushing, or review open PRs on GitHub. Most of this skill uses plain `git` via `bash`.

## Prerequisites

- Authenticated with GitHub (see `github-auth` skill)
- Inside a git repository

### Setup (for PR interactions)

Run via `bash`:
```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 1. Reviewing Local Changes (Pre-Push)

Pure `git` -- works everywhere, no API needed.

### Get the Diff

```bash
# Staged changes
git diff --staged

# All changes vs main (what a PR would contain)
git diff main...HEAD

# File names only
git diff main...HEAD --name-only

# Stat summary
git diff main...HEAD --stat
```

### Review Strategy

1. **Get the big picture first:**
```bash
git diff main...HEAD --stat
git log main..HEAD --oneline
```

2. **Review file by file** -- use `file_read` on changed files for full context, and the diff to see what changed:
```bash
git diff main...HEAD -- src/auth/login.py
```

3. **Check for common issues:**
```bash
# Debug statements, TODOs left behind
git diff main...HEAD | grep -n "print(\|console\.log\|TODO\|FIXME\|HACK\|debugger"

# Secrets or credential patterns
git diff main...HEAD | grep -in "password\|secret\|api_key\|token.*=\|private_key"

# Merge conflict markers
git diff main...HEAD | grep -n "<<<<<<\|>>>>>>\|======="
```

4. **Present structured feedback** to the user.

### Review Output Format

```
## Code Review Summary

### Critical
- **src/auth.py:45** -- SQL injection: user input passed directly to query.

### Warnings
- **src/models/user.py:23** -- Password stored in plaintext. Use bcrypt or argon2.

### Suggestions
- **src/utils/helpers.py:8** -- Duplicates logic in `src/core/utils.py:34`.

### Looks Good
- Clean separation of concerns in the middleware layer
```

---

## 2. Reviewing a Pull Request on GitHub

### View PR Details

**With gh:**
```bash
gh pr view 123
gh pr diff 123
```

**With git + curl:**
```bash
PR_NUMBER=123
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER
```

### Check Out PR Locally

```bash
git fetch origin pull/123/head:pr-123
git checkout pr-123
git diff main...pr-123
```

### Leave Comments on a PR

**General comment -- with gh:**
```bash
gh pr comment 123 --body "Overall looks good, a few suggestions below."
```

**General comment -- with curl:**
```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/issues/$PR_NUMBER/comments \
  -d '{"body": "Overall looks good, a few suggestions below."}'
```

### Submit a Formal Review

**With gh:**
```bash
gh pr review 123 --approve --body "LGTM!"
gh pr review 123 --request-changes --body "See inline comments."
```

**With curl -- multi-comment review:**
```bash
HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"COMMENT\",
    \"body\": \"Code review from Prometheus\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"Use parameterized queries.\"}
    ]
  }"
```

---

## 3. Review Checklist

### Correctness
- Does the code do what it claims?
- Edge cases handled?
- Error paths handled gracefully?

### Security
- No hardcoded secrets, credentials, or API keys
- Input validation on user-facing inputs
- No SQL injection, XSS, or path traversal

### Code Quality
- Clear naming
- No unnecessary complexity
- DRY -- no duplicated logic
- Functions are focused

### Testing
- New code paths tested?
- Happy path and error cases covered?

### Performance
- No N+1 queries or unnecessary loops
- Appropriate caching
- No blocking operations in async code

### Documentation
- Public APIs documented
- Non-obvious logic has comments explaining "why"

---

## 4. Pre-Push Review Workflow

1. `git diff main...HEAD --stat` -- see scope
2. `git diff main...HEAD` -- read full diff
3. For each file, use `file_read` for more context
4. Apply the checklist
5. Present findings (Critical / Warnings / Suggestions / Looks Good)
6. If critical issues found, offer to fix before pushing

---

## 5. PR Review Workflow (End-to-End)

1. **Gather context** -- view PR metadata and changed files
2. **Check out locally** -- `git fetch origin pull/N/head:pr-N`
3. **Read diff and understand changes** -- use `file_read` for full context
4. **Run tests locally** -- `bash` with appropriate test commands
5. **Apply checklist** -- Section 3
6. **Post review** -- approve, request changes, or comment
7. **Post summary comment** -- top-level overview
8. **Clean up** -- `git checkout main && git branch -D pr-N`

### Decision: Approve vs Request Changes vs Comment

- **Approve** -- no critical or warning issues
- **Request Changes** -- any critical or warning issue
- **Comment** -- observations and suggestions, nothing blocking

## Prometheus Context

- Run all git/curl commands via `bash`
- Use `file_read` to examine changed files in full
- Use `grep` to search for related code patterns
- Log review findings in LCM for tracking
- Use SENTINEL to monitor for follow-up changes
