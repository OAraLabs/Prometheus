---
name: gh-issues
description: "Fetch GitHub issues, implement fixes and open PRs, then monitor and address PR review comments. Usage: /gh-issues [owner/repo] [--label bug] [--limit 5] [--milestone v1.0] [--assignee @me] [--fork user/repo] [--watch] [--interval 5] [--reviews-only] [--dry-run] [--notify-channel <telegram_chat_id>]"
user-invocable: true
---

# gh-issues -- Auto-fix GitHub Issues

You are an orchestrator. Follow these 6 phases exactly. Do not skip phases.

Use `bash` with curl + the GitHub REST API for all GitHub operations. Pass GH_TOKEN as a Bearer token:

```
curl -s -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" ...
```

---

## Phase 1 -- Parse Arguments

Parse the arguments string provided after /gh-issues.

Positional:

- owner/repo -- optional. If omitted, detect from current git remote:
  `git remote get-url origin`
  Extract owner/repo from the URL (handles HTTPS and SSH).
  If not in a git repo or no remote found, stop with an error.

Flags (all optional):
| Flag | Default | Description |
|------|---------|-------------|
| --label | _(none)_ | Filter by label |
| --limit | 10 | Max issues to fetch |
| --milestone | _(none)_ | Filter by milestone title |
| --assignee | _(none)_ | Filter by assignee (`@me` for self) |
| --state | open | Issue state: open, closed, all |
| --fork | _(none)_ | Fork to push branches and open PRs from |
| --watch | false | Keep polling for new issues and PR reviews |
| --interval | 5 | Minutes between polls (only with `--watch`) |
| --dry-run | false | Fetch and display only -- no fixes |
| --yes | false | Skip confirmation and auto-process all |
| --reviews-only | false | Skip issue processing, only check PR reviews |
| --notify-channel | _(none)_ | Telegram chat ID for notifications |

Derived values:

- SOURCE_REPO = the positional owner/repo (where issues live)
- PUSH_REPO = --fork value if provided, otherwise same as SOURCE_REPO
- FORK_MODE = true if --fork was provided

---

## Phase 2 -- Fetch Issues

**Token Resolution:**
Check environment for GH_TOKEN:

```bash
echo $GH_TOKEN
```

If empty, check common config locations or prompt user.

Fetch issues via `bash`:

```bash
curl -s -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/{SOURCE_REPO}/issues?per_page={limit}&state={state}&{query_params}"
```

Where {query_params} is built from --label, --milestone, --assignee flags.

**Important:** The GitHub Issues API returns pull requests too. Filter them out -- exclude any item where `pull_request` key exists.

Error handling:
- HTTP 401/403: Report authentication failure
- Empty array: Report "No issues found matching filters"

---

## Phase 3 -- Present and Confirm

Display a markdown table of fetched issues:

| # | Title | Labels |
|---|-------|--------|
| 42 | Fix null pointer in parser | bug, critical |

If `--dry-run`: Display table and stop.
If `--yes`: Auto-process all listed issues.
Otherwise: Ask user to confirm which issues to process.

---

## Phase 4 -- Pre-flight Checks

Run via `bash`:

1. **Dirty working tree check:** `git status --porcelain`
2. **Record base branch:** `git rev-parse --abbrev-ref HEAD`
3. **Verify remote access:** `git ls-remote --exit-code origin HEAD`
4. **Verify GH_TOKEN validity:** Curl test to /user endpoint
5. **Check for existing PRs:** Query pulls API for each issue

---

## Phase 5 -- Implement Fixes

For each confirmed issue, implement the fix directly (Prometheus runs in a single agent context):

### Fix Workflow Per Issue

```
1. UNDERSTAND -- Read the issue carefully. Identify what needs to change.

2. BRANCH -- Create a feature branch:
   git checkout -b fix/issue-{N} {BASE_BRANCH}

3. ANALYZE -- Search the codebase with grep/file_read to find relevant files.

4. IMPLEMENT -- Make the minimal, focused fix using file_edit.
   - Follow existing code style
   - Change only what is necessary

5. TEST -- Run existing test suite if one exists.

6. COMMIT -- Stage and commit:
   git add {changed_files}
   git commit -m "fix: {short_description}

   Fixes {SOURCE_REPO}#{N}"

7. PUSH -- Push the branch:
   git push -u origin fix/issue-{N}

8. PR -- Create a pull request via curl:
   curl -s -X POST \
     -H "Authorization: Bearer $GH_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/{SOURCE_REPO}/pulls \
     -d '{
       "title": "fix: {title}",
       "head": "fix/issue-{N}",
       "base": "{BASE_BRANCH}",
       "body": "## Summary\n\n{description}\n\nFixes #{N}"
     }'
```

### Constraints
- No force-push, no modifying the base branch
- No unrelated changes or gratuitous refactoring
- No new dependencies without strong justification
- If the issue is unclear or too complex, report analysis instead of guessing

---

## Results Collection

Present a summary table:

| Issue | Status | PR | Notes |
|-------|--------|----|-------|
| #42 Fix null pointer | PR opened | https://github.com/.../pull/99 | 3 files changed |
| #37 Add retry logic | Failed | -- | Could not identify target code |

**Notify via Telegram (if --notify-channel is set):**
Use the message tool to send the summary to the specified Telegram chat.

---

## Phase 6 -- PR Review Handler

Monitors open PRs for review comments and addresses them.

### Step 6.1 -- Discover PRs to Monitor

Fetch open PRs with `fix/issue-` branch pattern:

```bash
curl -s -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/{SOURCE_REPO}/pulls?state=open&per_page=100"
```

Filter to PRs where `head.ref` starts with `fix/issue-`.

### Step 6.2 -- Fetch Review Comments

For each PR, fetch from multiple sources:
- PR reviews: `/pulls/{pr_number}/reviews`
- Inline comments: `/pulls/{pr_number}/comments`
- General comments: `/issues/{pr_number}/comments`

### Step 6.3 -- Analyze for Actionability

**NOT actionable:** Pure approvals, LGTM, informational bot comments.
**IS actionable:** CHANGES_REQUESTED reviews, specific code change requests, inline comments pointing out issues.

### Step 6.4 -- Address Review Comments

For each PR with actionable comments:

1. CHECKOUT the PR branch
2. UNDERSTAND all review comments
3. IMPLEMENT requested changes using `file_edit`
4. TEST to ensure no breakage
5. COMMIT and PUSH
6. REPLY to each comment via the API

---

## Watch Mode (if --watch is active)

1. Track processed issues and addressed comments across cycles
2. Sleep for {interval} minutes between polls
3. Re-run Phase 2 through Phase 6 each cycle
4. User can stop at any time

## Prometheus Context

- Use `bash` for all git and curl operations
- Use `file_read` and `file_edit` for code changes
- Use `grep` to search the codebase for relevant code
- Notify via Telegram using the message tool
- Log progress to LCM for tracking across sessions
