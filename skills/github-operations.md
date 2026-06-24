---
name: github-operations
description: GitHub operations via gh CLI or git+curl -- issues, PRs, CI runs, code review, API queries. Quick-reference for common GitHub interactions when gh is available.
version: 1.0.0
tags: [GitHub, Issues, PRs, CI, Code-Review]
---

# GitHub Operations

Quick reference for GitHub operations. Uses `gh` CLI when available, falls back to git + curl with `$GITHUB_TOKEN`.

## When to Use

- Checking PR status, reviews, or merge readiness
- Viewing CI/workflow run status and logs
- Creating, closing, or commenting on issues
- Creating or merging pull requests
- Querying GitHub API for repository data

## When NOT to Use

- Local git operations (commit, push, pull, branch) -- use `git` directly
- Non-GitHub repos (GitLab, Bitbucket) -- different CLIs
- Cloning repositories -- use `git clone`
- Complex multi-file code review -- use `file_read` / `grep` directly

## Setup

```bash
# Check if gh is available
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "gh CLI ready"
else
  echo "Using git + curl (needs GITHUB_TOKEN)"
fi
```

## Pull Requests

```bash
# List PRs
gh pr list --repo owner/repo

# Check CI status
gh pr checks 55 --repo owner/repo

# View PR details
gh pr view 55 --repo owner/repo

# Create PR
gh pr create --title "feat: add feature" --body "Description"

# Merge PR
gh pr merge 55 --squash --repo owner/repo
```

### Without gh (curl fallback)

```bash
# List open PRs
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls?state=open

# Create PR
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls \
  -d '{"title":"feat: add feature","head":"my-branch","base":"main","body":"Description"}'

# Merge PR
curl -s -X PUT -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/55/merge \
  -d '{"merge_method":"squash"}'
```

## Issues

```bash
# List issues
gh issue list --repo owner/repo --state open

# Create issue
gh issue create --title "Bug: something broken" --body "Details..."

# Close issue
gh issue close 42 --repo owner/repo
```

### Without gh

```bash
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/issues \
  -d '{"title":"Bug: something broken","body":"Details..."}'
```

## CI/Workflow Runs

```bash
# List recent runs
gh run list --repo owner/repo --limit 10

# View specific run
gh run view <run-id> --repo owner/repo

# View failed step logs only
gh run view <run-id> --repo owner/repo --log-failed

# Re-run failed jobs
gh run rerun <run-id> --failed --repo owner/repo
```

### Without gh

```bash
# List runs
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/actions/runs?per_page=10" \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin)['workflow_runs']:
    print(f\"Run {r['id']}  {r['name']:30}  {r['conclusion'] or r['status']}\")"

# Re-run
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/runs/<run-id>/rerun
```

## API Queries

```bash
# Get PR with specific fields
gh api repos/owner/repo/pulls/55 --jq '.title, .state, .user.login'

# List all labels
gh api repos/owner/repo/labels --jq '.[].name'

# Get repo stats
gh api repos/owner/repo --jq '{stars: .stargazers_count, forks: .forks_count}'
```

## JSON Output

Most gh commands support `--json` with `--jq` filtering:

```bash
gh issue list --repo owner/repo --json number,title --jq '.[] | "\(.number): \(.title)"'
gh pr list --json number,title,state,mergeable --jq '.[] | select(.mergeable == "MERGEABLE")'
```

## Templates

### PR Review Summary

```bash
PR=55 REPO=owner/repo
echo "## PR #$PR Summary"
gh pr view $PR --repo $REPO --json title,body,author,additions,deletions,changedFiles \
  --jq '"**\(.title)** by @\(.author.login)\n\n+\(.additions) -\(.deletions) across \(.changedFiles) files"'
gh pr checks $PR --repo $REPO
```

### Issue Triage

```bash
gh issue list --repo owner/repo --state open --json number,title,labels,createdAt \
  --jq '.[] | "[\(.number)] \(.title) - \([.labels[].name] | join(", ")) (\(.createdAt[:10]))"'
```

## Notes

- Always specify `--repo owner/repo` when not in a git directory
- Use URLs directly: `gh pr view https://github.com/owner/repo/pull/55`
- Rate limits apply; use `gh api --cache 1h` for repeated queries
- See `github-pr-workflow` skill for the full PR lifecycle
- See `github-repo-management` skill for repo creation, forking, releases
