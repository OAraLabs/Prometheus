---
name: github-auth
description: Set up GitHub authentication for the agent using git. Covers HTTPS tokens, SSH keys, credential helpers -- with a detection flow to pick the right method automatically.
version: 1.1.0
license: MIT
---

# GitHub Authentication Setup

This skill sets up authentication so Prometheus can work with GitHub repositories, PRs, issues, and CI. It covers two paths:

- **`git` (always available)** -- uses HTTPS personal access tokens or SSH keys
- **`gh` CLI (if installed)** -- richer GitHub API access with simpler auth flow

## Detection Flow

When a user asks to work with GitHub, run this check first via `bash`:

```bash
# Check what's available
git --version
gh --version 2>/dev/null || echo "gh not installed"

# Check if already authenticated
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

**Decision tree:**
1. If `gh auth status` shows authenticated -- use `gh` for everything
2. If `gh` is installed but not authenticated -- use "gh auth" method below
3. If `gh` is not installed -- use "git-only" method below (no sudo needed)

---

## Method 1: Git-Only Authentication (No gh, No sudo)

Works on any machine with `git` installed.

### Option A: HTTPS with Personal Access Token (Recommended)

**Step 1: Create a personal access token**

Tell the user to go to: **https://github.com/settings/tokens**
- Generate new token (classic)
- Scopes: `repo`, `workflow`, `read:org` (if needed)
- Set expiration (90 days default)

**Step 2: Configure git to store the token**

```bash
git config --global credential.helper store

# Test with a remote operation
git ls-remote https://github.com/<username>/<repo>.git
```

**Alternative: cache helper (credentials expire from memory)**
```bash
git config --global credential.helper 'cache --timeout=28800'
```

**Step 3: Configure git identity**
```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

### Option B: SSH Key Authentication

**Step 1: Check for existing keys**
```bash
ls -la ~/.ssh/id_*.pub 2>/dev/null || echo "No SSH keys found"
```

**Step 2: Generate if needed**
```bash
ssh-keygen -t ed25519 -C "their-email@example.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Tell user to add at **https://github.com/settings/keys**

**Step 3: Test and configure**
```bash
ssh -T git@github.com
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

---

## Method 2: gh CLI Authentication

### Interactive Browser Login (Desktop)
```bash
gh auth login
```

### Token-Based Login (Headless / SSH Servers)
```bash
echo "<TOKEN>" | gh auth login --with-token
gh auth setup-git
```

---

## Using the GitHub API Without gh

Access the full GitHub API using curl with a personal access token:

```bash
export GITHUB_TOKEN="<token>"

curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### Helper: Detect Auth Method

Use this pattern at the start of any GitHub workflow:

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "AUTH_METHOD=gh"
elif [ -n "$GITHUB_TOKEN" ]; then
  echo "AUTH_METHOD=curl"
elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
  export GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
  echo "AUTH_METHOD=curl"
else
  echo "AUTH_METHOD=none"
  echo "Need to set up authentication first"
fi
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `git push` asks for password | GitHub disabled password auth. Use a personal access token or SSH |
| `remote: Permission to X denied` | Token may lack `repo` scope -- regenerate with correct scopes |
| `fatal: Authentication failed` | Cached credentials stale -- run `git credential reject` then re-auth |
| SSH connection refused | Try SSH over HTTPS port: `Port 443`, `Hostname ssh.github.com` in `~/.ssh/config` |
| Credentials not persisting | Check `git config --global credential.helper` |
| Multiple GitHub accounts | Use SSH with different keys per host alias |

## Prometheus Context

- Run all auth commands via `bash`
- Never commit tokens to the repository (see LCM memory: feedback_no_commit_secrets)
- Store tokens in environment variables or secure config, not in code
- Use `file_read` to check existing git config when troubleshooting
