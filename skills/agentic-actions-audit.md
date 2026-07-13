---
name: agentic-actions-audit
description: "Audits CI/CD workflows for security vulnerabilities in AI agent integrations. Use when reviewing workflow files that invoke AI coding agents, auditing pipelines for prompt injection risks, or evaluating agentic action configurations."
version: 1.0.0
author: Trail of Bits (adapted for Prometheus)
license: CC-BY-SA-4.0
---

# Agentic Actions Auditor

Static security analysis guidance for CI/CD workflows that invoke AI coding agents. Covers how to discover workflow files, identify AI action steps, follow cross-file references, capture security-relevant configuration, and detect attack vectors where attacker-controlled input reaches an AI agent running in a CI/CD pipeline.

## When to Use

- Auditing a repository's CI/CD workflows for AI agent security
- Reviewing configurations that invoke AI coding agents (any provider)
- Checking whether attacker-controlled input can reach AI agent prompts
- Evaluating agentic action configurations (sandbox settings, tool permissions, user allowlists)
- Assessing trigger events that expose workflows to external input (`pull_request_target`, `issue_comment`, etc.)
- Investigating data flow from event context through environment blocks to AI prompt fields

## When NOT to Use

- Workflows that do NOT use any AI agent actions (use general CI security tools)
- Standalone composite actions outside of a caller workflow context
- Runtime prompt injection testing (this is static analysis, not exploitation)
- Non-GitHub CI/CD systems without adaptation (Jenkins, GitLab CI, CircleCI have different patterns)
- Auto-fixing workflow files (this skill reports findings, does not modify files)

## Rationalizations to Reject

**1. "It only runs on PRs from maintainers"**
Wrong because it ignores `pull_request_target`, `issue_comment`, and other triggers that expose actions to external input. Attackers do not need write access.

**2. "We use allowed_tools to restrict what it can do"**
Wrong because even restricted tools like `echo` can be abused for data exfiltration via subshell expansion (`echo $(env)`). A tool allowlist reduces attack surface but does not eliminate it.

**3. "There's no ${{ }} in the prompt, so it's safe"**
Wrong because data flows through `env:` blocks to the prompt field with zero visible expressions. The YAML looks clean but the AI agent still receives attacker-controlled input.

**4. "The sandbox prevents any real damage"**
Wrong because sandbox misconfigurations disable protections entirely. Even properly configured sandboxes leak secrets if the AI agent can read environment variables.

## Attacker-Controlled Context Expressions

These `github.event.*` expressions resolve to content an external attacker can influence:

**High-frequency:**
- `github.event.issue.body` -- issue body text
- `github.event.issue.title` -- issue title
- `github.event.comment.body` -- comment text on issues or PRs
- `github.event.pull_request.body` -- PR description
- `github.event.pull_request.title` -- PR title
- `github.event.pull_request.head.ref` -- PR source branch name
- `github.event.pull_request.head.sha` -- PR commit SHA

**Lower-frequency but still dangerous:**
- `github.event.review.body` -- review comment text
- `github.event.discussion.body`, `github.event.discussion.title`
- `github.event.commits.*.message`, `github.event.commits.*.author.email`
- `github.head_ref` -- branch name (attacker-controlled in fork PRs)

## How env: Blocks Enable Invisible Injection

Environment variables set at three scopes (workflow, job, step). `${{ }}` expressions in `env:` values are evaluated BEFORE the step runs. The step only sees the resolved string:

```yaml
env:
  ISSUE_BODY: ${{ github.event.issue.body }}   # Evaluated at parse time
# The AI agent receives raw attacker text through ISSUE_BODY
```

## Security-Relevant Trigger Events

| Trigger | Attacker-Controlled Data | Risk Level |
|---------|-------------------------|------------|
| `issues` (opened, edited) | Issue title, body | External users can create issues |
| `issue_comment` (created) | Comment body | External users can comment |
| `pull_request_target` | PR title, body, head ref, head SHA | Runs in base branch context WITH secrets |
| `pull_request` | Head ref, head SHA | Typically no secrets from forks |
| `discussion` / `discussion_comment` | Discussion title, body, comment body | External users can create discussions |
| `workflow_dispatch` | Input values | Triggering user controls all inputs |

## Audit Methodology

### Step 1: Discover Workflow Files

**Local analysis:** Use `glob` to find `.github/workflows/*.yml` and `.github/workflows/*.yaml`

**Remote analysis:** Use `bash` with git commands or curl to list workflow files:
```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/{owner}/{repo}/contents/.github/workflows" | \
  python3 -c "import sys,json; [print(f['name']) for f in json.load(sys.stdin)]"
```

### Step 2: Identify AI Action Steps

Check each step's `uses:` field against known AI action references:

| Action Reference | Action Type |
|-----------------|-------------|
| `anthropics/claude-code-action` | Claude Code Action |
| `google-github-actions/run-gemini-cli` | Gemini CLI |
| `google-gemini/gemini-cli-action` | Gemini CLI (legacy) |
| `openai/codex-action` | OpenAI Codex |
| `actions/ai-inference` | GitHub AI Inference |

Match as prefix before `@`. Also resolve step-level `uses:` with local paths and job-level `uses:` (reusable workflows) one level deep.

### Step 3: Capture Security Context

For each AI action step, capture:

**Step-level (from `with:` block):**
- Prompt fields, CLI args, allowed users/bots, settings, trigger phrases
- Sandbox configurations, safety strategies

**Workflow-level:**
- Trigger events (flag `pull_request_target`, `issue_comment`, `issues`)
- Environment variables with `${{ }}` expressions referencing event data
- Permissions (flag overly broad like `contents: write` combined with AI execution)

### Step 4: Analyze for Attack Vectors

| Vector | Name | Quick Check |
|--------|------|-------------|
| A | Env Var Intermediary | `env:` block with `${{ github.event.* }}` value + prompt reads that env var |
| B | Direct Expression Injection | `${{ github.event.* }}` inside prompt or system-prompt field |
| C | CLI Data Fetch | CLI commands fetching issue/PR data in prompt text |
| D | PR Target + Checkout | `pull_request_target` trigger + checkout with `ref:` pointing to PR head |
| E | Error Log Injection | CI logs, build output, or `workflow_dispatch` inputs passed to AI prompt |
| F | Subshell Expansion | Tool restriction list includes commands supporting `$()` expansion |
| G | Eval of AI Output | `eval`, `exec`, or `$()` in `run:` step consuming AI outputs |
| H | Dangerous Sandbox Configs | Full-access sandbox modes, wildcard tool permissions |
| I | Wildcard Allowlists | Wildcard user allowlists permitting any user to trigger |

Vectors H and I are configuration weaknesses that amplify co-occurring injection vectors (A-G).

### Step 5: Report Findings

**Finding structure:**
- Title (vector name as heading)
- Severity: High / Medium / Low / Info
- File and step reference with line number
- Impact (one sentence)
- Evidence (YAML snippet)
- Data Flow (numbered steps from attacker source to consequence)
- Remediation (action-specific)

**Severity factors:**
- External-facing triggers raise severity
- Dangerous sandbox modes raise severity
- Wildcard user allowlists raise severity
- Direct injection (Vector B) rates higher than indirect (Vector A, C, E)
- Elevated permissions + secrets raise severity

## Prometheus Integration

- Use `bash` to run `git` commands for local workflow analysis
- Use `file_read` to examine workflow YAML files
- Use `grep` to search for patterns across workflow files
- Log findings to LCM for tracking across audit sessions
- Use SENTINEL to monitor for workflow changes that introduce new risks
