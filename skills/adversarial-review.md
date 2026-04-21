---
name: adversarial-review
description: "Use when reviewing code changes, PRs, or files to break self-review blind spots. Runs three hostile personas (Saboteur, New Hire, Security Auditor) that each MUST find issues — no LGTM escapes."
version: 1.0.0
author: alirezarezvani/claude-skills
license: MIT
---
<!-- Provenance: alirezarezvani/claude-skills | engineering-team/adversarial-reviewer/SKILL.md | MIT -->

# Adversarial Code Review

## Overview

When an AI reviews code it wrote (or just read), it shares the same mental model, assumptions, and blind spots as the author. This produces rubber-stamp "LGTM" reviews that a fresh human reviewer would flag immediately.

This skill forces genuine perspective shifts through three adversarial personas -- each with different priorities, fears, and definitions of "bad code." Every persona MUST find at least one issue. Findings caught by 2+ personas are promoted one severity level.

## When to Use

- Before merging any PR -- especially self-authored PRs with no human reviewer
- After a long coding session -- fatigue produces blind spots
- When a previous review said "looks good" too easily
- On security-sensitive code -- auth, payments, data access, API endpoints
- When something "feels off" -- trust that instinct

## Review Workflow

### Step 1: Gather the Changes

- **No arguments:** `git diff` (unstaged) + `git diff --cached` (staged). If both empty, `git diff HEAD~1`.
- **Specific ref:** `git diff <ref>`.
- **Specific file:** Read the entire file.

If no changes found, stop: "Nothing to review."

### Step 2: Read Full Context

For every file in the diff:
1. Read the **full file** -- bugs hide in how new code interacts with existing code.
2. Identify the **purpose** of the change: bug fix, new feature, refactor, config change, test.
3. Note **project conventions** from CLAUDE.md, linting configs, or existing patterns.

### Step 3: Run All Three Personas

Execute each persona sequentially. Each MUST produce at least one finding. Do not soften findings. Do not hedge. Be direct.

### Step 4: Deduplicate and Synthesize

1. Merge duplicate findings (same issue caught by multiple personas).
2. Promote findings caught by 2+ personas to the next severity level.
3. Produce the final structured output.

## The Three Personas

### Persona 1: The Saboteur

**Mindset:** "I am trying to break this code in production."

**Priorities:**
- Input that was never validated
- State that can become inconsistent
- Concurrent access without synchronization
- Error paths that swallow exceptions or return misleading results
- Assumptions about data format, size, or availability that could be violated
- Off-by-one errors, integer overflow, null/undefined dereferences
- Resource leaks (file handles, connections, subscriptions, listeners)

**Process:**
1. For each function changed: "What is the worst input I could send this?"
2. For each external call: "What if this fails, times out, or returns garbage?"
3. For each state mutation: "What if this runs twice? Concurrently? Never?"
4. For each conditional: "What if neither branch is correct?"

MUST find at least one issue. If the code is genuinely bulletproof, note the most fragile assumption it relies on.

### Persona 2: The New Hire

**Mindset:** "I just joined this team. I need to understand and modify this code in 6 months with zero context."

**Priorities:**
- Names that don't communicate intent
- Logic requiring 3+ other files to understand
- Magic numbers, magic strings, unexplained constants
- Functions doing more than one thing
- Missing type information forcing call-chain tracing
- Inconsistency with surrounding code style
- Tests that test implementation details instead of behavior
- Comments describing *what* (redundant) instead of *why* (useful)

**Process:**
1. Read each function as if you've never seen the codebase. Can you understand it from name, parameters, and body alone?
2. Trace one code path end-to-end. How many files do you need to open?
3. Would a new contributor know where to add a similar feature?
4. Look for implicit knowledge baked into the code.

MUST find at least one issue. If crystal clear, note the most likely point of confusion for a newcomer.

### Persona 3: The Security Auditor

**Mindset:** "This code will be attacked. Find the vulnerability before an attacker does."

**OWASP-Informed Checklist:**

| Category | What to Look For |
|----------|-----------------|
| Injection | SQL, NoSQL, OS command, LDAP -- any user input reaching a query without parameterization |
| Broken Auth | Hardcoded credentials, missing auth checks on new endpoints, session tokens in URLs or logs |
| Data Exposure | Sensitive data in error messages, logs, or API responses; missing encryption |
| Insecure Defaults | Debug mode left on, permissive CORS, wildcard permissions, default passwords |
| Missing Access Control | IDOR, missing role checks, privilege escalation paths |
| Dependency Risk | New deps with known CVEs, pinned to vulnerable versions |
| Secrets | API keys, tokens, passwords in code, config, or comments |

**Process:**
1. Identify every trust boundary (user input, API calls, database, file system, env vars).
2. For each: is input validated? Is output sanitized? Is least privilege followed?
3. Could an authenticated user escalate privileges through this change?
4. Does this change expose new attack surface?

MUST find at least one issue.

## Severity Classification

| Severity | Definition | Action |
|----------|-----------|--------|
| CRITICAL | Data loss, security breach, or production outage. | Block merge. |
| WARNING | Edge-case bugs, performance degradation, or maintainability problems. | Fix or accept risk with justification. |
| NOTE | Style issue, minor improvement, or documentation gap. | Author's discretion. |

**Promotion rule:** Finding flagged by 2+ personas is promoted one level.

## Output Format

```markdown
## Adversarial Review: [brief description]

**Scope:** [files reviewed, lines changed, type of change]
**Verdict:** BLOCK / CONCERNS / CLEAN

### Critical Findings
[If any -- these block the merge]

### Warnings
[Should-fix items]

### Notes
[Nice-to-fix items]

### Summary
[2-3 sentences: overall risk profile, single most important thing to fix]
```

**Verdicts:**
- **BLOCK** -- 1+ CRITICAL findings. Do not merge.
- **CONCERNS** -- No criticals but 2+ warnings. Merge at your own risk.
- **CLEAN** -- Only notes. Safe to merge.

## Breaking the Self-Review Trap

When reviewing code you just wrote or read:
1. Read the code **bottom-up** (start from the last function, work backward).
2. For each function, state its contract **before** reading the body. Does the body match?
3. Assume every variable could be null/undefined until proven otherwise.
4. Assume every external call will fail.
5. Ask: "If I deleted this change entirely, what would break?" -- if "nothing," the change might be unnecessary.

## Anti-Patterns

| Anti-Pattern | Why It's Wrong |
|-------------|---------------|
| "LGTM, no issues found" | If you found nothing, you didn't look hard enough. |
| Cosmetic-only findings | Reporting only whitespace while missing a null dereference is worse than no review. |
| Pulling punches | "This might possibly be a minor concern..." -- No. Be direct. |
| Restating the diff | "This function was added to handle auth" is not a finding. What's WRONG? |
| Ignoring test gaps | New code without tests is a finding. Always. |
| Reviewing only changed lines | Bugs live in the interaction between new and existing code. Read the full file. |
