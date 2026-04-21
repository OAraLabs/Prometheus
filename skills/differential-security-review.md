---
name: differential-security-review
description: "Performs security-focused differential review of code changes (PRs, commits, diffs). Use when reviewing PRs, analyzing commit diffs, or auditing code changes for security regressions, blast radius, and test coverage gaps."
version: 1.0.0
author: Trail of Bits
license: CC-BY-SA-4.0
---
<!-- Provenance: trailofbits/skills | plugins/differential-review/skills/differential-review/SKILL.md | CC-BY-SA-4.0 -->

# Differential Security Review

Security-focused code review for PRs, commits, and diffs.

## Core Principles

1. **Risk-First**: Focus on auth, crypto, value transfer, external calls
2. **Evidence-Based**: Every finding backed by git history, line numbers, attack scenarios
3. **Adaptive**: Scale to codebase size (SMALL/MEDIUM/LARGE)
4. **Honest**: Explicitly state coverage limits and confidence level
5. **Output-Driven**: Always generate comprehensive markdown report file

## Rationalizations to Reject

| Rationalization | Why It's Wrong | Required Action |
|-----------------|----------------|-----------------|
| "Small PR, quick review" | Heartbleed was 2 lines | Classify by RISK, not size |
| "I know this codebase" | Familiarity breeds blind spots | Build explicit baseline context |
| "Git history takes too long" | History reveals regressions | Never skip Phase 1 |
| "Blast radius is obvious" | You'll miss transitive callers | Calculate quantitatively |
| "No tests = not my problem" | Missing tests = elevated risk rating | Flag in report, elevate severity |
| "Just a refactor, no security impact" | Refactors break invariants | Analyze as HIGH until proven LOW |
| "I'll explain verbally" | No artifact = findings lost | Always write report |

## Codebase Size Strategy

| Codebase Size | Strategy | Approach |
|---------------|----------|----------|
| SMALL (<20 files) | DEEP | Read all deps, full git blame |
| MEDIUM (20-200) | FOCUSED | 1-hop deps, priority files |
| LARGE (200+) | SURGICAL | Critical paths only |

## Risk Level Triggers

| Risk Level | Triggers |
|------------|----------|
| HIGH | Auth, crypto, external calls, value transfer, validation removal |
| MEDIUM | Business logic, state changes, new public APIs |
| LOW | Comments, tests, UI, logging |

## Workflow Overview

```
Pre-Analysis -> Phase 0: Triage -> Phase 1: Code Analysis -> Phase 2: Test Coverage
    |              |                    |                        |
Phase 3: Blast Radius -> Phase 4: Deep Context -> Phase 5: Adversarial -> Phase 6: Report
```

## When NOT to Use This Skill

- **Greenfield code** (no baseline to compare)
- **Documentation-only changes** (no security impact)
- **Formatting/linting** (cosmetic changes)
- **User explicitly requests quick summary only** (they accept risk)

For these cases, use standard code review instead.

## Red Flags (Stop and Investigate)

**Immediate escalation triggers:**
- Removed code from "security", "CVE", or "fix" commits
- Access control modifiers removed (onlyOwner, internal -> external)
- Validation removed without replacement
- External calls added without checks
- High blast radius (50+ callers) + HIGH risk change

These patterns require adversarial analysis even in quick triage.

---

## Pre-Analysis: Baseline Context Building

**FIRST ACTION - Build complete baseline understanding:**

1. Checkout baseline commit and build understanding of the codebase
2. Capture from baseline analysis:
   - System-wide invariants (what must ALWAYS be true across all code)
   - Trust boundaries and privilege levels (who can do what)
   - Validation patterns (what gets checked where -- defense-in-depth)
   - Complete call graphs for critical functions (who calls what)
   - State flow diagrams (how state changes)
   - External dependencies and trust assumptions

**Why this matters:**
- Understand what the code was SUPPOSED to do before changes
- Identify implicit security assumptions in baseline
- Detect when changes violate baseline invariants
- Know which patterns are system-wide vs local
- Catch when changes break defense-in-depth

---

## Phase 0: Intake and Triage

**Extract changes:**
```bash
# For commit range
git diff <base>..<head> --stat
git log <base>..<head> --oneline

# For PR
gh pr view <number> --json files,additions,deletions

# Get all changed files
git diff <base>..<head> --name-only
```

**Classify complexity:**
- **SMALL**: <20 files -- Deep analysis (read all deps)
- **MEDIUM**: 20-200 files -- Focused analysis (1-hop deps)
- **LARGE**: 200+ files -- Surgical (critical paths only)

**Risk score each file** using the Risk Level Triggers table above.

---

## Phase 1: Changed Code Analysis

For each changed file:

1. **Read both versions** (baseline and changed)
2. **Analyze each diff region:**
   - BEFORE: exact code
   - AFTER: exact code
   - CHANGE: behavioral impact
   - SECURITY: implications
3. **Git blame removed code:**
   ```bash
   git log -S "removed_code" --all --oneline
   git blame <baseline> -- file.sol | grep "pattern"
   ```
   Red flags: Removed code from "fix", "security", "CVE" commits = CRITICAL
4. **Check for regressions** (re-added code): Code added -> removed for security -> re-added now = REGRESSION
5. **Micro-adversarial analysis** for each change:
   - What attack did removed code prevent?
   - What new surface does new code expose?
   - Can modified logic be bypassed?
6. **Generate concrete attack scenarios** with preconditions, steps, impact

---

## Phase 2: Test Coverage Analysis

**Risk elevation rules:**
- NEW function + NO tests -> Elevate risk MEDIUM->HIGH
- MODIFIED validation + UNCHANGED tests -> HIGH RISK
- Complex logic (>20 lines) + NO tests -> HIGH RISK

---

## Phase 3: Blast Radius Analysis

**Classify blast radius:**
- 1-5 calls: LOW
- 6-20 calls: MEDIUM
- 21-50 calls: HIGH
- 50+ calls: CRITICAL

**Priority matrix:**

| Change Risk | Blast Radius | Priority | Analysis Depth |
|-------------|--------------|----------|----------------|
| HIGH | CRITICAL | P0 | Deep + all deps |
| HIGH | HIGH/MEDIUM | P1 | Deep |
| HIGH | LOW | P2 | Standard |
| MEDIUM | CRITICAL/HIGH | P1 | Standard + callers |

---

## Phase 4: Deep Context Analysis

For each HIGH RISK changed function, map:
1. Complete function flow (entry conditions, state reads/writes, external calls, return values)
2. Internal calls (recursive call graph)
3. External calls (trust boundaries, assumptions, reentrancy risks)
4. Invariants (what must ALWAYS be true, what must NEVER happen)
5. Five Whys root cause (WHY changed, WHY existed, WHY might break, WHY chosen, WHY could fail)

**Cross-cutting pattern detection:**
```bash
# Find repeated validation patterns
grep -r "require.*amount > 0" --include="*.sol" .
grep -r "onlyOwner" --include="*.sol" .

# Check if any removed in diff
git diff <range> | grep "^-.*require.*amount > 0"
```

Flag if removal breaks defense-in-depth.

---

## Phase 5: Adversarial Vulnerability Analysis

Apply to all HIGH RISK changes after deep context analysis.

### 1. Define Specific Attacker Model
- WHO (unauthenticated, authenticated, admin, compromised service)
- WHAT access/privileges they have
- WHERE they interact with the system

### 2. Identify Concrete Attack Vectors
```
ENTRY POINT: [Exact function/endpoint attacker can access]
ATTACK SEQUENCE:
1. [Specific API call/transaction with parameters]
2. [How this reaches the vulnerable code]
3. [What happens in the vulnerable code]
4. [Impact achieved]
PROOF OF ACCESSIBILITY: Show the function is reachable
```

### 3. Rate Realistic Exploitability
- **EASY:** Single call, public API, no special conditions
- **MEDIUM:** Multiple steps, timing requirements, elevated but obtainable privileges
- **HARD:** Admin privileges, rare edge cases, significant resources

### 4. Cross-Reference with Baseline
- Does this violate a system-wide invariant?
- Does this break a trust boundary?
- Does this bypass a validation pattern?
- Is this a regression of a previous fix?

---

## Phase 6: Report Generation

Generate markdown report with these mandatory sections:

1. **Executive Summary**: Severity distribution, risk assessment, recommendation (APPROVE/REJECT/CONDITIONAL)
2. **What Changed**: Commit timeline, file summary, lines changed
3. **Critical Findings**: For each HIGH/CRITICAL issue -- file, commit, blast radius, description, attack scenario, proof of concept, recommendation
4. **Test Coverage Analysis**: Coverage stats, untested changes, risk assessment
5. **Blast Radius Analysis**: High-impact functions, dependency graph
6. **Historical Context**: Security-related removals, regression risks
7. **Recommendations**: Immediate (blocking), before production, technical debt
8. **Analysis Methodology**: Strategy used, scope, techniques, limitations, confidence level

**Filename format:** `<PROJECT>_DIFFERENTIAL_REVIEW_<DATE>.md`

---

## Common Vulnerability Patterns Quick Reference

| Pattern | Detection | Key Risk |
|---------|-----------|----------|
| Security Regressions | `git log -S "pattern" --grep="security\|fix\|CVE"` | Previously-fixed code re-added |
| Double Decrease/Increase | Two state updates for same event | Accounting corruption |
| Missing Validation | `git diff | grep "^-.*require"` | Removed checks expose vulnerabilities |
| Reentrancy | External call before state update (CEI violation) | Recursive exploitation |
| Access Control Bypass | Removed permission checks | Unauthorized access |
| Race Conditions | Two-step process without commit-reveal | Front-running exploits |
| Unchecked Return Values | External call without success check | Silent failures |
| Denial of Service | Unbounded loops, external call reverts | Function becomes unusable |

## Quality Checklist

Before delivering:

- [ ] All changed files analyzed
- [ ] Git blame on removed security code
- [ ] Blast radius calculated for HIGH risk
- [ ] Attack scenarios are concrete (not generic)
- [ ] Findings reference specific line numbers + commits
- [ ] Report file generated
- [ ] User notified with summary
