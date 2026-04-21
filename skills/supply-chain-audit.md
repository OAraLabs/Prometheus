---
name: supply-chain-audit
description: "Identifies dependencies at heightened risk of exploitation or takeover. Use when assessing supply chain attack surface, evaluating dependency health, scoping security engagements, or auditing unmaintained/risky dependencies."
version: 1.0.0
author: Trail of Bits
license: CC-BY-SA-4.0
---
<!-- Provenance: trailofbits/skills | plugins/supply-chain-risk-auditor/skills/supply-chain-risk-auditor/SKILL.md | CC-BY-SA-4.0 -->

# Supply Chain Risk Auditor

Systematically evaluates all dependencies of a project to identify red flags that indicate a high risk of exploitation or takeover, then generates a summary report.

## When to Use

- Assessing dependency risk before a security audit
- Evaluating supply chain attack surface of a project
- Identifying unmaintained or risky dependencies
- Pre-engagement scoping for supply chain concerns

## When NOT to Use

- Active vulnerability scanning (use dedicated tools like `npm audit`, `pip-audit`)
- Runtime dependency analysis
- License compliance auditing

## Risk Criteria

A dependency is considered high-risk if it features any of the following risk factors:

### Single Maintainer or Small Team
The project is primarily maintained by a single individual or small group, not backed by an organization (Linux Foundation, etc.) or a company. If the individual is anonymous (GitHub identity not tied to real-world identity), the risk is significantly greater.
**Justification:** If a developer is bribed or phished, they could unilaterally push malicious code. Consider the left-pad incident.

### Unmaintained
The project is stale (no updates for a long period), explicitly deprecated/archived, or the maintainer has noted the project is inactive, understaffed, or seeking new maintainers. Large numbers of unresponded bug/security issues (not feature requests) are a signal.
**Justification:** If vulnerabilities are identified, they may not be patched in a timely manner.

### Low Popularity
Relatively low GitHub stars and/or downloads compared to other dependencies used by the target.
**Justification:** Fewer users means fewer eyes on the project. Malicious code introductions will not be noticed quickly.

### High-Risk Features
The project implements features especially prone to exploitation: FFI, deserialization, or third-party code execution.
**Justification:** These dependencies are key to the target's security posture and need a high bar of scrutiny.

### Presence of Past CVEs
High or critical severity CVEs, especially many relative to popularity and complexity.
**Justification:** Not necessarily concerning for extremely popular projects under more scrutiny, but a red flag for smaller projects.

### Absence of Security Contact
No security contact listed in `.github/SECURITY.md`, `CONTRIBUTING.md`, `README.md`, or the project's website.
**Justification:** Vulnerability reporters will have difficulty reporting safely and in a timely manner.

## Prerequisites

Ensure that the `gh` CLI tool is available before continuing. Ask the user to install if not found.

## Workflow

### Phase 1: Initial Setup

1. Create a `.supply-chain-risk-auditor` directory for workspace
2. Start a `results.md` report file based on the template below
3. Find all git repositories for direct dependencies
4. Normalize repository entries to URLs (prepend GitHub URL if just `name/project` format)

### Phase 2: Dependency Audit

For each dependency whose repository was identified:

1. Evaluate its risk according to the Risk Criteria above
2. For any criteria requiring data (stars, open issues, etc.), use the `gh` tool to query exact data. Numbers cited MUST be accurate. Round using `~` notation (e.g., "~4000 stars").
3. If a dependency satisfies any risk criteria, add it to the High-Risk Dependencies table in `results.md`, noting the reason for flagging
4. Skip low-risk dependencies; only note dependencies with at least one risk factor
5. Do not note "opposites" of risk factors (e.g., "organization backed")

### Phase 3: Post-Audit

1. For each high-risk dependency, fill out the Suggested Alternative field with an alternative that is more popular, better maintained, etc. Prefer direct successors and drop-in replacements.
2. Note total counts for each risk factor category in the Counts table
3. Summarize overall security posture in the Executive Summary
4. Summarize recommendations

## Report Template

```markdown
# Supply Chain Risk Report

## Metadata

- **Scan Date**: [YYYY-MM-DD HH:MM:SS]
- **Project**: [Project Name]
- **Repositories Scanned**: [X repositories]
- **Total Dependencies**: [Y dependencies]
- **Scan Duration**: [Duration]

## Executive Summary

### Counts by Risk Factor

| Risk Factor | Dependencies | Total |
|-------------|--------------|-------|
| Single maintainer | X, Y, Z | # |
| Unmaintained | X, Y, Z | # |
| Low popularity | X, Y, Z | # |
| High-risk features | X, Y, Z | # |
| Past CVEs | X, Y, Z | # |
| No security contact | X, Y, Z | # |
| **Total** | -- | **#** |

### High-Risk Dependencies

| Dependency Name | Risk Factors | Notes | Suggested Alternative |
|-----------------|--------------|-------|-----------------------|
| X | factors | summary | **Y** - justification |

## Suggested Alternatives

## Report Generated By

Supply Chain Risk Auditor Skill
Generated: [YYYY-MM-DD HH:MM:SS]
```
