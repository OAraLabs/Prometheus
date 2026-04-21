---
name: system-prompt-architecture
description: Use when designing or reviewing system prompts for AI agents. Covers layered identity construction, safety layers, behavioral constraints, and instruction priority hierarchies.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | patterns/01-system-prompt-architecture.md | MIT -->

# System Prompt Architecture

## Overview

A strong system prompt is the operating contract for a coding agent. It defines mission, scope, boundaries, and quality expectations before any task-specific instruction appears.

Organize it in layers: identity, non-negotiable constraints, execution workflow, and output format. This keeps high-priority behavior stable while allowing user requests to vary safely.

## Why Layered Architecture Matters

- Reduces ambiguity by making priorities explicit from the start.
- Prevents instruction conflicts through a predictable rule hierarchy.
- Improves output consistency across different tasks and users.
- Makes audits easier because behavior is tied to named sections.

## The Layers

### 1. Identity and Scope

Define what the agent is, what it does, and what it does not do.

```
PRIMARY OBJECTIVE
- Deliver correct, maintainable code changes that satisfy the user request.

ROLE AND SCOPE
- Operate as an implementation-focused engineer.
- Prefer concrete edits and verification over speculative discussion.
```

### 2. Non-Negotiable Constraints

These override everything else. Place them high in the prompt so they take priority.

```
NON-NEGOTIABLE RULES
- Follow instruction priority: system > developer > user > tool feedback.
- Do not perform destructive actions without explicit approval.
- Preserve unrelated local changes.
- Keep secrets out of logs, code, and commit text.
```

### 3. Execution Workflow

Define the standard sequence of operations the agent follows for any task.

```
EXECUTION WORKFLOW
1) Understand the request and identify affected files.
2) Inspect relevant code and dependencies.
3) Implement minimal, focused changes.
4) Run checks/tests for changed behavior.
5) Report what changed, why, and how it was verified.
```

### 4. Quality Bar

Set explicit standards for code quality that the agent must meet.

```
QUALITY BAR
- Favor readable, testable code.
- Keep backward compatibility unless asked otherwise.
- Document non-obvious decisions briefly.
```

### 5. Output Format

Specify how the agent communicates results.

```
OUTPUT
- Start with outcome.
- List key file changes.
- Include verification results and next actions if needed.
```

## Variations

- Add language-specific quality gates for Python, TypeScript, or Go projects.
- Add a "performance first" section for latency-sensitive services.
- Add a "migration safety" section for schema or API transition work.

## Checklist for Reviewing a System Prompt

1. Does it clearly define identity and scope?
2. Are non-negotiable constraints listed before task instructions?
3. Is the instruction priority hierarchy explicit?
4. Does the execution workflow cover discover, implement, verify, report?
5. Are quality standards concrete and testable (not aspirational)?
6. Is the output format specified?
