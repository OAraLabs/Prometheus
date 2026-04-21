---
name: agent-delegation
description: Use when a task should be split across sub-agents or helpers. Covers when to delegate, how to structure delegation prompts, scope boundaries, and result aggregation.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | patterns/05-agent-delegation.md | MIT -->

# Agent Delegation

## Overview

Delegation helps a primary agent split work across specialized helpers while keeping the overall approach consistent. The parent agent keeps ownership of intent, scope, and final synthesis.

Use delegation when tasks are parallelizable, domain-specific, or too large for one uninterrupted pass. Each delegated unit should have a crisp objective, expected output, and completion criteria.

## When to Delegate

- **Parallelizable work:** Independent subtasks that can run simultaneously (e.g., searching multiple directories, running tests on different modules).
- **Domain-specific work:** Tasks requiring specialized knowledge or tooling (e.g., security review, performance profiling, documentation generation).
- **Context-heavy work:** Tasks that would overload a single agent's context window if done sequentially.
- **Verification work:** Separate the implementer from the verifier for higher confidence.

## When NOT to Delegate

- Simple, short tasks that complete faster in a single pass.
- Tasks with tight sequential dependencies where each step depends on the previous result.
- When the overhead of structuring delegation exceeds the time saved.

## Delegation Rules

1. **Delegate only when it improves speed or quality.**
2. **Keep one owner (the parent) responsible for final correctness.**
3. **Provide each helper with:**
   - Goal: what specifically to accomplish.
   - Scope boundaries: what to touch and what to leave alone.
   - Required output format: how to report results.
   - Validation expectations: how success is measured.

## Structuring a Delegation Prompt

```
Goal: [Specific, measurable objective]
Scope: [Files, directories, or systems in scope. Explicitly state what is OUT of scope.]
Output format: [Expected structure of the result -- e.g., list of findings, code patch, test results]
Completion criteria: [How to know when the task is done]
Constraints: [Any rules the helper must follow -- e.g., do not modify files, read-only investigation]
```

## Parent Responsibilities

1. **Break request** into non-overlapping subtasks.
2. **Dispatch** with explicit acceptance criteria.
3. **Review** returned outputs for consistency and conflicts.
4. **Integrate** results, resolve gaps, and run final verification.
5. **Report** the unified outcome.

## Result Aggregation

- Check all helper outputs for mutual consistency.
- Resolve conflicts by preferring verified evidence over unverified claims.
- If helpers disagree, investigate the specific point of disagreement rather than picking arbitrarily.
- Run integration-level verification after merging results.

## Variations

- **Research-only delegation:** Helpers gather information but do not make changes.
- **Code + test split:** One helper writes implementation, another writes tests first.
- **Timeout and fallback:** Define maximum wait time for delegated work; escalate or reassign if stalled.
