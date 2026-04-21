---
name: multi-agent-coordination
description: Use when multiple agents must collaborate on a single outcome. Covers parallel vs sequential dispatch, role assignment, handoff protocols, conflict resolution, and result merging.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | patterns/08-multi-agent-coordination.md | MIT -->

# Multi-Agent Coordination

## Overview

Multi-agent coordination defines how several agents collaborate on one outcome without creating conflicting changes. It requires shared goals, clear ownership, and synchronization points.

Assign stable roles. Each role should produce specific artifacts so integration is mechanical instead of conversational guesswork. This pattern is most effective when coordination overhead stays small and each agent has a narrow, testable objective.

## Why It Matters

- Prevents duplicated or conflicting edits across parallel agents.
- Improves throughput by splitting work by responsibility.
- Increases quality through built-in review and verification lanes.
- Makes failures easier to isolate to a role or handoff point.

## Role Definitions

### Planner
- Defines scope and task graph.
- Breaks work into non-overlapping subtasks with acceptance criteria.
- Owns the overall architecture of the solution.

### Implementer
- Applies code changes within assigned scope.
- Returns changed files and decision notes.
- Does not exceed assigned scope boundaries.

### Reviewer
- Checks correctness, maintainability, and consistency with project conventions.
- Flags issues with actionable fixes (not vague suggestions).

### Verifier
- Runs tests, checks, and validation steps.
- Reports evidence (pass/fail with output), not opinions.

### Coordinator
- Resolves conflicts between agent outputs.
- Integrates results into the final deliverable.
- Publishes the unified outcome.

## Handoff Protocol

1. **Planner** issues scoped tasks with acceptance criteria to implementers.
2. **Implementer** returns changed files and decision notes to coordinator.
3. **Reviewer** flags issues with actionable fixes back to implementer or coordinator.
4. **Verifier** confirms behavior with explicit checks and reports evidence.
5. **Coordinator** resolves conflicts and publishes final integrated output.

## Parallel vs Sequential Dispatch

### Use Parallel Dispatch When:
- Subtasks are independent (no shared files or state).
- Each agent's scope is clearly bounded.
- Results can be merged mechanically.

### Use Sequential Dispatch When:
- Later tasks depend on earlier results.
- Shared state must be consistent (e.g., database migrations before code changes).
- Review must happen before the next implementation step.

## Conflict Resolution

- If two outputs disagree, prioritize **verified evidence** over unverified claims.
- Reroute unresolved items for rework rather than guessing which output is correct.
- When conflicts stem from ambiguous requirements, escalate to the user for clarification.

## Result Merging Checklist

1. Collect all agent outputs.
2. Check for file-level conflicts (two agents modified the same file).
3. Check for semantic conflicts (two agents made incompatible assumptions).
4. Run integration-level verification after merging.
5. Document any manual conflict resolutions with rationale.

## Variations

- Collapse planner/reviewer roles for smaller tasks.
- Add a security reviewer for sensitive repositories.
- Add a "single-writer policy" to avoid merge conflicts (only one agent writes to any given file).
