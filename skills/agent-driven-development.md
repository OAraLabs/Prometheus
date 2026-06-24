---
name: agent-driven-development
description: "Use when executing implementation plans with independent tasks - dispatches a fresh agent per task via the agent tool, with two-stage review (spec compliance then code quality) after each. Use for plan execution, parallel task dispatch, and structured code review workflows."
---

# Agent-Driven Development

Execute an implementation plan by dispatching a fresh agent per task via the agent tool, with two-stage review after each: spec compliance first, then code quality.

**Why agents:** Delegate tasks to specialized agents with isolated context. By precisely crafting their instructions and context, they stay focused and succeed. They should never inherit your session context -- you construct exactly what they need. This also preserves your own context for coordination.

**Core principle:** Fresh agent per task + two-stage review (spec then quality) = high quality, fast iteration

## When to Use

**Use when:**
- You have an implementation plan with discrete tasks
- Tasks are mostly independent
- You want structured code review on each piece

**Do not use when:**
- No implementation plan exists (brainstorm first)
- Tasks are tightly coupled (manual execution better)

## The Process

```
Read plan -> Extract all tasks with full text -> Track tasks

Per Task:
  1. Dispatch implementer agent
  2. If implementer asks questions -> answer, re-dispatch
  3. Implementer implements, tests, commits, self-reviews
  4. Dispatch spec reviewer agent
  5. If spec issues -> implementer fixes -> re-review
  6. Dispatch code quality reviewer agent
  7. If quality issues -> implementer fixes -> re-review
  8. Mark task complete

After all tasks:
  Dispatch final code reviewer for entire implementation
```

## Model Selection

Use the least powerful model that can handle each role to conserve cost and speed.

**Mechanical tasks** (isolated functions, clear specs, 1-2 files): fast, cheap model.
**Integration and judgment tasks** (multi-file, pattern matching, debugging): standard model.
**Architecture, design, and review tasks**: most capable available model.

## Handling Implementer Status

**DONE:** Proceed to spec compliance review.

**DONE_WITH_CONCERNS:** Read the concerns. If about correctness/scope, address before review. If observations, note and proceed.

**NEEDS_CONTEXT:** Provide missing context and re-dispatch.

**BLOCKED:** Assess the blocker:
1. Context problem -> provide more context, re-dispatch
2. Needs more reasoning -> re-dispatch with more capable model
3. Task too large -> break into smaller pieces
4. Plan is wrong -> escalate to the human

Never ignore an escalation or force the same model to retry without changes.

## Implementer Agent Prompt Template

```
You are implementing Task N: [task name]

## Task Description
[FULL TEXT of task from plan - paste it, do not make the agent read a file]

## Context
[Where this fits, dependencies, architectural context]

## Before You Begin
If you have questions about requirements, approach, dependencies, or anything unclear -- ask them now.

## Your Job
1. Implement exactly what the task specifies
2. Write tests
3. Verify implementation works (run via bash)
4. Commit your work (use bash with git commands)
5. Self-review (see below)
6. Report back

Work from: [directory]

## Code Organization
- Follow the file structure defined in the plan
- Each file should have one clear responsibility
- If a file grows beyond plan intent, stop and report DONE_WITH_CONCERNS
- In existing codebases, follow established patterns

## Self-Review Before Reporting
- Completeness: Did I implement everything in the spec?
- Quality: Are names clear? Is code clean and maintainable?
- Discipline: Did I avoid overbuilding? Only what was requested?
- Testing: Do tests verify behavior, not just mock it?

## Report Format
- Status: DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented
- What you tested and results
- Files changed
- Any issues or concerns
```

## Spec Compliance Reviewer Template

```
You are reviewing whether an implementation matches its specification.

## What Was Requested
[FULL TEXT of task requirements]

## What Implementer Claims They Built
[From implementer's report]

## CRITICAL: Do Not Trust the Report
Read the actual code. Compare to requirements line by line.

Check for:
- Missing requirements
- Extra/unneeded work
- Misunderstandings of spec

Report:
- PASS: Spec compliant
- FAIL: Issues found: [list with file:line references]
```

## Code Quality Reviewer Template

Only dispatch after spec compliance passes.

```
Review the implementation for code quality.

## What Was Implemented
[From implementer's report]

## Review Checklist
- Does each file have one clear responsibility?
- Are units decomposed for independent testing?
- Does it follow the file structure from the plan?
- Did this change create large files or significantly grow existing ones?

Report: Strengths, Issues (Critical/Important/Minor), Assessment (Approved/Changes Needed)
```

## Red Flags

**Never:**
- Start implementation on main/master without explicit user consent
- Skip reviews (spec compliance OR code quality)
- Proceed with unfixed issues
- Dispatch multiple implementation agents in parallel (conflicts)
- Make agent read plan file (provide full text instead)
- Skip scene-setting context
- Ignore agent questions
- Accept "close enough" on spec compliance
- Start code quality review before spec compliance passes

**If reviewer finds issues:**
- Implementer agent fixes them
- Reviewer reviews again
- Repeat until approved

**If agent fails task:**
- Dispatch fix agent with specific instructions
- Do not try to fix manually (context pollution)

## Integration with Prometheus

- Use the **agent tool** to dispatch implementer, reviewer, and quality agents
- Use **bash** with git commands for version control operations
- Use **file_read** / **file_edit** for code inspection and modification
- Reference **writing-plans** skill for creating the plan this skill executes
- Reference **finishing-development-branch** skill for post-implementation cleanup
