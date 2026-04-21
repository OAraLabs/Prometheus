---
name: subagent-driven-development
description: "Use when executing implementation plans with independent tasks in the current session - dispatches fresh subagent per task with two-stage review (spec compliance then code quality) after each."
version: 1.0.0
author: obra/superpowers
license: MIT
---
<!-- Provenance: obra/superpowers | skills/subagent-driven-development/SKILL.md | MIT -->

# Subagent-Driven Development

Execute plan by dispatching fresh subagent per task, with two-stage review after each: spec compliance review first, then code quality review.

**Why subagents:** You delegate tasks to specialized agents with isolated context. By precisely crafting their instructions and context, you ensure they stay focused and succeed at their task. They should never inherit your session's context or history -- you construct exactly what they need. This also preserves your own context for coordination work.

**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration

## When to Use

**Use when:**
- You have an implementation plan with discrete tasks
- Tasks are mostly independent
- You want to stay in the current session

**Don't use when:**
- No implementation plan exists (brainstorm first)
- Tasks are tightly coupled (manual execution better)

## The Process

```
Read plan -> Extract all tasks with full text -> Track tasks

Per Task:
  1. Dispatch implementer subagent
  2. If implementer asks questions -> answer, re-dispatch
  3. Implementer implements, tests, commits, self-reviews
  4. Dispatch spec reviewer subagent
  5. If spec issues -> implementer fixes -> re-review
  6. Dispatch code quality reviewer subagent
  7. If quality issues -> implementer fixes -> re-review
  8. Mark task complete

After all tasks:
  Dispatch final code reviewer for entire implementation
  Use finishing-development-branch skill
```

## Model Selection

Use the least powerful model that can handle each role to conserve cost and increase speed.

**Mechanical implementation tasks** (isolated functions, clear specs, 1-2 files): use a fast, cheap model. Most implementation tasks are mechanical when the plan is well-specified.

**Integration and judgment tasks** (multi-file coordination, pattern matching, debugging): use a standard model.

**Architecture, design, and review tasks**: use the most capable available model.

**Task complexity signals:**
- Touches 1-2 files with a complete spec -> cheap model
- Touches multiple files with integration concerns -> standard model
- Requires design judgment or broad codebase understanding -> most capable model

## Handling Implementer Status

Implementer subagents report one of four statuses. Handle each appropriately:

**DONE:** Proceed to spec compliance review.

**DONE_WITH_CONCERNS:** The implementer completed the work but flagged doubts. Read the concerns before proceeding. If the concerns are about correctness or scope, address them before review. If they're observations (e.g., "this file is getting large"), note them and proceed to review.

**NEEDS_CONTEXT:** The implementer needs information that wasn't provided. Provide the missing context and re-dispatch.

**BLOCKED:** The implementer cannot complete the task. Assess the blocker:
1. If it's a context problem, provide more context and re-dispatch with the same model
2. If the task requires more reasoning, re-dispatch with a more capable model
3. If the task is too large, break it into smaller pieces
4. If the plan itself is wrong, escalate to the human

**Never** ignore an escalation or force the same model to retry without changes. If the implementer said it's stuck, something needs to change.

## Implementer Subagent Prompt Template

```
You are implementing Task N: [task name]

## Task Description

[FULL TEXT of task from plan - paste it here, don't make subagent read file]

## Context

[Scene-setting: where this fits, dependencies, architectural context]

## Before You Begin

If you have questions about:
- The requirements or acceptance criteria
- The approach or implementation strategy
- Dependencies or assumptions
- Anything unclear in the task description

**Ask them now.** Raise any concerns before starting work.

## Your Job

Once you're clear on requirements:
1. Implement exactly what the task specifies
2. Write tests (following TDD if task says to)
3. Verify implementation works
4. Commit your work
5. Self-review (see below)
6. Report back

Work from: [directory]

**While you work:** If you encounter something unexpected or unclear, **ask questions**.
It's always OK to pause and clarify. Don't guess or make assumptions.

## Code Organization

- Follow the file structure defined in the plan
- Each file should have one clear responsibility with a well-defined interface
- If a file you're creating is growing beyond the plan's intent, stop and report
  it as DONE_WITH_CONCERNS -- don't split files on your own without plan guidance
- If an existing file you're modifying is already large or tangled, work carefully
  and note it as a concern in your report
- In existing codebases, follow established patterns

## When You're in Over Your Head

It is always OK to stop and say "this is too hard for me." Bad work is worse than
no work. You will not be penalized for escalating.

**STOP and escalate when:**
- The task requires architectural decisions with multiple valid approaches
- You need to understand code beyond what was provided and can't find clarity
- You feel uncertain about whether your approach is correct
- The task involves restructuring existing code in ways the plan didn't anticipate

**How to escalate:** Report back with status BLOCKED or NEEDS_CONTEXT.

## Before Reporting Back: Self-Review

Review your work with fresh eyes:

**Completeness:** Did I fully implement everything in the spec? Missing requirements? Unhandled edge cases?

**Quality:** Is this my best work? Are names clear and accurate? Is the code clean and maintainable?

**Discipline:** Did I avoid overbuilding (YAGNI)? Did I only build what was requested? Did I follow existing patterns?

**Testing:** Do tests actually verify behavior (not just mock behavior)? Are tests comprehensive?

If you find issues during self-review, fix them now before reporting.

## Report Format

When done, report:
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented (or what you attempted, if blocked)
- What you tested and test results
- Files changed
- Self-review findings (if any)
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

The implementer's report may be incomplete, inaccurate, or optimistic. You MUST verify
everything independently.

**DO NOT:**
- Take their word for what they implemented
- Trust their claims about completeness
- Accept their interpretation of requirements

**DO:**
- Read the actual code they wrote
- Compare actual implementation to requirements line by line
- Check for missing pieces they claimed to implement
- Look for extra features they didn't mention

## Your Job

Read the implementation code and verify:

**Missing requirements:** Did they implement everything requested? Requirements skipped or missed?

**Extra/unneeded work:** Did they build things that weren't requested? Over-engineer? Add unnecessary features?

**Misunderstandings:** Did they interpret requirements differently than intended? Solve the wrong problem?

**Verify by reading code, not by trusting report.**

Report:
- PASS: Spec compliant (if everything matches after code inspection)
- FAIL: Issues found: [list specifically what's missing or extra, with file:line references]
```

## Code Quality Reviewer Template

Only dispatch after spec compliance review passes.

```
Review the implementation for code quality:

## What Was Implemented

[From implementer's report]

## Review Scope

BASE_SHA: [commit before task]
HEAD_SHA: [current commit]

## Review Checklist

In addition to standard code quality concerns, check:
- Does each file have one clear responsibility with a well-defined interface?
- Are units decomposed so they can be understood and tested independently?
- Is the implementation following the file structure from the plan?
- Did this implementation create new files that are already large, or significantly
  grow existing files? (Don't flag pre-existing file sizes -- focus on what this
  change contributed.)

Report: Strengths, Issues (Critical/Important/Minor), Assessment (Approved/Changes Needed)
```

## Example Workflow

```
You: I'm using Subagent-Driven Development to execute this plan.

[Read plan file once]
[Extract all 5 tasks with full text and context]
[Track all tasks]

Task 1: Hook installation script

[Get Task 1 text and context (already extracted)]
[Dispatch implementation subagent with full task text + context]

Implementer: "Before I begin - should the hook be installed at user or system level?"

You: "User level (~/.config/hooks/)"

Implementer: "Got it. Implementing now..."
[Later] Implementer:
  - Implemented install-hook command
  - Added tests, 5/5 passing
  - Self-review: Found I missed --force flag, added it
  - Committed

[Dispatch spec compliance reviewer]
Spec reviewer: PASS - all requirements met, nothing extra

[Get git SHAs, dispatch code quality reviewer]
Code reviewer: Strengths: Good test coverage, clean. Issues: None. Approved.

[Mark Task 1 complete]

Task 2: Recovery modes
...
```

## Advantages

**vs. Manual execution:**
- Subagents follow TDD naturally
- Fresh context per task (no confusion)
- Parallel-safe (subagents don't interfere)
- Subagent can ask questions (before AND during work)

**Quality gates:**
- Self-review catches issues before handoff
- Two-stage review: spec compliance, then code quality
- Review loops ensure fixes actually work
- Spec compliance prevents over/under-building
- Code quality ensures implementation is well-built

## Red Flags

**Never:**
- Start implementation on main/master branch without explicit user consent
- Skip reviews (spec compliance OR code quality)
- Proceed with unfixed issues
- Dispatch multiple implementation subagents in parallel (conflicts)
- Make subagent read plan file (provide full text instead)
- Skip scene-setting context (subagent needs to understand where task fits)
- Ignore subagent questions (answer before letting them proceed)
- Accept "close enough" on spec compliance (reviewer found issues = not done)
- Skip review loops (reviewer found issues = implementer fixes = review again)
- Let implementer self-review replace actual review (both are needed)
- **Start code quality review before spec compliance passes** (wrong order)
- Move to next task while either review has open issues

**If subagent asks questions:**
- Answer clearly and completely
- Provide additional context if needed
- Don't rush them into implementation

**If reviewer finds issues:**
- Implementer (same subagent) fixes them
- Reviewer reviews again
- Repeat until approved
- Don't skip the re-review

**If subagent fails task:**
- Dispatch fix subagent with specific instructions
- Don't try to fix manually (context pollution)

## Integration

**Required workflow skills:**
- **git-worktrees** - Set up isolated workspace before starting
- **writing-plans** - Creates the plan this skill executes
- **finishing-development-branch** - Complete development after all tasks

**Subagents should use:**
- **test-driven-development** - Subagents follow TDD for each task
