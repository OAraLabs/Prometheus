---
name: memory-context-management
description: Use when managing agent memory across long or multi-step tasks. Covers context window optimization, what to persist vs derive, memory lifecycle, and preventing context drift.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | patterns/07-memory-and-context.md | MIT -->

# Memory and Context Management

## Overview

Agents need stable context to avoid repeating work or contradicting earlier decisions. This skill defines what to remember, for how long, and when to refresh from source files.

Track compact memory objects: task goals, constraints, decisions made, open questions, and verification status. Keep memory factual and source-linked rather than speculative.

Effective memory management reduces context drift and keeps long-running tasks coherent without bloating every response.

## Why It Matters

- Preserves intent across multi-step implementation sessions.
- Reduces repeated analysis and duplicate code edits.
- Improves consistency in decisions, naming, and architecture choices.
- Helps recover quickly after interruptions or context switches.

## The Memory Model

Track these five categories at all times during multi-step work:

### 1. Goal
What must be delivered. Keep this statement crisp and reference it when making trade-off decisions.

### 2. Constraints
Non-negotiable rules and boundaries. These come from the system prompt, the user, or discovered project conventions.

### 3. Decisions
Choices made during execution and short rationale for each. This prevents revisiting settled questions.

### 4. Open Questions
Unresolved items that block confidence. Tag each with what information would resolve it.

### 5. Verification State
What has been tested, what passed, what failed, and what remains untested.

## Memory Rules

1. **Update after each major step.** Do not let memory go stale across multiple actions.
2. **Prefer file-backed facts over inferred assumptions.** If you can read it from a file, do not guess from memory.
3. **Expire stale assumptions when new evidence appears.** If you assumed a function signature and then read the actual code, update immediately.
4. **Before final response, reconcile memory against current code.** Verify that your understanding matches the actual state of the files.

## What to Persist vs Derive

### Persist (save to memory/notes)
- Task goal and constraints
- Key decisions with rationale
- Verification results
- Known blockers and open questions
- File paths and line numbers of important locations

### Derive (re-read from source each time)
- Exact file contents (they may have changed)
- Current git state
- Test output (always re-run, never trust cached results)
- Environment variables and configuration values

## Context Window Optimization

- **Summarize completed work** rather than carrying full transcripts of finished subtasks.
- **Drop intermediate reasoning** once a decision is finalized. Keep the decision and rationale, discard the exploration.
- **Reference files by path** rather than quoting large blocks of code in memory.
- **Use structured formats** (lists, tables) over prose for memory objects -- they are more compact and scannable.

## Memory Lifecycle

1. **Initialize:** Set goal, constraints, and initial open questions at task start.
2. **Accumulate:** Add decisions, findings, and verification results as work progresses.
3. **Prune:** Remove resolved open questions, collapse completed subtask details into summaries.
4. **Reconcile:** Before final delivery, verify memory matches actual file state.
5. **Handoff:** If the session ends or context resets, produce a handoff note with current state.

## Variations

- Add per-file memory tags for large refactors.
- Add a "session handoff note" for async team workflows.
- Add strict assumption expiry for rapidly changing codebases.
