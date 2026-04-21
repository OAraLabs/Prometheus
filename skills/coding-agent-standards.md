---
name: coding-agent-standards
description: Use when writing or modifying code to ensure changes are predictable, reviewable, and safe. Covers implementation discipline, small focused changes, test expectations, and delivery reporting format.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | skills/coding-agent-standards/SKILL.md | MIT -->

# Coding Agent Standards

## Purpose

Use this skill to keep code changes predictable, reviewable, and safe.

## Default Behavior

1. Clarify the requested outcome and constraints before editing.
2. Prefer small, focused changes over broad refactors.
3. Preserve existing behavior unless a behavior change is requested.
4. Keep naming, structure, and style consistent with nearby code.
5. Add or update tests when behavior changes.

## Implementation Checklist

- Confirm the smallest viable file set to edit.
- Handle edge cases and explicit failure paths.
- Avoid speculative abstractions and dead code.
- Add concise comments only where intent is not obvious.
- Run available validation steps before finalizing.

## Delivery Format

When reporting completion:
- **What changed:** Files and intent.
- **Why it changed:** Problem solved or risk reduced.
- **How it was verified:** Tests, checks, or manual validation.
- **Remaining risks:** Assumptions or follow-ups.
