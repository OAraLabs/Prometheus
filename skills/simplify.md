---
name: simplify
description: "Review changed code for reuse opportunities, quality issues, and efficiency improvements, then fix any issues found. Use after writing or modifying code to ensure it's clean, DRY, and follows project conventions."
---
<!-- Provenance: whieber1/Prometheus | src/reference_data/subsystems/skills.json (simplify.ts reference) | MIT -->

# Simplify

Review recently changed code for quality, then fix issues.

## Triggers
- "simplify this code"
- "clean up what I just wrote"
- "review for quality"
- "make this DRY"

## When to Use
- After completing a feature implementation
- When code feels overly complex or repetitive
- Before committing, as a self-review step
- When asked to improve code quality

## Steps

1. **Identify changed files**: Use `git diff` to find recently modified files
2. **Review for issues**:
   - **Reuse**: Are there duplicated patterns that could be extracted?
   - **Complexity**: Can any logic be simplified without losing clarity?
   - **Naming**: Are variables and functions named clearly?
   - **Dead code**: Is there unused code that should be removed?
   - **Consistency**: Does the code follow existing project patterns?
3. **Fix issues found**: Apply improvements directly
   - Prefer small, targeted fixes over sweeping refactors
   - Don't change code that works fine just for style
   - Don't add abstractions for single-use patterns
4. **Verify**: Ensure changes don't break functionality

## Principles
- Less code is usually better code
- Don't add complexity to remove complexity
- If the fix is bigger than the problem, skip it
- Three similar lines are better than a premature abstraction

## Output
- List of issues found (with file:line references)
- Fixes applied
- Any issues deferred (too risky or out of scope)
