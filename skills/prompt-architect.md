---
name: prompt-architect
description: Use when designing new prompts, refining weak prompts, or establishing reusable prompting patterns for agent workflows. Covers prompt build sequence, quality checks, and iteration strategy.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | skills/prompt-architect/SKILL.md | MIT -->

# Prompt Architect

## Purpose

Use this skill to turn vague intent into reliable prompt instructions.

## Build Sequence

1. **Define objective, audience, and success criteria.** What must the prompt accomplish? Who or what will execute it? How do you measure success?
2. **Capture hard constraints.** Format, tools available, safety rules, style requirements.
3. **Specify reasoning boundaries and output structure.** What the agent should reason about, what it should not speculate on, and how results must be formatted.
4. **Add verification instructions and failure handling.** How should the agent confirm its work? What should it do when stuck?
5. **Iterate using concrete examples and observed errors.** Test the prompt, observe failures, tighten the wording.

## Prompt Quality Checks

- Is the task scope explicit and bounded?
- Are required outputs unambiguous?
- Are constraints actionable (not aspirational)?
- Is there a clear fallback when information is missing?
- Can you test whether the prompt was followed correctly?

## Common Anti-Patterns

- **Vague goals:** "Be helpful" instead of "Return a code patch that fixes the failing test."
- **Aspirational constraints:** "Try to be thorough" instead of "Run the test suite and report pass/fail counts."
- **Missing output format:** Letting the agent decide how to structure results leads to inconsistency.
- **No failure path:** If the prompt does not say what to do when stuck, the agent will guess or hallucinate.
- **Over-specification:** Prescribing every micro-step when the agent could handle sequencing itself.

## Refinement Loop

1. Run the prompt against a representative task.
2. Compare actual output to expected output.
3. Identify the gap: was the instruction missing, ambiguous, or contradicted?
4. Tighten the specific clause that caused the failure.
5. Re-run and verify the fix did not break other cases.
