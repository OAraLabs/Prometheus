---
name: plan
description: Structure an implementation plan before writing code — requirements, approach, steps, risks.
---

# plan

Think through an implementation before touching any code.

## Structure

### 1. Understand the requirement
- Restate what needs to be built in one sentence.
- Identify what success looks like (acceptance criteria).
- Clarify ambiguities before proceeding.

### 2. Read existing code
- Find the relevant files and read them.
- Understand existing patterns, conventions, and constraints.
- Do not design around imagined code — read the actual code first.

### 3. Design the approach
- Choose the simplest approach that satisfies the requirement.
- Identify what new files or classes are needed.
- Identify what existing files will change.
- Note any integration points with existing modules.

### 4. List implementation steps
- Break the work into discrete, ordered steps.
- Each step should be independently verifiable.
- Flag any steps that are risky or uncertain.

### 5. Identify risks
- What could go wrong?
- What dependencies or external systems are involved?
- What needs to be tested?

## Principles

- Do the simplest thing that works. Avoid speculative abstractions.
- Design for what is actually needed, not hypothetical future requirements.
- A plan that fits on one page is better than a plan that doesn't.
