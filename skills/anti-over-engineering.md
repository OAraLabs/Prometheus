---
name: anti-over-engineering
description: Use during any code modification to prevent scope creep, premature abstraction, and unnecessary complexity. Enforces strict rules against adding features not requested, speculative error handling, and gratuitous refactoring.
version: 1.0.0
author: RepoWise
license: MIT
---
<!-- Provenance: repowise-dev/claude-code-prompts | complete-prompts/system-prompt.md (Code Style section) | MIT -->

# Anti-Over-Engineering

## The Core Principle

Every line of code you add must be justified by the current request. If you cannot point to a specific user requirement that demands the code, do not write it.

## Rules

### 1. Limit Changes to What Was Requested

A bug fix does not warrant adjacent refactoring, style cleanup, or feature additions. Stay within the scope of the task.

**Test:** For every changed line, can you explain which part of the user's request requires it? If not, revert it.

### 2. No Speculative Error Handling

Do not insert defensive error handling, fallback logic, or input validation for conditions that cannot arise in the current code path. Trust the internal guarantees of the codebase.

**Bad:**
```python
# User asked to add a logger call
def process(items):
    if items is None:  # <-- not requested, items is never None here
        return []
    if not isinstance(items, list):  # <-- not requested, type is guaranteed
        raise TypeError("Expected list")
    for item in items:
        logger.info(f"Processing {item}")  # <-- this is what was requested
```

**Good:**
```python
def process(items):
    for item in items:
        logger.info(f"Processing {item}")
```

### 3. No Premature Abstraction

Do not extract helpers, utility functions, or shared abstractions for logic that appears only once. Three nearly identical lines are preferable to a premature generalization.

**Bad:**
```python
# User asked to add a retry to one API call
def _retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    """Generic retry helper."""  # <-- over-engineered for a single use
    ...

def fetch_data():
    return _retry_with_backoff(lambda: api.get("/data"))
```

**Good:**
```python
def fetch_data():
    for attempt in range(3):
        try:
            return api.get("/data")
        except APIError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
```

### 4. No Backward-Compatibility Scaffolding

Do not add: renaming variables to underscore-prefixed unused versions, re-exporting removed types, or inserting "this was removed" annotations. If something is removed, remove it cleanly.

### 5. Comments Only for Non-Obvious Reasoning

Only add code comments when the reasoning behind a decision is genuinely non-obvious -- hidden constraints, subtle invariants, non-intuitive workarounds. Never comment to narrate what the code does.

**Bad:**
```python
# Get the user from the database
user = db.get_user(user_id)
# Check if user exists
if user is None:
    return None
```

**Good:**
```python
user = db.get_user(user_id)
if user is None:
    return None
```

### 6. No Unrequested Documentation

Do not add docstrings, comments, or type annotations to code you did not modify. If the user did not ask for documentation, do not add it.

## Self-Check Before Submitting Changes

For every change you are about to make, ask:

1. **Was this requested?** If no, do not include it.
2. **Does this handle a condition that can actually occur?** If no, remove the guard.
3. **Is this abstraction used more than once?** If no, inline it.
4. **Am I adding compatibility shims for removed code?** If yes, remove them.
5. **Am I commenting what the code does (vs why)?** If yes, remove the comment.
6. **Am I touching files outside the scope of the request?** If yes, stop.

## The Bottom Line

The best code change is the smallest one that fully satisfies the request. Nothing more.
