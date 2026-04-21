---
name: debug
description: Systematic debugging workflow — reproduce, isolate, diagnose, fix, verify.
---

# debug

Diagnose and fix a bug methodically.

## Workflow

### 1. Reproduce
- Get the exact error message, stack trace, and steps to reproduce.
- Run the failing code or test to confirm you can reproduce it.

### 2. Isolate
- Narrow the failure to the smallest reproducible unit (single function, test, or input).
- Use binary search through recent commits (`git bisect`) if the regression is unclear.

### 3. Diagnose
- Read the relevant source files — do not guess before reading.
- Trace execution: add `print` / logging temporarily if needed.
- Check: off-by-one errors, None/null handling, async ordering, import errors, type mismatches.

### 4. Fix
- Make the minimal change that addresses the root cause.
- Do not refactor surrounding code unless directly relevant.
- Do not add error handling for impossible scenarios.

### 5. Verify
- Run the original failing test/command — confirm it passes.
- Run the broader test suite to check for regressions.
- Remove any temporary debug output.

## Principles

- Read before editing. Never modify code you haven't read.
- One change at a time. Don't change multiple things and hope it works.
- Root cause only. Fix the bug, not the symptom.
