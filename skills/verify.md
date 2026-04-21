---
name: verify
description: "Verify that code changes work correctly by running tests, type checks, linters, and build steps. Use after making changes and before committing to ensure nothing is broken."
---
<!-- Provenance: whieber1/Prometheus | src/reference_data/subsystems/skills.json (verify.ts reference) | MIT -->

# Verify

Run the full verification suite to confirm code changes are correct.

## Triggers
- "verify this works"
- "run the checks"
- "make sure nothing is broken"
- "test my changes"

## When to Use
- After implementing a feature or fix
- Before creating a commit
- When unsure if changes introduced regressions
- After a refactor

## Steps

1. **Detect project type** and available verification commands:
   - Check for `package.json` (npm/pnpm: `test`, `lint`, `typecheck`, `build`)
   - Check for `pyproject.toml` / `setup.py` (pytest, mypy, ruff)
   - Check for `Cargo.toml` (cargo test, cargo clippy, cargo fmt --check)
   - Check for `Makefile` (make test, make lint)
   - Check for CI config (`.github/workflows/`) to match what CI runs

2. **Run checks in order** (fail fast):
   1. **Format check**: Is the code formatted correctly?
   2. **Lint**: Are there any lint warnings or errors?
   3. **Type check**: Do types pass? (TypeScript, mypy, etc.)
   4. **Tests**: Do all tests pass?
   5. **Build**: Does the project build successfully?

3. **Report results**:
   - If all pass: "All checks pass. Ready to commit."
   - If any fail: Show the failure, suggest a fix, offer to fix it

4. **Fix failures** if asked:
   - Fix the issue
   - Re-run only the failed check to confirm
   - Then re-run the full suite

## Output
- Pass/fail status for each check
- Error details for any failures
- Suggested fixes for failures
