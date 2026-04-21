---
name: stuck
description: "Help recover when you're stuck, going in circles, or not making progress. Use when repeated attempts fail, when you keep getting the same error, or when you've lost track of what you were doing."
---
<!-- Provenance: whieber1/Prometheus | src/reference_data/subsystems/skills.json (stuck.ts reference) | MIT -->

# Stuck

Break out of unproductive loops and recover forward momentum.

## Triggers
- "I'm stuck"
- "this isn't working"
- "I keep getting the same error"
- "help me debug this"
- Agent detects repeated failed attempts

## When to Use
- The same approach has failed 2+ times
- You're going in circles (try → fail → revert → try again)
- Error messages are unclear or misleading
- You've lost context on what you were trying to do

## Steps

1. **Stop and assess**: Don't retry the same thing again
2. **State the problem clearly**: What are you trying to achieve? What's actually happening?
3. **Review what's been tried**: List the approaches attempted and why each failed
4. **Check assumptions**:
   - Is the file/path/function you're editing the right one?
   - Is the error message pointing to the actual problem?
   - Are there prerequisites you're missing (deps, config, env vars)?
   - Is the tool/command you're using the right one for this?
5. **Try a different angle**:
   - Read the error message literally — what is it actually saying?
   - Search for the exact error string
   - Try a minimal reproduction
   - Read the relevant source code instead of guessing
   - Ask the user for context you might be missing
6. **If still stuck**: Escalate to the user with a clear summary of what was tried

## Anti-patterns to avoid
- Retrying the identical command hoping for a different result
- Making the fix more complex each iteration
- Ignoring error messages and guessing at solutions
- Changing multiple things at once (can't tell what helped)

## Output
- Clear diagnosis of why you were stuck
- New approach to try
- Or: escalation to the user with context
