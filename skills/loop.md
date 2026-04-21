---
name: loop
description: "Run a prompt or skill on a recurring interval (e.g., every 5 minutes). Use when you need to poll for status, monitor a deploy, check build results, or run any recurring check. Default interval is 10 minutes."
---
<!-- Provenance: whieber1/Prometheus | src/reference_data/subsystems/skills.json (loop.ts reference) | MIT -->

# Loop

Run a prompt or skill repeatedly on a timer.

## Triggers
- "check every 5 minutes"
- "keep monitoring the deploy"
- "poll for status"
- "run this on repeat"

## When to Use
- Monitoring a deployment, build, or long-running process
- Polling an API or service for status changes
- Running periodic health checks during a session
- Watching for a condition to become true

## Steps

1. **Parse the interval**: Extract the repeat interval from the user's request (default: 10m)
   - Supported formats: `5m`, `30s`, `1h`, `5 minutes`
2. **Parse the action**: What to run each iteration
   - A shell command (`git status`, `curl ...`)
   - A skill invocation (`/healthcheck`)
   - A free-text prompt ("check if the tests pass")
3. **Create a cron job or background loop**:
   - Use `cron_create` for persistent schedules
   - Or use a background task with a sleep loop for session-scoped monitoring
4. **Report on each iteration**: Show the result, highlight changes from previous run
5. **Stop condition**: Stop when the user says so, or when a defined condition is met

## Example

```
User: "Check if the CI build passes every 2 minutes"

1. Every 2 minutes, run: gh run list --limit 1 --json status
2. Report: "Build still running..." / "Build passed!" / "Build failed!"
3. Stop when status != "in_progress"
```

## Output
- Iteration number and timestamp
- Result of each check
- Delta from previous check (if applicable)
- Final status when loop ends
