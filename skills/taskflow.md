---
name: taskflow
description: "Use when work should span one or more background agent tasks but still behave like one job with a single owner context. TaskFlow manages durable multi-step workflows with state persistence, child task tracking, and resume/cancel semantics."
---

# TaskFlow

Use TaskFlow when a job needs to outlive one prompt or one agent run, but you still want one owner session, one return context, and one place to inspect or resume the work.

## When to Use

- Multi-step background work with one owner
- Work that waits on dispatched agent tasks
- Jobs that may need to emit one clear update back to the owner
- Jobs that need small persisted state between steps
- Work that must survive restarts cleanly

## What TaskFlow Owns

- Flow identity and owner session
- Current step, persisted state, and wait conditions
- Linked child tasks and their parent flow ID
- Finish, fail, cancel, waiting, and blocked states
- Revision tracking for conflict-safe mutations

It does **not** own branching or business logic. Put that in the calling code or orchestration layer.

## Managed Flow Lifecycle

1. **Create** the flow with an ID, goal, initial step, and state
2. **Run tasks** linked to the flow (dispatched via agent tool)
3. **Set waiting** when blocked on external input (human reply, API callback)
4. **Resume** when the wait condition is met
5. **Finish** or **fail** the flow

## Design Constraints

- Use **managed** flows when your code owns the orchestration
- One-task mirrored flows are created by runtime for simple dispatched agent work
- Treat persisted state as the single state bag (no separate output API)
- Every mutating step is revision-checked -- carry forward the latest revision after each mutation
- Link child tasks to the flow rather than creating standalone tasks

## Example Pattern

```
1. Create flow: "triage inbox"
   - step: "classify"
   - state: { businessThreads: [], personalItems: [], summary: [] }

2. Dispatch classifier agent (linked to flow)
   - Agent classifies messages into categories

3. Set waiting: "await_business_reply"
   - wait condition: reply on specific thread
   - updated state with classified items

4. Resume on reply received
   - step: "finalize"

5. Finish flow with final state
```

## Operational Patterns

- Store only the minimum state needed to resume
- Put human-readable wait reasons in a summary field
- Use structured wait metadata for programmatic conditions
- When the orchestrator needs a compact health view, inspect child task status
- Use cancel when active linked child tasks should also stop

## Keep Conditionals Above the Runtime

Use the flow runtime for state and task linkage. Keep decisions in the orchestration layer:

- `business` -> notify team channel via Telegram and wait
- `personal` -> notify the owner now
- `later` -> append to an end-of-day summary bucket

## Integration with Prometheus

- Dispatch child tasks via the **agent tool**
- Use **bash** for any shell operations needed during flow steps
- Use **file_write** / **file_read** for persisting flow state to disk if needed
- Monitor flow health via **SENTINEL** telemetry
- Log flow transitions for debugging via session logs
