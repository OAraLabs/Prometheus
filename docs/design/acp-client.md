# The ACP client side: driving outside agents through the gate

Design, 2026-11-25. Prometheus as an ACP CLIENT. This note
says what the driver does, what it refuses, and where every permission
request goes. The protocol core is `docs/design/acp-core.md` (PR #957);
the approval rule for ACP-borne ANSWERS is `docs/design/acp-approval-channel.md`.

## 0. In brief

- Prometheus launches an outside agent (Claude Code first) as a subprocess
  speaking ACP v2 over stdio, sends prompts, streams the agent's updates.
- The agent's `session/request_permission` requests go through **the same
  gate** as Prometheus's own tool calls — never answered by the driver
  alone.
- **Exactly what Prometheus's own calls get (D13 as written)**: a gate
  ALLOW (a grant covers it, an allowlisted read, …) answers the agent's
  `allow_once` option directly, no queue — audited like any gate-allowed
  call, with a row naming the outside agent. A gate APPROVE goes to the
  owner's approval queue; anything the gate escalates goes to Beacon
  through the queue.
- The registry of drivable agents ships **EMPTY**; each entry is an explicit
  operator grant in `acp.clients`. Nothing outside the registry launches.
- **Hard-disabled for machine/benchmark sessions** (`bakeoff:`, `eval:`,
  `gym:`, `smoke:`, `coding:`, `system`): a driven agent has real side
  effects, never fixture material.
- OFF by default: nothing here is imported unless `acp.enabled` is true.

## 1. Layers (`src/prometheus/acp/client/`)

| Module | What it owns |
|---|---|
| `connection.py` | The client half of the NDJSON connection: requests out (ids matched), notifications to an observer, the agent's own requests answered through a handler. No permission handler → every ask answered `cancelled` (fail closed). |
| `registry.py` | `acp.clients` → named argv entries. A malformed entry never half-registers. A command string is shlex-split, never shell-executed. |
| `driver.py` | One driven session: initialize (v2 only; any other answer closes the connection), session/new, prompt, cancel. Permission routing below. |

## 2. Permission routing (the approval rule, applied)

An ask carries a subject (a command line, or a tool call) and the agent's
own options. The driver:

1. evaluates the subject through the gate (`evaluate`, command subjects map
   to `bash`; an unknown tool name is never assumed read-only);
2. a DENY-tier decision → the agent's `reject_once` option (or `cancelled`
   when none was offered). The agent MUST NOT proceed;
3. an ALLOW decision → the agent's `allow_once` option **directly, no
   queue** — the same call Prometheus itself would run without a prompt.
   The gate's `evaluate` writes its ALLOW audit row; the driver adds one
   naming the outside agent the allow answered (`acp-agent:<name>`), so
   the record says who ran it, not just what was allowed. The audit write
   is best-effort: a logger that cannot write never refuses a call the
   gate allowed;
4. an APPROVE decision → the **owner's approval queue**, approve-once
   (`once_only=True`, like a spawned worker's request — a driven agent's
   consent is per-call, never a lasting grant). The owner approves → the
   agent's `allow_once` option. **Never `allow_always`**, even when the
   agent offers it;
5. no gate, no `allow_once` option offered, no queue for an APPROVE, a
   timeout, a denial, or any failure along the path → refusal. Fail closed
   at every step.

The driver answers with a selection of the agent's OWN options (the
schema's contract), never an invented option id.

## 3. What this does NOT do

- No route or tool exposes the driver yet: wiring it to a surface (a
  `drive` tool, a coordinator hook) is the next slice, and each gets the
  gate's own review. This PR is the client machinery and its refusals.
- The agent-side endpoint (the `prometheus acp` shim, per-client pairing)
  is separate work; an unapproved client cannot connect there at all.
- v1 agents: initialize answers anything but v2 with a closed connection.

## 4. Tests

The far side is REAL code: the TCK's conforming v2 fixture agent
(Apache-2.0) is launched over stdio for the connection/driver round-trip
tests. Permission routing uses the real SecurityGate and ApprovalQueue:
a grant-covered ask is answered `allow_once` with NO queue entry and an
audit row naming the agent; an uncovered ask reaches the queue
approve-once; a DENY (a `denied_commands` entry) refuses even when a grant
would match it; an approval selects `allow_once` (never `allow_always`);
no gate refuses, and an APPROVE with no queue refuses while a gate ALLOW
without a queue still answers. The registry's refusals and the
machine-session hard disable are pinned, and the shipped template keeps
the registry empty and ACP off.