# The ACP approval channel: lower trust, recorded, refused where it must be

Design, 2026-11-25. The approval rule for ACP-borne approvals, decided
2026-11-25. This note says what an ACP answer is, where it is refused, and
what it can never do. It is the permission-side half of the ACP work; the
protocol core (framing, sessions) is `docs/design/acp-core.md` (PR #957).

## 0. In brief

- An approval that arrives over ACP is a **lower-trust channel** than a
  paired Beacon device. An ACP client may approve **ordinary** actions.
- Anything **critical** (`permissions/critical.py`: desktop prompts, hold
  lifts, cut arguments, long descriptions, the owner's `push.critical_*`
  lists) can only be approved from a paired Beacon device or the owner's
  chat — **never over ACP**.
- Every ACP answer is recorded with the **client that gave it**: a new
  `Approver` kind, `acp`, whose id is the paired client's own credential id.
  The shim never answers with the daemon's API token.
- The refusal lives in `ApprovalQueue.approve` — the ONE queue every surface
  reaches (REST, chat `/approve`, `/approve all`, the desktop channel's
  fan-in) — so no route can forget it.

## 1. Why the queue, not the routes

The surfaces already disagree about who may approve what: REST refuses a
scoped device, the desktop door refuses the API token, hold lifts require a
person, `/approve all` skips desktop/worker/hold/private requests. Adding
"and never a critical one over ACP" to each route would be a fifth copy of a
rule that must hold on every path, including paths not written yet. The queue
is the choke point: `approve()` already refuses a chat answer that is not the
owner's, and the ACP refusal sits beside it, before any mutation. A refused
request stays **pending** — a person can still answer it; nothing times out
early because a lower-trust channel tried.

Denial is different: saying **no** narrows what may run, so an ACP client may
deny anything, critical or not. The asymmetry is the point.

## 2. The credential

`Approver(kind="acp", id=<paired client id>)`, built by
`approver.acp_client(client_id, name)`. The id is the client's OWN revocable
credential from the device-pairing machinery (each ACP client pairs once,
approved in Beacon; the shim never uses the daemon's API token). The audit
row's `user_id` is `acp:<client id>`, so "who approved this?" has an answer
even when the answer came from an editor.

An unnamed client is recorded as `acp:unknown` — honestly, like every other
surface's unknown sender.

`acp` is NOT a chat kind: the chat-owner backstop (`not_the_owner`) does not
apply to it. Its own rule is the critical refusal above. It is also not a
person's credential for the desktop door (`computer/door.py` falls through
to "not a person's credential"), so a desktop prompt is doubly refused over
ACP: critical by construction, and the door would refuse the kind anyway.

## 3. What this does NOT do

- No ACP transport, shim or pairing route is built here. This is the rule
  every future ACP surface inherits; the pairing machinery (each client
  approved once in Beacon, its own revocable credential) is the agent-side
  work, and an unapproved client cannot connect at all.
- Outside agents that Prometheus DRIVES over ACP: their permission requests
  go through the same gate as Prometheus's own tool calls (the gate is
  unchanged here), anything the gate escalates goes to Beacon, and the
  outside agent runs in its strictest ask-every-time mode. That wiring is
  the client-side work; this note only fixes what an ACP-borne ANSWER means.
- Nothing changes for any existing surface: no current code constructs an
  `acp` approver, so every existing answer takes exactly the path it took
  before. The full suite is the proof.

## 4. Tests

`tests/test_acp_approval_channel.py`: ordinary approval over ACP lands and
records `acp:<id>`; a desktop prompt and an owner-list match are refused and
stay pending; the same critical action is still approvable by a device;
ACP may deny a critical action; an unnamed client records `acp:unknown`.