# ACP core: the protocol at the edge (experimental, off by default)

Design, 2026-11-25. This note says what
the shared ACP core is, what it pins, what it refuses, and what it deliberately
does not do yet. The approval rule for ACP-borne approvals is UNDECIDED; nothing
here touches the permission model, and the client/agent endpoint wiring is blocked
on that decision.

## 0. In brief

- ACP v2 only, pinned to ONE schema tag: `schema-v2.0.0-beta.4`. Moving to a new
  tag or to stable is one PR: regenerate the types, bump the pin, re-run the TCK.
- The types are GENERATED from the pinned schema, not hand-written and not the
  PyPI SDK (its released v2 module lags the pin — see §2).
- A v1-only peer is refused at initialize with `-32022` and keeps today's
  behavior: no ACP session exists for it, nothing in the daemon changed.
- Everything ships OFF. `acp.enabled` defaults to false; with it false, no
  module in `prometheus/acp/` is imported by any route, and the test suite,
  golden-trace replay and overhead are unchanged.
- ACP is spoken at the EDGE. The internal session model is not ACP; internal
  subagents stay in-process.

## 1. Layers (`src/prometheus/acp/`)

| Module | What it owns |
|---|---|
| `types.py` | Pydantic models generated from `schema/schema.json` (the pin). Never hand-edit. |
| `framing.py` | JSON-RPC 2.0 over newline-delimited JSON. v2 has NO batching: a JSON array is one Invalid Request. Request ids are string or integer, never null (null only on an error response whose request id could not be read). |
| `negotiation.py` | The initialize handshake. Answer the highest supported version ≤ the offer; we support only 2, so any other offer is `-32022`. |
| `sessions.py` | The ACP↔Prometheus session map: new/list/resume/delete, prompt INSERTION with durable idempotency, replay ending with the `replay_complete` marker. |
| `_vendor/` | Apache-2.0 runtime helpers the generated types import (upstream deserialization semantics), licence text included. |

What the core does NOT own: running turns, tool gating, permissions, transports.
Wiring an inserted prompt to an agent turn is later work and needs the approval
rule decided first.

## 2. Why generated types, not the SDK

The doc's rule: depend on the Python SDK's experimental v2 module IF it matches
the pinned tag; otherwise generate. Measured on 2026-11-25:

- PyPI `agent-client-protocol==1.0.0` (the latest release): its
  `acp/experimental/v2/schema.py` header says
  `Schema ref: refs/tags/schema-v2.0.0-alpha.5`. 87 of the pin's 214 `$defs`
  are absent from it (including `MessageId`, `IdempotencyKey`, `ReplayComplete`,
  `StateUpdate` — the prompt-lifecycle vocabulary itself). It does not match.
- The SDK repo's `main` generates from the UNSTABLE superset schema: 76 extra
  defs, 11 extra agent methods, 22 structurally changed defs versus the pin.
  Also not the pin.

So the types are generated with the upstream's own generator
(`scripts/gen_schema.py`, datamodel-code-generator) from the tag's
`schema/v2/schema.json`, exactly as the SDK builds its own — same base classes,
same deserialization semantics, vendored with their Apache-2.0 licence. The
generated file's header carries the tag and the schema's sha256; a test pins
both, and a second test walks every object def in the pin and checks the
generated model's required fields against it.

Regeneration (scratch, not part of the daemon):

1. Fetch `schema/v2/schema.json` and `meta.json` from the tag into a checkout
   of the SDK repo's `scripts/` + `schema/v2/`.
2. Run the generator (`render_schema` with the v2 semantics).
3. Rewrite the three `acp.*` imports to `prometheus.acp._vendor.*`.
4. Replace `src/prometheus/acp/types.py`, bump `SCHEMA_TAG`, re-run
   `tests/test_acp_core.py` and the TCK capture.

## 3. Sessions

An ACP session IS a Prometheus session under the `acp:` surface
(`acp:<uuid>`), so rows, titles, tombstones and workspaces live in the same LCM
store as every other conversation. The map is durable by construction: a daemon
restart needs no mapping table.

- `session/new` records the cwd in `session_workspaces` (so `session/list` can
  report a real cwd after a restart — `SessionInfo` requires it).
- `session/prompt` validates against the generated `PromptRequest` and INSERTS
  the user row through the store's own insert (ids, redaction, FTS). The
  response acknowledges INSERTION (`PromptResponse.messageId`), not completion —
  completion is a state update, per the schema's prompt lifecycle.
- Idempotency: `(sessionId, idempotencyKey)` claims live in the store's
  `message_client_ids` side table under `acp-key:<key>` — the same durable
  mechanism the REST send path uses. Lookup, insert and claim record run in
  ONE store transaction (`LCMConversationStore.transaction()`: BEGIN
  IMMEDIATE plus per-thread commit suppression). The connection is shared
  across threads and SQLite has ONE transaction per connection, so every
  write METHOD also takes the store's re-entrant write lock for its whole
  run: a bare writer on another thread waits for the unit to end instead of
  committing into it (which would end the unit early and make its rollback
  useless). Two concurrent prompts with the same key cannot both pass the
  lookup, and a crash mid-unit rolls the insert back with it. A retry after
  a restart returns the first insertion's message id and inserts nothing.
- `session/resume` replays history keyed by DURABLE rowid (never positional
  ids), ordered by prompt position, ending with the `replay_complete`
  SessionUpdate. The caller holds live updates for the session until the marker
  is out (schema `ReplayComplete`). Only the `start` cursor exists in the pin;
  an unsupported cursor is `-32602`, never a substituted cursor.
- `session/delete` tombstones (the store's durable forget), never deletes rows.

### Refusals (fail closed)

- Foreign session ids (anything not `acp:`) → `-32002` session not found.
- `idempotencyKey` longer than **120** chars → `-32602`, refused BY NAME. The
  schema allows 255; the stored claim is `acp-key:` + key in the daemon's
  128-char client-id column, so the KEY must fit 128 − 8 = 120. Checking the
  key against 128 instead would accept 121–128-char keys whose claim the
  store then silently drops — idempotency lost without a word. Truncating
  would merge distinct submissions, so the edge refuses by name. Widening
  the storage (263 chars, or a dedicated column) is a separate migration PR,
  before the agent endpoint; a test pins the 120/121/128 boundary.
- `messageId` is the durable SQLite rowid. `lcm_messages` has no INTEGER
  PRIMARY KEY, so rowid stability across the purge path's VACUUM is a
  practical property, not a promised one; a test runs the REAL purge and
  asserts surviving sessions' rowids are unchanged, so a SQLite change
  cannot silently break ACP message ids.
- Malformed params → `-32602` with the validation error; never a partial insert.

## 4. Fixtures

Fixtures come from the other side's real code. Two sources:

1. The pinned schema itself — required-field sets and const discriminators are
   read from `schema/schema.json` at test time.
2. The TCK's conforming v2 fixture agent (acp-tck, Apache-2.0). The capture
   driver (`tests/acp_fixtures/capture.py`) runs the REAL fixture over stdio
   and records its responses verbatim into `tck_exchanges.json`. The tests
   parse every captured line with our framing, validate every result and
   notification against the generated types, and pin the lifecycle facts the
   capture shows (retry returns the same messageId; replay ends with the
   marker before the response). The only edit to the capture is substituting
   the local checkout path with a neutral one (public-repo rule).

The full TCK v2 suite is the pass/fail gate for the endpoint work, when there is
an agent endpoint to drive. This core has no endpoint yet; the capture is the fixture
half of that gate, in place early.

## 5. Gates

- OFF by default: no route imports `prometheus.acp` unless `acp.enabled` is
  true; the config key ships false and labeled experimental.
- Unchanged test suite: the full suite runs with the package present and off.
- Golden-trace replay: the parity harness replays recorded turns byte-for-byte;
  nothing in this package sits on the request path with ACP off.
- Overhead with ACP off: the package is not imported on any hot path; the
  benchmark is the parity harness's fixed-overhead measurement plus an import
  probe (a daemon boot with ACP off must not import `prometheus.acp.types`).

## 6. Open, and deliberately not built

- The endpoints (Prometheus as ACP client, and as ACP agent):
  blocked on the approval rule. An ACP-borne approval is a lower-trust channel;
  until that rule is decided, no ACP peer can approve anything.
- Transports: stdio framing is built; a network transport is a later PR with
  two reviews (networking).
- v1: skipped. A v1-only peer gets today's behavior.