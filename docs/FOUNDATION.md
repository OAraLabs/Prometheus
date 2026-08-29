# Prometheus Foundation Spec, Version 3

**Status:** Adopted, pre-Beacon-release
**Scope:** The three architectural decisions that become irreversible once Beacon ships and third parties accumulate data.
**Rule:** Changes to anything in this document are breaking changes, not refactors. They require a version bump and a migration path.

---

## Why this document exists

Right now Prometheus has one user and every format is convention. Convention can be changed on a whim. The moment other people run it, accumulated state exists in whatever shape the code happened to write, and changing that shape breaks them silently.

Three things are genuinely hard to retrofit:

1. **Data format.** Users accumulate months of vault and telemetry. Reshaping it later means migrating everyone or breaking everyone.
2. **Extension surface.** If packs import internals, every refactor breaks third-party code and the internals become public API by accident.
3. **Identity.** Adding identity to a deployed fleet means every existing node is unidentified. There is no way to backfill it.

Everything else, pricing, packs, hosting, gateways, can be decided later. These three cannot.

---

## Part 1: Data Format

### 1.1 Versioning

Every vault carries a version marker. Prometheus writes it on adoption or creation and reads it on startup. The path below is the shipped default; the operator points `vault.root` (or `PROMETHEUS_VAULT`) at the real vault.

```
~/brain-vault/.prometheus-vault
```

```yaml
vault_format: 1
created: 2026-08-28T00:00:00Z
created_by: prometheus 0.1.0
instance_id: <see Part 3>
enrolled_nodes: []
```

**Behaviour on mismatch:**

| Condition | Action |
|---|---|
| Vault directory absent | Not an error. Long-standing stance: the tools report absence when called; startup does not |
| Marker absent | Legacy vault. Governed by `vault.format_check` (`off` \| `warn` \| `refuse`, default `warn` this release, `refuse` once adoption is routine). Adoption is the explicit `prometheus vault adopt` — never silent |
| `vault_format` equals current | Proceed |
| `vault_format` lower than current | Run migration, log every step, refuse on failure (no lower formats exist yet; today a lower value refuses, naming both) |
| `vault_format` higher than current | **Refuse to start**, regardless of `format_check`. Newer vault, older binary. Loud error naming both versions |

The last row matters. A newer vault read by older code is the silent-misread case, which is the project's dominant bug class. Refuse rather than guess. It is deliberately not configurable: `format_check` governs only the legacy states, because a parsed marker with the wrong format is always a deliberate state (a newer Prometheus wrote it, or a rollback ran), never a vault that merely predates markers.

The `warn` default for one release is itself a spec decision: the fleet that exists on the day this ships is entirely markerless, the daemon runs under systemd with a bounded restart budget, and "offer one-time migration" cannot be interactive there. Warn names the adopt command; refuse arrives once adoption is routine.

**Writer discipline:** the marker is the single machine-written file at the vault root, and its writer is `config/vault_marker.py` — never the vault tools, which are kept structurally read-only by a build-failing AST guard. Marker writes are atomic (tmp + rename) with stable key order, because the vault is a live git repo a human reads diffs of.

### 1.2 Vault layout

Zone ownership is already established and stays as is:

```
~/brain-vault/
  .prometheus-vault      version marker, machine-written
  BRAIN.md               router, human-editable
  raw/                   immutable, append only, never rewritten
  wiki/                  machine-owned, safe to delete and recompile
  notes/                 human-owned, machine never writes here
```

**Invariants, testable:**

- Nothing in `raw/` is ever modified after write. Rewrites are appends with a new timestamp.
- `wiki/` is fully derivable from `raw/`. Deleting it and recompiling loses nothing.
- No process writes to `notes/` except the human.
- Every file is plain markdown or plain SQLite. No proprietary container, no binary blob that only Prometheus reads.

That last invariant is deliberate. The switching cost you want comes from the *value* of accumulated state, not from making it hostage. A user who leaves should be able to keep their markdown. Anything else reads as a trap and will be called one in public.

### 1.3 Telemetry and traces

Telemetry schema is versioned separately from the vault, because it changes more often. This is the project's first real schema version — the existing mechanism is an additive column map with no version number and no mismatch detection — so the policy is stated in full:

```sql
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- schema_version, created_at, created_by
```

Same mismatch rules as 1.1. Older binary, newer schema, refuse. (The refusal case is real: rollbacks happen, and the additive migrator silently tolerates them today.)

**Every trace row carries `node_id`** (Part 3). A latency number without the hardware that produced it is not comparable across machines, and cross-machine comparison is the entire point of the fleet. The value is the node's public key. (`node_id` also names a summary-DAG concept in the LCM store — different database, and the collision is documented at both sites rather than renamed, because the telemetry name is the fleet-facing one.)

The telemetry DB path resolves through `get_config_dir()` like every other store. It did not always — the writer hardcoded the default while `reset-data` resolved properly, so a custom config dir split the DB from its eraser.

### 1.4 What is not covered

Session transcripts and LCM DAG nodes are in scope for version 4. They are stable enough not to block Beacon, but they are not yet specified.

The **SkillForge bundle format** is also deferred, for a different reason: it is genuinely still in testing and expected to change. The security rule around skills is pinned in 2.3, the file format is not. Do not let the pinned rule imply the format is settled. (As of v3 there is no bundle format at all: a skill is one flat markdown file with frontmatter. Anything richer is future.)

Also deferred, made explicit in v3: **relocating existing instance-owned state into the vault.** Part 3 defines what is instance-owned, and today much of it (`SOUL.md`, `memory.db`, the skill library) lives under `~/.prometheus`, not the vault. The ownership *rule* binds now, and new state must respect it; moving existing state is a migration that gets its own spec version, not a side effect of this one.

Noted here so none of it is rediscovered later.

---

## Part 2: Extension Surface

### 2.1 The rule

Third party code registers through a declared contract. It does not import Prometheus internals.

Consequence, stated plainly: **anything a pack can import becomes API you cannot change.** Keep the surface narrow on purpose.

### 2.2 What a pack is

A directory, dropped in a known location, discovered at startup.

```
~/.prometheus/packs/<pack-name>/
  pack.yaml           manifest
  skills/             skill bundles
  panels/             Beacon UI contributions, optional
                      (no tools/ — see 2.3a)
```

```yaml
# pack.yaml
name: example-pack
version: 1.0.0
pack_api: 1
requires:
  prometheus: ">=0.5.0"
provides:
  skills: [example-workflow]
  panels: [example-dashboard]
```

`pack_api: 1` is the contract version. It is not the Prometheus version and it moves independently.

**Behaviour on mismatch**, mirroring 1.1 deliberately — this is the same silent-misread
class, and the vault half was protected while this one was not:

| Condition | Action |
|---|---|
| `pack_api` absent | Refuse to load. Name the pack. A pack that does not declare a contract version has not agreed to one |
| `pack_api` equals current | Load |
| `pack_api` lower than current | Load if the older contract is still supported; log which version is in use. Refuse once support is dropped, naming both |
| `pack_api` higher than current | **Refuse to load.** Newer pack, older daemon. Loud error naming both versions |

The last row is the one that matters, for the same reason it mattered in 1.1: a pack
written against a contract the daemon does not have will fail somewhere unpredictable
if it is allowed to load. Refuse at the boundary, where the error can name the cause.

### 2.3 The contract

**Two** registration points, no more. Tools are deliberately not one of them — see 2.3a.

**Skills.** A pack ships skill bundles. No code required, which is the point: a domain pack can be mostly knowledge.

Two separable things here, and only one of them is pinned.

*Pinned, version 1:* a skill arriving from a pack is untrusted. It lands quarantined and requires an explicit promotion step before it can run, same posture as a vision-derived capture. This is a security rule and it does not change.

The quarantine is not new machinery: it is the existing `SkillDraftStore` lifecycle (`skills/drafts/` → REST accept/reject → the single machine-write path in `SkillCreator`). Reachability enforcement is directory membership — the loader globs exactly two directories, the skill tool re-reads disk on every call, and a draft matches neither glob. Pack skills land in that store and promote through that flow. A parallel mechanism would be a second thing to audit for the same property.

*Not pinned, deferred:* the SkillForge bundle format itself. See 1.4. Pack authors should expect the format to move before it is frozen.

**Panels.** Beacon exposes a panel contract: a route, a title, an icon, and a component. A panel talks to the daemon over the existing REST and WS surface only. It gets no privileged access.

*Scoped in v3:* the daemon side of the contract is **declaration and discovery** — the pack loader parses panel declarations and serves them read-only (`/api/packs`). Beacon *loading* third-party panel components is deferred until the sandboxing story is named in full: a panel component is third-party code in the operator's desktop app, and "no privileged handle" is necessary but not sufficient there. Discovery shipping first is deliberate (2.5: nothing has to consume the surface for declaring it to be worth it).

### 2.3a Tools are MCP, not packs

Version 1 of this spec had three registration points. Tools are removed from the pack
contract entirely. **A third party who wants to give Prometheus a tool writes an MCP
server.**

The reason is not tidiness. It is that the pack contract's hardest unsolved problem —
2.4's "enforced, not documented" — is only hard *for tools*, because a tool is
third-party code executing inside the daemon's process. Python has no reliable way to
stop in-process code importing whatever it likes; an import hook or AST scan is
fiddly and defeatable. A separate process **cannot** import our internals. The boundary
stops being policed and becomes structural.

What this buys, stated plainly:

- The isolation problem dissolves rather than being mitigated.
- A crashing or hanging tool cannot take the daemon with it.
- We inherit a specification we do not maintain, and an ecosystem that already exists.
- Prometheus already has an MCP client (`src/prometheus/mcp/`) that registers external
  tools as native `BaseTool` instances.

What it costs, also plainly:

- MCP has no concept of a skill. A domain pack that is mostly markdown knowledge does
  not map onto it at all — which is why skills stay in the pack contract.
- MCP has no concept of a Beacon panel. Same reason.
- An MCP tool runs out of process, so it cannot be as cheap as an in-process call.

**Prerequisites this creates.** Version 2 named one (`allowed_tools`). The survey
behind v3 found the true list is four, because the client that exists has never run
where the decision matters:

1. **Daemon wiring.** The MCP runtime is constructed only on the CLI path today; the
   daemon — Telegram, Beacon, cron — has never registered an MCP tool. Sanctioning
   MCP as *the* tool path means wiring it into the daemon's registry build first.
2. **Gate treatment.** An MCP call currently bypasses the SecurityGate in full: the
   adapter hardcodes read-only, its input model declares no fields so path extraction
   sees nothing, and there is no command — every call falls through to allow. The
   adapter must report honesty (`readOnlyHint` when the server declares it, else
   not-read-only), and a non-read-only MCP call requires confirmation. Blessing the
   path without this converts a dormant hole into the advertised way in.
3. **`allowed_tools`, enforced twice.** A per-server allowlist filtered at discovery
   *and* checked at `call_tool` — a registry-side filter alone is bypassable by a
   model that names an unregistered tool and gets it executed anyway.
4. **An advertisement decision.** Deferred tool loading filters by literal name
   membership in a static list, and the advertisement guard exempts dynamically
   registered tools — so today every MCP tool is silently unadvertised to local
   models, the exact `vault_search` failure class the guard exists to prevent.
   Dynamic tools must force an explicit advertise-or-defer decision.

Known limit, accepted: the client's HTTP/SSE transport is unimplemented (stdio only).
Local subprocess servers are the third-party story at v3; remote servers are not a
prerequisite, they are future work.

**Out of scope here.** Prometheus acting *as* an MCP server — so Claude Desktop or
Cursor can reach its tools — is a different direction and a separate decision. It does
not compete with the pack contract; exposure outward and extension inward are
orthogonal. Noted so the two are not conflated later.

### 2.4 What packs may not do

- Import from anywhere except the declared public module path
- Read or write the vault directly, they go through the vault tools
- Modify core configuration
- Declare a `pack_api` the daemon does not support (2.2)

These are enforced, not documented. A pack violating them fails loudly at load with the
pack name and the violation.

**How, specifically.** Version 1 left this as an assertion, which is the shape this
project has repeatedly found to be a note rather than a guard. Naming it:

Moving tools to MCP (2.3a) removes the hard case. What remains is **data and UI, not
executing code** — a skill is markdown, and a panel is a component that talks to the
daemon over the same REST and WS surface any client uses. So enforcement is a
**manifest and load-time check**, not runtime sandboxing:

1. **Skills.** No import surface at all. They are files. The check is that the bundle
   parses, declares its `pack_api`, and lands quarantined per 2.3 — a skill that has
   not been promoted is not reachable by the loop, and that is asserted by a test that
   drives the loop, not by inspecting a flag.
2. **Panels.** Loaded by Beacon, not the daemon. A panel gets the same bearer-gated
   REST and WS surface as any other client and no privileged handle. Enforcement is
   that no privileged handle is passed — verifiable by reading the one call site that
   constructs a panel, and pinned by a test asserting the argument list. (Deferred
   with panel loading itself — see 2.3.)
3. **Manifest integrity.** The loader refuses a pack whose `provides` block does not
   match what is on disk — a pack claiming a panel it does not ship, or shipping a
   skill it does not declare. Absence in the manifest is not permission.

If a future version reintroduces in-process third-party code, this section must name a
real mechanism before that lands. It must not inherit the current wording, which was
written for a contract that no longer has the hard case in it.

### 2.5 Timing

Declare the surface before Beacon ships. Nothing has to consume it. The cost of declaring it early is a document and a loader. The cost of retrofitting it is every pack ever written.

One hygiene rule, learned from the two dead MCP config keys: **a contract key lands in the same change as its reader and its tests, or it does not land.** `mcp_servers` is an open map the config-drift ratchet cannot see into, so nothing mechanical will catch a nested key that lies.

---

## Part 3: Identity

### 3.1 The distinction

Two identities, deliberately separate, because they have different lifetimes.

| | **Node** | **Instance** |
|---|---|---|
| Answers | which machine is this | which deployment is this |
| Lives in | `~/.prometheus/node/` | the vault |
| Travels with data | no | yes |
| Form | Ed25519 keypair | UUID |
| Cardinality | many per instance | one |

Copy your vault to a new machine and the instance ID follows. The node key does not. That is the whole design.

### 3.2 Node identity

Generated at first run. Ed25519 keypair. (`cryptography` moves to the base dependencies for this — identity must not inherit the optional-extra fragility the push stack has, where enabled-but-missing-deps is a boot-time failure mode.)

```
~/.prometheus/node/
  node.key      private, mode 0600, never leaves the machine
  node.pub      public, this is the node ID
```

Public key exposed on `/api/status` — the bearer-gated side. The unauthenticated `/health` endpoint does not carry it; that split is a standing security decision.

**Rules, binding:**

1. **The key is identity, never encryption.** No local file is ever encrypted with it. Losing the key must never make local data unreadable.
2. **The key is never the sole proof of anything.** Any future hosted service has an account layer above it with independent recovery. Keys enroll to accounts, many per account, revocable.
3. **Identity is not derived from hardware.** Swapping one GPU for another does not change the node. Generated key, not GPU fingerprint.
4. **Nothing is transmitted without opt-in.** The key exists locally and is inert until a user turns on something that uses it. Say this in the README before anyone asks.

Rule 4 exists because a self-hosted audience will see a keypair and ask what is phoning home. Answer it before the question is asked in public.

### 3.3 Instance identity

A UUID, generated when the vault is adopted or created, stored in the vault marker (1.1).

Instance-owned state: config, memory, wiki, skills, `SOUL.md`. Anything a user would expect to keep when they move machines. **The ownership rule binds now; the physical location of existing instance state does not move in this version** (see 1.4) — today much of it lives under `~/.prometheus`, and relocating it is a migration, not a refactor.

### 3.4 Node-owned state

Node-owned: hardware detection (`ANATOMY.md` and its history), traces, node key.

**Node-owned state must not share a directory with instance state**, and new node-owned state lands under:

```
~/.prometheus/
  node/                 node-owned, never copied to another machine
    node.key
    node.pub
```

The failure mode this prevents: a user moves to a new machine by copying state wholesale, carries the node key with it, and now two machines claim the same identity. Cheap to prevent now, ugly to diagnose later. The move-machines documentation says, in one line: copy the vault, never `node/`.

*Amended in v3:* `ANATOMY.md` itself stays at its current path for now. It has five independent reader sites, three of which carry only a bare filename resolved against the config dir, plus a boot-time writer — moving it without a single resolution point first recreates a documented incident class (the wiki root's nine-sites-three-derivations split). It relocates in a follow-on behind a `get_anatomy_path()` helper, not as a side effect of this spec.

### 3.5 The instance-to-node relationship

One instance, many nodes. The instance records which node keys are enrolled.

```yaml
# in the vault marker
instance_id: 550e8400-e29b-41d4-a716-446655440000
enrolled_nodes:
  - pubkey: <base64>
    label: studio-mini
    enrolled: 2026-08-28T00:00:00Z
  - pubkey: <base64>
    label: gpu-tower
    enrolled: 2026-08-29T00:00:00Z
```

That list is the fleet. In this version the local node self-enrolls when it first starts against an adopted vault — a single-machine reality where the human running `vault adopt` *is* the approval. The explicit approve-before-enroll step arrives with the fleet (3.6), and self-enrollment is the thing it replaces.

### 3.6 What identity does not solve

**Authentication is not authorization.** Two nodes with keypairs still do not know whether to trust each other. Something has to say node B may join instance A.

Today Tailscale provides this implicitly and that is acceptable. When the fleet becomes real, an explicit enrollment step is required: a node presents its public key, a human approves, the key lands in `enrolled_nodes`. Do not build it now. Do not forget it exists.

### 3.7 Deliberately deferred

- **Signing.** Traces and vault writes could be signed. They will not be, yet. Signing means every write path touches crypto and signature verification becomes a new failure mode. Generate the key, expose the public half, sign nothing until there is a reason.
- **Key rotation.** Needed before any hosted service. Not needed for a single local user.
- **Anatomy history.** Worth having eventually so benchmark runs can be attributed to the hardware that existed at the time. Currently a snapshot only. Noted, not scheduled.

---

## Part 4: Acceptance

Before Beacon ships, each of these is independently confirmed, not assumed. Owners are named because two of them cannot be confirmed from this repo.

**Prometheus, data format and identity:**

- [ ] Vault marker written on adopt/create, read on startup, refuses on higher version regardless of `format_check`, with a test asserting the refusal
- [ ] Telemetry `schema_meta` present, same refusal behaviour, same test
- [ ] Every trace row carries `node_id`, asserted by reading a row from a live run
- [ ] Node keypair generated at first run, private key mode 0600, verified on a clean machine (operator, post-deploy)
- [ ] `node.pub` visible on `/api/status`, verified by curl against the running daemon, not by unit test (operator, post-deploy)
- [ ] Node-owned files in `~/.prometheus/node/`, instance-owned files in the vault, verified by copying a vault to a second machine and confirming no key travels (operator)
- [ ] `instance_id` present in the vault marker and stable across daemon restarts

**Prometheus, extension surface:**

- [ ] MCP runtime constructed in the daemon path, not only the CLI
- [ ] An MCP tool from a configured server is *called* in a live loop, and per-server `allowed_tools` filtering demonstrably excludes a tool the server offers — at discovery and at `call_tool`
- [ ] A non-read-only MCP call reaches the SecurityGate and requires confirmation; a `readOnlyHint` tool does not
- [ ] Pack loader exists, discovers a fixture pack, and its skill is *promoted and then used in a live loop*, not merely discovered
- [ ] A pack declaring `pack_api: 999` is refused at load, with both versions named in the error — asserted by a fixture pack that declares it
- [ ] A pack whose `provides` block disagrees with its contents is refused, asserted by a fixture pack that lies
- [ ] An unpromoted pack skill is NOT reachable by the loop, asserted by driving the loop rather than by reading a flag
- [ ] Pack panels are discoverable read-only via the daemon (`/api/packs`)

**Beacon repo (not confirmable here):**

- [ ] A panel is constructed with no privileged handle, asserted against the call site's argument list
- [ ] Panel component loading, when it lands, names its sandbox before it ships

Structure passing is not function working. Each item above asserts a side effect or an outcome, per the standing principle.

---

## Change log

| Version | Date | Change |
|---|---|---|
| 1 | 2026-08-25 | Initial draft |
| 2 | 2026-08-28 | 2.2 gains a `pack_api` mismatch table mirroring 1.1. Tools removed from the pack contract and moved to MCP (2.3a); three registration points become two. 2.4 names an actual enforcement mechanism instead of asserting one, which the tools change made tractable. Part 4 acceptance items rewritten to match. |
| 3 | 2026-08-28 | Adopted after a full survey of the code the spec touches. 1.1: `format_check` knob (warn this release, refuse later), explicit `prometheus vault adopt`, absent-vault non-error, writer discipline, `enrolled_nodes` lives in the marker. 1.3: telemetry versioning acknowledged as net-new; DB-path unification folded in. 2.3: quarantine pinned to the existing `SkillDraftStore` lifecycle; daemon-side panels scoped to declaration + discovery. 2.3a: prerequisites grow from one to four (daemon wiring, gate treatment, allowlist enforced twice, advertisement decision); stdio-only accepted. 2.5: same-change rule for contract keys. 3.2: `cryptography` to base deps; `/health` exclusion stated. 3.3/3.4: ownership rule binds now, relocation of existing state (incl. `ANATOMY.md`) explicitly deferred. 3.5: v1 self-enrollment named as the thing fleet approval later replaces. Part 4 items annotated with owners; gate and discovery items added. |
