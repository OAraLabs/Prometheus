# Chat projects: one store for grouping, folders and instructions

Design, 2026-11-13. Two client contracts describe the same feature from two sides:

- **Grouping** (written for the phone): a project id on session rows, project routes with a chat-only/full device
  split, and a suggestions route that proposes a project from a session's tool calls.
- **Projects that carry context** (written for the desktop): a project is a folder on the daemon's computer with
  standing instructions (`PROMETHEUS.md`) and reference files, shared with "work in a folder".

This note makes them one design: one store, one set of routes, a `projects` capability, and one rule for chat-only
devices. Section 12 lists where the two contracts disagreed and what was settled, and why.

## 0. In brief

- A **chat project** is a named folder on this Prometheus's computer. Conversations filed in it work in that folder and
  read its `PROMETHEUS.md`. Reference files sit in `<folder>/reference/`.
- **Per instance.** A project belongs to one Prometheus. No project spans instances, and no id means anything to
  another instance. A client that talks to several daemons shows each one's projects separately.
- **One store**, in the conversation store beside `session_workspaces` and `session_pins`: `chat_projects`,
  `session_chat_projects`, and a change counter.
- **Routes** under `/api/chat-projects` (`/api/projects` is the kanban board, unchanged). Session rows carry
  `chat_project_id`.
- **The owner** (the global token or an owner device) does everything. **A chat-only device** sees only the projects
  its own sessions are in, as an id and a name, and nothing else: it cannot create, rename, delete or file, read
  instructions, or touch files. Fail closed.
- **`projects`** appears in the `subscribed` ack's `server_capabilities`. An app that does not see it keeps whatever it
  does today.
- **Suggestions** are computed on the daemon, over the full history, with the desktop's rules ported (plus one rule
  for the daemon's own project folders). Owner only, on a path of their own.
- **Filing binds the folder by default; a migration files as a label only** (`keep_workspace`), so moving a client's
  local projects in never changes where an old conversation writes.

## 1. The model

```json
{
  "id": "cp_7c1e02ab9d4f",
  "name": "Kitchen remodel",
  "folder": "/home/owner/prometheus-projects/kitchen-remodel",
  "folder_managed": true,
  "instructions_chars": 1240,
  "instructions_rev": "a41f09c2d77e10b5",
  "files": 6,
  "sessions": 14,
  "import_key": null,
  "created_at": 1794571270.1,
  "updated_at": 1794571270.1
}
```

- `id`: daemon-minted, `cp_` and 12 hex characters, opaque. Never reused.
- `name`: cleaned (trimmed, inner whitespace collapsed, at most 60 characters); unique without regard to case.
- `folder`: absolute, on this daemon's computer. `folder_managed: true` means the daemon made it under the projects
  root; `false` means the owner pointed the project at a folder that already existed ("work in a folder").
- Instructions are the content of `PROMETHEUS.md` at the folder's top level. The file is the store: there is no second
  copy, so the agent reads exactly what the owner wrote. `instructions_rev` is the first 16 hex of the SHA-256 of its
  bytes, `""` when there is no file.
- `files`: how many reference files are listed (3.5). `sessions`: how many conversations are filed in it, not counting
  deleted ones. `import_key`: the key a client migrated it under, or null.
- A conversation is in at most one project.

### 1.1 The projects root

Config `projects.root`, default `~/prometheus-projects`: a visible folder in the owner's home directory (Will,
2026-11-13), not inside the agent's workspace directory. No spaces, so the agent's shell commands need no quoting.
It is created on first use, mode 0700. It must pass `validate_workspace_path` like any folder a project uses
(absolute, an existing directory once made, not the filesystem root, not under `security.denied_paths`); if it does
not, creating a managed project is `400 bad_folder` with the validator's reason, and nothing is created. An existing
root is checked with `lstat`: a symlink, or a directory owned by another user, is refused the same way; one the daemon
did not make keeps its mode.

A managed folder is named from the project name: lowercase ASCII letters, digits and `-`, runs collapsed, at most 48
characters, `project` if nothing is left; `-2`, `-3`, … when the name is taken on disk. Renaming a project never
renames its folder.

## 2. Storage

In the conversation store (`memory/lcm_conversation_store.py`), beside `session_workspaces`:

```sql
CREATE TABLE IF NOT EXISTS chat_projects (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    name_folded    TEXT NOT NULL UNIQUE,     -- lower(cleaned name)
    folder         TEXT NOT NULL UNIQUE,     -- resolved absolute path
    folder_managed INTEGER NOT NULL,
    import_key     TEXT UNIQUE,              -- NULL unless a client migrated it
    position       REAL NOT NULL,            -- display order: creation order (no reorder route in v1)
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS session_chat_projects (
    session_id TEXT PRIMARY KEY,             -- a session is in at most one project
    project_id TEXT NOT NULL,
    filed_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_projects_rev (
    id  INTEGER PRIMARY KEY CHECK (id = 1),
    rev INTEGER NOT NULL
);
```

- Presence is the filing; unfiling deletes the row (the `session_pins` discipline).
- Deleting a project deletes its membership rows in the same transaction. No `ON DELETE CASCADE`: SQLite enforces
  foreign keys only with `PRAGMA foreign_keys=ON`.
- `rev` goes up by one on every change to a project or a membership, in the same transaction as the change.
- **The forget cascade names the new table:** `session_chat_projects` joins the enumerated list in
  `forget_session`. Projects are not per-session, so a forget leaves them alone.
- Instructions and reference files are plain files in the folder, never in the database.

## 3. Routes

All under `/api/`, behind the bearer middleware. "Owner" means `identity.is_operator` (the global token or an owner
device). Errors are `{"error": "<sentence>", "kind": "<slug>"}`.

### 3.1 Projects

| Route | Who | Request | Result |
|---|---|---|---|
| `GET /api/chat-projects` | owner, chat-only (filtered) | | `{"projects": [...], "rev": 41}` in display order |
| `POST /api/chat-projects` | owner | `{"name", "folder"?, "import_key"?}` | `201 {"project", "created": true}` or `200 {"project", "created": false}` |
| `GET /api/chat-projects/{id}` | owner, chat-only (filtered) | | the project plus `"session_ids": [...]` |
| `PUT /api/chat-projects/{id}` | owner | `{"name"}` | `{"project"}` |
| `DELETE /api/chat-projects/{id}` | owner | | `{"ok": true, "id", "unfiled": 3}` |

**Create** (both contracts' rules, reconciled):

1. `import_key` given and already stored: `200` with that project, `created: false`. A migration that is interrupted
   and run again creates nothing twice. Keys are per instance.
2. The cleaned name matches an existing project (any case):
   - `folder` absent, or the same resolved folder as that project's: `200` with it, `created: false`. Two devices
     that both make "Beacon" converge.
   - a different `folder`: `409 name_taken`.
3. `folder` given: it must pass `validate_workspace_path` (`400 bad_folder` with its reason). It may not be the home
   directory itself, the projects root itself, inside the agent's home (`~/.prometheus`), or inside or around another
   project's folder (`400 bad_folder`), so one project's conversations can never write into another's. No other
   project may use it (`409 folder_in_use`). The project is `folder_managed: false`; the folder is not created.
4. `folder` absent: a managed folder is made under the projects root (section 1.1). If the root lies inside another
   project's folder, that is `400 bad_folder` too (a configured root inside an unmanaged project would put every
   managed folder inside it).
5. An empty name after cleaning: `400 name_required`.
6. Two creates at once: the database's uniqueness decides. The loser gets the winner's project (`200`, `created:
   false`) when rule 2 would have returned it, else `409`; a managed folder it made a moment ago, still empty, is
   removed, so no orphan is left behind.

Every answer that changes something carries `"ok": true`, as in the desktop contract.

**Rename** onto another project's name is `409 name_taken`; renaming to the same project's name in another case is
allowed. The folder never changes.

**Delete** removes the project and its memberships, clears every workspace binding the project set (`set_by:
"project:<this id>"`, on members and on sessions since moved elsewhere as a label), and **never deletes the folder or
any file**. No route removes a folder.

### 3.2 Membership

`PUT /api/sessions/{session_id}/project`, owner only, `{"chat_project_id": "cp_…" | null, "replace_workspace"?: bool,
"keep_workspace"?: bool, "device_session"?: bool}` → `{"ok": true, "session_id", "chat_project_id", "workspace":
"<the session's workspace now, or null>"}`.

- **By default filing binds** the session's workspace to the project's folder with `set_by: "project:<id>"` (which
  project set it, so a later move, unfile or delete clears only that project's own binding): the
  conversation works in the folder and reads its instructions. The folder is validated again first, and its resolved
  path must still equal the stored one (a folder since swapped for a symlink, or fallen under `denied_paths`, is
  `400 bad_folder`). The change is logged at the level the workspace route logs it: it moves that conversation's write
  boundary.
- **`keep_workspace: true` files as a label only**: the membership is recorded and the workspace is left exactly as
  it is (none, or one set by hand). This is what a migration sends (section 9), so moving local projects in never
  changes where an old conversation writes.
- A session whose workspace was set another way (a `set_by` that does not start with `project:`) is `409 workspace_set` with the current
  `workspace`, unless `replace_workspace: true` or `keep_workspace: true`. A folder chosen by hand is never lost
  silently. After `replace_workspace`, unfiling leaves no workspace; it does not restore the old one.
- `null` unfiles, and clears the workspace only if this project set it. Deleting a project clears every binding it
  set, including on a session since moved to another project as a label.
- **Whose session.** Binding a session that a chat-only device owns (`device_store.session_owner` names a device that
  is not an owner device) is `409 device_session` unless `device_session: true`: it would give that device's turns the
  project's folder, instructions and reference names. A label-only filing of such a session needs no flag (the device
  then learns only the project's name). A machine session (`memory/session_kind.is_machine_session`: cron, workers,
  coding runs) is `400 machine_session`, always.
- The session must exist, live or durable: an unknown id is `404 unknown session` (a binding written for an id
  nobody has used yet could be inherited by whoever claims it first). To start a conversation already in a project,
  use `POST /api/sessions` with `chat_project_id`.
- Filing is idempotent. An unknown project is `404 unknown_project`. Membership, binding and `rev` change in one
  transaction. A forked session starts unfiled.

`POST /api/sessions` accepts an optional `chat_project_id`: owner only (a chat-only device sending it is `403
operator_only`, and no session is made). The new session is created filed and bound. An unknown project is `404`, and
no session is made.

### 3.3 Session rows

Every row of `GET /api/sessions` gains `chat_project_id: string | null`, beside `pinned`, on both branches: the durable
rows and the live overlay, which replaces the durable dict and must restate it (as it restates `pinned`). A chat-only
device sees only its own sessions there already.

### 3.4 Instructions (owner only)

`GET /api/chat-projects/{id}/instructions` → `{"content", "rev", "max_chars": 12000, "effective": [...]}`. No file:
`content: ""`, `rev: ""`. `effective` lists the instruction files the loader would actually read for a conversation in
this folder, deepest first, each as `{"path", "chars", "shadowed_by"?}`: the folder's own convention file (one per
folder level, `PROMETHEUS.md` first), its `.prometheus/rules/*.md`, and the parents' files. So an unmanaged repo whose
instructions are an `AGENTS.md` shows that, and the editor never pretends the folder has none.

`PUT /api/chat-projects/{id}/instructions` with `{"content", "base_rev"}` → `{"ok": true, "rev"}`.

- `base_rev` must match the file now; otherwise `409 conflict` with the current `content` and `rev`, and nothing is
  written (the memory editor's rule).
- Over `max_chars` (the loader's per-file cap, so what is saved is what the model sees): `400 too_long`.
- Written atomically (a temp file in the folder, then a rename). A `PROMETHEUS.md` that is a symlink is neither read
  nor written by these routes: `409 not_a_file`. `effective` marks symlinked files as such. The loader itself is not
  changed: it follows symlinks today in every workspace (shipped behaviour, and repositories link `CLAUDE.md` and
  `AGENTS.md`), and whether it should is a separate question for the owner, not part of this feature.
- Saving a `PROMETHEUS.md` where another convention file exists at the same level (which it would then shadow, since
  the loader takes one per level) is `409 would_shadow` with the file names, unless `"shadow_ok": true`.
- The owner's write over an owner-only route does not go through the tool gate. The agent editing the same file
  still asks, as today.

### 3.5 Reference files (owner only)

| Route | Result |
|---|---|
| `GET /api/chat-projects/{id}/files` | `{"files": [{"name", "size", "mtime"}]}`, top level of `reference/` only |
| `GET /api/chat-projects/{id}/files/{name}` | the bytes, `Cache-Control: no-store`, `Content-Type` from the name |
| `PUT /api/chat-projects/{id}/files/{name}` | raw body; `?replace=true` to overwrite → `{"ok": true, "file"}` |
| `DELETE /api/chat-projects/{id}/files/{name}` | `{"ok": true}` |

- `name` is one path segment: no `/` or `\`, no leading `.`, no control characters, at most 120 characters.
  Otherwise `400 bad_name`.
- The resolved path must be inside `<folder>/reference/`. A symlink there is not listed, not served, not replaced and
  not deleted, and `reference/` itself being a symlink makes every file route `409 not_a_file`.
- At most 20 MB a file (`413 too_large`), read with a running count, so a chunked body is cut off at the cap. At most
  200 files (`409 too_many`). An existing name without `replace=true` is `409 exists`.
- The routes list and serve only regular files whose names pass the `bad_name` rule; anything else in `reference/`
  (made by the agent, a shell, git) is not listed.
- Nothing in `reference/` is ever loaded as instructions. The daemon adds one generated section after the project's
  instructions in the prompt, listing the file names, so the agent knows they are there and reads them with its file
  tools: names only, never contents, each quoted, under a label that says they are reference data and not
  instructions, at most 50 names and 2,000 characters (then "and N more").
- Opened through the folder's directory handle with `O_NOFOLLOW`, never by a path checked first and opened later.

### 3.6 Suggestions (owner only)

`GET /api/chat-project-suggestions` (its own path: under `/api/chat-projects/` it would match the chat-only
`{id}` route shape) →

```json
{"suggestions": {"<session_id>": {"key": "beacon", "label": "beacon", "share": 0.82, "calls": 11,
                                  "project": {"id": "cp_…", "name": "Beacon"} | null}},
 "rev": 41}
```

- For **unfiled** sessions only, the 200 most recently active, leaving out machine sessions and sessions a chat-only
  device owns (filing those needs the explicit flag of 3.2).
- The desktop's rules, ported to Python and held to the desktop's own smoke cases (`session-projects-smoke.ts`
  sections 2 and 3, the same cases the phone's tests run), so the implementations answer alike: the project folder of
  each path in each tool call's input; a suggestion when at least 3 calls touched a project folder and one folder has
  at least 60% of them, with no tie at the top.
- **One deliberate addition:** a path inside a chat project's stored folder belongs to that project first (key: the
  project's id, label: its name), and the projects root is a container like `~/projects`. Without it every managed folder would read as
  the project "prometheus-projects". The desktop has no daemon folders to know about, so its rules lack this; the
  smoke cases still pass, and the additions have their own cases.
- `project` is the existing project that suggestion would file into: one whose folder's own name, folded the same way,
  is the suggestion's `key`, or else one whose name matches the `label` in any case. `null` means accepting it creates
  a project.
- Computed per session from its tool calls in the conversation store, kept by `(session_id, max_rowid)` so an
  unchanged session is not read again.
- A suggestion is only ever offered. Filing stays an explicit act, through 3.2.
- Dismissals stay on each device (section 8).

## 4. Who may call

| Route | Owner | Chat-only device |
|---|---|---|
| `GET /api/chat-projects` | all, full; `"role": "owner"` | only projects with at least one of its sessions; each as `{"id", "name", "sessions"}` where `sessions` counts its own; no `rev`; `"role": "chat_only"` |
| `GET /api/chat-projects/{id}` | full | `404` unless one of its sessions is in it; then `{"id", "name", "session_ids"}` with its own sessions only |
| everything else in section 3 | yes | `403 operator_only` (default-deny in `route_access`) |
| `chat_project_id` on its session rows | yes | yes, for its own sessions |
| `chat_project_id` on `POST /api/sessions` | yes | `403 operator_only` |
| `chat_project_changed` frame | yes | no |

Why this much and no more:

- **Filing moves a write boundary.** A filed session works in the project's folder. A chat-only device filing its own
  session would point its turns at the owner's project folder, so filing is the owner's act, always.
- **Through these routes, a chat-only device learns only the name.** It already sees its own sessions; the routes add
  the name and id of the project one of them is in. Not the folder path (the owner's filesystem), not the
  instructions, not other projects, not other sessions, not how busy projects are (`rev`). **But a filed session works
  in the project's folder:** its turns read the folder's instructions and the reference names, and can write files
  other conversations in that project will read. That is why filing a chat-only device's session takes an explicit
  `device_session: true` from the owner (3.2), and why suggestions never offer one.
- **Every owner-only handler also checks** `is_operator` itself, besides the route table: a path shape that happens to
  match a chat-only entry must not open an owner route. A test asserts no owner-only route matches any chat-only
  pattern.
- **"Not yours" is "not there".** A project with none of its sessions answers `404`, the same as an unknown id.

`web/route_access.py` lists `GET /api/chat-projects` and `GET /api/chat-projects/{id}` in `SCOPED_ALLOWED`; every other
new route goes in `OPERATOR_ONLY`; `tests/test_route_access.py` pins both. `role` lets a phone know up front whether to
offer "Move to project…".

## 5. The `projects` capability

The `subscribed` ack's `server_capabilities` lists `projects` on a daemon with these routes (no configuration needed).
An app that does not see it, or that gets `404` from `GET /api/chat-projects`, treats the daemon as having no projects:
it keeps its local projects as they are, offers no instructions or files, and never deletes local data.

## 6. Change notification

`chat_project_changed`, payload `{"rev", "project_id", "change": "created" | "updated" | "deleted" | "instructions" |
"files" | "membership", "session_id": "<for membership, else null>"}`. Sent **only to owner sockets**, through the
bridge's existing `send_to_operators` (the path pairing frames take). It carries ids and a change kind, never a name,
a path, instruction text or a file name. A client that sees a `rev` it does not have re-reads the list. A chat-only
device gets no frame and refreshes the way it refreshes pins. On a daemon with auth off no socket has a recorded
identity, so no socket gets the frame; clients refetch on refresh there too.

The SignalBus is not used for this: its events reach every socket as `sentinel_signal` frames, chat-only ones
included.

## 7. What the agent sees

A filed session's working directory is its project's folder (the existing workspace binding), so the instruction
loader already reads the folder's `PROMETHEUS.md` and `.prometheus/rules/*.md` within its caps, and the permission
gate already asks before a tool writes one of them. The one addition is the generated reference-file line (3.5).

## 8. Per device, and therefore local

These stay on each device: dismissed suggestions (`session_id → the key said no to`; a new key shows again), the
Unfiled-as-one-list toggle, and which groups are collapsed.

## 9. Migrating a client's local projects

The desktop keeps projects locally today (`{id, name}` and a session→project map, one computer only). On first run
against a daemon that advertises `projects`:

Nothing is dropped, and nothing changes where a conversation writes.

1. **Per instance.** Each local project's sessions are grouped by the Prometheus they belong to. A local project whose
   sessions are on several instances becomes one project on each of those instances, with the same name. A session is
   filed on the instance it lives on.
2. **A local project with no sessions** is created on the client's primary instance (the one it opens on start).
3. For each (instance, local project): `POST /api/chat-projects` with the name and an `import_key` derived from the
   local id (keys are per instance). With no `folder`, so the daemon makes a managed one, or returns the instance's
   existing project of that name.
4. `PUT /api/sessions/{id}/project` for each session **with `keep_workspace: true`**: a label only, so an old
   conversation keeps the workspace it had (none, or one chosen by hand), and neither `409 workspace_set` nor `409
   device_session` can happen, so every session lands. A session the daemon refuses for another reason (a machine
   session, an unknown id) stays in the client's local map for that instance, and the client says which ones.
5. **Per instance, the client retires its local data for that instance only** after the daemon has returned every
   migrated project from `GET /api/chat-projects` and every filing has answered. An instance that is offline, or that
   does not advertise `projects`, keeps its part locally until it does; the client migrates it then. Dismissals stay
   local.

## 10. Build order

1. This note.
2. The store (tables, rev, forget cascade) and the managed-folder maker, red tests first.
3. Projects, membership, session rows, `POST /api/sessions`, access, the capability and the frame (two reviews: auth).
4. Instructions and reference files, and the prompt's reference line (two reviews: file writes and paths).
5. Suggestions, with the desktop's smoke cases (one review).

Then the pastes: the desktop (move local projects into the daemon, per instance, losing nothing) and the phone (project
sections, "Move to project…" for owner devices, suggestions).

## 11. Not in this version

- A project that spans instances; moving a project between instances.
- Changing a project's folder; deleting a folder from the daemon.
- A reorder route (`position` is there so one can be added without a migration).
- A link from a chat project to a board project.
- Several instruction files managed by the routes (`.prometheus/rules/*.md` is still read by the loader).
- Chat-only devices filing or creating.

## 12. Where the contracts disagreed, and what was settled

Settled with the charter's principles (fail closed, capability gating, smallest change):

- **P1. The name.** `/api/chat-projects` (the desktop's), with `chat_project_id` on session rows (neither contract's
  exact field). The phone's contract warned that a `project_id` on session rows is one step from being read as a
  kanban link; `chat_project_id` keeps the resource and the field the same word.
- **P2. Chat-only devices.** The phone's contract let a chat-only device list every project, create one and file its
  own sessions; the desktop's made everything owner-only. Settled by Will's rule: a chat-only device sees only the
  projects its own sessions are in, as id and name, and does nothing else. Filing moves a write boundary, so it is
  the owner's.
- **P3. Creating a duplicate name.** The phone's contract returns the existing project; the desktop's says `409`. Both:
  the existing one when no different folder is asked for (devices converge), `409 name_taken` when a different folder
  is. `import_key` makes a migration safe to repeat either way.
- **P4. Uploads over REST, not a WebSocket frame.** The desktop contract adds files with a WebSocket frame because
  REST bodies are capped at 2 MB. A raw-body `PUT` read with a running count to 20 MB, plus one exception in the body
  check for that route, is simpler for clients and keeps `ws_server.py` untouched (other work goes on there).
- **P5. The change frame goes to owner sockets only**, not through the SignalBus, whose frames reach chat-only devices.
- **P6. Suggestions on the daemon** (the phone contract's option 1): one answer on every device, over the full history.
  Owner only, since accepting one files a session.
- **P7. Per instance** (Will, 2026-11-13): no project spans instances; the desktop's migration splits a local project
  across instances, keeping the name on each side, rather than dropping anything.
- **P8. One file for instructions**, `PROMETHEUS.md`, capped at the loader's per-file limit (12,000 characters).
- **P9. The projects root** is `~/prometheus-projects` by default: visible, in the home directory (Will's decision),
  without a space.
- **P10. Filing binds by default; migration files as a label.** The phone's contract treated filing as a label; the
  desktop's as a folder bind. A project that carries context only works if its conversations work in its folder, so an
  owner's "Move to project…" binds. But a migration must not move old conversations into folders (perhaps a real
  repository, if the name converges onto an unmanaged project), so it sends `keep_workspace: true`.
- **P11. A chat-only device's session needs an explicit flag to be filed**, and a machine session cannot be: filing
  gives a session the folder, its instructions and its reference names.
- **P12. Suggestions know the daemon's own folders** (3.6), a deliberate difference from the desktop's rules.
- **P13. The instruction loader is not changed.** Skipping symlinked instruction files would change shipped
  behaviour in every workspace; it is left to the owner as its own question. Only the instructions routes refuse a
  symlinked `PROMETHEUS.md`.
