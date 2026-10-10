# Device scoping contract, v1

**Contract id:** `device-scoping/1`
**Scope:** What a *device token* may see and do with chat sessions, over REST and the WebSocket bridge. Code: `web/session_scope.py` (the rule), `config/device_store.py` (the record), `web/server.py` and `web/ws_server.py` (the enforcement points).
**Written for:** whoever builds on the device model next (approve-to-pair): section 2 is what changed in it, and section 2.3 is what to keep true.

---

## 1. The rule

A **device token** (`POST /api/devices`, one row in `devices.db`) sees and manages only the sessions that device **owns**. The operator's **global token** is unrestricted, and so is a daemon deliberately run with auth off (an empty token).

1. A session belongs to the device that brought it into existence: by `POST /api/sessions`, by its first send, upload, `switch_session`, or write to an id that exists **nowhere** (no live session, no durable row), or by forking into a fresh id.
2. A device cannot take an id in a namespace the daemon writes into itself (`telegram:`, `slack:`, `discord:`, `cli:`, `api:`, `coding:`, and the bare id `system`). Otherwise it could send to `telegram:<chat id>` before the operator's chat existed, and own it.
3. A session nobody claimed belongs to the operator: a Telegram chat, a cron run, anything that predates this contract. A device sees none of them.
4. To a device, **"not yours" is the same as "not there"**: a 404 `{"error": "unknown session"}` over REST, an `error` frame with `kind: "not_found"` over the socket. The answer must not tell it whether an id exists.
5. A device can **revoke only itself** (`DELETE /api/devices/{its own id}`). Another id is a 403, before any lookup, so an unknown id and someone else's get the same answer. The global token revokes any device.

Ownership is first-writer-wins, never reassigned and never deleted. Revoking a device leaves its sessions owned by it, readable by the operator and claimable by no one.

## 2. The device-model change

### 2.1 What was added

One table in `devices.db`, one record per owned session:

```
device_sessions (session_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, claimed_at REAL NOT NULL)
```

It is created on the **first claim**, not when the database opens, like `computer_devices`: a daemon that never scopes a device keeps the exact schema it had, and the parity fixtures (which record every table in this file) are unchanged.

Three methods on `DeviceStore`:

| Method | Does |
|---|---|
| `claim_session(session_id, device_id) -> bool` | Take the session if nobody owns it. True if the device owns it afterwards. False for another owner, or a device that is unknown or revoked. It arbitrates between *devices* only; checking that the id exists nowhere else is `SessionAccess`'s job. |
| `session_owner(session_id) -> str \| None` | The owning device id; None = the operator's. Swallows "no such table" and nothing else. |
| `owned_session_ids(device_id) -> set[str]` | Everything one device owns. |

### 2.2 What did not change

`DeviceIdentity`, token minting, hashing, verification, revocation, `last_seen_at`, the `api_devices` schema, push registration, the computer-use mark, and every `/api/devices` route's shape. The one behavioural change on those routes is the revoke rule in section 1.

### 2.3 What approve-to-pair must keep true

- **A device id is the owner key.** Mint through `DeviceStore.mint`. A newly paired device owns nothing; it starts from an empty list.
- **A pairing request is not a device.** Until it is approved there is no token and no identity, so there is nothing to scope. A pairing endpoint must not read, list or create sessions.
- **Handing a session to another device** (if pairing wants it) is one `UPDATE device_sessions` in `DeviceStore`, written there and nowhere else, and mind that a turn in flight keeps streaming: its remaining frames follow the new owner. No such method exists yet. Do not delete rows, and do not add a second table that also says who owns a session.
- **A new route keyed by a device id** needs the same self-or-global check as revoke (`_scope(request)` in `server.py`), or it reintroduces the hole this contract closes.
- **A new route keyed by a session id** must name the path parameter `session_id`. The router-level guard keys on that name; `test_no_route_names_its_session_parameter_anything_but_session_id` fails the build otherwise. A session id in a body or query is the route's own job: `_access.admit` (may create) or `_access.owns` (may not).

## 3. Where it is enforced

| Surface | Enforcement |
|---|---|
| Every REST route with a `{session_id}` path parameter (messages, delete, purge, title, pin, fork, profile, workspace, checkpoints, `/api/lcm/{id}`, computer binding, model) | One router-level dependency (`_guard_session_path`), installed before the first route. A read needs ownership. A write may also claim a brand-new id. |
| `GET /api/sessions` | Filtered to the caller's sessions (durable rows and the live overlay alike). |
| `POST /api/sessions`, fork | The new id is the creator's from that moment. A caller-chosen fork target must be claimable. |
| `POST /api/chat/send`, `POST /api/chat` (as `web:<id>`), `POST /api/stories/{pk}/dispatch`, `POST /api/computer/tasks`, `GET /api/computer/tasks/{id}` | `_access.admit` / `owns` on the id in the body, before anything uses the session. |
| `POST /api/chat/interrupt`, WS `interrupt` | Someone else's session gets the answer a quiet one does (`stopped: false`), and nothing is stopped. |
| `POST /api/search` | A named session must be the caller's. With none named, a device searches each session it owns and merges by rank. |
| `GET /api/events/recent`, `GET /api/activity/recent` | Rows whose payload names a session are returned only for the caller's sessions; session-less (daemon-level) rows are returned to everyone. A device may get fewer than `limit` rows. |
| WS `switch_session`, `send_message`, `chat_upload` | Admitted like the REST equivalents; otherwise the `not_found` error frame and nothing happens. |
| WS broadcast | `WebSocketBridge._wants` drops, for a device socket, every frame that names a session the device does not own, **whatever it subscribed to**. The session id is found flat (`payload.session_id`) or nested (`payload.payload.session_id`, the `sentinel_signal` wrapper). A frame naming no session reaches everyone. The operator's socket is the firehose it always was. |

A socket whose identity was never recorded, on a daemon with auth on, owns nothing (fail closed).

## 4. What this does not cover

Stated so nobody reads "device scoping" as more than it is. None of these changed.

- **The agent.** A device's turn runs the operator's agent with the operator's tools; `lcm_grep` searches every session. This scopes the API, not the model.
- **Slash commands.** A command typed into a device's own session runs with the operator's authority: `/events` lists recent signals from every session, `/memory`, `/wiki`, `/grants`, `/revoke` and `/gate` read or change daemon-wide state. Their replies reach only the sockets of that session's owner, but what they read is not scoped.
- **Approvals** (`/api/approvals*`). Ordinary prompts carry no session id, so there is nothing to scope by. Approving another session's prompt, and `grants`, are open to every valid token. Needs the session id plumbed into the queue first.
- **Background and coding tasks** (`/api/tasks*`, `/api/code*`) are task-id keyed and owned by no device.
- **Stopping a desktop task** stays open to any valid token, by the door's design: a stop can only end a task.
- **`/api/tools/recent`** returns tool inputs from every session and its rows carry no session id.
- **`/api/media?path=`** serves any cached image to whoever holds the path (`img_<uuid>`); a capability, not a listing.
- **Operator-wide resources**: config, skills, memory, wiki, files, artifacts, cron, providers. Not sessions.
- **Push notifications** (`push/dispatcher.py`): `turn_completed`, `task_*` and approval pushes go to every registered device, whoever owns the session. A product decision, not a bug fix.
- **`GET /api/devices`** still lists every enrolled device (name, platform, last seen, push status).
- **An id-existence oracle remains** for ids outside the daemon namespaces: a device that sends to a guessed id learns whether it existed, and takes it if not. Real ids are `<gateway>:<uuid4>`.

## 5. Upgrading

Sessions that existed before this change are **unowned, so operator-only**. A device that used sessions before the upgrade no longer sees them over REST or the socket until the operator gets them back to it: there is no reassignment tool yet. New sessions are scoped from the first message.
