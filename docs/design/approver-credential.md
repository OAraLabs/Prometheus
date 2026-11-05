# W4 — an approval needs a person, not the API token

**Status: PROPOSAL, for Will's ruling. No code.** This is W4 from
`docs/design/computer-use-v1.1.md` (§8): "accepted, separate WP". It closes
D15 for every tool, not just the computer-use door (whose own W3 guard ships
with PR 5 either way). Line numbers are against `main` @ `37f5ca4`.

## The rule, in one line

An answer that **lets** something happen (approve, `/approve all`, `/gate off`)
must come from a **person's credential**: a Beacon device a person enrolled,
or a Telegram user the operator named. The daemon's API token can still read
the queue and **deny**, but it cannot approve.

## Why: what answers an approval today

An approval is answered by `ApprovalQueue.approve(request_id, *, scope, grant)`
(`permissions/approval_queue.py:730`). Nothing on that path knows who is
answering, and nothing records it.

| Surface | Code | What lets you in | Who approved, recorded? |
|---|---|---|---|
| `POST /api/approvals/{id}/approve` | `web/server.py:4417-4484` → `approve_detail` | Bearer middleware (`server.py:330-373`). The **global API token** is enough. The handler takes no `Request`, so it cannot see whether the caller was a device. | No |
| `POST /api/approvals/all/approve` | The path id is parsed as chat text, so `all` becomes `/approve all` (`gateway/commands.py:2618-2635`). | Same: the global token **drains the whole queue**. | No |
| `POST /api/approvals/{id}/deny` | `server.py:4547-4553` | Global token | No |
| `/gate off` via web chat (`/api/chat/send` or WS) | `commands.py:2060-2076`, `:4289-4301` → `gate.set_mode("autonomous")` | Global token. This **switches off the APPROVE tier** for every non-computer tool (`checker.py:1062-1072`). | No |
| Telegram `/approve`, `/deny` | `gateway/telegram.py:2555-2569`, `:2625-2636` | The **chat id** must be in `allowed_chat_ids` (`telegram.py:814-846`). There is no user check: anyone in an allowed group chat can approve. | No (the user id is logged only on a refusal) |
| Slack, Discord | `slack.py:1411-1450`; `discord.py:1572-1598` | Channel allowlist (open when it is empty); Discord DMs are open when `user_ids` is empty (`gateway/config.py:105-168`). | No |
| CLI | none (`__main__.py` has no `approve`; the CLI's gate has no queue, so APPROVE is refused) | — | — |

**Why that is a hole for a model, not just a person:**
- The global token sits in plaintext in `~/.config/prometheus/env`.
- With the read floor off (`bash_confinement` ships `"off"`), `cat` reads it. Before #659, every door other than `bash` read it even with the floor on.
- So a model can answer its own prompt with one `curl`, or `/gate off` itself.
- **Making approval "device-only" is not enough on its own**, because the global token also *mints* device tokens. `POST /api/devices` asks for nothing but the global token (`server.py:377-398`), and setup-mode pairing hands out the global token itself (`web/setup_server.py:624-650`).

**Recording:** `AuditEntry` already has a `user_id` column (`permissions/audit.py:54`, `:151`). The resolution row never fills it (`approval_queue.py:709-727`), and neither the `approval_resolved` signal nor a stored grant carries an approver.

**The auth layer can already tell who is calling.** `verify_token` returns a `DeviceIdentity`, and `is_global` separates the shared token from a device (`config/api_token.py:280-321`). REST stores it in `request.state.device_identity` (`server.py:372`). The WS bridge keeps one per socket (`ws_server.py:332`).

## The proposal

### 1. What counts as a person's credential

| Credential | Counts as a person when | New config/state |
|---|---|---|
| **Beacon device token** | The device row has `approver = 1`. Only an enrolment step a model cannot perform sets it (§3). | One column on `api_devices` (`config/device_store.py:67-77`). |
| **Telegram** | The *sender's user id* is in `gateway.approver_user_ids`. Default: in a **private** chat, the allowed chat id already *is* the user id (Telegram uses the same number), so a one-person setup needs no new config. | `gateway.approver_user_ids` (list), optional. |
| **Slack / Discord** | The sender's user id is in that platform's `approver_user_ids`. With none configured, these platforms can still **deny** but cannot approve. | Per-platform `approver_user_ids`. |

Everything else is not a person: the global token, a device without the `approver` flag, open mode (no token at all), Paperclip (which holds the global token), and any in-process caller.

### 2. What each credential may do

| Action | Person | Global token / non-approver device |
|---|---|---|
| List pending approvals, list grants | yes | yes (unchanged) |
| **Deny**, revoke a grant | yes | yes. This is the safe direction, so it stays open. |
| **Approve** (any scope), `/approve all` | yes | **no: 403** "approving needs a person's credential", and the request stays pending |
| `/gate off` (autonomous mode = approve everything) | yes | **no** |
| Enrol a device | yes, as an approver device (§3) | yes, but as a **non-approver** device (push and status keep working) |

The REST approve route stops parsing the path id as chat text. `{id}` must be a request id; `all` moves to its own route (`POST /api/approvals/approve-all`), which is person-only.

### 3. Enrolment: closing the minting hole

A device becomes an approver only through a step that needs a person **outside the model's reach**. Three ways in:

1. **Setup-mode pairing.** Setup mode runs before the daemon, so no agent loop exists yet. The 6-digit code is shown to the person at the console. Pairing then returns a **device token with `approver = 1`** instead of the global token. *(This is a Beacon change: see below.)*
2. **Confirmation from an existing person channel**, after setup. A new device enrolled with the global token starts as a non-approver and shows a short code. Either:
   - the operator sends `/devices allow <code>` from an approver Telegram user, or
   - an approver Beacon device taps "Allow this device to approve".
3. **On the box**: `oara devices allow <id>`. This works only with the write floor active, because without it a model's `bash` can write `~/.prometheus/data/devices.db` and do the same thing. So the command refuses unless `bash_write_confinement` is in force (state `active`), and says why. *(The tokens in that file are SHA-256 hashes, so reading it yields nothing usable; writing it is the threat.)*

### 4. Recording who answered

- `ApprovalQueue.approve`, `deny` and `approve_detail` gain a **required** keyword, `by: Approver`. It is required, not defaulted, so a new surface cannot forget it.
- `Approver` is `{kind: device|telegram|slack|discord|in_process, id, name}`.
- It is written in four places:
  - the audit row's existing `user_id`, as `device:<id>`, `telegram:<user_id>`, and so on;
  - the `approval_resolved` signal (`approved_by`);
  - a new `granted_by` field on a remembered `Grant`;
  - the WS frame, so Beacon can show "approved by Will's iPhone".
- `/grants` shows who granted each one.

### 5. Rollout: record → warn → enforce

| Step | Behaviour | Breaks |
|---|---|---|
| **1. Record** (one PR) | Thread `by:` everywhere, fill the audit `user_id`, and add the `approver` column (existing devices = 0). | Nothing |
| **2. Warn** (config `security.approvals.require_person: warn`, the default for a release) | A non-person approve still works, but logs WARNING, writes `user_id=global-token` to the audit row, and shows "approved with the API token" in `/grants` and Beacon. | Nothing; makes the remaining callers visible |
| **3. Enforce** (`require_person: enforce`, after Beacon desktop enrols) | §2 applies. | The callers in the next section |

## How scripts and the CLI approve today, and what breaks at "enforce"

| Caller | How it approves today | At enforce |
|---|---|---|
| **CLI** (`oara`) | It doesn't. There is no `approve` subcommand, and the CLI's `SecurityGate` has no queue, so APPROVE is refused (`__main__.py:839-860`). | Nothing breaks. The new `oara devices allow` is additive. |
| **Scripts**: `scripts/computer_use_probe.py:78`, `computer_use_multistep_probe.py:157`, `computer_use_grant_probe.py:110-125` | **In-process**, on a queue the script built itself (`queue.approve(...)` / `cmd_approve(...)`). None approves over REST. | One-line edit each: pass `by=Approver.in_process("<script>")`. The queue lives in the daemon's memory, so an in-process caller can only ever answer its own queue, never the daemon's. |
| **Docs / README** | No `curl` example hits an approve route. The routes are only listed (`docs/guide/api.md:188-190`). | Add a sentence saying approve needs a person's credential. |
| **Beacon desktop** | Pairs with the 6-digit code **and receives the global token** (`setup_server.py:624-650`), or takes a pasted token. Approves via REST with it (Mission Control card, exec-approval overlay; `docs/guide/beacon.md:38`, `:176`). | **Breaks** until it enrols as a device and is allowed (§3). This is the reason for the warn step. It needs a Beacon desktop release (other repo). |
| **Beacon iOS** | Enrols as a device (`POST /api/devices`) and approves with its device token. | Breaks until that device is allowed: one `/devices allow` from Telegram, or migrated by B2. |
| **Telegram, private chat** (the usual one-person setup) | `/approve` in an allowed chat | Keeps working: the private chat's id is the user's id (B3). |
| **Telegram group chat, Slack, Discord** | Any member of an allowed chat or channel | Approve needs `approver_user_ids`; deny keeps working. |
| **`/gate off` from Beacon chat** | Global token | Needs a person (from Telegram or an approver device). |
| **Paperclip** | Never approves. It holds the global token (`server.py:1299-1301`) for turns only. | Nothing. A Paperclip turn that hits APPROVE waits for a person, as it does today. |
| **Open mode** (an explicit empty `api_token`) | No credential at all; REST approve accepts anyone. | REST approve refused (B4). Telegram still works. |
| **Headless, no Telegram** | There is no queue at all: the queue is built only with Telegram and `security.approval_queue.enabled` (`daemon.py:1519-1551`), so APPROVE is refused (`checker.py:1674-1676`). | Nothing changes. |
| **Tests** | REST approves in open mode: `tests/test_api_approve_reports_what_it_did.py` (9 calls). In-process: about 33 `cmd_approve`/`approve_detail` calls, 12 `queue.approve`/`deny`, 4 `cmd_deny` (`test_approval_grants.py`, `test_consent_grants.py`, `test_typo_answers_immediately.py`, `test_gateway_command_pins.py`, `test_approval_push.py`, `test_approval_queue.py`, …). 19 gateway-handler tests (`test_discord.py`, `test_gateway_g1.py`, `test_control_plane_dispatch.py`, `test_gateway_command_authz.py`); parity pins `test_gateway_parity.py:267-281`. | Step 1 adds `by=` to the in-process calls (mechanical). Step 3 makes the 9 REST tests enrol an approver device, and the handler tests carry a user id. |

## What this does not fix, and what it depends on

- **It depends on the write floor.** A model that can write
  - `prometheus.yaml` (`permission_mode: autonomous`, or a grant),
  - `devices.db`, or
  - its own audit trail

  goes around any approval rule. The write floor makes `~/.prometheus` and `~/.config` read-only to every model shell, so this proposal is a boundary only where `bash_write_confinement` is `active`. There is one exception: a configured workspace root (or a `bash_write_allow` entry) that contains them. The shipped root is `~/.prometheus/workspace`, which does not contain them. A root of `~` would, and the enforce step should refuse to start in that configuration. #659 puts every model shell behind it. On a host without bubblewrap (macOS) it isn't, and `/api/status` should say so next to the approval mode.
- **The global token stays a secret worth keeping.** It can still deny, read the queue and turn on chat. The read floor (`bash_confinement`) is what keeps a model from reading it. See #659's proposal for shipping that as `"auto"`.
- **`PUT /api/sessions/{id}/workspace`** (`server.py:1704`) moves the write boundary, and so changes what prompts. It is in the same class as `/gate off`, but it is a separate change: listed here, not designed.
- **Revocation doesn't close an open WebSocket.** Any device can revoke any other (`server.py:481-488`). Both need fixing before approvals travel over WS, and neither does today.

## Decisions for Will

| # | Question | Recommendation |
|---|---|---|
| **B1** | May the global token still **deny**? | **Yes.** Deny is the safe direction, and refusing it would only make a stuck queue harder to clear. |
| **B2** | Existing devices at upgrade: start as **non-approvers**, or grandfather every existing device? | **Start as non-approvers.** Any of them could have been minted by whoever held the global token. Re-allowing the operator's phone is one `/devices allow` from Telegram. |
| **B3** | In a Telegram **private** chat, does the allowed chat id count as the person, with no new config? | **Yes.** In a private chat they are the same number. Group chats need `approver_user_ids`. |
| **B4** | **Open mode** (an explicit empty token): REST approvals refused? | **Yes.** No credential means no person. Telegram still works. |
| **B5** | Ship as **record → warn → enforce**, with enforce flipped only after Beacon desktop enrols as a device? | **Yes.** Steps 1 and 2 break nothing; step 3 is the only breaking change, and it waits on a Beacon release. |

## If ruled yes: the build

1. **Record.** Add `Approver`, the required `by:` keyword, the audit `user_id`, `approved_by`/`granted_by`, and the `approver` device column. The REST route stops parsing ids as chat text.
2. **Warn.** Add `require_person: warn`, plus the surfaces that show who approved.
3. **Enrol.**
   - Setup pairing mints an approver device.
   - `/devices allow <code>` on Telegram and an "Allow" action for approver devices.
   - `oara devices allow` gated on an active write floor.
4. **Beacon desktop** enrols as a device (other repo).
5. **Enforce.** Flip the default, after a Beacon desktop release.
