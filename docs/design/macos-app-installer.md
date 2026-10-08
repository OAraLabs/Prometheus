# Prometheus for Mac: a signed, notarized Prometheus.app

**Status: the build exists and has been run; nothing is published.** The launcher, the assembler, the
verifier and the release workflow are written, tested, and were run for real: a signed build was made
and verified on this Mac. Not done yet: notarization (no credentials on this Mac), the bind and pairing
changes the app depends on (separate PRs), and a run on a clean machine. Boot and unzip timings below are
from a prototype; the size row is from the real build.

## For Will (one page)

### What you decided

A signed and notarized `Prometheus.app`, zipped. No `.pkg`: there is no Developer ID Installer
certificate, and the Application certificate (team 53JM8W47RL) is enough. The app registers its own
daemon as a login item with `SMAppService` from a signed launcher, so no admin password. Beacon
downloads it, checks signature, team and notarization, moves it to `~/Applications` and runs the
launcher once. It is published as a GitHub Release asset with a stable `latest` URL, so oara.ai can
carry a "Download Prometheus for Mac" button next to Beacon's.

### What it weighs and how long it takes

arm64 only. Python 3.12.13 (python-build-standalone) plus every wheel pinned by `uv.lock`, pruned,
thinned to arm64 and precompiled.

| Bundle | Installed | Zip (the download) | Files | Mach-O to sign |
|---|---|---|---|---|
| **base** (what `pip install oara-prometheus` gives) | 187 MiB | **69 MiB** | 5,865 | 27 |
| **base + anthropic, mcp, slack, discord** (recommended) | 224 MiB | **81 MiB** | 9,700 | 37 |
| `[full]` (adds Playwright, voice output) | 591 MiB | 195 MiB | 15,093 | 178 |

**The real build, notarized** (the recommended bundle without PyMuPDF, signed with the Developer ID,
notarized by Apple and stapled; about 2.5 minutes on this Mac including Apple's wait): zip
**63,432,262 bytes (60.5 MiB)**, 150 MiB installed, 9,573 files, 30 Mach-O. Leaving PyMuPDF out is
24 MB off the download. (With PyMuPDF the same build was 87.7 MB.)

For scale, Beacon's dmg is 186 MB. The recommended bundle is inside the 120 MB target. Biggest
parts: PyMuPDF 52 MB, the interpreter about 50 MB before pruning, lxml about 11 MB (after thinning; it
ships x86_64 and arm64 in one file), cryptography 13 MB. Precompiled bytecode adds about 44 MiB
uncompressed and pays for itself: a signed bundle must not be written into at runtime, so it cannot
cache bytecode itself.

| Step | Measured |
|---|---|
| Download, 81 MiB | arithmetic, not measured: 27 s at 25 Mbps, 7 s at 100 Mbps, 1.4 s at 500 Mbps |
| Unzip with `ditto -x -k` | recommended bundle 7.6 to 8.1 s; base 4.4 to 4.8 s. File count drives it, not bytes |
| Fresh `setup --fast`, then daemon to an authenticated `/api/status` 200 (base bundle) | 0.8 to 0.9 s precompiled; 2.2 s cold with no bytecode |
| Setup-mode server (first run, nothing configured) answering `/api/setup/status` (base bundle) | 0.4 s warm, 1.5 to 1.8 s cold |

So after the download the on-device work is roughly 10 s, dominated by unzip. The boot rows were
timed on the base bundle only; the extras import lazily but I have not timed them.

**Not measured, and I will not claim it:** a clean Mac. These ran on this Mac (macOS 15.6.1, Apple
silicon) with a scratch `HOME` and `PATH=/usr/bin:/bin`, unsigned. That proves the runtime needs no
Homebrew and no system Python. It does not exercise Gatekeeper, notarization, quarantine,
`SMAppService`, the Background Items banner, signing time, verify time on ~10,000 files, or a
cross-volume move. Those need the real build and a clean machine.

### What I found that changes the plan

1. **The daemon cannot listen on localhost only today.** `web/launcher.py` defaults both hosts to
   `0.0.0.0`, the setup-mode server hard-codes it (`setup_server.py:870`), and the WebSocket bridge
   binds separately from uvicorn. In the prototype I forced uvicorn to loopback and `lsof` still showed
   `*:8010`. Any fix has to inventory every listener, not just REST.
2. **Nothing on the pairing path exists yet.** Beacon's plan needs per-device credentials, an
   approver role and device scoping first. There is no PR or branch for them (#660 is a design-only
   draft about approvals). The app can be built without them, but its pairing step can only hand out
   the global token until they land. I would not publish a release before then.
3. **On macOS the daemon has neither shell floor.** The prototype logged `bash WRITE floor
   UNAVAILABLE: bwrap is not installed` and `read floor unsupported on this platform (no AppArmor):
   every model-written shell ... can read ~/.ssh, ~/.gnupg and ~/.config/*/*env`. The API token lives
   in `~/.config/prometheus/env`. For a non-technical user that makes the app's default permission mode
   and the owner-device-token work release gates, not polish.
4. **`oara install-service` is wrong in two places.** On macOS it writes a systemd unit. And in
   `cli/service.py` the `FileNotFoundError` branch returns 0 after enabling nothing, which is the same
   lie on any Linux box without systemd. I would fix both.
5. **A bundled LaunchAgent plist is static.** `SMAppService` reads a plist shipped inside the app, so
   it cannot carry the user's home directory. Logs and working directory are set by the launcher at
   start, not by the plist.
6. **A separate bug, not mine:** the startup Doctor reports `config/prometheus.yaml not found` on every
   installed daemon, because it looks under a "repo root" that is the Python lib directory outside a
   checkout. Flagged as its own task.
7. **`unlink` is not an exactly-once claim on APFS.** Measured on macOS 15.6: with eight threads
   unlinking one path, up to seven calls returned success in a single trial. `rename` and
   `open(O_CREAT|O_EXCL)` gave exactly one winner in 300 of 300. "Delete the secret to use it once"
   would have let one secret be used several times; the pairing code claims by rename.
8. **A distributed bundle carries licence obligations a `pip install` does not put on us.** PyMuPDF is
   AGPL-3.0 (or Artifex commercial), python-telegram-bot is LGPL-3.0, certifi is MPL-2.0. **The app ships
   without PyMuPDF** (the default; `--include-pymupdf` opts back in; the manifest says what was left out).
   Without it two things degrade, measured on the notarized bundle: PDF text extraction returns the
   placeholder `[PDF file: name.pdf — install PyMuPDF to extract text]` instead of the text (which tells the
   user to install something an app user cannot install), and large images are sent to the model at full
   size instead of being shrunk first. Word and Excel files are unaffected. LGPL and MPL remain, with
   their texts in the bundle (`THIRD-PARTY-LICENSES.txt` and each package's own licence file)

### Decisions (Will, 2026-10-07: take the recommended option on each)

1. **Pre-release.** The first release carrying the app is published as a normal release, so
   `https://github.com/OAraLabs/Prometheus/releases/latest/download/Prometheus-mac-arm64.zip` resolves
   (every release so far is a Pre-release, which `releases/latest` skips). Publishing is Will's, and it
   waits until per-device credentials and a restrictive default permission mode have landed.
2. **Notarization credentials.** Will runs `xcrun notarytool store-credentials prometheus --apple-id <id>
   --team-id 53JM8W47RL` (it prompts for the app-specific password; it is never seen here). CI needs five
   repository secrets, listed in the header of `release-macos.yml`. Neither exists yet.
3. **A clean machine.** A Tart macOS VM at the acceptance step. The exact image and size are shown and
   Will is asked before it is pulled (about 20 GB of the 48 GiB free).
4. **Scope.** Bind, same-Mac pairing, the app and `install-service` as separate draft PRs. Per-device
   credentials, conversation scoping and the pairing-approval API stay separate.
5. **Service management.** Only the `install-service` fix now. `oara service ...`, doctor reporting
   service state and the "start at login" API come after the app.
6. **Phones.** Left out of v1. Loopback only, and no widening API in this stack.
7. **Extras.** The 81 MiB bundle: base plus anthropic, mcp, slack and discord.

**Decided since (Will, 2026-10-08):** PyMuPDF is left out of the app (below). The icon is the OAra "O."
mark. The clean-machine test is on a real MacBook, not a VM. **Still open:** the signing-team plan around
2027-02-01, and the release gates in the pull request.

## Shape

```
Prometheus.app/
  Contents/
    Info.plist                          bundle id com.oaralabs.prometheus, LSUIElement, min macOS 13
    MacOS/Prometheus                    signed native launcher (Swift, ~one file)
    Library/LaunchAgents/
      com.oaralabs.prometheus.daemon.plist
    Resources/python/                   relocatable CPython + site-packages, bytecode precompiled
```

The bundle is sealed at signing and never written into. State stays where it is today
(`~/.prometheus`, `~/.config/prometheus/env`) so the CLI and the app are one daemon with one set of
files. `PYTHONDONTWRITEBYTECODE=1`; bytecode is compiled before signing as `unchecked-hash`.

### The launcher

One binary, run with arguments by Beacon, with no arguments when a person double-clicks the app. Each
mode prints **one JSON line** and exits with a distinct code, so Beacon reads state instead of polling.

| Mode | Does | Exit codes |
|---|---|---|
| `--register` | registers the LaunchAgent via `SMAppService`; idempotent | 0 ok (`registered` or `already_registered`), 10 `requires_approval`, 11 `already_running` (a Prometheus answers on 127.0.0.1:8005 that is not ours), 12 `port_busy`, 13 `registration_failed`, 14 `unsupported_os` |
| `--run` | what launchd starts: sets paths, redirects logs to `~/Library/Logs/Prometheus/`, then `exec`s the bundled Python with `-m prometheus daemon --bind 127.0.0.1` and `PROMETHEUS_INSTALL_KIND=app` | the daemon's |
| `--unregister` | stops and unregisters the agent; the way to swap the app for an update | 0 / 13 |
| `--status` | `not_registered`, `registered`, `requires_approval`, `running` | 0 |
| `--pair` | writes a fresh pairing secret (re-pair after a Beacon reinstall) | 0 / 13 |
| `--uninstall [--purge-data]` | unregisters, removes the pairing file, trashes the app; keeps `~/.prometheus` unless `--purge-data` | 0 / 13 |
| *(none)* | one small window: running or not, Open Beacon (Get Beacon if absent), Pair again, Uninstall | |

Process names come out right because launchd starts `Prometheus --run` and the bundle identity is the
app's, so Login Items and prompts say "Prometheus", not "python3.12".

### The LaunchAgent

Label `com.oaralabs.prometheus.daemon`, `RunAtLoad`, `KeepAlive {SuccessfulExit: false}` with
`ThrottleInterval 10`, `AssociatedBundleIdentifiers` = the app's bundle id. That is the systemd unit's
`Restart=on-failure` and `RestartSec=10`, restated. The plist is rendered from one function that both
the app build and `oara install-service` call; a test pins the keys they share. Because the label is
the same, launchd itself refuses a second supervisor, and `install-service` detects an
app-owned registration and refuses with a clear message instead of clobbering it.

### `oara install-service` on macOS

On darwin it writes `~/Library/LaunchAgents/<label>.plist` for the current `oara`; `--now` runs
`launchctl bootstrap gui/<uid>`. It exits non-zero, with the reason, whenever it could not do what it
says: no `launchctl`, a failed bootstrap, an existing registration it will not replace. The Linux
no-`systemctl` branch gets the same rule. Without `--now` it does not start anything, matching today's
safety property.

### Listening on this Mac only

A `web.bind` setting, `--bind` on `oara daemon` and `PROMETHEUS_WEB_BIND` (setup mode has no config to
read). Routed through REST, the WebSocket bridge, the setup server, and every other listener found by
inventory. `POST /api/setup/configure` persists it. A config with no `web.bind` keeps `0.0.0.0`, so
your Mac mini reached over Tailscale does not change. When bound to loopback the server refuses a
`Host` header that is not `localhost`, `127.0.0.1` or `[::1]` (DNS rebinding). The app's launcher
passes `--bind 127.0.0.1`. The "let my other devices connect" API is a separate later change.

### Pairing with no typing

Settled with Beacon's side in writing, apart from the items marked as your call.

1. In setup mode, when `PROMETHEUS_INSTALL_KIND=app`, the daemon writes a one-time secret (32 random
   bytes, base64url, one line) to `~/Library/Application Support/Prometheus/pairing/pair.secret`:
   directory `0700`, file `0600`, created exclusively. This is outside `~/.prometheus` on purpose:
   setup mode is documented to create no `~/.prometheus` state.
2. Beacon reads it and calls the **existing** `POST /api/setup/pair` with `{"code": "<secret>"}`. The
   response shape (`token`, `api_base_port`, `ws_port`) does not change.
3. The secret is accepted only when the peer address is loopback **and** the `Host` header is loopback.
4. It stays valid until its first successful use, even if Beacon opens an hour after install, and
   survives a daemon restart. It is used exactly once, claimed by an atomic rename (not `unlink`, finding
   7), and deleted. Wrong attempts on the six-digit path never lock the secret path, and the secret path
   has no global lockout.
5. It is never logged.
6. The credential returned comes from one function. In this PR it is the global token, exactly what
   `/api/setup/pair` returns now. Per-device owner credentials (#696) change that function, not the routes:
   `/api/pair/local` then returns an owner device token, while setup mode keeps returning the global token
   (it can create no device). The credential is the OWNER tier only: a device approved from another device
   gets an ordinary token from its own mint path, never this one. **The swap is the release gate in finding 2.**
   One requirement from device scoping (#692): a device token sees only the sessions it owns, so the first
   Beacon on the Mac must be an owner device, operator-equivalent for scoping, or it would open to an empty
   session list. Approvals are not scoped by #692 either, so "approver" is its own piece of work.
7. A running daemon accepts the same secret at `POST /api/pair/local` so a Beacon reinstall is not a
   dead end. It is written by `--pair`. The route name is chosen to sit beside the other `/api/pair/*`
   routes planned for new devices. A configured daemon has no unauthenticated route today (the bearer
   middleware covers every `/api/` and `/v1/` path), so this route is a deliberate, tested exemption
   with its own loopback checks and an Origin refusal. Exemptions live in one exact allowlist,
   `web/public_routes.py`, shared with the pairing-approval API: a method and a path pattern, never a
   prefix, and a test that fails if any other route answers without a bearer. For the same reason Beacon's "is there already a Prometheus here"
   probe gets a 401 from a configured daemon until a discovery route exists.

### Signing and release

Reuses Beacon's approach: sign by SHA-1 fingerprint (two certificates in this keychain share one name,
so the name is ambiguous), inside-out, every Mach-O with hardened runtime and a secure timestamp,
entitlements starting at none and added only when a failing run proves one is needed. Then
`notarytool submit --wait`, **staple the app**, `ditto -c -k --keepParent`, and verify the *stapled app*
with `codesign --verify --deep --strict`, `xcrun stapler validate` and `syspolicy_check distribution`.
Not `spctl -t exec` on nested executables, which gave wrong answers on Beacon's own release.

Assets on the release: `Prometheus-<version>-arm64.zip`, the alias `Prometheus-mac-arm64.zip` (so the
`latest` URL never changes) and `prometheus-mac.json` (version, sha256, size, team id, bundle id, min
macOS). Beacon pins the team id in its own code and treats the manifest's as display only.
oara.ai gets a button that asks GitHub for the latest release, plus a first-party redirect
(`/download/prometheus-mac`) so the link never depends on GitHub's file naming. Publishing waits for
you.

The first release is built and notarized on this Mac; a `macos-latest` CI job follows with the same
secrets as Beacon's release job. If Apple's notary stalls past the job timeout I stop and tell you; I
do not raise the timeout.

## The contract for a client that installs and pairs the app (Beacon)

What a client may rely on, as built and run. Anything not listed here is not promised.

**Files and names.** Bundle id `com.oaralabs.prometheus`, signed by team 53JM8W47RL (a client should pin a
list of team ids). Launcher: `Prometheus.app/Contents/MacOS/Prometheus`. Agent label
`com.oaralabs.prometheus.daemon`. Release assets: `Prometheus-<version>-arm64.zip`, the stable alias
`Prometheus-mac-arm64.zip`, and `prometheus-mac.json` (version, sha256, size, team id, bundle id, minimum
macOS, the interpreter pin and the lock's hash). The zip is made with `ditto -c -k --keepParent`, holds one
`Prometheus.app`, and the app is stapled. Check, in this order, before moving it anywhere:
`codesign --verify --deep --strict`, the bundle id and team, `xcrun stapler validate`, and
`syspolicy_check distribution`. Do not use `spctl -t exec` on nested executables.

**Launcher modes** (run the executable directly; each prints one JSON line with `ok`, `state`, `detail`,
`agent`, `app_version`, and exits with the code shown):

| Mode | `state` | Exit |
|---|---|---|
| `--register` | `registered`, `already_registered` | 0 |
| | `requires_approval` (the person switched it off in Login Items; the launcher opens that pane) | 10 |
| | `already_running` (a Prometheus that is not ours answers on 127.0.0.1:8005) | 11 |
| | `port_busy` (something else holds 8005) | 12 |
| | `registration_failed` (`detail` says why) | 13 |
| | `unsupported_os` (before macOS 13) | 14 |
| `--unregister` | `not_registered` | 0 (13 on failure). Stops the agent: the first step of swapping the app for an update |
| `--status` | `not_registered`, `not_found` (never registered yet: treat the same), `registered`, `requires_approval`, `running` | 0 |
| `--pair` | `pair_secret_written` (adds `path`) | 0 (13 on failure) |
| `--uninstall [--purge-data]` | `uninstalled`, `uninstall_incomplete`; adds `removed`, `kept`, `failed` | 0 or 13. Moves the running app to the Trash; keeps `~/.prometheus` unless `--purge-data` |

**Is a Prometheus already there?** `GET http://127.0.0.1:8005/api/setup/status` answers
`{"setup_mode": true, "configured": false, "pairing": ..., "version": ...}` in setup mode, with no
authentication. A configured daemon has no unauthenticated route today and answers every `/api/` path with
401 and a JSON `error` that contains both "unauthorized" and "Bearer"; keep both words if that is ever
reworded. A `GET /api/hello` is planned by the pairing-approval work and is not built.

**Credentials.** Written against per-device owner credentials, PR #696 (stacked on the pairing PR, not
merged yet); what is marked *today* below holds without it.

**Pairing, fresh install** (no `~/.prometheus` config, so the daemon starts in setup mode): read
`~/Library/Application Support/Prometheus/pairing/pair.secret` and `POST /api/setup/pair` with
`{"code": "<secret>"}` from the same Mac, over `127.0.0.1` with a loopback `Host` and no `Origin` header.
The answer is `{"token", "api_base_port", "ws_port"}`, the same shape as pairing with the six-digit code,
and the token is the daemon's **global** token: setup mode authenticates its mutations against that token
alone and must create no `~/.prometheus` state, so it cannot mint a device. The secret is used once and
deleted. Then drive setup with `/api/setup/*` and finish with `POST /api/setup/complete`: the same process
becomes the configured daemon. **Then trade the global token for an owner device token** (#696): from a
loopback address, `POST /api/devices` with the global token and `{"owner": true, "name": "..."}` answers
201 `{id, name, platform, token, created_at, owner: true, revoked_previous}`. Keep ONLY that owner token and
drop the global one. *Today* (before #696) there is no such route option and the client keeps the global token.

**Pairing, existing install** (`~/.prometheus` already has a config, so there is no setup mode and no
secret at first start): run `Prometheus --pair`, read the file it names, and `POST /api/pair/local` with
`{"code": "<secret>", "name": "..."}` (`name` optional, default "Beacon on this Mac", clipped to 64
printable characters); same conditions. With #696 the answer is
`{"token", "api_base_port", "ws_port", "revoked_previous"}` and the token is an **owner device token**,
never the global one; `revoked_previous` is an integer. The route answers 404 unless this is the app install.

**Re-pairing revokes the old install's credentials.** Minting an owner device for this Mac revokes the
earlier owner devices minted for this Mac, in the same transaction (those minted by `POST /api/pair/local`
or by `POST /api/devices` with `owner: true` from a loopback peer). An owner device minted from a
non-loopback address, ordinary devices, and the new device are untouched. So an OLD Beacon install's token
now answers 401 and its open WebSocket is closed with code 4401: **treat 401 or 4401 as "re-pair with
`Prometheus --pair`".**

**What an owner device is.** Operator-equivalent for conversation scoping (sees every session, including
older ones and Telegram and CLI sessions), may list and revoke any device. Not root: it cannot enrol other
devices (`POST /api/devices` stays global-token-only) or define MCP servers. A device someone approves from
another device is an ordinary scoped device, minted by a different path, and never gets this credential.

**What is not true yet.** The global token stays valid and stays in `~/.config/prometheus/env`, where the
agent's own bash tool can read it, and on macOS the daemon has neither shell floor. Owner credentials stop a
paired *device* from holding the master key; they do not stop *the agent* from reading it. That is a
separate follow-up (design: keep the master key out of the agent's reach), and the app's release stays
gated on it and on a restrictive default permission mode. Requires-approval, `/Applications`, ad-hoc
signing and logout/login are unproven (see below).

## Order of work

Each is a draft PR, not merged, Auto-fix off. Where it can be tested, the test that fails on today's
behaviour is written and run first, and its red output kept.

1. **`install-service` and the shared plist.** Red first: on darwin with no `launchctl`, exit is 0
   today; the Linux no-`systemctl` branch the same.
2. **Bind.** Red first: with bind `127.0.0.1`, a connection to a non-loopback address of the machine
   on 8005 and 8010 is refused in setup mode and after configure; `Host: evil.example` is refused; a
   config without `web.bind` still binds `0.0.0.0`; a test lists the process's listening sockets and
   fails if any is wider than the setting.
3. **Same-Mac pairing.** Red first: secret from a non-loopback peer 403, wrong `Host` 403, second use
   401, file `0600` in a `0700` directory and gone after use, the six-digit path unchanged.
4. **The app.** Python tests for the plist, the layout and the manifest; a verify script exercised
   against fixtures; the launcher is Swift and is checked by a real signed run on a Mac.
5. **oara.ai button and redirect**, after your OK on the page.

## What has been proven, and what has not

**Proven by running it (2026-10-07):**

- A real signed build from a clean tree passes `verify_app.py`: every Mach-O is Developer ID signed with
  the hardened runtime and a secure timestamp, arm64 only, no entitlements.
- **The hardened runtime needs no entitlements.** The signed interpreter loads all 24 modules tried
  (including `slack_bolt`, `discord`, `anthropic`, `mcp`), runs a `ctypes` callback (libffi closures),
  renders with PyMuPDF and parses with lxml.
- **`SMAppService.register()` works when the launcher is exec'd directly**, from `~/Applications`, signed:
  idempotent, `KeepAlive` restarts a SIGTERMed agent after the throttle, `--unregister` leaves nothing
  behind (no launchd job, no process, no Login Items record). The agent row in Login Items is named after
  its PROGRAM file, which is why the agent runs the launcher.
- `--register` refuses correctly: 11 against a real setup-mode Prometheus, 12 against a plain HTTP server
  and against a look-alike login wall that says "unauthorized" without "Bearer".
- `--pair` writes a `0600` file in a `0700` directory, 43 base64url characters, replaced atomically.
  `--uninstall` removes the pairing directory and the app (moving the app it is running from to the
  Trash works) and leaves `~/.prometheus` untouched.
- **The whole path, with the real signed app and the real daemon** (the launcher's `--run`, in an isolated
  home, not under launchd): setup mode answers in 1.4 s on `127.0.0.1:8005` only, and a connection to the
  Mac's LAN address is refused; a foreign `Host` gets 403 and a browser `Origin` 400; a wrong secret gets
  401 and a right one 200, once; `configure` pins `bind: 127.0.0.1`; `complete` turns the same process
  into the configured daemon on `127.0.0.1:8005` and `:8010` only (authenticated `/api/status` in 1.1 s);
  then `--pair` and `POST /api/pair/local` return the daemon's token and consume the secret. Running it
  also showed the launcher and the daemon disagreed about the home directory (FileManager versus `$HOME`);
  the launcher now uses `$HOME`, like the daemon.

- **Notarized, stapled and accepted by Apple, 2026-10-08.** The `prometheus` keychain profile submitted the
  signed app and Apple accepted it on the first submission. On the zip: `verify_app.py --notarized` passes
  (signature, hardened runtime, timestamp, no entitlements, arm64, icon, stapled ticket); `stapler validate`
  "worked"; `syspolicy_check distribution` "passed all pre-distribution checks"; `spctl -a -t exec -vv`
  says `accepted, source=Notarized Developer ID, origin=Developer ID Application: William Hieber`, both
  without the downloaded-file flag (how Beacon's own download looks) and with the quarantine attribute set
  on every file (what a browser or AirDrop adds). `codesign` reports `Notarization Ticket=stapled`.
- The notarized build, run end to end in an isolated home, behaves exactly like the unnotarized one above.

**Not proven yet:**

- The `requires_approval` path (it needs the item switched off in System Settings), registration from
  `/Applications`, ad-hoc signed builds, and surviving a logout and login.
- That Login Items shows "Prometheus" for the real app (the BTM record was read for the dev-test app).
- A real double-click launch of a quarantined, notarized copy (this is the clean-machine test); everything
  short of the launch was checked (below).
- Anything on a clean machine, and the installed time end to end.
- The daemon started BY LAUNCHD through `SMAppService` (the end-to-end above ran `--run` directly, to keep
  it away from a real home directory). That needs a clean user or VM.
- Reproducibility across two build machines. The build pins the python-build-standalone release and
  sha256, installs from `uv export --hashes` with `--require-hashes --only-binary`, and records the lock's
  hash in the manifest; two runs have not yet been compared.
- The signing certificate that CI holds expires 2027-02-01, and Beacon expects a move to an OAra Labs
  certificate before then. Beacon pins a list of team ids, so a new team can be added in a Beacon release
  before builds signed with it ship. Existing timestamped signatures stay valid.

## One installer for both ("OAra for Mac")

**Recommendation: ship two buttons first.**

- The one-click flow already exists: a person who starts at Beacon clicks "Set up Prometheus on this
  Mac" and gets the same result. A combined installer only saves the person who starts at oara.ai one
  download.
- The release cadences differ (daemon 0.9.x, Beacon 0.3.x) and Beacon replaces itself on update. Bundling
  an 80 MB daemon inside it re-ships the daemon on every Beacon update, or forces both to move together.
- Beacon already pairs with daemons on other machines (a GPU box, a Mac mini over Tailscale). A local
  daemon in every Beacon install is the wrong default for those users.
- Licenses differ: Prometheus is MIT and public; Beacon is closed under its own terms. One bundle needs
  one notice story.
- A combined thing is a thin wrapper over two already-verified pieces. Building it first would delay
  the pieces to ship the wrapper.

One thing the two-button path needs: a Prometheus.app with no Beacon beside it has nothing to drive
setup. So its window must say "Prometheus needs Beacon to finish setup", open Beacon if it is installed,
and link to oara.ai/beacon if not. Beacon, in turn, on launch with no connection configured, can look
for a setup-mode Prometheus on 127.0.0.1 plus the secret file and offer "Connect to Prometheus on this
Mac"; that needs no deep link into Beacon.

Revisit after the clean-machine run, if people drop out between the two downloads. The shape would then
be a dmg holding both apps (works with the Application certificate), not a `.pkg`.

## How the numbers were taken

Prototype in a scratch directory, not in the repo. `uv python install 3.12` (arm64), dependencies from
`uv export --frozen` of the repo's `uv.lock`, the wheel from `uv build`, installed with `--target`.
Pruned: pip, ensurepip, idlelib, tkinter, turtledemo, lib2to3, Tcl/Tk, headers, `__pycache__`. Thinned
seven universal2 files (lxml) with `lipo -thin arm64`, saving 8.4 MiB. Compiled with `compileall
--invalidation-mode unchecked-hash`. Zipped with `ditto -c -k --keepParent`. Boot timings: spawn
`python -m prometheus daemon` with `env -i`, a fresh `HOME`, loopback forced by a throwaway
`sitecustomize` (so an unsigned interpreter does not raise the macOS firewall dialog), poll until HTTP
200, stop the process by its recorded PID. Imports checked under that environment: prometheus, fastapi,
uvicorn, websockets, pydantic_core, cryptography, lxml, pymupdf, yaml, watchdog, httpx, telegram, docx,
openpyxl, sqlite3, ssl, ctypes. All 27 Mach-O files in the base bundle are ad-hoc signed with no team id,
so every one is re-signed.
