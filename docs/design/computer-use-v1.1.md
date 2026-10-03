# Computer use v1.1 — the door, the cockpit, and the driver as an Integration

**Status: DESIGN PROPOSAL. Nothing in this document is implemented.** No
behaviour, config, test or registration changes ride with it.
`register_computer_tools` still has no call site and `computer.registered`
is still 0.

Written 2026-10-03 against `origin/main` @ `856ebb8`. The computer-use module
(`src/prometheus/computer/`) last changed in #527 (`c7e26aa`). Line numbers
are at that commit.

## How to read this

Every statement in this document is one of three kinds, and it says which:

* **Settled (plan).** Decided in `PLAN-COMPUTER-USE-AND-SKILLFORGE-2026-09-21`,
  **as summarized by Will** when this design was commissioned. The plan doc
  and its three companions (`MILESTONE-3-ADDITIVE-CHOOSER-ARCHITECTURE`,
  `COMPUTER-USE-WINDOW-IDENTITY-2026-09-20`, `EXTENSION-TAXONOMY`) were not
  available to the author. Where this document relies on them, it relies on
  that summary, not on inference. §1 lists every settled item.
* **Measured / read.** Cited as `path:line` at `856ebb8`, a path inside the
  installed `cua-driver` 0.28.2 wheel, or a URL that was fetched. Code that was
  run says so.
* **Proposed.** This document's own recommendation. Each one is marked, and
  the ones that need a decision are collected in §8.

Where evidence qualifies a settled decision, §1.2 says so explicitly rather
than quietly designing around it.

---

## 0. Summary

<!-- SUMMARY: filled last -->

---

## 1. Settled inputs (from the plan, summarized by Will)

### 1.1 The decisions

| # | Decision (PLAN-COMPUTER-USE-AND-SKILLFORGE, 2026-09-21) |
|---|---|
| 1 | Computer use is **started by the user**, not reached for by the model, at first. Model-initiated can come later without rework. |
| 2 | **One `computer_task(goal)` tool**, not seven verbs. A registered verb goes straight to `driver.act` and skips the candidate table and `validate_choice`. |
| 3 | **Local only.** Nothing leaves the machine. |
| 4 | The cua driver is an **Integration, never a builtin**: lifecycle, health known before dispatch, supervised like the backend registry. |
| 5 | **Daemon core, non-swappable:** the security gate, the consent extent, the candidate table, `validate_choice`, `ComputerUseLoop`. |
| 6 | **Beacon:** the session toggle IS the scope grant (picking the app is the consent), plus a live action log and a stop control. The log matters more than an overlay cursor, because the common case is a phone watching a machine you're not at. |
| 7 | **Beacon IA:** merge Connectors and Integrations. |
| 8 | **Don't port SkillForge's recorder** for desktop work. Watch mode on our own observe replaces it. |

Already ruled, not re-argued here:

* cua's `browser_*` subtree stays excluded (`actions.py:36-39`): it is an
  exfiltration primitive, and Prometheus has a browser tool.
* `pid` and `window_id` stay out of the extent. Observation is app-scoped
  (measured), and a fifth term silently invalidates grants.
* No shared element model across DOM / AT-SPI / pixels: `element_token` dies
  with the snapshot on purpose.

v1.1 order (settled): (1) answer platform pass-through and events; (2) the
door; (3) the Beacon action log and stop control, required before anyone
watches a cursor move; (4) the local chooser, Gemma + GBNF first; (5) watch
mode, then recorded-step-as-query; (6) split Record a Skill out and point
SkillForge's post-processing (parameterize, synthesize, quality gate) at the
new producer. Later: post-action capture into `VerifyInput`, the OmniParser
pixel fallback, per-run perception metrics.

Known open (settled): hazard (b), `_snapshots` never pruned after pid reuse
(`cua.py:134-158`), is deferred. The origin term must be decided while
`computer.registered` is 0, because it is free only until real grants exist.

On config (settled): fold the `computer:` block into the existing
`computer_use:` name if that is cleaner, and propose it either way (§5.3.4).

### 1.2 Where the evidence qualifies a settled decision

None of the evidence refutes a decision. Seven findings qualify one, and each
changes what the design must do. They are stated here, before the design, so
that nothing below quietly works around a settled item.

**Q1. Decision 3 (local only) is not true of the parts we would use, as they
ship.**

* **cua-driver sends telemetry by default, from the binary, not the library.**
  * Telemetry is on by default and content-free, sent to PostHog EU. It lives
    in the CLI **binary** crate (upstream `crates/cua-driver/src/telemetry.rs`).
  * The bundled `bin/cua-driver` contains `https://eu.i.posthog.com/capture/`.
  * `cua-driver doctor` reports `telemetry: enabled via default`.
  * Running the CLI here created `~/.cua-driver/.telemetry_id`, and the egress
    proxy refused its CONNECTs to `eu.i.posthog.com`.
  * **The in-process library we load today** (`CuaDriver.create()`,
    `cua.py:178`) contains no PostHog URL and no telemetry variable names, in
    0.28.2 or 0.33.1. That is a strings check, not a packet capture, so it is
    *probably* clean and *not established*.
  * **The private worker (§4.1) *is* the binary.** It registers telemetry
    before it starts, and its environment allowlist passes
    `CUA_DRIVER_RS_TELEMETRY_ENABLED` but **not** `DO_NOT_TRACK` (upstream
    `crates/cua-driver-sdk/src/embedded.rs:837-870`, `worker.rs:141-158`).
  * The update check is separate from telemetry, and unaudited.
  * Nothing in `src/` sets any opt-out.
* **Cua Bench and Lume send telemetry by default too.** Cua Bench uses
  cua-core (`CUA_TELEMETRY=0` or `DO_NOT_TRACK=1`); Lume uses
  `LUME_TELEMETRY_ENABLED`.
* **Record a Skill's step verifier can leave the box.**
  * It is built from the top-level `model:`, which may be a cloud provider
    (`web/server.py:4029-4064`).
  * It is on by default (`docs/guide/record-a-skill.md:33-37`).
  * Watch-mode output (§5.6) must not be pointed at it until it is restricted
    to local providers.
* **APNs approval pushes carry the app and the machine today.**
  * The alert body is `"<tool_name> — <first line of description>"`
    (`push/dispatcher.py:100-117`).
  * A computer action's description names the app and the target
    (`checker.py:1137-1140`).
  * So with `push.enabled` (off by default), both would pass through Apple.
    This contradicts the iOS extension's own header ("Apple learns that the
    daemon had something to say, never what", beacon-ios
    `NotificationService.swift:4-7`).
  * Computer pushes must be content-free (§5.2.2).

*Consequence:* the Integration sets `CUA_DRIVER_RS_TELEMETRY_ENABLED=0` and
`CUA_TELEMETRY_ENABLED=0` before the SDK loads **and** in the private worker's
environment, and the health check reports that it did (§5.3). These are a floor, not a config
key: a key that could turn telemetry on would contradict decision 3.

**Q2. Decision 6 (the toggle IS the grant) has no mechanism today, and needs
four rulings to be buildable.**

* **No grant is scoped to a session.** The only scopes are `until_restart`
  and `persistent`, and `until_restart` lives across every session and
  surface, because there is one gate per process
  (`permissions/checker.py:333-345`; `"session"` survives only as an alias,
  `approval_queue.py:84-97`).
* **Picking an app names no verb and no delivery**, so the toggle cannot be
  one existing grant. Grants are exact whole-value matches
  (`checker.py:391-409`).
* **Payload verbs are never rememberable** (`computer_extent.py:70-78`,
  `approval_queue.py:193-194`). The toggle therefore cannot cover typing
  unless that rule changes. This design keeps the rule: every `type_text`
  still prompts.
* **For a browser or Electron app, the app is every site**, including
  signed-in ones (§5.4).
* **Once the chooser is a model (v1.1 step 4), every click the toggle covers
  is chosen from app-displayed text.** The 2026-09-20 audit advised against
  combining remembered grants with model-chosen targets
  (`docs/audits/COMPUTER-USE-REGISTRATION.md:79-83`).

  This design reads decision 6 as accepting that risk, inside the one app the
  person picked and for the life of the toggle. It adds the mitigations in
  §5.1.6 rather than reversing the decision. §8 asks for confirmation.

**Q3. Decisions 1 and 2 (one user-started `computer_task`): the status block
would report it; the pin test would not catch it.**

* **The status block is correct.** `computer.registered` counts every registry
  tool whose name starts with `computer_` (`computer/status.py:79,177`). A
  registered `computer_task` would show as 1, however it got there.
* **The gap is narrower, and it is in the pin test only.**
  `test_the_daemon_registers_none_today` looks only for call sites of the
  literal `register_computer_tools(`
  (`tests/test_computer_status_block.py:324-343`). Registering `computer_task`
  any other way would leave that test green.
* **So the door PR makes the pin cover every `computer_*` registration path**
  (§6.1), not one function name.

In v1.1 the door is a **command, not a registered tool** (§5.1). Its
schema is the future tool's schema, so registering it later is a wrapper
with no rework. That later PR changes the ruling and is flagged in §6.

**Q4. Decision 4 ("supervised like the backend registry") holds for health
and refusal, not for lifecycle.**

* The backend registry "reports what a box serves; it never changes it…
  Prometheus does not launch it" (`providers/backends.py:19-26`). It has no
  start, restart or circuit breaker.
* Start/stop therefore copies `McpRuntime` (`mcp/runtime.py:128-306,
  396-405`).
* **The runtime is loaded inside the daemon process** (`cua.py:109-116`), so a
  native crash cannot be contained by any probe. Crash isolation would need
  the SDK's `PRIVATE_WORKER` or `DAEMON` execution mode
  (`_native.py:5074-5085`), which has not been evaluated (§8).

**Q5. The known-open item ("the origin term is free only until real grants
exist") is confirmed, with one qualification.**

* **Confirmed:**
  * `from_config_dict` drops any stored row whose term count is not
    `EXTENT_TERMS`, logging only a warning (`checker.py:452-476`).
  * The target-term precedent says terms go in "before anything is stored"
    (`computer_schema.py:47-53`).
* **Qualification:** unlike the target term, a site term has a safe padding
  value (`-`, "no web content", which never matches a web call). A later
  migration is therefore *possible*, but only by reversing the
  refuse-don't-pad rule. Doing it now is cheaper, not strictly the only
  option.
* **Also:** `computer.registered == 0` is a proxy for "no grants exist", not
  proof.
  * #527 minted a real persistent grant on the box through `/approve
    always`.
  * Its probe wrote to a throwaway config
    (`scripts/computer_use_grant_probe.py:38,58,148-149`).
  * The live `security.grants` should still be checked for `computer_action`
    rows before the term change lands.

**Q6. The already-ruled "a fifth term silently invalidates grants" does not
forbid a site term.**

* The ruling's *reason* is the cost of adding a term after grants exist.
* That cost is zero while `computer.registered` is 0, which is exactly the
  known-open item.
* So the fifth term recommended in §5.4 (`site`) is consistent with both,
  **provided it lands before the door**.
* `pid` and `window_id` stay out for their own reasons: unstable, unreadable,
  and not a real narrowing (`computer_schema.py:146-178`).

**Q7. Decision 8 holds for desktop work, and watch output is weaker evidence
than what it replaces for the web.**

* Polling observe sees **effects** (values, selection, structure, titles),
  not **causes**.
  * cua 0.28.2's typed elements have no focus field ("focus" has 0 hits in
    `_native_contract.py`).
  * The driver's only push feed cannot see the human (§2.2).
* So watch-mode steps belong in the **draft (human-review) tier**, as video
  does.
* The SkillForge Live DOM extension remains the better producer for web work.

Separately from the decisions, one measured defect blocks the door outright:
**under `/gate off` the gate allows every desktop action, and the loop has no
override** (§3, D1).

Will verified it at `checker.py:1026-1037` on 2026-10-03: the mode returns
allow before the computer rule. Fixing it is a **hard precondition for the
door**.

---

## 2. Part 1 — the open questions

### 2.1 (a) Does `CuaDriverAdapter` pass macOS/Windows through, or is it bound to Linux AT-SPI?

**Answer: the SDK and the adapter's calls are cross-platform. The module as a
whole only works on Linux/X11, because two layers in front of the driver are
Linux-shaped. The pin is `cua-driver>=0.28,<1`, locked at 0.28.2.**

| Layer | Platform posture | Evidence |
|---|---|---|
| SDK (`cua-driver` 0.28.2) | **Cross-platform.** Five wheels: `macosx_13_0_universal2`, `manylinux_2_31_x86_64`, `manylinux_2_31_aarch64`, `win_amd64`, `win_arm64`. The native loader branches by OS (`.dylib`/`.dll`/`.so`). There is a `Platform` enum and macOS TCC helpers (`current_mac_os_permission_status`, `request_mac_os_permissions`). The Linux build carries native Wayland input (portal/libei). | `uv.lock:1067-1076`; `cua_driver/_native.py:449-470`; `_native_contract.py:6483`; `_native.py:4442,8183,8225` |
| Adapter calls | **Platform-neutral.** Typed inputs only, no `sys.platform` branch anywhere. The only Linux text is an error message. | `computer/cua.py:94-106,178,222-238,423-462`; `cua.py:184-189` |
| Preconditions | **Linux/X11 + AT-SPI only, and this is the gate.** The act half needs a local `DISPLAY` socket that accepts a connection. The observe half asks `gdbus` for `org.a11y.Bus.GetAddress`. There is no darwin, win32 or Wayland-only branch. On macOS and Windows both halves come back unavailable/unknown, the rollup is `unknown`, and every step is `blocked` before the driver is reached (simulated). A GNOME Wayland session probes the Xwayland socket, which is not the route Cua uses there. | `computer/driver.py:158-227,230-276,328-362`; `loop.py:105-115` |
| Candidate building | **AT-SPI role names.** `_CLICKABLE_ROLES`/`_EDITABLE_ROLES` are AT-SPI spellings. Upstream macOS returns AX roles (`AXButton`), which match nothing, so only the three key candidates are offered. Upstream Windows returns UIA control types: `Button` matches; `Edit` fields are never offered for typing while static `Text` is. | `computer/candidates.py:42-49,81`; trycua/cua `platform-macos …/get_window_state.rs:947`, `platform-windows …/uia/mod.rs:1047-1090` (main, not the 0.28.2 tag) |
| Consent extent | **Platform-neutral** terms. The `app` term comes from the driver's `app_name`; on macOS a stable `bundle_id` exists and is unused. | `computer_extent.py:53-68`; `cua.py:265`; `_native_contract.py:1480` |
| Probes / scripts | GNOME default apps; all real-run probes go through preconditions. | `scripts/computer_use_grant_probe.py:253,256`; `computer_use_multistep_probe.py:246` |
| CI | **The extra is never installed.** Every leg syncs `web anthropic mcp`, including the macOS leg, so the computer tests pass on Darwin only because they use fixtures. | `.github/workflows/ci.yml:92,148,188,249,133-158` |

**macOS also has a process-identity question.**

* `CuaDriver.create()` loads the runtime into the Python daemon, so TCC
  attribution goes to the host process.
* Upstream treats a raw daemon with no stable bundle identity as unsupported.
  Its supported routes are:
  * the `CuaDriver.app` daemon reached with `connect()`;
  * an `EmbeddedCuaDriverHost` started from an app that owns the grants.
* That choice belongs to the Integration (§5.3) and is listed in §8.

**The pin.**

* `computer = ["cua-driver>=0.28,<1"]` (`pyproject.toml:97-105`), locked at
  **0.28.2** (`uv.lock:1067-1068`).
* PyPI's latest is **0.33.1** (2026-10-03), ten releases later. A plain relock
  would jump five minors on a leg CI cannot exercise.
* §5.3.1 proposes an exact pin plus a runtime version check.

**What macOS/Windows would need** (not v1.1 work, listed so v1.1 does not
block it):

1. A per-platform precondition provider inside the Integration: Linux X11 +
   AT-SPI as today plus a Wayland route; macOS TCC status; Windows
   interactive session.
2. Role normalisation in the adapter (AT-SPI / AX / UIA → one vocabulary), or
   selection by `actions` rather than role strings.
3. A host-identity decision on macOS.
4. Delivery-mode honesty per verb (§3, D5).
5. Per-platform probes.

None of these touches the gate, the extent, `validate_choice` or the loop.

### 2.2 (b) Can the driver or AT-SPI/AX deliver events a watch mode could subscribe to?

**Answer: from the driver, only events about its own actions. The human's
focus and activation can only be polled through Cua. The OS accessibility
stacks do push focus and activation events, but using them means a new
per-platform listener of our own, and every one of them is global.**

| Source | Push events? | Reachable today? | Caveats |
|---|---|---|---|
| cua-driver `DriverActivityObserver` | **Yes**, for the driver's **own** calls only: `AUTHORIZED_ACTION`, `AUTHORIZATION_REFUSED`, `ACTION_FAILED`, `GRANT_ISSUED/REVOKED`, `SESSION_STARTED/ENDED` | Only at construction (`create_configured_with_activity_observer`). We call plain `create()`, so it is not installed. | "A content-free lifecycle event … never carries arguments, page text, paths, typed input, images." Callbacks arrive on native threads. Measured: 8 SDK calls gave 10 events, all about the driver's calls. (`_native.py:2479-2493,2577-2592,5741-5745,7286-7295`; `cua.py:178`) |
| cua-driver request/response (`list_windows`, `list_apps`, `get_window_state`, `get_desktop_state`) | **No, state only** | Yes, by polling | Frontmost is the max `WindowInfo.z_index`. `AppInfo.active` is documented "always false" on Linux. Nothing in `src/` calls `list_windows` (`cua.py:147-157`). |
| cua-driver internals | Yes: AT-SPI `StateChanged`/`ActiveDescendantChanged` on Linux, `NSWorkspaceDidActivateApplication` on macOS | **No**, private | Could be asked of upstream; not callable. |
| AT-SPI2 (Linux) | **Yes**: `focus:`, `object:state-changed:focused`, `window:activate/deactivate/create/close` | Needs our own listener on the a11y bus | Device/mouse events are X11-only. A listener sees every app. |
| macOS AX + NSWorkspace | **Yes**: `kAXFocusedUIElementChanged`, `kAXFocusedWindowChanged`, `kAXApplicationActivated` (per app); `didActivateApplication` (global) | Needs our own listener | AXObserver is per-pid (no system-wide feed), needs a run loop and Accessibility trust. |
| Windows UIA / Win32 | **Yes**: UIA focus-changed (system-wide, cannot be narrowed); `SetWinEventHook(EVENT_SYSTEM_FOREGROUND)` | Needs our own listener | Non-UI thread / message loop required. |
| Prometheus today | **None** | — | The loop polls four times per step: preconditions, observe, act, re-observe (`loop.py:109,121,211,251`). |

**What it means.**

1. **The activity observer is at most a cross-check, never the log, and
   never watch mode.**
   * Constructing the driver with it gives a push feed of every driver call.
   * Its rows carry no content, so the log comes from our `StepResult`.
   * It is unavailable in the private-worker hosting this design recommends
     (§4.1, §5.3.3).
2. **v1.1 watch mode polls.** It uses `list_windows` for open, close, title
   and frontmost, and `get_window_state` on the granted app's frontmost window
   (§5.6).
3. **OS listeners are deferred behind a decision (§8).** Each is global: it
   would observe apps outside the consent grant, including their window
   titles, so events for other apps would have to be dropped before anything
   is logged. Asking upstream to expose the trackers the driver already runs
   may be cheaper than writing three listeners. That request does not break
   "we don't rebuild what Cua ships", because Cua does not ship this
   publicly.

### 2.3 (c) What does SkillForgeRecorder record, and how? (plan open question 6)

**Answer: screen video only, from a macOS menu-bar app. It captures no
input events and no accessibility data, and it uploads to a cloud endpoint.
Prometheus derives nothing from it. That supports decision 8.**

The repo has one commit, `9bfc86f` (2026-02-19). Paths below are under
`skillforgerecorder/SkillForgeRecorder/`.

| | Finding | Evidence |
|---|---|---|
| **What** | A macOS 14 menu-bar app (`LSUIElement`) that records **screen pixels** to MP4: H.264 High plus AAC, 60 fps / 10 Mbps by default. A JSON sidecar of file-level stats goes alongside. | `Services/ScreenRecorder.swift:67-95`; `Models/Preferences.swift:20-34,62`; `App/Info.plist:25-26`; `project.pbxproj:404,460` |
| **How** | ScreenCaptureKit `SCStream` → `AVAssetWriter`. It always uses the first display. Nothing is excluded from capture, so the app's own HUD ends up in the video. | `ScreenRecorder.swift:1-4,27-60,217-246` |
| **Modes** | Full screen; region (a drag overlay); window. Window mode has **no picker**: it takes `content.windows.first(where: isOnScreen)`. OCR throws `invalidMode`. Webcam and microphone toggles only set metadata flags; nothing captures from either device. | `ScreenRecorder.swift:36-47,84,191-192`; `Models/RecordingMode.swift:3-7` |
| **Hotkeys** | Carbon `RegisterEventHotKey`, fixed ⌘⇧ R/W/F/T/O/S/P | `Services/HotkeyManager.swift:29-41,53,62,91` |
| **Not captured** | **No input events.** 0 hits for `CGEvent`, `NSEvent`, `addGlobalMonitor`, `IOHID`. **No accessibility.** 0 hits for `AXUIElement`, `AXObserver`, `AXIsProcessTrusted`. **No app or window identity** (`frontmostApplication`, `CGWindowList` absent). No per-step timestamps. Accessibility and Input Monitoring are never requested. | grep over all 26 `.swift` files; `Services/PermissionManager.swift` |
| **Sidecar** | `SkillForgeMetadata`: id, file, duration (whole seconds), resolution (`"0x0"` outside region mode), codec (hard-coded), fps, `capturedAt` (taken at **stop**), mode, flags, `tags: []`, `notes: nil` | `Models/CaptureItem.swift:43-57`; `ScreenRecorder.swift:179-203` |
| **Where it goes** | Manual "Forge Skill" button. A Supabase password-grant JWT, then a multipart `POST {apiURL}/api/upload` with **the MP4 only** (the sidecar is not sent), to skillforge.sh. Config lives in a gitignored `Constants.swift` (from `Constants.swift.template`), so a fresh clone does not build. App sandbox is off. | `Services/SkillForgeIntegration.swift:59-95,139-169`; `App/SkillForgeRecorder.entitlements:5-16`; `README.md:15-28` |
| **Maturity** | Trim, export conversion, OCR, webcam, HUD stats and auto-upload are stubs (for example, trim's `onApply` discards both times). The vault says it was never used to make a recording; the raw chat shows it launched once, then looped on the screen-recording permission. SkillForge itself was deferred on 2026-04-17. | `Views/PostRecording/PostRecordingPreview.swift:46-47`; oara-brain `wiki/sources/projects/SkillForgeRecorder.md:22`; `raw/claude-chats/2026-02-14-…:4784-4940`; `wiki/sources/projects/SkillForge.md:109-114` |
| **Prometheus use** | **None as a format.** No hit for its file naming or sidecar fields in `src/`, `docs/` or `tests/`. Its MP4 would go through the generic `learning/video_ingest` path like any recording (that path came from the SkillForge *engine*, not this app). | `src/prometheus/learning/video_ingest/__init__.py:1-21`; `pipeline.py:59-66` |

**Implication for v1.1 (decision 8: confirmed, with one qualification).**

* There is nothing to port. It is macOS-only and pixel-only, it captures no
  cause data, and it ships its one output to a cloud endpoint (decision 3).
* Watch mode on our own observe replaces it as a desktop **step** producer.
* It does not replace it as a **pixel or narration** source, and nothing in
  v1.1 needs one. If that changes, an OS-native recording fed to
  `video_ingest` (drafts tier) serves without reviving this app.

### 2.4 (d) The tool catalogue the anti-bloat rule is calibrated against (plan open question 4)

**Answer: 51 tools registered. 12 are sent to the model each turn on a local
tier, and all 51 on a cloud tier. Seven computer verbs would add as much
schema as the whole local per-turn set; one `computer_task` adds about 6%.
In v1.1 the door adds zero, because it is a command, not a tool.**

The rule's own text is not in the repo. Here is the catalogue, measured **by
execution** at `856ebb8`.

* **How the measurement mirrors the daemon:**
  * it builds the registry the way the daemon does
    (`build_tool_registry`, `daemon.py:952` → `__main__.py:137-396`);
  * it applies `DynamicToolLoader` (`daemon.py:956`;
    `context/dynamic_tools.py:206-245`) and the profile filter
    (`engine/agent_loop.py:1609-1656`);
  * config is the shipped template, with no MCP servers and LSP off.
* A no-config run gives the same 51 and the same 12 names.
* The script is in Appendix A.

| Set | Tools | Schema chars (compact JSON) | ≈ tokens (chars/4) |
|---|---|---|---|
| Registered | **51** | 41,253 | 10,313 |
| Sent per turn, **local** tier (deferral `auto` → on) | **12** | 11,306 | 2,826 |
| Sent per turn, **cloud** tier (deferral `auto` → off) | 51 | 41,253 | 10,313 |
| Deferred when on (`tool_search` / exact name) | 39 | 29,948 | 7,487 |
| *+ the 7 `computer_*` tools* | 7 | **11,578** | 2,894 |
| *+ one `computer_task(goal, app?, text?)`* | 1 | **675** | 169 |

* **The 12 sent on a local tier:** `bash`, `task_create`, `read_file`,
  `write_file`, `edit_file`, `grep`, `glob`, `tool_search`, `skill`,
  `web_search`, `web_fetch`, `memory` (`config/shipped_defaults.py:44-61`).
  They are pinned to follow the default by
  `tests/test_always_loaded_follows_the_default.py`.
* **Token counts are chars/4 only.** `tiktoken` is not installed, and the
  proxy refused its encoding download.

**What the increment means:**

* **Seven verbs:**
  * They come to 11,578 characters, which is *larger than the entire local
    per-turn set*.
  * If they were always loaded, a local turn's catalogue would grow by 102%.
  * On a cloud tier they add 28% whether or not they are marked deferred,
    because `auto` sends everything to cloud tiers
    (`dynamic_tools.py:229-230,243-245`).
  * `computer_type_text` alone (2,168 characters) would be the third-largest
    schema in the catalogue.
* **One `computer_task`:**
  * It is 675 characters: +6.0% of the local per-turn set if always loaded,
    and +1.6% of the full catalogue.
  * Decision 2 is about 17× smaller in schema terms, besides keeping the
    table and `validate_choice` in the path.
  * When L1 registers it, it should be **deferred** (reachable by
    `tool_search`), not always loaded. That costs 0 per local turn and 675
    characters per cloud turn.

**What the rule was calibrated against: "was" figures stated in the tree,
vs now.**

| Stated | Where (date) | Now |
|---|---|---|
| "advertises 8 of 51 tools" | `tests/test_tool_advertisement.py:4` (2026-08-11) | 12 of 51 (`skill` added in #593, 2026-09-26) |
| "offered 11 tools" | `daemon.py:984` (2026-08-12) | 12 |
| "49 schemas, ~9.6k tokens, 60.7% of round 0" | `engine/agent_loop.py:1589-1590` (2026-07-31) | 51 schemas, ≈10.3k tokens |
| "all 55 schemas … 34,020-token request … 40 withheld" (live box) | `engine/agent_loop.py:1580-1582` (2026-09-11) | Shipped default 51/39. The live box adds MCP and other tools; not re-measured. |
| Advertised set "~74% below the full catalog" | `config/prometheus.yaml.default:231-236` (2026-08-11) | 72.6% below (by characters) |
| Deferral drops "~8k tokens" | `context/dynamic_tools.py:221`; `README.md:153` | ≈7.5k |
| Banner "37 tools" | `setup_wizard.py:282` (2026-04-20) | 51 |

**No numeric cap exists anywhere.**

* `AgentProfile.max_tool_schemas` exists (`config/profiles.py:36,273-274`),
  but no built-in profile sets it.
* `tool_search` returns a top 5 (`tools/tool_search.py:237`).
* The only enforced rule is a process rule. `tests/test_tool_advertisement.py`
  requires every tool to be advertised or to have a tested discovery path.

**If the anti-bloat rule is a number, it is calibrated on an 8-of-51 or
11-tool advertised set and a 49-55-schema catalogue. Today it is 12 of 51.**

---

## 3. Defects found on the way

None of these was asked about. Each was read from source and, where marked,
measured. Each is assigned to a PR in §6. None is fixed by this document.

| # | Defect | Evidence | Why it matters for v1.1 | Fixed in |
|---|---|---|---|---|
| **D1** | **`/gate off` allows every desktop action, and the loop has no override.** In `PermissionMode.AUTONOMOUS` the gate returns ALLOW before reaching the computer rule. `agent_loop` forces a prompt for an *unknown* extent in its own path; `ComputerUseLoop.step` has no such override. | `checker.py:1026-1037` runs before `checker.py:1130-1150`; `engine/agent_loop.py:4991-5008`; `computer/loop.py:173-207`. **Measured:** `SecurityGate(mode=AUTONOMOUS).evaluate(…)` returns `allowed=True, requires_confirmation=False` for both a known and an unknown computer extent; DEFAULT returns `False, True` for both. | A door on today's loop would click and **type** unprompted under `/gate off`. The payload rule would be bypassed too. | PR 3. **Hard precondition for the door** (verified by Will, 2026-10-03). |
| **D2** | **Typing goes to whatever has focus, not to the element the prompt names.** A `type-N` candidate says "Type the prepared text into the entry 'Search'" and carries that element's token. The typed SDK's `TypeTextInput` takes only an `ActionTarget`, whose variants are `WINDOW` and `DESKTOP`. The adapter drops the token. | `candidates.py:91-107`; `_native_contract.py:1657-1705,5862-5865`; `cua.py:451-455` | The approval sentence describes an action the driver is not asked to perform. That is a consent-honesty defect, not a bug in a corner. | PR 3: type candidates focus-then-verify, or are withheld |
| **D3** | **Non-click verdicts never raise.** `click` returns `ActionResult`, while `press_key`/`scroll`/`type_text`/`invoke_menu` return `ToolResult`, whose effect sits at `.action.effect` beside `is_error`/`error_code`. `_effect_name` reads only `result.effect`. | `_native.py:5568,5598,5624,5628,5652,4632-4646`; `cua.py:399-403`. **Simulated** with real SDK types: `ToolResult(is_error=True, error_code="background_unavailable", action.effect=SUSPECTED_NOOP)` maps to `UNVERIFIABLE`, `landed=False`, and does **not** raise. | `SUSPECTED_NOOP` was meant to raise (`cua.py:47-63`). On four of five verbs it cannot. | PR 1 |
| **D4** | **`editable` is always False.** The adapter reads `getattr(e, "editable", False)`, but `WindowElement` has no such field. The test fake supplies one, which hides it. | `cua.py:360`; `_native_contract.py:6101-6117`; `tests/test_cua_adapter.py:148-151` | Type candidates depend on role names alone, which are AT-SPI spellings (§2.1). | PR 1 |
| **D5** | **The `delivery` term is asserted, not enforced, for four verbs.** In 0.28.2 only `ClickInput` takes a delivery mode. Upstream documents that macOS `invoke_menu` activates the window. | `cua.py:413-462`; `_native_contract.py:4778,5863,4953,3697` | `…:press_key:background` names a property the driver is not asked to honour. That is harmless on Linux X11, but it is untrue as a consent sentence. | PR 1 (record it as driver-decided) |
| **D6** | **The driver's own warnings are dropped.** `WindowStateOutput` carries `degraded`, `degraded_reason`, `truncated`, `elements_complete`; elements carry `in_web_content`, `selected`, `enabled`, `parent_index`. `Observation`/`Element` keep none of them. | `_native_contract.py:6311-6336,6101-6117`; `cua.py:245-271,352-361`; `types.py` | A truncated or degraded tree is treated as complete: the "shaped like success" failure this module exists to refuse. `in_web_content` is the signal the site term (§5.4) needs. | PR 1 |
| **D7** | **The chooser is called synchronously inside an async step.** | `loop.py:136-137`, next to the stall comment at `loop.py:118-122` | Harmless for `RuleChooser`. A network chooser would stall the daemon loop for up to its timeout: the #416 shape. One-line fix, core but additive. | PR 10 |
| **D8** | **No discovery path.** `list_windows` and `list_apps` exist in the SDK and nothing calls them; callers must already know pid and window id. | `cua.py:147-157`; `_native.py:5602,5618` | The door has to resolve "my editor" to a window. | PR 4 |
| **D9** | **Record a Skill's archive is written unredacted, and its funnel is browser-shaped.** `_archive_upload` writes `events.json` raw and never calls `redact_capture`. **Measured:** three URL-less desktop actions in three apps pass `app_consistency` with "No app data to check", and are titled "Web - Enter Data" with "Captured with the live recorder browser extension". | `learning/live_recorder/service.py:218-238`; `security/log_redaction.py:190-214`; `quality_gate.py:149-150,253`; `synthesizer.py:192-193,219` | Watch mode's producer (plan step 6) would inherit both. | PR 14 |
| **D10** | **An empty tree still yields three actions.** `observe` treats any snapshot with an id as usable, and `build_candidates` always appends `key-return`, `key-tab` and `key-escape`. A zero-element observation therefore gives a 3-row table, and the loop's "abstained" branch never fires. | `cua.py:245-271`; `candidates.py:109-118`; `loop.py:129-133`. **Measured:** `Observation(elements=(), snapshot_id="snap-1")` → 3 candidates. | With a remembered `…:press_key:background` grant, Return could be pressed into a window we cannot see, unprompted. This is the class behind Hermes #32766 and #52014 (§4). | PR 1 |
| **D11** | **A failed health check is skipped on the next start.** `start()` keeps `self._driver` set after `is_available()` returns False, so the next `start()` returns early. | `cua.py:168-189`. **Measured** with a fake SDK: the first call raises `DriverUnavailable`, the second returns silently. | Contradicts decision 4's "health known before dispatch". | PR 1 (and PR 2's probe) |
| **D12** | **The pin admits a driver that breaks observe.** From 0.28.3, `GetWindowStateInput` has a *required* keyword `max_image_dimension`. Our call omits it, so every observe raises `TypeError`, which becomes `DriverUnavailable`. Only `uv.lock` prevents it. A `pip install 'oara-prometheus[computer]'` (the remedy `cua.py:100-103` itself prints) resolves 0.33.1. | `pyproject.toml:105`; `uv.lock:1067-1068`; `cua.py:222-243`. **Measured** by constructing the input with the 0.33.1 bindings. | Any install that is not from the lockfile gets computer use that can never observe. | PR 1 (exact pin) |
| **D13** | **A driver timeout reports failure for an action that may still land, and the adapter has no lock.** `.result(timeout=60)` does not cancel the SDK coroutine. | `cua.py:162-166,324-332`. **Probed:** `TimeoutError` at 1.9 s; the action landed at 2.7 s. Two concurrent steps on one adapter would interleave on its loop. | A step reported "failed" may have clicked. The stop guarantee (§5.1.7) needs one step at a time. | PR 1: a lock; a timeout reported as "outcome unknown", followed by an observe |
| **D14** | **Cancelling a task that waits on an approval leaks the pending entry.** No `approval_resolved` is emitted, and a later `/approve` returns True while the entry stays. | `permissions/approval_queue.py:640-659`; the comment at `:645` describes a `finally` that no longer exists. **Probed.** | A stop implemented as a cancel would leave ghost approvals on every surface. | PR 5 |
| — | `actions.py` says "Nine tools" and lists seven. | `actions.py:30-31` | Cosmetic. | PR 1 |

Already known and still deferred (settled): **hazard (b)**, `_snapshots` is
keyed on `(pid, window_id)` and never pruned (`cua.py:134-158`). The door's
per-step window re-resolution (§5.1.3) narrows the exposure but does not
close it.

---

## 4. Part 2 — overlap with Cua and Hermes

**The rule (settled):**

* We don't rebuild anything Cua ships: driver, cursor, VMs, benchmarks.
* We keep the consent gate, the candidate table, `validate_choice` and the
  loop.

**Sources:**

* **Cua:** the installed `cua-driver` 0.28.2 wheel, PyPI, and the upstream
  `trycua/cua` repository at the commit named in each row.
* **Hermes:** a clone of `NousResearch/hermes-agent` at `158fd638`
  (2026-10-03, MIT), cited as `H:path:line`, plus the GitHub issue pages.
  Some Cua doc hosts (`cua.ai`) are blocked by this environment's egress
  proxy; rows that depend on them say so.

### 4.1 Cua

**Upstream source:** `trycua/cua` at `0d274d0` (2026-10-03). Wheels 0.28.3
through 0.33.1 were downloaded and their bindings compared with the installed
0.28.2.

| Item | What Cua offers now | Prometheus today | Verdict |
|---|---|---|---|
| **Driver SDK versions** | **0.33.1** on PyPI (2026-10-03), ten releases after our 0.28.2 (2026-09-15). **0.28.3 adds a required keyword `max_image_dimension` to `GetWindowStateInput`.** Also: perception (typed `parse_visual_regions`, `CAPTURED_COORDINATES`), X11 background/foreground input, time-budgeted walks; `ActionResult.summary/error`; 0.31.0 BREAKING: "snapshot store invalidated on read" (#3873). Unchanged through 0.33.1: no `editable`/`focused` element field; only `ClickInput` takes a delivery mode. (`libs/cua-driver/rust/CHANGELOG.md`; PyPI JSON) | We wrap it. Pin `>=0.28,<1` (`pyproject.toml:105`), lock 0.28.2 (`uv.lock:1067-1068`). **Building our observe input with 0.33.1's bindings raises `TypeError: missing … 'max_image_dimension'`, which `cua.py:239-243` turns into `DriverUnavailable`.** Only the lockfile prevents it (D12). | **Use theirs, pinned exactly** (`==0.28.2`; `~=0.28.2` would still admit the break). Move to ≥ 0.33.1 deliberately, passing `max_image_dimension`/`timeout_ms` explicitly, for perception. |
| **Hosting modes** | `EMBEDDED` (`create()`, in-process). **`PRIVATE_WORKER`**: "one supervised child runtime over inherited pipes; no listener, closes with its channel", upstream's mode for "native crash containment"; it cannot take host callbacks, so there is no activity observer. `DAEMON` (`connect` to `cua-driver serve`). MCP (`cua-driver mcp`). And `REMOTE` (`connect_remote_channel`). (`docs/…/use-the-sdk.mdx:224-242`; `libs/cua-driver/docs/sdk-first-runtime-north-star.md:101,403,407`; `_native.py:5074-5085`) | `EMBEDDED` only (`cua.py:109-116,178`): no crash isolation. The docstring's "Cua has none" (no remote transport, `cua.py:113-115`) is stale against `REMOTE`. | **Use theirs: `PRIVATE_WORKER`** for the Integration (decision 4's supervision, with real containment), subject to an on-box check on X11 (§8 #10). **Not needed:** `DAEMON` (a shared socket); MCP (raw tools the gate cannot read, `actions.py:8-20`); `REMOTE` (decision 3). |
| **No-foreground contract** | Background means no window raise, no real pointer move, no frontmost switch. It "never retries in the foreground on its own"; impossibility returns `background_unavailable`. `ActionResult.escalation` is advice to the harness. `delivery.mode` reports what actually happened. The MCP tools accept `delivery_mode` on `type_text`/`press_key`/`scroll` (via `call_tool`); the typed SDK only on click. (`docs/…/how-cua-driver-works.mdx:41-43`; `action-result-contract.md:43-44,127-140`) | The extent's delivery term makes foreground a separate grant (`computer_schema.py:180-187`). Delivery is passed for click only (D5). `_verdict` drops `escalation` and `delivery.mode` (`cua.py:364-403`). | **Use their contract, keep our gate:** the same split. When the extent says background, require `delivery.mode ∈ {BACKGROUND, NOT_APPLICABLE}` and log `escalation` (PR 1). |
| **Agent cursor** | `set_agent_cursor_enabled/motion/theme`: an overlay with a badge naming the session, delivery mode and target type. "A visual aid, not an authorization signal". On by default; works in background delivery; kept out of captures. On macOS an in-process runtime returns `facility_unavailable` (a worker is needed). (`agent-cursor.mdx:13-80`; `operate.mdx:259-293`) | None. | **Use theirs.** Label the session with the task id so the badge names the task. **Off by default on X11** (§4.2, §5.2.3). |
| **Authorization layers** | Runtime-side, narrowing only: STANDARD (default; observe and input unprompted), **BOUNDED** (a manifest naming apps by bundle id or executable, a display flag and TTLs; deny by default), UNRESTRICTED. `TrustedSessionOptions` + `create_trusted_session` gives an immutable, connection-bound session; ending it removes its grants. "Code running inside the runtime's process can bypass them." (`permissions.mdx:8-110,180-187,266-279`; `_native.py:2363,3879-3893,7426-7451`) | Absent: `create()` means STANDARD. Our gate is the only control. | **Use theirs as a floor under our gate.** A BOUNDED manifest per task from the picked app: tools = `get_window_state` + our five verbs, TTL = the task budget, `end_session` on stop. It is a real boundary only out of process, which is another reason for `PRIVATE_WORKER`. Our gate stays, because it is the per-action consent. |
| **Activity observer** | Content-free events of the driver's own calls (§2.2). Unavailable in a private worker. | Absent. | **Keep ours** for the log (`StepResult`). Use the observer only as a cross-check, and only if we stay `EMBEDDED`. |
| **Telemetry** | cua-driver: on by default, content-free, to PostHog EU. Precedence: `DO_NOT_TRACK`, then `CUA_DRIVER_RS_TELEMETRY_ENABLED`, then `CUA_TELEMETRY_ENABLED`, then config. The update check is separate. The code lives in the **CLI binary crate**; `libcua_driver_sdk.so` has no PostHog strings (strings check only). **The private worker *is* the binary**, and its environment allowlist passes `CUA_DRIVER_RS_TELEMETRY_ENABLED` but not `DO_NOT_TRACK`. cua-bench/cua-agent use cua-core (`CUA_TELEMETRY=0`, `DO_NOT_TRACK=1`). Lume: `LUME_TELEMETRY_ENABLED`. (`operate.mdx:295-319`; `crates/cua-driver/src/telemetry.rs:212-241`; `crates/cua-driver-sdk/src/embedded.rs:837-870`, `worker.rs:141-158`) | No opt-out is set anywhere in `src/`. | **Decision 3 governs.** Set `CUA_DRIVER_RS_TELEMETRY_ENABLED=0` and `CUA_TELEMETRY_ENABLED=0` in the daemon *and* in `PrivateWorkerOptions.environment`, with a test. Audit the update check. Eval rigs run with all of the above plus `LUME_TELEMETRY_ENABLED=0`. |
| **Lume / Lumier** | Lume 0.6.0 (2026-10-01): Apple Virtualization VMs (macOS and Linux guests) on Apple silicon; CLI, HTTP API and MCP; at most two macOS guests. Lumier 0.1.3: VMs in Docker. (`libs/lume/CHANGELOG.md`; `docs/…/lume/index.mdx`) The PyPI package `lume` is unrelated. | Absent. | **Use theirs, for local evaluation VMs only** (macOS paths: AX roles, cursor, TCC). Never a runtime dependency. |
| **Cua Bench** | `cua-bench` 0.3.0 (2026-10-01), Python ≥ 3.12. Local gVisor/runc containers or QEMU/Lume VMs. Datasets: `cua-bench-basic` (13), kicad (25), workflows; adapters for OSWorld, ScreenSpot-Pro and others. Output: `result.json`, **ATIF-v1.8 `trajectory.json`**, pass@k. Agents subclass `BaseAgent.perform_task(…, session: DesktopSession, …)`. The sandbox's driver extra pins `cua-driver==0.27.0`. | Absent. The venv is 3.11; `cua.py:178` hard-codes `create()`. | **Use theirs to evaluate the loop** (§5.5.3), in a separate 3.12 venv, `--on local`, telemetry off. Needs a driver-factory seam in the adapter. Watch the 0.27/0.28 skew. |
| **Trajectory recording / export** | Driver MCP tools `start_recording`/`replay_trajectory` save every action **with arguments and screenshots**; replay re-invokes tool calls. Computer History is a nightly-only, metadata-only preview. Cua Bench writes ATIF. (`trajectories.mdx`; `recording.mdx:13-19,89-94`) | Excluded on purpose: "the recording/replay family" (`actions.py:42-43`). | **Not needed.** Recording persists typed text and frames; replay bypasses the table and the gate. Watch mode replaces it (decision 8). *Later, optional:* export our own log as ATIF for bench scoring. |
| **Grounding models** | cua-agent 0.9.0 loops for UI-TARS, GTA1, Holo1.5, OpenCUA, composed planner+grounder; `predict_click → (x, y)`; OmniParser via the deprecated AGPL `cua-som`. **`cua-perception` 0.2.1:** local CPU ONNX regions; the icon detector is AGPL-3.0 and PP-OCR is Apache-2.0; about 3.5-4 s per frame on Linux; capture-bound, single-use clicks (60 s expiry), typed from 0.28.3. Typed `WindowStateOutput` has no `capture_id` even in 0.33.1, so it may need `call_tool`. (`perception-extension.mdx`; `rfcs/3931…md:515-519`) | Absent; no screenshots. | **Use theirs for the later pixel tier: the perception extension's regions become table rows** (§5.5.2). The AGPL detector needs a licensing call. **Not needed:** cua-agent grounders as actors, because they emit raw coordinates (an open action space). UI-TARS and GTA1 at most rank, snapped to a region. |
| **jev-use / RFC 4268 / Cua-S1** | RFC 3931 (boundary) and **RFC 4268 "native accessibility candidates"**, completed 2026-09-29. It is the same design as ours: the chooser sees only `{id, description}`; `reobserve`/`abstain` are required; 2-32 candidates. It adds data we lack: a per-platform **role-class map** (the driver does not normalise roles); **eligibility filters** (enabled, on-screen, labelled, *not `in_web_content`*); **stable IDs** (`ax:<role_class>:<slug>`); **`set_value` by token** for text; **risk tags** (destructive / send / purchase / close_unsaved) removed unless allowed. Cua-S1 (local LoRA on Qwen3.5-4B, ≤ 26 options, research-stage). TypeSafe Jev is hosted. (`rfcs/4268-…md:175-272,421-562`; `examples/jev-use/decision-models.md:86-137`) | Same shape: `types.py:3-8` cites RFC #3931; `chooser_view` (`types.py:118-129`); `validate_choice` (`candidates.py:156-189`). Gaps: IDs are `click-<index>`; AT-SPI roles only; no filters; 40-row cap; D2. | **Keep ours** (core, and upstream converged on it). **Adopt RFC 4268's rules as data:** role map (L5); filters; `set_value` by token (the D2 fix); risk tags as the §5.1.6 list. **Not needed:** TypeSafe Jev (remote). Cua-S1-4B is an optional local comparison arm (§5.5.3). |

### 4.2 Hermes `computer_use`

| Item | What Hermes does now | Prometheus today | Verdict |
|---|---|---|---|
| **Tool shape** | **One** model-facing tool, `computer_use(action=…)`, with 14 actions: capture, click, double/right/middle click, drag, scroll, type, key, set_value, wait, list_apps, list_windows, focus_app. It takes raw coordinates, arbitrary key combos (`cmd+s`), drag and free text. The model sees a screenshot with numbered boxes plus an element list (index, role, label ≤ 120 chars, bounds); element tokens stay in the backend. It talks **MCP over stdio** to `cua-driver mcp`, with the lock pinning `cua-driver-rs` 0.21.0 (floor 0.20). Platforms: darwin, win32, linux. (`H:tools/computer_use/schema.py:17-185`; `tool.py:561-584,692-700`; `cua_backend.py:1-7`; `pm/lock.json`) | No model-facing tool (decision 1). A candidate table built from an in-process observe with no screenshot; the chooser returns an ID; `validate_choice` resolves it (`candidates.py:156-189`). Closed key set, no modifiers (`actions.py:217-220`). | **Keep ours.** A raw coordinate/key/text surface cannot pass through a candidate table and `validate_choice`, which are core (decision 5). **Borrow** the 120-char label cap (§5.1.6). |
| **Approval key** | **Confirmed:** `cua:<action>:<background\|foreground>` (`H:tool.py:395-398`), plus `cua:bring_to_front:<mode>` (`tool.py:345-349`). No app term, no machine term, no payload term. The prompt does not name the app (`tool.py:396-399,455-458`). | `target:app:verb:delivery`, exact whole-value match (`computer_extent.py:65-68`; `checker.py:391-409`). `computer_schema.py:32` describes the Hermes shape correctly. | **Keep ours.** The app and target terms are what make a grant refusable. Both keep foreground out of a background grant. |
| **Approval modes** | `approvals.mode` = manual \| smart \| off, plus `--yolo`/`/yolo`. Choices are once / session / always. "Session" is a real per-session set; "always" writes `command_allowlist`. Computer use never consults the "smart" LLM guardian and **fails closed with no human** (`H:tools/approval.py:250-253,286-292,381-393,793-866,973-1020`; `tool.py:403-405`). | Scopes are `until_restart` (process-wide) and `persistent`; there is no per-session scope (`checker.py:333-345`). No approver means refuse (`loop.py:195-206`). | **Keep ours.** Hermes's per-session store is the *shape* our toggle needs. It is built in our runner (§5.1.4), not ported, because theirs is keyed on their pattern strings. |
| **Typed text** | "Always" is offered for `type`, `key`, `set_value` and `drag`. One "always" on `type "hello"` stores `cua:type:background`: **any text, any app, permanently**. Typed text is screened by a shell-pattern denylist (`H:tool.py:42-69,343-344,475-476`). | Payload verbs are never rememberable (`computer_extent.py:70-78`; `approval_queue.py:193-194`). **Our gap is D2:** the text goes to focus, not to the named field. | **Keep ours**, and fix D2. Hermes is honest that its `type` has no element (`H:cua_backend_input.py:148-151`). |
| **Overlay cursor** | Cua's own overlay; Hermes builds none. It is **off by default** on macOS, headless Linux, WSL and **Linux X11**, where the always-on-top overlay can stick and block input (`H:cua_backend.py:45-65`; Hermes #28152, #83473). | None. | **Use Cua's**, and copy Hermes's X11-off default (§5.2.3). |
| **Driver bug #52014** | NousResearch/hermes-agent#52014, *"Windows computer_use reaches cua-driver, but capture returns only explorer.exe desktop layer"*. **Closed, not planned.** Upstream cause: trycua/cua#2013 (Windows Graphics Capture). `trycua/cua/issues/52014` is 404. | The Windows form cannot reach us (X11, explicit pid/window). **The class does:** a plausible but empty tree is treated as usable (D10). | **Keep ours.** The gate fails safe on a wrong app, because the extent will not match. **Fix D10.** |
| **Driver bug #32766** | NousResearch/hermes-agent#32766, *"computer_use (cua-driver backend) is too fragile and breaks auxiliary vision routing"*. **Open.** `list_windows(on_screen_only=true)` returning empty gives a 0×0 capture and can leave the backend broken. The fix, PR #33054 (fallback without the filter), is **open**, with changes requested. `trycua/cua/issues/32766` is 404. | We do not call `list_windows` yet. PR 4 will, and it must handle an empty on-screen list explicitly (ask, never guess). The "backend left broken" analogue is D11. | **Keep ours.** Fix D11. Build PR 4's empty-list case as a test. |
| **Driver bug #96328** | NousResearch/hermes-agent#96328, *"macOS computer_use rejects current notarised CUA Driver and misses symlinked app path"*. **Closed**, fixed by PR #96341, merged 2026-08-27 (realpath before the bundle search; a second signing team ID). `trycua/cua/issues/96328` is 404. | It cannot reach us: we never launch the app bundle. `create()` "never launches `cua-driver` and never opens daemon IPC" (`_native.py:5702-5706`). | **Not needed.** It informs L5 (macOS host identity): Hermes attaches TCC grants to `com.trycua.driver` (`H:tools/computer_use/permissions.py:1-6,21`). |
| **Surfaces** | `computer_use` is in every chat toolset, Telegram included, and is **model-started** mid-turn (`H:toolsets.py:12-41,217`). Approvals are chat buttons. Stop is the generic `/stop`. There is no per-action log. "Bot Screen" streams an Xvnc desktop to Hermes Desktop with Take over / Hand back, and a **lease-epoch fence** voids results produced across a takeover (`H:tool.py:333-380`; `bot-screen.md:122-176`). | Nothing yet. The loop goes from approval straight to act (`loop.py:183-211`). | **Keep ours** (user-started, consented app, per-action log). **Copy the epoch-fence pattern** (§5.1.7). A VNC live view is **not needed**: the machine is the person's own, and the log comes first (decision 6). |
| **Trajectory** | Nothing specific to computer use. Generic ShareGPT JSONL is off by default (`H:agent/trajectory.py:37-40`). | None. | **Not needed.** |
| **Local only** | **No.** Screenshots go to the main or auxiliary vision model (`H:tool.py:692-700,869-946`). It does set `CUA_DRIVER_RS_TELEMETRY_ENABLED=0` on every driver process and strips provider keys from the driver's environment (`H:cua_backend.py:33-34,68-70,137-180`). | No frames are taken (`cua.py:229-233`). **No telemetry opt-out is set anywhere in `src/`.** | **Keep ours. Use their telemetry practice** (Q1, §5.3.2). |

---

## 5. The v1.1 design

### 5.0 Shape at a glance

```
 any surface                    daemon core (non-swappable, decision 5)          Integration (decision 4)
 ───────────                    ──────────────────────────────────────          ────────────────────────
 /computer <goal>  ─┐
 Beacon toggle+goal ┼─► ComputerTaskRunner.start ─► ComputerUseLoop.step ─► SecurityGate ─► approve() ─► ComputerIntegration
 REST POST          ─┘        │  (ceilings, stop,       observe → table →      (extent:          │            ├ health / version / telemetry floor
                              │   window re-resolve)    chooser → validate      target:app:site   │            ├ CuaDriverAdapter (cua-driver 0.28.2)
                              │                                                 :verb:delivery)   │            └ DriverActivityObserver
                              ▼                                                                   ▼
                     computer_* frames ──► SignalBus ──► ws_server ──► Beacon (log + Stop)   session binding (the toggle)
                                                                                             or ApprovalQueue prompt
```

The runner and the Integration are new. The loop, the table, `validate_choice`
and the gate keep their contracts. The only core edits are additive and are
named where they occur: the `site` term, the D1 floor, D2, and the D7 seam.

### 5.1 The door: one user-started `computer_task(goal)`

#### 5.1.1 Shape

```python
class ComputerTaskInput(BaseModel):    # computer/task.py — also the future tool's schema
    goal: str                          # what to do, in the person's words
    app: str | None = None             # "my editor", "Firefox"; None → ask
    text: str | None = None            # the ONLY text that may be typed; never extracted from goal
    target: str | None = None          # a declared target name; None → the single declared local target
```

* **One entry point serves every surface:**
  `ComputerTaskRunner.start(ComputerTaskInput, *, session_id, surface,
  requested_by)`. It returns a task id at once; the task runs as an asyncio
  task owned by the runner.
* **In v1.1 nothing registers it as a tool** (decision 1).
  `computer.registered` stays 0. Registering this schema later is a thin
  wrapper (decision 2's "without rework"), and that PR needs a go (§6, L1).
* **`text` is separate from `goal` on purpose.** `text_to_type` must come
  from the caller and never from the chooser (`candidates.py:60-63`). A
  chooser asked to pull text out of the goal would be choosing the payload.

#### 5.1.2 Reachable from any message surface

Measured facts that shape this:

* **There is no surface-wide command table.** Each surface registers its own
  thin handlers around the shared `cmd_*` functions:
  * Telegram registers one `CommandHandler` per command
    (`gateway/telegram.py:394-470`);
  * Slack registers `/prometheus-<name>` (`gateway/slack.py:441-500`);
  * Discord builds an app-command tree (`gateway/discord.py:969-1062`);
  * Beacon goes through the web router, which reads `_SESSION_COMMANDS` and
    `_FORMATTER_COMMANDS` (`commands.py:1848-1872,2060-2076`;
    `web/slash_router.py:111-119`).
* **A parity guard covers the three chat surfaces.**
  `tests/test_gateway_parity.py` (`MANIFEST`, `:184`) requires every command
  family on Telegram, Slack and Discord, or a stated gap.
* **No gateway has inline-button callbacks or a message-edit primitive.**
  There is no `CallbackQueryHandler`, `callback_data` or `edit_message_text`
  in `src/`.
* **An answer must itself be a command.** Telegram sends non-command text to
  the agent as a turn (`telegram.py:496-499`).

| Surface | Start | "Which app may I use?" | Progress | Stop |
|---|---|---|---|---|
| Telegram / Slack / Discord | `/computer <goal> [app:<name>] [text:"…"]`. One shared `cmd_computer` core in `commands.py`, registered on each surface, plus a parity `MANIFEST` family. It **returns at once and spawns the task**, as `cmd_gepa` does (`commands.py:2999-3002`): a handler that awaited the task would block PTB's single update fetcher, and with it `/approve` and `/computer stop` (`telegram.py:472-495`). | **One match:** a yes/no `ApprovalQueue` prompt, "May Prometheus use gnome-text-editor for this task?", approve-once. **Several:** a numbered list, answered with `/computer use <n>`. | Milestone messages through the injected `send`: started (app, what is covered, task id); each approval as its own prompt; a "still running" heartbeat (the managed-task pattern, `gateway/heartbeat.py:368-415`); a terminal summary. **Not a message per step**, and no edited status message (no primitive exists; deferred). | `/computer stop [id]`. No stop command exists on these surfaces today. |
| Beacon desktop / iOS | Toggle on + app picker (the binding), then the composer in a "Do it on <app>" mode → `POST /api/computer/tasks`. A typed `/computer …` also works: unknown names already reach the daemon (beacon-desktop `slash-commands.ts:180`, `ChatShell.tsx:1267`) and route through `_SESSION_COMMANDS["computer"]`, whose `CommandContext` gains a sender and the runner (`commands.py:1643-1683`). | The picker, filled from `GET /api/computer/apps` | §5.2 | The existing chat Stop (§5.1.7) |
| REST | `POST /api/computer/tasks {session_id, goal, app?, text?, target?}` | `409 needs_app` with the candidate list | `GET /api/computer/tasks/{id}` | `POST /api/computer/tasks/{id}/stop` |

* **Phones are the common case.** Every route accepts a device token, and the
  audit row records the device identity. Only device minting and MCP writes
  require the global token today (`web/server.py:379,2334`).
* **Both Beacon slash catalogs gain the entry**, for autocomplete only.

#### 5.1.3 Resolving "my editor" to a real window (D8: `list_windows` is uncalled today)

1. **Forced health probe** of the Integration (§5.3.2). If it is down, refuse
   with the probe's reason, the way a dead backend is refused
   (`server.py:4919-4928`).
2. **`list_apps` + `list_windows(on_screen_only=True)`** through the
   Integration (`_native.py:5602,5618`; `WindowInfo` has `pid, window_id,
   app_name, title, z_index, is_on_screen, minimized`,
   `_native_contract.py:4367-4380`).
3. **Match the phrase**, in this order:
   * an operator alias (`computer_use.apps.aliases: {editor: [code,
     gnome-text-editor]}`);
   * a case-insensitive match on `AppInfo.name`, `bundle_id`, or the
     `launch_path` basename;
   * otherwise ask.

   Exactly one running match with an on-screen window → *propose* it, and the
   person confirms. Zero or several → ask "which app may I use?", listing
   names only.
4. **Never launch anything.** `launch_app` stays out (`actions.py:42-44`). "Open
   it first, then ask again" is the answer.
5. **The window is the app's frontmost on-screen window** (max `z_index`).
   * pid and window id are **re-resolved before every step**. A vanished
     window ends the task with that reason; the runner never guesses a
     replacement.
   * Neither number reaches the extent, the prompt or the log (already
     ruled).
   * Re-resolution also narrows hazard (b) without closing it.
6. **App term spelling.** Linux uses the driver's `app_name`, normalised as
   today (`computer_extent.py:213-225`). On macOS the stable `bundle_id`
   should be the term. That spelling is fixed by the first grant for the
   same reason the site term is (§5.4), so it is decided with it (§8).

#### 5.1.4 The answer is the consent grant (decision 6), without changing the gate

* **The answer creates a session binding** `(session_id, target, app)`.
  * It lives in the runner, not in `security.grants`. It records `set_by`
    (surface and device), `created_at` and `expires_at`.
  * It is modelled on `session_workspaces`, the existing security-relevant
    per-conversation binding (`memory/lcm_conversation_store.py:237-255`;
    `server.py:1672-1734`).
* **The gate is unchanged.** For any computer extent without a stored grant,
  the gate still returns APPROVE (`checker.py:1130-1150`). The loop then calls
  its injected `approve` callback (`loop.py:79,183-194,263-282`). For a door
  task that callback is `SessionConsent.approve`, which:
  1. re-derives the extent from the arguments with the **same**
     `computer_extent_for` and schema the gate used (`loop.py:170-172`);
  2. if the extent is covered by the binding, returns `True` and writes an
     audit row `confirm_approved: session binding <id>` through the existing
     redacting `AuditLogger`;
  3. otherwise forwards to the real approval path (`ApprovalQueue`), tagged
     with the task id.

  Stored grants from `/approve always` keep working exactly as they do today.
* **What the binding covers:**
  `target:app:-:{click,scroll,press_key}:background`.
  * Only with site `-`, i.e. no web content.
  * Only Return, Tab and Escape: the three keys the table offers
    (`candidates.py:109-118`).
  * The `press_key` extent does not name the key, and the wider closed set
    includes `backspace` and `delete` (`actions.py:217-220`).
    `SessionConsent.approve` therefore checks the key argument itself.
* **What it does not cover, each of which prompts every time:**
  * `type_text` and `invoke_menu` (payload: never rememberable);
  * any extent whose site is UNKNOWN (browser chrome in a window with web content; page content itself is not offered in v1.1, §5.4);
  * `foreground` (not in v1);
  * a label on the high-consequence list (§5.1.6).
* **The sentence the person reads** comes from the same `describe()`
  machinery, so the two cannot drift:

  > Prometheus may click, scroll and press Return/Tab/Escape in
  > gnome-text-editor on mini, in the background, and read everything shown
  > in its windows (not just the front one), until you turn this off.
  > Typing and menus ask every time. Web page content is not offered yet.

* **Lifetime:** until the toggle goes off, the session ends, or 8 hours pass.
  That ceiling mirrors cua's own absolute grant ceiling.
  * **Proposed:** not persisted across a daemon restart in v1.1, so a restart
    re-asks.
  * A durable table with boot-restore onto a healthy driver (the
    `restore_backend_overrides` pattern, `router/model_router.py:587-628`) is
    a follow-up (§8).

**Approvals have to reach the person who started the task.** Today they
cannot. Each item below was measured, and each is a PR 5 prerequisite:

1. **The queue exists only with Telegram plus a flag that ships off.** The
   `ApprovalQueue` is built only when Telegram is configured and
   `security.approval_queue.enabled` is set (`daemon.py:1496-1529`). With no
   queue, `gate.request_approval` returns False (`checker.py:1632-1634`) and
   the loop refuses, so **a Beacon-only install refuses every desktop
   action**. PR 5 builds the queue whenever `computer_use.enabled` is set,
   independent of Telegram.
2. **Prompts always go to the default Telegram chat.** That is
   `allowed_chat_ids[0]` (`checker.py:1639-1645`; `daemon.py:1506`),
   whichever surface started the work. PR 5 tags `PendingAction` with
   `task_id` and `session_id` (`approval_queue.py:54-79,509-517`;
   `serialize_pending` at `:412-464`) and routes the chat prompt to the
   starting chat. The tag is carried through `gate.request_approval` and
   `_call_approve` (`checker.py:1606-1611`; `loop.py:263-282`) by keyword or
   by a context variable the runner sets.
3. **`deny_task(task_id)`** resolves every pending approval of one task as
   denied. That is how stop unwinds a waiting step cleanly (§5.1.7).
4. **A cancelled waiter leaks its entry (D14).** Cancelling a task that waits
   on an approval leaves the entry in `pending` with no `approval_resolved`.
   A later `/approve` on it returns True and the entry still stays
   (`approval_queue.py:640-659`; the comment at `:645` describes a `finally`
   that no longer exists). Probed. PR 5 adds the `finally`.
5. **`/approve all` drains desktop approvals.** It approves every pending
   entry once, `computer_click`, `computer_type_text` and `bash` alike. So
   does `POST /api/approvals/all/approve` (`commands.py:2618-2635`;
   `server.py:4452,4465`). Probed. Excluding them is decision #7.
6. **iOS cannot show what it is asked to approve.**
   * `Approval` has no `arguments` field (beacon-ios `Models.swift:446-470`),
     so the text a `type_text` would type is never shown on a phone.
   * The lock-screen "Approve once" action is offered even when arguments
     exist (`NotificationController.swift:113-125`), which is the case desktop
     deliberately blocks (beacon-desktop `approval-push.ts:142-212`).
   * PR 6 (server) sends no APPROVAL category for an approval that carries
     arguments.
   * PR 8 (iOS) decodes `arguments` and shows a "With:" list.
   * Until PR 8 lands, a `type_text` approval should not be answerable from
     iOS (decision #15).

#### 5.1.5 Running a task

* **One task per target at a time.** A second start is refused, and watch
  mode is mutually exclusive with it (§5.6).
* **Each iteration:**
  1. stop check;
  2. re-resolve the window;
  3. `await loop.step(goal, target, app, pid, window_id, text_to_type=text,
     history=history)` (`loop.py:94-104`);
  4. emit a `computer_step` frame;
  5. branch on the step status:

| `StepResult.status` | Runner does |
|---|---|
| `executed` | Continue. `history` gains "Clicked push button 'Save' — window changed" (a richer entry than `loop.py:232`'s bare description, passed via `history=`, with no core change). |
| `reobserve` | Continue. Three in a row end the task. |
| `abstained` | End: "nothing in the window serves the goal, or it is done". |
| `refused` | End, with the reason: operator declined, invalid choice, or stale snapshot. |
| `blocked` | End, with the precondition reason. |

* **Ceilings**, all config:
  * `max_steps` 20 (the audit asked for "well below 500",
    `COMPUTER-USE-REGISTRATION.md:203-206`);
  * wall clock 10 min;
  * approvals per task 10;
  * reobserve streak 3.
* **D1 is fixed in the loop (PR 3).** This is a **hard precondition for
  the door** (Will verified the finding at `checker.py:1026-1037` on
  2026-10-03).
  * **The fix:** when the gate allows a computer extent (known or unknown) at
    `TrustLevel.AUTONOMOUS` (`checker.py:1037`), the loop routes it to
    `approve` anyway.
    * A grant match allows at `TrustLevel.AUTO` (`checker.py:1067-1071`),
      so it passes through untouched, and no reason strings are parsed.
    * This mirrors `agent_loop.py:4991-5008` and extends it to known
      extents.
  * **Parity:** under `/gate off` the gate never reaches its grants check.
    So `approve` consults the session binding and the stored grants (same
    exact match) before prompting. `/gate off` then neither waives nor
    tightens computer consent: "the floor is not a mode"
    (`checker.py:1021-1025`).

#### 5.1.6 Mitigations for model-chosen clicks under a binding (Q2)

The binding plus a model chooser is the combination the audit warned about.
Inside the picked app, for the life of the toggle, the design accepts it and
adds these:

1. **Typing and menus always prompt.** This already exists (payload rule).
2. **Web content is not offered in v1.1** (§5.4), so page-authored labels never reach the chooser's table. Browser chrome prompts.
3. **High-consequence labels prompt even when covered.** A description
   matching send, delete, remove, pay, transfer, purchase, submit, confirm,
   sign, or the same words in the app's language list, prompts. It is a
   denylist on app-supplied strings, so it is a **mitigation and not a
   control** (audit §1, direction 3).
4. **The prompt and the log mark descriptions as app-supplied text** (audit
   §1, direction 2). Labels are capped, for example at 120 characters as
   Hermes does (`hermes-agent tools/computer_use/tool.py:561-563`).
   `Element.describe()` is unbounded today (`types.py:59-66`), and it reaches
   the chooser, the prompt and the audit reason.
5. **Per-task approval ceiling, and computer approvals excluded from
   `/approve all`** (audit §5).
6. **The binding is per session and per app, never `until_restart`.**
7. **A live log, and a stop that works from a phone** (decision 6, §5.2).

#### 5.1.7 What stop guarantees

**The mechanism is cooperative, and built from the loop's existing seams.**
The runner hands `ComputerUseLoop` a chooser wrapper and the
`SessionConsent` approver, and both are injected already (`loop.py:73-92`).
A stop epoch is checked at four points:

1. by the runner, before each step;
2. by the chooser wrapper, after the inner chooser returns: if stopped, it
   answers `abstain`;
3. by the approver, **after any approval wait returns**: if stopped, it
   answers False;
4. by the runner again, after verify.

The only unguarded window is the instant between an *allow* (a grant match,
which calls no approver) and `driver.act`. Nothing awaits in that window.

This copies the *pattern* of Hermes's lease-epoch fence
(`H:tools/computer_use/tool.py:333-380`); no code is taken. **No core edit is
needed.**

**On stop:**

* `deny_task(task_id)` resolves the task's pending approvals. The waiting
  step then returns `refused` through its normal path (`loop.py:188-194`).
* `task.cancel()` is **not** the primary mechanism.
  * Cancelling mid-approval leaks the entry (D14).
  * Cancelling mid-act skips verify, so no `StepResult` describes the
    dispatched action (`loop.py:211-223`).

**One dispatched action can still land (probed).**

* The probe used the real `CuaDriverAdapter._await` with a 1-second fake SDK
  coroutine. Cancelling the step at 0.2 s raises `CancelledError` at once,
  yet the coroutine on the driver's loop runs to completion and the action
  lands at 1.0 s.
* `.result(timeout=…)` does not cancel it either. On timeout `act` raises
  `DriverUnavailable` while the action lands later: `TimeoutError` at 1.9 s,
  landed at 2.7 s (`cua.py:162-166,324-332`). That is D13.
* **The guarantee is therefore:** no new action starts after the stop is
  acknowledged, and at most one already-dispatched call may land.
  * That call is reported as "in flight at stop — may have landed", and is
    checked with one post-stop observe.
  * This holds only with **one step at a time per adapter**. The adapter has
    no lock today (D13), so the runner serialises, and PR 1 adds the lock.
* The UI says exactly this, not "stopped instantly".

**The approval-to-act gap.**

* Up to 1800 s can pass between an approval and the act
  (`approval_queue.py:44`).
* The adapter's staleness check compares only with its own last observation
  (`cua.py:281-308`).
* So if an approval wait exceeds 30 s (proposed), the approver returns False
  with "the window may have changed while you decided — looking again". The
  next step re-observes and asks again. That is fail-closed and stated.

**Every stop control reaches the task.**

* Beacon's existing Stop (`POST /api/chat/interrupt` and the WS `interrupt`
  frame) cancels only `_turn_tasks[session_id]` (`ws_server.py:487-497,790-805`;
  `server.py:1242-1272`).
* The runner registers each task there as the session's running turn, and
  `interrupt_turn` hands it to the runner's stop path instead of a bare
  `cancel()`.
* Both clients' Stop buttons therefore work with **no client change**:
  desktop `ChatShell.tsx:1238-1253`, iOS `ChatController.swift:238-245`.
* Chat surfaces get `/computer stop`.

### 5.2 The cockpit: a live action log and a stop control

The log comes first; the cursor is a nicety (decision 6). There are two
layers. The first needs no client change at all; the second is durable.

#### 5.2.1 Layer 1: the task renders today, as a turn

Because the runner registers the task as the session's turn (§5.1.7), it can
emit the frames both clients already render.

* **The task itself:**
  * `tool_call_start {session_id, call_id, tool_name: "computer_task",
    inputs: {goal, app, origin: "user_task"}}`;
  * `tool_call_end {session_id, call_id, tool_name, success, result}`.
  * iOS requires `call_id`, `tool_name` and `success` (beacon-ios
    `Frames.swift:125-153`).
* **Each step:** one nested `tool_call_start`/`tool_call_end` pair.
  * `call_id = <task_id>:<seq>`, `tool_name = computer_<verb>`;
  * `inputs = {description, extent}`, redacted;
  * `result` = the verification line.
  * Desktop renders these in its chat timeline and Tool feed, and iOS in its
    ToolStripView (beacon-desktop `ChatShell.tsx:193-215`,
    `gateway-events.ts:93-128`; beacon-ios `ChatStreamReducer.swift:173-187`).
* **Liveness:** `agent_progress` every 3 s with `phase: "tool"`
  (`ws_server.py:1503-1526`). It keeps desktop's activity line, iOS's reply
  watchdog and the Live Activity alive.
* **Stop:** `chat_done {interrupted: true}`.

**Costs, stated:**

* `tool_call_*` frames are never persisted (`gateway-events.ts:130-133`), so
  they vanish on a history reload.
* They look like model-issued calls. `origin: "user_task"` marks them, and
  clients ignore extra keys.

Hence layer 2.

#### 5.2.2 Layer 2: the durable log

* **Kinds are declared once, by the emitter.** `computer/livestream.py`
  declares `COMPUTER_FRAME_KINDS`. `ws_server._on_signal` promotes a kind to
  a first-class frame type **from that tuple**, exactly as
  `CODING_FRAME_KINDS` is promoted (`coding/livestream.py:44-63`;
  `web/ws_server.py:27-35,1640-1642`).
* **Why that matters:** a kind left unpromoted ships as a generic
  `sentinel_signal`, and every client gate keyed on `type` silently matches
  nothing. It happened to `coding_acceptance`/`coding_tool` and to
  `task_completed`/`task_failed` (`ws_server.py:1626-1660`).
* **Both pinning tests are copied:** *kinds match every emit site* and
  *every kind is promoted*.

The envelope is the existing `{type, timestamp, payload}`. Every payload
carries the **chat `session_id`**, because ProgressPane keys on it
(beacon-desktop `ProgressPane.tsx:138-162`).

| `type` | When | `payload` |
|---|---|---|
| `computer_binding` | Toggle on/off, expiry | `{session_id, state: "on"\|"off", target, app, describes, covers: ["click","scroll","press_key"], asks: ["type_text","invoke_menu","web","high_consequence"], set_by: {surface}, expires_at}` |
| `computer_task_started` | Task accepted | `{session_id, task_id, target, app, goal, chooser: "rule"\|"gemma", started_by: {surface}, limits: {max_steps, max_seconds, max_approvals}}` |
| `computer_step` | After every loop step, and on `awaiting_approval` | `{session_id, task_id, seq, status: "executed"\|"refused"\|"abstained"\|"reobserve"\|"blocked"\|"awaiting_approval", action: {verb, description, app_text: true}, extent, consent: "binding"\|"grant"\|"prompt"\|null, approval_request_id, chooser: {name, confidence, reason}, effect: "CONFIRMED"\|"PARTIAL"\|"UNVERIFIABLE"\|null, verified: true\|null, after_stop: false, candidates_offered, duration_ms, reason}` |
| `computer_task_ended` | Terminal | `{session_id, task_id, outcome: "done"\|"abstained"\|"stopped"\|"refused"\|"failed"\|"limit", reason, steps, approvals, duration_ms}` |
| `computer_stream_error` | The log itself failed (never the task) | `{session_id, task_id, detail}` |

* **`seq` is the de-dupe key**, as `seq` is for coding runs (beacon-desktop
  `coding.ts:339-343`).
* **Approvals reuse `approval_pending`/`approval_resolved`**, with optional
  `task_id`, `session_id` and `extent` added.
  * Both clients ignore extra keys (`approval-push.ts:81-101`; beacon-ios
    `Models.swift:463-469`).
  * Never drop or retype `extents` or `created_at`, or iOS drops the frame.
* **Persistence and backfill reuse what exists.**
  * Every SignalBus emission is written to `signal_events` before it is
    broadcast (`sentinel/signals.py:105-120`).
  * On connect the server replays nothing (`ws_server.py:244-249`).
  * `GET /api/events/recent` takes only `limit` and a single `type`
    (`server.py:3178-3202`), although the tracker underneath supports `since`
    and several types (`telemetry/tracker.py:1598-1605`).
  * PR 6 passes `since`, `types` and `session_id` through, so a returning
    phone catches up.
  * Retention is `computer_use.action_log.keep_per_session`, with the pruner
    in the same PR.
* **Older clients are safe, and that has a consequence.**
  * Desktop shows an unknown kind as a generic "system" row in the Activity
    feed, **with the whole payload, exportable** (`gateway-events.ts:84,286-287`;
    `ActivityFeed.tsx:120-162`). So the content policy below governs every
    field.
  * iOS decodes an unknown kind to `.unknown` and drops it (beacon-ios
    `Frames.swift:5-8,296-299`).
* **The iOS precedent to avoid.** A new kind needs a decoder-list entry *and*
  a reducer case, or it vanishes silently. `coding_tool`/`coding_acceptance`
  have been dropped on iOS since #503 (`Frames.swift:230-235`;
  `CodingRunReducer.swift:119-127`).
* **Desktop must keep `computer_step` out of the Activity feed,** as it keeps
  `coding_round` out (`gateway-store.tsx:104`). Otherwise it takes over
  Mission Control's three-item activity card (`MissionControl.tsx:697-710`).
* **Content policy, enforced by a test that greps every emitted payload:**
  * **Never on the wire:** element tokens, snapshot ids, pid or window id,
    screenshots, the labels of candidates that were *not* chosen.
  * **Typed text** appears as `{"text_chars": N}`. It reuses
    `redact_arguments`.
  * **Descriptions are app text**, flagged `app_text: true`, and capped
    (§5.1.6).
* **Push (APNs) carries no content.** Two server-side changes:
  * **A generic body for computer approvals.** Today the body is
    `"<tool_name> — <first line of description>"` (`push/dispatcher.py:100-117`).
    A computer description names the app and the machine
    (`checker.py:1137-1140`), so both would pass through Apple when
    `push.enabled` (off by default). The iOS service extension already
    rewrites the body locally over the tailnet (beacon-ios
    `BeaconNotify/NotificationService.swift:52-67`).
  * **No APPROVAL category when the request carries arguments**
    (§5.1.4, item 6).
* **Driver activity is a cross-check only when hosted `EMBEDDED`.** The
  observer is unavailable in a private worker (§4.1). The log never depends on
  it.

#### 5.2.3 Client → server

* **Stop:** the existing session interrupt (§5.1.7). Also a WS frame
  `{type: "computer_task_stop", payload: {task_id}}`, acked to the requesting
  socket as `computer_task_stop_ack {task_id, stopped}`, mirroring
  `interrupt`/`interrupt_ack` (`ws_server.py:487-497`). Also `POST
  /api/computer/tasks/{id}/stop` and `/computer stop`.
* **Toggle:**
  * `PUT /api/sessions/{id}/computer {target, app}` → `computer_binding on`;
  * `DELETE` → `off`;
  * `GET` → the current state.

  This is the `session_workspaces` route shape (`server.py:1672-1734`).

#### 5.2.4 Where it shows

| | Live log + Stop | Toggle + picker | Health |
|---|---|---|---|
| Beacon desktop | Layer 1 in the chat timeline today. Layer 2 in the ProgressPane's reserved computer-use section (`ProgressPane.tsx:1-13,237-243`, which today reads "Not connected…") through a `reduceCodingRuns`-style reducer. The existing chat Stop. | Thread header, beside the per-conversation autonomy chip (`ChatShell.tsx:1788-1799`) | The merged Integrations list (§5.3.5) |
| Beacon iOS | Layer 1 in the ToolStripView today. Layer 2 as a COMPUTER section beside Status → CODING (`StatusView.swift:166-253`), plus a task strip above the composer. The existing composer Stop. | The model line above the composer (`ChatView.swift:154-170`) | A `computer` row in the Status tab, next to the backend rows (`StatusView.swift:118-140`) |
| Chat surfaces | Milestone messages (§5.1.2) | `/computer` reply | `/computer status` |

**The cursor: use theirs, off on X11 by default.**

* cua-driver ships an agent cursor (`set_agent_cursor_enabled`,
  `set_agent_cursor_motion`, `set_agent_cursor_theme`): an overlay whose
  badge names the session, "a visual aid, not an authorization signal".
  Nothing is built for it here.
* Hermes, which uses the same overlay, turns it **off by default on Linux
  X11**: there it is a fullscreen always-on-top window that can stick over
  every workspace and block desktop input (Hermes #28152, #83473;
  `H:tools/computer_use/cua_backend.py:45-65`).
* X11 is our substrate, so the default is `computer_use.cursor: off`. `on` is
  an explicit choice for someone sitting at the machine, with the session
  labelled by the task id so the badge names the task.
* The log does not depend on any of this.

### 5.3 The driver as an Integration

The driver is never a builtin (decision 4). Health and refusal mirror
`BackendRegistry`; start/stop mirrors `McpRuntime` (Q4).

#### 5.3.1 Version pinning

* **The extra becomes an exact pin, in PR 1, not later:**
  `computer = ["cua-driver==0.28.2"]` (today `>=0.28,<1`,
  `pyproject.toml:105`).
  * D12 makes this urgent. Any non-lockfile install resolves 0.33.1, whose
    observe input our adapter cannot build.
  * `~=0.28.2` would still admit the break (0.28.3).
* **A code constant names the versions the adapter was validated against.**
  The probe compares it with `cua_driver.__version__`; a mismatch reports
  `down: version-mismatch`. Config cannot widen it, because the input
  builders are tied to the SDK's API (`cua.py:406-473`).
* **An upgrade is a PR that carries the on-box outcome check.** The driver
  leg is uncoverable by CI (`cua.py:1-17`).
* **The next upgrade has a reason:** 0.33.x brings capture-bound clicks for
  the pixel tier (§5.5.2).
  * It passes `max_image_dimension` and `timeout_ms` explicitly.
  * It re-checks D2, D3 and D5.
  * It reads what 0.31.0's "snapshot store invalidated on read" (#3873) means
    for observe → act → verify.

#### 5.3.2 Health known before every task

The probe body runs in `to_thread`, under a per-Integration lock, inside
`wait_for`. Failures are recorded, never raised (`providers/backends.py:386-421`).
It checks:

1. **The telemetry floor is in place.** `CUA_DRIVER_RS_TELEMETRY_ENABLED=0`
   and `CUA_TELEMETRY_ENABLED=0` are set before `import cua_driver`
   (`cua.py:94-106`), **and** in `PrivateWorkerOptions.environment`.
   `DO_NOT_TRACK` is not enough there: the worker's allowlist drops it.
2. **The SDK imports and its version is supported.**
3. **Platform preconditions both pass** (today's `check_preconditions`,
   `driver.py:158-174`). This step becomes a per-platform provider later
   (§2.1).
4. **The runtime starts and reports `is_available()`** (`cua.py:168-189`).
5. **`list_apps` returns at least one app.** The observe half is real, not an
   empty tree shaped like a working one.
6. **The target is bound.** `app.state.computer_targets` is finally assigned
   (`server.py:833-836`).
7. **If the chooser is `gemma`, its backend is healthy**
   (`BackendRegistry.status`).

When it runs, and what reads it:

* **Forced** at toggle-on and at every task start.
* **TTL-cached** (60 s, as `backend_probe.ttl_s`) for `/api/status`, which
  reads `snapshot()` from cache with no I/O. That replaces today's probe on
  every status call (`server.py:838`).
* **The per-step precondition stays as the floor** (`loop.py:105-115`).

#### 5.3.3 Lifecycle

* **States:**
  `disabled → stopped → starting → ready ⇄ degraded → failed`.
* **Start / stop:**
  * starts when enabled, at first toggle or at boot when enabled;
  * stops when disabled;
  * closes at daemon shutdown (`mcp/runtime.py:254-284,396-405`;
    `daemon.py:2828-2829`).
* **Boot probe:** bounded, non-fatal, and only when enabled
  (`daemon.py:891-906`).
* **No timer-driven restart.**
  * A failed start is retried only on the next forced probe, which is the LSP
    precedent of never retrying a broken server within a session
    (`lsp/orchestrator.py:59-61,94-96`).
  * `DriverUnavailable` during a task ends the task, marks the Integration
    `degraded`, and never retries the action.
* **Hosting: use Cua's `PRIVATE_WORKER`, subject to one on-box check (§8
  #10).**
  * Upstream's mode for "native crash containment": one supervised child
    runtime over inherited pipes, no listener, and it dies with its host
    (§4.1).
  * That is decision 4's "supervised" with a real process boundary. Today a
    native fault in `libcua_driver_sdk.so` takes the daemon down
    (`cua.py:109-116`).
  * It costs the activity observer, which the log never needed (§5.2.2).
  * `EMBEDDED` stays for the CI translation tests.
* **A driver-enforced floor under our gate (use theirs).**
  * Each task opens a cua trusted session (`TrustedSessionOptions`) in
    **BOUNDED** mode. The manifest names the picked app, allows only
    `get_window_state` and our five verbs, with `desktop.display: false`, and
    its TTL is the task budget.
  * Stop calls `end_session`, which removes the session's driver-side grants.
  * Upstream notes that code in the runtime's own process can bypass these
    layers. The floor is real only out of process: another reason for the
    worker.
  * Whether bounded sessions work with a private worker is *not
    established*; the PR 2 on-box check measures it.
  * Our gate stays: it is the per-action consent, and theirs is
    per-session scope.

#### 5.3.4 Config: one `computer_use:` block (the `computer:` keys fold into it)

* **Why this name:** the only existing reader already says `computer_use`
  (`targets.py:137-157`), with its error text and tests. It has no call site
  and no template entry, so the block can still be shaped freely.
* **The `/api/status` wire key stays `computer`** (`server.py:971`).

| Key | Default | Validation |
|---|---|---|
| `computer_use.enabled` | `false` | Only a literal `true` enables it. `resolve_telegram_enabled` uses `bool(value)`, so a quoted `"false"` would enable that one (`shipped_defaults.py:175-178`); this key must not repeat that. Off means: the driver is never constructed and nothing is registered. |
| `computer_use.targets` | `{}` | Existing grammar (`targets.py:50`). v1.1 accepts `kind: local` only; `remote` is refused under decision 3. A bad entry goes to `config_errors`, never a boot failure. |
| `computer_use.apps.aliases` | `{}` | Map of word → list of app names. |
| `computer_use.task.max_steps` / `max_seconds` / `max_approvals` | `20` / `600` / `10` | Positive ints (`shipped_defaults.py:254-271`). |
| `computer_use.chooser.kind` | `rule` | **Closed set** `{rule, gemma}`. An unknown value is a config error and the Integration reports down; there is no silent fallback. |
| `computer_use.chooser.backend` / `timeout_s` | — / `5` | The backend is a **name** in `backends:`, never a URL (`providers/backends.py:16-17`). |
| `computer_use.probe.ttl_s` / `timeout_s` | `60` / `5.0` | As `backend_probe` (`config/prometheus.yaml.default:1216-1220`). |
| `computer_use.cursor` | `off` | `on` \| `off`. Off on X11 by default (§5.2.3). |
| `computer_use.action_log.keep_per_session` | `200` | The pruner ships in the same PR. |
| `computer_use.watch.*` | (§5.6) | Lands with watch mode. |

* **Floors, deliberately not keys:**
  * telemetry off (Q1);
  * the supported driver versions;
  * `MAX_ELEMENTS`;
  * the browser exclusion;
  * background-only delivery.
* **The block must pass the existing guards:**
  * `test_config_defaults_equality.py`;
  * `test_config_drift.py`;
  * `test_template_key_placement.py`;
  * `test_absence_hostile_keys.py`;
  * the generated `docs/reference/config-keys.md` (`scripts/gen_reference.py`).

#### 5.3.5 Surfaces, and decision 7

* **`/api/status`.** The `computer` block gains `driver: {state, version,
  execution_mode, telemetry: "forced_off", checked_at}`.
* **Integration routes:** `GET /api/integrations/computer` (TTL-cached) and
  `POST …/probe` (forced). This is the `/api/backends` pattern
  (`server.py:4808-4838`).
* **Chat:** `/computer status` (the `cmd_backends` pattern,
  `commands.py:618-640`).
* **Merging Connectors and Integrations (decision 7).**
  * **Today they are different things:**
    * *Connectors* are daemon-side MCP servers (`GET /api/mcp/servers`),
      under Config.
    * *Integrations* are client-side control surfaces in `integrations.json`
      that Beacon reaches directly (control-surface contract v1), under
      Workspace.
  * **The merge is one list, one health vocabulary and one detail panel:**
    * rows come from a daemon aggregate (`GET /api/integrations`: MCP
      servers, the cua driver, backends), plus the Beacon-local entries;
    * health is `ok | degraded | down | disabled | unknown`.
  * **The computer row needs no new Beacon renderer.** The daemon serves it
    in the contract's own view types:
    * **vitals:** substrate halves, driver version, telemetry forced off,
      registered count, targets bound;
    * **events:** the action log, cursor-polled;
    * **controls:** Stop.
  * That keeps beacon-desktop's "zero integration-specific code" law
    (`src/shared/integrations.ts:1-8`).
  * Beacon's earlier plan to keep the two apart (`docs/PARITY-PLAN.md:200-204`)
    is superseded by decision 7. That file, `README.md:68` and `views.ts:55`
    need updating.
  * iOS has neither screen today; its Status tab gains the computer row.

### 5.4 The origin term — call it `site` — and why it is decided before the door

**The question.** In a browser, `mini:firefox:click:background` covers a
bank tab and a docs tab alike. "Which app may I use?" answered with "Firefox"
is consent to every site in it, signed-in sessions included.

**Naming.** The gate already uses *origin* for user-vs-system trust
(`checker.py:232-233,947`; `loop.py:80,178`). Reusing the word for a web
origin would put two meanings on one term in the same call. This document
says **site**.

#### 5.4.1 What can supply a site, and whether it can be trusted

* **Cua 0.28.2 supplies no URL at all.**
  * `WindowStateOutput`, `WindowElement` and `WindowInfo` have no
    url/origin/document field (`_native_contract.py:6311-6336, 6101-6117,
    4367-4380`; also measured from `list_tools_json()`).
  * Upstream's tree walk never reads a document URL.
  * Cua's only origin-attested path is the CDP `browser_*` surface
    (`Page.getFrameTree`, re-proved before each protected action), which is
    excluded by ruling (`actions.py:36-39`). **So v1.1 gets no site from Cua.**
* **The window title is page-controlled.** `document.title` sets it (MDN), so
  it is spoofable.
* **The address bar is browser chrome but editable.**
  * An unsubmitted edit changes its value, and the chooser can be offered that
    field (`candidates.py:47-49`).
  * Typing `docs.example.com` while on a bank page would make bank actions
    match a docs grant. That is a widening, so it is unusable.
* **The accessibility document URL is the right source, but each engine
  names it differently, and it is trustworthy only on the document node:**
  * Firefox/ATK `DocURL`;
  * Chromium on Linux `URI`;
  * macOS `AXURL` on `AXWebArea` only (on a link it is the page-controlled
    href);
  * Windows: the Document's Value.
  * A page can change its path but not its origin (`pushState` is
    same-origin), so the unit is the **origin**, not the URL.
* **What the driver does give:** `in_web_content` on every element
  (`_native_contract.py:6101-6117`), dropped today (D6).
* **Cua's own authorization layer is an inner ceiling, not consent.**
  * In standard mode, desktop input is "Routine" and allowed.
  * Its grants are pid/window/session-bound and expire in 30 min idle / 8 h.
  * It has no app or site term for desktop input; `origin` scopes only the
    `browser_*` tools.
  * It complements our gate and replaces nothing.

#### 5.4.2 Options

| | What a `firefox` grant means | When the site is unknown | Code now | Cost if done later | How "which app may I use?" reads |
|---|---|---|---|---|---|
| **A. No site term; say it** | Every site, every tab, forever | n/a | Reword `describe()` (`computer_extent.py:81-91`) and the grant row (`checker.py:538-563`) | Narrowing later = a fifth term after grants exist: every row dropped, or rows that mean "every site" | "Firefox — every page it shows, including sites you are signed in to" |
| **B. Fifth term `site` + build a provider now** | One site | Prompt | The 4→5 change (below) **plus** a per-engine provider over the a11y bus and act-time re-attestation | None | "Firefox, on docs.example.com" |
| **C. Fold the site into the app term** (`firefox@docs.example.com`) | One site | Prompt | Escaping `@`/`:` (an app named `x@bank` could forge a site-scoped grant: the colon-forging class, `test_computer_use_consent_unit.py:94-101`); grouping becomes substring parsing | None for the count; old `firefox` rows linger, matching nothing | `firefox@…` entries shown as if they were apps |
| **D1. Exclude browsers; use the browser tool** | n/a | n/a | Refuse web apps | — | The built-in `browser` is headless Chromium with an isolated context (`tools/builtin/browser.py:134,152`). It cannot act in the person's signed-in browser, so this removes a capability rather than moving it. `in_web_content` also catches Electron apps. |
| **D2. Web content is never rememberable** | Non-web only | Prompt | Small | Same as A if narrowing is wanted later | "Firefox — web pages ask every time" |
| **E. B's shape now; no web content until a provider exists** (recommended) | Non-web, or one site once a provider exists | Not rememberable: approve once (browser chrome, e.g. tabs and menus) | The 4→5 change, no provider. In v1.1 `build_candidates` drops `in_web_content` elements, which is Cua RFC 4268's rule (§4.1). | **None**: adding a provider later needs no migration | "Firefox — menus and tabs; page content is not offered yet" |

#### 5.4.3 Recommendation: E

* **The extent becomes `target:app:site:verb:delivery`.** Scope decreases
  left to right, which keeps the human grouping rule
  (`computer_schema.py:38-45`).
* **`site` takes one of three values:**
  * an origin `scheme://host[:port]`;
  * `-`, meaning *positively no web content*;
  * **UNKNOWN**.
* **UNKNOWN is gated and shown but not rememberable.** It reuses the
  mechanism that already makes payload verbs approve-once
  (`ComputerExtent.rememberable`, `approval_queue.py:193-194`).
  `derive_grant` never mints it, and `from_config_dict` refuses it as a
  stored value.
* **`-` only when all three hold:**
  * the walk is complete (`elements_complete`, not `truncated`, not
    `degraded`);
  * no element has `in_web_content`;
  * the platform path can detect web content at all. The upstream Windows
    MSAA fallback hard-codes it false, so that path never yields `-`.

  Anything else is UNKNOWN. The failure direction is over-prompting, never
  widening.
* **v1.1 ships with no site provider, and offers no web content.**
  * Elements with `in_web_content` are not built into the table (D6 passes
    the field through).
  * Browser chrome in such a window (tabs, toolbar, menus) has site UNKNOWN,
    so it is approve-once.
  * This is upstream's own rule: Cua's RFC 4268 leaves web content out of
    native candidates because "verify_state … treats web-content elements as
    an untrusted source" (`rfcs/4268-…md:211,470-472`).
  * It also removes the largest injection surface, page-authored labels,
    from the chooser's table. And it avoids the approval fatigue that an
    approve-every-web-click policy would breed under a model chooser (audit
    §5).
  * **The alternative (E′):** offer web content, approve-once, every time.
    It is listed in §8.
  * Excluding web content is a ◆ core edit to `build_candidates`: a filter,
    with the IDs and descriptions of everything else unchanged.
* **A later provider turns on per-site grants with no migration.** Linux
  first: `DocURL`/`URI` on the document node, read by Prometheus over the
  a11y bus.
* **Encoding and act-time checks:**
  * the site gets its own encoder (`_normalise_term` maps `:` to `_`, which
    would mangle a port, `computer_extent.py:213-225`);
  * it is re-checked at act time, the way `_assert_target` re-checks the
    machine (`cua.py:338-349`);
  * for `press_key`/`type_text` the site is the **focused** document's,
    because keys go to focus (D2).
* **Cost now (measured):**
  * 7 code sites hard-code four terms: `computer_extent.py:49,68,201-208`,
    `checker.py:472,560-563`, `scripts/computer_use_stall_probe.py:158`;
  * 27 four-term literals in tests, plus `count(":") == 3` at
    `test_computer_use_consent_unit.py:97`;
  * about 11 prose mentions.
  * **No client change:** both Beacons render the server's `describes`
    verbatim (beacon-ios `GrantRowView.swift:129-136`; beacon-desktop
    `ConfigPanel.tsx:545-557`).
* **Before it lands:** read the box's live `security.grants` for
  `computer_action` rows (Q5).

**Why this must be settled before the door ships.**

1. **Matching is exact and the term count is enforced.** `Grant.matches`
   compares the whole value (`checker.py:391-409`). `from_config_dict` drops
   any row whose term count differs, with only a warning
   (`checker.py:452-476`). The schema forbids ever turning matching into a
   prefix test (`computer_schema.py:41-45`).
2. **The codebase's own precedent says so.** "The term goes in before
   anything is stored" (`computer_schema.py:47-53`), and a fifth term
   "would silently invalidate every stored grant rather than failing"
   (`computer_schema.py:134-141`).
3. **The first grants fix what "app" means.** A row minted under "click
   anything in firefox" is consent to every site. A later change can drop it
   or orphan it. It cannot narrow it.
4. **The door is what creates the first real rows,** both through `/approve
   always` and through the toggle binding (§5.1.4). Before the door,
   `computer.registered` is 0 and the only caller is a human-run script.
5. **The Beacon picker's sentence is fixed at the same moment.** What the
   person reads when they pick an app *is* the consent (decision 6), so it
   has to name what is and is not covered.

**Who picked the element is not a site question, and stays out of the
extent.**

* Under decision 6 the toggle covers the picked app whichever chooser picks
  (rule, Gemma, or a recorded-step query, §5.6).
* So *who chose* is recorded in the audit row and the action-log frame
  (`chooser: rule|gemma|query`), not keyed into the grant.
* The audit's alternative (grants serve script origin only,
  `COMPUTER-USE-REGISTRATION.md:79-83`) is listed in §8 for confirmation.
  It is not adopted silently.

### 5.5 The chooser: a local model picking from the table, pixels later

**Its seat.**

* A chooser answers exactly one question: *which of these IDs* (`chooser.py:1-10`).
* It sees ID and description only (`types.py:118-129`).
* Whatever it returns is re-resolved by `validate_choice` against the table
  the client built and the live snapshot (`candidates.py:156-189`).
* Nothing here touches the gate, the extent, `build_candidates` or
  `validate_choice`.
* **Two core edits are needed, both additive:**
  * `loop.py:137` becomes `await asyncio.to_thread(self._chooser.choose,
    request)` (D7);
  * `Choice` gains `reason: str = ""` (`types.py:142-149`), so that
    timeout, backend down, grammar bypass and genuine abstain read
    differently in the log.

#### 5.5.1 `GemmaChooser` (plan step 4: Gemma + GBNF first)

* **Prompt.**
  * Built only from `ChoiceRequest`: goal, history, and `id<TAB>description`
    rows, plus `reobserve` and `abstain` rows.
  * Fixed instructions go first, so llama-server's prompt cache reuses the
    prefix across steps.
  * The prompt states that row text is copied from the application's screen
    and is data, not instructions.
  * Newlines are collapsed in every field, and each description is capped for
    the prompt only; the human prompt still shows full arguments.
  * `snapshot_id` is not sent. `validate_choice` enforces it.
* **Grammar.** Generated per request, on **one line**:

  ```
  root ::= "click-3" | "click-7" | "type-12" | "key-return" | "reobserve" | "abstain"
  ```

  * Validated against llama.cpp's own `test-gbnf-validator` (`836d571`) with
    a 44-ID table: every table ID is accepted. `click-40` (absent),
    `"click-1 "`, `Click-1`, `abstain.` and the empty string are rejected.
  * The leading-pipe multi-line form fails to parse. That is the 2026-07-02
    failure class (`adapter/enforcer.py:174-181`, pinned by
    `tests/test_force_search_provider.py:126-134`).
  * Every ID must match `^[a-z0-9][a-z0-9-]{0,63}$`, which all minted IDs do
    (`candidates.py:84,94,112`). Anything else means abstain; the table is
    never repaired.
* **Call contract.**
  * **One attempt.** A direct POST to a **llama-server** backend resolved by
    name through `BackendRegistry`. It deliberately does not go through
    `provider.stream_message`, which retries up to 3 times
    (`providers/llama_cpp.py:822-833`, `providers/retry.py:28-90`): that is
    exactly the "silently retried call in a per-step hot path" that
    `chooser.py:20-22` forbids.
  * It does not go through `LLMCallEnvelope` either, which would file a
    timeout as a user Stop (`learning/llm_envelope.py:269-273`).
  * Request: `temperature 0`, `max_tokens ≈ 16`, thinking off (as
    `llama_cpp.py:686-691` does for Gemma).
  * **Timeout means abstain**, with `reason="timeout"`.
  * **Health before dispatch:** abstain at once if
    `BackendRegistry.status(name)` is not ok, or if `/slots?fail_on_no_slot=1`
    says the slot is busy.
  * **Output outside the set** (grammar ignored, empty, truncated): abstain,
    plus a `silent_failures` row.
  * **Boot canary:** one request whose grammar admits only `abstain`. If the
    backend returns anything else it does not enforce grammars, and the
    chooser refuses to start. That covers Ollama: its OpenAI endpoint does
    not list `grammar`, though `OllamaProvider` sends it
    (`providers/ollama.py:339-346`).
* **Which Gemma.**
  * The registry carries the `gemma-4` family (`config/model_registry.yaml:13-41`).
    The ladder has a `gemma-4-26B-A4B-it` rung (`gym/ladder/rungs.yaml:172-178`).
  * The production rung is Qwen (`docs/MODEL-LADDER.md:532-534`), and
    llama-server serves one model per process (`providers/backends.py:21-25`).
    So "Gemma first" means a **second llama-server** process or box. A shared
    single-slot endpoint would turn a short timeout into frequent abstains.
* **Confidence** is recorded and evaluated, and **never authorizes anything.**
* **What GBNF plus `validate_choice` protect, and what they don't.**
  * *Protected:* the chooser has no tools, its output is one string from a
    finite set, and the action is the client's object, unchanged
    (`types.py:99-107`). Typed text is caller-supplied and never seen by the
    chooser (`candidates.py:60-63`).
  * *Not protected:*
    * picking the attacker's preferred *legitimate* row. The grammar makes
      this slightly more likely, because `abstain` is the model's only way to
      refuse;
    * history poisoning (history entries are app-derived descriptions,
      `loop.py:232`);
    * routing the person's text into the attacker's field. This is partly
      limited, because `type_text` always prompts.

  This is the gap §1.2 Q2 names. §5.1.6 lists the mitigations.

#### 5.5.2 Pixel fallback (later, per the plan): Cua's grounding, our table

* **Use theirs.** cua-driver's optional `cua-perception` extension parses a
  retained screenshot into text/icon regions (`parse_visual_regions`).
  * The worker is local, with no network and no Python.
  * A pixel click carries a **one-use `capture_id`** consumed before
    dispatch (`ClickPosition.CAPTURED_COORDINATES`).
  * It is in upstream 0.33.1 and **not in 0.28.2**, which offers only
    unbound `COORDINATES` and `ELEMENT` (`_native_contract.py:1788-1826`).
* **Keep ours.**
  * Each region becomes a `Candidate` (`region-<i>`), with a capture-bound,
    snapshot-bound token and a description such as "Click the icon region
    'Save'". The grammar, `validate_choice` and the gate are unchanged.
  * **Proposed:** a distinct verb, `click_region`, so an existing `click`
    grant never silently extends to OCR-derived targets.
  * Prefer a real accessibility element whenever one exists.
* **Grounding models only rank; they never dispatch a raw point.**
  * UI-TARS-1.5-7B and GTA1-7B expose only `predict_click(instruction,
    image) → (x, y)` in the Cua agent SDK.
  * Their point is snapped to the region candidate that contains it, or used
    as a ranking signal.
  * OmniParser-as-Set-of-Marks via `cua-som` is deprecated upstream.
  * Licences differ: older OmniParser icon detectors are AGPL, the 2026-07
    `icon_detect_v3` is MIT; the GTA1 weights' licence is *not established*.
* **It needs screenshots**, which `observe` deliberately does not take
  (`cua.py:229-233`). So the pixel fallback needs its own consent decision.
  Frames go only to the local parser, never to the chooser.

#### 5.5.3 Evaluation, with Cua Bench where it fits

**Cua Bench cannot score an accessibility-table chooser in isolation.**

* Its only no-environment mode (`DatasetSession`) is screenshot-plus-click.
* That serves OSWorld-G (564 items, 54 of them refusals, which correspond to
  our `abstain`) and ScreenSpot-Pro, which fits the *pixel* tier.
* So the evaluation has three tiers:

| Tier | What runs | Corpus | Measures | Where |
|---|---|---|---|---|
| **a. Offline chooser** | `chooser.choose(ChoiceRequest)` as a pure function; a second mode runs the rows through `ComputerUseLoop` + `FixtureDriver` + the real `SecurityGate`, asserting on `FixtureDriver.dispatched` (`driver.py:438-442`) | Recorded tables, via the existing `_RecordingChooser` (`scripts/computer_use_multistep_probe.py:63-81`): scrubbed, kept local, human-labelled. Plus synthetic rows tagged *negative*, *injection* (the audit §1 label), *history*, *duplicate descriptions*, *near-cap (38-40 rows)*, *goal already done* | Accuracy; abstain rate; unsafe-action rate (acted where gold is abstain); **invalid-ID rate on raw output, which must be 0**; timeout rate; latency p50/p95 (this sets `timeout_s`); calibration; run-to-run agreement | `gym/`: deterministic predicates, no judge (`docs/MODEL-LADDER.md:25-32`) |
| **b0. Real driver, local fixtures** | The real `CuaDriverAdapter` on a local VM or Xvfb desktop serving upstream's static fixtures (`libs/cua-driver-fixtures`), with independent state readback | Fixture pages and forms | The production driver path, which Cua Bench cannot exercise | On the box or in a Lume VM |
| **b. Cua Bench end to end** | A `PrometheusLoopAgent(BaseAgent)` wraps `ComputerUseLoop`. A `SessionDriver` implements our sync `Driver` over Cua Bench's async `DesktopSession` and mints its own snapshot IDs. An eval-only approver auto-approves and records every prompt. | `cua-bench-basic` (13 tasks) first; then OSWorld-Verified minus tasks that need internet | Task reward, `pass@k`, steps, approvals raised. **Chooser plus loop, not the cua-driver element path.** | `cua-bench` 0.3.0 needs Python ≥3.12, so it runs in a separate venv; `--on local` with docker/qemu/lume; `CUA_TELEMETRY=0` (Q1) |

Arms for tier a:

* `RuleChooser`, the baseline and the fallback the live chooser degrades to;
* Gemma E4B and 26B-A4B through the same grammar;
* the production Qwen through the same grammar;
* optionally upstream's local Cua-S1-4B decision model, as a comparison only.

Whether Cua Bench's images expose AT-SPI content to our observe is *not
established*. If they do not, tier b's tables are empty and every step
abstains, which is why tier b0 exists.

### 5.6 Watch mode and the Record-a-Skill split (plan steps 5-6)

This section only outlines plan steps 5-6. The door does not depend on any of
it.

#### 5.6.1 Watch mode on our own observe

* **Polling, not events (§2.2).**
  * `list_windows` about every 1 s.
  * `get_window_state` on the granted app's frontmost window only: about
    1 s while it is changing, 300 ms bursts for 3 s after a change, backing
    off to 3 s.
  * Never faster than 250 ms. Paused while the app has no on-screen window.
  * A poll timeout of about 5 s, not the adapter's 60 s (`cua.py:81`). Polls
    never queue.
  * None of these numbers is measured; the first on-box run records per-poll
    p50/p95 and CPU. That is also the seed of "per-run perception metrics".
* **A recorded step is a query key, never a token or an index.**
  * Each step stores `(target, app, verb, element: {role, label,
    ancestor_path, ordinal, in_web_content}, value_delta, title change,
    confidence, evidence)`.
  * No pid or window id, for the same reasons the extent omits them.
* **Inference from successive app-scoped trees.**

  | Diff | Inferred step | Confidence |
  |---|---|---|
  | Value changed on an editable role (merged until stable) | `type_text` (`FILL_FIELD`) | high |
  | `selected` flipped on a check box, radio or toggle | toggle | high |
  | `selected` moved within a list, tab or menu | select | high |
  | Title changed, or a window appeared | view change | medium |
  | Structure changed with no value delta | click, flagged *inferred* | low |
  | Key presses, scroll, hover, drag | not observable, not emitted | — |

  * A truncated or incomplete walk never yields "disappeared" (needs D6).
* **Redaction is fail-closed and happens before anything is persisted:**
  * a `password text` role never stores a value;
  * a label heuristic (password, PIN, OTP, CVV, token, API key…) withholds
    the value;
  * every string passes `redact_capture` (`security/log_redaction.py:190-214`);
  * only the delta's elements are stored, never a full snapshot (observation
    is app-scoped: two windows returned the same tree, each holding the other
    document's text, `computer_schema.py:164-169`);
  * no screenshots.
* **Consent.**
  * The session toggle's app, audited as `…:observe:background`, whose
    sentence reads "read window contents".
  * Every observation whose `app_name` does not normalise to the granted app
    is discarded. That also limits watch's exposure to hazard (b).
  * The toggle says: "Prometheus will read everything shown in every <App>
    window on <target>, not just the front one, until you press Stop."
  * Reading stays a separate consent from acting (audit §2).
* **Watch and a `computer_task` cannot share a target.** Every observe
  supersedes earlier snapshot tokens (`cua.py:286-312`), so a watch poll
  would kill a running task. Whether separate cua `session`s isolate this is
  *not established*.
* **Trust tier.** Clicks are inferred, not observed, so watch output goes to
  the **draft (human-review) tier**, like video. Low-confidence steps can be
  confirmed in the Beacon log ("which of these did you click?").

#### 5.6.2 Recorded step as a query

* A `QueryChooser` (behind the `Chooser` protocol) turns the recorded
  `(role, label)` into the description our own code would write. It picks
  the exact match, else a same-role near match above a threshold, else
  abstains.
* **Ties abstain.**
* `validate_choice`, the gate and the approver run unchanged. A replayed
  token or coordinate is never used: it would be refused as stale three times
  over.
* Typed text is the skill parameter bound at run time, never the matcher's.
* **Failure modes:**
  * label drift and localisation → abstain;
  * duplicate labels → abstain (the description format cannot carry the
    ancestor path without a core change, deferred);
  * targets beyond `MAX_ELEMENTS`/`max_candidates` → flagged unreplayable at
    record time;
  * a match whose `in_web_content` differs from the recording → refused
    (spoofing).
* A query-resolved choice is deterministic. The log records `chooser: query`
  (§5.4.3).

#### 5.6.3 Splitting Record a Skill out (plan step 6)

Today the shared post-processing lives *inside* the DOM producer's package:
`LiveRecorderService.handle_upload` hard-wires process → actions → gate →
verify → synth → persist (`learning/live_recorder/service.py:103-214`), and
video ingest imports the gate and synthesizer from there
(`video_ingest/pipeline.py:24-25`). The split:

1. **Move.** Move `quality_gate`, `synthesizer`, `step_verifier` and
   `extract_parameters` into `learning/record_skill/`. Leave a one-release
   re-export shim, with identity tests.
2. **Make the funnel producer-neutral.** Add a `FunnelAction` TypedDict. The
   gate reads `application` before the URL. The synthesizer's app name and
   wording follow `metadata.source`/`app`. The verifier prompt stops saying
   "browser recording". §3 D9's measurement becomes the regression test.
3. **Locality (decision 3).** The verifier runs only on providers in
   `_LOCAL_PROVIDERS` (`providers/registry.py:167`); otherwise it is skipped
   with the reason recorded. The guide's "nothing leaves the machine" is
   corrected to match.
4. **Extract the service.** Add `RecordSkillService.handle_actions(actions,
   parameters, metadata, trust)`, and send the archive through
   `redact_capture`.
5. **Point it at the new producer.** A watch→funnel adapter (the same shape
   as `video_ingest/funnel.py`). Stopping a watch session lands a draft for
   review in Beacon's Skills panel. Each desktop step's query is written into
   SKILL.md as an `**Element**` line, which the `QueryChooser` reads back.
6. **Docs.** Rework `record-a-skill.md` into three producers (DOM, video,
   watch) and one post-processing section.

---

## 6. Build order

Small PRs, in the plan's order. Each PR has to prove something specific, and
the PR description must show that proof.

**Flags:**

* ⚑ **ruling** — touches the "computer tools are not registered" ruling or
  its pin test, so it needs Will's go.
* ◆ **core** — edits a decision-5 component additively, so it is worth a
  deliberate look.

The driver leg cannot be covered by CI (`cua.py:1-17`). Any PR that changes
what reaches the real driver carries an **on-box outcome check**: refusal
first, then the positive case, verified by the target app's own output. That
is the #523 shape.

| PR | Repo | Plan step | Change | Proves | Tests | Flags |
|---|---|---|---|---|---|---|
| **0** | Prometheus | 1 | This document | — | — | — |
| **1** | Prometheus | 1 | **The adapter tells the truth.** D3: `ToolResult` verdicts raise on `SUSPECTED_NOOP`/`REFUSED`/`is_error`. D4: drop the phantom `editable`. D5: forward delivery where the SDK takes it; document it per verb. D6: pass through `truncated`, `degraded`, `elements_complete`, `window_title`, `in_web_content`, `selected`, `enabled`, `parent_index`, `depth`; `degraded` makes an observation unusable. D10: a zero-element observation is unusable, so no key candidates are offered into an unseen window. D11: a failed availability check resets the driver, so the next start re-checks. **D12: pin `cua-driver==0.28.2` exactly.** D13: an adapter lock; a timeout reported as "outcome unknown". Honour `delivery.mode` and log `escalation` (§4.1). Fix the "Nine tools" text and the stale "Cua has none" docstring (`cua.py:113-115`). | What the driver reports reaches the loop; a no-op raises for every verb; nothing acts on an empty tree | Translation tests built from **real `cua_driver` types** rather than fakes, so a phantom field cannot hide again. Candidate IDs and descriptions pinned unchanged. | ◆ `types.py` (additive fields) |
| **2** | Prometheus | 1 | **The driver as an Integration.** `ComputerIntegration` (from_config with no I/O; probe under lock + TTL; telemetry floor, worker environment included; supported-version check; `PRIVATE_WORKER` hosting behind an on-box check, with `EMBEDDED` as the fallback; a BOUNDED trusted session per task; cached snapshot for `/api/status`; `app.state.computer_targets`). The `computer_use` block with `enabled: false`. `GET /api/integrations/computer`, `/computer status`. | Health is known before dispatch; disabled constructs nothing; telemetry is off before import | The config guards; a probe-state matrix (disabled, version-mismatch, preconditions down, ready) over a fake SDK module; a subprocess test that the env is set before `import cua_driver`; the no-leak test extended; **`registered == 0` by execution with `enabled: true`** | — |
| **3** | Prometheus | 2 | **Consent before the first grant.** The `site` term (§5.4); the D1 floor in the loop; D2 (type candidates withheld unless focus is established); a runbook step to read the live `security.grants` for `computer_action` rows. | No remembered grant can cover web content; `/gate off` cannot bypass computer consent; the prompt describes only what executes | The 27 literals updated. New: unknown site is not rememberable; `derive_grant` returns None; `from_config_dict` refuses four-term and unknown rows; AUTONOMOUS still reaches the approver (`FixtureDriver.dispatched` empty when it declines). | ◆ extent + loop |
| **4** | Prometheus | 2 | **Discovery.** `Driver` gains `list_apps`/`list_windows` (fixture + adapter); `resolve_app(phrase, aliases)`; frontmost-window choice; per-step re-resolution | "My editor" becomes one window, or a question | 0/1/many matches; never launches; a vanished window ends the task | ◆ `Driver` protocol (additive) |
| **5** | Prometheus | 2 | **The door.** `ComputerTaskInput` + `ComputerTaskRunner`; `/computer` on every chat surface; REST routes; session binding + `SessionConsent`; ceilings; cooperative stop wired into session interrupt; chat progress. **Approvals that reach the starter (§5.1.4): the queue without Telegram, `task_id`/`session_id` tags, routing to the starting chat, `deny_task`, the D14 `finally`.** The task registered as the session's turn (§5.1.7). `enabled: false` by default. **The pin covers every `computer_*` registration path** (§6.1). **Cannot merge before PR 3's D1 fix.** | A person can start, consent, follow and stop a task end to end, and nothing is registered by any path | End to end with `FixtureDriver` + real `SecurityGate` + real `ApprovalQueue`. The binding approves only covered extents; `type_text` always prompts; high-consequence labels prompt; stop between steps denies pending approvals; one task per target. **The widened pin:** a full boot with shipped config and with `enabled: true`, a spy on `ToolRegistry.register`, and the shared `_TOOL_PREFIX`. | ⚑ **ruling** (touches the pin) |
| **6** | Prometheus | 3 | **The cockpit stream.** Layer 1 frames (§5.2.1); `COMPUTER_FRAME_KINDS` + promotion; toggle routes; stop frame; approval `task_id`/`extent`; `since`/`types`/`session_id` on `/api/events/recent`; push: a generic body, and no APPROVAL category when arguments are present | Every step is visible on a phone, and stop works from one | The two copied pinning tests; a content-policy grep test (no tokens, pids, typed text); reconnect backfill; stop ack | — |
| **7** | beacon-desktop | 3 | ProgressPane computer section from a `computer_*` reducer; thread-header toggle + picker; `computer_step` kept out of the Activity feed | — | Renderer smoke for every `computer_*` frame | — |
| **8** | beacon-ios | 3 | **`Approval.arguments` and a "With:" list** (required before any typing approval from a phone); `computer_*` kinds in the decoder list **and** a reducer; COMPUTER section + task strip; picker; Status row | — | Decoder tests for every frame. Pin the kinds list to the server's tuple, so the #503-style silent drop cannot recur. | — |
| **9** | beacon-desktop (+ Prometheus `GET /api/integrations`) | decision 7 | Connectors + Integrations merged; the computer row via contract views | — | The smoke's provider-name grep still passes | — |
| **10** | Prometheus | 4 | **Local chooser.** D7 (`to_thread`), `Choice.reason`, `GemmaChooser`, per-request grammar, boot canary, `chooser.kind: gemma` | The model can only answer with an ID from the table, and a slow model cannot stall the daemon | Grammar builder (one-line form) checked by llama.cpp's validator where available; timeout → abstain with **exactly one** HTTP call; bypass → abstain + `silent_failures`; the canary refuses a non-enforcing backend | ◆ loop (one line), `types.py` |
| **11** | Prometheus (`gym/`) | 4 | Chooser evaluation, tier a (§5.5.3); runbooks for b0 and b; the Cua Bench adapter in a separate 3.12 venv | Accuracy, unsafe-action rate, invalid-ID = 0, latency → `timeout_s` | The harness's own fixtures | — |
| **12** | Prometheus | 5 | **Watch mode.** Diff engine, `WatchSession`, routes, `computer_watch_*` frames | Steps are inferred from app-scoped trees, redacted before storage, and only for the granted app | Password never stored; foreign app refused; truncation never yields "disappeared"; stop within one tick; exclusive with tasks | — |
| **13** | Prometheus | 5 | `QueryChooser`: recorded step as a query | A recording replays through the table, never through a token | Exact match selects; drift, ties and localisation abstain; stale still refused | — |
| **14a-c** | Prometheus | 6 | Split Record a Skill: move + neutral funnel; locality + service + redaction; watch producer + docs (§5.6.3) | Desktop drafts are labelled as desktop; nothing leaves the box | Existing suites unchanged; D9's measurement becomes a regression test | — |

**Before `computer_use.enabled: true` on a real box:** PRs 1-8. That is the
plan's "Beacon action log + stop, required before anyone watches a cursor
move".

**Later, not v1.1:**

| | Change | Flags |
|---|---|---|
| **L1** | Register `computer_task` as a tool, so the model can initiate. It must first meet the registration audit's list (`COMPUTER-USE-REGISTRATION.md` §7). The pin is replaced by the audit's four assertions (§8 of that file). | ⚑ **ruling** |
| L2 | Post-action capture into `VerifyInput` (`actions.py:200-211`) | ◆ |
| L3 | Pixel fallback via cua-perception (driver ≥ 0.33), with screenshot consent | ◆ new verb |
| L4 | Per-run perception metrics | — |
| L5 | macOS / Windows: preconditions provider, role normalisation, host identity (§2.1) | — |
| L6 | OS focus/activation listeners (§2.2) | — |

### 6.1 The pin, precisely

`test_the_daemon_registers_none_today` (`tests/test_computer_status_block.py:324-343`)
asserts that no line in `src/` calls `register_computer_tools(`. That is one
registration path out of many.

The status block already counts every tool named `computer_*`
(`computer/status.py:79,177`), so its answer is right. The pin is what has to
widen.

**PR 5 requirement (Will, 2026-10-03): the pin covers every `computer_*`
registration path.** Concretely:

1. **By execution, through the real boot wiring.** The test builds the tool
   registry the way the daemon does: `build_tool_registry`
   (`daemon.py:952`), then the post-build registrations (LSP, MCP bootstrap,
   the SENTINEL re-registration, and whatever PR 2 and PR 5 add for the
   Integration and the door). It does this twice:
   * with the shipped config;
   * with `computer_use.enabled: true`.

   In both cases it asserts that **no tool name starts with `computer_`**.
2. **A spy on `ToolRegistry.register`** for the duration of that boot. Any
   `computer_*` registration fails the test with the caller's location,
   whatever function, wrapper or dynamic name produced it.
3. **One prefix, shared.** The pin uses `computer.status._TOOL_PREFIX`, the
   same constant the status block counts with, so the two cannot drift.
4. **The existing grep stays**, as the cheap static check for the named
   function.
5. **The shipped default for `computer_use.enabled` is off**, asserted
   against `shipped_defaults` (audit §8, assertion 2).

**Effect on the ruling:**

* PR 5 registers nothing, so every assertion is green.
* This strengthens the ruling rather than changing it. It still edits the
  pinned test file, hence the ⚑ flag.
* The test's message keeps its current spirit, naming the path that
  registered: "That is a deliberate decision; update this test and say so."

**What L1 would do:** change the ruling itself. It replaces the grep with the
four assertions, keeping the "That is a deliberate decision; update this test
and say so" message.

---

## 7. What this design does not do

* **Register anything** (decision 1). `computer.registered` stays 0 through
  PR 14.
* **Foreground delivery, remote targets, `browser_*`, clipboard, launching
  apps, the recording/replay family** (`actions.py:33-49`). They stay out for
  the reasons recorded there.
* **Screenshots.** Observe keeps `include_screenshot=False`. The pixel tier
  (L3) needs its own consent decision.
* **Cloud models or services anywhere in the path** (decision 3). That
  includes the chooser, the evaluation runs, push content, and the Record a
  Skill verifier once watch output reaches it.
* **macOS or Windows in v1.1.** §2.1 lists what they need, and nothing in
  v1.1 blocks it.
* **Rebuild anything Cua ships:** driver, cursor, VMs, benchmarks,
  perception (§4).

---

## 8. Decisions needed

Ordered by when they block. "Needed before" names the PR that cannot be
written without the answer.

| # | Decision | Recommendation | Needed before |
|---|---|---|---|
| **1** | **The origin term for the extent** (§5.4) | **E:** a fifth term `site` (`target:app:site:verb:delivery`). UNKNOWN is never rememberable. In v1.1, web content is **not offered** as candidates (Cua RFC 4268's rule), and browser chrome is approve-once. The alternative is E′: offer web content, approve-once, every time. Either way the term is free only while no real grants exist. | PR 3, and so the door |
| **2** | **The reading of decision 6 under a model chooser** (Q2) | The binding covers model-chosen clicks, scrolls and Return/Tab/Escape inside the picked app, with the §5.1.6 mitigations. The alternative is the audit's "grants serve script origin only", under which every model-chosen click prompts and the toggle grants little. | PR 5 |
| **3** | ~~Does `/gate off` waive computer consent?~~ **Settled by Will (2026-10-03):** the D1 fix is a hard precondition for the door. | **Mechanism:** in the loop, any computer extent the gate allows at `TrustLevel.AUTONOMOUS` (`checker.py:1037`) is routed to `approve`. A grant match allows at `TrustLevel.AUTO` (`checker.py:1067-1071`), so it is unaffected. **Open detail:** `approve` consults stored grants and the binding before prompting, so `/gate off` neither waives nor tightens computer consent. Recommended. | PR 3 |
| **4** | **Typing goes to focus, not to the named field** (D2) | Use RFC 4268's rule: `set_value` by element token, via `call_tool(name, arguments_json)` (`_native.py:5561`; the typed SDK has no element-targeted text input, `_native_contract.py:1657-1705,5862-5865`). Where that is unavailable, withhold `type-N` candidates. Never let the prompt name a field the driver will not target. | PR 3 |
| **5** | **Who picked the element: log, or consent term?** | Log only (`chooser: rule\|gemma\|query` in the audit row and the frame), never an extent term. | PR 3 |
| **6** | **Binding lifetime** | In memory, 8-hour ceiling, re-asked after a daemon restart. A durable binding is a follow-up. | PR 5 |
| **7** | **`/approve all` and computer approvals** | Exclude them, on chat and on `POST /api/approvals/all/approve` (both drain desktop approvals today; probed). | PR 5 |
| **8** | **Widening the pin in PR 5** (§6.1) | Go. It covers every `computer_*` registration path by execution, with a spy on `ToolRegistry.register`, and keeps the grep. | PR 5 ⚑ |
| **9** | **The driver pin** | Exact `cua-driver==0.28.2` **in PR 1** (D12), plus a code-side supported-versions check. Upgrade to 0.33.x deliberately, for the pixel tier. | PR 1 |
| **10** | **Hosting** (Q4) | Cua's `PRIVATE_WORKER`, with a BOUNDED trusted session per task: crash containment, and a driver-side floor. Adopt it if PR 2's on-box check shows that it works on X11 and that the telemetry opt-out reaches the worker. Otherwise stay `EMBEDDED` and accept that a native crash takes the daemon down. | PR 2 |
| **11** | **App-term spelling on macOS** | `bundle_id`, decided with #1, so the first grants on a Mac are never re-spelled. No effect on Linux. | Before any macOS work (L5) |
| **12** | **Where the chooser runs** | A dedicated small-Gemma llama-server (E4B first), not the production slot. A shared single slot turns a 5 s timeout into abstains. | PR 10 |
| **13** | **OS event listeners for watch mode** | Defer. Each is global and would see apps outside the grant. Ask upstream to expose the driver's existing trackers first. | L6 |
| **14** | **Registering `computer_task`** (model-initiated) | Out of v1.1. Revisit with the audit's §7 list. | L1 ⚑ |
| **15** | **Typing approvals on a phone before iOS can show them** | Until beacon-ios shows `arguments` (PR 8), a `type_text` approval is not answerable from iOS: no lock-screen action, and the card says "open on desktop or Telegram". | PR 6 |
| **16** | **The AGPL icon detector in `cua-perception`** | Decide before L3 ships. PP-OCR (Apache-2.0) alone is a possible first step. | L3 |

---

## Appendix A — measuring the tool catalogue

Run from the repo root with `uv run python measure_catalogue.py`. It is
read-only: config and data directories point at a temp dir. Measured at
`856ebb86437971d36ffedd90e9bc1970913057f3`.

```python
"""Measure the tool catalogue the daemon advertises at the checked-out commit.

Mirrors daemon.py: build_tool_registry(security_cfg=config["security"])
(daemon.py:952 -> daemon.py:222-232 -> __main__.create_tool_registry), then
DynamicToolLoader(registry, config["tools"]["deferred_loading"]) (daemon.py:956),
then schemas_for_run(deferred) as agent_loop.py:1595-1599 does, then the
profile filter (agent_loop.py:1641-1643). Config = the shipped template.
No MCP servers, lsp.enabled false, no SENTINEL re-registration.
"""
import json, os, tempfile
from pathlib import Path

SCRATCH = Path(tempfile.mkdtemp(prefix="promcfg-"))
for var in ("PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR",
            "PROMETHEUS_LOGS_DIR", "PROMETHEUS_WORKSPACE_DIR"):
    os.environ.setdefault(var, str(SCRATCH / var.lower()))

import yaml
cfg = yaml.safe_load(Path("config/prometheus.yaml.default").read_text())

from prometheus.daemon import build_tool_registry
from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.computer.tools import build_computer_tools   # NOT registered: sized only

def wire(s):
    return json.dumps(s, separators=(",", ":"), ensure_ascii=False)

def size(schemas):
    s = "[" + ",".join(wire(x) for x in schemas) + "]"
    return len(schemas), len(s), round(len(s) / 4)

registry = build_tool_registry(security_cfg=cfg.get("security", {}))
loader = DynamicToolLoader(registry, cfg.get("tools", {}).get("deferred_loading"))
print("registered      ", size(loader.schemas_for_run(False)))
print("advertised local", size(loader.schemas_for_run(True)))
print("7 computer_*    ", size([t.to_api_schema() for t in build_computer_tools(None)]))
print("computer_task   ", size([{
    "name": "computer_task",
    "description": "Hand a desktop goal to the computer-use subagent, which observes the "
                   "screen and drives the app step by step. Returns what it did and the final state.",
    "input_schema": {"type": "object", "required": ["goal"], "properties": {
        "goal": {"type": "string"}, "app": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]}}}}]))
```

Output of the script above at `856ebb8` (re-run for this document):

```
registered       (51, 41253, 10313)
advertised local (12, 11306, 2826)
7 computer_*     (7, 11578, 2894)
computer_task    (1, 352, 88)
```

* **The two `computer_task` figures.** The bare hand-written schema here is
  352 characters. The 675-character figure in §2.4 is the fuller
  pydantic-generated schema, with field descriptions, titles and `anyOf`
  nulls, that a real `BaseModel` input would emit. §2.4 uses the larger,
  realistic number.
* **The fuller run** adds the per-tool, per-category and per-profile
  breakdowns (profiles on a local tier: full 12, coder 7, research 3,
  assistant 2, minimal 2, symbiote 6). That run is not reproduced here.

## Appendix B — sources

<!-- APPENDIX B -->
