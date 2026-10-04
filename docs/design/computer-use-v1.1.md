# Computer use v1.1 — the door, the cockpit, and the driver as an Integration

## For Will (one page)

This PR is one document: a design. Nothing is built, and nothing
registers. The rest of the file is the cited reference behind this page.

### What is broken today

These must be fixed before anyone uses computer use for real.

1. **`/gate off` switches desktop consent off entirely.** In autonomous mode
   the gate approves every desktop action, typing included, before its
   computer rule runs, and the loop has no override. You verified this. The
   fix (PR 3) is a hard precondition for the door. *(D1)*
2. **A model can approve its own prompts.** The daemon's API token sits in a
   plain file that the always-loaded `bash` tool can read, and every approval
   route accepts it. The same token could start a desktop task through any
   door that accepts it. *(D15)*
3. **Typing goes wherever the focus is, not into the field the prompt
   names.** You approve "type into Search", and the driver types into
   whatever has focus. *(D2)*
4. **Our driver pin lets in versions our code cannot use.** `>=0.28,<1`
   admits 0.28.3 and later, which break every observation. Only the lockfile
   protects us today. *(D12)*
5. **A click prompt does not say what will be clicked**, only the app. That
   is fine while a script picks the button, but not once a model does.
   *(D16)*

### What v1.1 builds

* **The door.**
  * Start it with `/computer <goal>` from Telegram, or from a Beacon toggle,
    where picking the app *is* the consent.
  * Only a person can open it.
  * Nothing becomes a model tool, so `computer.registered` stays 0.
* **The cockpit.**
  * A task shows live in both Beacons today, as rows in the chat, and then in
    a proper action log.
  * Stop works from the phone. It says honestly that one click already in
    flight may still land.
  * After each action, a thumbnail of the approved window goes straight to
    your Beacon. It is kept in memory only, and skipped when a password field
    is showing.
* **The driver as an Integration.**
  * An exact pin, and a health check before every task.
  * Telemetry forced off.
  * The driver runs in Cua's crash-isolated worker.
* **Consent before any grant exists.**
  * A `site` term for browsers.
  * No web page content in v1.1.
* **Then:**
  * a local Gemma that can only answer with an ID from the table;
  * watch mode;
  * splitting Record a Skill out.

### In this order

Each step is a small PR (§6):

1. **PR 1:** the adapter tells the truth, plus the exact pin.
2. **PR 2:** the Integration.
3. **PR 3:** consent fixes: `site`, `/gate off`, typing.
4. **PR 4:** finding the app's window.
5. **PR 5:** **the door** (⚑, including the widened pin you asked for).
6. **PRs 6-8, with 6b:** the cockpit (daemon; per-step thumbnails of the
   approved window; desktop; iOS). **Only after these is it
   enabled on a real box.**
7. **PR 9:** Connectors and Integrations merged.
8. **PRs 10-11:** the Gemma chooser and its evaluation.
9. **PRs 12-13:** watch mode and replay.
10. **PR 14:** the Record a Skill split.

### Your decisions (answered 2026-10-04)

| | Decision | Decided | Before |
|---|---|---|---|
| **W1** | Browsers and Electron apps: how does consent tell a bank tab from a docs tab? | **Agreed.** Add a `site` term now, while no grants exist, and offer **no web page content** in v1.1. Browser menus and tabs ask every time. | PR 3 |
| **W2** | When a model picks the clicks, what does "I picked this app" cover? | **Agreed.** Clicks and Tab/Escape in that app only. Return, typing, menus and "Send/Delete/Pay"-type buttons always ask. Prompts never offer "always". The pick lasts for the Beacon session (8 h at most), or for one task on chat. | PR 5 |
| **W3** | Who may start computer use? ⚑ | **Agreed.** Only a person: an allowed chat user, or a Beacon device you have marked for computer use. Never the API token, which a model can read. | PR 5 |
| **W4** | Close D15 for *every* tool? | **Accepted, separate WP.** Agreed in principle but out of scope for computer use: it becomes its own work package once D15 is verified on the live daemon. The door keeps its own W3 guard regardless. | Not a v1.1 gate |
| **W5** | What may Telegram carry? | **Agreed.** Consent sentences and approval prompts (including the text to be typed), as it does for every tool today. Progress carries counts only. Slack and Discord wait. | PR 5 |

**Decided for you unless you object (Appendix A):**

* the exact pin;
* Cua's crash-isolated worker;
* **the agent cursor off by default on X11** (it can freeze input there), and
  opt-in when you are at the machine;
* `/approve all` skips desktop approvals;
* Gemma on its own llama-server;
* macOS spellings deferred.

**Part 1 answers and the Cua/Hermes table:** §0 and §4 below.

---

**Status: DESIGN PROPOSAL. Nothing in this document is implemented.** No
behaviour, config, test or registration changes ride with it.
`register_computer_tools` still has no call site and `computer.registered`
is still 0.

* **Base:** written 2026-10-03 against `origin/main` @ `856ebb8`. The
  computer-use module last changed in #527 (`c7e26aa`). **Line numbers are at
  `856ebb8`.**
* **Kinds of statement.** Every statement is one of three kinds:
  * **Settled (plan):** from `PLAN-COMPUTER-USE-AND-SKILLFORGE-2026-09-21`,
    **as summarized by Will**. The plan and its companions
    (`MILESTONE-3-ADDITIVE-CHOOSER-ARCHITECTURE`,
    `COMPUTER-USE-WINDOW-IDENTITY-2026-09-20`, `EXTENSION-TAXONOMY`) were not
    available to the author; §1 quotes the summary.
  * **Measured / read:** a `path:line`, a path in the `cua-driver` 0.28.2
    wheel, or a fetched URL.
  * **Proposed:** marked. Will answered the five decisions in §8 on
    2026-10-04; the rest are decided by recommendation in Appendix A.
* **Review.** It was checked by three citation critics, a security
  adversary and a completeness editor before publication.

---

## 0. Summary

**Part 1 (§2).**

* **(a) Platform.** The SDK (`cua-driver`, pinned `>=0.28,<1`, locked
  **0.28.2**, `pyproject.toml:105`, `uv.lock:1067-1068`) is cross-platform,
  and so are the adapter's calls.
  * Two layers bind the module to **Linux/X11 + AT-SPI**:
    * the preconditions (an X socket, plus `gdbus` for the a11y bus) block
      every macOS and Windows step before the driver
      (`driver.py:158-362`);
    * the candidate builder knows only AT-SPI role names
      (`candidates.py:42-49`).
  * The pin is unsafe: any install from 0.28.3 on cannot build our observe
    call (D12).
* **(b) Events.** The driver pushes events only about **its own** calls,
  content-free (`DriverActivityObserver`, `_native.py:2479-2493`).
  * The human's focus and activation can only be **polled** through Cua
    (`list_windows` z-order; uncalled today, `cua.py:147-157`).
  * AT-SPI, AX and UIA push them only to a listener we would write, and every
    such listener is global.
* **(c) SkillForgeRecorder** is a macOS menu-bar app recording **screen
  pixels to MP4** with ScreenCaptureKit (`ScreenRecorder.swift:67-95`).
  * It captures no input, accessibility or app identity, and uploads to
    skillforge.sh (`SkillForgeIntegration.swift:139-169`).
  * It was never shown to record. Decision 8 holds.
* **(d) Catalogue** (by execution, Appendix B): **51** tools registered;
  **12** sent per turn on a local tier, all 51 on a cloud tier (41,253
  characters, ≈10.3k tokens).
  * The seven `computer_*` verbs would add 11,578 characters, more than the
    whole local per-turn set.
  * One `computer_task` adds 865.
  * The door, a command, adds 0.

**Part 2 (§4).**

* **Use Cua's:** the driver (pinned exactly), private-worker hosting, bounded
  trusted sessions as a floor under our gate, the agent cursor (off on X11),
  Lume for evaluation VMs, Cua Bench for end-to-end evaluation, and
  perception regions for the later pixel tier.
* **Keep ours:** the gate, extent, table, `validate_choice` and loop. Cua's
  RFC 4268 converged on the same design, and its rules are adopted as data.
* **Hermes:**
  * It confirms the `cua:<action>:<mode>` key and has no app term; one
    "always" mints "type anything, anywhere".
  * Its driver bugs: #52014 is a tracking issue closed as not planned;
    #32766 is open; #96328 was fixed by #96341. Each maps to a defect class
    we fix or cannot hit.

**Part 3, the design (§5).**

1. **The door** is `/computer <goal>` on Telegram, plus a Beacon toggle and
   composer mode, all funnelled into one `ComputerTaskRunner`.
   * Only a person's credential opens it, never the global token a model can
     read (⚑).
   * The answer to "which app may I use?" becomes a binding, applied as the
     loop's approver, so **the gate is unchanged**.
   * `list_windows` resolves "my editor".
   * Nothing is registered.
2. **The cockpit:**
   * the task renders today in both Beacons' timelines;
   * a durable `computer_*` stream follows the coding live-stream pattern;
   * stop is a cooperative epoch fence ending at a `before_act` seam, and at
     most one in-flight action may land;
   * pushes carry no content.
3. **The driver is an Integration:** an exact pin, a probe before every task,
   `McpRuntime`-style lifecycle, private-worker hosting, and one
   `computer_use:` block, default off.
4. **The fifth extent term `site`** lands before the door, with `-` only on
   positive evidence (0.28.2 never reports a complete walk). v1.1 offers no
   web content.
5. **The chooser** is a local Gemma, constrained by a per-request GBNF
   grammar of the table's IDs (a timeout means abstain).
   * The pixel fallback is perception regions as table rows.
   * It is evaluated offline, then on a VM with the real driver, then on Cua
     Bench.
6. **Build order:** PRs 1-14 (§6).
   * PR 3 precedes the door: its D1 fix is a hard precondition (Will), and
     the site term must land before grants exist.
   * PR 5 (the door) is ⚑. It widens the pin to every `computer_*` path and
     enforces person-only credentials.

**Defects found on the way (§3): 19.** Three block the door:

* **D1:** `/gate off` lets every desktop action through.
* **D2:** typed text goes to focus, not to the field named in the prompt.
* **D15:** the global token, which a model can read, answers any approval.

**Decisions for Will: 5, answered 2026-10-04** (§8). W1, W2, W3 and W5 are
agreed. W4 is accepted as a separate work package, outside computer use.
Everything else is decided by recommendation in Appendix A, and can be
revisited there.

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

None of the evidence refutes a decision. Six findings qualify one, and each
changes what the design must do.

**Q1. Decision 3 (local only) is not true of the parts we would use, as they
ship.**

1. **cua-driver telemetry is on by default, in the binary.**
   * It is content-free and goes to PostHog EU, from the CLI binary crate
     (`libs/cua-driver/rust/crates/cua-driver/src/telemetry.rs`).
   * Evidence: `bin/cua-driver` contains `eu.i.posthog.com/capture/`; `doctor`
     says `enabled via default`; a run here wrote `~/.cua-driver/.telemetry_id`,
     and the proxy refused the upload.
   * **The pinned 0.28.2 binary honours only `CUA_DRIVER_RS_TELEMETRY_ENABLED`
     and `CUA_TELEMETRY_ENABLED`. It ignores `DO_NOT_TRACK`** (cua `fc18825`,
     `telemetry.rs:216-227`; main adds it).
   * The in-process library we load (`cua.py:178`) has no PostHog strings in
     0.28.2 or 0.33.1. That is a strings check only, so *not established*.
   * The private worker *is* the binary. Its environment allowlist passes the
     two variables but not `DO_NOT_TRACK`
     (`libs/cua-driver/rust/crates/cua-driver-sdk/src/embedded.rs:837-870`,
     `worker.rs:141-158`).
   * The update check is unaudited. `src/` sets no opt-out.
2. **Cua Bench and Lume default telemetry on** (`CUA_TELEMETRY` /
   `DO_NOT_TRACK`; `LUME_TELEMETRY_ENABLED`).
3. **Record a Skill's verifier uses the top-level `model:`**, possibly cloud,
   and is on by default (`web/server.py:4029-4064`;
   `docs/guide/record-a-skill.md:33-37`).
4. **APNs approval pushes carry `"<tool_name> — <description>"`**
   (`push/dispatcher.py:100-118`). A computer description names the app and
   the machine (`checker.py:1137-1140`). That contradicts beacon-ios's own
   header ("Apple learns that the daemon had something to say, never what",
   `NotificationService.swift:4-7`).
5. **Chat surfaces are off-box transports.** Telegram's servers see whatever
   the bot sends: today, every tool's approval prompt and its arguments. A
   computer task adds app names, goals and typed text in prompts (§8 W5).

*Consequence:*

* Both `*_TELEMETRY_ENABLED=0` variables are set before the SDK loads and in
  the worker's environment. This is a floor reported by the probe (§5.3.2).
* Watch output never reaches a non-local verifier (§5.6.3).
* Computer pushes are content-free (§5.2.2).
* Chat content is §8 W5.

**Q2. Decision 6 (the toggle IS the grant) has no mechanism today.**

* No grant is session-scoped. `until_restart` spans every session, because
  there is one gate per process (`checker.py:333-345`;
  `approval_queue.py:84-97`).
* Picking an app names no verb or delivery, and grants are exact whole-value
  matches (`checker.py:391-416`).
* Payload verbs are never rememberable (`computer_extent.py:70-78`), and this
  design keeps that.
* In a browser, the app is every site (§5.4).
* Under a model chooser, every covered click is chosen from app text, which
  the 2026-09-20 audit advised against (`COMPUTER-USE-REGISTRATION.md:79-83`).
* This design reads decision 6 as accepting that risk inside the one picked
  app. It applies the binding as the loop's approver (§5.1.4) and adds
  §5.1.6. Will agreed (§8 W2).

**Q3. Decisions 1 and 2: "user-started" needs a credential, and the pin is
narrow.**

* **The status block is right.** `computer.registered` counts every
  `computer_*` name (`status.py:79,177`), so a registered `computer_task`
  would show as 1.
* **The pin test is narrow.** It greps for `register_computer_tools(` call
  sites (`test_computer_status_block.py:324-343`). The door PR widens it to
  every `computer_*` registration path (§6.1).
* **Registering nothing is not enough.**
  * The global API token sits in plaintext in `~/.config/prometheus/env`
    (`config/api_token.py:1-12`), and `bash` is not path-floored
    (`checker.py:984-992`).
  * So a prompt-injected model can read the token and `curl` any route that
    accepts it, including a door route, with no `computer_*` tool registered.
  * The door therefore accepts only a person's credential (§5.1.1, ⚑).
  * The same exposure already lets a model answer *any* approval (D15).

**Q4. Decision 4's "like the backend registry" holds for health and refusal,
not lifecycle.**

* The registry never launches anything (`providers/backends.py:19-26`), so
  start and stop copy `McpRuntime` (`mcp/runtime.py:128-306,396-405`).
* Crash containment needs non-embedded hosting (§5.3.3; Appendix A).

**Q5. The origin term is free only before real grants exist, and the "fifth
term invalidates grants" ruling does not forbid it.**

* That ruling's reason is the cost after grants exist, which is zero while
  `computer.registered` is 0 (§5.4.3).
* **Qualifications:**
  * A padding value (`-`) would make a later migration *possible*, but only
    by reversing refuse-don't-pad (`checker.py:452-476`).
  * `registered == 0` is a proxy: #527 minted a real persistent grant, though
    its probe wrote to a throwaway config
    (`scripts/computer_use_grant_probe.py:38,58,148-149`). So PR 3 first reads
    the live `security.grants`.
* `pid`/`window_id` stay out for their own reasons
  (`computer_schema.py:146-178`).

**Q6. Decision 8 holds for desktop work.**

* Polling sees effects, not causes: there is no focus field in 0.28.2, and
  the driver's push feed cannot see the human (§2.2).
* So watch steps go to the draft tier (§5.6.1).
* The SkillForge Live DOM extension stays the better producer for the web.

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
| Probes / scripts | GNOME default apps. The loop-based probes go through preconditions (`computer_use_probe.py:82,103`; `computer_use_multistep_probe.py:166`; `computer_use_stall_probe.py:169`). `computer_use_grant_probe.py` drives the wrapped-tool path, which has none (D17). | `scripts/computer_use_grant_probe.py:171-183,253,256` |
| CI | **The extra is never installed.** The `uv sync` legs (test, test-macos, quality, security-floors) sync `web anthropic mcp`. The FIRSTLIGHT legs install `[full]` or nothing, and `full` excludes `computer`. So the computer tests pass on Darwin only because they use fixtures. | `.github/workflows/ci.yml:92,148,188,249,133-159,348-397`; `pyproject.toml:101-105` |

**macOS also has a process-identity question.**

* `CuaDriver.create()` loads the runtime into the Python daemon, so TCC
  attribution goes to the host process.
* Upstream treats a raw daemon with no stable bundle identity as unsupported.
  Its supported routes are:
  * the `CuaDriver.app` daemon reached with `connect()`;
  * an `EmbeddedCuaDriverHost` started from an app that owns the grants.
* That choice belongs to the Integration (§5.3) and is in Appendix A.

**The pin.**

* `cua-driver>=0.28,<1` (`pyproject.toml:97-105`), locked at 0.28.2
  (`uv.lock:1067-1068`).
* PyPI has 0.33.1 (2026-10-03), and any version from 0.28.3 on breaks our
  observe call (D12).
* §5.3.1 proposes an exact pin.

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

1. **The activity observer is at most a cross-check.** It is content-free,
   and unavailable in a private worker. The log comes from `StepResult`.
2. **v1.1 watch mode polls** (§5.6).
3. **OS listeners are deferred** (Appendix A).
   * Each is global, so events from apps outside the grant would have to be
     dropped before logging.
   * Asking upstream to expose the trackers the driver already runs may be
     cheaper. That is not "rebuilding what Cua ships", because Cua does not
     ship them publicly.

### 2.3 (c) What does SkillForgeRecorder record, and how? (plan open question 6)

**Answer: screen video only, from a macOS menu-bar app. It captures no
input events and no accessibility data, and it uploads to a cloud endpoint.
Prometheus derives nothing from it. That supports decision 8.**

The repo has two commits, both from 2026-02-19 (`91312dc`, then `9bfc86f`).
Paths below are under `skillforgerecorder/SkillForgeRecorder/`.

| | Finding | Evidence |
|---|---|---|
| **What** | A macOS 14 menu-bar app (`LSUIElement`) that records **screen pixels** to MP4: H.264 High plus AAC, 60 fps / 10 Mbps by default. A JSON sidecar of file stats goes alongside, and is not uploaded. | `Services/ScreenRecorder.swift:67-95`; `Models/Preferences.swift:20-34,62`; `App/Info.plist:25-26`; `project.pbxproj:404,460`; `Models/CaptureItem.swift:43-57` |
| **How** | ScreenCaptureKit `SCStream` → `AVAssetWriter`. It always uses the first display. Nothing is excluded from capture, so the app's own HUD ends up in the video. | `ScreenRecorder.swift:1-4,27-60,217-246` |
| **Modes** | Full screen; region (a drag overlay); window. Window mode has **no picker**: it takes `content.windows.first(where: isOnScreen)`. OCR throws `invalidMode`. Webcam and microphone toggles only set metadata flags; nothing captures from either device. | `ScreenRecorder.swift:36-47,84,191-192`; `Models/RecordingMode.swift:3-7` |
| **Not captured** | **No input events.** 0 hits for `CGEvent`, `NSEvent`, `addGlobalMonitor`, `IOHID`. **No accessibility.** 0 hits for `AXUIElement`, `AXObserver`, `AXIsProcessTrusted`. **No app or window identity** (`frontmostApplication`, `CGWindowList` absent). No per-step timestamps. Accessibility and Input Monitoring are never requested. | grep over all 27 `.swift` files; `Services/PermissionManager.swift` |
| **Where it goes** | Manual "Forge Skill" button. A Supabase password-grant JWT, then a multipart `POST {apiURL}/api/upload` with **the MP4 only** (the sidecar is not sent), to skillforge.sh. Config lives in a gitignored `Constants.swift` (from `Constants.swift.template`), so a fresh clone does not build. App sandbox is off. | `Services/SkillForgeIntegration.swift:59-95,139-169`; `App/SkillForgeRecorder.entitlements:5-16`; `README.md:15-28` |
| **Maturity** | Most features are stubs (e.g. trim's `onApply` discards both times). It was never used to make a recording; it launched once, then looped on the permission. SkillForge was deferred on 2026-04-17. | `Views/PostRecording/PostRecordingPreview.swift:46-47`; oara-brain `wiki/sources/projects/SkillForgeRecorder.md:22`; `raw/claude-chats/2026-02-14-…:4784-4940`; `wiki/sources/projects/SkillForge.md:109-114` |
| **Prometheus use** | **None as a format.** Its file naming and sidecar shape appear nowhere in `src/`, `docs/` or `tests/`. The one name collision, `capturedAt` (`learning/live_recorder/service.py:270`), belongs to the Live DOM extension. An MP4 from it would go through the generic `learning/video_ingest` path, which came from the SkillForge *engine*. | `src/prometheus/learning/video_ingest/__init__.py:1-21`; `pipeline.py:59-66` |

**Implication (decision 8 confirmed).**

* Nothing to port: it is macOS-only and pixel-only, and it uploads to the
  cloud.
* Watch mode replaces it as a step producer.
* If pixels or narration are ever needed, an OS recording fed to
  `video_ingest` serves.

### 2.4 (d) The tool catalogue the anti-bloat rule is calibrated against (plan open question 4)

**Answer: 51 tools registered. 12 are sent to the model each turn on a local
tier, and all 51 on a cloud tier. Seven computer verbs would add as much
schema as the whole local per-turn set; one `computer_task` adds about 8%.
In v1.1 the door adds zero, because it is a command, not a tool.**

The rule's own text is not in the repo. Here is the catalogue, measured **by
execution** at `856ebb8`.

* **How it was measured:**
  * It mirrors the daemon: `build_tool_registry` (`daemon.py:952` →
    `__main__.py:137-396`), then `DynamicToolLoader` (`daemon.py:956`;
    `context/dynamic_tools.py:204-252`).
  * Config is the shipped template, with no MCP and LSP off; a no-config run
    matches.
  * The profile filter (`engine/agent_loop.py:1608-1653`) is a no-op under
    the default profile `full`, so it is not applied.
  * The script is in Appendix B.

| Set | Tools | Schema chars (compact JSON) | ≈ tokens (chars/4) |
|---|---|---|---|
| Registered | **51** | 41,253 | 10,313 |
| Sent per turn, **local** tier (deferral `auto` → on) | **12** | 11,306 | 2,826 |
| Sent per turn, **cloud** tier (deferral `auto` → off) | 51 | 41,253 | 10,313 |
| Deferred when on (`tool_search` / exact name) | 39 | 29,948 | 7,487 |
| *+ the 7 `computer_*` tools* | 7 | **11,578** | 2,894 |
| *+ one `computer_task(goal, app?, text?, target?)`* | 1 | **865** | 216 |

* **The 12 sent on a local tier:** `bash`, `task_create`, `read_file`,
  `write_file`, `edit_file`, `grep`, `glob`, `tool_search`, `skill`,
  `web_search`, `web_fetch`, `memory` (`config/shipped_defaults.py:44-61`).
  They are pinned to follow the default by
  `tests/test_always_loaded_follows_the_default.py`.
* **Token counts are chars/4 only.** `tiktoken` is not installed, and the
  proxy refused its encoding download.

**What the increment means:**

* **Seven verbs:**
  * 11,578 characters: more than the whole local per-turn set (+102% if
    always loaded).
  * +28% on cloud tiers either way, because `auto` sends everything there
    (`dynamic_tools.py:237-238,250-252`).
  * `computer_type_text` alone would be the third-largest schema.
* **One `computer_task`:**
  * 865 characters: +7.7% local if always loaded, +2.1% full.
  * About 13× smaller, and it keeps the table and `validate_choice` in the
    path (decision 2).
  * When L1 registers it, defer it: 0 per local turn.

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
measured or probed (probe numbers come from scratch harnesses, not
re-runnable here without a display). Each is assigned to a PR in §6. None is
fixed by this document.

| # | Defect | Evidence | Why it matters for v1.1 | Fixed in |
|---|---|---|---|---|
| **D1** | **`/gate off` allows every desktop action, and the loop has no override.** In `PermissionMode.AUTONOMOUS` the gate returns ALLOW before reaching the computer rule. `agent_loop` forces a prompt for an *unknown* extent in its own path; `ComputerUseLoop.step` has no such override. | `checker.py:1026-1037` runs before `checker.py:1130-1150`; `engine/agent_loop.py:5000-5008` (unknown extents only); `computer/loop.py:173-207`. **Measured:** `SecurityGate(mode=AUTONOMOUS).evaluate(…)` returns `allowed=True, requires_confirmation=False` for both a known and an unknown computer extent; DEFAULT returns `False, True` for both. | A door on today's loop would click and **type** unprompted under `/gate off`. The payload rule would be bypassed too. | PR 3. **Hard precondition for the door** (verified by Will, 2026-10-03). |
| **D2** | **Typing goes to whatever has focus, not to the element the prompt names.** A `type-N` candidate says "Type the prepared text into the entry 'Search'" and carries that element's token. The typed SDK's `TypeTextInput` takes only an `ActionTarget`, whose variants are `WINDOW` and `DESKTOP`. The adapter drops the token. | `candidates.py:91-107`; `_native_contract.py:1657-1705,5862-5865`; `cua.py:451-455` | The approval sentence describes an action the driver is not asked to perform. That is a consent-honesty defect, not a bug in a corner. | PR 3: type candidates focus-then-verify, or are withheld |
| **D3** | **Non-click verdicts never raise.** `click` returns `ActionResult`, while `press_key`/`scroll`/`type_text`/`invoke_menu` return `ToolResult`, whose effect sits at `.action.effect` beside `is_error`/`error_code`. `_effect_name` reads only `result.effect`. | `_native.py:5568,5598,5624,5628,5652,4632-4646`; `cua.py:399-403`. **Simulated** with real SDK types: `ToolResult(is_error=True, error_code="background_unavailable", action.effect=SUSPECTED_NOOP)` maps to `UNVERIFIABLE`, `landed=False`, and does **not** raise. | `SUSPECTED_NOOP` was meant to raise (`cua.py:46-58`). On four of five verbs it cannot. | PR 1 |
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
| **D15** | **A model can answer any approval with the global API token, and could open any door that accepts it.** The token is persisted in plaintext to `~/.config/prometheus/env`, and `bash` (always loaded) is not path-floored, so `cat` reads it. The approval routes accept it. | `config/api_token.py:1-12`; `checker.py:984-992` (documents that `cat ~/.gnupg/x` is ALLOWed at user origin); the auth middleware accepts the global token on every `/api/` route (`server.py:357-372`), including `POST /api/approvals/{id}/approve` (`:4404`). Device enrolment needs only the global token (`server.py:377-385`). | Pre-existing, and it affects every tool's prompts. For the door it would make "user-started" meaningless. | PR 5 for computer routes and computer approvals (§5.1.1), whatever happens to the general fix. The general fix is its own work package once D15 is verified on the live daemon (§8 W4). |
| **D16** | **A click prompt does not say what will be clicked.** The prompt reason is `ComputerExtent.describe()` (app, machine, delivery). `element_token` is hidden as noise, and `Candidate.target_description` ("kept for … the prompt") is read nowhere. | `checker.py:1137-1140`; `permissions/argument_view.py:62-66`; `types.py:114-116` | Under a model chooser, the person approving cannot see the element the model picked. | PR 5 |
| **D17** | **The wrapped-tool path skips preconditions.** `_ComputerTool.execute` calls `driver.act` directly; only the loop checks the substrate. `computer_use_grant_probe.py` drives that path. | `computer/tools.py:57-95`; `scripts/computer_use_grant_probe.py:171-183` | Any registration that wraps the verbs, not the loop, would act on a dead display. | L1 (register `computer_task`, which runs the loop, never the verbs) |
| **D18** | **A legacy `tool` grant naming a computer tool matches every call of it.** It matches any app, any site, and even unassembled extents, because grants are checked before the computer rule. `from_config_dict` still loads `tool` rows. | `checker.py:398-399,452-455,1050-1071` | "No remembered grant can cover web content" is false while such a row exists. | PR 3 (◆ gate) |
| **D19** | **The extent's app term falls back to the caller's claim.** `Observation.app` is `out.app_name or app`, so when the driver reports no app name, the consent term is whatever the caller passed. | `cua.py:265` | A binding re-derived from the same claim would cover an unidentified window. | PR 4 |
| — | `actions.py` says "Nine tools" and lists seven. | `actions.py:31-32` | Cosmetic. | PR 1 |

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

* **Cua:** `trycua/cua` at `0d274d0` (2026-10-03). Upstream paths are
  relative to that repo. Wheels 0.28.3-0.33.1 were compared with the
  installed 0.28.2.
* **Hermes:** `NousResearch/hermes-agent` at `158fd638` (MIT), cited
  `H:path:line`.
* `cua.ai` is blocked from this environment (Appendix C).

### 4.1 Cua

| Item | What Cua offers now | Prometheus today | Verdict |
|---|---|---|---|
| **Driver SDK versions** | **0.33.1** on PyPI (2026-10-03), ten releases after 0.28.2 (2026-09-15). **0.28.3 makes `max_image_dimension` a required keyword of `GetWindowStateInput`** and adds typed perception (`parse_visual_regions`, `CAPTURED_COORDINATES`). Later: `ActionResult.summary/error`; 0.31.0 BREAKING "snapshot store invalidated on read" (#3873). Unchanged through 0.33.1: no `editable`/`focused` field; only `ClickInput` takes a delivery mode. (`libs/cua-driver/rust/CHANGELOG.md`; PyPI JSON) | Wrapped. Pin `>=0.28,<1`, lock 0.28.2. Any version from 0.28.3 on breaks our observe call (D12). | **Use theirs, pinned exactly** (`==0.28.2`). Upgrade deliberately, passing the new keywords. |
| **Hosting modes** | `EMBEDDED` (in-process). **`PRIVATE_WORKER`**: "one supervised child runtime over inherited pipes; no listener", upstream's mode for "native crash containment"; it takes no host callbacks. `DAEMON` (`cua-driver serve`). MCP (`cua-driver mcp`). `REMOTE`. (`docs/content/docs/cua-driver/guides/use-the-sdk.mdx:224-242`; `libs/cua-driver/docs/sdk-first-runtime-north-star.md:101,403,407`; `_native.py:5074-5085`) | `EMBEDDED` only (`cua.py:109-116,178`). "Cua has none" (no remote transport, `cua.py:113-115`) is stale. | **Use theirs: `PRIVATE_WORKER`**, subject to an on-box check (Appendix A). **Not needed:** `DAEMON`, MCP (raw tools the gate cannot read), `REMOTE` (decision 3). |
| **No-foreground contract** | Background means no raise, no pointer move, no frontmost switch, and it "never retries in the foreground on its own". When impossible it returns `background_unavailable`. `escalation` is advice; `delivery.mode` reports what happened. (`how-cua-driver-works.mdx:41-43`; `use-the-sdk.mdx:211`; `libs/cua-driver/docs/action-result-contract.md:43-44,127-140`) | Foreground is a separate grant (`computer_schema.py:180-187`). Delivery is passed for click only (D5); `_verdict` drops `escalation` and `delivery.mode` (`cua.py:364-403`). | **Use their contract, keep our gate.** Detect a `delivery.mode` mismatch (it is reported after the fact), end the task, and log `escalation` (PR 1). |
| **Agent cursor** | `set_agent_cursor_*`: an overlay whose badge names the session and delivery, "a visual aid, not an authorization signal". On by default; works in background; kept out of captures since 0.30.1 (#4199). On macOS an in-process runtime returns `facility_unavailable`, so a worker is needed. (`agent-cursor.mdx:13-80`; `operate.mdx:259-293`; `use-the-sdk.mdx:238-241`) | None. | **Use theirs.** Off by default on X11 (§4.2, §5.2.4). |
| **Authorization layers** | STANDARD (default; unprompted), **BOUNDED** (a manifest naming apps, a display flag and TTLs; deny by default), UNRESTRICTED. `TrustedSessionOptions` + `create_trusted_session`; ending a session removes its grants. "Code running inside the runtime's process can bypass them." (`permissions.mdx:8-110,180-187,266-279`; `_native.py:2363,3879-3893,7426-7451`) | Absent: `create()` means STANDARD. | **Use theirs as a floor under our gate:** a BOUNDED session per task (§5.3.3). It is real only out of process. Our gate stays the per-action consent. |
| **Activity observer** | Content-free events of the driver's own calls (§2.2); not available in a worker. | Absent. | **Keep ours** (`StepResult`) for the log. |
| **Telemetry** | On by default, to PostHog EU. On main, precedence runs `DO_NOT_TRACK` → `CUA_DRIVER_RS_TELEMETRY_ENABLED` → `CUA_TELEMETRY_ENABLED` → `CUA_TELEMETRY` → `~/.cua/config.toml` → persisted. **The pinned 0.28.2 binary ignores `DO_NOT_TRACK`** (`libs/cua-driver/rust/crates/cua-driver/src/telemetry.rs:216-241`; tag `fc18825` `:216-227`). Detail in Q1. | No opt-out is set in `src/`. | **Not needed** (their telemetry). Forced off by our floor (Q1, §5.3.2); eval rigs add `LUME_TELEMETRY_ENABLED=0`. |
| **Lume / Lumier** | Lume 0.6.0 (2026-10-01): macOS and Linux VMs on Apple silicon; at most two macOS guests. Lumier 0.1.3: VMs in Docker. (`libs/lume/CHANGELOG.md`). The PyPI package `lume` is unrelated. | Absent. | **Use theirs, for evaluation VMs only.** Never a runtime dependency. |
| **Cua Bench** | `cua-bench` 0.3.0 (2026-10-01), Python ≥ 3.12. Local gVisor/runc/QEMU/Lume runtimes. Datasets: `cua-bench-basic` (13), kicad (25), workflows; OSWorld and ScreenSpot-Pro adapters. ATIF-v1.8 trajectories; pass@k. Agents subclass `BaseAgent`. Its `cua-sandbox` 0.9.0 pins `cua-driver==0.28.2`, the same as our lock. | Absent. The venv is 3.11, and `cua.py:178` hard-codes `create()`. | **Use theirs to evaluate the loop** (§5.5.3), in a 3.12 venv, local, telemetry off, `cua-sandbox>=0.9`. Needs a driver-factory seam. |
| **Trajectory recording / export** | MCP `start_recording`/`replay_trajectory` save every action **with arguments and screenshots**, and replay re-invokes them. Computer History is a nightly-only preview. Cua Bench writes ATIF. (`trajectories.mdx`; `recording.mdx:13-19,89-94`) | Excluded: "the recording/replay family" (`actions.py:42-43`). | **Not needed.** Recording persists typed text and frames, and replay bypasses the table and the gate. *Later:* export our own log as ATIF. |
| **Grounding models** | cua-agent 0.9.0: UI-TARS (a `predict_step` loop and `predict_click`), GTA1 (`predict_click` only), all emitting raw `(x, y)`. OmniParser now needs the deprecated AGPL `cua-som`. **`cua-perception` 0.2.1:** local CPU ONNX regions (icon detector AGPL-3.0, PP-OCR Apache-2.0), with capture-bound single-use clicks typed since 0.28.3. (`libs/python/agent/cua_agent/loops/{uitars,gta1}.py`; `perception-extension.mdx`) | Absent; no screenshots. | **Use theirs for the pixel tier: perception regions as table rows** (§5.5.2). **Not needed:** grounders as actors (an open action space); they rank at most. |
| **jev-use / RFC 4268 / Cua-S1** | RFC 4268 "native accessibility candidates" (2026-09-29) is our design. The chooser sees `{id, description}`; `reobserve`/`abstain`; 2-32 rows. It adds: a per-platform role-class map; eligibility filters (enabled, on-screen, labelled, not web content); stable IDs; `set_value` by token; risk tags. Cua-S1: a local LoRA decision model (≤ 26 options). TypeSafe Jev is hosted. (`rfcs/4268-…md:175-272,421-562`; `libs/cua-driver/examples/jev-use/decision-models.md:86-137`) | Same shape (`types.py:3-8,118-129`; `candidates.py:156-189`). Gaps: index IDs, AT-SPI roles only, no filters, a 40-row cap, D2. | **Keep ours. Adopt RFC 4268's rules as data** (role map L5; filters; `set_value` = the D2 fix; risk tags = §5.1.6). **Not needed:** TypeSafe Jev (remote). Cua-S1-4B is an optional evaluation arm. |

### 4.2 Hermes `computer_use`

| Item | What Hermes does now | Prometheus today | Verdict |
|---|---|---|---|
| **Tool shape** | **One** model-facing tool, `computer_use(action=…)`, with 14 actions, including raw coordinates, key combos, drag and free text. It returns a screenshot plus a numbered element list (labels ≤ 120 chars). MCP stdio to `cua-driver mcp` (lock `cua-driver-rs` 0.21.0). darwin/win32/linux. (`H:tools/computer_use/schema.py:17-185`; `tool.py:561-584`; `pm/lock.json`) | No model-facing tool; a table, an ID chooser and `validate_choice`. | **Keep ours.** Raw coordinates and text cannot pass a table (decision 5). **Borrow** the label cap. |
| **Approval key** | **Confirmed:** `cua:<action>:<background\|foreground>` (`H:tool.py:395-398`), plus `cua:bring_to_front:<mode>`. No app, machine or payload term; the prompt does not name the app (`tool.py:396-399,455-458`). | `target:app:verb:delivery`, exact match (`computer_extent.py:65-68`; `checker.py:400-416`). `computer_schema.py:32` describes Hermes correctly. | **Keep ours.** The app term is what makes a grant refusable. |
| **Approval modes** | manual \| smart \| off, plus `/yolo`. Once / session / always: "session" is a real per-session set, and "always" writes `command_allowlist`. The smart guardian is never used for computer use. It **fails closed with no human, unless a cron, unattended or single-query context is configured to approve** (`H:tools/approval.py:250-253,286-292,381-393,973-1020`). | Process-wide or persistent grants only (`checker.py:333-345`); no approver means refuse (`loop.py:195-206`). | **Keep ours.** Their per-session store is the shape our binding needs, built in our runner (§5.1.4). |
| **Typed text** | "Always" is offered for `type`. One "always" on `type "hello"` stores `cua:type:background`: **any text, any app, permanently**. Plus a shell-pattern denylist (`H:tool.py:42-69,343-344`). | Payload verbs are never rememberable (`computer_extent.py:70-78`). **Our gap is D2.** | **Keep ours; fix D2.** |
| **Overlay cursor** | Cua's overlay. **Off by default** on macOS, headless Linux, WSL and **Linux X11**, where it can stick and block input (`H:tools/computer_use/cua_backend.py:45-65`; issues #28152 and #83473 are cited there, not fetched). | None. | **Use Cua's; copy the X11-off default.** |
| **Driver bug #52014** | *"Tracking: Windows computer_use reaches cua-driver, but capture returns only explorer.exe desktop layer"*. **Closed, not planned** (a tracking issue). Upstream cause: trycua/cua#2013. | The Windows form cannot reach us. **The class does:** a plausible but empty tree (D10). | **Keep ours** (a wrong app fails the extent); **fix D10.** |
| **Driver bug #32766** | *"computer_use (cua-driver backend) is too fragile and breaks auxiliary vision routing"*. **Open.** An empty `list_windows(on_screen_only)` gives a 0×0 capture and can leave the backend broken. Fix PR #33054 is open, with changes requested. | We do not call `list_windows` yet. The "left broken" analogue is D11. | **Keep ours;** fix D11, and make PR 4's empty-list case a test. |
| **Driver bug #96328** | *"[Bug]: macOS computer_use rejects current notarised CUA Driver and misses symlinked app path"*. **Closed** by PR #96341 (merged 2026-08-27). | We never launch the app bundle: `create()` "never launches `cua-driver`" (`_native.py:5702-5706`). | **Not needed;** it informs L5 (TCC identity `com.trycua.driver`, `H:tools/computer_use/permissions.py:1-6,21`). |
| **Surfaces** | Model-started in every chat toolset, Telegram included (`H:toolsets.py:12-41,217`); chat-button approvals; a generic `/stop`; no per-action log; "Bot Screen" VNC with a **lease-epoch fence** (`H:tool.py:333-380`). | Nothing yet. | **Keep ours** (user-started, consented app, a log). **Copy the fence pattern** (§5.1.7). **Not needed:** VNC (decision 6). |
| **Trajectory** | Nothing computer-specific. ShareGPT JSONL (`H:agent/trajectory.py:37-40`) is off by default (`save_trajectories: bool = False`, `H:run_agent.py:271`). | None. | **Not needed.** |
| **Local only** | **No:** screenshots go to a vision model (`H:tool.py:692-700,869-946`). It does set `CUA_DRIVER_RS_TELEMETRY_ENABLED=0` and strips provider keys from the driver env (`H:tools/computer_use/cua_backend.py:33-34,68-70,137-180`). | No frames; no telemetry opt-out set. | **Keep ours; use their telemetry practice** (Q1). |

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
                              │                                                 :verb:delivery)   │            └ DriverActivityObserver (EMBEDDED only)
                              ▼                                                                   ▼
                     computer_* frames ──► SignalBus ──► ws_server ──► Beacon (log + Stop)   binding (SessionConsent)
                                                                                             or computer approval channel
```

* **What is new:** the runner and the Integration.
* **What keeps its contract:** the loop, the table, `validate_choice` and the
  gate.
* **Core edits**, all additive and flagged ◆ in §6:
  * **PR 3:** the `site` term and its evidence rule (extent); the web-content
    filter and D2 (candidate table); the D1 floor, the `before_act` seam and
    `computer_approval_for` (loop); the legacy `tool`-grant fix D18 (gate).
  * **PR 1:** new `Observation`/`Element` fields.
  * **PR 4:** `Driver.list_apps/list_windows`.
  * **PR 5:** the task tag and the approve-once flag on approval requests
    (gate, `_call_approve`).
  * **PR 10:** the D7 `to_thread` seam and `Choice.reason`.

### 5.1 The door: one user-started `computer_task(goal)`

#### 5.1.1 Shape, and who may open it

```python
class ComputerTaskInput(BaseModel):    # computer/task.py — also the future tool's schema
    goal: str                          # what to do, in the person's words
    app: str | None = None             # "my editor", "Firefox"; None → ask
    text: str | None = None            # the ONLY text that may be typed; never extracted from goal
    target: str | None = None          # a declared target name; None → the single declared local target
```

* **One entry point serves every surface:**
  `ComputerTaskRunner.start(ComputerTaskInput, *, session_id, surface,
  requested_by)`. It returns a task id at once.
* **Nothing registers it as a tool in v1.1** (decision 1). Registering this
  schema later is a thin wrapper (decision 2's "without rework"); that is
  L1, and it needs a go.
* **`text` is separate from `goal`,** because `text_to_type` must come from
  the caller, never the chooser (`candidates.py:60-63`).

**"User-started" is enforced by credential, not by convention (Q3, D15).**
The global API token is in plaintext in `~/.config/prometheus/env`
(`config/api_token.py:1-12`), and `bash`, which is always loaded, is not
path-floored (`checker.py:984-992`). So any route that accepts the global
token is reachable by a prompt-injected model. PR 5 therefore requires:

* **Starting a task, setting or clearing a binding, and answering a computer
  approval** accept only a *person's* credential. That means either a message
  from an allowed chat user on a chat surface, or an enrolled **device
  identity** on Beacon. The global token gets 401.
  * iOS already enrols as a device; Beacon desktop must enrol too (PR 7).
  * The `/computer` slash path through `/api/chat` or the WS `send_message`
    follows the same rule.
  * **Enrolment needs a human step too.** `POST /api/devices` accepts the
    global token alone (`server.py:377-385`), so a model could mint itself a
    device token.
    * The door therefore honours only devices that a person marked
      `computer: true`, from Telegram or from an already-enrolled device.
    * Alternatively, enrolment itself gains a confirmation on an existing
      surface (§8 W3).
* **`ComputerTaskRunner.start` refuses inside a tool-execution context.**
  That is a context variable that `agent_loop` sets around tool execution, so
  no in-process path from a model reaches it either.
* **This is the substance of the registration ruling**, even with nothing
  registered. PR 5 carries ⚑ for it, not only for the pin.

#### 5.1.2 Reachable from any message surface

* **Each surface registers its own thin handlers** around the shared `cmd_*`
  core: Telegram `CommandHandler`s (`gateway/telegram.py:394-470`), Slack
  `/prometheus-<name>` (`gateway/slack.py:441-500`), a Discord app-command
  tree (`gateway/discord.py:969-1062`), and Beacon through the web router's
  `_SESSION_COMMANDS` (`commands.py:2060-2076`; `web/slash_router.py:111-119`).
* **A parity test covers all four** (`tests/test_gateway_parity.py`,
  `PLATFORMS` at `:94-120`, `MANIFEST` at `:184`).
* **No gateway has inline-button callbacks or a message-edit primitive,** and
  Telegram sends non-command text to the agent (`telegram.py:496-499`). So
  every answer is itself a command.

| Surface | Start | "Which app may I use?" | Progress | Stop |
|---|---|---|---|---|
| Telegram | `/computer <goal> [app:<name>] [text:"…"]`. A shared `cmd_computer` core **returns at once and spawns the task**, as `cmd_gepa` does (`commands.py:2999-3002`): awaiting it would block PTB's single update fetcher, and with it `/approve` and `/computer stop` (`telegram.py:472-495`). | One match: a yes/no prompt that renders the binding's own sentence (§5.1.4). Several: a numbered list answered with `/computer use <n>`. | Milestone messages: started, each approval, a "still running" heartbeat (`gateway/heartbeat.py:368-415`), the end. Not a message per step, and no edited status message. | `/computer stop [id]` |
| Slack / Discord | Registered, but **refuses in v1.1** with the reason. Approvals reach only Telegram and Beacon today (the queue's chat transport is Telegram, `daemon.py:1496-1529`). Recorded as a parity gap. | — | — | — |
| Beacon desktop / iOS | Toggle on + app picker (the binding), then the composer's "Do it on <app>" mode → `POST /api/computer/tasks`. A typed `/computer …` also routes, through `_SESSION_COMMANDS["computer"]`, whose `CommandContext` gains a sender and the runner (`commands.py:1643-1683`). | The picker, filled from `GET /api/computer/apps` | §5.2 | The existing chat Stop (§5.1.7) |
| REST | `POST /api/computer/tasks {session_id, goal, app?, text?, target?}` | `409 needs_app` with the candidate list | `GET /api/computer/tasks/{id}` | `POST /api/computer/tasks/{id}/stop` |

#### 5.1.3 Resolving "my editor" to a real window (D8: `list_windows` is uncalled today)

1. **A forced health probe** of the Integration (§5.3.2). If it is down,
   refuse with its reason, as a dead backend is refused
   (`server.py:4919-4928`).
2. **`list_apps` + `list_windows(on_screen_only=True)`**
   (`_native.py:5602,5618`; `WindowInfo` at `_native_contract.py:4367-4380`).
3. **Match the phrase**, in this order:
   * an operator alias (`computer_use.apps.aliases: {editor:
     [gnome-text-editor]}`);
   * a case-insensitive match on `AppInfo.name`, `bundle_id`, or the
     `launch_path` basename;
   * otherwise ask.

   One running match with an on-screen window → propose it. Zero or several →
   ask, listing names only.
4. **Never launch anything** (`launch_app` stays out, `actions.py:42-44`).
5. **The window is the app's frontmost on-screen window** (max `z_index`).
   * It is re-resolved before every step; a vanished window ends the task.
   * pid and window id never reach the extent, the prompt or the log.
6. **The app term must come from the driver (D19).**
   * Today `Observation.app` falls back to the caller's claim when the driver
     reports no `app_name` (`cua.py:265`).
   * For a door task, an observation is unusable unless the driver reports
     an `app_name` that normalises to the bound app.
7. **Electron apps are limited in v1.1.** VS Code, Slack, Discord and
   Obsidian put their UI under a document node, so they offer only
   approve-once rows until a site provider exists (§5.4.3). The picker says
   so.
8. **App-term spelling on macOS** (`bundle_id`) is fixed by the first grant,
   so it is decided with the site term (Appendix A).

#### 5.1.4 The answer is the consent grant (decision 6), without changing the gate

* **The answer creates a binding** `(session_id, target, app)`.
  * It lives in the runner, not in `security.grants`. It records `set_by`
    (surface and device identity), `created_at` and `expires_at`.
  * It is modelled on the security-relevant `session_workspaces` binding
    (`memory/lcm_conversation_store.py:243-248`; `server.py:1672-1731`).
* **Its scope depends on the surface,** so the sentence the person reads is
  exactly what is granted:
  * **Beacon:** the session, until the toggle goes off, the session ends, or
    8 hours pass. The 8 h is a floor, the same figure cua uses for its
    browser-profile grants (`libs/cua-driver/rust/crates/cua-driver-core/src/browser/grant.rs:14-15`).
  * **Chat surfaces:** **the task that asked**, and nothing after it.
  * **Neither** is persisted across a daemon restart in v1.1 (Appendix A).
* **The gate is unchanged.** For any computer extent with no stored grant, the
  gate returns APPROVE (`checker.py:1130-1150`), and the loop calls its
  injected `approve` (`loop.py:79,183-194,263-282`).
* **For a door task, that callback is `SessionConsent.approve(tool_name,
  reason, *, arguments)`.** `arguments` is keyword-only and required.
  1. It re-derives the extent with the **same** `computer_extent_for` and
     schema the gate used (`loop.py:170-172`). Missing arguments, an unknown
     extent, or any exception means **prompt**, never binding approval.
  2. If the extent and arguments are covered, it returns True and writes an
     audit row `confirm_approved: binding <id>` through the redacting
     `AuditLogger`.
  3. Otherwise it forwards to the computer approval channel (below), tagged
     with the task id.
* **What the binding covers:** `target:app:-:click:background`, and
  `press_key` with Tab or Escape. That is all, and it requires site `-`, i.e.
  positively no web content (§5.4.3).
  * The `press_key` extent does not name the key, and the closed set includes
    `backspace`/`delete` (`actions.py:217-220`). So the key argument is
    checked.
* **What always prompts:**
  * **Return**, which activates whatever has focus, a default "Send"
    included. The label list cannot see that target.
  * `type_text` and `invoke_menu` (payload: never rememberable).
  * Any extent with site UNKNOWN.
  * A label on the high-consequence list (§5.1.6).
  * Anything past the per-task approval ceiling.
  * There are no scroll rows: `build_candidates` offers none
    (`candidates.py:78-118`).
* **Door prompts are approve-once only.** A prompt forwarded for a door task
  offers no "until restart" or "always". A flag on the approval request
  suppresses the lasting scopes (◆ gate).
  * Otherwise one "always" on "Click the push button 'Send'" would mint
    `…:click:background` for every session, surface and origin, and outlive
    the binding.
  * The binding is the lasting consent. Stored grants from before keep
    working, under the `before_act` checks (§5.1.5).
* **The sentence** comes from the same `describe()` machinery:

  > Prometheus may click and press Tab/Escape in gnome-text-editor on mini, in
  > the background, and read everything shown in its windows (not just the
  > front one), until you turn this off [Beacon] / for this task [chat].
  > Return, typing, menus and anything on a web page ask every time.

**Approvals have to reach the person who started the task.** Today they
cannot (all measured):

1. **The queue exists only with Telegram plus a flag that ships off**
   (`daemon.py:1496-1529`). Without it, `gate.request_approval` returns False
   (`checker.py:1632-1634`), so a Beacon-only install refuses every desktop
   action. Building the gate-wide queue whenever computer use is on would
   override the operator's flag for *every* tool. So PR 5 adds a **computer
   approval channel**: an `ApprovalQueue` instance used only by
   `SessionConsent`, emitting the same frames. Other tools are unaffected
   (Appendix A).
2. **Prompts always go to the default Telegram chat** (`checker.py:1639-1645`;
   `daemon.py:1506`). The channel tags each entry with `task_id` and
   `session_id` (`approval_queue.py:54-79,412-464,509-517`) and routes it to
   Beacon and to the starting Telegram chat.
3. **`deny_task(task_id)`** resolves a task's pending approvals as denied.
   That is how stop unwinds a waiting step.
4. **D14:** add the missing `finally`.
5. **`/approve all` and `POST /api/approvals/all/approve` skip computer
   entries**, including the binding prompt, and report how many they skipped.
   Today both drain them (probed; `commands.py:2618-2635`; `server.py:4452,4465`).
6. **The prompt shows what will be acted on (D16).** That is the capped,
   app-text-marked element description, plus the typed text for `type_text`.
   Today a click prompt names only the app and the machine.
7. **A phone must see what it approves.**
   * iOS's `Approval` has no `arguments` field (beacon-ios
     `Models.swift:446-470`), yet both the lock screen and the in-app card
     offer "Approve once" (`NotificationController.swift:113-125`;
     `BeaconUI/Components/ApprovalCardView.swift:154`).
   * So the server refuses, with 409 and "open on desktop or Telegram", an
     approve of a payload-bearing computer approval from an iOS device that
     has not declared the capability. PR 8 builds declare it, e.g. an
     `X-Beacon-Caps: approval-arguments` header.
   * PR 6 sends no APPROVAL category for such requests.

#### 5.1.5 Running a task

* **One task per target at a time.** Watch mode is exclusive with it (§5.6).
* **Each iteration:**
  1. stop check;
  2. re-resolve the window;
  3. `await loop.step(goal, target, app, pid, window_id, text_to_type=text,
     history=history)` (`loop.py:94-104`);
  4. emit the step frame;
  5. branch on the step status:

| `StepResult.status` | Runner does |
|---|---|
| `executed` | Continue. History gains "Clicked push button 'Save' — window changed", passed via `history=`. |
| `reobserve` | Continue. `max_reobserve` (3) in a row ends the task. |
| `abstained` | End: nothing serves the goal, or it is done. |
| `refused` | End, with the reason. |
| `blocked` | End, with the precondition reason. |

* **Ceilings** (keys in §5.3.4): `max_steps` 20, wall clock 600 s, approvals
  per task 10, `max_reobserve` 3. The approval ceiling answers the audit's
  "per-turn ceiling on computer-use approvals (well below 500)"
  (`COMPUTER-USE-REGISTRATION.md:201-204`).
* **A `before_act` seam (◆ loop, PR 3).** `ComputerUseLoop` gains an optional
  synchronous `before_act(candidate, extent, decision) -> bool`. It is called
  with **no await between it and the `act` dispatch**, on every path: grant
  match, binding, prompt and AUTONOMOUS. The runner puts the per-action checks
  there, so they hold even when a stored grant allows and no approver is
  called:
  * the stop epoch;
  * the high-consequence list;
  * the key set;
  * the approval ceiling;
  * the `app_name` rule.
* **D1 is fixed in the loop (PR 3), a hard precondition for the door**
  (verified by Will at `checker.py:1026-1037`, 2026-10-03).
  * A computer extent the gate allows at `TrustLevel.AUTONOMOUS`
    (`checker.py:1037`) is routed to `approve`. A grant match allows at
    `TrustLevel.AUTO` (`checker.py:1067-1071`) and passes untouched.
  * This extends `agent_loop.py:5000-5008`, which forces only unknown extents,
    to known extents allowed at AUTONOMOUS. `agent_loop.py:4991-4999` is the
    legacy-gate branch.
  * **With no approver, an AUTONOMOUS allow is refused.**
  * The prompt's reason comes from a gate helper (`computer_approval_for(extent)`,
    the `describe()` sentence) that also registers its target. The
    reason-keyed target cache (`checker.py:1322-1341`) is never fed a bare
    "Auto-allowed".
  * **Parity:** under `/gate off` the gate never reaches its grants check. So
    `approve` consults the stored grants and, from PR 5, the binding before
    prompting. `/gate off` neither waives nor tightens computer consent
    ("the floor is not a mode", `checker.py:1021-1025`).

#### 5.1.6 Mitigations for model-chosen clicks under a binding (Q2)

The binding plus a model chooser is the combination the audit warned about.
Inside the one picked app, the design accepts it (§8 W2) and adds:

1. **Payload verbs and Return always prompt.**
2. **No web content in the table** (§5.4.3).
   * Browser chrome can still carry page-authored text: window titles,
     history and bookmark items.
   * Such rows have site UNKNOWN, so they prompt.
3. **High-consequence labels prompt even when covered, or under a stored
   grant** (checked in `before_act`). Send, delete, remove, pay, transfer,
   purchase, submit, confirm, sign: Cua RFC 4268's risk tags as data (§4.1).
   * It is a denylist on app text, so it is a mitigation, not a control.
   * It cannot see what Return or a Tab-then-Return would activate, hence
     item 1.
4. **App text is marked and capped where it crosses a boundary:** the
   chooser, the prompt and the log. The cap is 120 characters, as Hermes uses
   (`H:tools/computer_use/tool.py:561-563`).
   * It is applied there, not inside `Element.describe()` (`types.py:59-66`),
     so candidate descriptions stay pinned (PR 1).
5. **Door prompts are approve-once** (§5.1.4). The approval ceiling holds,
   and `/approve all` skips computer entries.
6. **The binding is per app and per session or task**, never
   `until_restart`.
7. **A live log, and a stop that works from a phone** (decision 6, §5.2).

#### 5.1.7 What stop guarantees

* **The mechanism is cooperative.** A stop epoch is checked:
  * by the runner before each step;
  * by the chooser wrapper after the chooser returns (if stopped, it answers
    `abstain`);
  * by `SessionConsent` **on entry, before enqueueing**, and on every return;
  * **in `before_act`**, the last synchronous point before dispatch.

  After the `before_act` seam there is no await before `act`. That holds
  whether or not the chooser later runs in a thread (D7). This copies the
  *pattern* of Hermes's lease-epoch fence (`H:tools/computer_use/tool.py:333-380`).
* **On stop**, `deny_task(task_id)` resolves the task's pending approvals. The
  waiting step returns `refused` through its normal path (`loop.py:188-194`).
  * `task.cancel()` is not the mechanism. Mid-approval it leaks the entry
    (D14); mid-act it skips verify, so no `StepResult` describes the
    dispatched action (`loop.py:211-223`).
* **One dispatched action can still land (D13).**
  * Cancelling the awaiting coroutine, or the 60 s timeout, does not stop the
    SDK call.
  * The guarantee is therefore: **no new action starts after the stop is
    acknowledged, and at most one already-dispatched call may land.**
  * It is reported as "in flight at stop — may have landed" and checked by
    one post-stop observe.
  * This needs one step at a time per adapter: the runner serialises, and
    PR 1 adds the lock.
  * The UI says this, not "stopped instantly".
* **The approval wait.** Up to 1800 s can pass between the observation a step
  was built from and its approval (`approval_queue.py:44,640`).
  * When the wait exceeds 30 s, a floor, `SessionConsent` returns False with
    "the window may have changed while you decided — looking again".
  * The next step re-observes and asks again.
* **Every stop control reaches the task, without taking over the chat turn.**
  * Beacon's Stop (`POST /api/chat/interrupt`, the WS `interrupt` frame)
    cancels `_turn_tasks[session_id]` (`ws_server.py:487-497,790-805`;
    `server.py:1242-1272`). That is one slot, owned by the chat turn's lock
    (`ws_server.py:1192-1226`).
  * The runner does **not** use it. Instead, `interrupt_turn(session_id)`
    first calls `runner.stop_session(session_id)`, which stops every computer
    task in the session, then cancels any chat turn as today. It returns
    "stopped" if either was running.
  * A chat message during a task therefore neither hides the task from Stop
    nor gets cancelled in its place.
  * Both Stop buttons work with no client change: desktop
    `ChatShell.tsx:1238-1253`, iOS `ChatController.swift:238-245`. Whether
    iOS shows Stop for a task it did not start is *not established*.
  * Chat surfaces use `/computer stop`.

### 5.2 The cockpit: a live action log and a stop control

The log comes first; the cursor is a nicety (decision 6). There are two
layers. The first needs no client change; the second is durable.

#### 5.2.1 Layer 1: the task renders today, in the session's timeline

The runner emits, under the task's chat `session_id`, the frames both clients
already render:

* **The task:** `tool_call_start {session_id, call_id, tool_name:
  "computer_task", inputs: {goal, app, origin: "user_task"}}` and
  `tool_call_end {session_id, call_id, tool_name, success, result}`. iOS
  requires `call_id`, `tool_name` and `success` (beacon-ios
  `Frames.swift:125-153`).
* **Each step:** a nested pair with `call_id = <task_id>:<seq>`, `tool_name =
  computer_<verb>`, `inputs = {description, extent}`, and `result` = the
  verification line.
  * Desktop renders these in its chat timeline and Tool feed, iOS in its
    ToolStripView (`ChatShell.tsx:193-215`; `gateway-events.ts:93-128`;
    `ChatStreamReducer.swift:173-187`).
* **Liveness:** `agent_progress` every 3 s with `phase: "tool"`
  (`ws_server.py:1503-1526`).
* **The end:** `chat_done {interrupted}`, **only if no chat turn is live in
  the session**. Otherwise it would clear iOS's `turnInFlight` in the middle
  of a chat turn (`ChatStreamReducer.swift:152-157`).
* **No `turn_completed`.** It would push a summary line to APNs
  (`push/dispatcher.py:126-136`).

**Costs:** `tool_call_*` frames are never persisted
(`gateway-events.ts:130-133`) and look like model-issued calls.
`origin: "user_task"` marks them. Hence layer 2.

#### 5.2.2 Layer 2: the durable log

* **Kinds are declared once, by the emitter.** `computer/livestream.py`
  declares `COMPUTER_FRAME_KINDS`, and `ws_server._on_signal` promotes from
  that tuple, exactly as for `CODING_FRAME_KINDS` (`coding/livestream.py:44-63`;
  `web/ws_server.py:27-35,1640-1642`).
  * An unpromoted kind ships as a generic `sentinel_signal` that every
    `type`-keyed client gate misses. That happened to `coding_tool` and to
    `task_completed` (`ws_server.py:1626-1660`).
  * Both pinning tests are copied.

The envelope is the existing `{type, timestamp, payload}`, and every payload
carries the chat `session_id` (ProgressPane keys on it, beacon-desktop
`ProgressPane.tsx:138-162`).

| `type` | When | `payload` |
|---|---|---|
| `computer_binding` | Toggle on/off, expiry, task end on chat | `{session_id, state: "on"\|"off", target, app, describes, scope: "session"\|"task", covers: ["click","press_key:tab","press_key:escape"], set_by: {surface}, expires_at}`. The device identity is in the audit row, not the frame. |
| `computer_task_started` | Task accepted | `{session_id, task_id, target, app, goal, chooser: "rule"\|"gemma"\|"query", started_by: {surface}, limits: {max_steps, max_seconds, max_approvals}}` |
| `computer_step` | After every step, and on `awaiting_approval` | `{session_id, task_id, seq, status: "executed"\|"refused"\|"abstained"\|"reobserve"\|"blocked"\|"awaiting_approval"\|"in_flight_at_stop", action: {verb, description, app_text: true}, extent, consent: "binding"\|"grant"\|"prompt"\|null, approval_request_id, chooser: {name: "rule"\|"gemma"\|"query", confidence, reason}, effect: "CONFIRMED"\|"PARTIAL"\|"UNVERIFIABLE"\|null, verified: true\|null, after_stop: bool, candidates_offered, duration_ms, reason}` |
| `computer_task_ended` | Terminal | `{session_id, task_id, outcome: "done"\|"abstained"\|"stopped"\|"refused"\|"failed"\|"limit", reason, steps, approvals, duration_ms}` |
| `computer_stream_error` | The log failed (never the task) | `{session_id, task_id, detail}` |

* **Field notes:**
  * `seq` is the de-dupe key (beacon-desktop `coding.ts:339-343`).
  * `after_stop: true` marks the in-flight call; its `effect` comes from the
    post-stop observe.
  * The `computer_watch_*` kinds are defined with watch mode (PR 12).
* **Approvals reuse `approval_pending`/`approval_resolved`,** with optional
  `task_id`, `session_id` and `extent`. Clients ignore extra keys
  (`approval-push.ts:81-101`; beacon-ios `Models.swift:463-469`). Never drop or
  retype `extents` or `created_at`.
* **Content policy.** It is enforced by a test over every emitted *and
  persisted* payload and over the backfill responses:
  * **Never:** element tokens, snapshot ids, pid or window id, screenshots,
    the labels of candidates *not* chosen. (The per-step thumbnail of
    §5.2.5 is not an exception: it never enters this stream. It is a
    direct-only frame, sent to a device and never persisted or backfilled.)
  * **Typed text** is `{"text_chars": N}`, computed by the emitter.
    `redact_arguments` alone would render it
    (`permissions/argument_view.py:51,69-101`).
  * **The one carve-out is `approval_pending`**, which must show the typed
    text for consent to mean anything. Its live frame carries it (with
    `pid`/`window_id` removed as noise). But the copy that SignalBus
    persists to `signal_events` (`sentinel/signals.py:105-120`) has
    arguments replaced by `{text_chars}`, and no backfill route ever returns
    arguments.
* **Backfill reuses what exists.**
  * On connect the server replays nothing (`ws_server.py:244-249`).
  * `GET /api/events/recent` takes only `limit` and one `type`
    (`server.py:3178-3202`), though the tracker supports `since` and several
    types (`telemetry/tracker.py:1598-1605`). PR 6 passes `since`, `types`
    and `session_id` through.
  * Retention is `computer_use.action_log.keep_per_session`, with the pruner
    in PR 6.
* **Older clients are safe.**
  * Desktop shows an unknown kind as a "system" row in the Activity feed,
    **with the whole payload, exportable** (`gateway-events.ts:84,286-287`;
    `ActivityFeed.tsx:120-162`). Hence the content policy.
  * iOS drops an unknown kind (beacon-ios `Frames.swift:5-8,296-299`).
* **The iOS trap.** A new kind needs a decoder-list entry *and* a reducer
  case, or it vanishes silently. `coding_tool`/`coding_acceptance` have been
  dropped on iOS since #503 (`Frames.swift:230-235`;
  `CodingRunReducer.swift:119-127`).
* **Desktop keeps `computer_step` out of the Activity feed**, as it keeps
  `coding_round` out (`gateway-store.tsx:104`). Otherwise it would take over
  Mission Control's activity card (`MissionControl.tsx:697-710`).
* **Push carries no content (Q1).** For computer approvals and the binding
  prompt, the push carries only `request_id` and `expires_at` with a fixed
  body ("Prometheus needs a decision"), and no top-level `tool_name` (today it
  has one, `push/dispatcher.py:100-118`). `test_approval_push_body_stays_narrow`
  is updated to assert exactly that. The iOS extension already fetches
  details over the tailnet (`BeaconNotify/NotificationService.swift:52-67`).
* **The driver activity observer** is at most a cross-check when hosted
  `EMBEDDED`. It does not exist in a private worker (§4.1).

#### 5.2.3 Client → server

* **Stop:** the session interrupt (§5.1.7). Also:
  * a WS `{type: "computer_task_stop", payload: {task_id}}` →
    `computer_task_stop_ack {task_id, stopped}`, which mirrors
    `interrupt`/`interrupt_ack` (`ws_server.py:487-497`);
  * `POST /api/computer/tasks/{id}/stop`;
  * `/computer stop`.
* **Toggle:** `PUT /api/sessions/{id}/computer {target, app}` (→ `on`),
  `DELETE` (→ `off`), `GET`. This is the `session_workspaces` route shape.
* **Credentials:** device identity only (§5.1.1).

#### 5.2.4 Where it shows

| | Live log + Stop | Toggle + picker | Health |
|---|---|---|---|
| Beacon desktop | Layer 1 in the chat timeline. Layer 2 in the ProgressPane's reserved computer-use section (`ProgressPane.tsx:1-13,237-243`, today "Not connected…"). The chat Stop. | Thread header, beside the per-conversation autonomy chip (`ChatShell.tsx:1788-1799`) | The merged Integrations list (§5.3.5) |
| Beacon iOS | Layer 1 in the ToolStripView. Layer 2 as a COMPUTER section beside Status → CODING (`StatusView.swift:166-253`), plus a task strip above the composer. The composer Stop. | The model line above the composer (`ChatView.swift:154-170`) | A `computer` row in Status (`StatusView.swift:118-140`) |
| Telegram | Milestone messages (§5.1.2) | `/computer` reply | `/computer status` |

**The cursor: use Cua's** (`set_agent_cursor_*`, §4.1); nothing is built.

* The default is `computer_use.cursor: off` on X11, where Hermes found the
  overlay can stick and block input (§4.2).
* `on` is opt-in for someone at the machine, and labels the session with the
  task id.
* This narrows the brief's "the driver's own cursor when you're at the
  machine" to opt-in (Appendix A).

#### 5.2.5 Cockpit level 2: a thumbnail of the approved window after each step

Asked for by Will, 2026-10-04. Level 1 is the log and Stop above (both of
its layers). Level 2 adds one picture after each action, for the person
watching from a phone. Level 3, a live view, is later (L7) and has no
design yet.

**What is captured: the approved window, and nothing else.**

* **The window:** the bound app's frontmost on-screen window, re-resolved
  as before every step (§5.1.3) and confirmed by the driver's own
  `app_name` at capture time (D19). If it vanished or now belongs to
  another app, nothing is captured.
* **Never** the desktop, a display or another app's window. Nothing calls
  `get_desktop_state` or a screen-capture verb.
* **How:** one window-scoped `get_window_state` call with
  `include_screenshot=True`, `include_accessibility_tree=True`,
  `screenshot_out_file=None` and `max_dimension` set to the thumbnail size.
  * The driver scales the image and returns it inline
    (`WindowStateOutput.images`, `SnapshotImage{mime_type, data_base64}`;
    `screenshot_width`/`screenshot_height`), so no image library is added.
  * The password check below reads the tree from **the same call**, so the
    check and the pixels describe the same moment.
* **Not established on 0.28.2, measured on the box:** whether an X11 window
  capture can contain other windows' pixels (an overlapping window, a
  notification). If the driver reports `screenshot_frame_valid: false`, or
  the on-box check shows overlap is possible, the capture is skipped.
* **When:** after each **executed** step, once that step's verification is
  done.
  * Not after a refused, abstained or blocked step.
  * Not while an approval is pending.
  * Not for the call that was in flight at a stop.
* **Size:** longest edge ≤ 480 px. A frame over 256 KB is skipped
  (`too_large`), never sent in pieces.
* **It never reaches a model.** Not the chooser, the step history, a
  prompt or the audit row. The capture ruling stays true for every decision
  path: `observe` still asks for no screenshot (`cua.py:229-233`). This is a
  picture for a person.

**Redaction: skip, never blur.**

* **No thumbnail when the window shows a password field.** That means any
  node of the **whole** walk, tokenless ones included (as for
  `web_content_seen`), whose role contains `password` (AT-SPI
  `password text`).
* **Positive evidence, as for the site term.** If the walk is degraded or
  truncated, a password field can't be ruled out, so there is no
  thumbnail.
* **A skip is visible.** The step frame says so, with a reason:
  * `password_field`;
  * `incomplete_walk`;
  * `window_changed`;
  * `capture_failed`;
  * `frame_invalid`;
  * `too_large`;
  * `no_viewer`.
* **Nothing is blurred or partly sent.** A masked image still shows the
  layout around a secret, and a blur only promises the pixels underneath
  are gone.
* **The limit:** it can't see a secret shown outside a password field (a
  token in a text box, a document). The thumbnail shows what a person
  sitting at the machine would see in the app they picked.

**Where it goes: straight to the device, over the existing WS.**

* **Who receives it:** the runner sends it directly to WS connections that
  meet three conditions:
  1. they authenticated with the device token of a device marked
     `computer: true` (§5.1.1, W3);
  2. they declare the `computer-thumbnails` capability;
  3. they are attached to the task's session.
* **It uses identity the bridge already keeps** per connection
  (`ws_server.py:328-332`; `DeviceIdentity.is_global`).
* **A global-token connection never receives one,** so a model holding that
  token can't watch the screen through it (D15).
* **The capability matters for older clients.** An old desktop renders an
  unknown kind's whole payload into its exportable Activity feed
  (`gateway-events.ts:84,286-287`), so a client that doesn't declare the
  capability is never sent the frame.
* **Not through SignalBus.** SignalBus persists every signal to
  `signal_events` (`sentinel/signals.py:105-120`), and a thumbnail must not
  be. So `computer_step_thumbnail` is a **direct-only** kind: it is outside
  `COMPUTER_FRAME_KINDS`, never promoted, never in `/api/events/recent`,
  never in a backfill.
* **Never** push notifications (APNs), Telegram, Slack or Discord, logs or
  telemetry.
* **No viewer, no capture.** If no eligible device is connected, nothing is
  captured (`no_viewer`). The picture exists only to be watched.

**Retention: memory only, unless Will opts in.**

* **By default** the frame is built, sent and dropped.
  * The runner keeps only the latest thumbnail of each running task in
    memory, so a device that reconnects mid-task gets the current picture
    once.
  * It is discarded when the task ends.
* **Opt-in:** `computer_use.thumbnails.persist: true` (default `false`).
  * Each sent thumbnail is written to
    `<data>/computer/thumbnails/<task_id>/<seq>.<ext>`, owner-only (0600).
  * It is pruned with the action log (`action_log.keep_per_session`).
  * Even then, no route serves them in v1.1, and they never enter
    `signal_events`.

**The frame.**

```
{"type": "computer_step_thumbnail", "timestamp": "...", "payload": {
  "session_id": "...", "task_id": "...",
  "seq": 7,                      # the computer_step it follows
  "app": "gedit",                # the bound app, as the extent names it
  "mime_type": "image/png",      # as the driver returned it
  "width": 480, "height": 300,   # after the driver's scaling
  "data_base64": "...",
  "captured_at": "2026-10-04T12:00:00Z"}}
```

* **Excluded:** pid, window id, window title, snapshot id and element data.
  The title is left out on purpose: in a browser it is page-authored text.
* **`computer_step` gains two fields:** `thumbnail: "sent" | "skipped" |
  null` and `thumbnail_skip_reason`. Those go through the ordinary
  persisted stream; the image never does.
* **Config, in `computer_use:`:**
  * `thumbnails.enabled` (`true`);
  * `thumbnails.max_dimension` (`480`);
  * `thumbnails.persist` (`false`).
  * The 256 KB cap and the password skip are floors, not keys.
* **Clients:**
  * desktop shows the latest thumbnail at the head of the ProgressPane's
    computer section;
  * iOS shows it in the COMPUTER section and the task strip;
  * both add the kind to their decoder lists (the iOS trap above) and send
    the capability.
* The log does not depend on it.

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
  * Cua Bench's `cua-sandbox` 0.9.0 pins the same 0.28.2.
* **A code constant names the versions the adapter was validated against.**
  The probe compares it with `cua_driver.__version__`; a mismatch reports
  `down: version-mismatch`. Config cannot widen it, because the input
  builders are tied to the SDK's API (`cua.py:222-238,413-473`).
* **An upgrade is a PR that carries the on-box outcome check.** The driver
  leg is uncoverable by CI (`cua.py:1-17`).
* **The next upgrade has a reason:** capture-bound clicks for the pixel tier
  (typed since 0.28.3; target the then-current release, ≥ 0.33.1).
  * It passes `max_image_dimension` and `timeout_ms` explicitly.
  * It re-checks D2, D3 and D5.
  * It reads what 0.31.0's "snapshot store invalidated on read" (#3873) means
    for observe → act → verify.

#### 5.3.2 Health known before every task

The probe body runs in `to_thread`, under a per-Integration lock, inside
`wait_for`. Failures are recorded, never raised (`providers/backends.py:386-421`).
It checks:

1. **The telemetry floor is in place.** This is new in PR 2: nothing in
   `src/` sets either variable today.
   * `CUA_DRIVER_RS_TELEMETRY_ENABLED=0` and `CUA_TELEMETRY_ENABLED=0` are set
     before the `import cua_driver` in `_require_sdk` (`cua.py:94-106`,
     import at `:97`), **and** in `PrivateWorkerOptions.environment`.
   * `DO_NOT_TRACK` is not enough: 0.28.2 ignores it, and the worker
     allowlist drops it.
2. **The SDK imports and its version is supported.**
3. **Platform preconditions both pass** (today's `check_preconditions`,
   `driver.py:158-174`). This step becomes a per-platform provider later
   (§2.1).
4. **The runtime starts and reports `is_available()`** (`cua.py:168-189`).
5. **`list_apps` returns at least one app.** The observe half is real, not an
   empty tree shaped like a working one.
6. **The target is bound.** `app.state.computer_targets` is finally assigned
   (`server.py:833-836`).
7. **No MCP server's command resolves to `cua-driver`.** Otherwise the probe
   reports `degraded: cua-driver-also-configured-as-mcp`, because raw
   `mcp__` tools would bypass the table (§6.1).
8. **If the chooser is `gemma`, its backend is healthy**
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
* **Hosting: Cua's `PRIVATE_WORKER`** (§4.1), subject to PR 2's on-box
  check (Appendix A).
  * It gives decision 4's "supervised" a real process boundary. Today a native
    fault takes the daemon down (`cua.py:109-116`).
  * It costs the activity observer, which the log never needed.
  * `EMBEDDED` stays for CI.
* **A driver-side floor (use theirs): a BOUNDED trusted session per task.**
  * The manifest names the picked app, allows only `get_window_state` and our
    five verbs, sets `desktop.display: false`, and its TTL is the task
    budget. Stop calls `end_session`.
  * It is real only out of process. Whether it works in a worker is *not
    established* (PR 2 measures it).
  * Our gate stays the per-action consent.

#### 5.3.4 Config: one `computer_use:` block (the `computer:` keys fold into it)

* **Why this name:** the only existing reader already says `computer_use`
  (`targets.py:137-157`), with its error text and tests. It has no call site
  and no template entry, so the block can still be shaped freely.
* **The `/api/status` wire key stays `computer`** (`server.py:971`).

| Key | Default | Validation |
|---|---|---|
| `computer_use.enabled` | `false` | Only a literal `true` enables it. `resolve_telegram_enabled` uses `bool(value)`, so a quoted `"false"` would enable that one (`shipped_defaults.py:255-267`, `bool(value)` at `:267`); this key must not repeat that. Off means: the driver is never constructed and nothing is registered. |
| `computer_use.targets` | `{}` | Existing grammar (`targets.py:50`). v1.1 accepts `kind: local` only; `remote` is refused under decision 3. A bad entry goes to `config_errors`, never a boot failure. |
| `computer_use.apps.aliases` | `{}` | Map of word → list of app names. |
| `computer_use.task.max_steps` / `max_seconds` / `max_approvals` / `max_reobserve` | `20` / `600` / `10` / `3` | Positive ints, through the existing `_positive_int` coercer (`shipped_defaults.py:343-360`). |
| `computer_use.chooser.kind` | `rule` | **Closed set** `{rule, gemma}`. An unknown value is a config error and the Integration reports down; there is no silent fallback. |
| `computer_use.chooser.backend` / `timeout_s` | — / `5` | The backend is a **name** in `backends:`, never a URL (`providers/backends.py:16-17`). |
| `computer_use.probe.ttl_s` / `timeout_s` | `60` / `5.0` | As `backend_probe` (`config/prometheus.yaml.default:1216-1220`). |
| `computer_use.cursor` | `off` | `on` \| `off` (§5.2.4). |
| `computer_use.action_log.keep_per_session` | `200` | The pruner ships in the same PR. |
| `computer_use.thumbnails.enabled` / `max_dimension` / `persist` | `true` / `480` / `false` | §5.2.5. `persist` is the only way a thumbnail reaches disk; the 256 KB cap and the password skip are floors. |
| `computer_use.watch.*` | (§5.6) | Lands with watch mode. |

* **Floors, deliberately not keys:**
  * telemetry off (Q1);
  * the supported driver versions;
  * `MAX_ELEMENTS`;
  * the browser exclusion;
  * background-only delivery;
  * the 8 h Beacon binding ceiling;
  * the 30 s approval-wait limit;
  * the browser/Electron floor list (§5.4.3).
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
  `commands.py:618-640`). It ships with the `/computer` family in PR 5.
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

**The question.** In a browser, `mini:firefox:click:background` covers a bank
tab and a docs tab alike. "Firefox" is consent to every site in it, signed-in
sessions included.

**Naming.** The gate already uses *origin* for user-vs-system trust
(`checker.py:232-233,947`; `loop.py:80,178`). This document says **site**.

#### 5.4.1 What can supply a site, and whether it can be trusted

* **Cua 0.28.2 supplies no URL.**
  * `WindowStateOutput`, `WindowElement` and `WindowInfo` have no url field
    (`_native_contract.py:6311-6336,6101-6117,4367-4380`; measured from
    `list_tools_json()`).
  * Its only origin-attested path is the excluded CDP `browser_*` surface
    (`actions.py:36-39`).
* **The window title is page-controlled.** `document.title` sets it (MDN).
* **The address bar is chrome, but editable.** An unsubmitted edit would let
  a bank page match a docs grant.
* **The accessibility document URL is the right source**, trustworthy only on
  the document node, and named differently per engine:
  * Firefox `DocURL`;
  * Chromium on Linux `URI`;
  * macOS `AXURL` on `AXWebArea` (on a link it is the page's href);
  * Windows the Document's Value.
  * A page can change its path but not its origin (`pushState`), so the
    unit is the origin.
* **The driver does give:**
  * `in_web_content` per element (`_native_contract.py:6101-6117`), dropped
    today (D6);
  * on Linux, the role strings that produce it. A node is web content when an
    ancestor's role contains `document` or equals `embedded`, or it sits on a
    WebKit bus (cua `fc18825` `platform-linux/src/atspi/native.rs:262-273`).
* **Cua's authorization layer is not a site source** (§4.1).

#### 5.4.2 Options

| | What a `firefox` grant means | Unknown site | Code now | Cost if done later | How the picker reads |
|---|---|---|---|---|---|
| **A. No site term; say it** | Every site, forever | n/a | Reword `describe()` (`computer_extent.py:81-91`, `checker.py:538-563`) | A fifth term after grants exist: rows dropped, or rows meaning "every site" | "Firefox — every page, including signed-in sites" |
| **B. Fifth term + a provider now** | One site | Prompt | 4→5 change **plus** a per-engine provider and act-time re-attestation | None | "Firefox, on docs.example.com" |
| **C. Fold the site into the app term** (`firefox@site`) | One site | Prompt | Escaping `@`/`:` (the colon-forging class, `test_computer_use_consent_unit.py:94-101`); substring grouping | Old rows linger, matching nothing | `firefox@…` shown as if they were apps |
| **F. Exclude browsers; use the browser tool** | n/a | n/a | Refuse web apps | — | The built-in `browser` is headless with an isolated context (`tools/builtin/browser.py:134,152`). It cannot act in the person's signed-in browser, so this removes a capability rather than moving it. |
| **G. Web content never rememberable** | Non-web only | Prompt | Small | Same as A if narrowing is wanted later | "Firefox — web pages ask every time" |
| **E. B's shape now; no web content until a provider exists** (recommended) | Non-web; one site once a provider exists | Approve once | 4→5 change; `build_candidates` drops web content (Cua RFC 4268's rule) | **None** | "Firefox — menus and tabs; page content not offered yet" |

#### 5.4.3 Recommendation: E

* **The extent becomes `target:app:site:verb:delivery`.** Scope still
  decreases left to right (`computer_schema.py:38-45`).
* **`site` takes three values:** an origin `scheme://host[:port]`; `-`
  (*positively* no web content); or **UNKNOWN**.
* **UNKNOWN is gated and shown but never rememberable**, through the
  mechanism that makes payload verbs approve-once
  (`ComputerExtent.rememberable`; `approval_queue.py:193-194`).
  `from_config_dict` refuses it as a stored value.
* **`-` needs positive evidence.** The absence of `in_web_content` flags is
  not enough: on Linux the flag only ever arrives as true, so a browser that
  has not exposed its page shows only unflagged chrome. All of these must
  hold:
  1. **The walk is complete by our own evidence:** not `degraded`, not
     `truncated`, and `returned_element_count == total_element_count`.
     ⚠ **cua-driver 0.28.2 hard-codes `elements_complete = false` on Linux**
     ("AT-SPI's current bounded walker does not surface an exhaustive-walk
     proof", cua `fc18825` `platform-linux/src/tools/impl_.rs:891-893`), so
     the driver's own flag cannot be used at the pinned version. Using it
     would make every extent UNKNOWN and the binding inert.
  2. **No node has a document-family role** anywhere in the walk (role
     contains `document`, or equals `embedded`), and none has
     `in_web_content`.
  3. **The app is not on a floor list of browsers, Electron and WebView
     hosts.** It is a floor, not a config key.
  4. **The platform path can detect web content at all.** The upstream
     Windows MSAA fallback cannot, so it never yields `-`.

  Anything else is UNKNOWN. The failure direction is over-prompting, never
  widening. `MAX_ELEMENTS = 200` (`cua.py:86`) truncates big apps, which then
  prompt.
* **PR 3's on-box check records which site a plain GTK app actually gets**
  (gnome-text-editor). If the evidence rule makes it UNKNOWN, the binding is
  inert on the pinned driver. The fix is then the deliberate driver upgrade
  (Appendix A), not a looser rule.
* **v1.1 offers no web content.**
  * `build_candidates` drops elements that have `in_web_content` or a
    document-family ancestor (◆ candidate table, PR 3). This is Cua RFC
    4268's rule: "verify_state … treats web-content elements as an untrusted
    source" (`rfcs/4268-…md:211,470-472`).
  * It removes page-authored labels from the chooser's table, and avoids
    approve-every-web-click fatigue (audit §5).
  * Browser chrome in such a window is UNKNOWN, so approve-once. It can still
    carry page-authored text (titles, history items), so those rows prompt.
  * **Consequence:** Electron apps (VS Code, Slack, Discord, Obsidian) offer
    only approve-once rows in v1.1.
  * The alternative E′ (offer web content, approve-once each time) was not
    chosen (§8 W1).
* **A later provider turns on per-site grants with no migration** (Linux
  first: `DocURL`/`URI` on the document node, over the a11y bus).
* **Encoding and act-time checks:**
  * the site gets its own encoder (`_normalise_term` maps `:` to `_`,
    `computer_extent.py:213-225`);
  * it is re-checked at act time, the way `_assert_target` re-checks the
    machine (`cua.py:338-349`);
  * keys and text go to focus (D2), so `press_key`/`type_text` take the
    window's site.
* **Cost now (measured):**
  * six code sites hard-code four terms (`computer_extent.py:49,68,201-208`;
    `checker.py:472,560-563`; `scripts/computer_use_stall_probe.py:158`);
  * 27 four-term literals in tests, plus `count(":") == 3` at
    `test_computer_use_consent_unit.py:97`;
  * no client change, because both Beacons render `describes` verbatim
    (beacon-ios `GrantRowView.swift:129-136`; beacon-desktop
    `ConfigPanel.tsx:545-557`).
* **Before it lands:** read the box's live `security.grants` for
  `computer_action` rows and for legacy `tool` rows naming `computer_*`
  (Q5, D18).

**Why this must be settled before the door ships.**

1. **Matching is exact, and the term count is enforced.** `Grant.matches`
   compares the whole value (`checker.py:391-416`). `from_config_dict` drops
   a row with the wrong term count, with only a warning
   (`checker.py:452-476`). Matching may never become a prefix test
   (`computer_schema.py:41-45`).
2. **The codebase's own precedent says so.** "The term goes in before
   anything is stored" (`computer_schema.py:47-53`); a fifth term "would
   silently invalidate every stored grant rather than failing"
   (`computer_schema.py:135-143`).
3. **The first grants fix what "app" means.** A row minted under "click
   anything in firefox" is consent to every site. A later change can drop it
   or orphan it, but it cannot narrow it.
4. **The door is where people start answering "which app may I use?".**
   * Door prompts are approve-once, and the binding lives in memory (§5.1.4).
   * But every extent the door shows and audits is keyed on these terms.
   * Anything that mints a lasting grant from then on (the existing `/approve
     always` path, a durable binding, L1) inherits whatever "app" meant.
5. **The picker's sentence is the consent** (decision 6), so it must say what
   is and is not covered from day one.

**Who picked the element is logged, not keyed.** The audit row and frame
record `chooser.name` (`rule|gemma|query`). The audit's "grants serve script
origin only" alternative was not chosen (§8 W2).

### 5.5 The chooser: a local model picking from the table, pixels later

**Its seat.**

* A chooser answers one question, *which of these IDs* (`chooser.py:1-10`).
* It sees only the ID and description (`types.py:118-129`).
* `validate_choice` re-resolves its answer against the client's table and
  the live snapshot (`candidates.py:156-189`).
* Two additive core edits (PR 10):
  * `loop.py:137` becomes `await asyncio.to_thread(self._chooser.choose,
    request)` (D7). The stop fence holds regardless, because of `before_act`
    (§5.1.7).
  * `Choice` gains `reason: str = ""` (`types.py:142-149`).

#### 5.5.1 `GemmaChooser` (plan step 4: Gemma + GBNF first)

* **Prompt.**
  * Built only from `ChoiceRequest`: goal, history, and `id<TAB>description`
    rows plus `reobserve`/`abstain`.
  * Fixed instructions first, so llama-server's prompt cache reuses them.
  * It states that row text is app data, not instructions.
  * Newlines are collapsed, and descriptions are capped (§5.1.6).
* **Grammar, one line per request:**

  ```
  root ::= "click-3" | "click-7" | "key-tab" | "reobserve" | "abstain"
  ```

  * **Validated** with llama.cpp's `test-gbnf-validator` (`836d571`) on a
    44-ID table. All IDs are accepted. Absent, whitespace-padded,
    case-changed and empty outputs are rejected.
  * The multi-line leading-pipe form fails to parse. That is the 2026-07-02
    class (`adapter/enforcer.py:174-181`).
  * IDs must match `^[a-z0-9][a-z0-9-]{0,63}$` (`candidates.py:84,93,112`).
    Anything else means abstain; the table is never repaired.
* **Call contract.**
  * **One direct POST** to a llama-server backend named in `backends:`.
    * Not `provider.stream_message`, which retries up to 3 times
      (`providers/llama_cpp.py:822-833`; `providers/retry.py:28-90`): the
      "silently retried call" that `chooser.py:20-22` forbids.
    * Not `LLMCallEnvelope`, which files a timeout as a user Stop
      (`learning/llm_envelope.py:269-273`).
  * `temperature 0`, `max_tokens ≈ 16`, thinking off (`llama_cpp.py:686-691`).
  * **A timeout means abstain.**
  * **Health first:** abstain if `BackendRegistry.status` is not ok, or if
    `/slots?fail_on_no_slot=1` says the slot is busy.
  * **Output outside the set** means abstain plus a `silent_failures` row.
  * **Boot canary:** a grammar admitting only `abstain`. Anything else means
    the backend ignores grammars, which covers Ollama, whose OpenAI endpoint
    does not list `grammar` (`providers/ollama.py:339-346`). The chooser
    then refuses to start.
* **Which Gemma.** A second llama-server: production is Qwen
  (`docs/MODEL-LADDER.md:532-534`), and there is one model per process
  (`providers/backends.py:21-25`). See Appendix A.
* **Confidence** is recorded and evaluated, and **never authorizes**.
* **The injection boundary.**
  * *Protected:* the chooser has no tools, a finite output, and no say in
    arguments or typed text (`types.py:99-107`; `candidates.py:60-63`).
  * *Not protected:*
    * picking the attacker's preferred legitimate row (`abstain` is its only
      refusal);
    * history poisoning (`loop.py:232`).

    Those are what §5.1.6 mitigates.

#### 5.5.2 Pixel fallback (later): Cua's grounding, our table

* **Use theirs:** cua-driver's `cua-perception` regions
  (`parse_visual_regions`).
  * Local, no network, no Python.
  * Capture-bound single-use clicks (`ClickPosition.CAPTURED_COORDINATES`),
    typed since 0.28.3, **not in 0.28.2** (`_native_contract.py:1788-1826`).
* **Keep ours:**
  * Each region becomes a `Candidate` (`region-<i>`) with a capture- and
    snapshot-bound token. The grammar, `validate_choice` and the gate are
    unchanged.
  * A distinct verb, `click_region`, so a `click` grant never extends to
    OCR-derived targets.
  * Prefer a real accessibility element when one exists.
* **Grounders only rank.**
  * GTA1-7B exposes only `predict_click → (x, y)`. UI-TARS-1.5-7B also has a
    `predict_step` loop, which emits raw coordinates too.
  * So their point is snapped to a region candidate, never dispatched.
* **It needs screenshots** (`cua.py:229-233` deliberately takes none), so it
  needs its own consent decision. Frames go only to the local parser.
* Licensing is in Appendix A.

#### 5.5.3 Evaluation, with Cua Bench where it fits

Cua Bench cannot score an accessibility-table chooser in isolation: its only
no-environment mode (`DatasetSession`) is screenshot-plus-click. That serves
OSWorld-G (564 items; its 54 refusals correspond to `abstain`) and
ScreenSpot-Pro: the *pixel* tier. Hence three tiers:

| Tier | What runs | Corpus | Measures | Where |
|---|---|---|---|---|
| **a. Offline chooser** | `chooser.choose(ChoiceRequest)` as a pure function. A second mode runs `ComputerUseLoop` + `FixtureDriver` + the real gate and asserts on `FixtureDriver.dispatched` (`driver.py:438-442`). | Recorded tables (via `_RecordingChooser`, `scripts/computer_use_multistep_probe.py:63-81`; scrubbed, local, human-labelled), plus synthetic rows: negative, injection, history, duplicates, near-cap, goal-done | Accuracy; abstain and unsafe-action rates; **invalid-ID = 0**; timeouts; p50/p95 (sets `timeout_s`); calibration | `gym/` (deterministic, no judge) |
| **b0. Real driver, local fixtures** | `CuaDriverAdapter` on a local VM or Xvfb desktop serving upstream's static fixtures (`libs/cua-driver-fixtures`), with state readback | Fixture pages and forms | The production driver path, which Cua Bench cannot reach | The box, or a Lume VM |
| **b. Cua Bench end to end** | `PrometheusLoopAgent(BaseAgent)` wraps the loop. A `SessionDriver` adapts our sync `Driver` to Cua Bench's async `DesktopSession`. An eval-only approver records every prompt. | `cua-bench-basic` (13), then offline-capable OSWorld | Reward, pass@k, steps, approvals. Chooser + loop, not the cua element path. | A 3.12 venv, `--on local`, telemetry off (Q1) |

* **Arms:** `RuleChooser` (baseline and fallback), Gemma E4B and 26B-A4B, the
  production Qwen, and optionally Cua-S1-4B.
* Whether Cua Bench images expose AT-SPI to our observe is *not
  established*. If they do not, tier b abstains everywhere, which is why b0
  exists.

### 5.6 Watch mode and the Record-a-Skill split (plan steps 5-6)

An outline only; the door depends on none of it.

#### 5.6.1 Watch mode on our own observe

* **Its own start control and sentence,** separate from the act toggle:
  "Prometheus will read everything shown in every <App> window on <target>
  until you press Stop." Reading is a separate consent (audit §2), audited as
  `…:observe:background`.
* **Polling (§2.2):**
  * `list_windows` about every 1 s, **filtered to the granted app before any
    diff or storage**, because titles of other apps are not consented;
  * `get_window_state` on that app's frontmost window: about 1 s, 300 ms
    bursts after a change, backing off to 3 s, never faster than 250 ms;
  * about 5 s per poll, with no queueing.
  * None of this is measured until the first on-box run, which seeds the
    per-run metrics.
* **A step is a query key:** `(target, app, verb, element {role, label,
  ancestor_path, ordinal}, value_delta, title change, confidence, evidence)`.
  No pid or window id.
* **Inference:**

  | Diff | Step | Confidence |
  |---|---|---|
  | Value change on an editable role (merged) | `type_text` | high |
  | `selected` flipped or moved | toggle / select | high |
  | Title change or new window | view change | medium |
  | Structure change, no value delta | click, *inferred* | low |
  | Keys, scroll, hover, drag | not observable | — |

  A truncated walk never yields "disappeared".
* **Redaction before persistence:**
  * no value for password roles or secret-like labels;
  * `redact_capture` on every string (`security/log_redaction.py:190-214`);
  * only the delta's elements are stored (`computer_schema.py:164-169`);
  * no screenshots;
  * **no web content**: the same filter as the act table (§5.4.3).
* **A foreign `app_name` is discarded. Watch is exclusive with tasks**
  (`cua.py:286-312`). **Draft tier**, like video.

#### 5.6.2 Recorded step as a query

* **How a step replays.** A `QueryChooser` turns the recorded `(role, label)`
  into our own description format. It picks the exact match, else a
  same-role near match, else abstains; ties abstain. `validate_choice`, the
  gate and the approver run unchanged, and no token or coordinate is ever
  replayed.
* **Failure modes:**
  * label drift and localisation abstain;
  * duplicate labels abstain (carrying the ancestor path would be a core
    change, deferred);
  * off-table targets are flagged at record time;
  * an `in_web_content` mismatch is refused.
* **Desktop skills replay only through a user-started `/computer run
  <skill>`** (or Beacon's Skills panel). Until L1, the model-facing `skill`
  tool refuses skills with `**Element**` lines. Otherwise a desktop skill
  would be a route to model-initiated computer use (PR 13 tests it).

#### 5.6.3 Splitting Record a Skill out (plan step 6)

The shared post-processing lives inside the DOM producer
(`learning/live_recorder/service.py:103-214`; `video_ingest/pipeline.py:24-25`).
The split:

1. **Move** gate, synthesizer, verifier and `extract_parameters` into
   `learning/record_skill/`, behind a shim.
2. **Make the funnel producer-neutral.** A `FunnelAction` carries a
   *producer-tagged* element payload (a DOM selector, an AT-SPI query key, or
   nothing for video). There is no common element type and no cross-producer
   resolution, per the ruling against a shared element model. The gate reads
   `application`; the synthesizer follows `metadata.source`/`app`. D9's
   measurement becomes the test.
3. **Locality:** the verifier runs only on `_LOCAL_PROVIDERS`
   (`providers/registry.py:167`).
4. **The service** gains `handle_actions(…, trust)`, and the archive goes
   through `redact_capture`.
5. **Watch producer → funnel**, with `**Element**` lines, as reviewable
   drafts.
6. **Docs:** three producers, one post-processing section.

---

## 6. Build order

Small PRs, in the plan's order. Each PR must show its "Proves" in its
description.

**Flags:**

* ⚑ **ruling** — touches the "computer tools are not registered" ruling, or
  its pin test, or who may start computer use. It needs Will's go.
* ◆ **core** — edits a decision-5 component, additively.

Any PR that changes what reaches the real driver carries an **on-box outcome
check**: refusal first, then the positive case, verified by the target app's
own output (the #523 shape). The driver leg is uncoverable by CI
(`cua.py:1-17`).

| PR | Repo | Plan step | Change | Proves | Tests | Flags |
|---|---|---|---|---|---|---|
| **0** | Prometheus | 1 | This document | — | — | — |
| **1** | Prometheus | 1 | **The adapter tells the truth:** D3-D6 (pass through the `WindowStateOutput` and element fields; `degraded` = unusable), D10 (in `observe`: no elements means unusable), D11, **D12 (exact pin)**, D13 (lock; a timeout means "outcome unknown", then observe). Detect a `delivery.mode` mismatch and log `escalation`. Fix the "Nine tools" and "Cua has none" texts. | What the driver reports reaches the loop; a no-op raises for every verb; nothing acts on an empty tree | Translation tests built from **real `cua_driver` types**. Candidate IDs and descriptions pinned unchanged. | ◆ `types.py` (additive fields) |
| **2** | Prometheus | 1 | **The driver as an Integration** (§5.3): probe, telemetry floor including the worker env, version check, hosting per Appendix A, BOUNDED session per task, cached status, `app.state.computer_targets`, MCP-cua-driver detection. The `computer_use` block with `enabled: false`. `GET /api/integrations/computer`. | Health is known before dispatch; disabled constructs nothing; telemetry is off before import | Config guards; a probe-state matrix over a fake SDK in a **new** `tests/test_computer_integration.py`; a subprocess test that the env is set before `import cua_driver` | — |
| **3** | Prometheus | 2 | **Consent before the first grant:** the `site` term and its evidence rule (§5.4.3); `build_candidates` drops web content; D1 floor + `before_act` seam + `computer_approval_for`; D2 (`set_value` by token, or withhold `type-N`); D18 (legacy `tool` grants). A runbook step reads the live grants. On-box: the site a GTK app gets. | No remembered grant can cover web content; `/gate off` cannot bypass consent; the prompt describes only what executes | 27 literals updated; UNKNOWN never rememberable; a document-node fixture and a chrome-only fixture both give UNKNOWN; AUTONOMOUS reaches the approver and, with no approver, refuses; a `tool` grant does not allow an UNKNOWN click | ◆ extent + gate + candidate table + loop |
| **4** | Prometheus | 2 | **Discovery:** `Driver.list_apps`/`list_windows` (fixture + adapter); `resolve_app`; frontmost window; per-step re-resolution; the D19 `app_name` rule | "My editor" becomes one window or a question; a window with no driver-reported app never acts | 0/1/many matches; never launches; an empty on-screen list asks; `app_name` None or mismatched is blocked | ◆ `Driver` protocol (additive) |
| **5** | Prometheus | 2 | **The door** (§5.1): runner; `/computer` (Telegram; Slack/Discord refuse, recorded as a parity gap; web table); REST; **person-only credentials**; binding + `SessionConsent`; the computer approval channel; §5.1.4 items 1-7; §5.1.6 mitigations; ceilings; cooperative stop wired into `interrupt_turn`; `/computer status`. **The widened pin** (§6.1). Cannot merge before PR 3. | A person can start, consent, follow and stop a task end to end. Nothing a model holds can start one. No `computer_*` tool is registered by any path. | End to end (`FixtureDriver`, real gate, real channel): the binding approves only covered extents; Return, `type_text` and high-consequence rows prompt; door prompts offer no lasting scope; `/approve all` skips computer entries; global token → 401 on every door route; a start from a tool context raises; stop denies pending approvals; a concurrent chat turn plus Stop still stops the task; an old-iOS approve of `type_text` → 409 | ⚑ **ruling** (pin; who may start); ◆ gate (request tag), loop (`_call_approve`) |
| **6** | Prometheus | 3 | **The cockpit stream** (§5.2): layer-1 frames; `COMPUTER_FRAME_KINDS` + promotion; toggle and stop routes; approval tags; redacted persistence of `approval_pending`; `since`/`types`/`session_id` on `/api/events/recent`; `action_log.keep_per_session` + pruner; content-free push | Every step is visible on a phone, and nothing on the wire or at rest carries tokens, pids or typed text outside the live approval | The two copied pinning tests; the content-policy test over emitted, persisted and backfilled payloads; reconnect backfill; stop ack; the narrow-push test | — |
| **6b** | Prometheus | 3 | **Cockpit level 2: per-step thumbnails** (§5.2.5, Will 2026-10-04): after each executed step, one window-scoped `get_window_state` with the screenshot and the tree; the whole-walk password skip and the incomplete-walk skip; the direct-only `computer_step_thumbnail` frame sent only to `computer: true` device connections that declare `computer-thumbnails`; `thumbnail`/`thumbnail_skip_reason` on `computer_step`; the latest frame per task in memory; `thumbnails.*` keys, with `persist` off by default | A phone sees the approved window after each action. Nothing else on the desktop is captured. No frame is stored (unless `persist` is on), backfilled, pushed or sent to a chat. A password field means no picture. | A password role anywhere in the walk, including a tokenless node, skips; a degraded or truncated walk skips; the payload's exact keys (no pid, window id, title or snapshot id); a SignalBus spy sees no thumbnail, and `signal_events` and `/api/events/recent` hold none; global-token and capability-less connections get nothing; no eligible viewer means no capture call; the 256 KB cap; `persist` off writes no file, `persist` on writes 0600 and is pruned; nothing is captured for the call in flight at a stop. On-box: the capture holds only the approved window | ◆ `Driver` protocol (additive capture) |
| **7** | beacon-desktop | 3 | Device enrolment; ProgressPane computer section from a `computer_*` reducer; thread-header toggle + picker; `computer_step` kept out of the Activity feed; the 6b thumbnail at the head of the section, declaring `computer-thumbnails` | Every `computer_*` frame renders; a reload backfills; Mission Control's card stays readable | Renderer smoke for every frame | — |
| **8** | beacon-ios | 3 | `Approval.arguments` + a "With:" list, and the capability header; `computer_*` kinds in the decoder list **and** a reducer; COMPUTER section + task strip; picker; Status row; the 6b thumbnail in the section and the strip, declaring `computer-thumbnails` | A phone shows the typed text before approving it, and no `computer_*` kind is silently dropped | Decoder tests pinned to the server's tuple | — |
| **9** | beacon-desktop (+ Prometheus `GET /api/integrations`) | decision 7 | Connectors + Integrations merged; the computer row via contract views | One list and one health vocabulary; no integration-specific renderer | The provider-name smoke grep still passes | — |
| **10** | Prometheus | 4 | **Local chooser** (§5.5): D7, `Choice.reason`, `GemmaChooser`, per-request grammar, boot canary | The model can only answer with a table ID, and a slow model cannot stall the daemon or slip past stop | One-line grammar checked by llama.cpp's validator; timeout → abstain with exactly one HTTP call; bypass → abstain + `silent_failures`; a non-enforcing backend refused; a stop while the chooser thread finishes dispatches nothing | ◆ loop (one line), `types.py` |
| **11** | Prometheus (`gym/`) | 4 | Chooser evaluation tier a; runbooks for b0 and b; the Cua Bench adapter in a separate 3.12 venv | Accuracy, unsafe-action rate, invalid-ID = 0, latency → `timeout_s` | The harness's own fixtures | — |
| **12** | Prometheus | 5 | **Watch mode** (§5.6.1), with its own start and sentence; `computer_watch_*` frames | Steps are inferred for the granted app only, redacted before storage, with no web content | Password never stored; foreign windows and titles dropped; truncation never yields "disappeared"; stop within one tick; exclusive with tasks | — |
| **13** | Prometheus | 5 | `QueryChooser`; desktop skills run only through `/computer run <skill>`; the skill tool refuses skills with `**Element**` lines | A recording replays through the table, never through a token, and never at a model's initiative | Exact match selects; drift, ties and localisation abstain; the skill tool refuses an Element-bearing skill | — |
| **14a** | Prometheus | 6 | Move the funnel into `learning/record_skill/` behind a re-export shim | Nothing behaves differently | Existing suites unchanged; shim identity tests | — |
| **14b** | Prometheus | 6 | Producer-neutral funnel; locality (verifier on local providers only); service extraction; archive through `redact_capture` | Desktop drafts are labelled as desktop; nothing leaves the box | D9's measurement as a regression test; a cloud `model:` means no verifier call | — |
| **14c** | Prometheus | 6 | Watch producer → funnel; `**Element**` lines; docs | A watch session yields a reviewable draft | Fixture end to end; the query line round-trips | — |

* **Before `computer_use.enabled: true` on a real box:** PRs 1-8. That is the
  plan's "action log + stop, required before anyone watches a cursor move".
* **Later, not v1.1:**
  * **L1:** register `computer_task` (model-initiated). Audit §7 list first;
    the pin becomes the audit's four assertions. ⚑
  * **L2:** post-action capture into `VerifyInput` (`actions.py:200-211`). ◆
  * **L3:** the pixel fallback via cua-perception (driver ≥ 0.28.3; target
    the then-current release), with screenshot consent. ◆ new verb
  * **L4:** per-run perception metrics.
  * **L5:** macOS and Windows (§2.1).
  * **L6:** OS focus and activation listeners (§2.2).
  * **L7:** cockpit level 3, a live view of the approved window. No design
    yet (Will, 2026-10-04).

### 6.1 The pin, precisely

**Today.** `test_the_daemon_registers_none_today`
(`tests/test_computer_status_block.py:324-343`) asserts that no line in `src/`
calls `register_computer_tools(`. The status block already counts every
`computer_*` tool (`computer/status.py:79,177`), so it is the pin that must
widen.

**PR 5 requirement (Will, 2026-10-03): the pin covers every `computer_*`
registration path.**

1. **By execution, at boot.** Build the registry through the real boot
   wiring: `build_tool_registry` (`daemon.py:952`), then LSP, MCP bootstrap,
   the SENTINEL re-registration, and whatever PR 2 and PR 5 add. Do it with
   the shipped config and with `computer_use.enabled: true`. Assert that no
   tool name starts with `computer.status._TOOL_PREFIX`.
2. **Through the door's runtime paths.** A spy on `ToolRegistry.register`
   stays installed through PR 5's end-to-end test (toggle on → `/computer` →
   steps → stop). Any registration fails the test with the caller's
   location, and `_registered_count(registry) == 0` afterwards.
3. **By schema, not only by name.** No registered tool's schema declares
   `x-prometheus-computer-verb`, so a wrapper named `desktop_task` cannot slip
   through.
4. **By caller.** `ComputerTaskRunner.start` raises inside a tool-execution
   context (§5.1.1).
5. **The grep stays.** The shipped default for `computer_use.enabled` is
   asserted off (`docs/audits/COMPUTER-USE-REGISTRATION.md` §8, item 2).

**Out of the pin's scope, but detected.** An operator-configured `cua-driver
mcp` MCP server appears as `mcp__*` tools, which prompt on every call (the
`mcp__` rule). The Integration's probe reports `degraded:
cua-driver-also-configured-as-mcp`.

**Effect.** PR 5 registers nothing, so everything stays green. It strengthens
the ruling and edits the pinned file, hence ⚑. L1 alone changes the ruling: it
replaces the grep with the audit's four assertions and keeps the message
"That is a deliberate decision; update this test and say so."

---

## 7. What this design does not do

v1.1 does none of the following:

* register anything (decision 1);
* foreground delivery, remote targets, `browser_*`, clipboard, launching
  apps, or recording/replay (`actions.py:33-49`);
* screenshots for any decision: the pixel tier is L3. The only capture is
  the per-step thumbnail for the person watching (§5.2.5), and a live view
  is L7;
* web content (§5.4.3);
* cloud models or services anywhere in the path (decision 3; Q1);
* macOS or Windows (L5);
* rebuild anything Cua ships (§4).

---

## 8. Decisions for Will

These five are product choices, security trade-offs, or touch the ruling.
**Will answered all five on 2026-10-04** (the "Will's answer" column).
Everything else is decided by recommendation in Appendix A, and can be
revisited there. Settled already: the D1 fix is a hard precondition for the
door, and the pin is widened in the door PR (Will, 2026-10-03).

| # | Decision | Recommendation | Will's answer (2026-10-04) | Must precede |
|---|---|---|---|---|
| **W1** | **The origin term for browsers and Electron apps** (§5.4) | **E:** a fifth extent term `site`, decided now while no grants exist. `-` (no web content) only on positive evidence. **v1.1 offers no web page content**; browser chrome is approve-once; Electron apps (VS Code, Slack) are limited to approve-once rows. The alternative E′ offers page content, approve-once every time: more capability, more injection surface and prompt fatigue. | **Agreed:** the site term now, no web content in v1.1. | PR 3 |
| **W2** | **What the app pick covers when a model chooses the clicks** (decision 6 + step 4; Q2) | The binding covers clicks and Tab/Escape in the picked app. Return, typing, menus, web content, high-consequence labels and anything past the ceiling prompt every time. Door prompts are approve-once, so a task never mints a lasting grant. Beacon binding: the session, at most 8 h; chat binding: the one task. The alternative ("grants serve script origin only") makes every model-chosen click prompt, and the toggle grants little. | **Agreed.** | PR 5 |
| **W3** | **Who may start computer use** (Q3; ⚑) | Only a person's credential: an allowed chat user, or a Beacon device a person marked `computer: true` from Telegram or an existing device. The global API token is refused with 401, because a model can read it with `bash`; device enrolment alone also needs only that token. `ComputerTaskRunner.start` refuses inside tool execution. | **Agreed.** | PR 5 |
| **W4** | **Close D15 for every tool, not just computer use** | Yes: approval answers require a person's credential for all tools, in a small PR outside the v1.1 numbering. Today a model holding the global token can approve its own `bash`/`write_file` prompts. The cost is that anything approving through REST with the global token (scripts) must move to a device token. | **Accepted, separate WP.** Agreed in principle, but out of scope for computer use. It becomes its own work package after D15 is verified on the live daemon. The door's own W3 guard (person-only credentials on computer routes and computer approvals, PR 5) stays regardless. | Not a v1.1 gate; its own WP |
| **W5** | **What chat surfaces may carry** (decision 3; Q1) | Telegram carries the consent sentence and approval prompts, including the text to be typed, as it already does for every tool's approvals. Milestones carry only the task id, app and counts, with no per-step log. Slack and Discord refuse `/computer` in v1.1. The stricter alternative: no content on chat at all, with Beacon as the only cockpit. | **Agreed.** | PR 5 |


---

## Appendix A — Decided (recommendation), revisit if you disagree

Each item is the design's recommendation, adopted so the work is unblocked.
Each is labelled **decided (recommendation), revisit if you disagree**.

| Item | Decided (recommendation), revisit if you disagree | Lands in |
|---|---|---|
| **The driver pin** | Exact `cua-driver==0.28.2` (D12) plus a code-side supported-versions check. Upgrade deliberately; capture-bound clicks are typed from 0.28.3. | PR 1 |
| **Hosting** (Q4) | Cua's `PRIVATE_WORKER` with a BOUNDED trusted session per task, if PR 2's on-box check shows it works on X11 and the telemetry opt-out reaches the worker. Otherwise `EMBEDDED`, accepting crash coupling. | PR 2 |
| **The agent cursor on X11** | Off by default; `on` is opt-in for someone at the machine (§5.2.4). Hermes found the overlay can stick and block input on X11. This narrows the brief's "the driver's own cursor when you're at the machine". | PR 2 |
| **`/gate off` and stored grants** | Under `/gate off`, `approve` consults stored grants and the binding before prompting, so the mode neither waives nor tightens computer consent. This is the detail of Will's settled D1 ruling. | PR 3 |
| **D2: typing goes to focus** | `set_value` by element token through `call_tool` (RFC 4268's rule; `_native.py:5561`), else withhold `type-N`. The prompt never names a field the driver will not target. | PR 3 |
| **Who picked the element** | Logged (`chooser.name`), never an extent term. | PR 3 |
| **The computer approval channel** | A computer-only `ApprovalQueue` instance, so other tools are unaffected and the operator's `security.approval_queue.enabled` keeps its meaning. | PR 5 |
| **`/approve all`** | Skips computer entries, on chat and REST, and says how many it skipped. | PR 5 |
| **Old iOS builds** | A payload-bearing computer approval from an iOS device without the capability header gets 409 "open on desktop or Telegram". | PR 5-6 |
| **Binding persistence** | In memory; re-asked after a daemon restart. A durable binding is a follow-up. | PR 5 |
| **Where the chooser runs** | A dedicated small-Gemma llama-server (E4B first), not the production slot. | PR 10 |
| **OS event listeners** | Deferred. Ask upstream to expose the trackers the driver already runs. | L6 |
| **App term on macOS** | `bundle_id`, so the first grants on a Mac are never re-spelled. | L5 |
| **The AGPL icon detector** | Start the pixel tier with PP-OCR (Apache-2.0) only, and revisit the AGPL detector before L3 ships. | L3 |
| **Registering `computer_task`** | Not in v1.1. L1 will come back as a ruling decision. | L1 ⚑ |


## Appendix B — measuring the tool catalogue

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

* **The two `computer_task` figures.** The bare hand-written three-field
  schema above is 352 characters. §2.4 uses the larger, realistic figure:
  **865 characters (≈216 tokens)** for the four-field input §5.1.1 proposes,
  generated by pydantic with field descriptions, titles and `anyOf` nulls,
  as a real `BaseModel` input would emit it. Measured with:

  ```python
  import json
  from pydantic import BaseModel, Field

  class ComputerTaskInput(BaseModel):
      goal: str = Field(description="What to do on the desktop, in the person's words.")
      app: str | None = Field(default=None, description="The app to use, e.g. 'my editor'; asked for if absent.")
      text: str | None = Field(default=None, description="The only text that may be typed; never taken from the goal.")
      target: str | None = Field(default=None, description="A declared machine name; defaults to the single local target.")

  schema = {
      "name": "computer_task",
      "description": ("Hand a desktop goal to the computer-use loop, which observes the chosen app and "
                      "acts step by step through the consent gate. Returns what it did."),
      "input_schema": ComputerTaskInput.model_json_schema(),
  }
  s = "[" + json.dumps(schema, separators=(",", ":"), ensure_ascii=False) + "]"
  print("computer_task (pydantic, 4 fields)", (1, len(s), round(len(s) / 4)))
  # -> computer_task (pydantic, 4 fields) (1, 865, 216)
  ```
* **The fuller run** adds the per-tool, per-category and per-profile
  breakdowns (profiles on a local tier: full 12, coder 7, research 3,
  assistant 2, minimal 2, symbiote 6). That run is not reproduced here.

## Appendix C — sources

Everything external cited in this document. URLs were fetched, and
repositories were cloned at the commits named, on 2026-10-03.

| Source | Where |
|---|---|
| Installed `cua-driver` 0.28.2 wheel | `.venv/lib/python3.11/site-packages/cua_driver/` (`_native.py`, `_native_contract.py`, `bin/cua-driver`, `libcua_driver_sdk.so`) |
| Later `cua-driver` wheels (0.28.3 … 0.33.1) | files.pythonhosted.org, via https://pypi.org/pypi/cua-driver/json |
| `trycua/cua` | github.com/trycua/cua at `0d274d0` (docs, `libs/cua-driver`, `libs/cua-bench`, `libs/lume`, `rfcs/3931…`, `rfcs/4268…`); also at `fbd6f96` (the chooser/eval research) and the `cua-driver-rs-v0.28.2` tag `fc18825` (the site-term research) |
| PyPI metadata | https://pypi.org/pypi/cua-bench/json, …/cua-agent/json, …/cua-sandbox/json, …/lume/json, …/pylume/json |
| Grounding models | `trycua/cua` `libs/python/agent/cua_agent/loops/{uitars,gta1}.py` (the UI-TARS-1.5 and GTA1 loops); `microsoft/OmniParser` README (licence and model notes); Cua's `perception-extension.mdx` |
| `NousResearch/hermes-agent` | github.com/NousResearch/hermes-agent at `158fd638` |
| Hermes issues | https://github.com/NousResearch/hermes-agent/issues/52014, /issues/32766, /pull/33054, /issues/96328, /pull/96341; https://github.com/trycua/cua/issues/2013. The same three numbers under trycua/cua returned 404 through the GitHub tools; that was not re-checked with an unauthenticated fetch. Hermes #28152 and #83473 are cited from `cua_backend.py`'s comments, not fetched. |
| llama.cpp | `ggml-org/llama.cpp` at `836d571` (`test-gbnf-validator`, `grammars/README.md`, `tools/server/README.md`) |
| AT-SPI2 / GTK | `GNOME/at-spi2-core` (`atspi/atspi-event-listener.c`, `registryd/deviceeventcontroller.c`, `xml/Event.xml`); `GNOME/gtk` (`gtk/a11y/gtkatspicontext.c`) |
| Apple AX / NSWorkspace | developer.apple.com documentation JSON for `AXObserverCreate`, `AXObserverAddNotification`, the `kAX…Notification` constants and `NSWorkspace.didActivateApplicationNotification` |
| Windows UIA / Win32 | `MicrosoftDocs/sdk-api` and `win32` sources on GitHub (learn.microsoft.com is blocked by this environment's proxy) |
| Browser a11y URLs | Mozilla `accessible/atk/nsMaiInterfaceDocument.cpp`, `DocAccessible.cpp`; Chromium `ax_platform_node_auralinux.cc`, `ax_platform_node_cocoa.mm`, `ax_platform_node_win.cc`; MDN `document.title`, `history.pushState` |
| OAraLabs repos (read-only) | `SkillForgeRecorder` at `9bfc86f`, `beacon-desktop` at `0a57da7`, `beacon-ios` at `8c1a383`, `OAra-Brain` at `28ffd86` |

**Not reachable from this environment:** cua.ai documentation, learn.microsoft.com,
api.github.com for some repositories, huggingface.co, and the tiktoken
encoding download. Claims that would have needed them are marked *not
established*.
