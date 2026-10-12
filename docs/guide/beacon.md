# Beacon — the desktop cockpit

Beacon is the native desktop client for the Prometheus daemon — an Electron app for macOS and Linux that bundles **no backend and no model keys**. It pairs to a running daemon over REST (`:8005`) and an authenticated WebSocket (`:8010`), stores your API token in the OS keychain, and works just as well over Tailscale as it does on localhost. The daemon is the single source of truth; Beacon is the window into it — chat with live tool timelines, Mission Control, a Loop Manager for coding runs, a documents editor with AI redlines, Kanban, telemetry feeds, and per-provider key management.

[← README](../../README.md)

![Beacon's Mission home — the default landing view, with the Armilla telemetry sphere beside the Mission Control HUD: agent state, scheduled jobs, local backends, and tool-call telemetry](../assets/shots/panel-mission-home.png)

## Getting Beacon

Honest status first: Beacon is free to use but not open source, and it is in beta. Every release publishes its builds — a `.dmg` for macOS (Apple silicon), an `.AppImage` and a `.deb` for Linux — from the [releases page](https://github.com/OAraLabs/beacon/releases/latest), and [oara.ai/beacon](https://oara.ai/beacon) reads whatever is newest there.

**Signing:** the macOS app and disk image are **signed, notarized and stapled** with an Apple Developer ID, so Gatekeeper opens them without ceremony. The Linux AppImage and `.deb` are **not** signed.

**Updates:** a packaged Beacon **updates itself.** It checks about 30 seconds after it opens and every four hours after that, downloads the new version in the background, and installs it the next time you quit (`⌘Q`). When one is ready the menu-bar icon also offers *"Beacon v\<version\> ready — restart to install"*, and **Connection Settings** has an **Updates** section showing the state, with a **Check for updates** button. The feed is the public `OAraLabs/beacon` releases page, so checking needs no credential and no token is shipped or stored. Two honest limits: a build **from source never checks** (the updater needs the packaged app's feed metadata), and the Linux update paths — the AppImage replacing its own file, and the `.deb` installing with `dpkg -i` behind a graphical sudo prompt — **have not been run on real Linux hardware yet**, only in CI.

### The first five minutes

On a Mac, the whole path from nothing to a first answer is Beacon doing the work:

1. **Download and open Beacon.** Drag it to Applications and open it. The published macOS build is signed and notarized, so Gatekeeper opens it without a warning.
2. Choose **"No, set it up on this Mac."** Beacon gets Prometheus for you, checks it's genuine, puts it in your Applications folder, and starts it. You won't type a command. It needs about 2 GB free and macOS 13 or later, on an Apple chip. macOS will tell you Prometheus was added to Login Items — that's expected, it's what keeps your assistant running when Beacon is closed.
3. **"How should your assistant think?"** Pick **With a cloud AI**, then **"Sign in with OpenRouter"** — one account that gives your assistant Claude, ChatGPT, Gemini and more, with no key to copy. OpenRouter opens in your browser; sign in (or make an account) and choose **Authorize**. OpenRouter needs a few dollars of credit before your assistant can answer: $5 is the smallest amount, plus a small card fee. Prefer a key? **Other options** takes an Anthropic or OpenAI key, pasted and checked before anything is saved.
4. Beacon finishes setup: saves your choices, starts your assistant, and **connects this Mac as its owner**.
5. **Say hello.** Send one of the suggested messages, or type your own. That's your first real round-trip.

If you already have Prometheus running, choose **"Yes, connect to it"** instead and Beacon connects to the daemon it finds.

![Beacon's setup wizard pairing with a daemon using a 6-digit code](../assets/shots/install-2-pairing.png)

The terminal-driven pairing path — the daemon in setup mode, a 6-digit code, `oara setup` — is in the [install guide](install.md). Both reach the same place.

The token you pair with goes into the OS keychain via Electron's `safeStorage` and never reaches the renderer process — Beacon's UI can see that a token is saved, but never the token itself.

## The nav

Six destinations in a rail that never expands — a surface's second level is a pill row at the top of that surface, not a tree in the rail:

- **Mission** — the cockpit landing, and the default view on launch
- **Chats** — Chat · Sessions
- **Coding** — Run · Loop Manager
- **Workspace** — Files · Artifacts · Board · Documents · Wiki · Checkpoints · Skills · Integrations
- **Activity** — Live · Tool Feed · Status
- **Config** — a tabbed panel, not a group: Models · Connectors · Skills · Telemetry · Usage · Memory · Sentinel · Wiki · Cron · Config · Profiles · Grants · **Devices** · Skin

The foot of the rail pins a global connection indicator — more on that in [Cross-cutting UX](#cross-cutting-ux).

## Mission

Mission is where Beacon lands you: a banner, a first-flight checklist (right after setup), a chat composer so you can start talking immediately, and two live instruments.

![Mission Control live — agent state hero, approvals, telemetry, Sentinel, and recent activity](../assets/shots/panel-mission-live.png)

**Mission Control** is the cockpit HUD. A hero card shows the agent's current state (it glows while thinking) alongside model, provider, active profile, and uptime, with quick actions underneath. Around it: an **Approvals** card with one-click Approve/Deny for anything the agent is blocked on, an **Attention inbox** collecting the things waiting on you, a telemetry headline, a **Sentinel** card (state plus dream count), a task ticker, and a recent-activity strip.

**The Armilla** is the armillary sphere you'll notice spinning above the composer. It isn't decoration-only — it's telemetry-driven. Its spin rate tracks how hard the agent is working, the core pulses with activity, each active session gets its own node on the ecliptic, and a 24-hour horizon dial marks the day's signal events. If the connection drops, the sky freezes — a glanceable "something's wrong." It renders static if your OS requests reduced motion.

After the setup wizard, a dismissible **first-flight checklist** sits on Mission home with three deep-linked starter tasks.

## Conversations

### Chat

The main chat surface streams markdown replies with syntax-highlighted code blocks (language chip, copy button, auto-collapse past 40 lines) and renders each turn's tool activity inline.

![A finished reply in Beacon — the session list, the answer, and the composer's Agent / Chat / Voice toggle and model picker](../assets/shots/chat-5-reply-done.png)

What you can actually do here:

- **Watch the tool timeline.** Consecutive tool calls merge into a single activity strip with live durations, and parallel dispatches show up as `‖N` batches — so a busy turn reads as one tidy row instead of a wall of cards. Individual tool calls expand into collapsible input/result cards.
- **Switch models per session.** The model chip in the top bar and composer opens a dropdown of everything the daemon's catalog offers. Each entry carries an **auth-mode badge** — "SuperGrok subscription" vs "API key" — and flags providers whose key is missing. Picking a preset sets a session-scoped override; Local/Default clears it.
- **Toggle Agent | Chat mode.** Agent (the default) lets the model use tools; Chat forces a plain no-tools reply. It's per-session and in-memory — it resets to Agent on every launch, by design.
- **Arm a forced search.** The 🔍 toggle arms the *next* send to open with a forced `web_search` call, then disarms itself. One-shot, Agent-mode only — for when you know the answer needs fresh information.
- **Retry or edit your last message.** The most recent user message gets Retry and Edit-last actions.
- **Attach files.** File picker, paste, or drag-and-drop — images get thumbnails, other files get chips. Agent-mode only.
- **Copy any message**, and rely on stick-to-bottom streaming with a "Latest ↓" jump chip when you scroll up. Long sessions stay fast because only the last ~120 rows render.
- **Download what the agent delivers.** Files the agent publishes to its [artifact outbox](api.md#artifacts--the-agents-outbox) appear as download chips in chat — click to fetch by content id.

![Mid-stream in Beacon — tool calls firing live while the reply streams](../assets/shots/chat-4-reply.png)

The session sidebar shows every session with **readable source chips** (Telegram, Desktop, Slack, Discord — parsed from the daemon's gateway stamping), unread dots, and relative timestamps, grouped by recency (Today / Yesterday / earlier). Machine-generated runs — bakeoffs, verification harnesses — fold into a single dashed **"Automated"** chip that's excluded from the default view, so your real conversations aren't buried under the system talking to itself. Inside a conversation, tool strips with more than 6 calls collapse into a per-tool tally instead of a wall of chips. You can filter/search, start a new chat, **rename** a session locally, or **Forget** it — a durable tombstone on the daemon: the session disappears from the index (and stays gone across restarts), the durable history is preserved, and new activity on the same id revives it.

### Sessions

A tree view of every active session on the daemon, with subagent sessions nested under their parents. Click any node to open it in Chat. Useful when the agent has spawned workers and you want to peek at what a subagent is actually doing.

## Coding

### Run

A single sandboxed **iterate-to-green** coding run: give it a repo path, a task, an acceptance command, and caps (max rounds, max wall-clock seconds), and fire.

![Live coding rounds streaming into Beacon — per-round outcome, tokens, and duration](../assets/shots/loop-2-live.png)

While the run is live, Beacon streams per-round progression over the authenticated WebSocket — each round's outcome, token count, duration, thinking flag, and stop reason — with an elapsed clock and a **Stop run** button.

![A finished run's artifact — verdict, acceptance exit code, and the reviewable diff](../assets/shots/run-2-artifact.png)

When the run finishes, the artifact view shows the verdict and acceptance exit code, the branch name, a diff-stat, a colorized diff viewer (truncated at 256 KB), and the final test output — plus **copy-able git commands** for checking out and merging the branch yourself. That last part is the whole model: **Beacon is review-only. It never merges and never pushes.** The run leaves a branch; you decide what happens to it.

### Loop Manager

The Loop Manager turns coding runs into a PM cockpit. Register local repos as projects (label, root path, color dot) and each gets a live `done/total` task ratio and an idle/running/done status.

![A Loop Manager project workspace — TASKS.md board, LOOP.md contract, and the compiled run](../assets/shots/panel-loop-workspace.png)

Opening a project gives you a three-step launch walkthrough:

1. **TASKS.md board** — parse, add, toggle, and delete tasks; tag them with `[Tag]` labels; pick which task the run targets. No TASKS.md yet? Create one from a seed.
2. **LOOP.md contract editor** — the run's rules of engagement: acceptance command, caps, conventions. You can generate a template, drop in an existing `.md`, or pull one **"From Documents…"** — browsing the daemon's documents library, which works even when Beacon is running on a different machine than the daemon.
3. **Compiled run** — a payload preview, a live readiness checklist with jump-links to anything unfinished, and a Fire button.

Runs launch in one of **three modes**: **Autonomous** (fire and forget), **Composed**, or **Supervised** — and in v1, Supervised means *watch plus Stop*. The daemon supports mid-run pause/inject/resume, but Beacon's UI for it is deferred; see [Known v1 limits](#known-v1-limits).

Every file the Loop Manager reads or writes goes through the daemon's `/api/project-file` route rather than Beacon's local filesystem — so a Beacon on your laptop can edit TASKS.md and LOOP.md on the daemon host across the tailnet.

## Workspace

### Documents

A calm writing surface over the daemon's confined documents folder. Browse with breadcrumbs, create files, and open them in a multi-tab editor with a centered reading column, debounced auto-save (about 900 ms, with an honest save-status indicator), **⌘S** to force-save, and a confirmation before closing a dirty tab.

![The Documents editor — multi-tab writing surface over the daemon's documents root](../assets/shots/panel-documents-editor.png)

The standout is **Ask AI**: type an instruction ("tighten the intro," "fix the dates") and the daemon returns proposed edits as `{find, replace, reason}` triples rendered as **inline tracked-changes** — redlines, exactly like a reviewer's markup.

![AI redlines pending in the Documents editor — each proposed edit shown inline with Accept/Reject](../assets/shots/panel-documents-redlines.png)

Each redline gets its own **Accept/Reject**, plus an **Accept all**. Nothing touches disk until you accept — accepting an edit is what applies it. Binary and oversized files are refused honestly rather than mangled.

### Files

A read-only browser for the agent's sandboxed workspace — the directory where the agent's own file tools operate. Navigate folders, click a file to open it in the **Hermes preview pane** (a global right-hand dock that renders text, images, and truncated/binary notices, with copy-path and copy-content), and see what the agent has actually been writing.

![The Files panel browsing the agent's sandboxed workspace](../assets/shots/panel-files.png)

### Board

A daemon-backed Kanban: projects and stories across four columns (todo / in progress / blocked / done) with native drag-and-drop. Create projects, create and edit stories (priority, labels, project), filter by project, delete what's stale — and **dispatch a story straight into a session** to have the agent work it. Stories can flow directly into coding runs; see the [Coding Mode guide](coding-mode.md).

![The Kanban board — four columns, drag-and-drop, and story dispatch](../assets/shots/panel-board.png)

### Integrations

A manifest-driven control surface for external integrations the daemon knows about. Each integration gets a health badge, Connect/Disconnect, and a Set-token action (stored in the keychain), plus whatever views its manifest declares: **Vitals** cards, a searchable **Browser** table, **Controls** with two-step confirmation, and a cursor-polled **Events** log. One v1 honesty note: there's **no UI for adding a new integration** — registering one means hand-editing `integrations.json` on the daemon.

## Activity

### Live

Every frame crossing the gateway, newest-first, with backfill from the daemon's recent-events API on connect. Filter by category pill or session, search, **pause** the stream while you read, clear it, copy any row's payload, or **export the whole feed as JSON** for offline digging.

![The Live activity feed — every gateway frame, filterable and exportable](../assets/shots/panel-activity.png)

### Tool Feed

Aggregated tool-call pairs across all sessions, with a HUD of totals — calls, ok, errors, currently live, success percentage. Filter by tool name and expand any call to see its inputs, result, or error. This is the fastest way to answer "which tool keeps failing, and on what input?"

![The Tool Feed — aggregated tool calls with success-rate HUD](../assets/shots/panel-toolfeed.png)

### Status

The daemon's vitals in one place: agent state, the active session's **context window** gauge (token usage, fresh vs summarized messages, compression ratio — the LCM at a glance), Sentinel state, and the list of agent profiles with a **Set active** button to switch between `full`, `coder`, `research`, and friends.

![The Status panel — agent state, context-window gauge, Sentinel, and profiles](../assets/shots/panel-status.png)

## Config

A tabbed panel of native surfaces over the daemon's configuration. Every tab shares a single honest not-authorized state that deep-links to Connection Settings if your token is rejected.

- **Models** — the per-provider key manager. Paste a key and it saves to the daemon, takes effect immediately, needs no restart, and is **never shown again** — the UI only knows a key exists. Each provider card carries a "Get a key ↗" link and an auth-mode badge, and there's a read-only card for your local model. This tab is also home to **"Sign in with SuperGrok"** — an OAuth device-code flow for xAI: start the login, approve it at accounts.x.ai, and Beacon polls until you're signed in. Your subscription then powers `/xai` with no API key at all; expiry is shown, and you can sign out anytime.

  **Main model** — *Which AI your assistant thinks with.* Pick a company and (optionally) a version; leave the version blank for that company's default. If the model you want is a cloud one you haven't keyed yet, **"Sign in with OpenRouter"** is right here too, the same browser sign-in as setup. A change here **takes effect when Prometheus restarts** — the panel shows `Now:` and `After a restart:` separately and offers the restart button, so you can see the pending change rather than wonder whether it took. This writes the daemon's configured main model; a per-chat `/qwen`-style override is separate and unaffected.

  ![The Models tab — per-provider keys, auth-mode badges, and SuperGrok subscription sign-in](../assets/shots/extra-models-tab.png)

- **Skills** — the agent's skill library, read-only, with **Pin/Unpin** (pinned skills are protected from automated pruning) and a View action that opens the skill in the preview pane.
- **Telemetry** — total tool calls, a success gauge, and per-tool stats.
- **Memory** — an **editor** for `MEMORY.md` and `USER.md` with character-budget gauges. Edits save to the daemon via `PUT /api/memory/current`; if the agent moved the file since you loaded it, the save is refused with a 409 carrying the current truth so you can rebase your draft instead of clobbering — and every write is snapshotted to the daemon's memory history first, so edits are reversible.
- **Sentinel** — state, dream count, timers, and a tail of the dream log.
- **Wiki** — page and entity counts for the knowledge base.
- **Cron** — full CRUD over scheduled jobs: create, **Run now**, enable/disable, delete, with next-run/last-run times and status.
- **Config** — the raw `prometheus.yaml`, for when you want the source of truth.
- **Profiles** — the profile list with the active one marked.

## Another computer, and a phone

### Let a second computer connect

By default **only this Mac can use Prometheus.** To reach it from another computer on the same network, open **Connection Settings** (`⌘,`) and turn on **"Let other computers on my home network connect."** That's home-network mode: it switches the daemon to serve your network over TLS, and the setting says plainly what it does — *"Beacon on your other computers can find this Prometheus and ask to connect. You approve each one."* Restart Prometheus to finish (the panel offers the restart), and if anything would make it unsafe — no way to share safely yet, or set to share but only listening on this Mac — it says so instead of flipping a switch that wouldn't work.

On the **second** computer, open Beacon and pick **"Look for another computer."** Beacon searches this Mac and your home network. If it finds Prometheus, choose it and **Connect**; it then waits for approval — *"Approve it in Beacon on a computer that already uses Prometheus"* — and shows the code to match. If it doesn't find it, **"Or type its address"** takes the computer's name or address directly (a port you type is kept and shown). A computer that joins this way gets an ordinary scoped device: it sees only its own conversations, and can't approve a worker's request.

### Make this Beacon an owner

If the second computer needs real authority — to approve a worker's request, the way the machine Prometheus runs on can — promote it to an **owner device**. In **Connection Settings** (`⌘,`) on that computer, find **"Make this Beacon an owner."** It reads:

> *An owner Beacon can approve worker requests. On the computer running Prometheus, run `oara pair --owner` and type the code it shows here. It works once, within 5 minutes.*

So: run `oara pair --owner` on the daemon's machine, which prints an 8-digit code (shown as `1234-5678`); type those eight digits into this Beacon. A 200 that says `owner: true` is the promotion — every other outcome (wrong, expired, used, replaced) is the one honest "that code didn't work" message, so a stale code can't look like success. The full command-side behavior — terminal-only printing, rate-limiting of wrong guesses, what can and can't mint a code — is in the [install guide](install.md#make-a-beacon-on-another-computer-the-owner).

### Pair a phone with the QR

In **Config → Devices**, name the device and click **"Mint a one-shot pairing QR."** Beacon mints a code and renders a QR — the token is held in memory and drawn locally, never sent to a remote renderer, and the device list refreshes immediately after. Scan it from the phone. The token is **one-shot**: the daemon returns the plaintext once and keeps only a SHA-256, so there's deliberately no "show it again" — if you dismiss the QR unscanned the credential still exists on the daemon, and the row stays visible so you can revoke it (an orphan you can't see is a credential you can't withdraw). Minting needs the global token, which only the desktop holds; a device token presented to `POST /api/devices` gets a 401 by design, so an already-enrolled phone can't enrol a second attacker device. A phone enrolled this way gets a scoped device like any other — promote it to an owner separately if it needs to approve workers.

## Cross-cutting UX

The things that follow you around every view:

- **Command palette (⌘K)** — fuzzy-ranked jump to any view, session, or action, including deep links into individual Config tabs.
- **Shortcut sheet (?)** — a keybinding cheat sheet, one keypress away.
- **Quick Capture (⌘⇧Space)** — a global, always-on-top mini composer, Spotlight-style. Fire a message at any session from anywhere on your desktop, watch the reply stream inline, and "Open in Beacon" if it turns into a conversation.
- **Global summon (⌘⇧B)** — toggle the Beacon window from any app.
- **Exec-approval overlay** — when the agent is blocked waiting on a tool approval, cards appear top-center with Approve/Deny. If Beacon isn't focused, you get a native notification and a tray badge instead of a silently stuck agent.
- **Tray / menu bar** — a live status line, an "N approvals waiting" counter, and Open/Settings/Quit. Closing the window **hides to tray** — the connection stays up and notifications keep flowing.
- **Native notifications** — replies, background-task completion and failure, and disconnects; clicking one focuses the relevant session.
- **Offline outbox** — messages you send while disconnected queue in a local SQLite cache and drain automatically on reconnect, with the queued count visible in the connection indicator.
- **Window-state persistence** — bounds, maximized state, and zoom survive relaunches.
- **Connection indicator** — the dot pinned at the foot of the nav rail. It shows WebSocket state, turns an honest **amber lock when the daemon is rejecting your token (401)** rather than pretending to be merely offline, and displays the outbox count. It also guards against the half-open-socket lie: a **20-second application-level ping** with a quiet-window terminate means a socket that has gone silent is torn down and reconnected instead of sitting there looking healthy — "Connected" can never point at a dead socket. Click it for a popover with the gateway address, reconnect attempts, outbox detail, and Reconnect / Open Settings actions. Connection Settings itself (⌘,) lets you test REST and WS independently before saving.
- **The ActivityLine** — movement means liveness. While a turn runs, a live line driven by the daemon's 3-second `agent_progress` pulse shows the current phase, tool, and round, so a long-thinking agent never looks hung. And when a turn fails, the structured error frame renders as a **cause with a hint** — "billing", "auth", "rate limit", "backend unreachable" plus one actionable sentence — instead of a bare stack-trace string.

## Themes — the Skin Codex

Beacon ships eight themes as a "Skin Codex," each a complete design-token plate with a live preview tile in the picker. Your choice persists across launches.

![The Skin Codex theme picker](../assets/shots/extra-theme-picker.png)

- **NOX — The Observatory** (dark, the default)
- **LVX — The Vellum** (light)
- **AVRVM**
- **VMBRA**
- **AERVGO**
- **CAERVLA** (dark)
- **RVBEDO**
- **AETHER** (light)

## Known v1 limits

Honesty section — the things Beacon does not do yet, so you're not surprised mid-task:

- **Supervised mode is watch + Stop only.** The daemon supports mid-run pause/inject/resume, but Beacon has no UI for it yet — Supervised in the Loop Manager means you can watch rounds stream and hit Stop, nothing more.
- **The live coding view only shows runs launched from this Beacon instance.** You can't attach to a run started from Telegram, the CLI, or another Beacon, and if you relaunch Beacon mid-run the live view doesn't recover it (the finished artifact is still reviewable).
- **Coding runs are review-only.** Beacon never merges or pushes — it hands you a branch and the git commands.
- **Per-round tool names and test pass/fail aren't on the wire** — the round stream shows outcome, tokens, and duration, not which tools fired inside each round.
- **The Loop Manager is local-only** — repos must live on the daemon host; no SSH/remote-host targets in v1.
- **Integrations has no add-integration UI** — new integrations mean hand-editing `integrations.json`.
- **Agent | Chat mode resets to Agent on every launch** — it's deliberately in-memory, not persisted.
- **The diff viewer and file preview truncate at 256 KB.**
- **The Linux update paths aren't run on real hardware yet.** macOS updates are signed, notarized and self-installing, but the AppImage and `.deb` self-update flows have only run in CI, not on a real Linux machine — the `.deb` in particular needs a graphical sudo prompt (`pkexec`/`gksudo`/`kdesudo`/`beesu`) to install.
