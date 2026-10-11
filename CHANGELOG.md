# Changelog

Notable changes in each release, newest first. Releases before 0.9.7 are described on the [GitHub releases page](https://github.com/OAraLabs/Prometheus/releases).

## 0.9.7

Prometheus can now be reached from more than one device. Phones and other computers pair with the owner's approval and get their own scoped tokens, and the home network gets TLS. This is also the first release with a Mac app and an official Docker image. Read "Upgrading from 0.9.6" before you restart the daemon: existing device tokens become scoped, and on macOS the shells a model writes lose `~/.ssh`.

### Security

- **Every shell a model writes runs behind the bash tool's floors (#659).**
  - This covers `task_create` shells and poll checks, model-created cron jobs and coding-run commands. They ran unconfined before, and three of them received the daemon's full environment, keys included. They now get the scrubbed environment and the read and write floors.
  - `security.bash_confinement` now defaults to `auto` (the 0.9.6 template shipped `off`). If a floor marked required isn't available, the command is blocked instead of run.
- **On macOS, model-written shells can't read your secrets (#712).** With `bash_confinement: auto`, every shell a model writes runs under a Seatbelt profile. It denies reading and writing `~/.ssh`, `~/.gnupg` and `~/.config/*/*env`, which holds the API token, provider keys and bot tokens. `security.bash_allow_ssh: true` re-opens `~/.ssh` only.
- **The listen address is yours to choose (#693).**
  - `--bind` > `PROMETHEUS_WEB_BIND` > `web.bind` > `0.0.0.0`. An invalid value refuses to start rather than guess.
  - A loopback bind refuses requests addressed to a foreign host name, which blocks DNS rebinding.
  - New configs are written with `web.bind: 127.0.0.1`. Existing configs are not changed.
- **An empty `PROMETHEUS_API_TOKEN=` no longer serves the API open (#717).** Before, that happened even when the env file held a token.
- **Agent tasks and coding runs start Python with `-P -s` (#709).** A file in the working directory can no longer shadow a real module.
- **Logs give the length of what a person sent, never the text (#641).** API keys and bot tokens show only their last 4 characters (#645).
- **Every approval answer records who gave it (#661):** token, device, or chat user. It goes in the audit log and the `approval_resolved` event.
- **Dependency updates in the lock: multidict 6.9.1 (CVE-2026-104874, #682) and strawberry-graphql 0.332.0 (CVE-2026-107727, CVE-2026-107728, #702).** These fix `uv` installs and the deploy only. `pip install -U` doesn't move either package: pyproject sets no multidict floor, and under pip the evals extra still resolves strawberry-graphql 0.316.0.

### More than one device: pairing, per-device tokens, home network

- **Request to join (#697, #698, #699).**
  - A new phone or computer calls `GET /api/hello` (name, version, fingerprint), then `POST /api/pair/requests`.
  - The owner approves in Beacon, in Telegram (opt-in: `pairing.telegram_prompts`), or with `oara pair list|approve|deny`.
  - The approved device receives its own token, encrypted to a key it sent.
  - Requests are on by default and rate-limited; `pairing.requests_enabled: false` turns them off.
- **Per-device tokens (#692, #696, #698).**
  - A device token sees and manages only the sessions it creates, and can revoke only itself. Every other route is operator-only.
  - Revoking a device closes its open connections.
  - Owner devices count as the operator, except that they can't enrol devices or define MCP servers.
- **Home network (#700, #721).**
  - `network.home_network: true` with a bind that reaches the LAN serves TLS on ports 8006 and 8011 (`web.tls_api_port`, `web.tls_ws_port`). The certificate is over the instance key: P-256, TLS 1.2 or later, AEAD ciphers only. Plain HTTP then listens on 127.0.0.1 only.
  - Clients pin the certificate. Its SHA-256 starts with the fingerprint `GET /api/hello` advertises.
  - Home network needs TLS. Plain HTTP on the LAN happens only with the explicit `network.allow_plaintext_lan: true`.
  - With the optional `[discovery]` extra (zeroconf), the daemon announces itself as `_prometheus._tcp` on private addresses, in home-network mode only.
- **`oara doctor` gains a Home network row (#700).** When `network.home_network` is on, it says whether home-network mode can work, whether it is safe, and whether the daemon will announce itself.
- **Switch from a client (#700, #724).** `GET`/`PUT /api/network` saves `this_mac` or `home_network` to the config file. `POST /api/restart` (operator only) applies it: the daemon re-execs in place, in the same process, re-reads the env file, and comes back with the same fingerprint. A request from a web page (one with an `Origin` header) is refused.

### Prometheus.app for Mac (#695, #711, #722, #728, #694, #696, #708)

- **A signed, notarized app in a zip, for Apple silicon on macOS 13 or later**, with its own Python 3.12. It installs the daemon as a login item. `oara install-service` on macOS now writes a LaunchAgent too (#691).
- **It listens on this Mac only** until the config says otherwise. A LAN bind on the app falls back to loopback unless home network (TLS) or `allow_plaintext_lan` is on.
- **New app installs start in the `cautious` permission mode:** every shell command the model runs, and every file write, asks first. `/gate cautious` selects it anywhere.
- **Beacon on the same Mac pairs with a one-time file secret** (`Prometheus --pair`, then `POST /api/pair/local`). It gets an owner device token, never the master key.
- The app ships without PyMuPDF (AGPL), so PDF text extraction and image downscaling are unavailable in it. It ships the `[discovery]` extra, and each copyleft package's license text is in `Contents/Resources/licenses/`.

### Docker image (#714)

- **`ghcr.io/oaralabs/prometheus`**, for linux/amd64 and linux/arm64, tagged `0.9.7`, `0.9` and `latest`.
  - It's published after the PyPI release, with the anthropic, mcp, slack, discord and push extras.
  - It runs as a non-root user and accepts `PROMETHEUS_API_TOKEN_FILE`.
  - A compose example is in `docs/guide/docker.md`.
- It leaves out mDNS discovery, which doesn't cross Docker's default bridge network.
- **Running in an OpenShell sandbox (#734).** `packaging/openshell/` holds a thin image on top of the official one, plus a default-deny policy.
  - The policy allows only the model server on the host, the cloud providers, Telegram, Slack and Discord.
  - Only `/sandbox` and `/tmp` are writable.
  - `docs/guide/openshell.md` says what was tested: OpenShell 0.1.3 on Linux amd64. You build that image yourself; the release doesn't publish it.

### Computer use (Linux, X11; off by default)

- **A person can run a desktop task** with `/computer <goal> [app:…] [text:"…"]` on Telegram, or from a Beacon device marked for it (#650–#657, #662, #663).
  - Only a person can start one: a Telegram sender on the allowlist, or a marked device. The API token and models can't.
  - Turning it on takes `computer_use.enabled: true`, a declared local target, and the `[computer]` extra (`cua-driver==0.28.2`).
- **Consent is the app the person picks.** It covers clicks and Tab/Escape in that app. Return, typing, menus, and buttons such as Send, Delete or Pay ask every time, as approve-once.
  - `/gate off` doesn't waive desktop prompts, `/approve all` skips them, and web content is never offered.
  - Each task has ceilings: 20 steps, 600 s and 10 approvals.
- **Stop always works:** `/computer stop`, Beacon's Stop, or the REST or WebSocket stop. An action already in flight may still land.
- Actions are chosen by fixed rules, not a model. A task ends when its one action has run (#672, #674, #664, #665, #710). After each step, a thumbnail of the approved window can go to marked devices (#701); this is experimental and not yet verified with the pinned driver.

### Spawned workers (#713, #715, #719, #720)

- **A worker runs on its conversation's model route,** or on the one worker model the operator chose (`workers.model`, or `GET`/`PUT /api/workers/model`). It never runs on a model the agent names.
- **It takes its brief as one turn and exits with a result.**
  - If it runs out of context on another route, it's retried once on the conversation's route.
  - When `sentinel.enabled` is on, its result comes back into the conversation that spawned it, Beacon included. SENTINEL ships off.
- **A worker can ask for approval.** The request shows up in the parent conversation as approve-once.
- **A daemon restart stops the task processes it can no longer supervise (#731).**
  - Each task process records its pid and start time when it launches.
  - On restart, a still-running task process is stopped with its whole process tree.
  - A pid that is gone, or now belongs to another process, is left alone.
  - Before, such a process outlived the daemon untracked.

### Models and the agent loop

- **OpenRouter is a cloud provider (#732).** Setup accepts `{"provider": "openrouter", "api_key": …}`. The default model is `anthropic/claude-sonnet-5`, a paid, tool-capable model.
  - `POST /api/setup/check-key` tests a key before it's saved. It covers OpenRouter, Anthropic and OpenAI, and answers `ok`, `bad_key`, `no_credit` or `unreachable`.
  - OpenRouter's `vendor/model` ids are no longer counted as free local traffic. Unpriced models are reported as unknown, not free.
- **`PROMETHEUS_OLLAMA_URL` and `PROMETHEUS_LLAMA_CPP_URL` now work (#725).** They set the main model's URL when it uses that provider; otherwise the daemon logs a warning. Before, they did nothing.
- **A prompt too long for the backend's window now ends the turn with a plain explanation and the server's numbers (#716),** not "400 Bad Request". The CLI and every worker now compact like the daemon, using the server's real context window (#718).
- **A `null` for an optional tool parameter takes its default (#666),** rather than failing or becoming a file named "None". This applies to local models; cloud models skip the adapter.
- Microcompaction never makes a tool result longer (#648). A cloud run's breaker recovery no longer starts microcompaction (#637).
- `/mimo` now defaults to mimo-v2.6-pro, ahead of v2.5's retirement on 2026-10-21. The picker labels gpt-6-astra "no tool calls" (#686).
- `POST /api/chat` uses the daemon's full runtime system prompt (#680).
- Every tool's example call, which a model is shown after a malformed call, now validates against that tool's own input model (#670).

### Memory, telemetry and learning

- **Edits to `MEMORY.md` and `USER.md` reach the next turn without a restart (#679).**
- Memory confidence decays by elapsed time, not once per AutoDream pass (#646). AutoDream and GEPA pause while Prometheus is in use (#644).
- `infrastructure.telemetry_enabled: false` now really turns telemetry off in the daemon (#639).
  - Local telemetry records each turn and response, and how each turn ended (#638, #640, #649). A repaired call that is then refused counts its repairs (#671).
  - It now keeps some content: redacted model prose, and the first 500 characters of each tool result, redacted.
- The golden-trace export runs off the event loop (#706).

### Other

- In cron jobs you create, `python3` is the daemon's own interpreter (#651). Model-created jobs don't get this.
- The daemon reports its real version (#642).
- `scripts/deploy.sh`'s B6 step checks the model choices instead of re-applying them (#643).
- The README leads with the always-on daemon (#658, #673). The README, the API and install guides and the config template now describe TLS in home-network mode and `oara install-service` on macOS, which they still denied.
- Project: the PR-gating workflows also run in a merge queue (#730), and the workflow actions are on current releases (#723).

### Upgrading from 0.9.6

- **Device tokens you minted before become scoped.** They can chat and manage only the sessions they create, see none of the earlier conversations, and get 403 `operator_only` elsewhere (including `/api/status` and `/api/config`). The global token is unchanged.
- **On macOS (terminal installs too), model-written shells lose `~/.ssh`, `~/.gnupg` and `~/.config/*/*env`.** Agent `git` over SSH and gpg-signed commits fail. Set `security.bash_allow_ssh: true` (SSH only) or `security.bash_confinement: off`.
- **Two new routes answer without a token, wherever the daemon listens:** `GET /api/hello` (name, version, fingerprint) and `POST /api/pair/requests` (rate-limited, and it needs your approval). `pairing.requests_enabled: false` turns join requests off.
- **Behind a reverse proxy or tunnel, name it in `web.trusted_proxies`.** Forwarded headers are otherwise ignored, so every client looks like 127.0.0.1.
- **A config without `web.bind` still listens on every interface.** `oara doctor` and the startup log now warn about it. Set `web.bind: 127.0.0.1` to keep the daemon on this machine.
- **If your config came from the 0.9.6 template, edit `slash_commands.mimo.model`.** It sets `mimo-v2.5-pro`, which overrides the new default and stops working on 2026-10-21.
- **The daemon writes `~/.prometheus/node/instance.key` and `instance.crt` at boot.** Keep them: the key is the fingerprint paired devices pin.
- `oara install-service` with no systemctl or launchctl now exits 1. `/gate on` restores your configured mode, and re-running `oara setup` keeps it.
