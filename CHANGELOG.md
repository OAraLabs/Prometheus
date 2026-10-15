# Changelog

Notable changes in each release, newest first. Releases before 0.9.7 are described on the [GitHub releases page](https://github.com/OAraLabs/Prometheus/releases).

## 0.9.11 (2026-10-15)

0.9.11 fixes coding mode on any repo whose code sits at the root: its search tools couldn't see root-level files. It's the one change since 0.9.10, released as a freeze exception.

### Coding mode sees files at the repo root (#802, fixes #801)

- **`code_glob` and `code_grep` now match `**/` against zero directories, too.** They read `**/` as "at least one directory", so `**/*.py` never listed `export.py` at the root, `code_grep`'s default `**/*` never searched a root-level file, and `src/**/*.py` missed the files directly under `src/`. In the runs reported in #801, a model that could see only the test file rewrote it and broke it.
- Only matches are added: everything else about the patterns is unchanged (`*` still spans `/`, and `?` and `[...]` work as before).
- The tools' descriptions are unchanged. Their examples still read `src/**/*.py`, which now also matches files directly under `src/`.

### Upgrading from 0.9.10

- Nothing to do. A coding run's `code_glob` and `code_grep` can now return more files than before, never fewer.

### Known issues

- **The memory extractor can save no facts, silently, when the model pretty-prints its JSON (#798).** Each extraction call is recorded as a success and nothing is parsed, so no new facts are saved and the wiki doesn't grow. Seen with a 7B model on Ollama. The log line `MemoryExtractor: persisted 0 memories from N messages` is the only sign.
- The known issues listed for 0.9.10 still apply.

## 0.9.10 (2026-10-14)

0.9.10 keeps the work of a turn that hits its round cap, stops a model that breaks down at a very large context from poisoning the conversation, makes compaction cover every message and start earlier on large windows, and closes two gaps around subagents. Read "Upgrading from 0.9.9": the compaction threshold moved, `context.compression_trigger` is retired, and built-in subagents now have tools.

### Turns that run long (#775, #782)

- **A turn that hits its round cap keeps its work (#775).** On 2026-10-13 a Beacon turn ran 200 model rounds and 223 tool calls, hit the cap, and the whole turn was rolled back; on Telegram, Slack and Discord the user's own message went with it. The cap is a budget stop, not a failure, so every round that completed is now kept, the same way a user stop keeps it.
  - The loop raises `MaxTurnsExhausted` (still a `RuntimeError` with the old message, so gym, ladder and coding mode are unchanged), answers any tool call left without a result, and ends the turn on a plain note naming the limit.
  - Web and Beacon save the turn, stream the note, and add `stop_reason: "max_turns_exhausted"` and `max_turns` to `chat_done`. Telegram, Slack, Discord and REST `/api/chat` save it and reply with the note. The CLI keeps it. A worker reports `max_turns_exhausted` as its reason instead of a generic error. A real failure still rolls back.
  - **`model.max_turns` is a config key** (default 200). It counts model rounds, separately from `max_tool_iterations_cloud`, which counts tool calls; that is why a configured 500 never bound the turn above.
- **`/api/status` reports the live round cap (#782):** `iteration_ceilings.max_turns`, read from the live loop, and `null` when the bridge isn't wired.

### A model that breaks down no longer poisons the conversation (#777, #786)

- **A garbage round is dropped, retried once, and otherwise ends the turn.** On 2026-10-13 `qwen3.8-max` broke down twice at about 241k tokens of context, once with digits and C0 control bytes in the visible text and once with 6,416 tokens of `"0 00 000"` in its thinking. Both rounds were saved as ordinary history, and a background task's completion then woke the session and read them back.
  - The check runs right after the model returns, before any tool is dispatched or the round is saved, on the visible text and the thinking separately: any C0 control byte except tab, newline and carriage return, or a run of at least 200 non-space characters that is under 5% letters and at least 85% one character.
  - The first bad round is dropped (never appended, saved or dispatched) and the model is called once more with a nudge. A second in a row ends the turn on a note naming the context size, model and provider, with stop reason `degenerate_output`. A clean round resets it.
  - Streamed garbage is cut from what the turn keeps: the web bridge sends a `chat_discard` frame.
  - A background task whose session's last round was dropped and ended no longer re-engages it; its result is still notified.
- **Every firing is evidence.** Each firing and recovery writes a `subsystem_runs` row (input and cached tokens, window, compaction threshold and its source, requested and served model, provider, which test fired and where) plus a WARNING, and `/health` shows a summary by model and provider. That is the data per-model compaction thresholds get set from.
- **The incident round is pinned verbatim (#786):** the test now uses the exact 448 characters and 28 C0 bytes the daemon persisted.

### Compaction (#778, #784, #787)

- **The threshold is 80% of the serving model's window (#778).** It was 75% of the window less the output reserve, and on a 1,000,000-token cloud window that meant compacting at 746,928 tokens, long after the model above broke down. Set a model's own line with `context.model_overrides.<model>.compaction_threshold_pct` or `compaction_threshold_tokens` (the smaller wins if both are set; none ship). `/context`, the startup line and every `context_compaction` event name which rule decided.
  - A pass also runs when the next round would not fit: the current size, plus the request's `max_tokens`, plus the largest tool result of the last five rounds, compared with the window. The event says `trigger=threshold` or `trigger=projected`.
- **Compaction no longer leaves messages out.** The summarizer used to get one prompt capped at 18,000 characters, and every message past that was dropped: in live telemetry 309 of 3,334 compactions did this, and the worst replaced 1,685 messages with a summary of 46. The span is now packed into chunks that fit the summarizer's own window, each chunk is summarized, and the parts are merged. A single message bigger than a chunk is cut to fit and counted.
  - **Cost:** on a small local summarizer, a very large span becomes many calls. A span of about 900k tokens is roughly 34 calls before that turn's model call, on the order of ten minutes or more. At ordinary local sizes a span still fits one call, as before.
- **Passes now get under the line.** A pass aims at 70% of the line, not just under it. The protected tail never starts on a tool result (that left a result whose call had been summarised away, which every provider rejects). If the tail doesn't fit, the largest older tool results in it are trimmed first (marked, and the last complete tool round is kept), then turns are yielded. The question the current turn is answering is never lost: it rides along verbatim. A session that still can't get under is marked stuck once and not re-summarised while the turn is unchanged.
- **`context.compression_trigger` is retired.** Only a class nothing used ever read it. A config that still sets it gets one startup warning naming `compaction.threshold_pct`, and `migrate` maps Hermes' `compression.threshold` to it.
- **Beacon can see compaction happen (#784).** A new session-scoped `context_compaction` frame reports `start` (with `chunks_total`), `progress` (`chunks_done`, `chunks_total`, `progress`), `done` (`tokens_before`, `tokens_after`, `duration_ms`, `converged`), `failed` (with a reason that names an exception type only) and `noop`, each with the threshold, its source and the trigger, and `agent_progress.phase` is `compacting` while a pass runs. A pass that pays for no summary reports nothing. The frame is documented in `docs/guide/api.md`; the existing `context_compaction*` telemetry events are unchanged. This is the daemon half of beacon-desktop#283.
- **A Stop during compaction stops the turn (#787).** A cancellation was caught by the call wrapper and read as "the summarizer returned nothing", so the turn went on to the model. It now propagates under every `on_failure` mode, still after the run row is written. The same fix means a cancelled memory-extraction batch is re-read next pass instead of skipped, and a Stop during a teacher escalation stands instead of being recorded as a failed teacher call.

### Relative paths in a workspace session (#770, #771, #785)

- **A relative output path resolves to the session's workspace and no longer prompts.** In a session bound to a workspace, a relative path on `write_file`, `edit_file` and `notebook_edit` (#770) and now `download_file`, `tts` and `image_generate` (#785) is ruled on, and written, at `<workspace>/<path>`. Before, each prompted ("relative path … target is unknown"), and the model then made the same write through bash where nothing asked.
  - `../` that escapes the workspace still prompts, on the workspace boundary. With no session workspace a relative path still prompts, as before. Strict and cautious modes still prompt on every write. The resolved path still goes through the always-denied floor and `denied_paths`.
  - `download_file`, `tts` and `image_generate` now resolve a relative path against the session's working directory (#771). They used the daemon's own, so an approved `destination="x.bin"` landed beside the daemon.
  - `image_generate` may write under the session workspace's `images/` folder, but only when that folder resolves inside the workspace.
  - The file-mutation verifier looks for a relative write in the workspace, not in the daemon's directory, so it no longer reports "claimed but file absent" for a file that is there (#770).

### Subagents (#793, #794, #795)

- **The router-escalation subagent runs its tool under the turn's gate (#789, security).** With `router.escalation.enabled`, a tool call that failed validation too often was handed to a subagent on a stronger model whose loop had no permission checker, so its `bash` call ran with no approval for the commands the gate holds, no denied paths and no audit row. It now runs under the same gate. Escalation is off by default.
- **Built-in subagents have their tools (#790).** Every built-in agent definition named its tools `Bash`, `FileRead` and so on, which this daemon doesn't have, so each ran with none. They now name the real tools: `general-purpose` and `worker` get `bash`, `read_file`, `write_file`, `edit_file`, `glob` and `grep`; `explorer` and `planner` get `read_file`, `glob` and `grep`; `verification` gets `bash` and the read tools. A name that does not resolve is logged.
  - **A subagent only gets tools something named.** An agent type with no definition is refused with the known types in the error (it used to get every tool the parent has), and a definition that lists no tools gets none (it used to get all of them too). An explicit tool list still works for any type.
- **Ending a turn withdraws what its subagent was waiting on (#791).** When the parent turn was cancelled while a subagent waited for approval, the request left the queue but no client was told, so a card and a Telegram prompt stayed up. It is now withdrawn the way a worker's request is: `approval_resolved` with `resolution: "withdrawn"`, a notice to the chat that was asked, and a resolution row. A late answer is still 410. An end-to-end test now covers the whole path: it reaches an owner device, blocks until answered, a denial refuses the call, an unanswered request is denied at expiry, and each answer is audited against whoever gave it.

### Rejected WebSocket logins say who (#779)

- **A rejected WebSocket auth now logs which client and why.** Each rejection has its own reason (`malformed auth frame`, `not an auth frame`, `missing token`, `unknown token`, `revoked device`, `auth frame timeout`), and the line names the peer, the revoked device's id and name when the token belonged to one, and the handshake's `Origin` and `User-Agent`, truncated. Repeats from the same peer and reason within five minutes are counted ("and N more"). The client is still told only `unauthorized`.
- **A token leak found while testing is closed.** At DEBUG level the `websockets` library logs every frame it receives verbatim, including `{"type": "auth", "token": …}`. It is off by default, but anyone who turned DEBUG on to chase a connection problem would have written the token to their log. Auth frames are now redacted from that log.

### Upgrading from 0.9.9

- **The compaction threshold moved to 80% of the window,** with the projected-overflow rule. On a small window that is a few thousand tokens later than before. Pin a model's own line under `context.model_overrides` if you want it earlier.
- **A config that sets `context.compression_trigger` logs one warning at startup** ("config key context.compression_trigger is retired and does nothing; use compaction.threshold_pct instead"). The key never did anything; remove it.
- **`chat_done` may carry `stop_reason` and `max_turns`; there are two new session frames,** `context_compaction` and `chat_discard`, and `/api/status` gains `iteration_ceilings.max_turns`. A client that ignores frames it doesn't know is unaffected.
- **In a workspace session, a relative output path for `download_file`, `tts` and `image_generate` is allowed without a prompt** and written in the workspace.
- **A model's `agent` call now does something.** An unlisted `subagent_type` is an error naming the known types, and the built-in types now have real tools: `general-purpose` and `worker` can run `bash` and write files, under the gate and as the `system` origin, so a command the gate holds for approval is held.

### Known issues

- **Beacon does not yet use the new frames.** Until it follows (beacon-desktop#283), Beacon shows no compaction progress, and shows a dropped round's text followed by the retry or the error note, because it doesn't handle `chat_discard`.
- **An approval raised inside a subagent has only been checked as far as the queue, the gate, the REST route and the notifier.** Delivery to a real Beacon and a real Telegram client is unchecked. The request is filed under the subagent's own id (`subagent:<parent conversation>:<id>`), not the parent conversation, so a scoped device that owns the parent conversation does not see it, and the shared API token can answer it as it can any request that isn't a worker's.
- **`read_only` on an agent definition is a label nothing enforces:** `verification` is marked read-only but has `bash`.

## 0.9.9 (2026-10-13)

0.9.9 closes a way for a model-written shell on a Mac to reach the same-Mac pairing secret, and a way around the read floor on both Mac and Linux, makes the `agent` tool work, reads PDFs in the Mac app without PyMuPDF (AGPL), drops the unused Anthropic SDK, and adds a startup banner. Read "Upgrading from 0.9.8": Linux hosts with the AppArmor floor must reload its profile, `--version` prints different text, and an earlier default `denied_paths` list now resolves to the new one.

### Security (#764)

- **A model-written shell on a Mac can no longer read, write or move the same-Mac pairing secret.** `pair.secret`, in `~/Library/Application Support/Prometheus/pairing/`, is what `POST /api/pair/local` (and `POST /api/setup/pair` in setup mode) trades for an owner device token, and the request has to come from loopback, which every shell on the same Mac meets. Measured on macOS 15 before the change: `cat` printed the secret, `read_file` on it was allowed without a prompt, `write_file` could plant a secret of the agent's choosing, and `grep --root ~/Library` returned its contents.
  - The Seatbelt floor denies reading and writing the pairing folder, and the folder named by `PROMETHEUS_LOCAL_PAIRING_DIR` when it is set. `bash_allow_ssh` does not change this.
  - The gate's always-denied list, the shipped `denied_paths` and the config template now include the pairing folders for the Mac and Linux locations, and the folder this process would actually mint into. Differently-cased spellings are refused on macOS.
  - `grep` and `glob` now prune with that built-in floor as well as your `denied_paths`. Before, `denied_paths: []`, or a config holding an earlier list, let `grep --root ~` print a file the gate itself floors.
- **Renaming a parent folder no longer escapes the Seatbelt floor.** `mv ~/.config ~/cfg && cat ~/cfg/prometheus/env` printed the env file (the master key), because `~/.config` itself matched no rule; moving `Application Support/Prometheus` did the same to the pairing secret. The floor now refuses to rename, replace or re-mode each folder above a protected one (`~/.config`, `~/Library`, `~/Library/Application Support` and its `Prometheus` folder), up to but not including the home folder. Creating, writing and deleting files inside them still works. Hard links, symlinks and moving a prepared folder into place are refused too.
- **The boot probe now proves both refusals by outcome:** the pairing secret can't be read, and `~/.config` can't be renamed.
- **On Linux, renaming `~/.config` no longer escapes the AppArmor floor (#766, #776).** The same move worked there: `mv ~/.config ~/cfg && cat ~/cfg/prometheus/env` printed the file (measured on Linux 7.0 with the shipped profile loaded, against a dummy env file). The profile now refuses writes to `~/.config` itself, which a rename or replace needs. Creating, writing and deleting files inside it, and `touch ~/.config`, still work. The boot probe checks the rule by outcome, by renaming `~/.config` onto itself, which is refused under the rule and moves nothing without it. If the loaded profile is older, shells stay confined by it and the daemon logs an ERROR with the reload command, and `oara doctor` shows the same.

### The agent tool works (#762)

- **The `agent` tool no longer always fails.** It answered "SubagentSpawner not configured" on every call, because the daemon never gave it a spawner. It now runs a subagent on the same model as the turn that called it and returns the subagent's answer; the calling turn waits for it.
- **A subagent runs under the same gate as the turn that spawned it.** Its tool calls are checked like any other, approvals apply and a denial stops the call. A subagent can't spawn another.

### The `anthropic` extra installs nothing (#769)

- **`[anthropic]` no longer installs the Anthropic SDK, which nothing in Prometheus imports:** the Anthropic provider speaks HTTP itself. The SDK and three packages only it brought (jiter, docstring-parser, sniffio) leave Prometheus.app, the Docker image and `[full]`. The fourth, `distro`, becomes a declared dependency on Linux: the system prompt's OS line uses it, and it had only ever been installed by accident, through the SDK. The extra's name stays, empty, so `pip install 'oara-prometheus[anthropic]'` and `uv sync --extra anthropic` still work. Nothing changes for using Claude models.

### PDFs and large images without PyMuPDF (#767)

- **PDF text comes from pypdfium2 and image downscaling from Pillow; PyMuPDF (AGPL-3.0) is no longer a dependency.** The Mac app, which shipped without PyMuPDF, now reads PDFs and shrinks large images like every other install. The comparison on ten real PDFs (two-column papers, a 59-page IRS publication with tables, forms, scans, Chinese and Japanese) is in `docs/audits/pdf-extraction/`: the same text in the same order once ligatures and line-break hyphens are normalised, and 1.3 to 2 times faster.
- **A filled PDF form keeps what was typed.** Each page's filled form fields follow its text as a `[Form fields]` block of `name: value` lines (8 of 8 values in the test form; PyMuPDF's text carried 6). Empty fields and unchecked boxes are left out.
- **A large image is shrunk to exactly 2,576 px on its long edge,** not by halving, which could leave it up to twice that. Images up to twice the cap are still sent as captured.
- Scanned PDFs without a text layer still give no text: neither library does OCR.

### A default style for pages, reports and PDFs (#765)

- **`beacon-plate` is now a builtin skill.** When the agent builds an HTML page, report, dashboard or PDF, it loads this house style unless you ask for another one or have your own brand. It was in the copy-in `skills/` library before.
- It adds one line to every system prompt (its name and one sentence, about 35 words). The full guide, about 2,000 tokens, arrives only when the agent loads it.
- It covers print and PDF: page backgrounds that reach the sheet edge, page numbers, break rules, and the headless Chrome command.
- Its Google Fonts link is a request to Google; the guide says so and gives the offline fallback.

### Startup banner (#759)

- **`oara daemon`, `oara setup` and the interactive chat open with a banner on a terminal.** It shows the wordmark, the port the daemon listens on and the model it serves, each only once it is known; never an address, a token or a path. Nothing is printed when stdout isn't a terminal, for `oara --once`, with `--quiet` (`-q`) or `OARA_NO_BANNER=1`, and `NO_COLOR` makes it plain text.
- **`oara --version` prints `◉ prometheus <version> · OAra Labs`.** It used to print `Prometheus <version>`. On a terminal that can't show `◉` it prints `prometheus <version> - OAra Labs`.

### Fixes

- **A photo described for a model that can't see is described once, not twice (#755).** The description was added to the message twice, glued together.

### Upgrading from 0.9.8

- **`oara --version` has new text** (see above). A script that matches the old `Prometheus <version>` should match the first `x.y.z` in the line instead.
- **A default `denied_paths` list from an earlier release now resolves to the new one (#764).** Setup copies the template, so each install holds the list of its day. A config that holds a previously shipped list exactly, in order, gets the new list, and the daemon logs one INFO line at boot saying so; a config already holding the new list is silent. Any other list, `[]` included, is yours: it is used as written and keeps the override warning. One effect: the oldest shipped list has a relative entry the gate refuses at boot, so a config still carrying it now boots with the new list instead of refusing.
- **Every system prompt has one more line, for `beacon-plate` (#765).** If you copied `beacon-plate.md` into `~/.prometheus/skills/` earlier, your copy is the one used.
- **A model's `agent` call now succeeds,** so a turn that used to get an error from it now waits for the subagent.
- **Linux hosts using the AppArmor floor must reload the profile after upgrading:** `sudo apparmor_parser -r -W /etc/apparmor.d/prometheus-bash`, after copying the new `deploy/apparmor/prometheus-bash` to `/etc/apparmor.d/`. It is a root-loaded file, so upgrading Prometheus doesn't change it. Until it's reloaded, `~/.config` can still be renamed, and the daemon says so at boot. The new profile works with the old daemon, so reload it before restarting the daemon.

### Known issues

- **Built-in subagent types get no tools.** Their definitions name tools (`Bash`, `FileRead`, ...) this daemon doesn't have, so `general-purpose`, `explorer`, `planner`, `worker` and `verification` subagents answer from the model alone.
- **How an approval request raised inside a subagent appears in Beacon and Telegram has not been checked.** A subagent has no session of its own.

## 0.9.8

A Beacon on another computer can now become the owner's own device, so it can approve workers. Workers also behave better at the edges: their approval requests end when they do, and their time limit is a setting that counts only working time. Read "Upgrading from 0.9.7": a worker's time limit no longer comes from the model, and a late answer to an approval is now a 410.

### Owner devices on other computers (#748, #750)

- **`oara pair --owner` makes a Beacon on another computer an owner device (#748).**
  - Run it on the machine Prometheus runs on. It asks the daemon for a one-time code (8 digits, 5 minutes, one use), prints it, and waits to report who used it and from which address. Ctrl-C cancels it.
  - Only the API token, from that machine, can mint a code (`POST /api/pair/owner-codes`; `GET`/`DELETE /api/pair/owner-codes/{id}` report on it and cancel it). An owner device can't mint one.
  - The other computer's Beacon sends the code to `POST /api/pair/owner` (`{code, device_name, platform}`, no token needed) and gets an owner device token of its own, listed with `owner_source: "owner-code"`. Unlike the API token, that device can approve a worker's request (#744). The token is in the answer in clear, so redeem over TLS, a tailnet or loopback.
  - Re-pairing this Mac never revokes it, and neither does another computer's redemption. Every failure is one `401 invalid_code`.
  - Each code issued or redeemed is announced to the owner's open Beacons (an `owner_code` frame).
- **Wrong guesses can't switch an owner code off (#750).**
  - Five wrong tries shut out one address (an IPv6 /64 counts as one). At most ten wrong tries a minute are checked from everyone together; past that, tries are refused unchecked until the minute passes, the right code included.
  - No number of guesses ends a code, so a stranger can delay a redemption but never cancel it. A code gets about 50 chances in 10^8.
  - The terminal warns as wrong tries come in, and the code's status carries `wrong_attempts`.
  - `oara pair --owner` shows a code only on a terminal (over ssh, `ssh -t`), so asking the agent to run it, or piping it, can't put the code in a transcript or a log.

### Workers (#751, #752)

- **A worker's approval requests end when the worker does (#751).** Before, a worker stopped at its time limit left its request pending for as long as the approval window lasted.
  - When a worker's process ends for any reason (it finished, ran out of time, was stopped or killed, or a context-overflow retry replaced it), its requests leave the queue at once.
  - Every client gets `approval_resolved` with `resolution: "withdrawn"` and a `reason`, so the card drops everywhere. The chat that was asked gets a notice that it was withdrawn and nothing ran. Anything still waiting on it gets a denial.
  - A daemon shutting down or restarting withdraws everything still pending, the same way.
  - Approving or denying a request that has ended is **410** `no_longer_active`, with how it ended: withdrawn and why, already approved, already denied, expired, or no record (the daemon restarted). The body keeps `ok: false` and the usual keys. A scoped device still gets 404 for a request that isn't its own conversation's.
- **A worker's time limit is `workers.timeout_minutes`, default 60 (#752).**
  - The limit counts working time only: time spent waiting for a person to approve something doesn't count.
  - The agent can't set it. `task_create`'s `timeout_seconds` is for shell, watch and poll tasks, and is ignored for a worker; `tasks.default_timeout_seconds` no longer applies to workers either.
  - A worker stopped for time tells its conversation so: "stopped after N minutes — time limit", with its limit, any approval wait that didn't count, and what to try next.

### Models

- **Gemini and xAI keys can be checked (#740)** by `POST /api/setup/check-key` and `POST /api/model/check-key`, which answered `unsupported_provider` for them before. Both services answer an invalid key with HTTP 400 rather than 401, so the check reads each service's own error code for a bad key. Gemini's key is checked against its native model list.

### Upgrading from 0.9.7

- **Workers now stop after 60 minutes of working time** unless you set `workers.timeout_minutes`. A limit the agent passed to `task_create`, or `tasks.default_timeout_seconds`, no longer applies to workers.
- **A late approve or deny is now 410 `no_longer_active`.** It used to be `200` with `ok: false`. A client that reads only the status code should treat 410 as "this request is gone".
- **One more route answers without a token:** `POST /api/pair/owner`. It needs a code minted on the daemon's own machine, and wrong guesses are rate-limited per address and in total.
- The `owner_code` frame's `locked` event and the code status `locked` no longer occur: wrong guesses don't end a code.

### Known issues

- If `web.bind` is set to a specific LAN address (not `0.0.0.0` or loopback), worker approval requests are refused (fail closed); use `0.0.0.0` or home-network mode.

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
- **Setup refuses a key or token with a line break or another control character (#738).** The env file is line-based, so such a value could add a second variable, such as an API token. Keys and tokens are also capped in length, and a value that isn't text is refused.
- **A worker never gets the API token, and only a person approves its request (#744).**
  - Every child the daemon starts (a worker, a spawned agent, a coding run) gets the bash tool's scrubbed environment, with only the provider keys put back. The API token, `PROMETHEUS_API_TOKEN_FILE` and the gateway tokens are never passed on.
  - Approving a worker's request takes a device token: the owner's device, or the scoped device whose conversation it is. The API token and an open daemon get `401 not_a_person`, and the request stays pending.
- **A desktop thumbnail goes only to a device that owns the task's session (#742).** A scoped device used to receive every session's thumbnails, or any session it named. The owner's device still sees every session, and a task started on Telegram shows only there.
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
- **`oara doctor` reports home-network mode (#700, #727).** When `network.home_network` is on, the Home network row says whether it can work, whether it is safe, and whether the daemon will announce itself. The Web bind row says when TLS serves the LAN and plain HTTP stays on 127.0.0.1. Both rows read the same prediction the daemon's boot makes.
- **A second computer can join over TLS (#745).** An approved join request that came in on the TLS listener is now told to connect back on `https://` and `wss://` (the TLS ports), and is given the certificate pin to check. Before, it got plain `http`/`ws` addresses that only listen on 127.0.0.1 in home-network mode, so the second computer paired and could never connect.
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
  - The policy allows only the model server on the host, the cloud providers (OpenRouter included, #735), Telegram, Slack and Discord.
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
- Actions are chosen by fixed rules, not a model. A task ends when its one action has run (#672, #674, #664, #665, #710). After each step, a thumbnail of the approved window can go to marked devices that own the task's session (#701, #742); this is experimental and not yet verified with the pinned driver.

### Spawned workers (#713, #715, #719, #720)

- **A worker runs on its conversation's model route,** or on the one worker model the operator chose (`workers.model`, or `GET`/`PUT /api/workers/model`). It never runs on a model the agent names.
- **It takes its brief as one turn and exits with a result.**
  - If it runs out of context on another route, it's retried once on the conversation's route.
  - When `sentinel.enabled` is on, its result comes back into the conversation that spawned it, Beacon included. SENTINEL ships off.
- **A worker can ask for approval.** The request shows up in the parent conversation as approve-once, and only a person can approve it, with a device token (#744).
- **A daemon restart stops the task processes it can no longer supervise (#731).**
  - Each task process records its pid and start time when it launches.
  - On restart, a still-running task process is stopped with its whole process tree.
  - A pid that is gone, or now belongs to another process, is left alone.
  - Before, such a process outlived the daemon untracked.

### Models and the agent loop

- **OpenRouter is a cloud provider (#732).** Setup accepts `{"provider": "openrouter", "api_key": …}`. The default model is `anthropic/claude-sonnet-5`, a paid, tool-capable model.
  - `POST /api/setup/check-key` tests a key before it's saved. It covers OpenRouter, Anthropic and OpenAI, and answers `ok`, `bad_key`, `no_credit` or `unreachable`.
  - OpenRouter's `vendor/model` ids are no longer counted as free local traffic. Unpriced models are reported as unknown, not free.
- **Change the provider, model and API key after setup (#737).** `GET`/`PUT /api/model` and `POST /api/model/check-key`, operator only.
  - A change is saved to the config file and applied on restart (`POST /api/restart`). Cloud providers only.
  - The key is checked first, and a rejected key writes nothing. It goes to the env file only, never the config.
  - Switching providers drops the old vendor's key, base URL and key variable from `model:`. A value pinned by `PROMETHEUS_PROVIDER` or `PROMETHEUS_MODEL` answers `409 model_overridden`.
- **Gemini and xAI can be chosen in setup (#739):** `POST /api/setup/configure`, `oara setup --provider gemini` or `xai`, and `GET /api/setup/detect` offer them, and `PUT /api/model` can switch to them.
  - The env-file template now names `GEMINI_API_KEY`. It said `GOOGLE_API_KEY`, which the daemon never read.
  - Their keys are saved without a check: the key checks cover OpenRouter, Anthropic and OpenAI.
- **Every setup failure answers with a stable code (#738)**, listed in `docs/contracts/setup-errors.md`. Before, some answered in bare text.
  - Setup now refuses a provider it doesn't offer. Before, one was written into the config.
  - An unexpected error is coded JSON that never echoes the exception's message.
- **`PROMETHEUS_OLLAMA_URL` and `PROMETHEUS_LLAMA_CPP_URL` now work (#725).** They set the main model's URL when it uses that provider; otherwise the daemon logs a warning. Before, they did nothing.
- **A prompt too long for the backend's window now ends the turn with a plain explanation and the server's numbers (#716),** not "400 Bad Request". The CLI and every worker now compact like the daemon, using the server's real context window (#718).
- **A `null` for an optional tool parameter takes its default (#666),** rather than failing or becoming a file named "None". This applies to local models; cloud models skip the adapter.
- Microcompaction never makes a tool result longer (#648). A cloud run's breaker recovery no longer starts microcompaction (#637).
- `/mimo` now defaults to mimo-v2.6-pro, ahead of v2.5's retirement on 2026-10-21. The picker labels gpt-6-astra "no tool calls" (#686).
- `POST /api/chat` uses the daemon's full runtime system prompt (#680).
- **MiMo refuses forced tool calls with a clear error (#726).** MiMo supports only `tool_choice: auto` and silently drops anything else, so a forced tool call now fails with an error that says so; automatic tool choice is unchanged. Contributed by Sepehr-Taji (@sepehrtaji-dev).
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
- The guides cover this release's features: TLS and the network modes, cautious mode, restart, workers, the Seatbelt floor and Docker (#733).
- The README leads with the always-on daemon (#658, #673). The README, the API and install guides and the config template now describe TLS in home-network mode and `oara install-service` on macOS, which they still denied.
- Project: the PR-gating workflows also run in a merge queue (#730), and the workflow actions are on current releases (#723). A Linux test switches a real daemon to home-network mode, restarts it, and checks that it comes back on TLS (#729). Tests pin that a marked device reaches the computer-use binding of its own sessions only (#743).

### Upgrading from 0.9.6

- **Device tokens you minted before become scoped.** They can chat and manage only the sessions they create, see none of the earlier conversations, and get 403 `operator_only` elsewhere (including `/api/status` and `/api/config`). The global token is unchanged.
- **On macOS (terminal installs too), model-written shells lose `~/.ssh`, `~/.gnupg` and `~/.config/*/*env`.** Agent `git` over SSH and gpg-signed commits fail. Set `security.bash_allow_ssh: true` (SSH only) or `security.bash_confinement: off`.
- **Two new routes answer without a token, wherever the daemon listens:** `GET /api/hello` (name, version, fingerprint) and `POST /api/pair/requests` (rate-limited, and it needs your approval). `pairing.requests_enabled: false` turns join requests off.
- **Behind a reverse proxy or tunnel, name it in `web.trusted_proxies`.** Forwarded headers are otherwise ignored, so every client looks like 127.0.0.1.
- **A config without `web.bind` still listens on every interface.** `oara doctor` and the startup log now warn about it. Set `web.bind: 127.0.0.1` to keep the daemon on this machine.
- **If your config came from the 0.9.6 template, edit `slash_commands.mimo.model`.** It sets `mimo-v2.5-pro`, which overrides the new default and stops working on 2026-10-21.
- **The daemon writes `~/.prometheus/node/instance.key` and `instance.crt` at boot.** Keep them: the key is the fingerprint paired devices pin.
- `oara install-service` with no systemctl or launchctl now exits 1. `/gate on` restores your configured mode, and re-running `oara setup` keeps it.

### Known issues

- If `web.bind` is set to a specific LAN address (not `0.0.0.0` or loopback), worker approval requests are refused (fail closed); use `0.0.0.0` or home-network mode.
