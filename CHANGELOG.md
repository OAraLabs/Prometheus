# Changelog

Notable changes in each release, newest first. Releases before 0.9.7 are described on the [GitHub releases page](https://github.com/OAraLabs/Prometheus/releases).

## 0.9.15 (2026-10-19)

0.9.15 raises the aiohttp floor to 3.14.4, which fixes three advisories against 3.14.3 and earlier, and makes the `slack` extra install aiohttp, which the Slack gateway needs. Example device names in the docs and tests are now generic.

### aiohttp 3.14.4 (#829)

- **What changed.** uv.lock moves aiohttp from 3.14.3 to 3.14.4, and yarl from 1.24.2 to 1.25.1 (aiohttp 3.14.4 requires it). 3.14.4 fixes CVE-2026-111063, CVE-2026-111064 and GHSA-2g87-pp6c-29x5.
- **The floor.** The `discord` and `full` extras now require `aiohttp>=3.14.4`, so `pip install -U` cannot keep an installed 3.14.3.
- **Where it applies.** Prometheus uses aiohttp only as a client: discord.py for the Discord gateway, and slack-bolt for the Slack gateway. It runs no aiohttp server. The 0.9.14 Mac app and Docker image shipped 3.14.3; the 0.9.15 builds ship 3.14.4.

### The `slack` extra installs aiohttp (#PR)

- **What went wrong.** The Slack gateway uses slack-bolt's `AsyncApp` with Socket Mode, which imports aiohttp. Neither slack-bolt nor slack-sdk declares aiohttp, and the `slack` extra didn't either, so `pip install 'oara-prometheus[slack]'` without `discord` or `full` installed none. The gateway then failed to start with "slack-bolt is required", although slack-bolt was installed. The Mac app and Docker image install the `discord` extra too, so they were not affected.
- **What changed.**
  - The `slack` extra requires `aiohttp>=3.14.4`, the same floor as `discord` and `full`.
  - When a module the Slack gateway needs is missing, the error names that module and points at `pip install 'oara-prometheus[slack]'`.

### Generic example device name (#829)

- Pairing docs, comments, tests and fixtures use "Alex's MacBook" as the example device name. The pairing seal test vector was re-sealed with its own key and nonce, so only its plaintext and ciphertext changed.

### Upgrading from 0.9.14

- **No config change and no migration.**
- A pip install with the `slack` extra gains aiohttp on upgrade.

### Known issues

- The known issues listed for 0.9.13 still apply.

## 0.9.14 (2026-10-17)

0.9.14 fixes two faults that appear when Prometheus is installed as a package rather than run from a Prometheus checkout, for example with Homebrew. The daemon no longer runs `git fetch` in whatever repository happens to surround its install, and its status no longer reports that repository's commit as its own. A Prometheus checkout, such as a deploy clone, behaves exactly as before.

### The origin refresh runs only in a Prometheus checkout (#826)

- **What went wrong.** Every 5 minutes the daemon refreshed `origin/main` by running `git fetch` three directories above its own code, in whatever repository git found there.
  - On a Homebrew install that is Homebrew's own repository (`/opt/homebrew`), so a brew-installed daemon fetched Homebrew's repository and wrote `refs/remotes/origin/main` into it every 5 minutes.
  - A package installed into a virtual environment inside one of your own projects did the same to that project.
  - No credentials were involved.
- **What changed.** The refresh runs only from a Prometheus checkout, which must pass three checks:
  - the code sits at `<root>/src/prometheus`;
  - `<root>/pyproject.toml` names `oara-prometheus`;
  - git's top level is `<root>` itself.

  Anywhere else (Homebrew, pip, `uv tool`, the Mac app), the refresh is skipped without a log line, and `/api/status` reports `origin_fetch.state: "not_checkout"`.

### Status reports commits only from a Prometheus checkout (#827)

- **What went wrong.** The same lookup fed `/api/status`: `running_sha`, `tree_head`, `stale` and the deployment block all described the surrounding repository. On a Homebrew install:
  - they showed Homebrew's commit, and Beacon on iOS displayed it as the RUNNING version;
  - after `brew update`, the heartbeat sent "new code on disk this process isn't executing. Restart to go live." when nothing about Prometheus had changed.
- **What changed.**
  - Outside a Prometheus checkout, `running_sha` and `tree_head` are `null`, `stale` is `false`, and the deployment state reads `not_checkout`, so the false restart nudge is gone.
  - If git can't be run to decide, the fields read `unknown`, as before.
  - In a checkout nothing changes.

### Upgrading from 0.9.13

- **On an install that isn't a Prometheus checkout** (Homebrew, pip, `uv tool`, the Mac app), `/api/status` changes:
  - `running_sha` and `tree_head` are `null`, where they read Homebrew's commit or `"unknown"`;
  - `deployment.state`, and `/health`'s `deployment_state`, read `not_checkout`;
  - `checkout` gains `source_checkout`: `false`, or `null` when git could not be asked.

  Beacon on iOS hides its RUNNING row instead of showing another repository's commit. A client that ignores unknown fields and accepts `null` there is unaffected.
- **No config change and no migration.** A Prometheus checkout, such as a deploy clone, is unaffected.

### Known issues

- The known issues listed for 0.9.13 still apply.

## 0.9.13 (2026-10-16)

0.9.13 fixes tool calling on local models. A local turn can now call every tool, where since late July it could reach only the dozen or so tools always sent to it. It also does the following:
- shrinks the first request of a local turn from about 19.2k to about 13.8k tokens on the reference install;
- puts the llama.cpp reply cap back to 4096, so a 32k session no longer compacts early;
- makes Beacon show the text of turns the loop ends itself;
- sends Ollama the context window Prometheus plans for;
- reports context-window guesses and dead overrides at startup;
- makes `oara doctor` say what a local turn is actually offered.

Read "Upgrading from 0.9.12": the daemon no longer puts project files from its own directory into the prompt, `always_loaded` now applies only to the old catalog, and `context.reserved_output` is no longer read.

### Every tool can be called on a local turn (#819)

- **What went wrong.** Since #121 (31 July), a local turn sent the model only the `always_loaded` tools, about a dozen. The others were meant to be found with `tool_search`, but that never worked on llama.cpp:
  - A model does not call a tool that isn't in the request's tool list.
  - llama.cpp's grammar for Qwen's tool format only allows listed tool names and their declared arguments. `tool_search` returned a tool's schema as text, which doesn't add the tool to that list.

  On 0.9.12 the 27B on the reference install called none of 7 plain requests that needed such a tool (scheduling a job, searching past conversations, the wiki, an MCP tool, a subagent, and so on).
- **What changed.** A local turn now offers every tool:
  - Ten common tools carry their full schema: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `web_search`, `web_fetch`, `agent` and `ask_user`.
  - Every other tool is a short stub: its name, a one-line description, and its argument names with their types. An enum's values are listed when there are 8 or fewer.
  - The list is sorted and built once per turn, so it is the same in every round and the server's prompt cache keeps working.
  - A call to a stubbed tool is checked against the tool's full schema. A wrong call is refused with the full schema and a minimal valid example, at most twice per tool per turn.
- **Measured** on the 27B with 12 plain requests, each needing one of the stubbed tools and none naming it:
  - The right tool was called in 12 of 12, and called first in 12 of 12.
  - The first call's arguments were right in 11 of 12. The twelfth, a `memory` update, left out an argument and got it right two calls later, after one refusal.
  - Before choosing stubs we compared them with a single `call_tool` dispatcher. Both found the right tool equally often, but the dispatcher got the arguments right on the first try in 9 of 12, against 11 of 12 for stubs.
- **Fallback rounds.** When a round is served by another model (the provider fallback after an authentication or billing failure, or the circuit breaker's model fallback), its tools now follow the model that serves it: stubs for a local model, every full schema for a cloud one.
- **Cloud turns are unchanged.** They still get every tool with its full schema.
- **To keep the 0.9.12 catalog,** set `tools.deferred_loading.local_catalog: names`.

### Smaller first request on local turns (#820)

- **Round 0** is a turn's first request, before any conversation. With every tool offered, it was 19,198 tokens on the reference install, 59% of a 32k window. It is now 13,809.
- **What was cut:**
  - On llama.cpp at tier light, the Qwen formatter no longer adds a prose tool list and a JSON example to the system prompt. The tools are already in the request, and the example contradicted Qwen's own format. About 1.4k tokens.
  - The daemon no longer puts the project files from the directory it was started in into the prompt (about 2.8k tokens on the reference install, where that directory is the deploy checkout and the file is Prometheus's own developer guide). A session bound to a workspace with `/workspace` still gets that workspace's files.
  - Saved files appear as a count and a location instead of a full listing. The model lists them with `glob` when it needs a name. About 1.1k tokens.
  - Stub descriptions are capped at 80 characters.
- **The project-file and saved-file changes apply to cloud turns too.** The stub cap affects only local turns, the only place stubs are used.
- **A new check:** the regression gate now fails if Prometheus's own part of round 0 goes over 7,400 tokens. The operator's files (memory, USER.md, SOUL.md and so on) are not counted.

### The llama.cpp reply cap is back to 4096, so a 32k session compacts on time (#813)

- 0.9.12 asked llama.cpp for up to 8,192 reply tokens, a quarter of a 32k window. Compaction keeps room for the next reply, so on the reference install the compaction line fell from 26,214 to about 22.3k tokens, and a first turn was summarised after two or three tool results.
- The cap is 4096 again, as it was before 0.9.12. A per-model `max_output_tokens` still overrides it, and a reply cut at the cap is still told to send its work in smaller pieces.

### Beacon shows the text of turns the loop ends itself (#811)

- **What went wrong.** Some turns end with text the loop writes rather than the model streams. That happens for the repeat guard, the iteration cap, two empty replies, the context pre-flight, an overflow, the circuit breaker, a boundary escape and the divergence halt. The WebSocket bridge forwarded only streamed text, so Beacon showed an empty reply while the text sat in history.
- **What changed.** The bridge now sends any part of the final text that didn't stream, and `chat_done` carries the loop's `stop_reason` when the loop ended the turn. The field is absent when the model chose to stop.

### Ollama gets the context window Prometheus plans for (#815)

- **What went wrong.** Prometheus spoke to Ollama through its OpenAI-compatible `/v1` route, which can't set the context window. Prometheus planned for the model's trained length (262,144 tokens for qwen3.5:9b) while Ollama served its default window. A prompt longer than the served window was cut from the front with no error. Measured: 36.8k tokens sent, 32,768 kept, the start of the system prompt gone.
- **What changed.**
  - Prometheus now uses Ollama's native `/api/chat` and sends `num_ctx`: the window it planned for, never above the model's trained length.
  - It also sends `num_predict`, the reply cap.
  - A request without a planned window uses the window the model is already loaded at, so it doesn't force a reload.
  - Thinking is still off by default, now as `think: false`.
  - The backend probe reports the window a loaded model serves.

### Context-window problems are reported at startup (#814)

- **Warnings at boot** for:
  - a window that is a guess: the local server reported none, or a backend has neither a measured window nor a `context_limit`;
  - each `context.model_overrides` entry that no served or routed model matches;
  - a backend planned above its trained length;
  - `context.reserved_output`, which is no longer read.

  `/api/status` reports them too, in its context block (`fallback`, `unmatched_overrides`).
- **One way to match overrides.** The window, the reply cap and the compaction threshold all match a model's override the same way: by exact id, then by GGUF file name (`/models/a/X.gguf`, `X.gguf` and `X` are the same model). A moved GGUF keeps its override; a renamed one does not. If two keys name the same file, neither applies, and a warning says so.
- **Known cloud windows.** grok-4.5 is planned at 500,000 tokens and kimi-k2.6 at 262,144, from the vendors' figures. Before, both got the 1,000,000 default. Any other cloud model on the default gets one warning.
- **One reply room.** `/context`, document sizing and the token budget now reserve the request's actual reply cap, the same number compaction uses, instead of `reserved_output` (2,000).

### `oara doctor` says what a local turn is offered (#819)

- Under the new catalog, the "Advertised tools" row counts every tool a local turn gets, for example "51 of 51 registered offered to a local model (10 with their full schema, 41 as stubs)". Before, it counted the `always_loaded` list, and it warned about a list that no longer decides anything.
- The chat `/doctor`, the boot line and `/api/tools/deferred` report the same.

### `tool_search` finds what the model means (#816)

- `select` accepts near misses such as "cron create" or "the lcm_grep tool", and resolves one only when exactly one tool fits.
- A miss lists the 5 closest names instead of every tool and skill, which was about 1,700 tokens.
- Each tool declares one or two plain requests it answers ("schedule a recurring job") that the search ranks on. These never reach the schema.
- This matters less now that every tool is offered, but it still applies under `local_catalog: names` and for skills.

### Telemetry records the tools a round actually sent (#818)

- The lucky-guess marker ("a tool was called without being offered") recomputed what had been offered, and got it wrong after routing or a fallback. 25 of the 28 lucky guesses recorded on the reference install were mislabels.
- Each round now records the tool names it sent, and a round that sent a different set writes its own `tool_sets` row. Expect far fewer lucky guesses on dashboards.

### Under the hood (#810, #812)

- **A regression gate** for local tool calling and context: hermetic tests run in CI, plus a real-model suite for a local GPU in `scripts/local_gate/`. The audit behind this release is in `docs/audits/local-tools-audit-2026-10-16/`.
- **The parity harness** matches the session-title and summary side calls by kind, in any order. The turn's own requests still have to match exactly.
- Hardening: daemon process is non-dumpable on Linux; the service unit no longer passes keys through the environment

### Upgrading from 0.9.12

- **Local turns offer every tool.** No config change is needed.
  - `tools.deferred_loading.always_loaded` is ignored under the default `local_catalog: stubs`. If you added a tool to it to make it reachable (`agent`, say), that is no longer needed.
  - `local_catalog: names` brings back the 0.9.12 catalog.
- **A hand-set 4096 reply cap for a local model** (`context.model_overrides.<model>.max_output_tokens: 4096`) is now the default for llama.cpp, so the override can go.
- **The daemon no longer reads project files (PROMETHEUS.md, CLAUDE.md, AGENTS.md and the like) from the directory it was started in.** If you started the daemon inside a project to give it that project's instructions, bind a session to the project with `/workspace` instead. The CLI is unchanged.
- **`context.reserved_output` is no longer read.** It logs a warning at startup and can be removed. The reply room is each model's reply cap.
- **New startup warnings** for unmatched overrides and guessed windows change nothing at runtime, but they point at config worth checking.
- **Ollama is reached at `/api/chat`, not `/v1`.** A proxy in front of Ollama has to pass `/api/chat`. Prometheus now asks for a window explicitly: if that differs from the window a model is loaded at, Ollama loads the model again at the requested size, and a larger window uses more memory.
- **New wire fields:**
  - `chat_done.stop_reason`;
  - `/api/tools/deferred`: `full_schema_count`, `stub_count`, `local_catalog`, `always_loaded_applies`;
  - `/api/status`'s context block: `fallback` and `unmatched_overrides`;
  - the `tool_advertisement` telemetry row: `catalog` and `stubs`.

  A client that ignores unknown fields is unaffected.

### Known issues

- **Local models still make some argument mistakes on stubbed tools.** In the release run, one first call in 12 left out a required argument. The refusal with the full schema led to a correct call, but it cost two extra calls.
- **Round 0 is still about 13.8k tokens of a 32k window on the reference install,** about 5.8k of it the operator's own files. Long local sessions still compact. Beacon can't show the breakdown yet: no per-component overhead reaches the client.
- **The known issues listed for 0.9.12 still apply.** These include a call the cap cuts off still being able to end the turn through the circuit breaker, the gaps in cut-off checking, #798, #799 and #800.

## 0.9.12 (2026-10-15)

0.9.12 makes compaction count in the server's own tokens and retry once when a server says the prompt is too long, lets Beacon sessions use subagents, gives each model its own reply-size cap and refuses a tool call that the cap cut off, and fixes two smaller faults: `web_search` failing with a blank error, and Beacon badging every reply "untrusted". It also carries the one change from 0.9.11, which was tagged but never published. Read "Upgrading from 0.9.10": local sessions will compact more often, and the reply cap is no longer a flat 4096.

### Compaction counts in the server's tokens, and retries once (#805)

- **What went wrong.** On 2026-10-15 a Beacon session on a local model with a 32,768-token window reached 28,915 prompt tokens by llama.cpp's own count (88%) without compacting. The window was right and so was the threshold (26,214, which is 80%). The trigger compared them with a characters-divided-by-four estimate that reads low: a median of 1.067 against the server's count over 279 compactions on the reference install, and 1.19 to 1.32 on that session's web-search results. Past 32,768 llama.cpp refuses the request; since #716 that ends the turn with "too long… /reset", and the turn's work is lost. There have been 133 such refusals since August.
- **Calibration.** After each round the loop tells the compactor what the provider counted for the prompt it sent (`usage.input_tokens`). The compactor keeps that ratio over its own estimate, per session; a session's first round borrows the model's last ratio. The ratio is clamped to 1.0 to 1.5: never below 1.0, because Ollama's `prompt_eval_count` leaves a cached prefix out and a low count must not make compaction fire later than before; never above 1.5, so one odd count, such as an image, can't shrink the window to nothing. It is used by the threshold, the projected-overflow rule, the pre-flight check, the fallback window check, and the summarizer's own chunk size. At a ratio of 1.0 every number is what it was, and a degraded round (another model served it) is skipped.
- **Retry once on a "too long" refusal.** A `context_overflow` (llama.cpp, OpenAI-compatible and Anthropic shapes) carries the prompt's real size and the real window. The real size becomes a calibration observation, and the stated window is learned, which only ever lowers a window and lasts for the process. That covers a cloud model with an unlisted window, which was budgeted at 1,000,000 tokens and never compacted before the provider refused. The round then runs again with compaction seeing the prompt over the line, once per round. A second refusal, a degraded round, or no compactor ends the turn with the same explanation as before; if the pre-flight, now holding the server's numbers, sees that the compacted prompt still can't fit, it ends the turn without a second request. The retry is recorded as a `context_overflow_retry` row in `subsystem_runs`.
- **Events.** `context_compaction` and `context_compaction_noop` now give token counts in the server's tokens, plus `calibration` and `tokens_before_estimated`.
- **Not changed.** `/context` and Beacon's meter still show the bare estimate. A steer drained into the refused request isn't re-sent with the retry (the same as the degenerate-output retry). A learned window isn't saved; a restart forgets it until the next refusal.

### Subagents work from Beacon (#806)

- **The `agent` tool from Beacon, REST and WebSocket turns.** `run_daemon()` builds two loops. #761 and #762 attached the subagent spawner to the one that Telegram and the CLI use, but the web context that every Beacon turn runs on had no tool metadata at all, so from Beacon `agent` always answered "SubagentSpawner not configured". The printing-press registry was missing there the same way. The web context now gets the other loop's own metadata, not a copy.
- **A boot check.** `loop_tool_metadata_divergence` logs an error if the two loops ever carry different keys.
- **Unchanged.** A subagent runs on the daemon's primary model, as it already does from Telegram; a session's `/qwen` route does not carry over.

### Each model gets its own reply cap; a cut-off tool call is refused and explained (#806)

- **The cap.** `max_tokens` was hard-coded to 4096 for every request. It is now 4096 only for models Prometheus knows nothing about. The order is: `context.model_overrides.<model>.max_output_tokens`; the route's own `max_tokens` (which was ignored until now); a table of known cloud models (qwen3.8-max 16384, and 32768 for models that think inside the cap); 8192 for llama.cpp; then 4096. The result is clamped to the vendor's documented maximum. On a window a local server measured, one reply never asks for more than a quarter of it (32768 gives 8192, 16384 gives 4096). The compactor's projected-overflow rule plans around the cap actually sent. It is wired into both daemon loops, the CLI and coding runs.
- **A cut-off tool call is never run.** On 2026-10-15 a Beacon session on qwen3.8-max wrote a report into `write_file` and `bash` over five rounds, and each call arrived as `{}`, until the circuit breaker ended the turn. Providers now name every call whose arguments don't read as a JSON object, and the loop never runs one. If the reply ended at the limit, the model is told the call was cut off at N tokens, that nothing ran, and to send it in smaller pieces. "Ended at the limit" means the stop reason says so, or the output tokens reached the cap; the second signal matters because DashScope reported `finish_reason: tool_calls` on all five cut replies. A call with unreadable arguments in a reply that ended normally is refused as "not valid JSON, not run". These refusals stay out of the repeat guard, so a second cut call isn't blocked as a repeat of the first `{}`.
- **Replaces #685.** That pull request is superseded; its tests are carried over.

### `web_search` names the error and retries once (#806)

- The tool printed only the exception's text, and several httpx errors have an empty one, so every failure read "web_search failed: " with nothing after it. The failures took 60 to 125 ms, against 600 to 900 ms for a good search, and on a scratch daemon the next try a second later worked.
- It now reads `web_search failed: ReadError (tried 2 times)`, after one retry with a 1 s pause.

### Beacon no longer badges every reply "untrusted" (#806)

- `ConversationMessage.is_trusted` defaults to false and nothing set it on the model's own replies, so every assistant row since 2026-06-10 is stored as untrusted and Beacon badged every reply that way. `GET /api/sessions/{id}/messages` now reports an assistant reply of user provenance as trusted and every other row as stored. There is no migration, and the model never saw the difference: its untrusted banner applies only to user-role turns of non-user provenance.

### Coding mode sees files at the repo root (#802, fixes #801; first tagged as 0.9.11)

- **`code_glob` and `code_grep` now match `**/` against zero directories, too.** They read `**/` as "at least one directory", so `**/*.py` never listed `export.py` at the root, `code_grep`'s default `**/*` never searched a root-level file, and `src/**/*.py` missed the files directly under `src/`. In the runs reported in #801, a model that could see only the test file rewrote it and broke it.
- Only matches are added: everything else about the patterns is unchanged (`*` still spans `/`, and `?` and `[...]` work as before). The tools' descriptions are unchanged.

### Upgrading from 0.9.10

- **Local sessions will compact more often.** On a 32k window about 19k tokens are the system prompt and tool schemas before any conversation, so the 80% line leaves about 7k tokens of real conversation. That is what 80% means once it is measured in the server's units. If it proves too eager, `compaction.threshold_pct`, or a per-model `compaction_threshold_pct`, is the knob. No config change is needed.
- **The reply cap changed for every request.** A llama.cpp model now asks for up to 8192 tokens (and no more than a quarter of a measured window), a known cloud model gets its table value, and an unknown model keeps 4096. A route that sets its own `max_tokens` now has it honoured. Set `context.model_overrides.<model>.max_output_tokens` to pin one model's cap; `config/prometheus.yaml.default` has a commented example.
- **Two fields changed on the wire.** `context_compaction` and `context_compaction_noop` carry server-token counts plus `calibration` and `tokens_before_estimated`, and `GET /api/sessions/{id}/messages` reports an assistant reply as `is_trusted: true`. A client that ignores fields it doesn't know is unaffected.
- **A coding run's `code_glob` and `code_grep` can return more files than before,** never fewer.

### Known issues

- **A call the cap cuts off can still end the turn.** The circuit breaker counts repeated cut-off refusals like repeated malformed calls: three in a row trip it, and on a local model its remedy is to raise the adapter tier, which doesn't help with length. A model that keeps sending an oversized call may end the turn instead of sending smaller pieces. Not changed here, because loosening the breaker is a gate decision.
- **Cut-off checking has gaps.** It doesn't cover a call that a local adapter extracts from text, a subagent still runs with a flat 4096-token cap, and MiMo documents `max_completion_tokens`, not the `max_tokens` we send, so our cap may not apply there (#688).
- **A daemon started with `--config <path>` outside the search path does not hand that path on (#799, #800).** A managed coding run is accepted and then refuses with "coding mode is disabled" (#799). The `prometheus.jobs.daily_briefing` cron job reads its config through the search path instead (#800; found by reading the source, not yet seen in a run). An install whose config is at `config/prometheus.yaml` in the checkout or at `~/.prometheus/prometheus.yaml` is unaffected.
- **The memory extractor can save no facts, silently, when the model pretty-prints its JSON (#798).** Each extraction call is recorded as a success and nothing is parsed, so no new facts are saved and the wiki doesn't grow. Seen with a 7B model on Ollama. The log line `MemoryExtractor: persisted 0 memories from N messages` is the only sign.
- The known issues listed for 0.9.10 still apply.

## 0.9.11

0.9.11 was tagged (`v0.9.11`) but never published: there is no release, wheel or app for it. Its one change, the `**/` fix for `code_glob` and `code_grep` (#802), is in 0.9.12.

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
