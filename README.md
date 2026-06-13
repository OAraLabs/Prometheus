# Prometheus

A sovereign agent harness for local LLMs — the validation layer that makes open models actually reliable in a tool loop.

**The model is the agent. The harness is the vehicle.**

```bash
pip install oara-prometheus[full]
prometheus init           # auto-detects llama.cpp / Ollama / LM Studio / vLLM
prometheus daemon         # starts agent, Beacon dashboard, gateways
```

`prometheus init` probes for a running local inference server, asks two or three questions, writes a working `~/.prometheus/prometheus.yaml`, and points you at the Beacon dashboard. Five minutes from `pip install` to a working agent.

**What it gives you:**

- **Reliable tool calls on open models** — a Model Adapter Layer validates every call, auto-repairs common errors (fuzzy names, JSON inside markdown fences, type coercion), and enforces output schemas at the token level via GBNF for llama.cpp.
- **Always-on gateways** — Telegram and Slack at parity (23 slash commands each), with mid-turn `/steer` and `/queue` for durability while the agent is mid-task.
- **Visible memory and skills** — `MEMORY.md` and `USER.md` you can read, skills the agent writes for itself that you can pin, plus a weekly Curator pass that consolidates and prunes.
- **Lossless context** — DAG-based compression with full-text search so long sessions don't drop facts.
- **Beacon dashboard** — live activity feed, memory viewer, skills browser at `http://localhost:8005` once the daemon is up.

> **Status:** Active development. Expect rough edges. Fixes land weekly. Feedback welcome.

Deeper docs live under [`docs/`](docs/) — architecture, model registry, adapter strictness tuning, sprint reports. The rest of this README is the high-level tour; jump to the section you care about.

---

## Prior art and influences

**Not a fork, and not a port.** Prometheus is an original Python codebase. Several of its subsystems were designed by studying the *architecture and behavior* of proven open-source projects, then reimplementing those ideas from scratch — no source code was copied or translated. The table credits the prior art that shaped each subsystem's design:

| Subsystem | Influence | What the design drew on |
|-----------|-----------|-------------------------|
| Agent loop | Claude Code patterns (via Sigrid Jin's clean-room analysis) | The tool-registry / hook-pipeline / permission-governance shape |
| Always-on gateway | Hermes | The always-on messaging + cron + credential-rotation model |
| Context management | Lossless-Claw / OpenClaw | The DAG-based lossless-compression-with-FTS approach |
| Production patterns | OpenClaw | Memory-extractor / archive-bridge / heartbeat patterns |
| Knowledge base | Karpathy's LLM Wiki (concept) | Memory extraction → wiki pages → cross-referenced knowledge |

Where a subsystem's design was shaped by external prior art, the relevant source file credits that influence in a header comment. The implementation is Prometheus's own.

## What we built from scratch

Three subsystems had no meaningful prior art and are wholly original:

- **Model Adapter Layer** — the gap between Claude-quality tool-calling and what open models actually produce. Validates, auto-repairs, enforces output schemas, retries with specific error context.
- **SENTINEL** — a proactive layer that watches for idle time and acts, instead of only reacting to prompts. Nudges, dreams, synthesizes.
- **Wiki Knowledge System** — turns every conversation into a compounding knowledge base that cross-references itself over time.

These are where most of the interesting work lives.

---

## The Problem Nobody Else Solves

Open models are getting good at conversation. They're still terrible at *doing things*. Ask Qwen to call a tool and it hallucinates the tool name. Ask Gemma to return JSON and it wraps it in markdown. Ask Llama to chain three tool calls and it drops a required parameter on the second one.

Every other agent harness — LangChain, CrewAI, AutoGen — assumes the model will get tool calls right. That works fine when you're paying OpenAI. It falls apart the moment you point it at a local model.

Prometheus fixes this with a Model Adapter Layer that sits between your agent loop and whatever LLM you're running. Every tool call gets validated before execution, common errors get auto-repaired (fuzzy name matching, JSON extraction from markdown fences, type coercion), and when something still fails, the model gets specific error feedback with the actual schema — not a generic "try again." For llama.cpp, it goes further: GBNF grammar constraints force valid JSON at the token level, so the model literally can't produce malformed output.

The result: open models that reliably call tools, chain multi-step tasks, and run autonomously — without you babysitting every interaction.

## What Makes This Different

Prometheus isn't a wrapper around `ollama.chat()`. It's a complete agent operating system with novel systems that don't exist in other harnesses:

**The Model Adapter Layer is the core innovation.** Four cascading extraction strategies handle whatever mess the model produces. A retry engine feeds specific schema errors back to the model. GBNF grammar enforcement at the llama.cpp level makes invalid JSON structurally impossible. Telemetry tracks success rates per model per tool so you know exactly where your model struggles. Nothing else does this — other harnesses either assume clean output or crash.

**Lossless Context Management means your agent never forgets.** Every message is persisted to SQLite. When context fills up, a two-tier compression system kicks in: Tier 1 strips `tool_result` content from old messages (free — the output was already acted on). Tier 2 uses LLM-powered batch summarization when pruning alone isn't enough. But the originals are always recoverable — old messages get summarized into a DAG structure, and the agent can expand any summary back to full detail on demand. Full-text search across your entire conversation history. The DAG-compression approach is informed by Lossless-Claw; the Python implementation is original to Prometheus.

**SENTINEL transforms the agent from reactive to proactive.** Most agents sit idle until you talk to them. Prometheus has a background intelligence layer that watches tool performance patterns, consolidates memory, lints its own knowledge base, and discovers cross-entity insights — all while you're away. Three of four phases use zero LLM calls. The fourth is budget-capped at 2,000 tokens. It nudges you via Telegram when it finds something interesting but never acts without permission.

**A compounding knowledge base inspired by Karpathy's LLM Wiki concept.** Every 30 minutes, a Memory Extractor pulls structured facts from your conversations. A WikiCompiler builds entity pages with cross-references. The wiki grows and connects itself over time. Point Obsidian at the markdown files and the graph view lights up.

**Infrastructure self-awareness via the AnatomyScanner.** At startup, Prometheus scans your hardware (CPU, RAM, GPU VRAM), detects the loaded model and its quantization, maps your Tailscale network peers, checks disk usage, and generates `ANATOMY.md` with Mermaid architecture diagrams of your entire setup. The agent knows exactly what machine it's running on, what model is loaded, and what resources are available — and it uses this to answer questions about its own infrastructure. No other agent harness does this.

**An evaluation framework with a local LLM judge.** Most agent evals require API calls to GPT-4. Prometheus uses constrained-decoding on your local model to judge task completion, tool usage accuracy, and hallucination — zero API cost. 21 atomic tests + 5 multi-step tests, with failure classification (model vs harness vs unclear) and trend tracking across models and runs.

**LSP integration for compiler-grade code intelligence.** Instead of grepping for function names, the agent queries language servers for real symbol definitions, type errors, and references. After every file edit, a diagnostics hook automatically checks for type errors and feeds them back to the model in the same turn. This compensates for open models being weaker at code reasoning — the LSP gives them ground truth.

## Open Models First, APIs Welcome

Prometheus is built for local inference. That's the whole point — sovereignty, privacy, no subscriptions. But it's not religious about it. If you want to use cloud APIs, the same harness works with:

- OpenAI (GPT-4o, o3-mini)
- Anthropic (Claude Sonnet, Haiku)
- Google Gemini (Flash, Pro)
- xAI (Grok)
- Any OpenAI-compatible endpoint (vLLM, LiteLLM, Together, etc.)

The setup wizard lets you pick. The Model Router can even mix them — route coding tasks to your local 70B model, quick answers to a fast API, and fall back to cloud when your GPU is busy. The adapter layer adjusts its strictness automatically: full validation for open models, passthrough for APIs that already handle tool calling well.

The architecture doesn't care where the tokens come from. It cares that the tools get called correctly.

## Features

### Model Independence

- Runs any model llama.cpp or Ollama can serve — Qwen, Gemma, Llama, Mistral, Phi, DeepSeek, Command-R
- Optimized formatters for Qwen and Gemma, default formatter works with everything else
- Auto-detects whatever model is loaded — swap the GGUF, restart, done
- 6+ providers: llama.cpp, Ollama, OpenAI-compatible (covers OpenAI/Gemini/xAI), Anthropic, stub
- Configurable adapter strictness: STRICT (small models), MEDIUM (Qwen/Gemma), NONE (cloud APIs)

### Model Router + Fallback Chains

- Automatic task-based provider selection — route by task type (code, reasoning, quick answer)
- Fallback chains — if your primary provider fails, try the next one automatically
- Divergence detection — checkpoint/rollback for long autonomous tasks, goal-alignment scoring, auto-rollback at configurable trust levels
- Credential pool rotation with dead-key cooldown for multi-key API setups

### 43 Builtin Tools

`bash`, `file_read`, `file_write`, `file_edit`, `grep`, `glob`, `web_search`, `web_fetch`, `cron_create`, `cron_delete`, `cron_list`, `task_create`, `task_get`, `task_list`, `task_update`, `task_stop`, `task_output`, `todo_write`, `skill`, `agent` (subagent spawning), `ask_user`, `message`, `tts`, `notebook_edit`, `dashboard`, `browser`, `vision`, `anatomy`, `audit_query`, `lcm_grep`, `lcm_expand`, `lcm_describe`, `lcm_expand_query`, `wiki_compile`, `wiki_query`, `wiki_lint`, `sentinel_status`, `mcp_status`, `lsp` (7 actions), `sessions_list`, `sessions_send`, `sessions_spawn`, plus dynamic MCP tools

### MCP Integration

- Dynamic tool discovery from any MCP server
- Collision-free naming (`mcp__{server}__{tool}`), Stdio/HTTP/SSE transport, config fingerprinting
- Context7 ships ready for up-to-date library documentation

### 92 Builtin Skills

Markdown skill files in `skills/` that teach the agent patterns — from code review to subagent-driven development to git workflows. The SkillCreator auto-generates new skills from successful task traces. The SkillRefiner compares execution traces to skill prescriptions and updates skills when deviations improve outcomes.

### Coding Mode

Point the agent at a coding task and it runs in a sandbox, iterating until the build and tests pass — "iterate-to-green." LSP diagnostics feed type errors back into the same loop, so the model self-corrects against compiler ground truth instead of guessing. The result is a reviewable diff: nothing lands on your branches until you merge it.

### Identity System

- **SOUL.md** — persistent identity loaded into every prompt. Survives `/reset`. Generated from templates at setup — no hardcoded names.
- **AGENTS.md** — agent registry with specializations for subagent spawning
- **ANATOMY.md** — live infrastructure snapshot with Mermaid architecture diagrams. AnatomyScanner detects hardware, GPU VRAM, loaded model + quantization, Tailscale peers, disk usage. Supports named project configurations (daily-driver, multi-model, A/B eval) with VRAM requirement checks. Updated at startup and periodically via heartbeat. The agent queries its own infrastructure via the anatomy tool (scan, status, projects, switch, diagram, history).
- **MEMORY.md + USER.md** — the agent learns who you are over time (bounded: 12K + 8K chars)
- **Agent Profiles** — switch between `full`, `coder`, `research`, `assistant`, `minimal` via `/profile` to optimize your context budget

### Security

- 4-level trust model (BLOCKED → APPROVE → AUTO → AUTONOMOUS)
- 33+ always-blocked patterns, workspace boundary enforcement, bash intent analysis
- Env var config overrides — secrets via `PROMETHEUS_TELEGRAM_TOKEN` or secret files
- Audit logging (SQLite + JSONL), exfiltration detection, prompt injection defense
- Approval queue — `/approve`, `/deny`, `/pending` via Telegram
- Credential pool rotation with dead key cooldown
- Authenticated control plane — bearer-token auth on the REST API plus first-frame token auth on the WebSocket bridge (`PROMETHEUS_API_TOKEN`)

### Always-On

- Telegram gateway with photo, voice, document (20+ formats), and sticker handling
- Slack gateway (Socket Mode) at Telegram-parity: 23 slash commands, thread-based long replies, channel-scoped permissions, skill/memory/curator notification routing
- Vision support (VisionTool) and voice transcription (Whisper STT)
- Cron scheduler, heartbeat monitoring, systemd service
- 40+ slash commands (Telegram), 23 slash commands (Slack)
- Hook hot reload — modify `prometheus.yaml` at runtime, hooks rebuild automatically via mtime polling

### Migration Tool

- `prometheus migrate --from hermes` or `--from openclaw`
- Auto-detects existing installations (`~/.hermes`, `~/.openclaw`, `~/.clawdbot`)
- Migrates config, identity, memory, skills with conflict resolution (skip/overwrite/rename)
- Dry-run mode and timestamped migration reports

### Parallel Tool Execution

- Read-only tools run concurrently via `asyncio.gather`
- Mutating tools run sequentially — no race conditions
- Security hooks still run on every call

### Durable Background Tasks

- Long-running work is registered as a managed task with durable storage (`tasks.db`) that survives restarts
- Event-driven completion detection emits a signal and notifies you (e.g. via Telegram) the moment a task finishes
- An honesty check catches the agent promising "I'll let you know when it's done" without actually registering a task to back that promise — so async commitments are real, not hallucinated

### Fine-Tuning Flywheel (in progress)

- Successful tool-call traces and adapter repair-pairs are captured, stored, and mined into an exportable dataset (capture → store → miner → export)
- Browse what's been collected with `/pairs`
- This is the data-collection half of a LoRA fine-tuning loop for the local model; the training step itself is still on the roadmap

### Observability

- Tool call telemetry (SQLite) — success rates per model per tool
- Every model call is wrapped in an `LLMCallEnvelope` — per-round token/usage accounting and capture of silent failures that older code paths used to swallow
- Phoenix/OpenTelemetry tracing — env-gated (`PROMETHEUS_TRACING=1`), zero-cost no-ops when off
- Failure classification in evals (model issue vs harness issue vs unclear)
- Trend tracking across evaluation runs

## Quick Start

### Prerequisites

- Python 3.11+
- llama.cpp or Ollama running with any model loaded (or a cloud API key)
- A Telegram bot token (from @BotFather) — optional, CLI works without it

### Install

```bash
git clone https://github.com/OAraLabs/Prometheus.git
cd Prometheus
pip install -e .
python3 -m prometheus --setup
```

The setup wizard generates your personalized identity, detects your hardware, connects to your LLM, and runs a smoke test.

### Run

```bash
# Interactive CLI
python3 -m prometheus

# One-shot query
python3 -m prometheus --once "List all Python files in this directory"

# Daemon mode (Telegram + Slack + cron + heartbeat + SENTINEL)
python3 -m prometheus daemon

# As a systemd service (Linux)
systemctl --user enable --now prometheus
```

### Multi-Machine Setup

Run the agent on a storage machine, point it at a GPU machine for inference:

```yaml
# config/prometheus.yaml
model:
  provider: "llama_cpp"
  base_url: "http://gpu-machine:8080"
  fallback:
    - provider: "ollama"
      base_url: "http://gpu-machine:11434"
    - provider: "anthropic"
      api_key_env: "ANTHROPIC_API_KEY"
      model: "claude-haiku-4-5-20251001"
```

Connect via Tailscale, WireGuard, or any network. Prometheus talks HTTP — localhost or remote, it doesn't care.

### What About Smaller GPUs?

16GB VRAM runs Gemma 2 9B or Qwen 2.5 14B (Q4 quantized). Set `strictness: STRICT` — the adapter compensates with more validation and retries. No GPU at all? Use a cloud API provider and you still get the full harness: memory, wiki, SENTINEL, security, profiles, all of it.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                        │
│  Telegram │ Slack │ CLI │ Beacon (Web UI)                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                  ALWAYS-ON LAYER                          │
│  Heartbeat │ Cron │ SENTINEL │ Memory Extractor          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 ORCHESTRATION LAYER                       │
│  Agent Loop → Model Adapter → Tool Dispatch              │
│  ┌─────────────────────────────────────────────────┐     │
│  │  MODEL ADAPTER LAYER                             │     │
│  │  Validator │ Formatter │ Enforcer │ Retry │ Telem│     │
│  └─────────────────────────────────────────────────┘     │
│  Model Router │ Divergence Detector │ LSP │ MCP          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│               IDENTITY & KNOWLEDGE LAYER                  │
│  SOUL.md │ AGENTS.md │ ANATOMY.md │ Profiles             │
│  LCM (DAG compression) │ Wiki │ MEMORY.md │ USER.md      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 MODEL PROVIDER LAYER                      │
│  llama.cpp │ Ollama │ OpenAI │ Anthropic │ Gemini │ xAI  │
└─────────────────────────────────────────────────────────┘
```

## Configuration

```yaml
model:
  provider: "llama_cpp"              # or ollama, openai, anthropic, gemini, xai
  base_url: "http://localhost:8080"
  # model auto-detected from llama.cpp on startup

context:
  effective_limit: 24000
  compression_trigger: 0.75

security:
  permission_mode: "default"
  workspace_root: "~/.prometheus/workspace"

gateway:
  telegram_enabled: true
  # token via env: PROMETHEUS_TELEGRAM_TOKEN

sentinel:
  enabled: true
  dream_budget_tokens: 2000

profile:
  active: "full"   # full | coder | research | assistant | minimal
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/status` | Model, uptime, tools, memory stats, SENTINEL state |
| `/help` | List commands and capabilities |
| `/reset` | Clear conversation (identity persists) |
| `/clear` | Clear conversation context |
| `/model` | Current model and provider |
| `/wiki` | Wiki stats — page count, recent updates |
| `/sentinel` | SENTINEL status and last dream results |
| `/benchmark` | Run evaluation suite |
| `/context` | Token budget breakdown with visual progress bar |
| `/skills` | Loaded skills |
| `/profile` | Switch agent profiles |
| `/anatomy` | Infrastructure snapshot |
| `/beacon` | Web dashboard status, start/stop/restart |
| `/steer` | Inject a mid-turn course-correction while the agent is working |
| `/queue` | Queue a message for the agent to pick up after the current task |
| `/tasks` | List managed background tasks and their status |
| `/pairs` | Browse captured repair-pairs / golden traces |
| `/approve` `/deny` `/pending` | Manage approval queue |
| `/claude` `/gpt` `/gemini` `/xai` | Per-session cloud provider override |
| `/local` | Clear override, return to primary local model |
| `/route` | Show this chat's current provider + model |

### Configuring cloud slash commands

`/claude`, `/gpt`, `/gemini`, and `/xai` each route to a configurable provider + model. Edit `config/prometheus.yaml`:

```yaml
slash_commands:
  claude:
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-5    # alias OR dated snapshot — Anthropic resolves either
  gpt:
    provider: openai
    api_key_env: OPENAI_API_KEY
    model: gpt-4o
  gemini:
    provider: gemini
    api_key_env: GEMINI_API_KEY
    model: gemini-2.5-pro
  xai:
    provider: xai
    api_key_env: XAI_API_KEY
    model: grok-3
```

Restart the daemon and grep the journal to verify the wiring:

```bash
systemctl --user restart prometheus.service
journalctl --user -u prometheus.service | grep slash_commands
# INFO  slash_commands.claude  → anthropic / claude-sonnet-4-5
# INFO  slash_commands.gpt     → openai / gpt-4o
# ...
```

Omit a command (or the whole section) to fall back to the conservative defaults baked into the source — the daemon logs a one-time WARN noting the fallback so the source-archaeology problem isn't load-bearing.

## Project Structure

```
prometheus/
├── src/prometheus/
│   ├── engine/          # Agent loop, sessions, streaming
│   ├── adapter/         # Model Adapter Layer (validator, formatter, enforcer, retry, router)
│   ├── providers/       # llama_cpp, ollama, openai_compat, anthropic, stub, registry
│   ├── tools/builtin/   # 43 builtin tools
│   ├── hooks/           # PreToolUse / PostToolUse + hot reload + LSP diagnostics
│   ├── permissions/     # Security gate + audit + exfiltration + approval queue
│   ├── memory/          # LCM engine, wiki compiler, extractor, store
│   ├── context/         # Token budget, 2-tier compression, prompt assembly
│   ├── gateway/         # Telegram, Slack, cron, heartbeat, media cache, archive
│   ├── sentinel/        # Observer, AutoDream, wiki lint, memory consolidation, telemetry digest
│   ├── mcp/             # MCP runtime, transport, adapter
│   ├── lsp/             # Language server client, orchestrator, diagnostics hook
│   ├── evals/           # LLM judge, metrics, failure classifier, trend tracking
│   ├── coordinator/     # Subagent spawning, divergence detection, checkpoint/rollback
│   ├── learning/        # Skill creator, skill refiner, periodic nudge
│   ├── infra/           # AnatomyScanner, project configs
│   ├── cli/             # Identity generation, migration tool
│   ├── skills/          # Skill loader + registry
│   ├── tasks/           # Background task manager (bash + agent tasks)
│   ├── telemetry/       # Tool call tracking + cost tracking
│   ├── tracing/         # Phoenix/OpenTelemetry spans
│   └── config/          # Settings, paths, env var overrides, profiles
├── templates/           # Identity templates (no personal data)
├── skills/              # 92 builtin skill files (.md)
├── tests/               # 1,179+ tests across 53 files
├── docs/                # Architecture, model registry, sprint reports
├── audits/              # Timestamped audit reports (diffable across runs)
├── gym/                 # Eval & fine-tuning gym (repair-pair capture, benchmark tasks)
├── .githooks/           # Pre-commit hook (blocks secrets, Tailscale IPs, host refs)
├── config/
│   └── prometheus.yaml.default   # Reference config (no secrets)
├── scripts/
│   └── daemon.py        # Always-on daemon entry point
└── PROMETHEUS.md        # Agent instructions (like CLAUDE.md)
```

## Benchmarks

```bash
python -m prometheus.benchmarks.runner --model gemma4-26b --tier 1
```

Latest results (Gemma 4 26B, RTX 4090):

```
Tasks: 19  |  OK: 19  |  Errors: 0
Avg latency: 1.4s  |  Total: 27s

Tool Usage      : 97.4%
Task Completion : 100%
No Hallucination: 84.7%
```

All evaluation runs locally — the LLM judge uses constrained decoding on your own hardware.

## Stats

- ~30,000 lines of production Python
- 1,179+ tests across 53 test files
- 43 builtin tools + dynamic MCP tools
- 92 builtin skills (.md instruction files)
- 6+ model providers (local and cloud)
- 21+ eval tasks with local LLM judge
- 8 LCM modules (DAG compression + FTS5 search)

## Roadmap

- [x] Core agent loop with tool dispatch
- [x] Model Adapter Layer (validation, repair, GBNF, retry, telemetry)
- [x] 43 builtin tools + MCP dynamic tools
- [x] Lossless Context Management (DAG compression, FTS5 search)
- [x] Security (4-level trust, audit, exfiltration, approval queue)
- [x] Telegram gateway with media/vision/voice
- [x] Slack gateway with Socket Mode + Telegram-parity command surface (23 commands, thread-based long replies, signal-bus notifications)
- [x] Wiki knowledge system (Karpathy-inspired, Obsidian-compatible)
- [x] SENTINEL proactive layer (observer + AutoDream)
- [x] Model router with fallback chains + divergence detection
- [x] Evaluation framework with local LLM judge
- [x] LSP integration (compiler-grade code intelligence)
- [x] Identity system with template-based setup
- [x] Agent profiles (context budget optimization)
- [x] Infrastructure self-awareness (AnatomyScanner + ANATOMY.md)
- [x] Migration tool (Hermes + OpenClaw)
- [x] Phoenix/OpenTelemetry tracing
- [x] Web UI for setup and monitoring (Beacon dashboard)
- [ ] Fine-tuning flywheel (LoRA on collected traces) — *capture/export pipeline shipped (#30); training loop pending*

## License

MIT

## Credits

Built by [Will Hieber](https://github.com/OAraLabs) / OAra Labs.

Prometheus is an original implementation. Its architecture was informed by studying the design of [OpenHarness](https://github.com/HKUDS/OpenHarness), the [Hermes Agent](https://github.com/NousResearch/hermes-agent), [Lossless-Claw](https://github.com/Martian-Engineering/lossless-claw), and Andrej Karpathy's [LLM Wiki concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f); the Claude Code agent-loop patterns were reconstructed from [Sigrid Jin's](https://github.com/instructkr) clean-room analysis rather than from any original source. No source code from these projects was copied or translated into Prometheus.
