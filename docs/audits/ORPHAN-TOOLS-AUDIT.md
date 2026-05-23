# Orphan Tool Inventory

**Verified against:** `origin/main` at `39072429dc5f36bb1992e9aa88b5c53ef894d9ec`
**Audited:** 2026-05-22
**Method:** Read-only grep + cross-reference. No code changes.

## Background

Sprint 4 A4 found `MemoryTool` was a load-bearing orphan — class existed,
never registered, `MEMORY.md` was 0 bytes for 6 weeks. Audit B3 surfaced two
more candidates (`Anatomy`, `Whisper`). This audit produces the definitive
list of every `BaseTool` subclass in the codebase and classifies each one.

## Method

- **Step 1:** `grep -rn "class.*Tool.*:"` across `src/prometheus/tools/`,
  `src/prometheus/learning/`, `src/prometheus/memory/`, and the rest of
  `src/prometheus/` to enumerate every `BaseTool` subclass.
- **Step 2:** `grep -rn "register_tool\|registry\.register\|create_tool_registry"`
  across `src/prometheus/` and `scripts/` to enumerate every registration
  site.
- **Step 3:** For every class, cross-reference against registration sites and
  classify (REGISTERED / ORPHAN / CONDITIONALLY-REGISTERED / INTENTIONAL-NON-TOOL).
- **Step 4:** For every CONDITIONALLY-REGISTERED entry, inspect the gate. The
  `try/except: pass` blocks in `__main__.py` are the suspects — a failed
  import silently disables a registration with no log or `/health` signal.

`BaseTool` subclasses only; utility classes like `ToolRegistry`,
`ToolResultTruncator`, `ToolCallTelemetry`, `ToolDashboard`,
`ToolCallValidator`, `DynamicToolLoader`, and Pydantic `*Input` models are
excluded — they're not registrable tools.

## Summary

| Category | Count |
|---|---|
| REGISTERED (unconditional) | 28 |
| CONDITIONALLY-REGISTERED (safe — config gate or logged failure) | 3 |
| CONDITIONALLY-REGISTERED (silently failable) | 17 |
| ORPHAN | 2 |
| INTENTIONAL-NON-TOOL | 3 |
| **Total `BaseTool` subclasses inventoried** | **53** |

Total includes `McpToolAdapter` (a base class for dynamically-registered
MCP wrappers — classified as INTENTIONAL-NON-TOOL since it's never
instantiated as a single tool).

## REGISTERED (unconditional)

Registered in the `for tool in [...]:` loop in
`src/prometheus/__main__.py:157-193` and in the unconditional registration
of `ToolSearchTool` at line 204.

| Tool | Site |
|---|---|
| AgentTool | `src/prometheus/__main__.py:185` |
| AskUserTool | `src/prometheus/__main__.py:186` |
| BashTool | `src/prometheus/__main__.py:159` |
| CronCreateTool | `src/prometheus/__main__.py:166` |
| CronDeleteTool | `src/prometheus/__main__.py:167` |
| CronListTool | `src/prometheus/__main__.py:168` |
| DashboardTool | `src/prometheus/__main__.py:182` |
| DownloadFileTool | `src/prometheus/__main__.py:178` |
| FileEditTool | `src/prometheus/__main__.py:162` |
| FileReadTool | `src/prometheus/__main__.py:160` |
| FileWriteTool | `src/prometheus/__main__.py:161` |
| GlobTool | `src/prometheus/__main__.py:164` |
| GrepTool | `src/prometheus/__main__.py:163` |
| LCMDescribeTool | `src/prometheus/__main__.py:170` |
| LCMExpandTool | `src/prometheus/__main__.py:171` |
| LCMExpandQueryTool | `src/prometheus/__main__.py:173` |
| LCMGrepTool | `src/prometheus/__main__.py:172` |
| MessageTool | `src/prometheus/__main__.py:179` |
| NotebookEditTool | `src/prometheus/__main__.py:183` |
| SentinelStatusTool | `src/prometheus/__main__.py:191` |
| ToolSearchTool | `src/prometheus/__main__.py:204` |
| TTSTool | `src/prometheus/__main__.py:180` |
| WebFetchTool | `src/prometheus/__main__.py:176` |
| WebSearchTool | `src/prometheus/__main__.py:175` |
| WikiCompileTool | `src/prometheus/__main__.py:188` |
| WikiLintTool | `src/prometheus/__main__.py:190` |
| WikiQueryTool | `src/prometheus/__main__.py:189` |
| YouTubeTranscriptTool | `src/prometheus/__main__.py:177` |

## CONDITIONALLY-REGISTERED (safe)

These have explicit gates with surfaced failures (logged warnings or
guard-clauses on real preconditions).

| Tool | Site | Gate | Why safe |
|---|---|---|---|
| AuditQueryTool | `src/prometheus/__main__.py:207-208` | `if security_gate and hasattr(security_gate, '_audit') and security_gate._audit` | Guard-clause, not silenced exception. Skipped only when audit logger is genuinely absent. |
| LSPTool | `scripts/daemon.py:248-269` | `try/except logger.warning("LSP not available: %s", exc)` | Failure is logged loudly. |
| McpStatusTool | `src/prometheus/__main__.py:382-397` | `try/except log.warning("MCP runtime not available: %s", exc)` | Failure is logged loudly. |

## CONDITIONALLY-REGISTERED (silently failable)

These are wrapped in `try/except: pass` blocks. A failed import or
constructor crashes silently, the tool never registers, no log line is
emitted, and the agent loses the capability with no diagnostic. Flagged as
drive-by findings in Sprint 4 PR #4.

| Tool | Site | Block |
|---|---|---|
| GitHubSearchTool | `src/prometheus/__main__.py:221` | SYMBIOTE block (213-227) |
| SymbioteScoutTool | `src/prometheus/__main__.py:222` | SYMBIOTE block (213-227) |
| SymbioteHarvestTool | `src/prometheus/__main__.py:223` | SYMBIOTE block (213-227) |
| SymbioteGraftTool | `src/prometheus/__main__.py:224` | SYMBIOTE block (213-227) |
| SymbioteStatusTool | `src/prometheus/__main__.py:225` | SYMBIOTE block (213-227) |
| SkillTool | `src/prometheus/__main__.py:232` | Optional-tools (230-234) |
| TodoWriteTool | `src/prometheus/__main__.py:237` | Optional-tools (235-239) |
| BrowserTool | `src/prometheus/__main__.py:244` | Optional-tools (242-246) |
| SessionsListTool | `src/prometheus/__main__.py:253` | Session block (249-257) |
| SessionsSendTool | `src/prometheus/__main__.py:254` | Session block (249-257) |
| SessionsSpawnTool | `src/prometheus/__main__.py:255` | Session block (249-257) |
| TaskCreateTool | `src/prometheus/__main__.py:267` | Task block (260-274) |
| TaskGetTool | `src/prometheus/__main__.py:268` | Task block (260-274) |
| TaskListTool | `src/prometheus/__main__.py:269` | Task block (260-274) |
| TaskUpdateTool | `src/prometheus/__main__.py:270` | Task block (260-274) |
| TaskStopTool | `src/prometheus/__main__.py:271` | Task block (260-274) |
| TaskOutputTool | `src/prometheus/__main__.py:272` | Task block (260-274) |

Each block fails as a unit. If any import inside a block raises, every tool
in that block disappears together with no telemetry.

Note also: `ToolSearchTool`'s skill-registry hookup at
`src/prometheus/__main__.py:199-203` is `try/except: pass`. The
`ToolSearchTool` itself is unconditionally registered (line 204), but if
`load_skill_registry()` raises, the `ToolSearchTool` is silently downgraded
to deferred-tool-only mode (no skill search). Same pattern as the blocks
above. Not counted as a separate tool in the table since the registration
itself isn't gated.

Also conditional but redundant: `scripts/daemon.py:80-87` re-registers
`WikiCompileTool` and `WikiQueryTool` (already registered in
`create_tool_registry`) inside a `try/except: pass`. The redundancy reduces
risk — if it fails, the tools are still registered from the main path —
but the redundant registration itself is dead code.

## ORPHAN (no registration site)

Class fully implemented as `BaseTool` subclass with `execute()`, but no
registration site references it.

### 1. MemoryTool

- **File:** `src/prometheus/memory/hermes_memory_tool.py:259`
- **Purpose:** Agent's write path to `MEMORY.md` and `USER.md`. Wraps
  `FileMemoryStore` with `add`/`update`/`remove`/`list` actions and emits
  `memory_updated` events.
- **Status on `origin/main`:** ORPHAN. Class exists, `daemon.py:691-697`
  wires `set_memory_signal_bus()` but never calls `registry.register(MemoryTool())`.
  This is the exact load-bearing orphan Sprint 4 A4 found.
- **Recommendation:** REGISTER. Sprint 4 PR #4 (open as of audit) is the
  fix — adds `registry.register(MemoryTool())` to `create_tool_registry`.
  Merging PR #4 closes this.

### 2. AnatomyTool

- **File:** `src/prometheus/tools/builtin/anatomy.py:48`
- **Purpose:** Agent's query path to infrastructure state — hardware,
  loaded model, VRAM, services, project configurations, architecture
  diagrams. Actions: `scan`, `status`, `projects`, `switch`, `diagram`,
  `history`.
- **Status:** ORPHAN. Daemon wires `set_anatomy_components(scanner, writer, store)`
  at `scripts/daemon.py:533-554` — substantial setup effort — but
  `AnatomyTool()` is never instantiated or registered. Not exported from
  `src/prometheus/tools/builtin/__init__.py` either, so it's not even
  importable via the conventional path.
- **Recommendation:** REGISTER. Add `registry.register(AnatomyTool())` in
  `scripts/daemon.py` immediately after `set_anatomy_components(...)`
  (around line 550). Optionally add the export to `tools/builtin/__init__.py`
  for symmetry with other tools. The fact that all the wiring exists and
  only registration is missing is the same antipattern as `MemoryTool`.

## INTENTIONAL-NON-TOOL

`BaseTool` subclasses that exist but are invoked as utility objects
(`tool = X(); await tool.execute(...)`), not through the registry/agent loop.

### 1. VisionTool

- **File:** `src/prometheus/tools/builtin/vision.py:56`
- **Purpose:** Describe images via multimodal LLM. Used by the Telegram
  gateway and WebSocket server to convert incoming image attachments to
  text for the agent's context.
- **Usage sites:**
  - `src/prometheus/gateway/telegram.py:2645-2652` — `_describe_image()`
    instantiates and calls `.execute()` directly.
  - `src/prometheus/web/ws_server.py:226-227` — same pattern.
- **Recommendation:** DOCUMENT-AS-UTILITY. The current pattern (gateway
  calls `VisionTool().execute()` to preprocess inbound images) is
  reasonable — the agent itself doesn't typically hold raw image paths.
  Consider also registering it for direct agent use if there's a workflow
  where the agent needs to describe an image it produced. Otherwise, leave
  as utility and add a one-line module docstring noting the intent.

### 2. WhisperSTTTool

- **File:** `src/prometheus/tools/builtin/whisper_stt.py:29`
- **Purpose:** Transcribe audio via Whisper. Used by the Telegram gateway
  to convert incoming voice messages to text for the agent's context.
- **Usage sites:**
  - `src/prometheus/gateway/telegram.py:2662-2669` — `_transcribe_audio()`
    instantiates and calls `.execute()` directly.
- **Recommendation:** DOCUMENT-AS-UTILITY. Same reasoning as `VisionTool`.

### 3. McpToolAdapter

- **File:** `src/prometheus/mcp/adapter.py:32`
- **Purpose:** Generic `BaseTool` wrapper that forwards calls to an MCP
  server-side tool. Instantiated per-MCP-tool by `register_mcp_tools()`
  (`src/prometheus/mcp/adapter.py:99-114`).
- **Recommendation:** Leave as-is. Not orphan — it's a base/wrapper class
  consumed dynamically. There's exactly one `registry.register(adapter)`
  call (line 114) that handles every MCP-adapted tool.

## Recommended Phase 2 actions

Ranked by ratio of (capability restored) ÷ (lines of code).

### 1. Register `MemoryTool` — single line

Add `registry.register(MemoryTool())` in `create_tool_registry`. Sprint 4
PR #4 already does this. Just rebase and merge.

### 2. Register `AnatomyTool` — single line in daemon

Add `registry.register(AnatomyTool())` in `scripts/daemon.py` immediately
after `set_anatomy_components(...)` around line 550. Optional one-line
export addition to `tools/builtin/__init__.py`.

### 3. Convert the 7 silently-failable `try/except: pass` blocks to logged failures

Replace each `except: pass` with `except Exception as exc: log.warning("X tool block failed: %s", exc)`.
Order by load-bearing impact:

| Block | Tools at risk | Why high impact |
|---|---|---|
| Task block (`__main__.py:260-274`) | 6 task tools | `/loop`, `/task`, agent's work-tracking primitives. Single import failure disables all 6. |
| SYMBIOTE block (`__main__.py:213-227`) | 5 tools + GitHubSearchTool | Entire research/graft pipeline. |
| Session block (`__main__.py:249-257`) | 3 session tools | Multi-session orchestration. |
| Optional-tools individual blocks (`__main__.py:230-246`) | SkillTool, TodoWriteTool, BrowserTool | Each is one tool, one block. SkillTool block is highest impact: gateway to `/skills`. |
| ToolSearchTool skill-registry attach (`__main__.py:199-203`) | partial degrade of skill discovery | Less critical (the tool itself still registers) but worth logging. |
| Daemon wiki re-register (`daemon.py:80-87`) | redundant — already in main path | Low risk. Can be deleted entirely. |

### 4. Surface registration health to `/health` endpoint

Once failures are logged (action 3), the next step is to surface them
proactively. Add a registration-health field to `/health` listing any
expected-but-missing tools so soak monitoring catches it. A reasonable
shape: for each `try/except` block, record a tag (`symbiote`, `task`,
`session`, etc.) and whether registration succeeded.

### 5. Consider whether `VisionTool`/`WhisperSTTTool` should also be agent-callable

Design question, not a clear bug. Today they only preprocess inbound media
from gateways. If there's a workflow where the agent should be able to
invoke them directly (e.g., "describe this screenshot I just captured"),
register them. Otherwise leave as utility and add a module-level docstring
explaining the intent so the next auditor doesn't re-flag them.

### 6. Delete redundant daemon wiki re-registration

`scripts/daemon.py:80-87` re-registers `WikiCompileTool` and `WikiQueryTool`
which are already unconditionally registered by `create_tool_registry`.
Dead code. Safe to delete.

## Provenance

- Tools inventoried: every `class .*Tool.*:` match in `src/prometheus/` that
  subclasses `BaseTool`.
- Registration sites: every `register_tool`, `registry.register`, and
  `create_tool_registry` match in `src/prometheus/` and `scripts/`.
- All file:line citations verified against `origin/main` at
  `39072429dc5f36bb1992e9aa88b5c53ef894d9ec`.
- No code changes made in the course of this audit.
