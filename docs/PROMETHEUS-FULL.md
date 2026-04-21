# PROMETHEUS.md — Agent Instructions

This is Prometheus, a sovereign AI agent harness. It runs local LLMs (llama.cpp, Ollama)
via an abstract Model Adapter Layer — no Anthropic API dependency in the agent loop.

## Architecture

```
prometheus/
  engine/       AgentLoop — the main turn loop (Sprint 1) ✓
  adapter/      Model Adapter Layer + Model Router (Sprint 3, 10)
  tools/        Tool registry + builtin tools (Sprint 2)
  hooks/        PreToolUse / PostToolUse hooks (Sprint 2)
  permissions/  Security gate + audit + exfiltration detection (Sprint 4, 11)
  context/      Context management + compression (Sprint 4)
  providers/    ModelProvider ABC + StubProvider (Sprint 1) ✓
  gateway/      Telegram / messaging interface + media handling (Sprint 6, 15b)
  learning/     Learning loop + skill creation (Sprint 7) ✓
  tasks/        Task persistence (Sprint 5)
  memory/       LCM + persistent memory (Sprint 5)
  skills/       Skill loading from .md files (Sprint 5)
  coordinator/  Multi-agent coordination + divergence detection (Sprint 8, 10) ✓
  benchmarks/   Benchmark suite + runner (Sprint 8) ✓
  telemetry/    Tool call tracking (Sprint 3, 15)
  config/       Settings + path management + env var overrides (Sprint 11) ✓
  mcp/          MCP integration — tool servers + Context7 (Sprint 12) ✓
  evals/        DeepEval evaluation suite + G-Eval metrics (Sprint 13) ✓
  tracing/      Phoenix/OpenTelemetry trace visualization (Sprint 13) ✓
  infra/        AnatomyScanner + Doctor diagnostics (Sprint 18, 24) ✓
```

## Key Conventions

- All extracted donor code has a provenance header: Source, Original path, License, Modified
- Imports use `from prometheus.` not `from openharness.` or `from hermes.`
- Config is loaded from `config/prometheus.yaml` via `prometheus.config`
- Paths resolve through `prometheus.config.paths` (adapted from OpenHarness)
- Python 3.11+, managed with `uv`

---

## Sprint 0: Skeleton ✓

### `prometheus.config.paths`
`src/prometheus/config/paths.py` — adapted from OpenHarness (MIT)

```python
get_config_dir() -> Path          # ~/.prometheus/ (or $PROMETHEUS_CONFIG_DIR)
get_data_dir() -> Path            # ~/.prometheus/data/
get_logs_dir() -> Path            # ~/.prometheus/logs/
get_sessions_dir() -> Path        # ~/.prometheus/data/sessions/
get_tasks_dir() -> Path           # ~/.prometheus/data/tasks/
get_cron_registry_path() -> Path  # ~/.prometheus/data/cron_jobs.json
get_project_config_dir(cwd) -> Path  # <cwd>/.prometheus/
get_workspace_dir() -> Path       # ~/.prometheus/workspace/
```

### `prometheus.config.defaults`
`src/prometheus/config/defaults.py` — constants: `DEFAULT_MODEL_PROVIDER`, `DEFAULT_CONTEXT_LIMIT`, etc.

### `config/prometheus.yaml`
Root config file. Keys: `system`, `model`, `context`, `security`, `infrastructure`, `gateway`, `learning`, `model_router`, `divergence`, `sentinel`.

### CLI data management

```
python -m prometheus --reset-telemetry   # Delete telemetry.db (with y/N confirmation)
python -m prometheus --reset-data        # Delete all user data (with y/N confirmation)
```

`--reset-data` deletes: `telemetry.db`, `memory.db`, `data/lcm.db`, `data/security/audit.db`,
`eval_results/`, `wiki/`, `sentinel/`, `skills/auto/`. Preserves config files.

### Querying telemetry

```bash
sqlite3 ~/.prometheus/telemetry.db "SELECT tool_name, success, latency_ms, error_type FROM tool_calls ORDER BY timestamp DESC LIMIT 20;"
sqlite3 ~/.prometheus/telemetry.db "SELECT tool_name, COUNT(*), AVG(latency_ms), SUM(success)*1.0/COUNT(*) AS rate FROM tool_calls GROUP BY tool_name;"
```

---

## Sprint 1: Agent Loop ✓

### `prometheus.engine.messages`
`src/prometheus/engine/messages.py` — adapted from OpenHarness (MIT)

```python
class TextBlock(BaseModel):         # type="text", text: str
class ToolUseBlock(BaseModel):      # type="tool_use", id, name, input: dict
class ToolResultBlock(BaseModel):   # type="tool_result", tool_use_id, content, is_error
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]
    @classmethod from_user_text(text) -> ConversationMessage
    @property text -> str           # concatenated TextBlocks
    @property tool_uses -> list[ToolUseBlock]
    def to_api_param() -> dict      # OpenAI wire format
```

### `prometheus.engine.usage`
`src/prometheus/engine/usage.py`

```python
class UsageSnapshot(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    @property total_tokens -> int
```

### `prometheus.engine.cost_tracker`
`src/prometheus/engine/cost_tracker.py`

```python
class CostTracker:
    def add(usage: UsageSnapshot) -> None
    @property total -> UsageSnapshot
```

### `prometheus.engine.stream_events`
`src/prometheus/engine/stream_events.py`

```python
@dataclass AssistantTextDelta:      # text: str
@dataclass AssistantTurnComplete:   # message: ConversationMessage, usage: UsageSnapshot
@dataclass ToolExecutionStarted:    # tool_name: str, tool_input: dict
@dataclass ToolExecutionCompleted:  # tool_name: str, output: str, is_error: bool
StreamEvent = AssistantTextDelta | AssistantTurnComplete | ToolExecutionStarted | ToolExecutionCompleted
```

### `prometheus.engine.agent_loop`
`src/prometheus/engine/agent_loop.py` — adapted from OpenHarness `query.py` (MIT)

```python
@dataclass RunResult:
    text: str
    messages: list[ConversationMessage]
    usage: UsageSnapshot
    turns: int

@dataclass LoopContext:
    provider: ModelProvider         # required
    model: str
    system_prompt: str
    max_tokens: int
    tool_registry: object | None    # wired in Sprint 2
    permission_checker: object | None  # wired in Sprint 4
    hook_executor: object | None    # wired in Sprint 2
    cwd: Path
    max_turns: int = 200

async def run_loop(context: LoopContext, messages: list[ConversationMessage])
    -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]

class AgentLoop:
    def __init__(provider, model="qwen3.5-32b", max_tokens=4096, max_turns=200,
                 tool_registry=None, hook_executor=None, permission_checker=None, cwd=None)
    async def run_async(system_prompt, user_message, tools=None) -> RunResult
    def run(system_prompt, user_message, tools=None) -> RunResult  # asyncio.run() wrapper
```

### `prometheus.providers.base`
`src/prometheus/providers/base.py` — replaces OpenHarness `SupportsStreamingMessages` Protocol

```python
@dataclass ApiMessageRequest:
    model: str
    messages: list[ConversationMessage]
    system_prompt: str | None
    max_tokens: int = 4096
    tools: list[dict] = []

@dataclass ApiTextDeltaEvent:       # text: str
@dataclass ApiMessageCompleteEvent: # message, usage, stop_reason

class ModelProvider(ABC):
    @abstractmethod
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

### `prometheus.providers.stub`
`src/prometheus/providers/stub.py` — new code, implements `ModelProvider`

```python
class StubProvider(ModelProvider):
    def __init__(base_url="http://localhost:8080", timeout=120.0)
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
    # Connects to /v1/chat/completions (OpenAI-compatible)
    # Handles SSE streaming, text + tool_calls, retry on 429/500/502/503
```

### Wiring

```python
from prometheus.engine import AgentLoop
from prometheus.providers.stub import StubProvider

provider = StubProvider(base_url="http://localhost:8080")
loop = AgentLoop(provider=provider)
result = loop.run(system_prompt="You are helpful.", user_message="Hello")
print(result.text)
```

---

## Sprint 2: Tools + Hooks ✓

### `prometheus.tools.base`
`src/prometheus/tools/base.py` — adapted from OpenHarness (MIT)

```python
@dataclass ToolExecutionContext:
    cwd: Path
    metadata: dict[str, Any]          # carries tool_registry, ask_user_prompt, etc.

@dataclass(frozen=True) ToolResult:
    output: str
    is_error: bool = False
    metadata: dict[str, Any]

class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]
    async def execute(arguments, context) -> ToolResult
    def is_read_only(arguments) -> bool
    def to_api_schema() -> dict        # Anthropic format: {name, description, input_schema}
    def to_openai_schema() -> dict     # OpenAI format: {type: "function", function: {...}}

class ToolRegistry:
    def register(tool: BaseTool) -> None
    def get(name: str) -> BaseTool | None
    def list_tools() -> list[BaseTool]
    def to_api_schema() -> list[dict]          # Anthropic format
    def list_schemas() -> list[dict]           # alias for to_api_schema()
    def to_openai_schemas() -> list[dict]      # OpenAI function-calling format
    def list_schemas_for_task(task_description: str) -> list[dict]
    # keyword-matches task against tool name+description; falls back to all schemas
```

### `prometheus.tools.builtin`
`src/prometheus/tools/builtin/` — adapted from OpenHarness (MIT)

```python
# All import from prometheus.tools.builtin

BashTool(workspace=None, max_output=10_000)
    # name="bash" — runs /bin/bash -lc <command>
    # workspace locking: raises error if cwd resolves outside workspace root
    # configurable timeout (default 30s, max 600s)
    # output truncation at max_output chars

FileReadTool()
    # name="read_file" — reads text files with line numbers
    # supports offset + limit for windowed reads (default 200 lines)
    # rejects binary files (null bytes)

FileWriteTool()
    # name="write_file" — creates or overwrites files
    # auto-creates parent directories

FileEditTool()
    # name="edit_file" — str_replace semantics (old_str → new_str)
    # replace_all=True for global replacement

GrepTool()
    # name="grep" — pure-Python regex search across file tree
    # file_glob filter, case_sensitive flag, limit (default 200 matches)

GlobTool()
    # name="glob" — glob pattern file listing
    # optional root override, limit (default 200 paths)
```

### `prometheus.hooks`
`src/prometheus/hooks/` — adapted from OpenHarness (MIT)

```python
# events.py
class HookEvent(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"

# schemas.py — Pydantic hook definitions
CommandHookDefinition(command, timeout_seconds=30, matcher=None, block_on_failure=False)
HttpHookDefinition(url, headers={}, timeout_seconds=30, matcher=None, block_on_failure=False)
PromptHookDefinition(prompt, model=None, timeout_seconds=30, matcher=None, block_on_failure=True)
AgentHookDefinition(prompt, model=None, timeout_seconds=60, matcher=None, block_on_failure=True)
# matcher: fnmatch against tool_name — None matches all tools

# types.py
@dataclass(frozen=True) HookResult:
    hook_type: str; success: bool; output: str; blocked: bool; reason: str; metadata: dict

@dataclass(frozen=True) AggregatedHookResult:
    results: list[HookResult]
    @property blocked -> bool     # True if any result has blocked=True
    @property reason -> str       # first blocking reason

# registry.py — in-memory registry (loader.py deferred to Sprint 5)
class HookRegistry:
    def add(event: HookEvent, hook: HookDefinition) -> None
    def get(event: HookEvent) -> list[HookDefinition]
    def clear(event: HookEvent | None = None) -> None

# executor.py
@dataclass HookExecutionContext:
    cwd: Path
    provider: ModelProvider       # used by prompt/agent hooks
    default_model: str

class HookExecutor:
    def __init__(registry: HookRegistry, context: HookExecutionContext)
    async def execute(event: HookEvent, payload: dict) -> AggregatedHookResult
    def update_registry(registry: HookRegistry) -> None
    # Runs all hooks registered for event whose matcher matches payload["tool_name"]
    # Command hooks: /bin/bash -lc; sets PROMETHEUS_HOOK_EVENT + PROMETHEUS_HOOK_PAYLOAD env vars
    # HTTP hooks: POST {event, payload} JSON
    # Prompt/agent hooks: calls provider.stream_message(), parses {"ok": bool} JSON response
```

### Wiring into AgentLoop

`agent_loop._execute_tool_call()` already wires both hooks:

```python
# Before tool dispatch:
pre = await hook_executor.execute(HookEvent.PRE_TOOL_USE, {tool_name, tool_input, event})
if pre.blocked:
    return ToolResultBlock(is_error=True, content=pre.reason)

# Dispatch tool via registry.get(tool_name).execute(parsed_input, ToolExecutionContext(...))

# After tool dispatch:
await hook_executor.execute(HookEvent.POST_TOOL_USE, {tool_name, tool_input, tool_output, ...})
```

Full wiring example:

```python
from prometheus.engine import AgentLoop
from prometheus.providers.stub import StubProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool
from prometheus.hooks import HookExecutor, HookExecutionContext, HookRegistry, HookEvent
from prometheus.hooks.schemas import CommandHookDefinition

registry = ToolRegistry()
registry.register(BashTool(workspace="/tmp/prometheus-test"))
registry.register(FileReadTool())
registry.register(FileWriteTool())

hook_reg = HookRegistry()
hook_reg.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="echo pre-hook ok"))

provider = StubProvider(base_url="http://localhost:8080")
hook_executor = HookExecutor(
    hook_reg,
    HookExecutionContext(cwd=Path.cwd(), provider=provider, default_model="qwen3.5-32b"),
)

loop = AgentLoop(provider=provider, tool_registry=registry, hook_executor=hook_executor)
result = loop.run(
    system_prompt="You are a coding assistant with access to tools.",
    user_message='Create hello.py with print("hello world"), then run it.',
    tools=registry.list_schemas(),
)
print(result.text)
```

---

## Sprint 3: Model Adapter Layer ✓

### `prometheus.adapter` — ModelAdapter
`src/prometheus/adapter/__init__.py` — novel code

```python
class ModelAdapter:
    def __init__(formatter=None, strictness="NONE", max_retries=3)
    # formatter  defaults to AnthropicFormatter (passthrough)
    # strictness "NONE" | "MEDIUM" | "STRICT" — controls validation depth

    def format_request(system_prompt, tools) -> (str, list[dict])
    # Calls formatter.format_system_prompt() + formatter.format_tools()
    # Returns (formatted_system_prompt, formatted_tools)

    def validate_and_repair(tool_name, tool_input, tool_registry) -> (str, dict, list[str])
    # Returns (final_name, final_input, repairs_made) or raises ValueError

    def extract_tool_calls(text, tool_registry=None) -> list[ToolUseBlock]
    # Delegates to StructuredOutputEnforcer

    def handle_retry(tool_name, error, tool_registry) -> (RetryAction, str)
    # Delegates to RetryEngine
```

### `prometheus.adapter.validator` — ToolCallValidator
`src/prometheus/adapter/validator.py` — novel code

```python
class Strictness(str, Enum):
    NONE = "NONE"    # skip all validation (Claude API)
    MEDIUM = "MEDIUM"  # validate + auto-repair (Qwen, Mistral)
    STRICT = "STRICT"  # validate + repair + coerce aggressively

@dataclass ValidationResult:
    valid: bool; error: str; error_type: str
    # error_type: unknown_tool | invalid_json | missing_param | wrong_type | extra_param

@dataclass RepairResult:
    repaired: bool; tool_name: str; tool_input: dict
    repairs_made: list[str]; error: str

class ToolCallValidator:
    def __init__(strictness: Strictness | str = Strictness.NONE)
    def validate(tool_name, tool_input, tool_registry) -> ValidationResult
    def repair(tool_name, tool_input, error, tool_registry) -> RepairResult
    # Repair strategies: fuzzy Levenshtein name match, JSON extraction from markdown,
    # type coercion (str "5" → int 5), strip unknown params
```

### `prometheus.adapter.formatter` — Model-specific prompt formatting
`src/prometheus/adapter/formatter.py` — novel code

```python
class ModelPromptFormatter(ABC):
    def format_tools(tools: list[dict]) -> list[dict]
    def format_system_prompt(base_prompt, tools, context=None) -> str
    def parse_tool_calls(raw_response: str) -> list[ToolUseBlock]

class AnthropicFormatter(ModelPromptFormatter)   # passthrough — Anthropic handles natively
class QwenFormatter(ModelPromptFormatter)        # OpenAI format + explicit examples in system prompt
class GemmaFormatter(ModelPromptFormatter)       # <tool_call>...</tool_call> native format
```

### `prometheus.adapter.retry` — RetryEngine
`src/prometheus/adapter/retry.py` — novel code

```python
class RetryAction(str, Enum):
    RETRY = "RETRY"   # send retry prompt to model
    ABORT = "ABORT"   # max retries exceeded

class RetryEngine:
    def __init__(max_retries: int = 3)
    def handle_failure(tool_name, error, tool_registry, session_key=None) -> (RetryAction, str)
    def build_retry_prompt(tool_name, error, tool_registry) -> str
    def retry_count(session_key: str) -> int
    def reset(session_key: str | None = None) -> None
```

### `prometheus.adapter.enforcer` — StructuredOutputEnforcer
`src/prometheus/adapter/enforcer.py` — novel code

```python
class StructuredOutputEnforcer:
    def extract_tool_calls(raw_response, tool_registry=None) -> list[ToolUseBlock]
    # Handles: clean JSON, JSON in ```json...``` blocks, mixed prose+JSON,
    # multiple calls, partial/truncated JSON (best-effort repair)
    # Optionally filters against tool_registry

    def generate_grammar(tool_schemas: list[dict]) -> str
    # Returns GBNF grammar string for llama.cpp constrained decoding
    # Pass to LlamaCppProvider(grammar=...) to constrain model output
```

### `prometheus.telemetry.tracker` — ToolCallTelemetry
`src/prometheus/telemetry/tracker.py` — novel code

```python
class ToolCallTelemetry:
    def __init__(db_path: str | Path = "~/.prometheus/telemetry.db")
    # SQLite storage — auto-creates parent dirs

    def record(model, tool_name, success, retries=0, latency_ms=0.0,
               error_type=None, error_detail=None) -> None

    def report() -> dict
    # Returns {
    #   "models": {"<model>": {"<tool>": {calls, successes, failures,
    #                                      success_rate, avg_retries, avg_latency_ms}}},
    #   "tools":  {"<tool>": {calls, success_rate, avg_retries, avg_latency_ms, error_types}},
    #   "total_calls": int,
    #   "overall_success_rate": float,
    # }

    def close() -> None
```

### Real Provider Implementations

#### `prometheus.providers.llama_cpp` — LlamaCppProvider
`src/prometheus/providers/llama_cpp.py` — novel code

```python
class LlamaCppProvider(ModelProvider):
    def __init__(base_url="http://localhost:8080", timeout=120.0, grammar=None)
    # grammar: optional GBNF grammar string from StructuredOutputEnforcer.generate_grammar()
    # Passes grammar to llama-server via the `grammar` request field
    def set_grammar(grammar: str | None) -> None
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

#### `prometheus.providers.ollama` — OllamaProvider
`src/prometheus/providers/ollama.py` — novel code

```python
class OllamaProvider(ModelProvider):
    def __init__(base_url="http://localhost:11434", timeout=120.0, force_json=False)
    # force_json=True adds format="json" to requests for structured output
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

#### `prometheus.providers.anthropic` — AnthropicProvider
`src/prometheus/providers/anthropic.py` — novel code

```python
class AnthropicProvider(ModelProvider):
    def __init__(api_key=None, model="claude-sonnet-4-6", timeout=120.0,
                 prompt_caching=False, base_url=...)
    # api_key: reads ANTHROPIC_API_KEY env var if not provided
    # prompt_caching=True adds cache_control headers on long system prompts (≥1024 chars)
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

### Updated AgentLoop signature (Sprint 3 additions)

```python
class AgentLoop:
    def __init__(provider, model="qwen3.5-32b", max_tokens=4096, max_turns=200,
                 tool_registry=None, hook_executor=None, permission_checker=None,
                 adapter=None,    # NEW: ModelAdapter
                 telemetry=None,  # NEW: ToolCallTelemetry
                 cwd=None)
```

### LoopContext additions

```python
@dataclass LoopContext:
    ...
    adapter: object | None = None    # ModelAdapter — wired in Sprint 3
    telemetry: object | None = None  # ToolCallTelemetry — wired in Sprint 3
```

### Sprint 3 wiring in the agent loop

```
Before LLM call:
  adapter.format_request(system_prompt, tools)  → formatted_system, formatted_tools

After LLM response:
  if no tool_uses but text contains JSON:
    adapter.extract_tool_calls(text, registry)  → inject as ToolUseBlocks

Before tool execution:
  adapter.validate_and_repair(name, input, registry)  → repaired name + input
  on failure: adapter.handle_retry(...)  → RetryAction.RETRY + retry_prompt (returned to model)

After tool execution:
  telemetry.record(model, tool_name, success, retries, latency_ms, error_type)

Error paths (all record telemetry before returning):
  hook_blocked      — pre-tool hook denies execution
  no_registry       — tool registry not configured
  unknown_tool      — tool name not found in registry
  input_validation  — Pydantic model_validate() failure
  permission_denied — SecurityGate or user denial
  parallel_exception — uncaught exception in asyncio.gather read-only dispatch
```

### Sprint 3 full wiring example

```python
from prometheus.engine import AgentLoop
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool, GrepTool, GlobTool
from prometheus.adapter import ModelAdapter
from prometheus.adapter.formatter import QwenFormatter
from prometheus.telemetry.tracker import ToolCallTelemetry

registry = ToolRegistry()
for tool in [BashTool('/tmp/prom-test'), FileReadTool(), FileWriteTool(), GrepTool(), GlobTool()]:
    registry.register(tool)

adapter = ModelAdapter(formatter=QwenFormatter(), strictness='MEDIUM')
telemetry = ToolCallTelemetry('~/.prometheus/telemetry.db')
provider = LlamaCppProvider(base_url='http://localhost:8080')

loop = AgentLoop(
    provider=provider,
    tool_registry=registry,
    adapter=adapter,
    telemetry=telemetry,
)
result = loop.run(
    system_prompt='You are a coding assistant.',
    user_message='List all Python files in /tmp/prom-test, then create test.py.',
    tools=registry.list_schemas(),
)
print(result.text)
print(telemetry.report())
```

---

## Sprint 4: Security + Context Management ✓

### `prometheus.permissions.modes`
`src/prometheus/permissions/modes.py` — new code

```python
class TrustLevel(IntEnum):
    BLOCKED = 0     # always deny
    APPROVE = 1     # requires user confirmation
    AUTO = 2        # allow automatically
    AUTONOMOUS = 3  # allow (background ops)

class PermissionMode(str, Enum):
    DEFAULT = "default"       # destructive ops require approval
    STRICT = "strict"         # file writes + network require approval
    AUTONOMOUS = "autonomous" # no user confirmations
```

### `prometheus.permissions.checker`
`src/prometheus/permissions/checker.py` — new code

```python
@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool              # False for DENY and APPROVE; True for ALLOW
    requires_confirmation: bool  # True only for APPROVE
    reason: str
    action: str          # "ALLOW" | "DENY" | "APPROVE"
    trust_level: TrustLevel

    @classmethod allow(reason="", level=AUTO) -> PermissionDecision   # allowed=True
    @classmethod approve(reason="") -> PermissionDecision             # allowed=False, requires_confirmation=True
    @classmethod deny(reason) -> PermissionDecision                   # allowed=False

class SecurityGate:
    def __init__(denied_commands=None, denied_paths=None,
                 workspace_root=None, mode=PermissionMode.DEFAULT)
    @classmethod from_config(config_path=None) -> SecurityGate
    # Loads security section from prometheus.yaml

    def evaluate(tool_name, *, is_read_only=False,
                 file_path=None, command=None) -> PermissionDecision
    # Used by agent_loop.py permission_checker slot

    def pre_tool_use(tool_name, tool_input: dict, context: dict) -> PermissionDecision
    # Acceptance-test interface: gate.pre_tool_use('bash', {'command': 'rm -rf /'}, {})
```

Trust rules applied in order:
1. AUTONOMOUS mode → ALLOW all except always-blocked patterns
2. Always-blocked regex (rm -rf /, mkfs, dd to /dev/, fork bomb …) → DENY
3. denied_commands list from config → DENY
4. denied_paths list from config → DENY
5. write_file/edit_file in STRICT mode → APPROVE
6. write_file/edit_file outside workspace_root → APPROVE
7. bash with git push / curl / wget / ssh / pip install → APPROVE
8. Everything else → ALLOW

### `prometheus.permissions.sandbox`
`src/prometheus/permissions/sandbox.py` — new code

```python
class SandboxedExecution:
    def __init__(workspace: str | Path, timeout=30, max_output=10_000,
                 strip_env_keys: list[str] | None = None)
    async def run(command: str, env_override=None) -> ToolResult
    @property workspace -> Path
    # Strips API keys / tokens / secrets from subprocess env automatically
    # Enforces timeout; truncates output at max_output chars
```

### `prometheus.context.token_estimation`
`src/prometheus/context/token_estimation.py` — new code

```python
def estimate_tokens(text: str) -> int
# len(text) // 4 — 4-chars-per-token heuristic; fast, no dependencies
```

### `prometheus.context.budget`
`src/prometheus/context/budget.py` — new code

```python
@dataclass
class TokenBudget:
    effective_limit: int
    reserved_output: int = 2000
    model_overrides: dict[str, int]

    @classmethod from_config(model=None, config_path=None) -> TokenBudget
    # Reads context section from prometheus.yaml; applies model_overrides

    def add(category: str, text: str) -> None   # accumulates by category
    def reset() -> None
    @property used -> int                        # sum across all categories
    def usage_by_category() -> dict[str, int]
    def headroom() -> int                        # effective_limit - reserved_output - used
    def is_approaching_limit(threshold=0.75) -> bool
```

### `prometheus.context.truncation`
`src/prometheus/context/truncation.py` — new code

```python
class ToolResultTruncator:
    def __init__(max_tokens=4000)
    @classmethod from_config(config_path=None) -> ToolResultTruncator
    def truncate(tool_name: str, output: str) -> str
    def __call__(tool_name, output) -> str   # callable interface

    # Strategies:
    # bash      → keep last 100 lines
    # read_file → first 50 + last 50 lines + "[... N lines truncated ...]"
    # grep      → top 20 results
    # default   → hard-truncate at max_tokens*4 chars + "[truncated at N tokens]"
```

### `prometheus.context.compression`
`src/prometheus/context/compression.py` — new code

```python
class ContextCompressor:
    def __init__(budget: TokenBudget, fresh_tail_count=32)
    @classmethod from_config(budget, config_path=None) -> ContextCompressor

    def maybe_compress(messages: list[ConversationMessage]) -> list[ConversationMessage]
    # No-op if budget.is_approaching_limit() is False
    # Prunes ToolResultBlock.content from messages older than fresh_tail_count user turns
    # Replaces pruned content with "[content pruned — context compression]"
    # Full LCM summarization deferred to Sprint 7
```

### `prometheus.context.dynamic_tools`
`src/prometheus/context/dynamic_tools.py` — new code

```python
CORE_TOOLS: frozenset[str]   # {"bash", "read_file", "write_file"}

class DynamicToolLoader:
    def __init__(registry: ToolRegistry)
    def active_schemas(task_description: str | None = None) -> list[dict]
    # None → all tools; otherwise CORE_TOOLS + keyword-matched extras
    # Keywords: grep/search → grep; find/list/files → glob;
    #           edit/modify/replace/patch → edit_file
    def on_demand(tool_name: str) -> dict | None
    # Returns schema if tool registered, else None
    def all_schemas() -> list[dict]
```

### Sprint 4 wiring into AgentLoop

```python
from prometheus.permissions import SecurityGate
from prometheus.context import TokenBudget, ToolResultTruncator, ContextCompressor, DynamicToolLoader

# Permission gate (slots into existing permission_checker= param)
gate = SecurityGate.from_config()   # reads config/prometheus.yaml security section

# Context management
budget = TokenBudget.from_config(model="qwen3.5-32b")
truncator = ToolResultTruncator.from_config()
compressor = ContextCompressor.from_config(budget)

loop = AgentLoop(
    provider=provider,
    tool_registry=registry,
    permission_checker=gate,   # evaluate() called before every tool execution
    # APPROVE decisions (allowed=False, requires_confirmation=True) trigger
    # permission_prompt callback in agent_loop — if no prompt is wired, they block.
)

# Truncation + compression used outside AgentLoop (Sprint 5 will wire fully):
output = truncator.truncate(tool_name, raw_output)
messages = compressor.maybe_compress(messages)
```

---

## Sprint Status

- [x] Sprint 0: Skeleton
- [x] Sprint 1: Agent loop
- [x] Sprint 2: Tools + hooks (extract OpenHarness tools/ + hooks/)
- [x] Sprint 3: Model Adapter Layer (novel code)
- [x] Sprint 4: Security + context management
- [x] Sprint 5: Skills + memory + tasks
- [x] Sprint 6: Gateway (Telegram) + Cron + Daemon
- [ ] Sprint 7: Learning loop + LCM
- [ ] Sprint 8: Multi-agent + benchmarks
- [x] Sprint 9: SENTINEL — Proactive Daemon + AutoDream
- [x] Sprint 10: Model Router + Divergence Detector
- [x] Sprint 11: Security Hardening — env overrides, audit logging, exfiltration detection
- [x] Sprint 12: MCP Integration + Context7
- [x] Sprint 13: DeepEval + Phoenix — G-Eval metrics, trend tracking, nightly cron
- [x] Sprint 14: Constrained Decoding Judge — grammar-forced JSON, failure classifier, A/B tested
- [x] Sprint 15: Telemetry Wiring Fix — daemon pipeline, error-path coverage, data reset CLI
- [x] Sprint 15b: GRAFT Phase 1 — Telegram media/vision/voice, scoped lock, sticker cache
- [x] Sprint 15c: GRAFT Phase 2 — Hook hot reload, compression Tier 2, approval queue, credential pool
- [x] Sprint 16: GRAFT-THREAD — Gateway-agnostic conversation memory
- [x] Sprint 17: BOOTSTRAP — Layer 1 identity files (SOUL.md, AGENTS.md, memory wiring)
- [x] Sprint 18: ANATOMY — Infrastructure self-awareness (hardware, model, VRAM, project configs)
- [x] Sprint 19: PROFILES — Agent profiles for context-efficient tool/bootstrap loading

---

## Sprint 5: Skills + Memory + Tasks

### `prometheus.skills`

**`SkillDefinition`** *(frozen dataclass)* — `src/prometheus/skills/types.py`
```python
SkillDefinition(name: str, description: str, content: str, source: str, path: str | None)
```

**`SkillRegistry`** — `src/prometheus/skills/registry.py`
```python
SkillRegistry()
  .register(skill: SkillDefinition) -> None
  .get(name: str) -> SkillDefinition | None          # case-insensitive fallback
  .list_skills() -> list[SkillDefinition]             # sorted by name
```

**`loader`** — `src/prometheus/skills/loader.py`
```python
get_builtin_skills() -> list[SkillDefinition]         # reads skills/builtin/*.md
load_user_skills() -> list[SkillDefinition]           # reads ~/.prometheus/skills/*.md
load_skill_registry(cwd=None) -> SkillRegistry        # builtin + user merged
```

**Builtin skills** — `src/prometheus/skills/builtin/`: `commit.md`, `debug.md`, `plan.md`

**Wires into**: `SkillTool` (`tools/builtin/skill.py`) exposes registry to agent via `skill` tool name.

---

### `prometheus.memory`

**`MemoryStore`** *(SQLite + FTS5)* — `src/prometheus/memory/store.py`
```python
MemoryStore(db_path: str | Path | None = None)        # default: ~/.prometheus/memory.db
  # Messages
  .add_message(session_id, role, content, *, message_id, compressed) -> str
  .get_messages(session_id, *, since, compressed, limit) -> list[dict]
  # Memories
  .persist_memory(entity_type, entity_name, fact, confidence, *, relationship,
                  source_event_ids, tags, memory_id) -> str   # deduplicates on name+fact
  .search_memories(*, query, entity, entity_type, min_confidence, limit) -> list[dict]
  .get_memory(memory_id: str) -> dict | None
  # Summaries
  .add_summary(summary_text, source_message_ids, *, level, summary_id) -> str
  .get_summaries(*, level, limit) -> list[dict]
  .close() / context manager
```

Tables: `messages` (FTS5), `memories` (FTS5), `summaries`.

**`MemoryPointer`** — `src/prometheus/memory/pointer.py`
```python
MemoryPointer(pointer_path: str | Path | None = None, max_chars: int = 8000)
  .add_pointer(text: str) -> None
  .remove_pointer(text: str) -> bool
  .replace_pointer(old_text, new_text) -> bool
  .get_all() -> list[str]
  .format_for_prompt() -> str                         # "## Memory Pointers\n- ..."
  .clear() -> None
```
Backed by `~/.prometheus/MEMORY.md` with `fcntl` exclusive locking and char-limit pruning.

**`FileMemoryStore`** — `src/prometheus/memory/hermes_memory_tool.py`
```python
FileMemoryStore(path: Path, max_chars: int)
  .add(entry: str) -> str
  .replace(old_text, new_text) -> str
  .remove(text: str) -> str
  .list_entries() -> list[str]
  .format_for_prompt(header: str) -> str

get_memory_store() -> FileMemoryStore       # MEMORY.md, 12 000 char cap
get_user_store() -> FileMemoryStore         # USER.md, 8 000 char cap
format_memory_for_prompt() -> str           # both sections combined
```

**`MemoryTool`** *(BaseTool)* — `src/prometheus/memory/hermes_memory_tool.py`
```
name="memory"  inputs: operation (add|replace|remove|list), target (memory|user),
               entry, old_entry
```

**`MemoryExtractor`** — `src/prometheus/memory/extractor.py`
```python
MemoryExtractor(store: MemoryStore, provider: ModelProvider, *, model, obsidian_writer, batch_size)
  .run_once(session_id=None) -> int                   # returns # memories persisted
  .run_forever(interval=1800, session_id=None) -> None # 30-min background loop
```
Reads `messages` table → batches of 15 → LLM extraction prompt → writes to `memories` + optional Obsidian vault.

**`ObsidianWriter`** — `src/prometheus/memory/extractor.py`
```python
ObsidianWriter(vault_path: str | Path)
  .write_fact(fact: dict) -> None   # appends to vault/Memory/<entity_name>.md
```

---

### `prometheus.tasks`

**`TaskType`** — `"local_bash" | "local_agent" | "remote_agent" | "in_process_teammate"`
**`TaskStatus`** — `"pending" | "running" | "completed" | "failed" | "killed"`

**`TaskRecord`** *(dataclass)* — `src/prometheus/tasks/types.py`
```python
TaskRecord(id, type, status, description, cwd, output_file, command, prompt,
           created_at, started_at, ended_at, return_code, metadata)
```

**`BackgroundTaskManager`** — `src/prometheus/tasks/manager.py`
```python
BackgroundTaskManager()
  # Async create
  await .create_shell_task(*, command, description, cwd, task_type) -> TaskRecord
  await .create_agent_task(*, prompt, description, cwd, task_type, model, api_key, command) -> TaskRecord
  # Read
  .get_task(task_id: str) -> TaskRecord | None
  .list_tasks(*, status: TaskStatus | None) -> list[TaskRecord]
  .read_task_output(task_id, *, max_bytes=12000) -> str
  # Mutate
  .update_task(task_id, *, description, progress, status_note) -> TaskRecord
  await .stop_task(task_id) -> TaskRecord
  await .write_to_task(task_id, data: str) -> None

get_task_manager() -> BackgroundTaskManager   # process-wide singleton
```
Output streamed to `~/.prometheus/data/tasks/<id>.log`. Agent tasks restart on broken pipe.

---

### New tools (`prometheus.tools.builtin`)

| File | Tool name | Input fields |
|------|-----------|-------------|
| `skill.py` | `skill` | `name` |
| `task_create.py` | `task_create` | `type`, `description`, `command`, `prompt`, `model` |
| `task_get.py` | `task_get` | `task_id` |
| `task_list.py` | `task_list` | `status?` |
| `task_update.py` | `task_update` | `task_id`, `description?`, `progress?`, `status_note?` |
| `task_stop.py` | `task_stop` | `task_id` |
| `task_output.py` | `task_output` | `task_id`, `max_bytes?` |
| `todo_write.py` | `todo_write` | `item`, `checked?`, `path?` |

All extend `BaseTool` from `prometheus.tools.base`.

### Wiki Maintenance

Transforms extracted memory facts into a persistent, cross-linked Markdown
wiki at `~/.prometheus/wiki/`. Inspired by the LLM Wiki pattern — knowledge
is compiled once and kept current, not re-derived on every query.

```
~/.prometheus/wiki/
├── index.md            # Auto-generated, organized by category
├── log.md              # Append-only chronological compile log
├── .last_compile_ts    # Watermark for incremental compilation
├── people/             # person entities
├── clients/            # organization entities
├── projects/           # task + tool entities
├── topics/             # concept, place, preference entities
└── queries/            # Auto-filed query results (compounding loop)
```

**`WikiCompiler`** — `src/prometheus/memory/wiki_compiler.py`
```python
WikiCompiler(store: MemoryStore, wiki_root: Path | None = None)  # default: ~/.prometheus/wiki/
  .compile(new_facts: list[dict]) -> None   # thread-safe (threading.Lock)
  .get_watermark() -> float                 # last compile timestamp from .last_compile_ts
  .wiki_root -> Path                        # property
```

Entity type mapping: `person→people/`, `organization→clients/`,
`task|tool→projects/`, `concept|place|preference→topics/`.

Page creation requires 2+ `mention_count` in `MemoryStore`. Pages use YAML
frontmatter (`type`, `first_seen`, `last_updated`, `source_count`) and
`[[wiki-links]]` for cross-references detected by substring match (3+ char
entity names). Regenerates `index.md` and appends to `log.md` after each pass.

**Wiring into `MemoryExtractor`** (Sprint 5):
```python
# extractor.py — new parameter
MemoryExtractor(..., post_extract_callback: Callable[[list[dict]], None] | None = None)
  .run_once(session_id) -> tuple[int, list[dict]]   # was: -> int
  # run_forever() calls post_extract_callback(facts) after each pass
```
`daemon.py` passes `WikiCompiler.compile` as `post_extract_callback`.
`lcm_engine.py:238` updated to unpack the new tuple return.

**`WikiCompileTool`** *(BaseTool)* — `src/prometheus/tools/builtin/wiki_compile.py`
```
name="wiki_compile"  inputs: entity_name? (str | None)
```
Module-level `set_wiki_compiler(compiler, store)` setter (same pattern as
`lcm_grep.py`). Reads `.last_compile_ts` watermark, queries `MemoryStore`
for newer facts, calls `compile()`.

**`WikiQueryTool`** *(BaseTool)* — `src/prometheus/tools/builtin/wiki_query.py`
```
name="wiki_query"  inputs: query (str)
```
Reads `index.md`, scores entries by keyword overlap, reads top-5 pages,
returns concatenated content. **Compounding loop:** files result to
`wiki/queries/` when it spans 2+ pages *and* exceeds 200 chars — prevents
trivial lookups from becoming wiki pages.

Both tools registered in `build_tool_registry()` (`scripts/daemon.py`) and
exported from `prometheus.tools.builtin.__init__`.

---

## Sprint 6: Gateway (Telegram) + Cron + Daemon

### `prometheus.gateway.config`
`src/prometheus/gateway/config.py` — novel code

```python
class Platform(str, Enum):           # TELEGRAM, CLI, API

@dataclass
class PlatformConfig:
    platform: Platform
    token: str = ""
    webhook_url: str | None = None
    allowed_chat_ids: list[int] = []
    proxy_url: str | None = None
    max_message_length: int = 4096
    parse_mode: str = "MarkdownV2"
    connect_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    extra: dict[str, Any] = {}

    @property is_restricted -> bool
    def chat_allowed(chat_id: int) -> bool
```

### `prometheus.gateway.platform_base`
`src/prometheus/gateway/platform_base.py` — novel code (architecture inspired by Hermes)

```python
class MessageType(str, Enum):        # TEXT, COMMAND, CALLBACK, EDITED, PHOTO, DOCUMENT, VOICE

@dataclass
class MessageEvent:
    chat_id: int
    user_id: int
    text: str
    message_id: int
    platform: Platform
    message_type: MessageType = TEXT
    username: str | None = None
    timestamp: datetime = now(utc)
    raw: dict[str, Any] = {}
    def session_key() -> str         # "{platform}:{chat_id}"

@dataclass(frozen=True)
class SendResult:
    success: bool
    message_id: int | None = None
    error: str | None = None

class BasePlatformAdapter(ABC):
    def __init__(config: PlatformConfig)
    @property platform -> Platform
    @property running -> bool
    async def start() -> None
    async def stop() -> None
    async def send(chat_id, text, *, reply_to=None, parse_mode=None) -> SendResult
    async def on_message(event: MessageEvent) -> None
```

### `prometheus.gateway.telegram`
`src/prometheus/gateway/telegram.py` — novel code (architecture inspired by Hermes)

```python
def escape_markdown_v2(text: str) -> str
def chunk_message(text: str, max_length=4096) -> list[str]

class TelegramAdapter(BasePlatformAdapter):
    def __init__(config: PlatformConfig, agent_loop: AgentLoop,
                 tool_registry: ToolRegistry,
                 system_prompt="You are Prometheus, a helpful AI assistant.")
    async def start() -> None           # builds Application, registers handlers, starts polling
    async def stop() -> None            # graceful shutdown
    async def send(chat_id, text, *, reply_to=None, parse_mode=None) -> SendResult
    async def on_message(event: MessageEvent) -> None
    # Internal:
    async def _cmd_start(update, context) -> None       # /start handler
    async def _cmd_clear(update, context) -> None       # /clear handler
    async def _handle_text(update, context) -> None     # text message handler
    async def _dispatch_to_agent(event: MessageEvent) -> None
    # Dispatches to agent_loop.run_async(system_prompt, event.text, tools)
    # Sends result.text back via self.send()
```

### `prometheus.gateway.telegram_network`
`src/prometheus/gateway/telegram_network.py` — novel code

```python
@dataclass
class TelegramNetworkConfig:
    proxy_url: str | None = None
    base_url: str | None = None
    base_file_url: str | None = None
    connect_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0
    def to_bot_kwargs() -> dict
    def to_request_kwargs() -> dict
```

### `prometheus.gateway.cron_service`
`src/prometheus/gateway/cron_service.py` — adapted from OpenHarness (MIT)

```python
def load_cron_jobs() -> list[dict]
def save_cron_jobs(jobs: list[dict]) -> None
def validate_cron_expression(expression: str) -> bool
def next_run_time(expression: str, base: datetime | None) -> datetime
def upsert_cron_job(job: dict) -> None
def delete_cron_job(name: str) -> bool
def get_cron_job(name: str) -> dict | None
def set_job_enabled(name: str, enabled: bool) -> bool
def mark_job_run(name: str, *, success: bool) -> None
# JSON registry at ~/.prometheus/data/cron_jobs.json via get_cron_registry_path()
```

### `prometheus.gateway.cron_scheduler`
`src/prometheus/gateway/cron_scheduler.py` — adapted from OpenHarness (MIT)

```python
TICK_INTERVAL_SECONDS = 30

# History
def append_history(entry: dict) -> None         # JSONL at data/cron_history.jsonl
def load_history(*, limit=50, job_name=None) -> list[dict]

# PID management
def read_pid() -> int | None
def write_pid() -> None
def remove_pid() -> None
def is_scheduler_running() -> bool
def stop_scheduler() -> bool

# Job execution
async def execute_job(job: dict) -> dict        # /bin/bash -lc, 300s timeout
async def run_scheduler_loop(*, once=False) -> None   # main loop, signal-aware
def scheduler_status() -> dict
```

### `prometheus.gateway.heartbeat`
`src/prometheus/gateway/heartbeat.py` — novel code

```python
class Heartbeat:
    def __init__(*, interval=30, gateway: BasePlatformAdapter | None = None,
                 task_manager: BackgroundTaskManager | None = None)
    async def check() -> dict           # cron_jobs_due, gateway_running, tasks_running
    async def run_forever() -> None     # loop until stop()
    def stop() -> None
```

### `prometheus.gateway.archive_writer`
`src/prometheus/gateway/archive_writer.py` — novel code (inspired by OpenClaw)

```python
class ArchiveWriter:
    def __init__(path: str | Path | None = None)   # default: ~/.prometheus/data/archive.jsonl
    def archive_event(event_type: str, data: dict | None = None) -> None
    def read_events(*, event_type=None, limit=100) -> list[dict]
```

### New tools (`prometheus.tools.builtin`)

| File | Tool name | Input fields |
|------|-----------|-------------|
| `cron_create.py` | `cron_create` | `name`, `schedule`, `command`, `cwd?`, `enabled?` |
| `cron_delete.py` | `cron_delete` | `name` |
| `cron_list.py` | `cron_list` | *(none)* — read-only |

All adapted from OpenHarness (MIT), imports changed to `prometheus.*`.

### `scripts/daemon.py` — Main daemon entry point
```python
def load_config(config_path=None) -> dict
def build_tool_registry(workspace=None) -> ToolRegistry
async def run_daemon(args) -> None
def main() -> None

# Wires: LlamaCppProvider → ToolRegistry (builtin + cron tools) → AgentLoop
#        → TelegramAdapter + Heartbeat + CronScheduler + MemoryExtractor
# All run via asyncio.gather with SIGTERM/SIGINT graceful shutdown
```

Usage:
```bash
python scripts/daemon.py --telegram-only --debug
python scripts/daemon.py --config config/prometheus.yaml
```

### `scripts/health_check.sh`
Checks cron scheduler PID, daemon log freshness, cron registry, archive events.

### `scripts/prometheus.service`
Systemd unit file for running daemon as a service.

### Sprint 6 wiring

```python
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool, CronCreateTool, ...
from prometheus.engine import AgentLoop
from prometheus.gateway import TelegramAdapter, PlatformConfig, Platform, Heartbeat
from prometheus.gateway.cron_scheduler import run_scheduler_loop

provider = LlamaCppProvider(base_url="http://localhost:8080")
registry = ToolRegistry()
# register all builtin + cron tools ...
telemetry = ToolCallTelemetry()  # shared with SENTINEL digest
agent_loop = AgentLoop(provider=provider, tool_registry=registry, telemetry=telemetry)

tg_config = PlatformConfig(platform=Platform.TELEGRAM, token="BOT_TOKEN")
telegram = TelegramAdapter(config=tg_config, agent_loop=agent_loop,
                           tool_registry=registry)

await asyncio.gather(
    telegram.start(),
    run_scheduler_loop(),
    Heartbeat(gateway=telegram).run_forever(),
)
```

### Dependencies added
- `python-telegram-bot>=21.0` — Telegram Bot API
- `croniter>=2.0` — cron expression parsing

---

## Sprint 7: Learning Loop + LCM ✓

### File tree additions

```
learning/
  __init__.py         PeriodicNudge, SkillCreator, SkillRefiner
  nudge.py            Self-evaluation injection every N turns
  skill_creator.py    Auto-generate SKILL.md from tool traces
  skill_refiner.py    Refine skills based on execution deviations

memory/
  lcm_types.py        MessagePart, SummaryNode, CompactionConfig, AssemblyResult, CompactionResult, LCMStats
  lcm_fts5.py         sanitize_fts5_query(), tokenize_for_fts5()
  lcm_conversation_store.py  SQLite message store with FTS5
  lcm_summary_store.py       SQLite DAG summary store with FTS5
  lcm_compaction.py    Incremental DAG compaction engine
  lcm_assembler.py     Context assembly from summaries + fresh tail
  lcm_summarize.py     LLM summarization with circuit breaker
  lcm_engine.py        Top-level LCM orchestrator

tools/builtin/
  lcm_grep.py          FTS5 search over messages + summaries
  lcm_expand.py        Expand summary back to source messages
  lcm_describe.py      Inspect summary metadata / LCM stats

context/
  system_prompt.py     Prometheus identity + SYSTEM_PROMPT_DYNAMIC_BOUNDARY
  prompt_assembler.py  Runtime system prompt assembly (static + dynamic)
  prometheusmd.py      PROMETHEUS.md discovery and loading
```

### `prometheus.learning.nudge.PeriodicNudge`
`src/prometheus/learning/nudge.py`

```python
@dataclass
class PeriodicNudge:
    interval: int = 15          # turns between nudges
    prompt: str = _NUDGE_PROMPT # < 200 tokens
    enabled: bool = True

    @classmethod
    def from_config(cls, config_path: str | None = None) -> PeriodicNudge
    def maybe_inject(self, turn_count: int) -> dict | None  # returns nudge msg or None
    def reset(self) -> None
```

### `prometheus.learning.skill_creator.SkillCreator`
`src/prometheus/learning/skill_creator.py`

```python
class SkillCreator:
    def __init__(self, provider: ModelProvider, *, model: str = "default",
                 min_tool_calls: int = 3, auto_dir: Path | None = None) -> None

    @classmethod
    def from_config(cls, provider: ModelProvider, config_path: str | None = None) -> SkillCreator
    async def maybe_create(self, task_description: str,
                           tool_trace: list[dict]) -> Path | None
    # Writes to ~/.prometheus/skills/auto/<slug>.md
```

### `prometheus.learning.skill_refiner.SkillRefiner`
`src/prometheus/learning/skill_refiner.py`

```python
class SkillRefiner:
    def __init__(self, provider: ModelProvider, *, model: str = "default") -> None

    async def maybe_refine(self, skill_path: Path,
                           tool_trace: list[dict], outcome: str) -> bool
    # Creates .bak backup before updating
```

### `prometheus.memory.lcm_types`
`src/prometheus/memory/lcm_types.py`

```python
@dataclass
class MessagePart:
    role: str; content: str; timestamp: float; message_id: str
    session_id: str = ""; turn_index: int = 0; token_count: int = 0

@dataclass
class SummaryNode:
    id: str; parent_ids: list[str]; source_message_ids: list[str]
    summary_text: str; depth: int = 0; token_count: int = 0
    created_at: float; is_leaf: bool = True

@dataclass
class CompactionConfig:
    context_threshold: int = 18_000; fresh_tail_count: int = 32
    summary_model: str = "default"; max_summary_depth: int = 5
    compaction_batch_size: int = 10

@dataclass
class AssemblyResult:
    summaries: list[SummaryNode]; fresh_messages: list[MessagePart]
    total_tokens: int = 0; compression_ratio: float = 1.0

@dataclass
class CompactionResult:
    summaries_created: int = 0; messages_compacted: int = 0
    new_depth: int = 0; tokens_saved: int = 0

@dataclass
class LCMStats:
    total_messages: int = 0; total_summaries: int = 0; max_depth: int = 0
    total_compactions: int = 0; last_compaction_at: float | None = None
```

### `prometheus.memory.lcm_conversation_store.LCMConversationStore`
`src/prometheus/memory/lcm_conversation_store.py` — SQLite + FTS5, WAL mode, shared `lcm.db`

```python
class LCMConversationStore:
    def __init__(self, db_path: Path | None = None) -> None

    def insert_message(self, msg: MessagePart) -> str
    def get_messages(self, session_id: str, *, since_turn: int | None = None,
                     limit: int = 500) -> list[MessagePart]
    def get_fresh_tail(self, session_id: str, count: int) -> list[MessagePart]
    def mark_compacted(self, message_ids: list[str]) -> int
    def search(self, query: str, *, session_id: str | None = None,
               limit: int = 20) -> list[MessagePart]
    def count_uncompacted(self, session_id: str) -> int
    def close(self) -> None
```

### `prometheus.memory.lcm_summary_store.LCMSummaryStore`
`src/prometheus/memory/lcm_summary_store.py` — DAG summary storage, shared `lcm.db`

```python
class LCMSummaryStore:
    def __init__(self, db_path: Path | None = None) -> None

    def insert_summary(self, node: SummaryNode) -> str  # marks parents non-leaf
    def get_by_id(self, summary_id: str) -> SummaryNode | None
    def get_leaves(self, *, max_depth: int | None = None) -> list[SummaryNode]
    def get_by_depth(self, depth: int, *, limit: int = 100) -> list[SummaryNode]
    def get_roots(self) -> list[SummaryNode]
    def get_children(self, parent_id: str) -> list[SummaryNode]
    def get_ancestors(self, node_id: str) -> list[SummaryNode]  # BFS walk up DAG
    def search(self, query: str, *, limit: int = 20) -> list[SummaryNode]
    def get_stats(self) -> dict
    def close(self) -> None
```

### `prometheus.memory.lcm_compaction.LCMCompactor`
`src/prometheus/memory/lcm_compaction.py` — the heart of LCM

```python
class LCMCompactor:
    def __init__(self, conversation_store: LCMConversationStore,
                 summary_store: LCMSummaryStore,
                 summarizer: LCMSummarizer,
                 config: CompactionConfig) -> None

    async def compact(self, session_id: str) -> CompactionResult
    def should_compact(self, session_id: str) -> bool
    # Internal: _summarize_messages, _cascade_summaries (6+ leaves → merge up),
    #           _summarize_nodes
```

### `prometheus.memory.lcm_assembler.LCMAssembler`
`src/prometheus/memory/lcm_assembler.py`

```python
class LCMAssembler:
    def __init__(self, conversation_store: LCMConversationStore,
                 summary_store: LCMSummaryStore,
                 config: CompactionConfig) -> None

    def assemble(self, session_id: str, token_budget: int) -> AssemblyResult
    def format_summary_preamble(self, summaries: list[SummaryNode]) -> str
```

### `prometheus.memory.lcm_summarize.LCMSummarizer`
`src/prometheus/memory/lcm_summarize.py`

```python
class LCMCircuitBreakerOpen(Exception): ...

class LCMSummarizer:
    def __init__(self, provider: ModelProvider, *, model: str = "default",
                 max_retries: int = 2) -> None

    async def summarize_messages(self, messages: list[MessagePart]) -> str
    async def summarize_summaries(self, summaries: list[SummaryNode]) -> str
    def reset(self) -> None
    @property
    def circuit_open(self) -> bool
    # Circuit breaker: 3 consecutive failures → LCMCircuitBreakerOpen
```

### `prometheus.memory.lcm_engine.LCMEngine`
`src/prometheus/memory/lcm_engine.py` — top-level orchestrator

```python
class LCMEngine:
    def __init__(self, provider: ModelProvider, *,
                 config: CompactionConfig | None = None,
                 db_path: Path | None = None) -> None

    async def ingest(self, session_id: str, role: str, content: str,
                     *, turn_index: int = 0) -> str
    def assemble(self, session_id: str, token_budget: int) -> AssemblyResult
    async def compact(self, session_id: str) -> CompactionResult
    async def maybe_compact(self, session_id: str) -> CompactionResult | None
    async def pre_compaction_flush(self, session_id: str) -> None
    def get_stats(self, session_id: str) -> LCMStats
    def close(self) -> None
    # Context manager support: __enter__, __exit__
```

### LCM Agent Tools
`src/prometheus/tools/builtin/lcm_grep.py`, `lcm_expand.py`, `lcm_describe.py`

```python
# lcm_grep — FTS5 search over messages + summaries
class LCMGrepTool(BaseTool):
    name = "lcm_grep"
    # Input: query, session_id?, search_target ("messages"|"summaries"|"both"), limit

# lcm_expand — expand a summary back to originals
class LCMExpandTool(BaseTool):
    name = "lcm_expand"
    # Input: summary_id, depth?

# lcm_describe — inspect summary metadata or overall stats
class LCMDescribeTool(BaseTool):
    name = "lcm_describe"
    # Input: summary_id?, session_id?

# Engine wiring:
set_lcm_engine(engine: LCMEngine) -> None  # module-level setter for tool access
```

### `prometheus.context.system_prompt`
`src/prometheus/context/system_prompt.py`

```python
SYSTEM_PROMPT_DYNAMIC_BOUNDARY: str = "--- SYSTEM_PROMPT_DYNAMIC_BOUNDARY ---"

def build_system_prompt(custom_prompt: str | None = None,
                        env: EnvironmentInfo | None = None,
                        cwd: str | None = None) -> str
```

### `prometheus.context.prompt_assembler`
`src/prometheus/context/prompt_assembler.py`

```python
def build_runtime_system_prompt(
    *, cwd: str, config: dict | None = None,
    memory_content: str = "", skills: list | None = None,
    task_state: str = "", loaded_skill_content: str = "",
) -> str
# Assembles: STATIC (base prompt + env) + DYNAMIC_BOUNDARY + DYNAMIC (PROMETHEUS.md, memory, skills, tasks)
```

### `prometheus.context.prometheusmd`
`src/prometheus/context/prometheusmd.py`

```python
def discover_prometheus_md_files(cwd: str | Path) -> list[Path]
def load_prometheus_md_prompt(cwd: str | Path, *, max_chars_per_file: int = 12000) -> str | None
```

### Sprint 7 wiring

```python
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.learning import PeriodicNudge, SkillCreator, SkillRefiner
from prometheus.tools.builtin.lcm_grep import set_lcm_engine
from prometheus.context.prompt_assembler import build_runtime_system_prompt

provider = LlamaCppProvider(base_url="http://localhost:8080")

# LCM engine — manages conversation persistence + DAG compaction
lcm = LCMEngine(provider=provider)
set_lcm_engine(lcm)  # wire into lcm_grep/expand/describe tools

# Learning loop
nudge = PeriodicNudge.from_config()
skill_creator = SkillCreator(provider)
skill_refiner = SkillRefiner(provider)

# Agent loop integration:
# 1. On each message: await lcm.ingest(session_id, role, content, turn_index=n)
# 2. Before LLM call: result = lcm.assemble(session_id, token_budget=24000)
# 3. Every N turns: nudge_msg = nudge.maybe_inject(turn_count)
# 4. After task: skill_path = await skill_creator.maybe_create(desc, trace)
# 5. After skill-guided task: await skill_refiner.maybe_refine(path, trace, outcome)
# 6. Periodically: await lcm.maybe_compact(session_id)
```

---

## Running

```bash
uv sync
uv run prometheus
# or
./scripts/start.sh

# Daemon mode (Sprint 6):
python scripts/daemon.py --telegram-only --debug

# Benchmarks (Sprint 8):
python -m prometheus.benchmarks.runner --model qwen3.5-32b --tier 1
python -m prometheus.benchmarks.runner --tier 2 --json
python -m prometheus.benchmarks.runner --case t1_bash_echo --verbose
```

---

## Sprint 8: Multi-Agent + Benchmark Suite ✓

### `prometheus.coordinator.agent_definitions` — AgentDefinition
`src/prometheus/coordinator/agent_definitions.py` — adapted from OpenHarness (MIT)

```python
@dataclass
class AgentDefinition:
    def __init__(name, description, system_prompt="", tools=[], model="",
                 read_only=False, max_turns=50, metadata={})

# Built-in agents: general-purpose, explorer, planner, worker, verification

get_all_agent_definitions() -> dict[str, AgentDefinition]
get_agent_definition(name) -> AgentDefinition | None
register_agent_definition(defn: AgentDefinition) -> None
```

### `prometheus.coordinator.coordinator_mode` — TeamRegistry
`src/prometheus/coordinator/coordinator_mode.py` — adapted from OpenHarness (MIT)

```python
@dataclass
class TeamRecord:
    name: str; description: str; agents: list[str]; metadata: dict

class TeamRegistry:
    def create_team(name, description="", agents=None) -> TeamRecord
    def get_team(name) -> TeamRecord | None
    def list_teams() -> list[TeamRecord]
    def add_agent_to_team(team_name, agent_name) -> bool
    def remove_agent_from_team(team_name, agent_name) -> bool

get_team_registry() -> TeamRegistry   # module-level singleton
is_coordinator_mode(agent_count) -> bool
get_coordinator_system_prompt(team=None) -> str
```

### `prometheus.coordinator.subagent` — SubagentSpawner
`src/prometheus/coordinator/subagent.py` — novel code

```python
@dataclass(frozen=True)
class SubagentResult:
    agent_id: str; agent_type: str; text: str; turns: int
    success: bool; error: str | None; metadata: dict

class SubagentSpawner:
    def __init__(provider, *, parent_tool_registry=None, model="qwen3.5-32b",
                 max_tokens=4096, cwd=None, adapter=None, telemetry=None)

    async def spawn(task, *, agent_type="general-purpose", tools_subset=None,
                    model=None, system_prompt=None, max_turns=None) -> SubagentResult
    # Spawns a fresh AgentLoop with isolated messages[]. Result returned
    # without polluting the parent conversation.

    async def spawn_parallel(tasks: list[dict]) -> list[SubagentResult]
    # Runs multiple subagents concurrently via asyncio.gather()
```

**Wiring**: `SubagentSpawner` creates a new `AgentLoop` (Sprint 1) per subagent, optionally filtering the parent `ToolRegistry` (Sprint 2) to a tool subset. Agent definitions resolve via `get_agent_definition()`.

### `prometheus.tools.builtin.agent` — AgentTool
`src/prometheus/tools/builtin/agent.py` — adapted from OpenHarness agent_tool.py (MIT)

```python
class AgentToolInput(BaseModel):
    description: str; prompt: str
    subagent_type: str = "general-purpose"
    model: str | None = None

class AgentTool(BaseTool):
    name = "Agent"
    # Reads SubagentSpawner from context.metadata["subagent_spawner"]
    async def execute(arguments: AgentToolInput, context: ToolExecutionContext) -> ToolResult
```

**Wiring**: Registered in `tools/builtin/__init__.py`. Uses `BaseTool` interface (Sprint 2). Requires `subagent_spawner` key in `ToolExecutionContext.metadata`.

### `prometheus.coordinator.health` — HealthMonitor
`src/prometheus/coordinator/health.py` — novel code

```python
class HealthState(str, Enum): HEALTHY, DEGRADED, CRITICAL

@dataclass
class ComponentHealth:
    name: str; healthy: bool; detail: str; latency_ms: float

@dataclass
class HealthStatus:
    state: HealthState; components: list[ComponentHealth]; timestamp: float
    @property degraded_components -> list[ComponentHealth]
    def summary() -> str

# Individual check functions:
check_llama_cpp(base_url="http://127.0.0.1:8080") -> ComponentHealth
check_sqlite(db_path=None) -> ComponentHealth
check_tailscale() -> ComponentHealth
check_gpu_memory() -> ComponentHealth
check_disk(path="/") -> ComponentHealth

class HealthMonitor:
    def __init__(*, interval=60, llama_url="...", db_path=None, disk_path="/",
                 alert_callback=None, checks=["llama_cpp","sqlite","tailscale","gpu_memory","disk"])

    async def check() -> HealthStatus
    async def run_forever() -> None  # alerts via callback on state transition to degraded
    def stop() -> None
```

**Wiring**: Extends Sprint 6's `Heartbeat` (gateway) with infrastructure checks. `alert_callback` can post to Telegram via the gateway adapter.

### `prometheus.benchmarks.suite` — BenchmarkSuite
`src/prometheus/benchmarks/suite.py` — novel code

```python
class TestTier(IntEnum): TIER_1 = 1, TIER_2 = 2

@dataclass
class TestCase:
    id: str; name: str; tier: int; prompt: str
    expected_tools: list[str]; expected_output_contains: list[str]
    expected_output_not_contains: list[str]; expected_file_exists: list[str]
    expected_file_contains: dict[str, str]; max_turns: int
    setup_commands: list[str]; teardown_commands: list[str]; tags: list[str]

class BenchmarkSuite:
    def filter_tier(tier) -> list[TestCase]
    def filter_tags(tags) -> list[TestCase]
    def get(case_id) -> TestCase | None
    def to_yaml() -> str
    @classmethod from_yaml(yaml_str) -> BenchmarkSuite
    @classmethod from_file(path) -> BenchmarkSuite

load_suite(tier=None) -> BenchmarkSuite
# Built-in: 22 Tier 1 atomic tests + 5 Tier 2 multi-step tests
```

### `prometheus.benchmarks.runner` — BenchmarkRunner
`src/prometheus/benchmarks/runner.py` — novel code

```python
class Score(str, Enum): SUCCESS, PARTIAL, RETRY_SUCCESS, FAIL, CRASH

@dataclass
class ScoreResult:
    case_id: str; case_name: str; score: Score
    turns: int; latency_ms: float; details: str; tool_calls: list[str]

class BenchmarkRunner:
    def __init__(provider, tool_registry, *, model="qwen3.5-32b",
                 max_tokens=4096, cwd=None, adapter=None)

    async def run_case(case: TestCase) -> ScoreResult
    async def run_suite(suite, *, concurrency=1) -> list[ScoreResult]

# CLI: python -m prometheus.benchmarks.runner --model MODEL --tier 1|2
#      --json --verbose --case CASE_ID --concurrency N
```

**Wiring**: `BenchmarkRunner` creates an `AgentLoop` (Sprint 1) per test case with a `ToolRegistry` (Sprint 2). Scoring evaluates tool calls from `ConversationMessage.tool_uses`, output text, and file system assertions.

### Sprint 8 File Tree

```
coordinator/
  __init__.py             — package exports
  agent_definitions.py    — AgentDefinition + builtin agents (adapted from OpenHarness)
  coordinator_mode.py     — TeamRecord, TeamRegistry, coordinator system prompt
  subagent.py             — SubagentSpawner, SubagentResult (novel)
  health.py               — HealthMonitor + 5 check functions (novel)
tools/builtin/
  agent.py                — AgentTool (BaseTool adapter for SubagentSpawner)
benchmarks/
  __init__.py             — package exports
  __main__.py             — CLI entry point
  suite.py                — TestCase, BenchmarkSuite, 27 built-in test cases
  runner.py               — BenchmarkRunner, Score, ScoreResult, CLI
tests/
  test_coordinator.py     — 37 tests (definitions, teams, subagent, health)
  test_benchmarks.py      — 31 tests (suite, scoring, runner, agent tool)
```

---

## Sprint 9 — SENTINEL: Proactive Daemon + AutoDream

Transforms Prometheus from reactive (waits for messages) to proactive (observes, acts, consolidates). Two subsystems: **Activity Observer** watches signals and sends Telegram nudges, **AutoDream Engine** uses idle time for wiki maintenance, memory consolidation, telemetry analysis, and knowledge synthesis.

### Architecture

```
SENTINEL LAYER (signal-driven)
├── SignalBus              — async pub/sub for ActivitySignal events
├── Activity Observer      — pattern detection → Telegram nudges
│   ├── Extraction spike   — high fact count in one pass
│   ├── Error streak       — consecutive tool failures
│   └── Nudge cooldown     — per-type cooldown prevents spam
│
└── AutoDream Engine       — triggered by idle_start signal
    ├── Phase 1: WikiLint             — orphans, broken links, stale pages (no LLM)
    ├── Phase 2: MemoryConsolidation  — dedup, decay, tombstone (no LLM)
    ├── Phase 3: TelemetryDigest      — anomaly detection (no LLM)
    └── Phase 4: KnowledgeSynthesis   — cross-entity insights (LLM, budget-capped)
```

### New Classes

**`sentinel/signals.py`** — Signal bus

```python
@dataclass
class ActivitySignal:
    kind: str               # "idle_start", "idle_end", "extraction_complete", etc.
    timestamp: float
    payload: dict[str, Any]
    source: str

class SignalBus:
    def subscribe(kind: str, callback: SignalCallback) -> None   # "*" for wildcard
    async def emit(signal: ActivitySignal) -> None               # broadcast, exceptions caught
    def recent(kind: str | None, *, limit: int) -> list[ActivitySignal]
```

**`sentinel/observer.py`** — Activity observer

```python
class ActivityObserver:
    def __init__(bus: SignalBus, gateway: BasePlatformAdapter | None, *, config: dict)
    async def start() -> None              # subscribes to "*" on bus
    async def _send_nudge(nudge_type: str, message: str) -> None  # respects cooldown
    # Properties: started, last_activity, pending_nudges, nudge_history
```

**`sentinel/autodream.py`** — AutoDream engine

```python
@dataclass
class DreamResult:
    phase: str
    duration_seconds: float
    summary: dict[str, Any]
    error: str | None

class AutoDreamEngine:
    def __init__(bus: SignalBus, *, wiki_linter, memory_consolidator,
                 telemetry_digest, knowledge_synth, config: dict)
    async def start() -> None              # subscribes to idle_start/idle_end
    async def run_cycle() -> list[DreamResult]  # runs all 4 phases
    # Properties: dreaming, cycle_count, last_cycle_time, last_results
```

**`sentinel/wiki_lint.py`** — Wiki linter

```python
@dataclass
class LintIssue:
    severity: str    # "error" | "warning" | "info"
    category: str    # "orphan" | "broken_link" | "stale" | "duplicate" | "missing_crossref" | "imbalance"
    page: str
    detail: str
    fixable: bool

class WikiLinter:
    def __init__(wiki_root: Path | None)
    def lint() -> LintResult              # scans for 6 issue types
    def auto_fix(result: LintResult) -> int  # fixes safe issues
    def summary(issues: list[LintIssue]) -> str
```

**`sentinel/memory_consolidator.py`** — Memory cleaner

```python
class MemoryConsolidator:
    def __init__(store: MemoryStore, *, decay_rate=0.05, min_confidence=0.1,
                 stale_days=90, similarity_threshold=0.80)
    def consolidate() -> ConsolidationResult  # dedup + decay + tombstone
```

**`sentinel/telemetry_digest.py`** — Telemetry health report

```python
class TelemetryDigest:
    def __init__(telemetry: ToolCallTelemetry, *, period_hours=24, baseline_hours=168)
    def generate() -> DigestResult        # compares current vs baseline
    # Flags: success_rate_drop >5%, retry_increase >10%, latency_spike >50%
```

**`sentinel/knowledge_synth.py`** — LLM-powered insight generation

```python
class KnowledgeSynthesizer:
    def __init__(store: MemoryStore, provider: ModelProvider, *, model, budget_tokens=2000)
    async def synthesize(budget_tokens: int | None) -> list[SynthInsight]
    # Writes to wiki/queries/insight-{date}-{topic}.md
```

### New Tools

**`tools/builtin/sentinel_status.py`** — `sentinel_status` tool

```python
class SentinelStatusTool(BaseTool):
    # Returns: observer state, dream state, signal bus stats, pending nudges
    # Input: SentinelStatusInput(verbose: bool = False)
```

**`tools/builtin/wiki_lint_tool.py`** — `wiki_lint` tool

```python
class WikiLintTool(BaseTool):
    # Triggers wiki lint on demand, returns issue list
    # Input: WikiLintInput(severity: str | None, auto_fix: bool = False)
```

### Modified Files

| File | Change |
|------|--------|
| `gateway/heartbeat.py` | Added `signal_bus` property + setter, idle detection via `_check_idle()`, emits `idle_start`/`idle_end` |
| `memory/extractor.py` | Added `signal_bus` property + setter, emits `extraction_complete` after `run_once()` |
| `memory/store.py` | Added `update_memory(id, **fields)`, `delete_memory(id)`, `get_all_memories(min_confidence, limit)` |
| `telemetry/tracker.py` | Added `since: float | None` param to `report()` for time-windowed queries |
| `tools/builtin/__init__.py` | Exports `SentinelStatusTool`, `WikiLintTool` |
| `scripts/daemon.py` | Creates SignalBus + all SENTINEL components, wires into heartbeat/extractor, registers tools |
| `config/prometheus.yaml` | Added `sentinel:` section with 10 config keys |

### Daemon Wiring (scripts/daemon.py)

SENTINEL is wired after the memory extractor block. Signal bus is passed to heartbeat and extractor via property setters to avoid restructuring existing init order.

```python
signal_bus = SignalBus()
heartbeat.signal_bus = signal_bus       # enables idle detection
extractor.signal_bus = signal_bus       # emits extraction_complete
observer = ActivityObserver(signal_bus, gateway=telegram, config=sentinel_config)
autodream = AutoDreamEngine(signal_bus, wiki_linter=..., ...)
await observer.start()                  # subscribes to "*"
await autodream.start()                 # subscribes to idle_start/idle_end
```

### Config (`prometheus.yaml`)

```yaml
sentinel:
  enabled: true
  idle_threshold_minutes: 15
  nudge_cooldown_minutes: 60
  dream_interval_minutes: 30
  dream_budget_tokens: 2000
  stale_threshold_days: 90
  confidence_decay_rate: 0.05
  digest_lookback_hours: 24
  auto_fix_wiki: true
  synthesis_enabled: true
```

### Sprint 9 File Tree

```
sentinel/
  __init__.py               — package exports (SignalBus, ActivitySignal)
  signals.py                — ActivitySignal dataclass, SignalBus async pub/sub
  observer.py               — ActivityObserver, PendingNudge
  autodream.py              — AutoDreamEngine, DreamResult
  wiki_lint.py              — WikiLinter, LintIssue, LintResult
  memory_consolidator.py    — MemoryConsolidator, ConsolidationResult
  telemetry_digest.py       — TelemetryDigest, DigestAnomaly, DigestResult
  knowledge_synth.py        — KnowledgeSynthesizer, SynthInsight
tools/builtin/
  sentinel_status.py          — SentinelStatusTool + set_sentinel_components()
  wiki_lint_tool.py         — WikiLintTool + set_wiki_linter()
tests/
  test_sentinel.py            — 33 tests (signals, observer, autodream, linter, consolidator, digest, tools)
```

---

## Parallel Tool Dispatch — Agent Loop Enhancement

Cuts 1–2 seconds per skipped LLM round-trip when the model returns multiple tool calls. Read-only tools execute simultaneously; mutating tools run sequentially after. Each tool still goes through the full pipeline (pre-hooks → validate → permission → execute → telemetry → post-hooks).

### Architecture

```
Before:  LLM → file_read → LLM → grep → LLM → glob → LLM
After:   LLM → [file_read + grep + glob parallel] → LLM
```

### New Functions (`engine/agent_loop.py`)

```python
async def _dispatch_tool_calls(
    context: LoopContext,
    tool_calls: list[ToolUseBlock],
) -> list[ToolResultBlock]
    # Single call: direct execution, no partitioning
    # Multiple calls: partition by is_read_only → parallel + sequential
    # Results re-sorted to match original tool_calls order

def _is_tool_read_only(tool: object, tool_input: dict) -> bool
    # Calls tool.is_read_only(parsed_input) if method exists
    # Falls back to getattr(tool, "is_read_only", False) for attribute pattern
```

### Wiring into Existing Modules

| Module | Integration |
|--------|-------------|
| `engine/agent_loop.py:run_loop()` | Replaced inline dispatch (lines 141–168) with `_dispatch_tool_calls()` call |
| `tools/base.py:BaseTool.is_read_only()` | Existing method (Sprint 2) — no change needed; already returns `bool` per tool |
| `hooks/executor.py` | Unchanged — each parallel tool still calls pre/post hooks individually |
| `telemetry/tracker.py` | Unchanged — each tool still records its own telemetry |
| `permissions/checker.py` | Unchanged — `is_read_only` result still passed to `evaluate()` per tool |

### Read-Only Tools (parallel-safe)

| Tool | File |
|------|------|
| `file_read` | `tools/builtin/file_read.py` |
| `grep` | `tools/builtin/grep.py` |
| `glob` | `tools/builtin/glob.py` |
| `lcm_grep` | `tools/builtin/lcm_grep.py` |
| `lcm_describe` | `tools/builtin/lcm_describe.py` |
| `lcm_expand` | `tools/builtin/lcm_expand.py` |
| `lcm_expand_query` | `tools/builtin/lcm_expand_query.py` |
| `wiki_query` | `tools/builtin/wiki_query.py` |
| `sentinel_status` | `tools/builtin/sentinel_status.py` |
| `cron_list` | `tools/builtin/cron_list.py` |
| `task_list` | `tools/builtin/task_list.py` |
| `task_get` | `tools/builtin/task_get.py` |
| `task_output` | `tools/builtin/task_output.py` |

All other tools (`bash`, `file_write`, `file_edit`, `cron_create`, `cron_delete`, `wiki_compile`, `wiki_lint`, `agent`, etc.) default to `is_read_only() → False` and run sequentially.

### Error Isolation

`asyncio.gather(return_exceptions=True)` — one failed read-only tool returns an error `ToolResultBlock` without blocking others.

### File Tree

```
engine/
  agent_loop.py            — _dispatch_tool_calls(), _is_tool_read_only() added
tests/
  test_parallel_dispatch.py — 12 tests (read-only helper, single call, parallel timing,
                               order preservation, mixed dispatch, sequential mutating,
                               error isolation, unknown tools, pre-hook deny)
```

---

## Sprint 10: Model Router + Divergence Detector ✓

### `prometheus.adapter.router`
`src/prometheus/adapter/router.py` — adapted from leaky `runtime.py` (token scoring) + Hermes profiles (fallback chain)

```python
class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    QUICK_ANSWER = "quick_answer"
    CREATIVE = "creative"
    TOOL_HEAVY = "tool_heavy"

@dataclass
class TaskClassification:
    task_type: TaskType
    confidence: float           # 0.0–1.0
    matched_tokens: list[str]
    reason: str

class TaskClassifier:
    TASK_TOKENS: dict[TaskType, set[str]]   # keyword sets per task type
    def classify(message: str, tool_mentions: list[str] | None = None) -> TaskClassification

@dataclass
class ProviderConfig:
    provider: str               # "llama_cpp", "ollama", "anthropic"
    model: str
    base_url: str | None
    reason: str

@dataclass
class RoutingRule:
    task_type: TaskType
    provider: str
    model: str
    base_url: str | None
    min_confidence: float

class ModelRouter:
    def __init__(config: dict)              # reads config["model_router"] + config["model"]
    def route(message: str, tool_mentions: list[str] | None = None,
              force_provider: str | None = None) -> ProviderConfig
    def get_fallback(failed_provider: str) -> ProviderConfig | None
```

### `prometheus.coordinator.divergence`
`src/prometheus/coordinator/divergence.py` — adapted from OpenClaw `memory_extractor` (fact extraction) + LCM DAG (checkpoint persistence)

```python
# Goal extraction (standalone functions)
def extract_objectives(message: str) -> list[str]   # imperative verb detection, max 5
def extract_entities(message: str) -> list[str]      # file paths, quoted strings, capitalized words

@dataclass
class TaskGoal:
    original_message: str
    goal_hash: str              # sha256[:16]
    key_objectives: list[str]
    key_entities: list[str]

class GoalTracker:
    current_goal: TaskGoal | None
    def set_goal(message: str) -> TaskGoal
    def check_alignment(recent_messages: list[dict], tool_results: list[dict]) -> float  # 0.0–1.0
    def clear() -> None

@dataclass
class Checkpoint:
    task_id: str
    step_number: int
    goal_description: str
    goal_hash: str
    messages_snapshot: list[dict]
    tool_calls: list[dict]
    timestamp: float
    divergence_score: float
    def to_db_row() -> tuple
    @classmethod def from_db_row(row: tuple) -> Checkpoint

class CheckpointStore:
    """Persists checkpoints in lcm.db (shared with LCMConversationStore/LCMSummaryStore)."""
    def __init__(db_path: Path | None = None)       # defaults to ~/.prometheus/lcm.db
    def save(checkpoint: Checkpoint) -> None
    def get_latest(task_id: str) -> Checkpoint | None
    def delete_after(task_id: str, step_number: int) -> None

@dataclass
class DivergenceResult:
    score: float                # 0.0 = on track, 1.0 = off track
    should_rollback: bool
    reason: str
    checkpoint: Checkpoint | None

class DivergenceDetector:
    def __init__(config: dict, checkpoint_store: CheckpointStore | None = None,
                 notify_callback: Callable[[str], None] | None = None)
    def start_task(task_id: str, goal_message: str) -> None
    def record_tool_call(tool_name: str, args: dict, result: object, success: bool) -> None
    def maybe_checkpoint(messages: list[dict]) -> Checkpoint | None
    def evaluate(messages: list[dict], tool_results: list[dict]) -> DivergenceResult
    def rollback(checkpoint: Checkpoint, trust_level: int) -> tuple[bool, list[dict]]
    def end_task() -> None
```

Divergence scoring heuristics (no LLM cost):
1. Goal alignment — keyword overlap between objectives/entities and recent activity (inverted)
2. Tool failure rate — proportion of failed tool calls in recent window
3. Repetition detection — same tool called 3+ times consecutively
4. Context growth anomaly — messages-per-step ratio > 5

### LCM Schema Extension
`src/prometheus/memory/lcm_conversation_store.py` — added to `_apply_schema()`

```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    goal_hash TEXT NOT NULL,
    goal_description TEXT,
    messages_json TEXT NOT NULL,
    tool_calls_json TEXT NOT NULL,
    divergence_score REAL DEFAULT 0.0,
    created_at REAL NOT NULL,
    UNIQUE(task_id, step_number)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id, step_number DESC);
```

### Config Schema
`config/prometheus.yaml` — new top-level keys

```yaml
model_router:
  enabled: true
  rules:                        # list of {task_type, provider, model, base_url?, min_confidence?}
  fallback_chain:               # list of {provider, base_url?, model}

divergence:
  enabled: true
  checkpoint_interval: 5        # checkpoint every N tool calls
  threshold: 0.7                # score triggering rollback consideration
  auto_rollback_trust_level: 3  # only AUTONOMOUS trust auto-rolls back
  max_rollbacks: 2              # prevent infinite rollback loops
  use_llm_eval: false           # optional LLM-based eval (budget-capped)
  llm_eval_budget: 500
```

### Wiring into Existing Modules

| Module | Integration |
|--------|-------------|
| `engine/agent_loop.py:LoopContext` | Added `model_router` and `divergence_detector` fields |
| `engine/agent_loop.py:_execute_tool_call()` | Records each tool call via `divergence_detector.record_tool_call()` after telemetry |
| `engine/agent_loop.py:run_loop()` | After tool dispatch: calls `maybe_checkpoint()` + `evaluate()` + `rollback()` |
| `engine/agent_loop.py:AgentLoop.__init__()` | Accepts `model_router` and `divergence_detector` kwargs, passes to `LoopContext` |
| `__main__.py` | `create_model_router(config)` + `create_divergence_detector(config)` factories wired into CLI |
| `adapter/__init__.py` | Exports `ModelRouter`, `TaskClassifier`, `TaskType`, `ProviderConfig` |
| `coordinator/__init__.py` | Exports `DivergenceDetector`, `DivergenceResult`, `GoalTracker`, `Checkpoint`, `CheckpointStore` |
| `memory/lcm_conversation_store.py` | `checkpoints` table added to shared `lcm.db` schema (no separate database) |

### File Tree

```
adapter/
  router.py                 — TaskClassifier, ModelRouter, ProviderConfig, RoutingRule
coordinator/
  divergence.py             — GoalTracker, CheckpointStore, DivergenceDetector
memory/
  lcm_conversation_store.py — checkpoints table added to _apply_schema()
engine/
  agent_loop.py             — LoopContext + _execute_tool_call() + run_loop() wiring
config/
  prometheus.yaml            — model_router + divergence sections added
tests/
  test_router.py             — 16 tests (classifier accuracy, routing rules, fallback chain)
  test_divergence.py         — 23 tests (goal extraction, alignment, checkpoint CRUD, rollback)
```

---

## Sprint 11: Security Hardening ✓

### `prometheus.config.env_override`
`src/prometheus/config/env_override.py` — env var overrides + secret file loading (OpenClaw `secret-file.ts` pattern)

```python
# Env var → config path mapping (17 supported vars)
ENV_OVERRIDES: dict[str, tuple[str, ...]]     # e.g. "PROMETHEUS_TELEGRAM_TOKEN" → ("gateway", "telegram_token")
SECRET_FILE_VARS: dict[str, tuple[str, ...]]  # e.g. "PROMETHEUS_TELEGRAM_TOKEN_FILE" → ("gateway", "telegram_token")

def read_secret_file(file_path: str, label: str, max_bytes: int = 16384) -> str | None
    # Rejects symlinks (checked before resolve()), enforces size limit, strips whitespace

def apply_env_overrides(config: dict) -> dict
    # Applies secret files first, then direct env vars (higher priority). Mutates + returns config.
```

### `prometheus.permissions.audit`
`src/prometheus/permissions/audit.py` — dual-write audit logger (SQLite + JSONL)

```python
class AuditDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM_PENDING = "confirm_pending"
    CONFIRM_APPROVED = "confirm_approved"
    CONFIRM_REJECTED = "confirm_rejected"

@dataclass
class AuditEntry:
    timestamp: float
    tool_name: str
    decision: AuditDecision
    trust_level: int
    reason: str
    tool_input_summary: str             # redacted + truncated
    user_id: str | None
    session_id: str | None
    def to_dict() -> dict
    def to_json() -> str

class AuditLogger:
    def __init__(data_dir: Path, max_input_chars: int = 200)
        # Creates data_dir/audit.db (SQLite) + data_dir/permission_audit.jsonl

    def log(tool_name: str, decision: AuditDecision, trust_level: int, reason: str,
            tool_input: dict | str | None = None, user_id: str | None = None,
            session_id: str | None = None) -> AuditEntry
        # Writes to JSONL + SQLite + standard logger. Redacts tokens/keys in tool_input.

    def query_recent(limit: int = 50, decision: AuditDecision | None = None,
                     tool_name: str | None = None) -> list[AuditEntry]

    def stats(hours: int = 24) -> dict[str, int]    # {"allow": 42, "deny": 3}
```

### `prometheus.permissions.exfiltration`
`src/prometheus/permissions/exfiltration.py` — secret exfiltration pattern matcher

```python
@dataclass
class ExfiltrationMatch:
    pattern_name: str               # "network_sensitive_file", "subshell_exfil", etc.
    matched_text: str
    severity: str                   # "critical", "high", "medium"
    reason: str

class ExfiltrationDetector:
    SENSITIVE_PATHS: list[str]      # ~/.ssh/, ~/.aws/, .env, id_rsa, prometheus.yaml, etc.
    NETWORK_COMMANDS: list[str]     # curl, wget, nc, scp, rsync, etc.
    SECRET_ENV_PATTERNS: list[str]  # $*KEY, $*TOKEN, $*SECRET, $ANTHROPIC*, etc.

    def check_command(command: str) -> ExfiltrationMatch | None
        # Detects: network+sensitive_path, network+secret_env, subshell exfil,
        #          pipe exfil, base64 exfil, redirect exfil

    def check_url(url: str) -> ExfiltrationMatch | None
        # Detects secret env var patterns embedded in URLs
```

### `prometheus.tools.builtin.audit_query`
`src/prometheus/tools/builtin/audit_query.py` — in-agent audit inspection

```python
class AuditQueryTool(BaseTool):
    name = "audit_query"
    input_model = AuditQueryInput    # limit: int, decision: str, tool: str | None

    def __init__(audit_logger: AuditLogger)
    async def execute(arguments, context) -> ToolResult
    def is_read_only(arguments) -> True
```

### Wiring into Existing Modules

| Module | Integration |
|--------|-------------|
| `permissions/checker.py:SecurityGate.__init__()` | Added optional `audit_logger: AuditLogger` and `exfiltration_detector: ExfiltrationDetector` params |
| `permissions/checker.py:SecurityGate.evaluate()` | Exfiltration check runs first (before all other checks, including AUTONOMOUS bypass). Every decision logged via `_audit_log()` helper. |
| `permissions/checker.py:SecurityGate.from_config()` | Reads `security.audit.enabled` and `security.exfiltration.enabled` from YAML to optionally create logger + detector |
| `__main__.py:load_config()` | Now calls `apply_env_overrides(config)` after YAML load |
| `__main__.py:create_security_gate()` | Creates `AuditLogger` + `ExfiltrationDetector` and passes to `SecurityGate` |
| `__main__.py:create_tool_registry()` | Accepts optional `security_gate` kwarg; registers `AuditQueryTool` if audit logger is available |
| `__main__.py:main()` | Build order changed: `security_gate` created before `registry` so audit tool can be wired |
| `permissions/__init__.py` | Exports `AuditDecision`, `AuditEntry`, `AuditLogger`, `ExfiltrationDetector`, `ExfiltrationMatch` |
| `tools/builtin/__init__.py` | Exports `AuditQueryTool` |
| `config/prometheus.yaml` | Telegram token blanked (use env var). Added `security.audit` and `security.exfiltration` sections. |

### Config Schema
`config/prometheus.yaml` — new keys under `security:`

```yaml
security:
  audit:
    enabled: true               # write to ~/.prometheus/data/security/audit.db + .jsonl
    retention_days: 30
  exfiltration:
    enabled: true               # block network commands touching sensitive paths/env vars
```

### File Tree

```
config/
  env_override.py              — apply_env_overrides(), read_secret_file()
permissions/
  audit.py                     — AuditLogger, AuditDecision, AuditEntry
  exfiltration.py              — ExfiltrationDetector, ExfiltrationMatch
  checker.py                   — SecurityGate updated (audit + exfil params)
  __init__.py                  — exports updated
tools/builtin/
  audit_query.py               — AuditQueryTool
  __init__.py                  — exports updated
__main__.py                    — load_config env overrides, create_security_gate audit/exfil, registry wiring
config/prometheus.yaml         — token blanked, audit + exfiltration sections added
tests/
  test_config_env.py           — 13 tests (env overrides, secret file loading, symlink rejection)
  test_exfiltration.py         — 18 tests (block patterns + allow patterns + URL check)
  test_audit.py                — 16 tests (JSONL/SQLite write, query, redaction, gate integration)
```

---

## Sprint 12: MCP Integration + Context7 ✓

### `prometheus.mcp.types`
`src/prometheus/mcp/types.py` — catalog and status types

```python
@dataclass
class McpServerCatalog:
    server_name: str
    launch_summary: str
    tool_count: int

@dataclass
class McpCatalogTool:
    server_name: str
    safe_server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]

@dataclass
class McpToolCatalog:
    version: int = 1
    generated_at: float = 0.0
    servers: dict[str, McpServerCatalog]
    tools: list[McpCatalogTool]

@dataclass
class McpConnectionStatus:
    name: str
    state: Literal["connected", "failed", "pending", "disabled"]
    transport: str = "unknown"
    detail: str = ""
    tool_count: int = 0

def create_config_fingerprint(servers: dict) -> str   # sha1 for change detection
```

### `prometheus.mcp.transport`
`src/prometheus/mcp/transport.py` — transport resolution (OpenClaw `mcp-transport-config.ts`)

```python
@dataclass
class ResolvedStdioTransport:
    kind: Literal["stdio"]
    command: str
    args: list[str]
    env: dict[str, str] | None
    cwd: str | None
    timeout_ms: int
    @property description -> str

@dataclass
class ResolvedHttpTransport:
    kind: Literal["http"]
    transport_type: Literal["sse", "streamable-http"]
    url: str
    headers: dict[str, str] | None
    timeout_ms: int
    @property description -> str         # redacts passwords

def resolve_stdio_config(raw: dict) -> ResolvedStdioTransport | None
def resolve_http_config(raw: dict, transport_type: str = "sse") -> ResolvedHttpTransport | None
def resolve_transport(server_name: str, raw: dict) -> ResolvedTransport | None
    # Priority: stdio (command) > streamable-http > sse (url)
```

### `prometheus.mcp.names`
`src/prometheus/mcp/names.py` — safe tool naming (OpenClaw `pi-bundle-mcp-names.ts`)

```python
def sanitize_server_name(name: str, used_names: set[str]) -> str
def sanitize_tool_name(name: str) -> str
def build_safe_tool_name(server_name: str, tool_name: str, reserved_names: set[str]) -> str
    # Returns "mcp__{server}__{tool}" with collision suffix if needed
```

### `prometheus.mcp.runtime`
`src/prometheus/mcp/runtime.py` — connection management (OpenClaw `pi-bundle-mcp-runtime.ts`)

```python
class McpConnectionError(Exception): ...

class McpRuntime:
    def __init__(server_configs: dict[str, dict])
    @property config_fingerprint -> str              # detect config changes

    async def connect_all() -> None                  # connect + discover tools
    def get_catalog() -> McpToolCatalog
    def list_statuses() -> list[McpConnectionStatus]
    def list_tools() -> list[McpCatalogTool]
    async def call_tool(server_name: str, tool_name: str, arguments: dict) -> str
    async def close() -> None
```

Internals ported from OpenClaw:
- `_connect_with_timeout(session, timeout_ms)` — `asyncio.wait_for` wrapper
- `_list_all_tools(session)` — paginated tool listing

### `prometheus.mcp.adapter`
`src/prometheus/mcp/adapter.py` — wraps MCP tools as `BaseTool` instances

```python
class McpToolAdapter(BaseTool):
    input_model = _McpDynamicInput            # ConfigDict(extra="allow") for arbitrary MCP args
    def __init__(runtime: McpRuntime, tool_info: McpCatalogTool, safe_name: str)
    def to_api_schema() -> dict               # returns MCP-provided schema, not pydantic
    def to_openai_schema() -> dict            # same, in OpenAI format
    async def execute(arguments, context) -> ToolResult
    def is_read_only(arguments) -> True

def register_mcp_tools(registry: ToolRegistry, runtime: McpRuntime) -> int
    # Registers all MCP tools with collision-safe names. Returns count.
```

### `prometheus.tools.builtin.mcp_status`
`src/prometheus/tools/builtin/mcp_status.py` — connection status tool

```python
class McpStatusTool(BaseTool):
    name = "mcp_status"
    input_model = McpStatusInput              # server: str | None
    def __init__(runtime: McpRuntime)
    async def execute(arguments, context) -> ToolResult
    def is_read_only(arguments) -> True
```

### Wiring into Existing Modules

| Module | Integration |
|--------|-------------|
| `__main__.py` | `create_mcp_runtime(config, registry)` factory: creates `McpRuntime`, calls `connect_all()`, registers tools + `McpStatusTool` |
| `__main__.py:main()` | MCP init moved into `_async_main()` so connections live in same event loop as agent loop |
| `__main__.py:_async_main()` | `await create_mcp_runtime(config, registry)` before run_interactive/run_once; `await mcp_runtime.close()` in finally |
| `pyproject.toml` | `mcp = ["mcp>=1.0"]` added to optional dependencies |
| `config/prometheus.yaml` | `mcp_servers:` section with Context7 configured |

### Config Schema
`config/prometheus.yaml` — new top-level key

```yaml
mcp_servers:
  context7:
    command: npx
    args: ["-y", "@upstash/context7-mcp"]
    connectionTimeoutMs: 60000

  # HTTP/SSE example (not yet implemented):
  # remote_server:
  #   url: http://localhost:3000/mcp
  #   transport: sse
  #   headers:
  #     Authorization: "Bearer xxx"
```

### File Tree

```
mcp/
  __init__.py                  — package exports
  types.py                     — McpCatalogTool, McpToolCatalog, McpConnectionStatus
  transport.py                 — ResolvedStdioTransport, ResolvedHttpTransport, resolve_transport()
  names.py                     — sanitize_server_name(), build_safe_tool_name()
  runtime.py                   — McpRuntime (connect, discover, call, close)
  adapter.py                   — McpToolAdapter (BaseTool wrapper), register_mcp_tools()
tools/builtin/
  mcp_status.py                — McpStatusTool
__main__.py                    — create_mcp_runtime(), _async_main() lifecycle
config/prometheus.yaml         — mcp_servers section added
pyproject.toml                 — mcp optional dependency
tests/
  test_mcp_transport.py        — 20 tests (stdio, HTTP, resolve priority)
  test_mcp_names.py            — 12 tests (sanitization, collision avoidance)
  test_mcp_adapter.py          — 17 tests (adapter schema, execute, registry, runtime unit)
```

## Sprint 13: DeepEval + Phoenix Evaluation Suite ✓

Automated quality measurement with LLM-as-judge scoring, visual trace debugging, and nightly cron.

### `prometheus.evals.golden_dataset`
`src/prometheus/evals/golden_dataset.py` — 26 canonical evaluation tasks

```python
class TaskTier(IntEnum):
    TIER_1 = 1  # Atomic single-tool (21 tasks)
    TIER_2 = 2  # Multi-step workflows (5 tasks)

@dataclass
class GoldenTask:
    id: str
    name: str
    tier: int
    input: str                   # Prompt for agent
    expected_behavior: str       # Free-text for LLM judge
    expected_tools: list[str]
    tags: list[str]
    requires_network: bool       # True for web_search/web_fetch tasks
    max_turns: int = 10

load_golden_dataset(tier=None, skip_network=True) -> list[GoldenTask]
```

### `prometheus.evals.judge`
`src/prometheus/evals/judge.py` — LLM-as-judge via llama.cpp `/v1/chat/completions`

```python
@dataclass
class JudgeVerdict:
    score: float         # 0.0-1.0
    reasoning: str
    raw_response: str

class PrometheusJudge:
    def __init__(self, base_url="http://100.110.140.39:8080", model=None, timeout=120.0)

    # Constrained JSON scoring (primary — used by metrics, Sprint 14)
    async def evaluate(task_input, agent_output, expected_behavior, tool_trace=None) -> JudgeVerdict

    # G-Eval: chain-of-thought scoring (kept for manual/debug use)
    async def evaluate_geval(criteria: list[str], context: str) -> JudgeVerdict
```

`evaluate()` uses constrained decoding via `response_format` with `json_schema` — see Sprint 14 below.

### `prometheus.evals.metrics`
`src/prometheus/evals/metrics.py` — three evaluation metrics (DeepEval-compatible stubs when deepeval not installed)

```python
class TaskCompletionMetric(BaseMetric):     # Constrained JSON, threshold=0.7
    def __init__(self, judge: PrometheusJudge, threshold=0.7)

class ToolUsageMetric(BaseMetric):          # Deterministic, no LLM needed, threshold=0.5
    def __init__(self, threshold=0.5)

class NoHallucinationMetric(BaseMetric):    # Constrained JSON, threshold=0.8
    def __init__(self, judge: PrometheusJudge, threshold=0.8)
```

`TaskCompletionMetric` and `NoHallucinationMetric` use `evaluate()` with constrained decoding (Sprint 14). `ToolUsageMetric` is deterministic — compares tool trace against expected tools, no LLM call.

### `prometheus.evals.runner`
`src/prometheus/evals/runner.py` — orchestrates golden tasks through `AgentLoop`

```python
@dataclass
class MetricScore:
    metric_name: str; score: float; threshold: float; passed: bool; reasoning: str

@dataclass
class EvalResult:
    task_id: str; task_name: str; tier: int; agent_output: str
    turns: int; latency_ms: float; tool_trace: list[dict]
    metrics: list[MetricScore]; error: str | None
    failure_source: str       # "pass", "model", "harness", "unclear"
    failure_category: str     # e.g. "model:wrong_tool", "harness:tool_crash"
    failure_detail: str

class EvalRunner:
    def __init__(self, agent_loop: AgentLoop, judge: PrometheusJudge, system_prompt: str, *, config=None)
    async def run_task(task: GoldenTask) -> EvalResult
    async def run_all(tasks=None, *, tier=None, skip_network=True) -> list[EvalResult]
    def save_results(results, output_dir=None) -> Path     # JSON + trend record
    def print_summary(results, output_dir=None) -> None    # stdout with trend comparison
```

Critical: copies `AgentLoop._tool_trace` immediately after `run_async()` (reset at line 504, cleared by post-task hook at line 533).

### `prometheus.evals.trends`
`src/prometheus/evals/trends.py` — score tracking over time via SQLite

```python
@dataclass
class TrendRow:
    timestamp: str; task_count: int; completed: int; errored: int
    avg_latency_ms: float; metric_averages: dict[str, float]

class TrendTracker:
    def __init__(self, db_path=None)               # default: ~/.prometheus/eval_results/trends.db
    def record(summary: dict) -> None              # append run summary row
    def get_latest(n=10) -> list[TrendRow]         # most recent N runs
    def get_previous() -> TrendRow | None          # last run before current
    def format_trend_comparison(current, previous) -> str  # "Task Completion: 0.850 (+0.130 vs prev)"
```

`save_results()` automatically calls `TrendTracker.record()`. `print_summary()` shows deltas vs previous run.

### `prometheus.tracing`
`src/prometheus/tracing/` — optional Phoenix/OpenTelemetry integration

```python
# phoenix.py
init_tracing(config=None) -> TracerProvider | None   # gated by PROMETHEUS_TRACING=1
get_tracer() -> Tracer | _NoOpTracer                 # no-op when disabled
shutdown_tracing() -> None                           # flush + cleanup

# spans.py
@traced(name=None, attributes=None)                  # decorator for sync/async functions
span_context(name, attributes=None)                  # context manager for ad-hoc spans
```

Zero-cost when disabled: `_NoOpTracer` and `_NoOpSpan` avoid any overhead in the hot path.

### Nightly Script
`scripts/run_nightly_evals.py` — cron-ready standalone runner

```bash
# Cron: 0 3 * * * cd ~/Prometheus && .venv/bin/python scripts/run_nightly_evals.py
python scripts/run_nightly_evals.py              # default: skip network tasks
python scripts/run_nightly_evals.py --tier 1     # tier 1 only
python scripts/run_nightly_evals.py --no-skip-network  # include web tasks
PROMETHEUS_TRACING=1 python scripts/run_nightly_evals.py  # with Phoenix spans
```

Imports factory functions from `__main__.py` (same pattern as `scripts/daemon.py`). Health-checks judge endpoint before running.

### Wiring Into Existing Modules

| Existing Module | How Sprint 13 Connects |
|--------|-------------|
| `engine/agent_loop.py` | `EvalRunner` calls `AgentLoop.run_async()`, reads `_tool_trace` for metrics |
| `__main__.py` | Nightly script imports `load_config`, `create_provider`, `create_tool_registry`, `create_adapter`, `create_security_gate`, `build_system_prompt` |
| `benchmarks/` | Golden dataset covers same tool categories as Sprint 8 suite but uses `input`/`expected_behavior` for LLM judging instead of deterministic checks |
| `telemetry/tracker.py` | Complements existing `ToolCallTelemetry` — evals measure quality, telemetry measures reliability |
| `pyproject.toml` | `evals = ["deepeval>=2.0", "arize-phoenix>=5.0", "opentelemetry-api>=1.20", "opentelemetry-sdk>=1.20"]` optional deps |
| `config/prometheus.yaml` | `evals:` (judge_base_url, results_dir, skip_network_tasks) and `tracing:` (enabled, phoenix_endpoint, service_name) sections |

### File Tree

```
evals/
  __init__.py                  — package exports
  golden_dataset.py            — GoldenTask, TaskTier, load_golden_dataset()
  judge.py                     — PrometheusJudge (constrained decoding + G-Eval)
  metrics.py                   — TaskCompletionMetric, ToolUsageMetric, NoHallucinationMetric
  runner.py                    — EvalRunner, EvalResult, MetricScore
  classifier.py                — FailureClassification, classify_failure()
  trends.py                    — TrendTracker, TrendRow (SQLite)
tracing/
  __init__.py                  — package exports
  phoenix.py                   — init_tracing(), get_tracer(), shutdown_tracing()
  spans.py                     — @traced decorator, span_context()
scripts/
  run_nightly_evals.py         — cron-ready nightly eval runner
config/prometheus.yaml         — evals + tracing sections added
pyproject.toml                 — evals optional dependency group
tests/
  test_evals.py                — 66 tests (dataset, judge, constrained decoding, metrics, runner, classifier, trends, tracing)
```

## Sprint 14: Constrained Decoding Judge ✓

Fixes the #1 failure mode in nightly evals: judge returning empty or unparseable output. Forces valid JSON at the token level via llama.cpp grammar constraints.

### Constrained Decoding (`evals/judge.py`)

```python
# Schema passed to llama.cpp response_format — converted to GBNF grammar
JUDGE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}
```

`evaluate()` now passes three things to `/v1/chat/completions`:

```python
response_format={"type": "json_schema", "json_schema": {"schema": JUDGE_SCORE_SCHEMA, "strict": True}}
chat_template_kwargs={"enable_thinking": False}   # suppresses <think> blocks at template level
```

`_call_llm()` also checks `reasoning_content` if `content` is empty (Qwen 3.5 bug where thinking leaks despite suppression).

### Fallback Parser (`_parse_verdict`)

4-layer parsing for servers without constrained decoding (Ollama, etc.):
1. Direct `json.loads(raw)` — constrained output
2. Strip markdown fences (`` ```json ... ``` ``) — Ollama pattern
3. Extract `{…}` JSON substring — preamble text
4. Regex number fallback + safe 0.0 default on empty

### Failure Classifier (`evals/classifier.py`)

```python
class FailureSource(str, Enum):
    PASS = "pass"           # No failure
    MODEL = "model"         # LLM model issue
    HARNESS = "harness"     # Harness bug (tool crash, permission block)
    UNCLEAR = "unclear"     # Can't determine

class FailureCategory(str, Enum):
    # Model failures
    NO_TOOL_CALL = "model:no_tool_call"
    WRONG_TOOL = "model:wrong_tool"
    BAD_ARGS = "model:bad_args"
    HALLUCINATED_OUTPUT = "model:hallucinated"
    INCOMPLETE = "model:incomplete"
    # Harness failures
    TOOL_ERROR = "harness:tool_error"
    TOOL_CRASH = "harness:tool_crash"
    PERMISSION_DENIED = "harness:permission"
    VALIDATION_FAIL = "harness:validation"

def classify_failure(task_id, expected_tools, tool_trace, agent_output, error, metric_scores) -> FailureClassification
```

Pure logic on trace data — no LLM call. Analyzes tool traces and metric scores to determine whether a failure is the model's fault or a harness bug. Critical for A/B testing models.

### Metrics Switch

Both `TaskCompletionMetric` and `NoHallucinationMetric` switched from `evaluate_geval()` to `evaluate()`. G-Eval was a workaround for bad JSON output — grammar constraints fix it at the source.

### Wiring Into Existing Modules

| Existing Module | How Sprint 14 Connects |
|--------|-------------|
| `evals/judge.py` | `_call_llm()` gains `response_format` + `chat_template_kwargs` params; `evaluate()` passes schema |
| `evals/metrics.py` | Both LLM metrics call `judge.evaluate()` instead of `judge.evaluate_geval()` |
| `evals/runner.py` | `run_task()` calls `classify_failure()` after metrics; `EvalResult` gains `failure_source`, `failure_category`, `failure_detail` fields; `print_summary()` shows PASS/MDL/HRN/ERR status + breakdown |

### Impact (Gemma4-26B, 24 tasks)

| Metric | Before (Sprint 13) | After (Sprint 14) |
|--------|--------------------|--------------------|
| PASS count | 14 / 24 | 22 / 24 |
| Task Completion | 0.867 | 1.000 |
| No Hallucination | 0.667 | 0.867 |
| Parse failures | ~30% | 0% |
| Empty responses | ~4-10/run | 0 |

---

## Sprint 15: Wiring Audit + Data Reset CLI ✓

Full wiring audit of Sprints 0-14.  Six components were built and unit-tested but never connected at runtime.  All six fixed; 36 integration tests added.

### Daemon wiring fix (`scripts/daemon.py`)

The daemon constructed `AgentLoop` with only `provider`, `model`, and `tool_registry`. Every other Sprint 2-10 component was skipped. Now matches the CLI path:

```python
from prometheus.__main__ import (
    create_adapter, create_divergence_detector,
    create_model_router, create_security_gate, create_tool_registry,
)
from prometheus.hooks.executor import HookExecutor, HookExecutionContext
from prometheus.hooks.registry import HookRegistry

adapter        = create_adapter(model_config)
security_gate  = create_security_gate(security_config)
model_router   = create_model_router(config)
divergence_det = create_divergence_detector(config)
telemetry      = ToolCallTelemetry()
hook_executor  = HookExecutor(
    registry=HookRegistry(),
    context=HookExecutionContext(cwd=Path.cwd(), provider=provider, default_model=model_name),
)

agent_loop = AgentLoop(
    provider=provider, model=model_name, tool_registry=registry,
    adapter=adapter, permission_checker=security_gate,
    hook_executor=hook_executor, telemetry=telemetry,
    model_router=model_router, divergence_detector=divergence_det,
)
```

### HookExecutor wired (`src/prometheus/__main__.py`)

Sprint 2 built `HookExecutor` + `HookRegistry` but never instantiated either. Now created with an empty registry (ready for user-configured hooks) and passed to `LoopContext`:

```python
hook_executor = HookExecutor(
    registry: HookRegistry,                              # empty — no hooks registered yet
    context: HookExecutionContext(cwd, provider, model),  # needed for prompt/agent hooks
)
# → LoopContext(hook_executor=hook_executor)
```

### ModelRouter invoked (`src/prometheus/engine/agent_loop.py`)

Sprint 10 created and passed `ModelRouter` to `LoopContext` but `run_loop()` never called it. Now invoked at the top of the loop:

```python
# run_loop(), before the turn loop:
if context.model_router is not None and messages:
    route = context.model_router.route(first_user_message)
    log.debug("ModelRouter: %s → %s/%s (%s)", msg[:60], route.provider, route.model, route.reason)
```

### Telemetry error-path coverage (`src/prometheus/engine/agent_loop.py`)

Sprint 3 recorded telemetry at two points; six error paths returned before reaching them. All now record:

| `error_type` | Trigger |
|---|---|
| `hook_blocked` | Pre-tool hook returns `blocked=True` |
| `no_registry` | `tool_registry` is `None` |
| `unknown_tool` | `registry.get(name)` returns `None` |
| `input_validation` | `input_model.model_validate()` raises |
| `permission_denied` | `SecurityGate` denies or user rejects confirmation |
| `parallel_exception` | `asyncio.gather` catches exception in read-only dispatch |

### Data reset helpers (`src/prometheus/__main__.py`)

```python
def _reset_telemetry() -> None
    # Deletes ~/.prometheus/telemetry.db (+ WAL/SHM) after y/N confirmation.

def _reset_data() -> None
    # Deletes all user data after y/N confirmation. Preserves config files.
    # Files: telemetry.db, memory.db, data/lcm.db, data/security/audit.db
    # Dirs:  eval_results/, wiki/, sentinel/, skills/auto/
```

```
python -m prometheus --reset-telemetry
python -m prometheus --reset-data
```

### Querying telemetry

```bash
sqlite3 ~/.prometheus/telemetry.db "SELECT tool_name, success, latency_ms, error_type FROM tool_calls ORDER BY timestamp DESC LIMIT 20;"
sqlite3 ~/.prometheus/telemetry.db "SELECT tool_name, COUNT(*), AVG(latency_ms), SUM(success)*1.0/COUNT(*) AS rate FROM tool_calls GROUP BY tool_name;"
```

### Wiring audit results

| Sprint | Component | Pre-fix status | Fix |
|---|---|---|---|
| 2 | HookExecutor | Built, never instantiated | Created in `__main__.py` + `daemon.py`, passed to `LoopContext` |
| 3 | ModelAdapter | CLI only | Wired into daemon via `create_adapter()` |
| 3 | ToolCallTelemetry | Instantiated, never passed to daemon `AgentLoop` | Passed as `telemetry=`; SENTINEL reuses same instance |
| 4 | SecurityGate | CLI only | Wired into daemon via `create_security_gate()` |
| 10 | ModelRouter | Passed to `LoopContext`, never invoked | `run_loop()` now calls `router.route()` |
| 10 | DivergenceDetector | CLI only | Wired into daemon via `create_divergence_detector()` |

Components confirmed wired (no fix needed): ToolRegistry, AuditLogger, ExfiltrationDetector, MemoryStore, MemoryExtractor, LCMEngine, TelegramAdapter, Heartbeat, CronScheduler, ArchiveWriter, SignalBus, ActivityObserver, AutoDreamEngine, WikiLinter, MemoryConsolidator, TelemetryDigest, KnowledgeSynthesizer, McpRuntime, PrometheusJudge.

Deferred (by design): ContextCompressor, ToolResultTruncator, TokenBudget, DynamicToolLoader.

### Wiring into existing modules

| Existing module | How Sprint 15 connects |
|---|---|
| `scripts/daemon.py` (Sprint 6) | `AgentLoop()` gains 6 kwargs: `adapter`, `security_gate`, `hook_executor`, `telemetry`, `model_router`, `divergence_detector` |
| `engine/agent_loop.py` (Sprint 1) | `run_loop()` calls `model_router.route()`; `_execute_tool_call()` gains 7 telemetry recording sites |
| `__main__.py` (Sprint 0) | Creates `HookExecutor` + `LoopContext`; adds `--reset-telemetry` / `--reset-data` argparse flags |
| `pyproject.toml` | Registers `integration` pytest marker |

### Integration wiring tests

`tests/test_wiring.py` — 25 tests verifying runtime invocation across all sprints.
`tests/test_telemetry_wiring.py` — 11 tests for telemetry recording + reset commands.

```bash
pytest -m integration tests/test_wiring.py -v   # wiring tests only
pytest tests/test_telemetry_wiring.py -v          # telemetry + reset tests
```

**Policy: every future sprint must add wiring tests to `test_wiring.py`.**

---

## Sprint 15b: GRAFT Phase 1 — Telegram Media + Vision + Voice ✓

Restores donor features stripped during Sprint 6. Additive only — no existing handlers, commands, or behavior changed.

### `prometheus.gateway.media_cache`
`src/prometheus/gateway/media_cache.py` — disk-backed media cache (Hermes `base.py` pattern)

```python
# Module-level cache functions (not a class — matches Hermes pattern)
cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str       # → ~/.prometheus/cache/images/img_{uuid}.jpg
cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str       # → ~/.prometheus/cache/audio/audio_{uuid}.ogg
cache_document_from_bytes(data: bytes, original_filename: str) -> str  # → ~/.prometheus/cache/documents/doc_{uuid}_{name}
extract_text_from_document(path: str) -> str | None                 # inline text for .txt/.md/.py etc. (≤100KB)
sniff_image_extension(file_path: str | None) -> str                 # guess ext from Telegram file_path
cleanup_cache(subdir: str, max_age_hours: int = 24) -> int          # TTL-based eviction

SUPPORTED_DOCUMENT_TYPES: dict[str, str]  # 20 extensions → MIME mappings
```

### `prometheus.gateway.sticker_cache`
`src/prometheus/gateway/sticker_cache.py` — JSON file cache for sticker descriptions (Hermes `sticker_cache.py` pattern)

```python
get_cached_description(file_unique_id: str) -> dict | None
cache_sticker_description(file_unique_id: str, description: str, emoji: str = "", set_name: str = "") -> None
build_sticker_injection(description: str, emoji: str = "", set_name: str = "") -> str
build_animated_sticker_injection(emoji: str = "") -> str

STICKER_VISION_PROMPT: str  # "Describe this sticker in 1-2 sentences..."
```

### `prometheus.gateway.status`
`src/prometheus/gateway/status.py` — scoped daemon lock (Hermes `status.py` pattern)

```python
acquire_daemon_lock() -> tuple[bool, str]   # (ok, reason) — atomic O_CREAT|O_EXCL, /proc stale detection
release_daemon_lock() -> None               # only releases if current PID owns the lock
```

### `prometheus.tools.builtin.vision`
`src/prometheus/tools/builtin/vision.py` — image analysis via multimodal LLM (Hermes `vision_tools.py` pattern)

```python
class VisionInput(BaseModel):
    image_path: str
    question: str = "Describe this image in detail."

class VisionTool(BaseTool):
    name = "vision_analyze"
    async execute(arguments: VisionInput, context: ToolExecutionContext) -> ToolResult
        # reads image → base64 data URL → sends to provider as multimodal content block
    def is_read_only(...) -> True
```

### `prometheus.tools.builtin.whisper_stt`
`src/prometheus/tools/builtin/whisper_stt.py` — speech-to-text (counterpart to existing `tts.py`)

```python
class WhisperSTTInput(BaseModel):
    audio_path: str
    language: str = "en"
    model: str = "base"     # tiny, base, small, medium, large

class WhisperSTTTool(BaseTool):
    name = "whisper_stt"
    async execute(arguments: WhisperSTTInput, context: ToolExecutionContext) -> ToolResult
        # .ogg → ffmpeg → .wav → whisper/faster-whisper CLI → transcription text
    def is_read_only(...) -> True
```

### Telegram media handlers (`src/prometheus/gateway/telegram.py`)

Four handlers added alongside existing text/command handlers:

```python
# Registered in start() — existing handlers untouched
self._app.add_handler(MessageHandler(filters.PHOTO,        self._handle_photo))
self._app.add_handler(MessageHandler(filters.VOICE,        self._handle_voice))
self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
self._app.add_handler(MessageHandler(filters.Sticker.ALL,  self._handle_sticker))
```

| Handler | Flow |
|---|---|
| `_handle_photo` | download → `cache_image_from_bytes` → `VisionTool` describes → dispatch enriched text |
| `_handle_voice` | download .ogg → `cache_audio_from_bytes` → `WhisperSTTTool` transcribes → dispatch transcription |
| `_handle_document` | download → `cache_document_from_bytes` → `extract_text_from_document` inline injection → dispatch |
| `_handle_sticker` | cache check → if miss: download .webp → `VisionTool` → `cache_sticker_description` → dispatch |

### `platform_base.py` additions

```python
class MessageType(str, Enum):
    ...
    STICKER = "sticker"   # NEW
    AUDIO = "audio"       # NEW
    VIDEO = "video"       # NEW

@dataclass
class MessageEvent:
    ...
    media_urls: list[str] = field(default_factory=list)   # NEW — local cached paths
    media_types: list[str] = field(default_factory=list)  # NEW — MIME strings
    caption: str | None = None                            # NEW
```

### `config/prometheus.yaml` additions

```yaml
gateway:
  media:
    max_file_size_mb: 20
    cache_dir: "~/.prometheus/cache/media"
    sticker_cache_dir: "~/.prometheus/cache/stickers"
  rate_limits:
    messages_per_minute: 30
    media_downloads_per_minute: 10

whisper:
  enabled: true
  model: "base"
  device: "auto"
  language: "en"
```

### Wiring into existing modules

| Existing module | How Sprint 15b connects |
|---|---|
| `gateway/telegram.py` (Sprint 6) | 4 new media handlers registered in `start()` alongside existing text/command handlers |
| `gateway/platform_base.py` (Sprint 6) | `MessageEvent` gains `media_urls`, `media_types`, `caption` fields; `MessageType` gains STICKER/AUDIO/VIDEO |
| `gateway/config.py` (Sprint 6) | `PlatformConfig` gains `max_file_size_mb`, `media_cache_dir`, rate limit fields |
| `scripts/daemon.py` (Sprint 6) | `acquire_daemon_lock()` at startup, `release_daemon_lock()` in signal handler |
| `tools/builtin/tts.py` (Sprint 5) | `whisper_stt.py` mirrors TTS pattern (Pydantic input, async subprocess, engine auto-detection) |

---

## Sprint 15c: GRAFT Phase 2 — Hook Reload + Compression + Approval + Credentials ✓

Four independent additive restorations. No existing behavior changed.

### `prometheus.hooks.loader`
`src/prometheus/hooks/loader.py` — builds HookRegistry from YAML config (OpenHarness `loader.py` pattern)

```python
def load_hook_registry(hooks_config: dict[str, list[dict]]) -> HookRegistry
    # hooks_config: {"pre_tool_use": [{"type": "command", "command": "echo"}], ...}
    # Returns populated HookRegistry. Skips unknown events/types gracefully.
```

### `prometheus.hooks.hot_reload`
`src/prometheus/hooks/hot_reload.py` — mtime-based lazy reloader (OpenHarness `hot_reload.py` pattern)

```python
class HookReloader:
    def __init__(self, config_path: Path) -> None
    def current_registry(self) -> HookRegistry       # lazy check — reloads if mtime changed
    async def start_watching(self, interval: float = 5.0, on_reload: callable = None) -> None
    def stop_watching(self) -> None
```

### `prometheus.context.compression` — Tier 2 addition
`src/prometheus/context/compression.py` — extended with async summarization

```python
class ContextCompressor:
    # Existing Tier 1 (unchanged):
    def maybe_compress(self, messages) -> list[ConversationMessage]

    # New Tier 2:
    async def maybe_compress_async(self, messages, provider: ModelProvider = None) -> list[ConversationMessage]
        # Runs Tier 1 first. If budget still over 90% and provider available,
        # summarizes old message batches (3-5 sentences each via LLM).
        # Protected messages (last fresh_tail_count) never summarized.
```

### `prometheus.permissions.approval_queue`
`src/prometheus/permissions/approval_queue.py` — Telegram confirmation flow for LEVEL 1 actions

```python
class ApprovalResult(str, Enum):
    APPROVED, DENIED, TIMEOUT

class ApprovalQueue:
    def __init__(self, telegram_adapter=None, timeout_seconds: int = 300, default_chat_id: int = None)
    async def request_approval(self, tool_name: str, description: str, chat_id: int = None) -> ApprovalResult
    async def approve(self, request_id: str) -> bool
    async def deny(self, request_id: str) -> bool
    def list_pending(self) -> list[PendingAction]
```

Telegram commands: `/approve {id}`, `/deny {id}`, `/pending`

### `prometheus.providers.credential_pool`
`src/prometheus/providers/credential_pool.py` — multi-key rotation (Hermes `auxiliary_client.py` pattern)

```python
class CredentialPool:
    def __init__(self, api_keys: list[str], dead_key_cooldown_seconds: int = 300)
    def get_next(self) -> str             # round-robin, skips dead keys
    def report_success(self, key: str)
    def report_error(self, key: str, status_code: int)  # 429→rotate, 401→mark dead
    @property
    def active_count(self) -> int
```

### `config/prometheus.yaml` additions

```yaml
security:
  approval_queue:
    enabled: true
    timeout_seconds: 300
```

### Wiring into existing modules

| Existing module | How Sprint 15c connects |
|---|---|
| `hooks/executor.py` (Sprint 2) | `update_registry()` called by `HookReloader` on config change |
| `context/compression.py` (Sprint 4) | `maybe_compress_async()` added alongside existing `maybe_compress()` |
| `permissions/checker.py` (Sprint 4) | `SecurityGate.__init__` gains optional `approval_queue` parameter |
| `gateway/telegram.py` (Sprint 6) | 3 new commands: `/approve`, `/deny`, `/pending` |
| `scripts/daemon.py` (Sprint 6) | `ApprovalQueue` wired to `SecurityGate` + `TelegramAdapter` when enabled |

## Sprint 16: GRAFT-THREAD — Gateway-Agnostic Conversation Memory ✓

Every Telegram and Slack message was dispatched as a brand-new conversation — no
history survived between turns. The LCM system existed but was never wired to any
gateway. This sprint adds a shared session layer so all gateways get multi-turn
memory for free.

### `prometheus.engine.session`
`src/prometheus/engine/session.py` — novel code

```python
MAX_SESSION_MESSAGES = 50

class ChatSession:
    def __init__(self, session_id: str) -> None
    def add_user_message(self, text: str) -> None
    def add_result_messages(self, result_messages: list[ConversationMessage], original_len: int) -> None
    def rollback_last(self) -> None             # undo last append (error recovery)
    def get_messages(self) -> list[ConversationMessage]
    def clear(self) -> None
    def trim(self, max_messages: int = MAX_SESSION_MESSAGES) -> None

class SessionManager:
    MAX_SESSION_MESSAGES = 50
    def __init__(self) -> None
    def get_or_create(self, session_id: str) -> ChatSession
    def clear(self, session_id: str) -> None    # empties history, keeps object
    def remove(self, session_id: str) -> None   # deletes session entirely
```

### `prometheus.engine.agent_loop` — updated signature
`src/prometheus/engine/agent_loop.py`

```python
class AgentLoop:
    async def run_async(
        self,
        system_prompt: str,
        user_message: str = "",
        *,
        messages: list[ConversationMessage] | None = None,  # NEW — pre-built history
        tools: list | None = None,
    ) -> RunResult
    # When messages= is provided, uses it (shallow-copied) instead of
    # creating a fresh single-message list. Backward compatible — all
    # existing callers pass user_message= only.
    def run(self, ...) -> RunResult   # same signature change, wraps run_async
```

### Wiring into existing modules

| Existing module | How Sprint 16 connects |
|---|---|
| `engine/agent_loop.py` (Sprint 1) | `run_async()` gains optional `messages=` parameter for pre-built history |
| `gateway/telegram.py` (Sprint 6) | `_dispatch_to_agent()` uses `SessionManager` to accumulate per-chat history; `/reset` and `/clear` call `session_manager.clear()` |
| `gateway/slack.py` (Sprint 6) | `_dispatch_to_agent()` uses same `SessionManager`; `/prometheus-reset` clears via `session_manager.clear()` |
| `scripts/daemon.py` (Sprint 6) | Creates one `SessionManager` and passes it to both `TelegramAdapter` and `SlackAdapter` |
| `engine/__init__.py` (Sprint 1) | Exports `ChatSession` and `SessionManager` |

### `prometheus.providers.stub._build_openai_messages` — multimodal passthrough
`src/prometheus/providers/stub.py` — used by both `StubProvider` and `LlamaCppProvider`

```python
def _build_openai_messages(request: ApiMessageRequest) -> list[dict[str, Any]]
    # Now passes through pre-formatted dicts (e.g. multimodal image_url
    # content blocks from VisionTool) instead of crashing on them.
    # ConversationMessage objects are converted as before.
```

### `prometheus.gateway.telegram._handle_photo` — improved fallback
`src/prometheus/gateway/telegram.py`

When vision analysis fails (model doesn't support it, or server error), the photo
handler now injects `[The user sent a photo with caption:] ...` so the LLM knows
an image was attached even without a description. Previously only the bare caption
was passed, giving the model no indication a photo existed.

### Infrastructure: llama-server vision on 4090

```
~/.config/systemd/user/llama-server.service  (on oara-4090)
  --mmproj /home/oara-4090/models/mmproj-BF16.gguf   # Gemma 4 vision projector
  --ubatch-size 2048 --batch-size 2048                # required for image tokens
```

### Additional wiring

| Existing module | How Sprint 16 connects |
|---|---|
| `providers/stub.py` (Sprint 1) | `_build_openai_messages()` passes through dict messages for multimodal content |
| `tools/builtin/vision.py` (Sprint 15b) | VisionTool's multimodal request now reaches llama.cpp instead of crashing |
| `gateway/telegram.py` (Sprint 15b) | `_handle_photo()` fallback text improved when vision unavailable |
| `~/.config/prometheus/env` (new) | `EnvironmentFile` for secrets; `prometheus.service` reads token from here |

## Sprint 17: BOOTSTRAP — Layer 1 Identity Files ✓

Prometheus had no persistent identity. The model didn't know who it was, who
the owner was, or what agents it could spawn. This sprint adds the Layer 1
bootstrap system — permanent files that load into every system prompt, plus
wiring for MEMORY.md and USER.md which existed but were never injected.

### Bootstrap files (`~/.prometheus/`)

```
~/.prometheus/
├── SOUL.md     — identity, hardware, capabilities, behavioral rules (STATIC, every prompt)
├── AGENTS.md   — agent registry, specializations, spawn rules (STATIC, every prompt)
├── MEMORY.md   — persistent facts (DYNAMIC, auto-loaded per turn)
└── USER.md     — user model (DYNAMIC, auto-loaded per turn)
```

### `prometheus.context.prompt_assembler` — updated
`src/prometheus/context/prompt_assembler.py`

```python
def _load_bootstrap_file(filename: str) -> str | None
    """Load a bootstrap file from ~/.prometheus/. Returns content or None."""

def _load_memory_and_user() -> str
    """Load MEMORY.md + USER.md via format_memory_for_prompt(). Returns formatted string."""

def build_runtime_system_prompt(
    *, cwd: str, config: dict | None = None,
    memory_content: str = "",       # if empty, auto-loads MEMORY.md + USER.md
    skills: list | None = None,
    task_state: str = "",
    loaded_skill_content: str = "",
) -> str
    # Assembly order:
    # STATIC:  SOUL.md → AGENTS.md → base prompt + environment
    # DYNAMIC: reasoning → skills → PROMETHEUS.md → MEMORY.md + USER.md → task → skill
    # Config key "bootstrap" controls load_soul / load_agents toggles
```

### `config/prometheus.yaml` — new section

```yaml
bootstrap:
  load_soul: true       # ~/.prometheus/SOUL.md
  load_agents: true     # ~/.prometheus/AGENTS.md
```

### Wiring into existing modules

| Existing module | How Sprint 17 connects |
|---|---|
| `context/prompt_assembler.py` (Sprint 1) | `build_runtime_system_prompt()` extended with `_load_bootstrap_file()` for SOUL.md + AGENTS.md in static section; `_load_memory_and_user()` auto-populates dynamic section |
| `memory/hermes_memory_tool.py` (Sprint 5) | `format_memory_for_prompt()` now called by `_load_memory_and_user()` — was previously defined but never invoked |
| `config/prometheus.yaml` (Sprint 0) | New `bootstrap:` block with `load_soul` and `load_agents` toggles |
| `scripts/daemon.py` (Sprint 6) | No changes — `build_runtime_system_prompt(config=config)` picks up bootstrap config automatically |
| `engine/__init__.py` (Sprint 1) | Lazy `__getattr__` for `AgentLoop`/`RunResult` to break circular import chain (`providers.base` ↔ `engine.__init__` ↔ `engine.agent_loop`) |

## Sprint 18: ANATOMY — Infrastructure Self-Awareness ✓

Prometheus had no awareness of its own hardware, loaded models, VRAM usage,
or how the Mini and 4090 are wired together. ANATOMY gives Prometheus a view
of its own body — hardware, engines, resources — and the ability to track
different project configurations across context switches.

### `prometheus.infra.anatomy`
`src/prometheus/infra/anatomy.py` — novel code

```python
@dataclass
class AnatomyState:
    hostname: str; platform: str; cpu: str
    ram_total_gb: float; ram_available_gb: float
    gpu_name: str | None; gpu_vram_total_mb: int | None
    gpu_vram_used_mb: int | None; gpu_vram_free_mb: int | None
    model_name: str | None; model_file: str | None; model_quantization: str | None
    inference_engine: str; inference_url: str; inference_features: list[str]
    vision_enabled: bool; whisper_model: str | None
    tailscale_ip: str | None; tailscale_peers: list[str]
    disk_total_gb: float; disk_free_gb: float; prometheus_data_size_mb: float
    scanned_at: str

class AnatomyScanner:
    def __init__(self, llama_cpp_url: str, ollama_url: str, inference_engine: str) -> None
    async def scan(self) -> AnatomyState        # full infra scan
    async def quick_scan(self) -> AnatomyState   # model + VRAM only
```

Detection methods: `_detect_gpu` (nvidia-smi), `_detect_model` (llama.cpp `/v1/models` + `/props` + `/slots`; Ollama `/api/tags`), `_detect_tailscale` (`tailscale status --json`), `_detect_disk` (`shutil.disk_usage`), `_check_cmdline_vision` (pgrep `--mmproj`).

### `prometheus.infra.anatomy_writer`
`src/prometheus/infra/anatomy_writer.py` — novel code

```python
class AnatomyWriter:
    def __init__(self, anatomy_path: Path | None = None) -> None
    def write(self, state: AnatomyState, project_summaries: list[dict]) -> str
    def update_active_section(self, state: AnatomyState) -> None
    def render_mermaid(self, state: AnatomyState) -> str
    def render_summary(self, state: AnatomyState, project_names: list[str]) -> str
```

### `prometheus.infra.project_configs`
`src/prometheus/infra/project_configs.py` — novel code

```python
@dataclass
class ModelSlot:
    name: str; role: str; engine: str; machine: str
    vram_estimate_gb: float; port: int; gguf_file: str | None; extra_flags: list[str]

@dataclass
class ProjectConfig:
    name: str; description: str; models: list[ModelSlot]; services: list[str]
    notes: str; last_used: str | None; active: bool

class ProjectConfigStore:
    def __init__(self, projects_dir: Path | None = None) -> None
    def list_projects(self) -> list[ProjectConfig]
    def get(self, name: str) -> ProjectConfig | None
    def get_active(self) -> ProjectConfig | None
    def save(self, config: ProjectConfig) -> None
    def activate(self, name: str) -> bool
    def summaries(self) -> list[dict]
```

Project configs stored as YAML in `~/.prometheus/anatomy/projects/*.yaml`.

### `prometheus.tools.builtin.anatomy`
`src/prometheus/tools/builtin/anatomy.py` — novel code

```python
class AnatomyTool(BaseTool):
    name = "anatomy"
    # Actions: scan, status, projects, switch, diagram, history
    async def execute(self, arguments: AnatomyInput, context: ToolExecutionContext) -> ToolResult

def set_anatomy_components(scanner, writer, project_store) -> None
```

### `prometheus.context.prompt_assembler` — updated
```python
def _load_anatomy_summary() -> str | None
    # Extracts Active Configuration section from ANATOMY.md
    # Injects into static section after AGENTS.md, before base prompt
```

### `config/prometheus.yaml` — new section
```yaml
anatomy:
  enabled: true
  scan_on_startup: true
  periodic_quick_scan: true
  quick_scan_interval_seconds: 300
  include_in_system_prompt: true
```

### Wiring into existing modules

| Existing module | How Sprint 18 connects |
|---|---|
| `context/prompt_assembler.py` (Sprint 17) | `_load_anatomy_summary()` extracts Active Configuration from ANATOMY.md into static section (after AGENTS.md, before base prompt) |
| `gateway/commands.py` (Sprint 6) | New `cmd_anatomy()` function; `/anatomy` added to `/help` listing |
| `gateway/telegram.py` (Sprint 6) | New `_cmd_anatomy` handler + `CommandHandler("anatomy", ...)` registration |
| `scripts/daemon.py` (Sprint 6) | `AnatomyScanner` + `AnatomyWriter` + `ProjectConfigStore` created at startup; `set_anatomy_components()` wires to tool; startup scan writes ANATOMY.md |
| `config/prometheus.yaml` (Sprint 0) | New `anatomy:` block controls scan-on-startup, periodic scan, and system prompt inclusion |

## Sprint 19: PROFILES — Agent Profiles ✓

Every Prometheus session loaded all tools, all bootstrap files, all subsystems.
On a 24K context window, this wastes ~20% before a single user message. Profiles
let you select a configuration that loads only what's needed.

### `prometheus.config.profiles`
`src/prometheus/config/profiles.py` — novel code

```python
@dataclass
class AgentProfile:
    name: str                    # "full", "coder", "research", "assistant", "minimal"
    description: str
    bootstrap_files: list[str]   # which .md files to load
    tools: list[str] | None      # tool names to include, None = all
    exclude_tools: list[str]     # tools to exclude (applied after include)
    subsystems: dict[str, bool]  # {"sentinel": True, "wiki": True, ...}
    max_tool_schemas: int | None # cap on tool schemas, None = no cap

class ProfileStore:
    def __init__(self, custom_dir: Path | None = None) -> None
    def get(self, name: str) -> AgentProfile | None
    def list_profiles(self) -> list[AgentProfile]
    def names(self) -> list[str]

def get_profile_store() -> ProfileStore
def filter_tools_by_profile(all_schemas: list[dict], profile: AgentProfile) -> list[dict]
```

Builtin profiles: `full` (all tools, all bootstrap), `coder` (9 tools, SOUL.md only),
`research` (9 read-only tools), `assistant` (7 tools, memory-rich), `minimal` (2 tools).
Custom profiles via YAML in `~/.prometheus/profiles/` override builtins with same name.

### `prometheus.tools.base.ToolRegistry` — updated
```python
class ToolRegistry:
    def schemas_for_names(self, names: list[str]) -> list[dict[str, Any]]
        # Return tool schemas for specific tool names (preserving order)
```

### `prometheus.context.prompt_assembler` — updated
```python
def build_runtime_system_prompt(
    *, cwd, config, memory_content, skills, task_state,
    loaded_skill_content,
    profile: AgentProfile | None = None,  # NEW — controls bootstrap file loading
) -> str
    # When profile is set, only bootstrap files in profile.bootstrap_files are loaded
    # When profile is None, falls back to legacy bootstrap config toggles
```

### `config/prometheus.yaml` — new section
```yaml
profiles:
  default: "full"
  custom_dir: "~/.prometheus/profiles"
```

### Wiring into existing modules

| Existing module | How Sprint 19 connects |
|---|---|
| `context/prompt_assembler.py` (Sprint 17) | `build_runtime_system_prompt()` gains `profile=` parameter; when set, profile controls which bootstrap files load instead of config toggles |
| `tools/base.py` (Sprint 2) | `ToolRegistry.schemas_for_names()` added for profile-filtered schema lists |
| `gateway/commands.py` (Sprint 6) | New `cmd_profile()` function; `/profile` added to `/help` listing |
| `gateway/telegram.py` (Sprint 6) | New `_cmd_profile` handler + `CommandHandler("profile", ...)` registration; stores `_active_profile_name` per adapter |
| `config/prometheus.yaml` (Sprint 0) | New `profiles:` block with default profile and custom directory |

## Sprint 20: LSP — Language Server Protocol Integration ✓

Prometheus used grep and text matching to understand code structure. LSP gives it
real compiler-grade intelligence — go-to-definition, find-references, type info,
diagnostics. Three-layer architecture modeled after OpenCode; the `context` action
(Claude Code's symbolContext concept) packages definition + references + type info
into one call instead of three round trips. The diagnostics hook auto-checks for
type errors after every file mutation — the model sees "your edit broke 3 type
checks" in the same turn and can fix immediately.

### `prometheus.lsp.languages`
`src/prometheus/lsp/languages.py` — novel code

```python
@dataclass
class LSPServerDef:
    language_id: str              # "python", "typescript", "go", "rust", "c"
    extensions: list[str]         # [".py", ".pyi"]
    command: list[str]            # ["pyright-langserver", "--stdio"]
    root_markers: list[str]      # ["pyproject.toml", ".git"]
    install_command: list[str] | None
    initialization_options: dict[str, Any]

BUILTIN_SERVERS: dict[str, LSPServerDef]  # python, typescript, go, rust, c
EXTENSION_TO_LANGUAGE: dict[str, str]     # .py → "python", .ts → "typescript", etc.

def find_project_root(filepath: Path, root_markers: list[str]) -> Path
def get_server_for_file(filepath, custom_servers=None) -> LSPServerDef | None
```

Built-in servers: `python` (pyright), `typescript`, `go` (gopls), `rust` (rust-analyzer), `c` (clangd).
Custom servers from `prometheus.yaml` `lsp.servers` override builtins or add new languages.

### `prometheus.lsp.client`
`src/prometheus/lsp/client.py` — novel code

```python
@dataclass
class Location:    # path, line, col (1-indexed)
@dataclass
class Diagnostic:  # path, line, col, severity (1=Error..4=Hint), message, source
@dataclass
class HoverInfo:   # contents (type/doc string)
@dataclass
class DocumentSymbol:  # name, kind, range, detail, children

class LSPClient:
    def __init__(self, server_def: LSPServerDef, project_root: Path) -> None
    async def start(self) -> None           # spawn process, initialize handshake
    async def stop(self) -> None            # shutdown + exit + kill
    async def did_open(self, filepath) -> None
    async def did_change(self, filepath) -> None
    async def did_close(self, filepath) -> None
    async def get_definition(self, filepath, line, col) -> list[Location]
    async def get_references(self, filepath, line, col) -> list[Location]
    async def get_hover(self, filepath, line, col) -> HoverInfo | None
    async def get_diagnostics(self, filepath) -> list[Diagnostic]
    async def get_document_symbols(self, filepath) -> list[DocumentSymbol]
    async def rename_symbol(self, filepath, line, col, new_name) -> dict
```

JSON-RPC 2.0 over stdin/stdout with Content-Length framing. Full document sync
(no incremental). Diagnostics cached from `textDocument/publishDiagnostics`
notifications. Async reader loop dispatches responses via `asyncio.Future`.

### `prometheus.lsp.orchestrator`
`src/prometheus/lsp/orchestrator.py` — novel code

```python
class LSPOrchestrator:
    def __init__(self, custom_servers: dict | None = None) -> None
    async def ensure_server(self, filepath) -> LSPClient | None  # lazy spawn
    async def get_definition(self, filepath, line, col) -> list[Location]
    async def get_references(self, filepath, line, col) -> list[Location]
    async def get_hover(self, filepath, line, col) -> HoverInfo | None
    async def get_diagnostics(self, filepath) -> list[Diagnostic]
    async def get_symbols(self, filepath) -> list[DocumentSymbol]
    async def rename(self, filepath, line, col, new_name) -> dict
    async def get_symbol_context(self, filepath, line, col) -> str  # THE power move
    async def notify_file_changed(self, filepath) -> None
    async def shutdown_all(self) -> None
```

Manages multiple concurrent LSP clients keyed by `language_id:project_root`.
Lazy spawning (server starts on first file access), broken server tracking
(failed servers blacklisted for the session), promise coalescing (concurrent
requests for the same server share one spawn task).

`get_symbol_context` fans out definition + references + hover concurrently
and packages everything into one formatted text block.

### `prometheus.tools.builtin.lsp`
`src/prometheus/tools/builtin/lsp.py` — novel code

```python
class LSPTool(BaseTool):
    name = "lsp"
    # Actions: definition, references, hover, diagnostics, symbols, rename, context
    # Input: action, file, line?, column?, symbol?, new_name?
    # is_read_only = True (except rename)

def set_lsp_orchestrator(orch) -> None  # module-level wiring
```

Single tool with 7 actions. The `context` action is the recommended default —
one call instead of three. Symbol name resolution: if `symbol` is given without
`line`/`column`, the file is searched for the first occurrence.

### `prometheus.hooks.lsp_diagnostics`
`src/prometheus/hooks/lsp_diagnostics.py` — novel code

```python
class LSPDiagnosticsHook:
    def __init__(self, orchestrator, delay_ms=500, enabled=True) -> None
    async def __call__(self, tool_name, tool_input, tool_result) -> ToolResultBlock
```

Post-result hook that fires after `write_file` and `edit_file`. Notifies the
LSP server of the change, waits for diagnostics to settle, and appends any
errors/warnings to the tool result text. The model sees type errors in the
same turn as the edit.

### `prometheus.engine.agent_loop` — updated
```python
@dataclass
class LoopContext:
    post_result_hooks: list[object] | None = None  # NEW — modify result after execution

class AgentLoop:
    def __init__(self, ..., post_result_hooks=None) -> None  # NEW param
```

Post-result hooks run after tool execution and the existing POST_TOOL_USE hook.
Each hook is an async callable `(tool_name, tool_input, tool_result) → tool_result`.

### `config/prometheus.yaml` — new section
```yaml
lsp:
  enabled: true
  auto_diagnostics: true
  diagnostics_delay_ms: 500
  auto_install: false
  servers: {}
```

### File tree
```
src/prometheus/lsp/
  __init__.py          — package exports
  languages.py         — server definitions, extension mapping, root detection
  client.py            — JSON-RPC client over stdin/stdout
  orchestrator.py      — lifecycle management, routing, symbol context
src/prometheus/tools/builtin/lsp.py    — LSPTool (7 actions)
src/prometheus/hooks/lsp_diagnostics.py — post-result diagnostics hook
tests/test_lsp_client.py               — 22 tests
tests/test_lsp_orchestrator.py         — 10 tests
tests/test_lsp_tool.py                 — 16 tests
tests/test_lsp_diagnostics_hook.py     — 11 tests
```

### Wiring into existing modules

| Existing module | How Sprint 20 connects |
|---|---|
| `engine/agent_loop.py` (Sprint 1) | `LoopContext` gains `post_result_hooks`; `_execute_tool_call` iterates hooks after tool execution to allow result modification |
| `config/profiles.py` (Sprint 19) | `"lsp"` added to coder profile's tool list |
| `tools/builtin/__init__.py` (Sprint 2) | `LSPTool` exported |
| `hooks/__init__.py` (Sprint 2) | `LSPDiagnosticsHook` exported |
| `scripts/daemon.py` (Sprint 6) | `LSPOrchestrator` created if `lsp.enabled`; `set_lsp_orchestrator()` wires to tool; `LSPDiagnosticsHook` registered as post-result hook; `orchestrator.shutdown_all()` in graceful shutdown |
| `config/prometheus.yaml` (Sprint 0) | New `lsp:` block controls enablement, auto-diagnostics, delay, and custom servers |

## Sprint 21: Cloud API Providers ✓

Prometheus was locked to local inference (llama.cpp, Ollama). This sprint
adds OpenAI, Anthropic, Gemini, and xAI as cloud API providers alongside
existing local providers, plus a provider registry, cost tracking, and
setup wizard extensions. No new pip dependencies — all use raw `httpx`.

### `prometheus.providers.openai_compat`
`src/prometheus/providers/openai_compat.py` — novel code

```python
class OpenAICompatProvider(ModelProvider):
    """Provider for any OpenAI-compatible chat completions API.
    Covers: OpenAI, Google Gemini, xAI Grok, vLLM, LiteLLM."""

    def __init__(
        self, base_url: str, api_key: str, model: str = "",
        default_max_tokens: int = 4096, timeout: float = 120.0,
    ) -> None

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
        # SSE streaming to /v1/chat/completions with Bearer auth
        # Handles tool_calls delta accumulation, usage extraction
        # Exponential backoff retry on 429/5xx
```

### `prometheus.providers.anthropic` — existing (Sprint 1 fallback)
`src/prometheus/providers/anthropic.py` — already implemented, now wired via registry

```python
class AnthropicProvider(ModelProvider):
    def __init__(
        self, api_key: str | None = None, model: str = "claude-sonnet-4-6",
        timeout: float = 120.0, prompt_caching: bool = False,
    ) -> None
```

### `prometheus.providers.registry`
`src/prometheus/providers/registry.py` — novel code

```python
class ProviderRegistry:
    @staticmethod
    def create(config: dict) -> ModelProvider
        # Factory: "openai"|"gemini"|"xai" → OpenAICompatProvider
        #          "anthropic" → AnthropicProvider
        #          "llama_cpp" → LlamaCppProvider
        #          "ollama" → OllamaProvider, "stub" → StubProvider
        # API keys resolved via config["api_key_env"] → os.environ

    @staticmethod
    def is_cloud(provider_name: str) -> bool
        # True for openai, anthropic, gemini, xai

    @staticmethod
    def list_providers() -> list[str]
        # All 7 supported provider names
```

### `prometheus.adapter.formatter.PassthroughFormatter`
`src/prometheus/adapter/formatter.py` — added to existing file

```python
class PassthroughFormatter(ModelPromptFormatter):
    """For cloud API models with native tool calling.
    format_tools/format_system_prompt return inputs unchanged."""
```

### `prometheus.telemetry.cost`
`src/prometheus/telemetry/cost.py` — novel code

```python
PRICING: dict[str, tuple[float, float]]  # model → (input $/1M, output $/1M)
    # gpt-4o, gpt-4o-mini, o3-mini, claude-sonnet-4-6, claude-haiku-4-5,
    # gemini-2.5-flash, gemini-2.5-pro, grok-3, grok-3-mini

class CostTracker:
    def record(self, model: str, input_tokens: int, output_tokens: int) -> float
        # Returns cost in USD. Prefix-matches model names for versioned IDs.
    def report(self) -> str
        # "Session cost: $0.0342 (14,230 input + 2,891 output tokens)"
    def to_dict(self) -> dict[str, Any]
    # Properties: total_cost, total_input_tokens, total_output_tokens, total_tokens
```

### `prometheus.__main__` — updated
```python
def create_adapter(model_cfg: dict) -> ModelAdapter
    # "anthropic" → AnthropicFormatter, strictness=NONE
    # "openai"|"gemini"|"xai" → PassthroughFormatter, strictness=NONE
    # "llama_cpp"|"ollama" + "gemma" in model → GemmaFormatter, strictness=MEDIUM
    # "llama_cpp"|"ollama" + other model → QwenFormatter, strictness=MEDIUM
```

### `prometheus.setup_wizard` — updated
```python
class SetupWizard:
    def _step_provider(self) -> None
        # 7 options: llama.cpp, Ollama, OpenAI, Anthropic, Gemini, xAI, none
    def _step_cloud_provider(self, provider: str) -> None
        # Collects API key/env var, model selection with pricing, sets base_url
```

### `config/prometheus.yaml` — updated
```yaml
model:
  # Commented examples for all 4 cloud providers:
  # provider: "openai"  / "anthropic" / "gemini" / "xai"
  # api_key_env: "OPENAI_API_KEY"  # reads from env var, never stored
  # model: "gpt-4o"
```

### File tree
```
src/prometheus/providers/
  openai_compat.py       — OpenAICompatProvider (OpenAI, Gemini, xAI)
  anthropic.py           — AnthropicProvider (existing, now registered)
  registry.py            — ProviderRegistry factory
src/prometheus/adapter/
  formatter.py           — PassthroughFormatter added
src/prometheus/telemetry/
  cost.py                — CostTracker + PRICING table
tests/test_cloud_providers.py  — 37 tests
```

### Wiring into existing modules

| Existing module | How Sprint 21 connects |
|---|---|
| `providers/base.py` (Sprint 1) | `OpenAICompatProvider` implements `ModelProvider` ABC |
| `providers/stub.py` (Sprint 1) | `OpenAICompatProvider` reuses `_build_openai_messages` and `_parse_assistant_message` |
| `adapter/__init__.py` (Sprint 3) | `create_adapter()` routes cloud providers to `PassthroughFormatter` with `strictness=NONE` |
| `scripts/daemon.py` (Sprint 6) | Provider instantiation replaced with `ProviderRegistry.create(model_config)`; `CostTracker` created for cloud providers and attached to Telegram adapter |
| `gateway/telegram.py` (Sprint 6) | `TelegramAdapter` gains `cost_tracker` attribute; `/status` handler shows session cost when set |
| `gateway/commands.py` (Sprint 6) | `cmd_status()` gains optional `cost_tracker` parameter |
| `setup_wizard.py` (Sprint 6) | `_step_provider()` expanded from 3 to 7 options; `_step_cloud_provider()` added; `_apply_wizard_fields()` handles `api_key_env` and provider-specific context limits; smoke test supports Anthropic and OpenAI-compat APIs |
| `config/prometheus.yaml` (Sprint 0) | Commented examples for all 4 cloud providers added under `model:` |

## Sprint 22: MIGRATE — Hermes/OpenClaw Migration Tool ✓

Users coming from Hermes Agent or OpenClaw have identity files, memories,
skills, and config they don't want to lose. This sprint adds a CLI migration
tool that runs pre-agent (no model, no API keys) and a setup wizard
auto-detection hook.

### `prometheus.cli.migrate`
`src/prometheus/cli/migrate.py` — novel code

```python
@dataclass
class MigrationItem:
    category: str          # "identity", "memory", "skills", "config", "secrets"
    source_path: Path
    dest_path: Path
    description: str
    action: str = "copy"   # "copy", "remap", "skip", "manual"
    status: str = "pending"  # "pending", "done", "skipped", "conflict", "error"
    conflict: str | None = None

@dataclass
class MigrationReport:
    source: str            # "hermes" or "openclaw"
    source_path: Path
    timestamp: str
    items: list[MigrationItem]
    # Properties: migrated, skipped, errors, manual

@dataclass
class MigrationOptions:
    source: str            # "hermes" or "openclaw"
    source_path: Path
    dest_path: Path | None = None
    dry_run: bool = False
    overwrite: bool = False
    preset: str = "user-data"      # "full" includes secrets (as manual items)
    skill_conflict: str = "skip"   # "skip", "overwrite", "rename"

def detect_sources() -> dict[str, Path]
    # Finds ~/.hermes (config.yaml), ~/.openclaw|.clawdbot|.moldbot (openclaw.json)

class HermesMigrator(_BaseMigrator):
    def scan(self) -> MigrationReport
        # Scans: SOUL.md, AGENTS.md, memories/{MEMORY,USER}.md, memories/daily/,
        #        skills/, config.yaml (remap), cron/, .env (manual)
    def execute(self) -> MigrationReport
        # Copies files, remaps config, archives overflow, writes report

class OpenClawMigrator(_BaseMigrator):
    def scan(self) -> MigrationReport
        # Finds workspace via openclaw.json agents.*.workspace or ~/clawd/
        # Scans workspace: SOUL.md, MEMORY.md, USER.md, AGENTS.md, skills/, memory/
        # Scans config: openclaw.json (remap)
    def execute(self) -> MigrationReport

def run_migration(args) -> bool
    # CLI entry point — detect, scan, confirm, execute, report
```

Key behaviors:
- Memory overflow: MEMORY.md > 12K chars trimmed (most-recent kept), overflow archived
- Config remap: Hermes YAML keys mapped to Prometheus YAML keys; provider names mapped
- Secrets: never auto-copied, only printed as guidance (`action="manual"`)
- Overwrite: archives original to `~/.prometheus/migration/<source>/<timestamp>/archive/`

### CLI (`__main__.py` — updated)
```
python -m prometheus migrate --from hermes
python -m prometheus migrate --from openclaw
python -m prometheus migrate --from hermes --dry-run
python -m prometheus migrate --from openclaw --source ~/.clawdbot
python -m prometheus migrate --from hermes --overwrite --preset full
```

Flags: `--from` (required), `--dry-run`, `--source`, `--overwrite`,
`--preset` (user-data|full), `--skill-conflict` (skip|overwrite|rename), `--yes`

### `prometheus.setup_wizard` — updated
```python
class SetupWizard:
    def _offer_migration(self) -> None
        # Called at start of run() when no existing config
        # Auto-detects Hermes/OpenClaw, prompts user, runs migration
```

### File tree
```
src/prometheus/cli/
  __init__.py
  migrate.py             — detect_sources, HermesMigrator, OpenClawMigrator
tests/test_migrate.py    — 30 tests
```

### Wiring into existing modules

| Existing module | How Sprint 22 connects |
|---|---|
| `__main__.py` (Sprint 0) | `migrate` subcommand added to argparse; dispatches to `run_migration()` before any agent/model setup |
| `setup_wizard.py` (Sprint 6) | `_offer_migration()` called at start of `run()` when no config exists; uses `detect_sources()` + `run_migration()` |
| `memory/hermes_memory_tool.py` (Sprint 5) | Memory char limits (`_MEMORY_MAX_CHARS=12000`, `_USER_MAX_CHARS=8000`) matched by overflow trimming |
| `config/prometheus.yaml` (Sprint 0) | Config remap target — Hermes/OpenClaw keys mapped to Prometheus keys |

---

## Sprint 23: CLEAN-SLATE + VISION-DETECT

### CLEAN-SLATE — identity templates for shareable repo

```python
# src/prometheus/cli/generate_identity.py

def detect_hardware() -> dict
    # → {hostname, os, arch, cpu, ram_gb, gpu, has_gpu}

def render_soul_md(owner_name, hardware, hardware_layout="single",
                   gpu_machine_name=None, brain_machine_name=None,
                   owner_description="", vision_available=None) -> str
    # Fills {{OWNER_NAME}}, {{HARDWARE_SECTION}}, {{VISION_LINE}}, etc.

def render_agents_md() -> str

def generate_identity_files(owner_name, hardware, ..., dest=None) -> dict[str, str]
    # Creates SOUL.md, AGENTS.md; creates (never overwrites) MEMORY.md, USER.md

# setup_wizard.py additions
class SetupWizard:
    def _setup_identity(self) -> None    # interactive: name, desc, layout → generate
```

### VISION-DETECT — detect vision at startup, graceful photo fallback

```python
# providers/base.py addition
class ModelProvider(ABC):
    supports_vision: bool = False
    async def detect_vision(self) -> bool   # default False, override in subclasses

# providers/llama_cpp.py addition
class LlamaCppProvider(ModelProvider):
    async def detect_vision(self) -> bool
        # GET /props → checks multimodal flag (top-level + nested)
        # Graceful on connection error — returns False, never crashes

# setup_wizard.py addition
class SetupWizard:
    def _check_vision_hint(self) -> None    # probes provider after smoke test

# __main__.py addition
identity_parser  # subcommand: identity --show | --regenerate
```

### File tree
```
templates/
  SOUL.md.template           — {{OWNER_NAME}}, {{HARDWARE_SECTION}}, etc.
  AGENTS.md.template         — generic, no placeholders
config/
  prometheus.yaml.default    — reference config, zero secrets
src/prometheus/cli/
  generate_identity.py       — hardware detect + template render + file gen
tests/
  test_clean_slate.py        — 18 tests
  test_vision_detect.py      — 7 tests
  test_wiring.py             — +9 integration tests (CleanSlate + VisionDetect classes)
```

### Wiring into existing modules

| Existing module | How Sprint 23 connects |
|---|---|
| `providers/base.py` (Sprint 1) | `supports_vision: bool` + `detect_vision()` added to `ModelProvider` ABC |
| `providers/llama_cpp.py` (Sprint 1) | `detect_vision()` checks `/props` endpoint for multimodal flag |
| `setup_wizard.py` (Sprint 6) | `_setup_identity()` before provider step; `_check_vision_hint()` after smoke test |
| `__main__.py` (Sprint 0) | `identity` subcommand with `--show` and `--regenerate` flags |
| `scripts/daemon.py` (Sprint 6) | Calls `detect_vision()` after `detect_loaded_model()`, logs hint for vision-capable models |

---

## Sprint 24: DOCTOR — Model Registry + Diagnostics

### Model Registry

```yaml
# config/model_registry.yaml
# Maps model families → capabilities + match patterns
# 9 families: gemma-4, gemma-3, qwen-3, qwen-2.5, llama-3, mistral, deepseek, phi, command-r

models:
  gemma-4:
    display_name: "Google Gemma 4"
    match_patterns: ["gemma-4", "gemma_4", "gemma4"]
    capabilities:
      vision:    { supported: true,  requires: "mmproj", setup_hint: "..." }
      function_calling: { supported: true }
      streaming: { supported: true }
    context_sizes: [8192, 32768, 131072]
```

```python
# src/prometheus/infra/doctor.py

def match_model(model_name: str, registry: dict) -> dict | None
    # Case-insensitive substring match against match_patterns
    # Returns the matched family dict or None
```

### Doctor class

```python
# src/prometheus/infra/doctor.py

@dataclass
class DiagnosticCheck:
    name: str          # "Vision", "GPU", "Python"
    category: str      # "platform" | "connectivity" | "model" | "resources"
    status: str        # "ok" | "warning" | "error" | "info"
    message: str       # Human-readable result
    fix: str | None    # Actionable remediation (for warnings/errors)

@dataclass
class DiagnosticReport:
    model_name: str | None
    model_family: str | None
    checks: list[DiagnosticCheck]
    timestamp: str
    CATEGORY_ORDER = ["platform", "connectivity", "model", "resources"]

    @property has_warnings -> bool
    @property has_errors -> bool
    def checks_by_category() -> dict[str, list[DiagnosticCheck]]

class Doctor:
    def __init__(self, config: dict | None = None)
        # Loads model_registry.yaml from config dir
    async def diagnose(self, state: AnatomyState) -> DiagnosticReport
        # Runs all checks grouped by category:
        #   Platform:     python_version, uv, config_valid, data_dir, bootstrap_files, dependencies
        #   Connectivity: inference, telegram_token, tailscale
        #   Model:        model_loaded, vision, function_calling
        #   Resources:    gpu, disk, whisper
```

### /doctor command

```python
# src/prometheus/gateway/commands.py

async def cmd_doctor(config: dict | None = None) -> str
    # Scans via AnatomyScanner, runs Doctor.diagnose(), formats report
    # Output grouped by category with section headers:
    #   -- Platform --
    #   -- Connectivity --
    #   -- Model --
    #   -- Resources --
    # Status icons: ok, warning, error, info
    # Fix instructions shown for warnings/errors
    # Summary line: error count / warning count / all passed
```

### RetryEngine escalation fix

```python
# src/prometheus/adapter/retry.py

class RetryAction(str, Enum):
    RETRY = "RETRY"
    ABORT = "ABORT"
    ESCALATE = "ESCALATE"   # NEW — escalate to stronger model via router

class RetryEngine:
    def __init__(self, max_retries: int = 3, router: Any = None)
        # router: optional ModelRouter; if provided and escalation_enabled,
        # returns ESCALATE instead of ABORT when retries exhausted
```

### File tree
```
config/
  model_registry.yaml          — 9 model families, capabilities, match patterns
  prometheus.yaml               — +doctor: section (startup_check, registry_file)
src/prometheus/infra/
  doctor.py                    — Doctor class, DiagnosticCheck/Report, match_model()
src/prometheus/gateway/
  commands.py                  — +cmd_doctor() formatter, +/doctor in cmd_help()
  telegram.py                  — +_cmd_doctor handler, +prometheus_config param, +BotCommand("doctor")
src/prometheus/adapter/
  retry.py                     — +RetryAction.ESCALATE, +router kwarg
scripts/
  daemon.py                   — +prometheus_config passthrough, +startup doctor check
tests/
  test_doctor.py               — 59 tests (registry matching, all checks, categories, formatting)
  test_wiring.py               — +integration tests (DoctorWiring class)
```

### Wiring into existing modules

| Existing module | How Sprint 24 connects |
|---|---|
| `infra/anatomy.py` (Sprint 18) | Doctor.diagnose() takes AnatomyState from scanner.scan() |
| `gateway/telegram.py` (Sprint 6) | +`prometheus_config` constructor param, `_cmd_doctor` handler, BotCommand menu |
| `gateway/commands.py` (Sprint 6) | +`cmd_doctor()`, `/doctor` added to `cmd_help()` listing |
| `scripts/daemon.py` (Sprint 6) | Passes full config to TelegramAdapter; runs startup doctor check after anatomy scan |
| `adapter/retry.py` (Sprint 3) | +`ESCALATE` action, +`router` kwarg — escalates to stronger model when retries exhausted |
| `config/prometheus.yaml` (Sprint 0) | +`doctor:` section with `startup_check` and `registry_file` |

---

## Sprint 25: Tool Calling Middle Layer

### Feature 1: Deferred Tool Loading + ToolSearchTool

```python
# src/prometheus/tools/tool_search.py
class ToolSearchTool(BaseTool):
    name = "tool_search"
    # search: fuzzy match query against names + descriptions (top 5)
    # select: exact name lookup, return full schema
    input_model = ToolSearchInput  # query: str, action: "search" | "select"
    def set_registry(registry: ToolRegistry) -> None  # inject after construction
```

```yaml
# config/prometheus.yaml
tools:
  deferred_loading:
    enabled: false          # all schemas in prompt by default
    always_loaded: [bash, read_file, write_file, edit_file, grep, glob, tool_search]
    mcp_always_deferred: true
    search_mcp: true
```

- `DynamicToolLoader` extended with `deferred_config` — delegates to `always_loaded` when enabled, falls back to keyword matching when disabled
- Wired into `LoopContext.tool_loader` → `run_loop()` uses it for `tool_schema`
- Lucky guess telemetry: if model calls a deferred tool, it executes + logs `error_type="lucky_guess"`

### Feature 2: Cross-Result Token Budget

```yaml
context:
  tool_results_turn_budget: 8000  # total budget for ALL results in one turn
```

- `_apply_cross_result_budget()` in `agent_loop.py` — runs after per-result truncation, before conversation injection
- Proportionally truncates results to fit budget; read-only tools truncated first, mutating tools preserved

### Feature 3: Tool Result MicroCompaction

```yaml
context:
  microcompact_after_turns: 3
  microcompact_keep_chars: 200
  microcompact_keep_chars_no_lcm: 500  # if LCM hasn't ingested
```

- `_microcompact_old_results()` in `agent_loop.py` — runs at start of each turn, before LCM/compression
- Replaces old ToolResultBlock content with `[microcompacted] {first_line}...\n{summary}`
- Skips error results, already-pruned results, and results in the fresh window

### Feature 4: Telemetry Dashboard

```python
# src/prometheus/telemetry/dashboard.py
class ToolDashboard:
    def get_stats(hours: int = 24) -> dict
    # Returns: success_rate_by_tool, most_called, avg_latency_by_tool,
    #          circuit_breaker_trips, lucky_guesses, adapter_repairs
```

- `/tools` Telegram command calls `ToolDashboard.get_stats()` and formats for chat
- Presentation-agnostic: returns dict, can be served via Beacon `/api/tools/stats`

### Feature 5: Adaptive Per-Tool Strictness

```yaml
adapter:
  adaptive_strictness: true
  strictness_window: 100
  strictness_threshold: 0.8
```

- `ModelAdapter` tracks per-tool success history in `_tool_call_history`
- When success rate drops below threshold (10+ calls), bumps strictness: NONE→MEDIUM→STRICT
- Precedence: manual override > adaptive > cloud auto-config > model default
- `set_tool_strictness()` for manual per-tool overrides

### Feature 6: Structured Error Feedback

```python
# Added to BaseTool:
example_call: dict[str, Any] | None = None
# Populated on 6 core tools (bash, file_read, file_write, file_edit, grep, glob)
```

- `_build_structured_error()` in `validator.py` — includes: what failed, available tools, expected format, example call
- Called from `validate()` and `repair()` failure paths

### Adapter: Model Capability-Based Configuration

`create_adapter()` now checks `model_registry.yaml` capabilities:
- `function_calling.supported: true` + `requires: null` → PassthroughFormatter/NONE (server handles it)
- Fallback: model-specific formatter + MEDIUM strictness
- Gemma 4, Qwen on llama.cpp → skip adapter (native tool calling)

### Wiring Summary

| Existing module | How this sprint connects |
|---|---|
| `engine/agent_loop.py` | +`tool_loader`, +`_apply_cross_result_budget()`, +`_microcompact_old_results()`, +lucky_guess telemetry |
| `adapter/__init__.py` | +`adaptive_strictness`, +`record_tool_call()`, +`get_effective_strictness()`, +`set_tool_strictness()` |
| `adapter/validator.py` | +`_build_structured_error()` — structured errors with examples |
| `tools/base.py` | +`example_call` class attribute on BaseTool |
| `context/dynamic_tools.py` | +`deferred_config`, +`_deferred_schemas()`, +`deferred_count` |
| `tools/tool_search.py` (new) | ToolSearchTool — fuzzy search + exact select against registry |
| `telemetry/dashboard.py` (new) | ToolDashboard — structured stats from telemetry SQLite |
| `gateway/telegram.py` | +`/tools` command handler |
| `__main__.py` | +`_has_native_tool_calling()` — model_registry.yaml capability check |
| `config/prometheus.yaml` | +`tools.deferred_loading`, +`adapter`, +`context.tool_results_turn_budget`, +microcompact settings |
