"""Prometheus CLI entry point.

Provides interactive and one-shot modes for running the agent loop,
plus a ``daemon`` subcommand that delegates to ``scripts/daemon.py``.

Usage:
    prometheus                          # interactive REPL
    prometheus --once "List files"      # single query, then exit
    prometheus daemon                   # start always-on daemon
    prometheus daemon --telegram-only   # daemon with Telegram only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from prometheus import __version__
from prometheus.config.paths import get_config_dir, get_data_dir, get_logs_dir, get_wiki_root
from prometheus.engine.agent_loop import AgentLoop, RunResult, run_loop, LoopContext
from prometheus.config.shipped_defaults import resolve_max_tool_iterations, resolve_max_tool_iterations_cloud
from prometheus.engine.messages import ConversationMessage
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ModelProvider
from prometheus.engine.fallback import build_fallback_target

log = logging.getLogger("prometheus")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROMETHEUS_YAML = Path(__file__).resolve().parents[2] / "config" / "prometheus.yaml"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load prometheus.yaml with env var overrides applied.

    Config file search order (also documented in the README and in
    config/prometheus.yaml.default):

    1. an explicit ``--config`` path
    2. the repo-local ``config/prometheus.yaml`` (checkout installs)
    3. ``$PROMETHEUS_CONFIG_DIR/prometheus.yaml`` — default
       ``~/.prometheus/prometheus.yaml`` (pip installs; written by
       ``prometheus setup``)

    Value precedence within the loaded file:
    env vars > secret files > YAML > defaults.
    """
    from prometheus.config.env_override import apply_env_overrides

    path = Path(config_path) if config_path else _PROMETHEUS_YAML
    if not path.exists():
        alt = get_config_dir() / "prometheus.yaml"
        if alt.exists():
            path = alt
        else:
            log.debug("No config file found — using defaults")
            return apply_env_overrides({})
    with path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return apply_env_overrides(config)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _detect_model_or_fallback(base_url: str, config_model: str) -> str:
    """Query /v1/models to discover the loaded model; fall back to config value."""
    import httpx as _httpx
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        resp = _httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            detected = models[0].get("id")
            if detected:
                log.info("Detected loaded model: %s", detected)
                return detected
    except Exception as exc:
        log.warning("Could not detect model from %s: %s", url, exc)
    log.info("Using model name from config: %s", config_model)
    return config_model


def create_provider(model_cfg: dict[str, Any]) -> tuple[ModelProvider, str]:
    """Instantiate the model provider from config.  Returns (provider, model_name)."""
    provider_name = model_cfg.get("provider", "llama_cpp")
    model_name = model_cfg.get("model", "qwen3.5-32b")
    base_url = model_cfg.get("base_url", "http://localhost:8080")

    if provider_name == "llama_cpp":
        from prometheus.providers.llama_cpp import LlamaCppProvider
        return LlamaCppProvider(base_url=base_url), model_name

    if provider_name == "ollama":
        from prometheus.providers.ollama import OllamaProvider
        url = model_cfg.get("fallback_url", "http://localhost:11434")
        return OllamaProvider(base_url=url), model_name

    if provider_name == "anthropic":
        from prometheus.providers.anthropic import AnthropicProvider
        return AnthropicProvider(), model_name

    # Fallback — treat as llama_cpp-compatible
    from prometheus.providers.llama_cpp import LlamaCppProvider
    log.warning("Unknown provider %r — falling back to llama_cpp", provider_name)
    return LlamaCppProvider(base_url=base_url), model_name


# ---------------------------------------------------------------------------
# Tool registry factory
# ---------------------------------------------------------------------------

def create_tool_registry(security_cfg: dict[str, Any], security_gate=None) -> Any:
    """Build the default tool registry with all builtin tools."""
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin import (
        AgentTool,
        AskUserTool,
        AuditQueryTool,
        BashTool,
        DashboardTool,
        DownloadFileTool,
        FileEditTool,
        FileReadTool,
        FileWriteTool,
        GlobTool,
        GrepTool,
        LCMDescribeTool,
        LCMExpandTool,
        LCMExpandQueryTool,
        LCMGrepTool,
        MessageTool,
        NotebookEditTool,
        SentinelStatusTool,
        TTSTool,
        WebFetchTool,
        WebSearchTool,
        YouTubeTranscriptTool,
        WikiCompileTool,
        WikiLintTool,
        WikiQueryTool,
    )
    from prometheus.tools.builtin.cron_create import CronCreateTool
    from prometheus.tools.builtin.cron_delete import CronDeleteTool
    from prometheus.tools.builtin.cron_list import CronListTool

    from prometheus.config.shipped_defaults import (
        resolve_denied_paths, resolve_workspace_root)
    workspace = resolve_workspace_root(security_cfg)
    # Same injection shape as BashTool's workspace: the tools get a
    # RESOLVED value, never the config. grep/glob prune denied paths
    # out of their results — the gate refuses a denied ROOT, this
    # covers a legitimate root that CONTAINS one.
    #
    # ⚠ THE SECOND READER. This one read the raw config while the gate read
    # the resolver, so on a config omitting the key the gate would have been
    # fixed and grep/glob pruning would still have seen []. Two readers of one
    # absence-hostile key, and fixing one is worse than fixing neither —
    # it makes the boundary look repaired while half of it is still inert.
    _denied = resolve_denied_paths(security_cfg)
    registry = ToolRegistry()

    # Funnel (honest-async-promises Layer d): register bash and task_create FIRST
    # and adjacent, so the model always sees task_create right beside bash and
    # routes work that outlives the turn to the managed-task path instead of
    # `nohup … &`. task_create stays on the fault-tolerant try_register (it needs
    # the task manager) so a missing optional dep degrades gracefully rather than
    # breaking the whole registry build.
    from prometheus.tools.registration import try_register
    # security.bash_confinement gates the kernel floor for bash. Default
    # "off": Prometheus runs on hosts without AppArmor, and defaulting to
    # "required" would refuse every bash call on them. That means the floor
    # is NOT in force until an operator turns it on — stated plainly here
    # because a security control that looks default-on and isn't is the
    # false assurance this whole line of work exists to remove.
    registry.register(BashTool(
        workspace=workspace,
        confinement=security_cfg.get("bash_confinement", "off"),
    ))
    try_register(registry, "TaskCreateTool",
                 "prometheus.tools.builtin.task_create", "TaskCreateTool")

    for tool in [
        # Core file/shell tools
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GrepTool(denied_paths=_denied),
        GlobTool(denied_paths=_denied),
        # Cron
        CronCreateTool(),
        CronDeleteTool(),
        CronListTool(),
        # LCM (long-context memory)
        LCMDescribeTool(),
        LCMExpandTool(),
        LCMGrepTool(),
        LCMExpandQueryTool(),
        # Web + messaging
        WebSearchTool(),
        WebFetchTool(),
        YouTubeTranscriptTool(),
        DownloadFileTool(),
        MessageTool(),
        TTSTool(),
        # Visualization
        DashboardTool(),
        NotebookEditTool(),
        # Agent delegation
        AgentTool(),
        AskUserTool(),
        # Wiki + SENTINEL
        WikiCompileTool(),
        WikiQueryTool(),
        WikiLintTool(),
        SentinelStatusTool(),
    ]:
        registry.register(tool)

    # ToolSearchTool — always loaded, enables deferred tool loading + skill search
    from prometheus.tools.tool_search import ToolSearchTool
    ts = ToolSearchTool()
    ts.set_registry(registry)
    try:
        from prometheus.skills.loader import load_skill_registry
        ts.set_skill_registry(load_skill_registry())
    except Exception as exc:
        # Daemon-reachable (daemon.py delegates here): a corrupt skill file or
        # import error must NOT silently leave skill search dead forever — the
        # MEMORY.md-at-0-bytes failure shape. Log loudly (audit H8).
        log.warning("skill registry load failed — skill search disabled: %s",
                    exc, exc_info=True)
    registry.register(ts)

    # Sprint 11: Audit query tool (requires audit logger from security gate)
    if security_gate and hasattr(security_gate, '_audit') and security_gate._audit:
        registry.register(AuditQueryTool(security_gate._audit))

    # ------------------------------------------------------------------
    # Per-tool fault-tolerant registration (Phase 2 — orphan-tool fix).
    #
    # Previously, several tool groups were wrapped in ``try/except: pass``
    # blocks. A single import failure silently disabled the whole group
    # with no log line and no /health signal. See
    # docs/audits/ORPHAN-TOOLS-AUDIT.md (Phase 1).
    #
    # ``try_register`` from prometheus.tools.registration:
    #   - imports + instantiates + registers each tool independently
    #   - logs INFO on success, WARN with traceback on failure
    #   - writes a row to telemetry.subsystem_runs (subsystem="tool_registration",
    #     operation=<tool_name>) so /health can surface failures
    #   - never raises — the caller below stays linear and readable
    # ------------------------------------------------------------------
    from prometheus.tools.registration import try_register

    # GRAFT-SYMBIOTE Session A: GitHub research + scout/harvest/graft tools.
    # The SYMBIOTE tools route through prometheus.symbiote.get_coordinator(),
    # which the daemon sets at startup if symbiote.enabled is true.
    def _make_github_search_tool():
        from prometheus.symbiote.github_search import GitHubSearchTool, GitHubClient
        return GitHubSearchTool(client=GitHubClient.from_config(None))

    try_register(registry, "GitHubSearchTool",
                 "prometheus.symbiote.github_search", "GitHubSearchTool",
                 factory=_make_github_search_tool)
    try_register(registry, "SymbioteScoutTool",
                 "prometheus.tools.builtin.symbiote_scout", "SymbioteScoutTool")
    try_register(registry, "SymbioteHarvestTool",
                 "prometheus.tools.builtin.symbiote_harvest", "SymbioteHarvestTool")
    try_register(registry, "SymbioteGraftTool",
                 "prometheus.tools.builtin.symbiote_graft", "SymbioteGraftTool")
    try_register(registry, "SymbioteStatusTool",
                 "prometheus.tools.builtin.symbiote_status", "SymbioteStatusTool")

    # Optional tools — don't fail if deps missing
    try_register(registry, "SkillTool",
                 "prometheus.tools.builtin.skill", "SkillTool")
    # Sprint 4 A4: MemoryTool was defined since the initial commit but never
    # registered — root cause of MEMORY.md / USER.md sitting at 0 bytes for
    # weeks per Phase 1 baseline. format_memory_for_prompt() reads these
    # files into every system prompt; without a tool, the agent had no way
    # to write them. See docs/audits/SILENT-FAILURE-AUDIT.md.
    try_register(registry, "MemoryTool",
                 "prometheus.memory.hermes_memory_tool", "MemoryTool")
    try_register(registry, "TodoWriteTool",
                 "prometheus.tools.builtin.todo_write", "TodoWriteTool")

    # Brain vault (~/brain-vault) — READ ONLY. Registered via try_register so an
    # absent or broken vault surfaces as a WARN + subsystem_runs row rather
    # than a quietly missing capability. Distinct from WikiQueryTool, which
    # reads the Prometheus wiki: different root, different corpus.
    try_register(registry, "VaultSearchTool",
                 "prometheus.tools.builtin.vault", "VaultSearchTool")
    try_register(registry, "VaultReadTool",
                 "prometheus.tools.builtin.vault", "VaultReadTool")

    # AnatomyTool — daemon wires ``set_anatomy_components`` at startup
    # (scripts/daemon.py:550); execute() degrades gracefully when the
    # scanner isn't initialised (e.g. CLI mode). Audit B3 / Phase 1 orphan.
    try_register(registry, "AnatomyTool",
                 "prometheus.tools.builtin.anatomy", "AnatomyTool")

    # Browser — requires optional playwright dependency
    try_register(registry, "BrowserTool",
                 "prometheus.tools.builtin.browser", "BrowserTool")

    # Session tools — require task manager
    try_register(registry, "SessionsListTool",
                 "prometheus.tools.builtin.sessions_list", "SessionsListTool")
    try_register(registry, "SessionsSendTool",
                 "prometheus.tools.builtin.sessions_send", "SessionsSendTool")
    try_register(registry, "SessionsSpawnTool",
                 "prometheus.tools.builtin.sessions_spawn", "SessionsSpawnTool")

    # Task tools — require task manager. (task_create is registered early, beside
    # bash, as the long-running-work funnel — see top of create_tool_registry.)
    try_register(registry, "TaskGetTool",
                 "prometheus.tools.builtin.task_get", "TaskGetTool")
    try_register(registry, "TaskListTool",
                 "prometheus.tools.builtin.task_list", "TaskListTool")
    try_register(registry, "TaskUpdateTool",
                 "prometheus.tools.builtin.task_update", "TaskUpdateTool")
    try_register(registry, "TaskStopTool",
                 "prometheus.tools.builtin.task_stop", "TaskStopTool")
    try_register(registry, "TaskOutputTool",
                 "prometheus.tools.builtin.task_output", "TaskOutputTool")

    # Image generation — free Pollinations.ai endpoint, no API key
    # (backend=dashscope adds paid WAN 2.5, opt-in, DASHSCOPE_API_KEY)
    try_register(registry, "ImageGenerateTool",
                 "prometheus.tools.builtin.image_generate", "ImageGenerateTool")

    # Video generation — Kling 3.0 (CLOUD EXPANSION 2026-07). PAID +
    # dormant-until-keyed: registration is unconditional (same pattern as
    # image_generate — this is the single registry both the CLI and the
    # daemon build via create_tool_registry); without KLING_ACCESS_KEY +
    # KLING_SECRET_KEY the tool returns an actionable error, never bills.
    try_register(registry, "KlingVideoTool",
                 "prometheus.tools.builtin.video_generate", "KlingVideoTool")

    return registry


# ---------------------------------------------------------------------------
# Adapter + Security
# ---------------------------------------------------------------------------

def _has_native_tool_calling(model_name: str) -> bool:
    """Check model_registry.yaml for native function_calling capability."""
    import yaml
    from pathlib import Path

    registry_path = Path(__file__).resolve().parents[2] / "config" / "model_registry.yaml"
    if not registry_path.exists():
        return False
    try:
        data = yaml.safe_load(registry_path.read_text())
        models = data.get("models", {})
        name_lower = model_name.lower()
        for _key, meta in models.items():
            patterns = meta.get("match_patterns", [])
            if any(p.lower() in name_lower for p in patterns):
                fc = meta.get("capabilities", {}).get("function_calling", {})
                return fc.get("supported", False) and fc.get("requires") is None
    except Exception:
        pass
    return False


def _get_adapter_tier(provider_name: str, model_name: str) -> str:
    """Determine the adapter tier from provider + model capabilities.

    Returns "off", "light", or "full".
    """
    from prometheus.providers.registry import ProviderRegistry

    # Tier 1: API enforces structure (Anthropic, OpenAI, cloud providers)
    if provider_name == "anthropic" or ProviderRegistry.is_cloud(provider_name):
        return "off"

    # Tier 2: Model has native tool calling, but server doesn't guarantee structure
    if _has_native_tool_calling(model_name):
        return "light"

    # Tier 3: Full adapter pipeline
    return "full"


def create_adapter(model_cfg: dict[str, Any], adapter_cfg: dict[str, Any] | None = None):
    """Create the model adapter layer with three tiers.

    Tier "off"   — API enforces structure. Passthrough, no validation.
    Tier "light" — Model trained for tool calling on local server.
                   Keep model-specific formatter, GBNF on, validator NONE,
                   max_retries=1, adaptive strictness on.
    Tier "full"  — No native tool calling. Full validation + repair.
    """
    from prometheus.adapter import ModelAdapter
    from prometheus.adapter.formatter import (
        QwenFormatter,
        GemmaFormatter,
        AnthropicFormatter,
        PassthroughFormatter,
    )

    provider_name = model_cfg.get("provider", "llama_cpp")
    model_name = model_cfg.get("model", "")
    acfg = adapter_cfg or {}
    adaptive_kwargs = {
        "adaptive_strictness": acfg.get("adaptive_strictness", False),
        "strictness_threshold": acfg.get("strictness_threshold", 0.8),
        "strictness_window": acfg.get("strictness_window", 100),
        # Phase 4: conservative dict-wrap unwrapping, per tool. Default off.
        "unwrap_tools": acfg.get("unwrap_dict_args") or (),
    }

    tier = _get_adapter_tier(provider_name, model_name)

    if tier == "off":
        formatter = AnthropicFormatter() if provider_name == "anthropic" else PassthroughFormatter()
        return ModelAdapter(formatter=formatter, tier="off", **adaptive_kwargs)

    if tier == "light":
        # Keep model-specific formatter — the model was trained for a format
        if "gemma" in model_name.lower():
            formatter = GemmaFormatter()
        elif "qwen" in model_name.lower():
            formatter = QwenFormatter()
        else:
            formatter = QwenFormatter()  # safe default for tool-calling models
        log.info("Adapter tier=light for %s (native tool calling, GBNF + light validation)", model_name)
        return ModelAdapter(formatter=formatter, tier="light", **adaptive_kwargs)

    # tier == "full"
    if "gemma" in model_name.lower():
        formatter = GemmaFormatter()
    else:
        formatter = QwenFormatter()
    return ModelAdapter(formatter=formatter, strictness="MEDIUM", tier="full", **adaptive_kwargs)


async def create_mcp_runtime(
    config: dict[str, Any], registry: Any, tool_loader: Any | None = None
) -> Any:
    """Delegate to prometheus.mcp.bootstrap (FOUNDATION 2.3a) — the one
    construction path the daemon now shares, so MCP can never again be
    wired for one entry point and dark on the other.

    The import stays inside the function: prometheus.mcp's package
    __init__ eagerly pulls the optional ``mcp`` SDK, and this module (home
    of create_tool_registry) must import cleanly without it.
    """
    from prometheus.mcp.bootstrap import create_mcp_runtime as _impl
    return await _impl(config, registry, tool_loader=tool_loader)


def create_model_router(
    config: dict[str, Any],
    primary_provider: Any,
    primary_adapter: Any,
    primary_model: str,
):
    """Create the model router (Sprint 10 / GRAFT-ROUTER-WIRE Phase 2).

    Reads the `router:` section from prometheus.yaml. If the deprecated
    `model_router:` key is present, logs a migration warning and runs with
    primary-only routing until the user renames the block.
    """
    from prometheus.router import ModelRouter, load_router_config

    # Phase 2 migration (I3): warn if the deprecated model_router: key is present
    if "model_router" in config:
        log.warning(
            "model_router: config key is deprecated (renamed to router: in "
            "GRAFT-ROUTER-WIRE v3). Your existing rules are not being applied. "
            "Migrate the block manually: rename 'model_router:' to 'router:' "
            "and update the 'rules:' schema if it differs. Running with "
            "primary-only routing until migrated."
        )

    router_config = load_router_config(config)
    return ModelRouter(
        config=router_config,
        primary_provider=primary_provider,
        primary_adapter=primary_adapter,
        primary_model=primary_model,
    )


def create_divergence_detector(config: dict[str, Any]):
    """Create the divergence detector (Sprint 10)."""
    from prometheus.coordinator.divergence import DivergenceDetector, CheckpointStore
    try:
        store = CheckpointStore()
        return DivergenceDetector(config, checkpoint_store=store)
    except Exception as exc:
        log.warning("Divergence detector not available: %s", exc)
        return None


def create_security_gate(security_cfg: dict[str, Any], config_path: str | None = None):
    """Create the permission checker (Sprint 4 + Sprint 11 audit/exfil)."""
    from prometheus.permissions.audit import AuditLogger
    from prometheus.permissions.checker import SecurityGate
    from prometheus.permissions.exfiltration import ExfiltrationDetector

    # Sprint 11: audit logger
    audit_logger = None
    audit_cfg = security_cfg.get("audit", {})
    if audit_cfg.get("enabled", True):
        audit_logger = AuditLogger(get_data_dir() / "security")

    # Sprint 11: exfiltration detector
    from prometheus.config.shipped_defaults import (
        resolve_denied_paths, resolve_workspace_root)

    exfil_detector = None
    exfil_cfg = security_cfg.get("exfiltration", {})
    if exfil_cfg.get("enabled", True):
        exfil_detector = ExfiltrationDetector()

    return SecurityGate(
        mode=security_cfg.get("permission_mode", "default"),
        workspace_root=resolve_workspace_root(security_cfg),
        denied_commands=security_cfg.get("denied_commands"),
        denied_paths=resolve_denied_paths(security_cfg),
        allowed_commands=security_cfg.get("allowed_commands"),
        audit_logger=audit_logger,
        exfiltration_detector=exfil_detector,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# LCM wiring
# ---------------------------------------------------------------------------

def create_lcm_engine(provider: ModelProvider):
    """Create and wire the LCM engine + tools."""
    try:
        from prometheus.memory.lcm_engine import LCMEngine
        from prometheus.tools.builtin.lcm_grep import set_lcm_engine
        engine = LCMEngine(provider)
        set_lcm_engine(engine)
        return engine
    except Exception as exc:
        log.warning("LCM engine not available: %s", exc)
        return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(config: dict[str, Any]) -> str:
    """Assemble the full system prompt."""
    try:
        from prometheus.context.prompt_assembler import build_runtime_system_prompt
        skills_list = None
        try:
            from prometheus.skills.loader import load_skill_registry
            sr = load_skill_registry()
            if sr.list_skills():
                skills_list = [{"name": s.name, "description": s.description} for s in sr.list_skills()]
        except Exception:
            pass
        return build_runtime_system_prompt(cwd=str(Path.cwd()), config=config, skills=skills_list)
    except Exception:
        return config.get("gateway", {}).get(
            "system_prompt",
            "You are Prometheus, a sovereign AI agent. Be concise and helpful.",
        )


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def run_interactive(
    context: LoopContext,
    lcm_engine: Any | None,
    session_id: str,
    *,
    voice_mode: bool = False,
    voice_config: dict[str, Any] | None = None,
) -> None:
    """Run an interactive conversation loop with streaming output.

    Voice mode (opt-in via ``--voice`` flag or ``:voice on`` REPL command):
    every turn captures mic audio, transcribes via Whisper, and speaks
    the agent reply via piper. ``:voice`` / ``:v`` does a single voice
    turn from text mode without flipping the sticky flag. ``:text`` or
    ``:voice off`` leaves voice mode.
    """
    messages: list[ConversationMessage] = []
    turn_index = 0
    voice_config = voice_config or {}

    print(f"Prometheus {__version__} — interactive mode")
    print(f"Model: {context.model} | Provider: type(provider)")
    if voice_mode:
        print("Voice mode ON. Press Enter to record, ':text' to type instead.")
    else:
        print("Type your message. ':v' for one voice turn, ':voice on' for sticky.")
    print("(Ctrl+D or 'exit' to quit)\n")

    while True:
        # Each turn decides whether THIS reply should be spoken — sticky
        # voice_mode applies always; the per-turn ``:v`` command sets it
        # locally without flipping the sticky flag.
        voice_this_turn = voice_mode
        user_input: str | None = None

        try:
            if voice_mode:
                # In sticky voice mode: a bare Enter triggers recording;
                # typed text is interpreted as a command (exit, :text).
                print(">>> 🎤", flush=True)
                line = sys.stdin.readline()
                if not line:  # EOF
                    print("Goodbye.")
                    break
                line = line.strip().lower()
                if line in ("exit", "quit", "/exit", "/quit"):
                    print("Goodbye.")
                    break
                if line in (":text", ":t", ":voice off"):
                    voice_mode = False
                    print("Switched to text mode.\n")
                    continue
                # Anything else (including empty): start the capture
                from prometheus.cli.voice import cli_voice_capture
                user_input = await cli_voice_capture()
                if not user_input:
                    continue
            else:
                line = input(">>> ").strip()
                if not line:
                    continue
                lower = line.lower()
                if lower in ("exit", "quit", "/exit", "/quit"):
                    print("Goodbye.")
                    break
                if lower in (":voice", ":v"):
                    from prometheus.cli.voice import cli_voice_capture
                    user_input = await cli_voice_capture()
                    voice_this_turn = True
                    if not user_input:
                        continue
                elif lower in (":voice on", ":voice mode"):
                    voice_mode = True
                    print(
                        "Voice mode ON. Enter to record, "
                        "':text' to switch back.\n"
                    )
                    continue
                elif lower == ":voice off":
                    # Idempotent — already off
                    print("Voice mode already OFF.\n")
                    continue
                else:
                    user_input = line
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Goodbye.")
            break

        # Ingest to LCM
        if lcm_engine:
            await lcm_engine.ingest(session_id, "user", user_input, turn_index=turn_index)

        messages.append(ConversationMessage.from_user_text(user_input))

        # Stream the response
        response_text = ""
        try:
            async for event, usage in run_loop(context, messages):
                if isinstance(event, AssistantTextDelta):
                    print(event.text, end="", flush=True)
                    response_text += event.text
                elif isinstance(event, ToolExecutionStarted):
                    print(f"\n[tool] {event.tool_name}...", flush=True)
                elif isinstance(event, ToolExecutionCompleted):
                    status = "error" if event.is_error else "ok"
                    output_preview = event.output[:200] if event.output else ""
                    print(f"[tool] {event.tool_name} -> {status}", flush=True)
                    if output_preview:
                        print(f"  {output_preview}", flush=True)
                elif isinstance(event, AssistantTurnComplete):
                    if event.message.text and not response_text:
                        print(event.message.text, end="", flush=True)
                        response_text = event.message.text
        except Exception as exc:
            print(f"\n[error] {exc}")
            # Remove the last user message so we can retry
            if messages and messages[-1].role == "user":
                messages.pop()
            continue

        print()  # newline after response

        # Ingest assistant response to LCM
        if lcm_engine and response_text:
            await lcm_engine.ingest(
                session_id, "assistant", response_text, turn_index=turn_index
            )
            await lcm_engine.maybe_compact(session_id)

        # Voice output — speak the reply if this turn was voice-flagged
        # (sticky voice_mode OR per-turn ``:v``). Silent fallback if
        # piper isn't installed or the model path is bad.
        if voice_this_turn and response_text:
            try:
                from prometheus.cli.voice import cli_voice_speak
                await cli_voice_speak(response_text, voice_config)
            except Exception as exc:
                log.debug("Voice playback failed: %s", exc)

        turn_index += 1


# ---------------------------------------------------------------------------
# Coding mode (SPRINT-coding-mode v2) — one sandboxed iterate-to-green run
# ---------------------------------------------------------------------------

def run_coding_task(args) -> int:
    """`prometheus code` — clone, run the session, print the JSON report.

    Deliberately minimal construction compared to chat mode: the coding
    session pins its provider (no router/model fallback mid-run — that
    would confound the acceptance metric), builds its own tool registry
    around the sandbox, and writes its own system prompt. Thinking is ON
    by default (the addendum's #1 lever), overridable via
    --suppress-thinking for ablation runs.
    """
    import json as _json
    import time as _time
    from dataclasses import asdict
    from uuid import uuid4 as _uuid4

    from prometheus.coding.sandbox import clone_repo_for_sandbox
    from prometheus.coding.session import CodingSession, CodingTask

    config = load_config(args.config)

    # coding.enabled is the master switch, and it now actually switches.
    # It shipped in prometheus.yaml.default documented as "enabled: false"
    # and NOTHING read it: not this path, not POST /api/code. An operator
    # who set it false — or who simply took the shipped default — got a
    # coding mode that ran anyway. A documented key with a false default
    # that gates nothing is worse than no key at all, because it answers
    # "is this off?" with a confident yes.
    if not (config.get("coding", {}) or {}).get("enabled", False):
        print(_json.dumps({
            "ok": False,
            "error": "coding mode is disabled",
            "detail": (
                "coding.enabled is false (its shipped default). Coding runs "
                "execute model-authored commands against a clone of your "
                "repo, so they are opt-in. Set `coding.enabled: true` in "
                "prometheus.yaml to allow them."
            ),
        }))
        return 2

    model_cfg = dict(config.get("model", {}))
    provider, model_name = create_provider(model_cfg)
    adapter = create_adapter(model_cfg, config.get("adapter"))

    telemetry = None
    if config.get("infrastructure", {}).get("telemetry_enabled", True):
        try:
            from prometheus.telemetry.tracker import ToolCallTelemetry
            telemetry = ToolCallTelemetry()
        except Exception:
            pass

    task_id = args.task_id or f"c{_uuid4().hex[:8]}"
    sandbox_parent = args.sandbox_parent or str(
        get_data_dir().parent / "coding"
    )
    clone_name = f"{task_id}-{int(_time.time())}"
    # coding.sandbox_type is read HERE because this is the only construction
    # site for a coding run's sandbox. A backend selected in config but not
    # threaded to this call is a setting that silently does nothing.
    coding_cfg = config.get("coding", {}) or {}
    sandbox_backend = str(coding_cfg.get("sandbox_type", "process"))
    try:
        sandbox = clone_repo_for_sandbox(
            args.repo,
            sandbox_parent,
            name=clone_name,
            backend=sandbox_backend,
            task_id=task_id,
            network_isolation=bool(coding_cfg.get("network_isolation", False)),
            image=coding_cfg.get("docker_image") or None,
        )
    except Exception as exc:
        # A requested-but-unavailable backend raises rather than degrading, so
        # this abandons the run instead of executing it under weaker
        # confinement than the operator asked for.
        print(_json.dumps({
            "task_id": task_id, "status": "failed_abandoned",
            "reason": f"sandbox setup failed (backend={sandbox_backend!r}): {exc}",
        }))
        return 1

    session = CodingSession(
        provider=provider,
        model=model_name,
        sandbox=sandbox,
        task=CodingTask(
            task_id=task_id,
            description=args.task_description,
            acceptance_command=args.acceptance_command,
        ),
        adapter=adapter,
        telemetry=telemetry,
        # Precedence: explicit CLI flag > config > built-in default. Without
        # the config leg, coding.max_iterations and
        # coding.max_task_duration_minutes were near-twins of these flags
        # that nothing read — a config that looks tunable and is not.
        max_rounds=(
            args.max_rounds
            if args.max_rounds is not None
            else int(coding_cfg.get("max_iterations", 30))
        ),
        max_wall_seconds=float(
            args.max_wall_seconds
            if args.max_wall_seconds is not None
            else float(coding_cfg.get("max_task_duration_minutes", 20)) * 60.0
        ),
        suppress_thinking=True if args.suppress_thinking else False,
        control_dir=args.control_dir,
    )
    # A coding run must ALWAYS emit a JSON report and a verdict exit code —
    # an uncaught exception mid-run (a model-output edge case, a provider
    # hiccup) otherwise leaves a caller with exit 1 and no report to read.
    # Emit a structured failed report instead; the traceback goes to stderr
    # for forensics.
    try:
        report = asyncio.run(session.run())
        payload = asdict(report)
        payload["sandbox_root"] = str(sandbox.root)
        print(_json.dumps(payload, indent=2))
        return 0 if report.status == "success" else 1
    except Exception as exc:  # noqa: BLE001 — reported, not swallowed
        import traceback as _tb
        _tb.print_exc()
        print(_json.dumps({
            "task_id": task_id,
            "status": "failed_error",
            "reason": f"uncaught {type(exc).__name__}: {exc}",
            "sandbox_root": str(sandbox.root),
        }, indent=2))
        return 1


# ---------------------------------------------------------------------------
# One-shot mode
# ---------------------------------------------------------------------------

async def run_once(context: LoopContext, query: str) -> None:
    """Run a single query and print the result."""
    messages = [ConversationMessage.from_user_text(query)]
    response_text = ""

    async for event, usage in run_loop(context, messages):
        if isinstance(event, AssistantTextDelta):
            print(event.text, end="", flush=True)
            response_text += event.text
        elif isinstance(event, ToolExecutionStarted):
            print(f"\n[tool] {event.tool_name}...", end="", flush=True)
        elif isinstance(event, ToolExecutionCompleted):
            status = "error" if event.is_error else "ok"
            print(f" {status}", flush=True)
        elif isinstance(event, AssistantTurnComplete):
            if event.message.text and not response_text:
                print(event.message.text, end="", flush=True)

    print()


# ---------------------------------------------------------------------------
# Data reset helpers
# ---------------------------------------------------------------------------

def _reset_telemetry() -> None:
    """Delete telemetry.db after user confirmation."""
    from prometheus.config.paths import get_config_dir

    db_path = get_config_dir() / "telemetry.db"
    if not db_path.exists():
        print(f"No telemetry database found at {db_path}")
        return
    print(f"Will delete: {db_path}")
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    db_path.unlink()
    for suffix in ("-wal", "-shm"):
        p = db_path.parent / (db_path.name + suffix)
        if p.exists():
            p.unlink()
    print(f"Deleted {db_path}")


def _reset_data() -> None:
    """Delete all user data after confirmation.  Preserves config files."""
    import shutil
    from prometheus.config.paths import (
        get_config_dir,
        get_data_dir,
        get_lcm_db_path,
        get_legacy_lcm_db_path,
    )

    config_dir = get_config_dir()
    data_dir = get_data_dir()

    file_targets = [
        ("telemetry.db", config_dir / "telemetry.db"),
        ("memory.db", config_dir / "memory.db"),
        ("data/lcm.db", get_lcm_db_path()),
        ("data/security/audit.db", data_dir / "security" / "audit.db"),
        # Installs predating the single-resolution-point fix (2026-08-12) grew
        # a second lcm.db in the config root, written by CheckpointStore, whose
        # checkpoints.messages_json holds full conversation messages. It is
        # never written to now, but "delete all user data" has to mean it.
        # Prints "(not found)" on a fresh install, so the listing stays honest.
        ("lcm.db (legacy)", get_legacy_lcm_db_path()),
    ]
    dir_targets = [
        ("eval_results/", config_dir / "eval_results"),
        ("wiki/", get_wiki_root()),
        ("sentinel/", config_dir / "sentinel"),
        ("skills/auto/", config_dir / "skills" / "auto"),
    ]

    print("The following will be deleted:")
    for label, path in file_targets:
        status = "(exists)" if path.exists() else "(not found)"
        print(f"  {label}: {path} {status}")
    for label, path in dir_targets:
        status = "(exists)" if path.exists() else "(not found)"
        print(f"  {label}: {path} {status}")

    confirm = input("\nDelete all listed data? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    for label, path in file_targets:
        if path.exists():
            path.unlink()
            for suffix in ("-wal", "-shm"):
                p = path.parent / (path.name + suffix)
                if p.exists():
                    p.unlink()
            print(f"  Deleted {label}")

    for label, path in dir_targets:
        if path.exists():
            shutil.rmtree(path)
            print(f"  Deleted {label}")

    print("Done. Config files preserved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Prometheus CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="prometheus",
        description="Prometheus — sovereign AI agent harness",
    )
    parser.add_argument(
        "--version", action="version", version=f"Prometheus {__version__}"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to prometheus.yaml config file",
    )
    parser.add_argument(
        "--once", type=str, default=None, metavar="QUERY",
        help="Run a single query then exit (non-interactive)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Override provider (llama_cpp, ollama, anthropic)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show INFO logs on the console in chat/--once mode (they always "
             "land in ~/.prometheus/logs/cli.log; the console default is "
             "WARNING so the reply isn't buried in operator logging)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="[alias for `prometheus setup`] Run first-time setup wizard",
    )
    parser.add_argument(
        "--setup-gateway-only", action="store_true",
        help="[alias for `prometheus setup --gateway-only`] Add or change gateway only",
    )
    parser.add_argument(
        "--reset-telemetry", action="store_true",
        help="Delete telemetry.db and exit",
    )
    parser.add_argument(
        "--reset-data", action="store_true",
        help="Delete all user data (telemetry, memory, LCM, audit, evals, wiki, sentinel, skills/auto) and exit",
    )
    parser.add_argument(
        "--voice", action="store_true",
        help="Start interactive mode with voice in/out enabled (push-to-talk via Enter).",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Onboarding Phase 0: the canonical setup/diagnostics surface.
    from prometheus.cli.doctor import add_doctor_subparser
    from prometheus.cli.service import add_install_service_subparser
    from prometheus.cli.setup import add_setup_subparser
    from prometheus.cli.token import add_token_subparser
    add_setup_subparser(subparsers)
    add_token_subparser(subparsers)
    add_doctor_subparser(subparsers)
    add_install_service_subparser(subparsers)

    daemon_parser = subparsers.add_parser("daemon", help="Start always-on daemon")
    daemon_parser.add_argument(
        "--telegram-only", action="store_true",
        help="Only start Telegram adapter",
    )

    identity_parser = subparsers.add_parser(
        "identity", help="Manage identity files (SOUL.md, AGENTS.md)",
    )
    identity_parser.add_argument(
        "--show", action="store_true", help="Print current SOUL.md",
    )
    identity_parser.add_argument(
        "--regenerate", action="store_true",
        help="Regenerate SOUL.md and AGENTS.md interactively",
    )

    migrate_parser = subparsers.add_parser(
        "migrate", help="Import data from Hermes Agent or OpenClaw",
    )
    migrate_parser.add_argument(
        "--from", dest="source_type", required=True,
        choices=["hermes", "openclaw"],
        help="Source agent to migrate from",
    )
    migrate_parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview migration without writing files",
    )
    migrate_parser.add_argument(
        "--source", dest="source_path",
        help="Custom source directory path",
    )
    migrate_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing Prometheus files (archives originals)",
    )
    migrate_parser.add_argument(
        "--preset", choices=["user-data", "full"], default="user-data",
        help="Migration preset (default: user-data, excludes secrets)",
    )
    migrate_parser.add_argument(
        "--skill-conflict", choices=["skip", "overwrite", "rename"],
        default="skip", help="How to handle skill name collisions",
    )
    migrate_parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt",
    )

    # SPRINT-coding-mode v2: sandboxed iterate-to-green coding run.
    code_parser = subparsers.add_parser(
        "code", help="Run a coding task in a sandboxed clone (iterate-to-green)",
    )
    code_parser.add_argument(
        "--repo", required=True,
        help="Path to the target git repository (cloned, never touched)",
    )
    code_parser.add_argument(
        "--task", required=True, dest="task_description",
        help="What to do, in plain language",
    )
    code_parser.add_argument(
        "--acceptance", required=True, dest="acceptance_command",
        help="Command that must exit 0 for the task to count as done",
    )
    code_parser.add_argument(
        "--task-id", default=None,
        help="Task id (default: generated; names the branch coding/<id>)",
    )
    code_parser.add_argument(
        # default=None so the config can supply it; the effective fallback
        # lives at the use site. A literal default here would silently win
        # over coding.max_iterations, which is how that key came to look
        # settable while doing nothing.
        "--max-rounds", type=int, default=None,
        help="Model-round cap across the whole run "
             "(default: coding.max_iterations, else 30)",
    )
    code_parser.add_argument(
        "--max-wall-seconds", type=int, default=None,
        help="Wall-clock cap for the run "
             "(default: coding.max_task_duration_minutes, else 1200)",
    )
    code_parser.add_argument(
        "--sandbox-parent", default=None,
        help="Where the dedicated clone lives (default: ~/.prometheus/coding)",
    )
    code_parser.add_argument(
        "--control-dir", default=None,
        help="Per-run dir for the mid-run control file (pause/inject/resume). Absent → the "
             "control channel is OFF (default, byte-identical un-supervised behavior).",
    )
    code_parser.add_argument(
        "--suppress-thinking", action="store_true",
        help="Run with thinking suppressed (default: coding turns THINK — "
             "the addendum's highest-leverage lever)",
    )

    # Record-a-skill phase 2: video/YouTube -> skill DRAFT (human review).
    ingest_parser = subparsers.add_parser(
        "ingest-video",
        help="Turn a screen recording or YouTube URL into a skill draft "
             "for review in Beacon",
    )
    ingest_parser.add_argument("source", help="Video file path or http(s)/YouTube URL")
    ingest_parser.add_argument("--model", default=None, help="Vision model override")
    ingest_parser.add_argument("--provider", default=None, help="Provider type override")
    ingest_parser.add_argument("--base-url", default=None, help="Vision endpoint override")
    ingest_parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate")
    ingest_parser.add_argument("--no-audio", action="store_true", help="Skip narration transcription")
    ingest_parser.add_argument(
        "--work-dir", default=None,
        help="Session directory (reuse to resume a crashed digest)",
    )
    ingest_parser.add_argument(
        "--force", action="store_true",
        help="Run even if model_registry.yaml says the model lacks vision support",
    )

    # Record-a-skill phase 2: score a candidate VLM against the annotated
    # session corpus before enabling it for video ingestion.
    bakeoff_parser = subparsers.add_parser(
        "bakeoff-vlm",
        help="Score a vision model against the annotated skill corpus "
             "(golden-SKILL.md diff, hallucination-penalized)",
    )
    bakeoff_parser.add_argument(
        "--corpus", required=True,
        help="Corpus root containing videos and a ground_truth/ dir "
             "(e.g. ~/projects/skillforge)",
    )
    bakeoff_parser.add_argument("--model", required=True, help="Vision model name (registry entry)")
    bakeoff_parser.add_argument(
        "--provider", default="llama_cpp",
        help="Provider type for the vision model (default: llama_cpp)",
    )
    bakeoff_parser.add_argument(
        "--base-url", default="http://localhost:8080",
        help="Vision model endpoint (default: http://localhost:8080)",
    )
    bakeoff_parser.add_argument("--limit", type=int, default=0, help="Max videos to score (0 = all)")
    bakeoff_parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate")
    bakeoff_parser.add_argument("--no-audio", action="store_true", help="Skip audio transcription")
    bakeoff_parser.add_argument(
        "--force", action="store_true",
        help="Run even if model_registry.yaml says the model lacks vision support",
    )
    bakeoff_parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Mean accuracy required for a PASS verdict (default: 0.75)",
    )
    bakeoff_parser.add_argument("--output", default=None, help="Write a JSON report to this path")

    # SUNRISE: export-traces — write golden tool-call traces to JSONL for fine-tuning.
    export_parser = subparsers.add_parser(
        "export-traces", help="Export golden tool-call traces to a JSONL file",
    )
    export_parser.add_argument(
        "--limit", type=int, default=100,
        help="Maximum number of traces to export (default: 100)",
    )
    export_parser.add_argument(
        "--output", default="~/.prometheus/trajectories/",
        help="Output directory (default: ~/.prometheus/trajectories/)",
    )
    export_parser.add_argument(
        "--tool", default=None,
        help="Filter by tool name (default: all tools)",
    )

    # `prometheus config --show-defaults` — print the SHIPPED TEMPLATE.
    #
    # Three artefacts could have gone here and only one is useful. The
    # EFFECTIVE MERGE is disqualified by its own name: under `--show-defaults`
    # it prints the operator's CHOICES, the opposite of a default, plus every
    # token in their config. A hand-built table of THE CODE'S DEFAULTS would be
    # a 364-entry drift surface whose only job is to agree with code it does
    # not live beside, and it would throw away every comment — the comments
    # being most of the template's value.
    #
    # So: print the template, and make the template TRUE rather than merely
    # printed. tests/test_config_defaults_equality pins each documented value
    # equal to the reader's own fallback, both sides extracted programmatically.
    # Round-tripping is then a CONSEQUENCE — feeding this output back yields
    # the same effective values because every value already equals what the
    # reader would have chosen — not a property this command has to arrange.
    config_parser = subparsers.add_parser(
        "config", help="Inspect Prometheus configuration",
    )
    config_parser.add_argument(
        "--show-defaults", action="store_true",
        help="Print the shipped config template (every documented key, with "
             "comments) to stdout",
    )

    args = parser.parse_args()

    # Logging — FIRSTLIGHT GAP-3. On the CLI SURFACES (interactive chat and
    # --once; args.command is None on that fallthrough) the console defaults
    # to WARNING so a stranger's first answer isn't buried in httpx/audit/
    # loop INFO lines, while a file handler keeps the full INFO stream in
    # ~/.prometheus/logs/cli.log (before this, the console spam was also the
    # ONLY record of a CLI run). -v/--verbose restores the old console
    # stream verbatim (same format, same stderr); --debug is unchanged.
    # Subcommands — including `prometheus daemon`, which layers its own
    # file logging on top — keep the exact pre-GAP-3 configuration: root at
    # INFO with a single stderr handler.
    log_level = logging.DEBUG if args.debug else logging.INFO
    cli_surface = args.command is None
    console = logging.StreamHandler(sys.stderr)
    if cli_surface and not args.verbose and not args.debug:
        console.setLevel(logging.WARNING)
    handlers: list[logging.Handler] = [console]
    if cli_surface:
        try:
            from prometheus.config.paths import get_logs_dir
            cli_log = logging.FileHandler(
                get_logs_dir() / "cli.log", encoding="utf-8"
            )
            handlers.append(cli_log)
        except Exception:
            pass  # an unwritable logs dir must not kill the CLI
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    # `prometheus setup` — the ONE canonical wizard (Onboarding Phase 0).
    # `--setup` / `--setup-gateway-only` are thin forwarding aliases.
    if args.command == "setup" or args.setup or args.setup_gateway_only:
        from prometheus.cli.setup import run_setup
        if args.command != "setup":
            print("note: `prometheus --setup` is now `prometheus setup` — "
                  "forwarding.\n")
            args.gateway_only = args.setup_gateway_only
            args.fast = False
            args.noninteractive = False
        sys.exit(run_setup(args))

    # `prometheus token show|rotate` — web API token management.
    if args.command == "config":
        if getattr(args, "show_defaults", False):
            from prometheus.config.template import (
                TemplateNotFound, get_template_path, read_template_text)
            try:
                sys.stdout.write(read_template_text())
            except TemplateNotFound as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0
        config_parser.print_help()
        return 0

    if args.command == "token":
        from prometheus.cli.token import run_token_command
        sys.exit(run_token_command(args, load_config(args.config)))

    # `prometheus doctor` — install diagnostics (exit nonzero on errors).
    if args.command == "doctor":
        from prometheus.cli.doctor import run_doctor_command
        sys.exit(run_doctor_command(args))

    # `prometheus bakeoff-vlm` — vision-model corpus scoring (record-a-skill).
    if args.command == "bakeoff-vlm":
        from prometheus.cli.bakeoff import run_bakeoff_command
        sys.exit(run_bakeoff_command(args))

    # `prometheus ingest-video` — video/YouTube -> skill draft (record-a-skill).
    if args.command == "ingest-video":
        from prometheus.cli.ingest_video import run_ingest_video_command
        sys.exit(run_ingest_video_command(args, load_config(args.config)))

    # `prometheus install-service` — systemd user unit installer.
    if args.command == "install-service":
        from prometheus.cli.service import run_install_service_command
        sys.exit(run_install_service_command(args))

    # Identity subcommand — manage SOUL.md / AGENTS.md
    if args.command == "identity":
        if args.show:
            soul = get_config_dir() / "SOUL.md"
            if soul.exists():
                print(soul.read_text())
            else:
                print("No SOUL.md found. Run: prometheus setup")
                sys.exit(1)
        elif args.regenerate:
            from prometheus.setup_wizard import SetupWizard
            wizard = SetupWizard()
            wizard._setup_identity()
        else:
            print("Usage: python -m prometheus identity [--show | --regenerate]")
        sys.exit(0)

    # Migration subcommand — runs pre-agent, no model needed
    if args.command == "migrate":
        from prometheus.cli.migrate import run_migration
        success = run_migration(args)
        sys.exit(0 if success else 1)

    # SPRINT-coding-mode v2: one sandboxed coding run, then exit. The JSON
    # report goes to stdout (a managed task's output file captures it);
    # the exit code is the verdict.
    if args.command == "code":
        sys.exit(run_coding_task(args))

    # SUNRISE: export-traces — manual trigger for golden trace JSONL export.
    if args.command == "export-traces":
        from prometheus.telemetry.tracker import ToolCallTelemetry
        telemetry = ToolCallTelemetry()
        try:
            # Same pairing the daemon does — a trace without the conversation
            # that prompted it is not a training example, so the CLI dump
            # resolves context too rather than emitting context-free rows.
            from prometheus.memory.lcm_conversation_store import (
                LCMConversationStore,
            )
            from prometheus.sentinel.golden_trace_exporter import (
                lcm_context_resolver,
            )
            try:
                _conv_store = LCMConversationStore()
            except Exception:
                _conv_store = None
            output_path = telemetry.export_golden_traces(
                tool_name=args.tool,
                limit=args.limit,
                output_dir=args.output,
                context_resolver=lcm_context_resolver(_conv_store),
            )
            print(f"Exported: {output_path}")
            sys.exit(0)
        except Exception as exc:
            print(f"Export failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # Data reset commands
    if args.reset_telemetry:
        _reset_telemetry()
        sys.exit(0)

    if args.reset_data:
        _reset_data()
        sys.exit(0)

    # Daemon subcommand — delegate to the packaged daemon module so wheel
    # installs work (scripts/ is not part of the wheel).
    if args.command == "daemon":
        from prometheus.daemon import main as daemon_main
        # Re-inject args so daemon sees them
        sys.argv = ["prometheus-daemon"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.telegram_only:
            sys.argv.append("--telegram-only")
        if args.debug:
            sys.argv.append("--debug")
        daemon_main()
        return

    # Load config — hint about setup wizard if no config exists
    config = load_config(args.config)
    if not config and not _PROMETHEUS_YAML.exists() and not (get_config_dir() / "prometheus.yaml").exists():
        print("No configuration found. Run the setup wizard:\n")
        print("  prometheus setup\n")
        print("Or create config/prometheus.yaml manually.")
        sys.exit(1)
    model_cfg = config.get("model", {})
    security_cfg = config.get("security", {})

    # Apply CLI overrides
    if args.model:
        model_cfg["model"] = args.model
    if args.provider:
        model_cfg["provider"] = args.provider

    # Build components
    provider, model_name = create_provider(model_cfg)

    # Detect actual loaded model from the server (falls back to config)
    if model_cfg.get("provider", "llama_cpp") in ("llama_cpp",):
        model_name = _detect_model_or_fallback(
            model_cfg.get("base_url", "http://localhost:8080"), model_name,
        )
        model_cfg["model"] = model_name

    security_gate = create_security_gate(security_cfg, getattr(args, "config", None))
    registry = create_tool_registry(security_cfg, security_gate=security_gate)
    adapter = create_adapter(model_cfg, config.get("adapter"))
    lcm_engine = create_lcm_engine(provider)
    system_prompt = build_system_prompt(config)

    # Telemetry (optional)
    telemetry = None
    if config.get("infrastructure", {}).get("telemetry_enabled", True):
        try:
            from prometheus.telemetry.tracker import ToolCallTelemetry
            telemetry = ToolCallTelemetry()
        except Exception:
            pass

    # DynamicToolLoader — deferred loading support
    from prometheus.context.dynamic_tools import DynamicToolLoader
    tool_loader = DynamicToolLoader(registry, config.get("tools", {}).get("deferred_loading"))

    # Sprint 10 / Phase 2: Model Router + Divergence Detector
    # Router now requires primary provider + adapter + model built beforehand.
    model_router = create_model_router(config, provider, adapter, model_name)
    # Phase 3: wire router back into adapter.retry so RetryEngine can escalate
    if adapter is not None and hasattr(adapter, "retry"):
        adapter.retry.router = model_router
    divergence_detector = create_divergence_detector(config)

    # Sprint 15 wiring fix: HookExecutor was built (Sprint 2) but never created
    hook_executor = None
    try:
        from prometheus.hooks.executor import HookExecutor, HookExecutionContext
        from prometheus.hooks.loader import load_hook_registry
        # H3: load YAML-defined hooks from the config `hooks:` section. Empty or
        # absent → empty registry (no behavior change).
        hook_registry = load_hook_registry(config.get("hooks", {}) or {})
        hook_executor = HookExecutor(
            registry=hook_registry,
            context=HookExecutionContext(
                cwd=Path.cwd(),
                provider=provider,
                default_model=model_name,
            ),
        )
    except Exception:
        log.warning("HookExecutor init failed — hooks disabled", exc_info=True)

    # Generate session ID (Phase 3.5: threaded into LoopContext below so the
    # router's per-session override lookup has a unique namespace per CLI run).
    import uuid
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    # PASSIVE RECALL (MEMORY-3 follow-up): the CLI chats against the same
    # ~/.prometheus/memory.db the daemon writes (WAL — cross-process readers
    # are safe), so interactive CLI turns recall stored facts like the
    # gateway surfaces do. Fail-open: no store, no recall, no crash.
    memory_recall = None
    try:
        from prometheus.memory.recall import MemoryRecall, RecallConfig
        from prometheus.memory.store import MemoryStore

        recall_cfg = RecallConfig.from_config(config)
        if recall_cfg.enabled:
            memory_recall = MemoryRecall(store=MemoryStore(), config=recall_cfg)
    except Exception:
        log.warning("Memory recall unavailable for CLI session", exc_info=True)

    ctx_cfg = config.get("context", {})

    # Agent profile: the CLI has no /profile switching, so the holder stays on
    # profiles.default for the session — but it must still filter, or the CLI
    # advertises a different catalog than the daemon under the same config.
    profile_resolver = None
    try:
        from prometheus.config.profiles import ActiveProfileState, get_profile_store
        profile_resolver = ActiveProfileState(
            get_profile_store(), config.get("profiles", {}).get("default", "full"),
        ).get
    except Exception:
        log.warning("profile store unavailable — advertisement unfiltered", exc_info=True)

    context = LoopContext(
        fallback=build_fallback_target(model_cfg, detected_model=model_name),
        provider=provider,
        model=model_name,
        system_prompt=system_prompt,
        max_tokens=4096,
        tool_registry=registry,
        permission_checker=security_gate,
        hook_executor=hook_executor,
        adapter=adapter,
        telemetry=telemetry,
        model_router=model_router,
        divergence_detector=divergence_detector,
        max_tool_iterations=resolve_max_tool_iterations(config.get("model", {})),
        max_tool_iterations_cloud=resolve_max_tool_iterations_cloud(config.get("model", {})),
        tool_result_max=ctx_cfg.get("tool_result_max", 4000),
        tool_results_turn_budget=ctx_cfg.get("tool_results_turn_budget", 8000),
        microcompact_after_turns=ctx_cfg.get("microcompact_after_turns", 3),
        microcompact_on_cloud=ctx_cfg.get("microcompact_on_cloud", False),
        microcompact_keep_chars=ctx_cfg.get("microcompact_keep_chars", 200),
        microcompact_keep_chars_no_lcm=ctx_cfg.get("microcompact_keep_chars_no_lcm", 500),
        tool_loader=tool_loader,
        session_id=session_id,
        memory_recall=memory_recall,
        # The engine built above (create_lcm_engine) — the CLI ingests its
        # turns into it, and the microcompactor checks it; None if unavailable.
        lcm_engine=lcm_engine,
        profile_resolver=profile_resolver,
    )

    async def _async_main() -> None:
        # Sprint 12: MCP servers (must live in same async context as agent loop)
        mcp_runtime = None
        if config.get("mcp_servers"):
            mcp_runtime = await create_mcp_runtime(
                config, registry, tool_loader=tool_loader
            )

        # Wire GBNF grammar for llama.cpp constrained decoding
        model_cfg = config.get("model", {})
        if (
            model_cfg.get("grammar_enforcement", True)
            and hasattr(provider, "set_grammar")
            and adapter is not None
        ):
            grammar = adapter.generate_grammar(registry)
            if grammar:
                provider.set_grammar(grammar)
                # force-search (IGNITION): hand over the grammar SOURCE so per-call
                # required/{tool:X} grammars derive via the same enforcer path.
                if hasattr(provider, "set_grammar_source"):
                    provider.set_grammar_source(adapter.enforcer, registry.to_api_schema())
                log.info("GBNF grammar enforcement enabled for tool calls")

        try:
            if args.once:
                await run_once(context, args.once)
            else:
                # Voice-mode plumbing: pull gateway.voice block from the
                # loaded config so CLI and Telegram share engine/model
                # settings, and respect --voice for sticky-on at start.
                from prometheus.cli.voice import get_voice_config
                voice_cfg = get_voice_config(config)
                await run_interactive(
                    context, lcm_engine, session_id,
                    voice_mode=bool(getattr(args, "voice", False)),
                    voice_config=voice_cfg,
                )
        finally:
            if mcp_runtime:
                await mcp_runtime.close()

    asyncio.run(_async_main())

    # Cleanup
    if lcm_engine:
        lcm_engine.close()


if __name__ == "__main__":
    main()
