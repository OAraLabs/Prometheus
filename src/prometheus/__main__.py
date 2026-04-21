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
from prometheus.config.paths import get_config_dir, get_data_dir, get_logs_dir
from prometheus.engine.agent_loop import AgentLoop, RunResult, run_loop, LoopContext
from prometheus.engine.messages import ConversationMessage
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ModelProvider

log = logging.getLogger("prometheus")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROMETHEUS_YAML = Path(__file__).resolve().parents[2] / "config" / "prometheus.yaml"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load prometheus.yaml with env var overrides applied.

    Precedence: env vars > secret files > YAML > defaults.
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
        WikiCompileTool,
        WikiLintTool,
        WikiQueryTool,
    )
    from prometheus.tools.builtin.cron_create import CronCreateTool
    from prometheus.tools.builtin.cron_delete import CronDeleteTool
    from prometheus.tools.builtin.cron_list import CronListTool

    workspace = security_cfg.get("workspace_root")
    registry = ToolRegistry()
    for tool in [
        # Core file/shell tools
        BashTool(workspace=workspace),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GrepTool(),
        GlobTool(),
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
    except Exception:
        pass
    registry.register(ts)

    # Sprint 11: Audit query tool (requires audit logger from security gate)
    if security_gate and hasattr(security_gate, '_audit') and security_gate._audit:
        registry.register(AuditQueryTool(security_gate._audit))

    # Optional tools — don't fail if deps missing
    try:
        from prometheus.tools.builtin.skill import SkillTool
        registry.register(SkillTool())
    except Exception:
        pass
    try:
        from prometheus.tools.builtin.todo_write import TodoWriteTool
        registry.register(TodoWriteTool())
    except Exception:
        pass

    # Browser — requires optional playwright dependency
    try:
        from prometheus.tools.builtin.browser import BrowserTool
        registry.register(BrowserTool())
    except Exception:
        pass

    # Session tools — require task manager
    try:
        from prometheus.tools.builtin.sessions_list import SessionsListTool
        from prometheus.tools.builtin.sessions_send import SessionsSendTool
        from prometheus.tools.builtin.sessions_spawn import SessionsSpawnTool
        registry.register(SessionsListTool())
        registry.register(SessionsSendTool())
        registry.register(SessionsSpawnTool())
    except Exception:
        pass

    # Task tools — require task manager
    try:
        from prometheus.tools.builtin.task_create import TaskCreateTool
        from prometheus.tools.builtin.task_get import TaskGetTool
        from prometheus.tools.builtin.task_list import TaskListTool
        from prometheus.tools.builtin.task_update import TaskUpdateTool
        from prometheus.tools.builtin.task_stop import TaskStopTool
        from prometheus.tools.builtin.task_output import TaskOutputTool
        registry.register(TaskCreateTool())
        registry.register(TaskGetTool())
        registry.register(TaskListTool())
        registry.register(TaskUpdateTool())
        registry.register(TaskStopTool())
        registry.register(TaskOutputTool())
    except Exception:
        pass

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


async def create_mcp_runtime(config: dict[str, Any], registry: Any) -> Any:
    """Create MCP runtime, connect servers, register tools (Sprint 12)."""
    mcp_servers = config.get("mcp_servers", {})
    if not mcp_servers:
        log.debug("MCP: no servers configured")
        return None

    try:
        from prometheus.mcp.runtime import McpRuntime
        from prometheus.mcp.adapter import register_mcp_tools
        from prometheus.tools.builtin.mcp_status import McpStatusTool

        runtime = McpRuntime(mcp_servers)
        await runtime.connect_all()

        count = register_mcp_tools(registry, runtime)
        registry.register(McpStatusTool(runtime))
        log.info("MCP: registered %d tools + mcp_status", count)

        return runtime
    except Exception as exc:
        log.warning("MCP runtime not available: %s", exc)
        return None


def create_model_router(config: dict[str, Any]):
    """Create the model router (Sprint 10)."""
    from prometheus.adapter.router import ModelRouter
    return ModelRouter(config)


def create_divergence_detector(config: dict[str, Any]):
    """Create the divergence detector (Sprint 10)."""
    from prometheus.coordinator.divergence import DivergenceDetector, CheckpointStore
    try:
        store = CheckpointStore()
        return DivergenceDetector(config, checkpoint_store=store)
    except Exception as exc:
        log.warning("Divergence detector not available: %s", exc)
        return None


def create_security_gate(security_cfg: dict[str, Any]):
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
    exfil_detector = None
    exfil_cfg = security_cfg.get("exfiltration", {})
    if exfil_cfg.get("enabled", True):
        exfil_detector = ExfiltrationDetector()

    return SecurityGate(
        mode=security_cfg.get("permission_mode", "default"),
        workspace_root=security_cfg.get("workspace_root"),
        denied_commands=security_cfg.get("denied_commands"),
        denied_paths=security_cfg.get("denied_paths"),
        audit_logger=audit_logger,
        exfiltration_detector=exfil_detector,
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
) -> None:
    """Run an interactive conversation loop with streaming output."""
    messages: list[ConversationMessage] = []
    turn_index = 0

    print(f"Prometheus {__version__} — interactive mode")
    print(f"Model: {context.model} | Provider: type(provider)")
    print("Type your message (Ctrl+D or 'exit' to quit)\n")

    while True:
        try:
            user_input = input(">>> ").strip()
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

        turn_index += 1


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
    from prometheus.config.paths import get_config_dir, get_data_dir

    config_dir = get_config_dir()
    data_dir = get_data_dir()

    file_targets = [
        ("telemetry.db", config_dir / "telemetry.db"),
        ("memory.db", config_dir / "memory.db"),
        ("data/lcm.db", data_dir / "lcm.db"),
        ("data/security/audit.db", data_dir / "security" / "audit.db"),
    ]
    dir_targets = [
        ("eval_results/", config_dir / "eval_results"),
        ("wiki/", config_dir / "wiki"),
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
        "--setup", action="store_true",
        help="Run first-time setup wizard",
    )
    parser.add_argument(
        "--setup-gateway-only", action="store_true",
        help="Add or change gateway only (skip model provider setup)",
    )
    parser.add_argument(
        "--reset-telemetry", action="store_true",
        help="Delete telemetry.db and exit",
    )
    parser.add_argument(
        "--reset-data", action="store_true",
        help="Delete all user data (telemetry, memory, LCM, audit, evals, wiki, sentinel, skills/auto) and exit",
    )

    subparsers = parser.add_subparsers(dest="command")
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

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Setup wizard — runs before anything else
    if args.setup or args.setup_gateway_only:
        from prometheus.setup_wizard import SetupWizard
        wizard = SetupWizard(gateway_only=args.setup_gateway_only)
        success = wizard.run()
        sys.exit(0 if success else 1)

    # Identity subcommand — manage SOUL.md / AGENTS.md
    if args.command == "identity":
        if args.show:
            soul = get_config_dir() / "SOUL.md"
            if soul.exists():
                print(soul.read_text())
            else:
                print("No SOUL.md found. Run: python -m prometheus --setup")
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

    # Data reset commands
    if args.reset_telemetry:
        _reset_telemetry()
        sys.exit(0)

    if args.reset_data:
        _reset_data()
        sys.exit(0)

    # Daemon subcommand — delegate to scripts/daemon.py
    if args.command == "daemon":
        from scripts.daemon import main as daemon_main
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
        print("  python3 -m prometheus --setup\n")
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

    security_gate = create_security_gate(security_cfg)
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

    # Sprint 10: Model Router + Divergence Detector
    model_router = create_model_router(config)
    divergence_detector = create_divergence_detector(config)

    # Sprint 15 wiring fix: HookExecutor was built (Sprint 2) but never created
    hook_executor = None
    try:
        from prometheus.hooks.executor import HookExecutor, HookExecutionContext
        from prometheus.hooks.registry import HookRegistry
        hook_registry = HookRegistry()
        hook_executor = HookExecutor(
            registry=hook_registry,
            context=HookExecutionContext(
                cwd=Path.cwd(),
                provider=provider,
                default_model=model_name,
            ),
        )
    except Exception:
        pass

    ctx_cfg = config.get("context", {})
    context = LoopContext(
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
        max_tool_iterations=config.get("model", {}).get("max_tool_iterations", 25),
        tool_results_turn_budget=ctx_cfg.get("tool_results_turn_budget", 8000),
        microcompact_after_turns=ctx_cfg.get("microcompact_after_turns", 3),
        microcompact_keep_chars=ctx_cfg.get("microcompact_keep_chars", 200),
        microcompact_keep_chars_no_lcm=ctx_cfg.get("microcompact_keep_chars_no_lcm", 500),
        tool_loader=tool_loader,
    )

    # Generate session ID
    import uuid
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    async def _async_main() -> None:
        # Sprint 12: MCP servers (must live in same async context as agent loop)
        mcp_runtime = None
        if config.get("mcp_servers"):
            mcp_runtime = await create_mcp_runtime(config, registry)

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
                log.info("GBNF grammar enforcement enabled for tool calls")

        try:
            if args.once:
                await run_once(context, args.once)
            else:
                await run_interactive(context, lcm_engine, session_id)
        finally:
            if mcp_runtime:
                await mcp_runtime.close()

    asyncio.run(_async_main())

    # Cleanup
    if lcm_engine:
        lcm_engine.close()


if __name__ == "__main__":
    main()
