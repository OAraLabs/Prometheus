#!/usr/bin/env python3
"""Prometheus daemon — main entry point for always-on operation.

Source: Novel code for Prometheus Sprint 6.
Starts Telegram adapter, cron scheduler, heartbeat, and memory extractor.
Signal handling for graceful shutdown.

Usage:
    python -m prometheus.scripts.daemon --telegram-only --debug
    python scripts/daemon.py --config config/prometheus.yaml
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

from prometheus.config.paths import get_config_dir, get_logs_dir
from prometheus.engine.agent_loop import AgentLoop
from prometheus.gateway.archive_writer import ArchiveWriter
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.cron_scheduler import run_scheduler_loop
from prometheus.gateway.heartbeat import Heartbeat
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.registry import ProviderRegistry
from prometheus.__main__ import (
    create_adapter,
    create_divergence_detector,
    create_model_router,
    create_security_gate,
    create_tool_registry,
)
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import ToolRegistry

logger = logging.getLogger("prometheus.daemon")

DEFAULT_SYSTEM_PROMPT = (
    "You are Prometheus, a sovereign AI agent. You have access to tools for "
    "file operations, shell commands, and cron job management. Be concise and helpful."
)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load prometheus.yaml configuration."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path("config/prometheus.yaml")
        if not path.exists():
            path = get_config_dir() / "prometheus.yaml"

    if not path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return {}

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_tool_registry(security_cfg: dict[str, Any] | None = None) -> ToolRegistry:
    """Create the tool registry with all builtin tools (same as CLI).

    Reuses create_tool_registry() from __main__ so daemon and CLI
    always have the same tool set.
    """
    if security_cfg is None:
        security_cfg = {}
    registry = create_tool_registry(security_cfg)

    # Add wiki tools (daemon-specific, not in CLI)
    try:
        from prometheus.tools.builtin.wiki_compile import WikiCompileTool
        from prometheus.tools.builtin.wiki_query import WikiQueryTool
        registry.register(WikiCompileTool())
        registry.register(WikiQueryTool())
    except Exception:
        pass

    return registry


async def run_daemon(args: argparse.Namespace) -> None:
    """Main async entry point — start all subsystems."""
    config = load_config(args.config)
    model_config = config.get("model", {})
    gateway_config = config.get("gateway", {})
    security_config = config.get("security", {})

    # ── Config drift guard (opt-in) ─────────────────────────────────────
    # Users can create ~/.prometheus/config_pins.yaml to pin critical
    # config values. If a pinned value drifts, the daemon auto-corrects
    # both in-memory and on disk.
    #
    # Example ~/.prometheus/config_pins.yaml:
    #   model.provider: llama_cpp
    #   model.base_url: http://<gpu-host>:8080
    #
    # If the file doesn't exist, no checking happens — users who run
    # Ollama, Anthropic, etc. are unaffected.
    _pins_path = get_config_dir() / "config_pins.yaml"
    if _pins_path.is_file():
        try:
            with _pins_path.open(encoding="utf-8") as fh:
                _pins = yaml.safe_load(fh) or {}
        except Exception:
            _pins = {}

        _drifted = False
        for dotpath, expected in _pins.items():
            if expected is None:
                continue
            parts = str(dotpath).split(".")
            val = config
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else None
            if val and str(val) != str(expected):
                logger.warning(
                    "CONFIG DRIFT DETECTED: %s = %r (pinned: %r). Auto-fixing.",
                    dotpath, val, expected,
                )
                obj = config
                for p in parts[:-1]:
                    obj = obj.setdefault(p, {})
                obj[parts[-1]] = expected
                _drifted = True

        if _drifted:
            config_path = Path(args.config) if args.config else Path("config/prometheus.yaml")
            try:
                with config_path.open("w", encoding="utf-8") as fh:
                    yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
                logger.info("CONFIG AUTO-FIXED: wrote corrected values to %s", config_path)
            except Exception as exc:
                logger.error("Failed to auto-fix config file: %s", exc)

    # Sprint 15 GRAFT: scoped daemon lock — prevent duplicate instances
    from prometheus.gateway.status import acquire_daemon_lock, release_daemon_lock
    lock_ok, lock_reason = acquire_daemon_lock()
    if not lock_ok:
        logger.error("Cannot start daemon: %s", lock_reason)
        sys.exit(1)

    # Archive writer
    archive = ArchiveWriter()
    archive.archive_event("daemon_start", {"args": vars(args)})

    # Write daemon start time for uptime tracking
    import time as _time
    uptime_path = Path(get_config_dir()) / ".daemon_started"
    uptime_path.write_text(str(_time.time()), encoding="utf-8")

    # Model provider — ProviderRegistry handles all provider types
    provider = ProviderRegistry.create(model_config)

    # Detect actual loaded model from the server (local providers only)
    config_model = model_config.get("model", "qwen3.5-32b")
    if hasattr(provider, "detect_loaded_model"):
        detected = await provider.detect_loaded_model()
        model_name = detected or config_model
        if detected:
            model_config["model"] = detected
        else:
            logger.info("Using model name from config: %s", config_model)
    else:
        model_name = config_model
        logger.info("Cloud provider: %s, model: %s", model_config.get("provider"), model_name)

    # Context size detection
    if hasattr(provider, "detect_context_size"):
        ctx_size = await provider.detect_context_size()
        if ctx_size:
            logger.info("Server context size: %d tokens", ctx_size)

    # Vision detection
    if hasattr(provider, "detect_vision"):
        has_vision = await provider.detect_vision()
        if has_vision:
            logger.info("Vision: enabled (multimodal)")
        else:
            logger.info("Vision: not available")
            vision_capable = ("gemma", "llava", "qwen-vl", "pixtral", "minicpm-v")
            if any(v in model_name.lower() for v in vision_capable):
                logger.info(
                    "Hint: %s supports vision. Restart llama.cpp with "
                    "--mmproj to enable image analysis.",
                    model_name,
                )

    # Cost tracker for cloud providers
    cost_tracker = None
    if ProviderRegistry.is_cloud(model_config.get("provider", "")):
        from prometheus.telemetry.cost import CostTracker
        cost_tracker = CostTracker()

    # Tool registry — same tools as CLI mode
    registry = build_tool_registry(security_cfg=security_config)

    # DynamicToolLoader — deferred loading support
    from prometheus.context.dynamic_tools import DynamicToolLoader
    tool_loader = DynamicToolLoader(registry, config.get("tools", {}).get("deferred_loading"))

    # Sprint 15 wiring fix: daemon was missing adapter, security_gate,
    # model_router, and divergence_detector — all were built but not connected.
    # Phase 2: router now requires primary provider + adapter + model built first.
    adapter = create_adapter(model_config, config.get("adapter"))
    security_gate = create_security_gate(security_config)
    model_router = create_model_router(config, provider, adapter, model_name)
    # Phase 3: wire the router back into the adapter's RetryEngine so it can
    # consult router.config.escalation_enabled at tool-retry exhaustion and
    # return RetryAction.ESCALATE instead of ABORT.
    if adapter is not None and hasattr(adapter, "retry"):
        adapter.retry.router = model_router
    divergence_detector = create_divergence_detector(config)

    # Telemetry — shared instance for AgentLoop and SENTINEL digest
    telemetry = ToolCallTelemetry()

    # Sprint 15 wiring fix: HookExecutor was built but never created in daemon
    hook_executor = None
    try:
        from prometheus.hooks.executor import HookExecutor, HookExecutionContext
        from prometheus.hooks.registry import HookRegistry
        hook_executor = HookExecutor(
            registry=HookRegistry(),
            context=HookExecutionContext(
                cwd=Path.cwd(),
                provider=provider,
                default_model=model_name,
            ),
        )
    except Exception:
        pass

    # Sprint 20: LSP orchestrator + diagnostics hook
    lsp_orchestrator = None
    post_result_hooks: list = []
    lsp_config = config.get("lsp", {})
    if lsp_config.get("enabled", False):
        try:
            from prometheus.lsp.orchestrator import LSPOrchestrator
            from prometheus.hooks.lsp_diagnostics import LSPDiagnosticsHook
            from prometheus.tools.builtin.lsp import LSPTool, set_lsp_orchestrator

            lsp_orchestrator = LSPOrchestrator(
                custom_servers=lsp_config.get("servers") or {},
            )
            set_lsp_orchestrator(lsp_orchestrator)
            registry.register(LSPTool())
            logger.info("LSP orchestrator initialised")

            if lsp_config.get("auto_diagnostics", True):
                diag_hook = LSPDiagnosticsHook(
                    orchestrator=lsp_orchestrator,
                    delay_ms=lsp_config.get("diagnostics_delay_ms", 500),
                )
                post_result_hooks.append(diag_hook)
                logger.info("LSP diagnostics hook registered")
        except Exception as exc:
            logger.warning("LSP not available: %s", exc)

    # Helper: regenerate GBNF grammar after tool set changes
    def _update_grammar() -> None:
        if (
            model_config.get("grammar_enforcement", True)
            and hasattr(provider, "set_grammar")
            and adapter is not None
        ):
            grammar = adapter.generate_grammar(registry)
            if grammar:
                provider.set_grammar(grammar)
                logger.info(
                    "GBNF grammar updated (%d tool schemas)",
                    len(registry.list_tools()),
                )

    _update_grammar()

    # Agent loop
    agent_loop = AgentLoop(
        provider=provider,
        model=model_name,
        tool_registry=registry,
        adapter=adapter,
        permission_checker=security_gate,
        hook_executor=hook_executor,
        telemetry=telemetry,
        model_router=model_router,
        divergence_detector=divergence_detector,
        post_result_hooks=post_result_hooks or None,
        max_tool_iterations=model_config.get("max_tool_iterations", 25),
        tool_loader=tool_loader,
    )

    # Shared session manager for all gateways
    from prometheus.engine.session import SessionManager
    session_manager = SessionManager()

    # Collect async tasks to run
    tasks: list[asyncio.Task] = []
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        release_daemon_lock()
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Telegram adapter
    telegram: TelegramAdapter | None = None
    telegram_token = gateway_config.get("telegram_token", "") or os.environ.get("PROMETHEUS_TELEGRAM_TOKEN", "")
    if telegram_token and gateway_config.get("telegram_enabled", True):
        tg_config = PlatformConfig(
            platform=Platform.TELEGRAM,
            token=telegram_token,
            allowed_chat_ids=gateway_config.get("allowed_chat_ids", []),
            proxy_url=gateway_config.get("proxy_url"),
        )
        from prometheus.context.prompt_assembler import build_runtime_system_prompt
        system_prompt = build_runtime_system_prompt(
            cwd=str(Path.cwd()), config=config,
        )
        telegram = TelegramAdapter(
            config=tg_config,
            agent_loop=agent_loop,
            tool_registry=registry,
            system_prompt=system_prompt,
            model_name=model_name,
            model_provider=model_config.get("provider", "llama_cpp"),
            session_manager=session_manager,
            prometheus_config=config,
        )
        if cost_tracker is not None:
            telegram.cost_tracker = cost_tracker
        await telegram.start()
        archive.archive_event("telegram_started")
        logger.info("Telegram adapter started")

        # Sprint 15b GRAFT: wire approval queue if enabled
        approval_cfg = security_config.get("approval_queue", {})
        if approval_cfg.get("enabled", False):
            from prometheus.permissions.approval_queue import ApprovalQueue
            default_chat = (gateway_config.get("allowed_chat_ids") or [None])[0]
            approval_queue = ApprovalQueue(
                telegram_adapter=telegram,
                timeout_seconds=approval_cfg.get("timeout_seconds", 300),
                default_chat_id=default_chat,
            )
            security_gate._approval_queue = approval_queue
            telegram._approval_queue = approval_queue
            logger.info("Approval queue wired to Telegram adapter")

    # Slack adapter
    slack_adapter = None
    slack_bot_token = gateway_config.get("slack_bot_token", "") or os.environ.get("PROMETHEUS_SLACK_BOT_TOKEN", "")
    slack_app_token = gateway_config.get("slack_app_token", "") or os.environ.get("PROMETHEUS_SLACK_APP_TOKEN", "")
    if slack_bot_token and slack_app_token and gateway_config.get("slack_enabled", False):
        try:
            from prometheus.gateway.slack import SlackAdapter
            slack_config = PlatformConfig(
                platform=Platform.SLACK,
                token=slack_bot_token,
                app_token=slack_app_token,
                allowed_channels=gateway_config.get("slack_channels", []),
            )
            if "system_prompt" not in dir():
                from prometheus.context.prompt_assembler import build_runtime_system_prompt
                system_prompt = build_runtime_system_prompt(
                    cwd=str(Path.cwd()), config=config,
                )
            slack_adapter = SlackAdapter(
                config=slack_config,
                agent_loop=agent_loop,
                tool_registry=registry,
                system_prompt=system_prompt,
                model_name=model_name,
                model_provider=model_config.get("provider", "llama_cpp"),
                session_manager=session_manager,
            )
            await slack_adapter.start()
            archive.archive_event("slack_started")
            logger.info("Slack adapter started (Socket Mode)")
        except ImportError:
            logger.warning(
                "Slack is enabled but slack-bolt is not installed. "
                "Install with: pip install 'prometheus[slack]'"
            )
        except Exception as exc:
            logger.error("Failed to start Slack adapter: %s", exc)

    # Heartbeat
    heartbeat = Heartbeat(gateway=telegram)
    heartbeat_task = asyncio.create_task(heartbeat.run_forever())
    tasks.append(heartbeat_task)

    # Cron scheduler (skip if --telegram-only)
    if not args.telegram_only:
        cron_task = asyncio.create_task(run_scheduler_loop())
        tasks.append(cron_task)
        logger.info("Cron scheduler started")

    # LCM engine (optional, from Sprint 7)
    lcm_engine = None
    try:
        from prometheus.memory.lcm_engine import LCMEngine
        from prometheus.tools.builtin.lcm_grep import set_lcm_engine
        lcm_engine = LCMEngine(provider)
        set_lcm_engine(lcm_engine)
        logger.info("LCM engine initialised")
    except Exception as exc:
        logger.warning("LCM engine not available: %s", exc)

    # Memory extractor (optional, from Sprint 5)
    try:
        from prometheus.memory.extractor import MemoryExtractor
        from prometheus.memory.store import MemoryStore
        from prometheus.memory.wiki_compiler import WikiCompiler
        from prometheus.tools.builtin.wiki_compile import set_wiki_compiler

        memory_store = MemoryStore()

        # Wiki compiler — auto-compiles after each extraction run
        wiki_compiler = WikiCompiler(store=memory_store)
        set_wiki_compiler(wiki_compiler, memory_store)
        logger.info("Wiki compiler initialised at %s", wiki_compiler.wiki_root)

        extractor = MemoryExtractor(
            store=memory_store,
            provider=provider,
            model=model_name,
            post_extract_callback=wiki_compiler.compile,
        )

        # Wire extractor into LCM for pre-compaction flush
        if lcm_engine is not None:
            lcm_engine.set_memory_extractor(extractor)
            logger.info("Memory extractor wired to LCM pre-compaction flush")

        extractor_task = asyncio.create_task(extractor.run_forever())
        tasks.append(extractor_task)
        logger.info("Memory extractor started")
    except Exception as exc:
        logger.warning("Memory extractor not available: %s", exc)

    # Infrastructure self-awareness — AnatomyScanner (Sprint 18 ANATOMY)
    anatomy_config = config.get("anatomy", {})
    if anatomy_config.get("enabled", True):
        try:
            from prometheus.infra.anatomy import AnatomyScanner
            from prometheus.infra.anatomy_writer import AnatomyWriter
            from prometheus.infra.project_configs import ProjectConfigStore
            from prometheus.tools.builtin.anatomy import set_anatomy_components

            scanner = AnatomyScanner(
                llama_cpp_url=model_config.get("base_url", "http://localhost:8080"),
                ollama_url=model_config.get("fallback_url", "http://localhost:11434"),
                inference_engine=model_config.get("provider", "llama_cpp"),
                ssh_user=anatomy_config.get("ssh_user"),
                ssh_key=anatomy_config.get("ssh_key"),
            )
            anatomy_writer = AnatomyWriter()
            project_store = ProjectConfigStore()
            set_anatomy_components(scanner, anatomy_writer, project_store)

            if anatomy_config.get("scan_on_startup", True):
                state = await scanner.scan()
                anatomy_writer.write(state, project_store.summaries())
                logger.info("Infrastructure scan complete: %s, model=%s",
                            state.hostname, state.model_name)

                # Doctor startup check — log warnings/errors from diagnostics
                doctor_cfg = config.get("doctor", {})
                if doctor_cfg.get("startup_check", True):
                    try:
                        from prometheus.infra.doctor import Doctor
                        doctor = Doctor(config)
                        report = await doctor.diagnose(state)
                        for check in report.checks:
                            if check.status == "error":
                                logger.error("Doctor: %s — %s", check.name, check.message)
                                if check.fix:
                                    logger.error("  Fix: %s", check.fix.strip().split("\n")[0])
                            elif check.status == "warning":
                                logger.warning("Doctor: %s — %s", check.name, check.message)
                                if check.fix:
                                    logger.warning("  Fix: %s", check.fix.strip().split("\n")[0])
                        if report.has_errors:
                            logger.error("Doctor: %d error(s) found at startup. Run /doctor for details.",
                                         sum(1 for c in report.checks if c.status == "error"))
                        elif report.has_warnings:
                            logger.warning("Doctor: %d warning(s) at startup. Run /doctor for details.",
                                           sum(1 for c in report.checks if c.status == "warning"))
                        else:
                            logger.info("Doctor: all checks passed")
                    except Exception as exc:
                        logger.debug("Doctor startup check skipped: %s", exc)
        except Exception as exc:
            logger.warning("Anatomy system not available: %s", exc)

    # Learning loop — SkillCreator (auto-generate skills from successful tasks)
    try:
        from prometheus.learning.skill_creator import SkillCreator
        skill_creator = SkillCreator(provider, model=model_name)
        agent_loop.set_post_task_hook(skill_creator.maybe_create)
        logger.info("SkillCreator wired to agent loop post-task hook")
    except Exception as exc:
        logger.warning("SkillCreator not available: %s", exc)

    # SENTINEL proactive subsystem (Sprint 9)
    sentinel_config = config.get("sentinel", {})
    if sentinel_config.get("enabled", True):
        try:
            from prometheus.sentinel.signals import SignalBus
            from prometheus.sentinel.autodream import AutoDreamEngine
            from prometheus.sentinel.observer import ActivityObserver
            from prometheus.sentinel.wiki_lint import WikiLinter
            from prometheus.sentinel.memory_consolidator import MemoryConsolidator
            from prometheus.sentinel.telemetry_digest import TelemetryDigest
            from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer
            from prometheus.tools.builtin.sentinel_status import set_sentinel_components
            from prometheus.tools.builtin.wiki_lint_tool import (
                set_wiki_linter as set_lint_wiki_linter,
            )
            from prometheus.tools.builtin.sentinel_status import SentinelStatusTool
            from prometheus.tools.builtin.wiki_lint_tool import WikiLintTool

            signal_bus = SignalBus()

            # Leaf components
            wiki_linter = WikiLinter()
            set_lint_wiki_linter(wiki_linter)

            mem_consolidator = None
            if "memory_store" in dir():
                mem_consolidator = MemoryConsolidator(
                    memory_store,
                    stale_days=sentinel_config.get("stale_threshold_days", 90),
                    decay_rate=sentinel_config.get("confidence_decay_rate", 0.05),
                )

            tel_digest = None
            try:
                tel_digest = TelemetryDigest(
                    telemetry,
                    period_hours=sentinel_config.get("digest_lookback_hours", 24),
                )
            except Exception:
                logger.debug("SENTINEL: telemetry digest not available")

            knowledge_synth = None
            if "memory_store" in dir() and sentinel_config.get("synthesis_enabled", True):
                knowledge_synth = KnowledgeSynthesizer(
                    store=memory_store,
                    provider=provider,
                    model=model_name,
                    budget_tokens=sentinel_config.get("dream_budget_tokens", 2000),
                )

            # Orchestrators
            autodream = AutoDreamEngine(
                signal_bus,
                wiki_linter=wiki_linter,
                memory_consolidator=mem_consolidator,
                telemetry_digest=tel_digest,
                knowledge_synth=knowledge_synth,
                config=sentinel_config,
            )
            observer = ActivityObserver(
                signal_bus,
                gateway=telegram,
                config=sentinel_config,
            )

            # Wire signal bus into existing subsystems
            heartbeat.signal_bus = signal_bus
            if "extractor" in dir():
                extractor.signal_bus = signal_bus

            # Start (signal-reactive, no separate tasks needed)
            await observer.start()
            await autodream.start()

            # Wire tool singletons and register
            set_sentinel_components(signal_bus, observer, autodream)
            registry.register(SentinelStatusTool())
            registry.register(WikiLintTool())

            logger.info("SENTINEL proactive subsystem started")
            _update_grammar()  # Regenerate grammar to include SENTINEL tools
        except Exception as exc:
            logger.warning("SENTINEL not available: %s", exc)

    # Web bridge (Beacon dashboard backend)
    web_config = config.get("web", {})
    if web_config.get("enabled", False):
        try:
            from prometheus.web.launcher import launch_web
            from prometheus.engine.agent_loop import LoopContext

            # Build system prompt if not already available
            if "system_prompt" not in dir():
                from prometheus.context.prompt_assembler import build_runtime_system_prompt
                system_prompt = build_runtime_system_prompt(
                    cwd=str(Path.cwd()), config=config,
                )

            loop_context = LoopContext(
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
            )

            api_port = web_config.get("api_port", 8005)
            ws_port = web_config.get("ws_port", 8010)
            web_task = asyncio.create_task(launch_web(
                config=config,
                signal_bus=signal_bus if "signal_bus" in dir() else None,
                session_mgr=session_manager,
                telemetry=telemetry,
                lcm_engine=lcm_engine if "lcm_engine" in dir() else None,
                agent_loop=agent_loop,
                approval_queue=approval_queue if "approval_queue" in dir() else None,
                loop_context=loop_context,
                api_port=api_port,
                ws_port=ws_port,
            ))
            tasks.append(web_task)
            logger.info("Web bridge started (REST :%d, WS :%d)", api_port, ws_port)
        except Exception as exc:
            logger.warning("Web bridge not available: %s", exc)

    logger.info("Prometheus daemon running. Press Ctrl+C to stop.")

    # Wait for shutdown
    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    archive.archive_event("daemon_shutdown")

    if lsp_orchestrator:
        await lsp_orchestrator.shutdown_all()

    if telegram:
        await telegram.stop()

    if slack_adapter:
        await slack_adapter.stop()

    heartbeat.stop()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Prometheus daemon stopped")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prometheus daemon")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to prometheus.yaml"
    )
    parser.add_argument(
        "--telegram-only",
        action="store_true",
        help="Only start Telegram adapter (skip cron scheduler)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = get_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "daemon.log"),
        ],
    )

    asyncio.run(run_daemon(args))


if __name__ == "__main__":
    main()
