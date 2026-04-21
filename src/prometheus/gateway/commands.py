"""Shared command handlers for gateway adapters (Telegram, Slack).

Platform-agnostic command logic. Each function returns a string
that the adapter sends via its own transport.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prometheus.tools.base import ToolRegistry


def cmd_help() -> str:
    """Return help text listing available commands."""
    return (
        "Prometheus — Sovereign AI Agent\n"
        "\n"
        "Commands:\n"
        "/status    — Model, uptime, tools, memory, SENTINEL\n"
        "/model     — Current model name and provider\n"
        "/profile   — Show/switch agent profiles\n"
        "/anatomy   — Hardware, GPU, VRAM, infrastructure\n"
        "/doctor    — Diagnostic health check against model registry\n"
        "/wiki      — Wiki stats and recent entries\n"
        "/sentinel  — SENTINEL subsystem status\n"
        "/benchmark — Run a quick smoke test\n"
        "/context   — Context window usage\n"
        "/skills    — List available skills\n"
        "/reset     — Clear conversation context\n"
        "/help      — This message\n"
        "\n"
        "Send any message to chat with the agent."
    )


def cmd_model(model_name: str, model_provider: str) -> str:
    """Return model info text."""
    name = model_name or "(unknown)"
    provider = model_provider or "(unknown)"
    return f"Model: {name}\nProvider: {provider}"


def cmd_status(
    model_name: str,
    model_provider: str,
    start_time: float,
    tool_registry: ToolRegistry,
    cost_tracker: Any = None,
) -> str:
    """Return full status text."""
    lines: list[str] = ["Prometheus Status\n"]

    lines.append(f"Model: {model_name or '(unknown)'}")
    lines.append(f"Provider: {model_provider or '(unknown)'}")

    # Cloud provider cost info
    if cost_tracker is not None:
        lines.append(cost_tracker.report())

    if start_time:
        elapsed = int(time.monotonic() - start_time)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        lines.append(f"Uptime: {h}h {m}m {s}s")

    lines.append(f"Tools: {len(tool_registry.list_tools())}")

    # Memory stats
    try:
        from prometheus.tools.builtin.wiki_compile import _memory_store

        if _memory_store is not None:
            facts = _memory_store.get_all_memories(limit=10000)
            lines.append(f"Memory facts: {len(facts)}")
        else:
            lines.append("Memory: not initialized")
    except Exception:
        lines.append("Memory: unavailable")

    # SENTINEL state
    try:
        from prometheus.tools.builtin.sentinel_status import (
            _autodream,
            _observer,
        )

        if _observer is not None and _autodream is not None:
            state = (
                "dreaming"
                if _autodream.dreaming
                else ("active" if _observer.started else "idle")
            )
            lines.append(f"\nSENTINEL: {state}")
            lines.append(f"Dream cycles: {_autodream.cycle_count}")
            if _autodream.last_results:
                lines.append("Last dream results:")
                for r in _autodream.last_results:
                    status = "OK" if not r.error else f"FAIL: {r.error}"
                    lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
        else:
            lines.append("\nSENTINEL: not initialized")
    except Exception:
        lines.append("\nSENTINEL: unavailable")

    return "\n".join(lines)


def cmd_wiki() -> str:
    """Return wiki stats text."""
    wiki_index = Path.home() / ".prometheus" / "wiki" / "index.md"
    if not wiki_index.exists():
        return "Wiki: no index found at ~/.prometheus/wiki/index.md"

    try:
        content = wiki_index.read_text(encoding="utf-8")
        entries: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ["):
                entries.append(stripped)

        lines = [f"Wiki: {len(entries)} pages"]

        mtime = wiki_index.stat().st_mtime
        from datetime import datetime, timezone

        updated = datetime.fromtimestamp(mtime, tz=timezone.utc)
        lines.append(f"Last updated: {updated.strftime('%Y-%m-%d %H:%M UTC')}")

        if entries:
            lines.append("\nRecent entries:")
            for entry in entries[-5:]:
                lines.append(f"  {entry}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Wiki: error reading index — {exc}"


def cmd_sentinel() -> str:
    """Return SENTINEL subsystem status text."""
    try:
        from prometheus.sentinel.signals import SignalBus
        from prometheus.tools.builtin.sentinel_status import (
            _autodream,
            _observer,
            _signal_bus,
        )
    except ImportError:
        return "SENTINEL module not available."

    if _signal_bus is None or _observer is None or _autodream is None:
        return "SENTINEL not initialized. Is the daemon running with sentinel enabled?"

    lines: list[str] = ["SENTINEL Status\n"]

    idle_secs = int(time.time() - _observer.last_activity)
    lines.append("Observer:")
    lines.append(f"  Active: {_observer.started}")
    lines.append(f"  Last activity: {idle_secs}s ago")
    lines.append(f"  Pending nudges: {len(_observer.pending_nudges)}")

    lines.append("\nAutoDream Engine:")
    lines.append(f"  Dreaming: {_autodream.dreaming}")
    lines.append(f"  Cycles completed: {_autodream.cycle_count}")
    if _autodream.last_cycle_time:
        ago = int(time.time() - _autodream.last_cycle_time)
        lines.append(f"  Last cycle: {ago}s ago")

    lines.append("\nSignal Bus:")
    lines.append(f"  Total signals: {_signal_bus.signal_count}")
    lines.append(f"  Subscribers: {_signal_bus.subscriber_count}")

    recent = _signal_bus.recent(limit=10)
    if recent:
        lines.append("\nRecent Signals:")
        for sig in recent:
            ago = int(time.time() - sig.timestamp)
            lines.append(f"  [{sig.kind}] from {sig.source} ({ago}s ago)")

    if _autodream.last_results:
        lines.append("\nLast Dream Cycle:")
        for r in _autodream.last_results:
            status = "OK" if not r.error else f"FAIL: {r.error}"
            lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
            for k, v in r.summary.items():
                lines.append(f"    {k}: {v}")

    if _observer.pending_nudges:
        lines.append("\nPending Nudges:")
        for nudge in _observer.pending_nudges[:5]:
            lines.append(f"  [{nudge.nudge_type}] {nudge.message[:80]}")

    return "\n".join(lines)


def cmd_context(system_prompt: str, model_name: str) -> str:
    """Return context window usage text."""
    from prometheus.context.token_estimation import estimate_tokens

    try:
        from prometheus.context.budget import TokenBudget

        budget = TokenBudget.from_config(model=model_name)
        effective_limit = budget.effective_limit
        reserved_output = budget.reserved_output
    except Exception:
        effective_limit = 24000
        reserved_output = 2000

    prompt_tokens = estimate_tokens(system_prompt)
    available = effective_limit - reserved_output
    headroom = max(0, available - prompt_tokens)
    usage_pct = (prompt_tokens / available * 100) if available > 0 else 0

    lines = [
        "Context Window\n",
        f"Window size:    {effective_limit:,} tokens",
        f"Reserved output: {reserved_output:,} tokens",
        f"Available:       {available:,} tokens",
        "",
        f"System prompt:   {prompt_tokens:,} tokens ({usage_pct:.0f}%)",
        f"Headroom:        {headroom:,} tokens",
        "",
        f"Model: {model_name or '(unknown)'}",
    ]

    bar_len = 20
    filled = round(usage_pct / 100 * bar_len)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    lines.append(f"[{bar}] {usage_pct:.0f}% used")

    return "\n".join(lines)


def cmd_profile(arg: str = "", current: str = "full") -> str:
    """Handle /profile command — show or switch profiles."""
    try:
        from prometheus.config.profiles import ProfileStore
    except ImportError:
        return "Profile system not available."

    store = ProfileStore()

    if not arg:
        # Show current + list available
        lines = [f"Current profile: {current}\n", "Available profiles:"]
        for p in store.list_profiles():
            marker = " [active]" if p.name == current else ""
            lines.append(f"  {p.name}{marker} -- {p.description}")
        lines.append("\nSwitch: /profile <name>")
        return "\n".join(lines)

    # Switch to named profile
    profile = store.get(arg.strip())
    if profile is None:
        names = ", ".join(store.names())
        return f"Unknown profile: {arg}\nAvailable: {names}"

    tools_desc = "all" if profile.tools is None else ", ".join(profile.tools)
    bootstrap_desc = ", ".join(profile.bootstrap_files) or "(none)"
    lines = [
        f"Switched to: {profile.name}",
        f"  {profile.description}",
        f"  Tools: {tools_desc}",
        f"  Bootstrap: {bootstrap_desc}",
    ]
    for sub, enabled in profile.subsystems.items():
        lines.append(f"  {sub}: {'enabled' if enabled else 'disabled'}")
    return "\n".join(lines)


async def cmd_anatomy() -> str:
    """Return infrastructure summary text."""
    try:
        from prometheus.tools.builtin.anatomy import _scanner, _writer, _project_store
    except ImportError:
        return "Anatomy module not available."

    if _scanner is None or _writer is None:
        return "Anatomy not initialized. Is the daemon running with anatomy enabled?"

    from prometheus.infra.anatomy import AnatomyScanner
    from prometheus.infra.anatomy_writer import AnatomyWriter

    scanner: AnatomyScanner = _scanner  # type: ignore[assignment]

    try:
        state = await scanner.scan()
    except Exception as exc:
        return f"Anatomy scan failed: {exc}"

    from urllib.parse import urlparse

    # Determine if inference is remote
    parsed_url = urlparse(state.inference_url)
    inf_host = parsed_url.hostname or ""
    is_remote = inf_host not in ("", "localhost", "127.0.0.1", "::1")

    lines: list[str] = ["\U0001f527 Prometheus Anatomy\n"]

    # Host line
    host_line = f"Host: {state.hostname} ({state.platform})"
    if is_remote:
        host_line += f" + {inf_host} (remote inference)"
    lines.append(host_line)

    if state.cpu:
        lines.append(f"CPU: {state.cpu[:60]}")
    if state.ram_total_gb:
        lines.append(f"RAM: {state.ram_available_gb:.1f} GB free / {state.ram_total_gb:.0f} GB")

    # Model
    if state.model_name:
        lines.append(f"\nModel: {state.model_name}")
        if state.model_quantization:
            lines.append(f"Quant: {state.model_quantization}")
    engine_label = state.inference_engine.replace("_", ".")
    lines.append(f"Engine: {engine_label} @ {state.inference_url}")
    lines.append(f"Vision: {'enabled' if state.vision_enabled else 'disabled'}")

    # GPU
    if state.gpu_name:
        gpu_label = state.gpu_name
        if is_remote:
            gpu_label += " (remote)"
        lines.append(f"\nGPU: {gpu_label}")
        if state.gpu_vram_total_mb:
            used_gb = (state.gpu_vram_used_mb or 0) / 1024
            free_gb = (state.gpu_vram_free_mb or 0) / 1024
            total_gb = state.gpu_vram_total_mb / 1024
            lines.append(f"VRAM: {used_gb:.1f} / {total_gb:.1f} GB ({free_gb:.1f} GB free)")
    elif is_remote:
        lines.append("\nGPU: remote (stats unavailable)")

    # Tailscale
    if state.tailscale_ip:
        online_count = sum(
            1 for p in state.tailscale_peers
            if isinstance(p, dict) and p.get("online")
        )
        lines.append(f"\nTailscale: {len(state.tailscale_peers)} peers ({online_count} online)")
        for peer in state.tailscale_peers:
            if isinstance(peer, dict):
                icon = "\u2705" if peer.get("online") else "\u274c"
                lines.append(f"  {peer['name']}: {peer.get('ip', '?')} {icon}")
            else:
                lines.append(f"  {peer}")

    # Disk
    if state.disk_total_gb:
        lines.append(f"\nDisk: {state.disk_free_gb} GB free / {state.disk_total_gb} GB")
        if state.prometheus_data_size_mb:
            lines.append(f"Prometheus data: {state.prometheus_data_size_mb:.0f} MB")

    # Uptime
    uptime_str = _read_uptime()
    if uptime_str:
        lines.append(f"\nUptime: {uptime_str}")

    if state.whisper_model:
        lines.append(f"Whisper: {state.whisper_model}")

    # Active project
    if _project_store is not None:
        from prometheus.infra.project_configs import ProjectConfigStore
        store: ProjectConfigStore = _project_store  # type: ignore[assignment]
        active = store.get_active()
        if active:
            lines.append(f"Config: {active.name} ({active.description})")

    return "\n".join(lines)


def _read_uptime() -> str | None:
    """Read daemon start time and return human-readable uptime."""
    import time
    from prometheus.config.paths import get_config_dir
    uptime_path = get_config_dir() / ".daemon_started"
    try:
        started = float(uptime_path.read_text().strip())
        elapsed = time.time() - started
        if elapsed < 0:
            return None
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return None


async def cmd_doctor(config: dict | None = None) -> str:
    """Run diagnostic health check and return formatted report."""
    try:
        from prometheus.tools.builtin.anatomy import _scanner
    except ImportError:
        return "Doctor requires anatomy module."

    if _scanner is None:
        return "Doctor: anatomy not initialized. Is the daemon running?"

    from prometheus.infra.anatomy import AnatomyScanner
    from prometheus.infra.doctor import Doctor, DiagnosticReport

    scanner: AnatomyScanner = _scanner  # type: ignore[assignment]

    try:
        state = await scanner.scan()
    except Exception as exc:
        return f"Doctor: scan failed — {exc}"

    doctor = Doctor(config)
    report = await doctor.diagnose(state)

    status_icons = {"ok": "\u2705", "warning": "\u26a0\ufe0f", "error": "\u274c", "info": "\u2139\ufe0f"}

    lines: list[str] = ["\U0001fa7a Prometheus Doctor\n"]

    if report.model_family:
        lines.append(f"Model: {report.model_name} ({report.model_family})")
    elif report.model_name:
        lines.append(f"Model: {report.model_name} (not in registry)")
    else:
        lines.append("Model: none detected")

    # Group by category
    for cat, cat_checks in report.checks_by_category().items():
        label = report.CATEGORY_LABELS.get(cat, cat)
        lines.append(f"\n\u2500\u2500 {label} \u2500\u2500")
        for check in cat_checks:
            icon = status_icons.get(check.status, "\u2753")
            lines.append(f"{icon} {check.name}: {check.message}")
            if check.fix and check.status in ("warning", "error"):
                for fix_line in check.fix.strip().split("\n"):
                    lines.append(f"   \u2192 {fix_line}")

    lines.append("")
    ok_count = sum(1 for c in report.checks if c.status == "ok")
    warn_count = sum(1 for c in report.checks if c.status == "warning")
    err_count = sum(1 for c in report.checks if c.status == "error")

    if err_count:
        lines.append(f"\U0001f534 {err_count} error(s), {warn_count} warning(s), {ok_count} ok")
    elif warn_count:
        lines.append(f"\U0001f7e1 {warn_count} warning(s), {ok_count} ok — check items above")
    else:
        lines.append(f"\U0001f7e2 All {ok_count} checks passed")

    return "\n".join(lines)


def cmd_skills() -> str:
    """Return available skills list text."""
    try:
        from prometheus.skills.loader import load_skill_registry

        registry = load_skill_registry()
        skills = registry.list_skills()
    except Exception as exc:
        return f"Skills: error loading registry — {exc}"

    if not skills:
        return "No skills available."

    lines = [f"Skills ({len(skills)})\n"]
    for skill in skills:
        source_tag = f" [{skill.source}]" if skill.source else ""
        lines.append(f"  {skill.name}{source_tag}")
        if skill.description:
            lines.append(f"    {skill.description[:80]}")

    lines.append("\nUse the skill tool to load a skill by name.")
    return "\n".join(lines)


def cmd_beacon(config: dict) -> str:
    """Report web bridge / Beacon dashboard status."""
    import platform as _platform

    web = config.get("web", {})
    if not web.get("enabled", False):
        return (
            "Beacon: not running\n\n"
            "Enable in config/prometheus.yaml:\n"
            "  web:\n"
            "    enabled: true"
        )

    host = _platform.node()
    api_port = web.get("api_port", 8005)
    ws_port = web.get("ws_port", 8010)
    api_url = f"http://{host}:{api_port}"
    ws_url = f"ws://{host}:{ws_port}"
    dash_url = f"http://{host}:3000"
    return (
        f"<b>Beacon</b>\n\n"
        f'  <a href="{api_url}/api/status">Open REST API</a>  ({host}:{api_port})\n'
        f"  WebSocket: <code>{ws_url}</code>\n"
        f'  <a href="{dash_url}">Open Dashboard</a>  ({host}:3000)'
    )
