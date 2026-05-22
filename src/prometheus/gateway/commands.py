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


# ---------------------------------------------------------------------------
# Sprint S1 Stream 3: visible memory & skills inspection commands
# ---------------------------------------------------------------------------


def cmd_memory_show(target: str = "memory") -> str:
    """Display the contents of MEMORY.md or USER.md with char count + limit.

    Args:
        target: ``"memory"`` (MEMORY.md, the agent's note-to-self file) or
            ``"user"`` (USER.md, the persistent user-model file).
    """
    from prometheus.memory.hermes_memory_tool import (
        _MEMORY_MAX_CHARS,
        _USER_MAX_CHARS,
        get_memory_store,
        get_user_store,
    )

    if target == "user":
        store = get_user_store()
        limit = _USER_MAX_CHARS
        label = "USER.md"
    else:
        store = get_memory_store()
        limit = _MEMORY_MAX_CHARS
        label = "MEMORY.md"

    entries = store.list_entries()
    chars = sum(len(e) + 1 for e in entries)
    header = f"{label} — {chars}/{limit} chars, {len(entries)} entries"
    if not entries:
        return f"{header}\n\n(empty)"
    body = "\n".join(f"- {e}" for e in entries)
    # Cap reply size; full content readable via the file on disk.
    if len(body) > 3500:
        body = body[:3500] + f"\n\n…truncated. Read {label} on disk for full content."
    return f"{header}\n\n{body}"


def cmd_memory_limits() -> str:
    """Show the hard char ceilings + current usage for MEMORY.md and USER.md."""
    from prometheus.memory.hermes_memory_tool import (
        _MEMORY_MAX_CHARS,
        _USER_MAX_CHARS,
        get_memory_store,
        get_user_store,
    )

    mem_entries = get_memory_store().list_entries()
    mem_chars = sum(len(e) + 1 for e in mem_entries)
    user_entries = get_user_store().list_entries()
    user_chars = sum(len(e) + 1 for e in user_entries)

    return (
        "Memory limits (hard ceiling + prune-oldest)\n"
        f"  MEMORY.md: {mem_chars}/{_MEMORY_MAX_CHARS} chars, {len(mem_entries)} entries\n"
        f"  USER.md:   {user_chars}/{_USER_MAX_CHARS} chars, {len(user_entries)} entries\n"
        "\n"
        "Single entries larger than the limit raise MemoryOverflowError\n"
        "and the agent loop is expected to consolidate before retry."
    )


def cmd_skills_auto_list() -> str:
    """List skills under ~/.prometheus/skills/auto/ with state + mtime.

    Uses the SkillStateStore for `pinned` and `state`; mtime as last_used.
    """
    import time as _time
    from datetime import datetime
    from prometheus.config.paths import get_config_dir
    from prometheus.learning.skill_state import SkillStateStore

    auto_dir = get_config_dir() / "skills" / "auto"
    if not auto_dir.is_dir():
        return "Auto skills directory not initialised yet (no skills created)."

    store = SkillStateStore()
    skills = sorted(auto_dir.glob("*.md"))
    if not skills:
        return "No auto-skills yet. They appear here when SkillCreator fires."

    now = _time.time()
    lines: list[str] = [f"Auto-skills ({len(skills)})"]
    for path in skills:
        rec = store.get_skill(path.stem)
        try:
            mtime = path.stat().st_mtime
            days = int((now - mtime) / 86400)
            last_used = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except OSError:
            days = 0
            last_used = "?"
        pin = " 📌" if rec.pinned else ""
        state_tag = "" if rec.state == "active" else f" [{rec.state}]"
        lines.append(
            f"  {path.stem}{pin}{state_tag} — last used {last_used} ({days}d ago)"
        )
    lines.append("")
    lines.append("Use /skills show <name> · /skills pin <name> · /skills history <name>")
    return "\n".join(lines)


def cmd_skills_show(name: str) -> str:
    """Display the SKILL.md content for *name* (from auto/ first, then root)."""
    from prometheus.config.paths import get_config_dir

    if not name:
        return "Usage: /skills show <name>"

    skills_root = get_config_dir() / "skills"
    candidates = [
        skills_root / "auto" / f"{name}.md",
        skills_root / f"{name}.md",
    ]
    for path in candidates:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if len(content) > 3500:
                content = content[:3500] + "\n\n…truncated."
            return f"{path}\n\n{content}"
    return f"Skill not found: {name}"


def cmd_skills_pin(name: str) -> str:
    """Mark *name* as pinned so the Curator never auto-prunes it."""
    from prometheus.learning.skill_state import SkillStateStore

    if not name:
        return "Usage: /skills pin <name>"
    store = SkillStateStore()
    rec = store.set_pinned(name, True)
    return f"📌 Pinned: {name} (state: {rec.state})"


def cmd_skills_unpin(name: str) -> str:
    from prometheus.learning.skill_state import SkillStateStore

    if not name:
        return "Usage: /skills unpin <name>"
    store = SkillStateStore()
    rec = store.set_pinned(name, False)
    return f"Unpinned: {name} (state: {rec.state})"


def cmd_skills_history(name: str) -> str:
    """List the SkillRefiner-created backups for *name*."""
    from datetime import datetime
    from prometheus.config.paths import get_config_dir

    if not name:
        return "Usage: /skills history <name>"

    auto_dir = get_config_dir() / "skills" / "auto"
    if not auto_dir.is_dir():
        return "No auto-skills directory."

    # SkillRefiner writes <name>.bak-<ts>.md alongside <name>.md before
    # each refine. List those sorted by timestamp.
    backups = sorted(
        auto_dir.glob(f"{name}.bak-*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    current = auto_dir / f"{name}.md"

    lines: list[str] = [f"History for {name}"]
    if current.exists():
        when = datetime.fromtimestamp(current.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  current: {when}")
    else:
        lines.append("  current: (file not found in auto/)")

    if not backups:
        lines.append("  no refinement backups")
    else:
        for b in backups[:20]:
            when = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {b.name} — {when}")
        if len(backups) > 20:
            lines.append(f"  …and {len(backups) - 20} older")
    return "\n".join(lines)


def cmd_curator_show() -> str:
    """Display the most recent Curator REPORT.md."""
    from prometheus.config.paths import get_config_dir
    from prometheus.learning.skill_state import SkillStateStore

    store = SkillStateStore()
    cur = store.curator()
    path = cur.last_report_path
    if path:
        try:
            content = Path(path).read_text(encoding="utf-8")
            if len(content) > 3500:
                content = content[:3500] + "\n\n…truncated."
            return content
        except OSError:
            pass
    # Fall back to most-recent on disk under ~/.prometheus/curator/.
    reports_root = get_config_dir() / "curator"
    if reports_root.is_dir():
        runs = sorted(reports_root.glob("*/REPORT.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            content = runs[0].read_text(encoding="utf-8")
            if len(content) > 3500:
                content = content[:3500] + "\n\n…truncated."
            return content
    return "No Curator runs yet. Try /curator run to trigger one."


def cmd_curator_status() -> str:
    """Show last/next run, run count, pinned skills."""
    import time as _time
    from datetime import datetime
    from prometheus.learning.curator import get_curator
    from prometheus.learning.skill_state import SkillStateStore

    store = SkillStateStore()
    cur = store.curator()
    skills = store.list_skills()
    pinned = [name for name, rec in skills.items() if rec.pinned]

    lines: list[str] = ["Curator status"]
    instance = get_curator()
    if instance is None:
        lines.append("  daemon: not wired (disabled in config?)")
    else:
        lines.append(
            f"  interval: {instance._interval}s "
            f"({instance._interval / 86400:.1f}d)"
        )

    if cur.last_run_at:
        when = datetime.fromtimestamp(cur.last_run_at).strftime("%Y-%m-%d %H:%M")
        ago = int((_time.time() - cur.last_run_at) / 60)
        lines.append(f"  last run: {when} ({ago}m ago)")
        if instance is not None:
            next_at = cur.last_run_at + instance._interval
            lines.append(
                f"  next run: {datetime.fromtimestamp(next_at).strftime('%Y-%m-%d %H:%M')}"
            )
    else:
        lines.append("  last run: (none yet)")

    lines.append(f"  total runs: {cur.run_count}")
    lines.append(f"  paused: {cur.paused}")
    lines.append(f"  pinned skills: {len(pinned)}")
    if pinned:
        lines.append(f"    {', '.join(pinned[:10])}")
        if len(pinned) > 10:
            lines.append(f"    …and {len(pinned) - 10} more")
    if cur.last_report_path:
        lines.append(f"  last report: {cur.last_report_path}")
    return "\n".join(lines)


async def cmd_curator_run(*, dry_run: bool = False) -> str:
    """Trigger an immediate Curator pass. Returns a one-line summary."""
    from prometheus.learning.curator import get_curator

    instance = get_curator()
    if instance is None:
        return "Curator not wired (check learning.curator_enabled in config)."
    try:
        run = await instance.run_once(dry_run=dry_run)
    except Exception as exc:
        return f"Curator run failed: {exc}"
    mode = " (dry-run)" if dry_run else ""
    return (
        f"📋 Curator run{mode} — {run.skills_reviewed} reviewed, "
        f"{len(run.auto_transitions)} auto-transitions, "
        f"{len(run.consolidations)} consolidation suggestions, "
        f"{len(run.prunings)} archived.\n"
        f"Report: {run.report_path}"
    )


# Notification mode persistence — /notifications quiet|verbose|off
_NOTIFICATIONS_MODE_FILE = "notifications_mode"
_VALID_NOTIF_MODES = ("off", "quiet", "verbose")


def get_notifications_mode(default: str = "quiet") -> str:
    """Read the runtime notification override (or *default*)."""
    from prometheus.config.paths import get_config_dir

    path = get_config_dir() / _NOTIFICATIONS_MODE_FILE
    try:
        mode = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return default
    return mode if mode in _VALID_NOTIF_MODES else default


def _render_tool_registration_section(
    lines: list[str],
    tel: Any,
    since_ts: float,
    aggregate: dict[str, Any],
) -> None:
    """Render the per-tool registration-failure breakdown into ``lines``.

    Phase 2 (orphan-tools): every ``try_register`` call writes a
    ``subsystem_runs`` row with ``subsystem='tool_registration'`` and
    ``operation=<display_name>``. Failures carry the exception type/msg
    in ``summary_json`` so we can show *which* tool broke *how*, not just
    a count.

    Only called when ``aggregate['failed'] > 0`` — the section is
    suppressed for clean startups per the audit spec.
    """
    import json as _json
    runs = int(aggregate.get("runs", 0))
    success = int(aggregate.get("success", 0))
    failed = int(aggregate.get("failed", 0))

    lines.append("")
    lines.append("🛠  Tool registration:")
    lines.append(f"   ✅ {success} of {runs} registered")
    lines.append(f"   ⚠  {failed} failed:")

    try:
        rows = tel.runs_since(since_ts, subsystem="tool_registration", limit=500)
    except Exception:
        rows = []

    # Group failures by (tool, exception_type) and tally occurrences.
    # The most-recent message and module_path for the group are kept so the
    # rendered line is "this is what happened the last time it broke".
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("outcome") != "failed":
            continue
        tool = row.get("operation") or "?"
        summary_json = row.get("summary_json") or "{}"
        try:
            summary = _json.loads(summary_json) if summary_json else {}
        except Exception:
            summary = {}
        exc_type = str(summary.get("exception_type") or "Exception")
        key = (tool, exc_type)
        entry = grouped.setdefault(key, {
            "occurrences": 0,
            "exception_msg": summary.get("exception_msg") or "",
            "module_path": summary.get("module_path") or "",
        })
        entry["occurrences"] += 1
        # Newer rows come first from runs_since (ORDER BY timestamp DESC);
        # the first one we see for a group is the most recent.
        if not entry["exception_msg"] and summary.get("exception_msg"):
            entry["exception_msg"] = summary.get("exception_msg") or ""

    # Render in stable order: most recent failure first
    for (tool, exc_type), entry in sorted(
        grouped.items(),
        key=lambda kv: (-int(kv[1]["occurrences"]), kv[0][0]),
    ):
        msg = (entry["exception_msg"] or "").strip()
        msg = (msg[:80] + "…") if len(msg) > 80 else msg
        occ = int(entry["occurrences"])
        suffix = f" [{occ} occurrences]" if occ > 1 else ""
        if msg:
            lines.append(f"      - {tool}: {exc_type} ({msg}){suffix}")
        else:
            lines.append(f"      - {tool}: {exc_type}{suffix}")


def cmd_health(verbose: bool = False, since_hours: float = 24.0) -> str:
    """Surface silent-failure telemetry — Sprint 4 Work Stream 3.

    Reads from the live ``ToolCallTelemetry`` singleton (set by daemon.py)
    so the same content powers Telegram, Slack, and any future surface
    without threading the handle through.

    Args:
        verbose: when True, include the full per-subsystem breakdown +
            traceback excerpts from the 5 most recent silent failures.
        since_hours: lookback window (default 24h).

    Output shape::

        🩺 Prometheus Health — last 24h

        ✅ Tool calls:        1,247  (3 failures, 0 silent)
        ✅ curator:           3 / 3 successful
        ⚠  skill_creator:     21 invocations, 19 successful, 2 SILENT FAILURES
        ...

        Silent failures (most recent 5):
          2026-05-21 14:32  skill_creator/_call_model  ValueError: …
          ...
    """
    import time as _time
    from datetime import datetime
    from prometheus.telemetry.tracker import get_telemetry_handle

    tel = get_telemetry_handle()
    if tel is None:
        return (
            "🩺 Health: telemetry not wired.\n"
            "(daemon hasn't called set_telemetry_handle — restart required)"
        )

    since_ts = _time.time() - max(0.0, float(since_hours)) * 3600.0
    try:
        summary = tel.health_summary(since=since_ts)
    except Exception as exc:
        return f"🩺 Health: telemetry query failed — {exc}"

    lines: list[str] = [
        f"🩺 Prometheus Health — last {since_hours:.0f}h",
        "",
    ]

    tc = summary.get("tool_calls", {})
    tc_total = int(tc.get("total", 0))
    tc_fail = int(tc.get("failures", 0))
    tc_succ_rate = float(tc.get("success_rate", 0.0))
    tc_icon = "✅" if tc_fail == 0 else ("⚠ " if tc_succ_rate >= 0.8 else "❌")
    # With zero invocations, "0.0% success" is mathematically meaningless —
    # render "n/a" instead. Otherwise use a 1-decimal success rate so
    # "4 failures in 1247" doesn't render as "100% success" via rounding.
    if tc_total == 0:
        rate_str = "n/a"
    else:
        rate_str = (
            f"{tc_succ_rate * 100:.1f}% success"
            if tc_succ_rate < 1.0 else "100% success"
        )
    lines.append(
        f"{tc_icon} Tool calls:        {tc_total:>5,}  "
        f"({tc_fail} failures, {rate_str})"
    )

    subsystems = summary.get("subsystems", {}) or {}
    # Phase 2 (orphan-tools): tool_registration gets its own dedicated
    # section below — pop it here so the generic loop doesn't render the
    # row twice. The detailed section is suppressed entirely when clean
    # (no failures), per docs/audits/ORPHAN-TOOLS-AUDIT.md Phase 2 spec.
    tool_reg_aggregate = subsystems.pop("tool_registration", None)
    if not subsystems:
        lines.append("")
        lines.append(
            "_no autonomous-subsystem activity recorded in this window_"
        )
    else:
        lines.append("")
        for name in sorted(subsystems):
            row = subsystems[name]
            runs = int(row.get("runs", 0))
            success = int(row.get("success", 0))
            partial = int(row.get("partial", 0))
            failed = int(row.get("failed", 0))
            silent = int(row.get("silent_failures", 0))
            icon = "✅" if (failed == 0 and silent == 0) else (
                "⚠ " if (success + partial) > 0 else "❌"
            )
            tail = (
                f"{success} / {runs} successful"
                if (runs and silent == 0)
                else (
                    f"{runs} invocations, {success} successful, "
                    f"{silent} SILENT FAILURES" if silent else
                    f"{runs} runs ({success} ok, {partial} partial, {failed} failed)"
                )
            )
            lines.append(f"{icon} {name:<18} {tail}")

    # Tool registration section (Phase 2 — orphan-tools). Shown only when
    # at least one tool failed to register in the window. Each per-tool
    # row carries the import-time exception, grouped by (tool, type).
    if tool_reg_aggregate and int(tool_reg_aggregate.get("failed", 0)) > 0:
        _render_tool_registration_section(
            lines, tel, since_ts, tool_reg_aggregate,
        )

    recent = summary.get("recent_silent_failures", []) or []
    if recent:
        lines.append("")
        lines.append(f"Silent failures (most recent {len(recent)}):")
        for row in recent:
            when = datetime.fromtimestamp(float(row["timestamp"])).strftime(
                "%Y-%m-%d %H:%M"
            )
            sub = row.get("subsystem", "?")
            op = row.get("operation", "?")
            exc_type = row.get("exception_type", "?")
            msg = (row.get("exception_msg") or "")[:100]
            lines.append(f"  {when}  {sub}/{op}  {exc_type}: {msg}")

    if verbose:
        # Add traceback excerpts for the recent failures.
        if recent:
            lines.append("")
            lines.append("--- Recent silent-failure tracebacks (verbose) ---")
            for row in recent:
                lines.append("")
                lines.append(
                    f"# {row.get('subsystem','?')}/{row.get('operation','?')} "
                    f"@ {datetime.fromtimestamp(float(row['timestamp']))}"
                )
                tb = (row.get("traceback") or "").strip()
                if tb:
                    # Cap each traceback at ~800 chars to keep Telegram-friendly.
                    if len(tb) > 800:
                        tb = tb[:800] + "\n  …truncated…"
                    lines.append(tb)
        # Show context JSON for each failure when present.
        for row in recent:
            ctx = row.get("context")
            if ctx:
                lines.append(f"  context: {ctx}")

    if not recent:
        lines.append("")
        lines.append("_no silent failures detected in this window_  ✨")

    lines.append("")
    lines.append(
        "Run `/health verbose` for tracebacks. Window: `/health 168` for 7d."
    )
    return "\n".join(lines)


def cmd_notifications(mode: str = "") -> str:
    """Show or set the notification mode (off | quiet | verbose)."""
    from prometheus.config.paths import get_config_dir

    arg = (mode or "").strip().lower()
    current = get_notifications_mode()
    if not arg:
        return (
            f"Notifications: {current}\n"
            "Set: /notifications off | quiet | verbose"
        )
    if arg not in _VALID_NOTIF_MODES:
        return f"Unknown mode: {arg}. Use one of: {' | '.join(_VALID_NOTIF_MODES)}"
    path = get_config_dir() / _NOTIFICATIONS_MODE_FILE
    path.write_text(arg + "\n", encoding="utf-8")
    return f"Notifications: {arg} (was {current})"


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
