"""Shared command handlers for gateway adapters (Telegram, Slack).

Platform-agnostic command logic. Each function returns a string
that the adapter sends via its own transport.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prometheus.tools.base import ToolRegistry

log = logging.getLogger(__name__)


def cmd_help() -> str:
    """Return help text listing available commands."""
    return (
        "Prometheus — Sovereign AI Agent\n"
        "\n"
        "Commands:\n"
        "/status    — Model, uptime, tools, memory, SENTINEL, queues\n"
        "/model     — Current model name and provider\n"
        "/profile   — Show/switch agent profiles\n"
        "/anatomy   — Hardware, GPU, VRAM, infrastructure\n"
        "/doctor    — Diagnostic health check against model registry\n"
        "/wiki      — Wiki stats and recent entries\n"
        "/note      — Save a manual fact: /note [@entity] <text>\n"
        "/sentinel  — SENTINEL subsystem status\n"
        "/events    — Recent signal-bus events (recent | skills | memory | curator | show <id>)\n"
        "/steer     — Inject mid-turn guidance (arrives after next tool call)\n"
        "/queue     — Queue a follow-up turn for after the current one ends\n"
        "/unqueue   — Drop the most recently queued prompt\n"
        "/clearsteers — Drop all pending steers without surfacing them\n"
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


def cmd_note(store, raw: str) -> str:
    """Write a manual, max-trust note to memory.db (canonical store only).

    ``/note <text>`` files under the default ``Notes`` entity; ``/note @Entity
    <text>`` targets ``Entity``. Never writes the wiki — compile projects it,
    and a manual fact earns a page on first mention (the >= 2 threshold is
    bypassed). On a dedup hit it upgrades the existing row to manual rather
    than duplicating it.
    """
    text = (raw or "").strip()
    if not text:
        return "Usage: /note [@entity] <text>"
    entity = "Notes"
    if text.startswith("@"):
        head, _, rest = text.partition(" ")
        entity = head[1:].strip() or "Notes"
        text = rest.strip()
        if not text:
            return "Usage: /note @entity <text>"
    if store is None:
        return "Memory store unavailable — note not saved."
    # Ack ONLY on a confirmed, committed write. persist_memory raises on
    # failure (it never swallows); surface that as an explicit "not saved"
    # rather than a false "Noted." Closes the false-ack pattern.
    try:
        store.persist_memory(
            "note",
            entity,
            text,
            1.0,
            source_event_ids=["manual"],
            manual=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface, never false-ack
        log.warning("cmd_note: persist failed for [[%s]]: %s", entity, exc)
        return (
            f"⚠ Note NOT saved ({type(exc).__name__}) — nothing was written. "
            "Try again."
        )
    return f"Noted under [[{entity}]] — manual, max trust."


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


def _format_gpu_processes(procs: list) -> list[str]:
    """Render the per-GPU process list under a GPU section.

    Returns one indented line per process (PID, name, VRAM in MB or GB).
    Empty list yields no lines so the section just shows VRAM totals
    without an awkward "Processes:" header followed by nothing.
    Processes are sorted by VRAM descending so the heaviest user shows
    first — that's almost always the question the reader is asking.
    """
    if not procs:
        return []
    sorted_procs = sorted(procs, key=lambda p: p.memory_mb, reverse=True)
    out: list[str] = ["Processes:"]
    for p in sorted_procs:
        # Display VRAM as GB when >=1024 MB, else MB. Keeps the most
        # common case (model loaded, multi-GB) clean while still
        # reporting tiny helpers (browser/X server) honestly.
        if p.memory_mb >= 1024:
            mem_str = f"{p.memory_mb / 1024:.1f} GB"
        else:
            mem_str = f"{p.memory_mb} MB"
        out.append(f"  {p.name} (PID {p.pid}): {mem_str}")
    return out


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

    # GPU — honest two-card reporting.
    #
    # state.gpu_* now always reflects the *inference* GPU (the box where
    # the LLM runs). When inference is remote, state.local_gpu_* may
    # also be populated for this box's own card (used by ComfyUI / local
    # Ollama / etc.). We label each line so the agent can't conflate them.
    inference_host_label = state.gpu_inference_host or inf_host
    if state.gpu_name:
        if state.gpu_is_remote:
            header = f"\nGPU (inference, remote @ {inference_host_label}): {state.gpu_name}"
        else:
            header = f"\nGPU (inference, local): {state.gpu_name}"
        lines.append(header)
        if state.gpu_vram_total_mb:
            used_gb = (state.gpu_vram_used_mb or 0) / 1024
            free_gb = (state.gpu_vram_free_mb or 0) / 1024
            total_gb = state.gpu_vram_total_mb / 1024
            lines.append(
                f"VRAM: {used_gb:.1f} / {total_gb:.1f} GB ({free_gb:.1f} GB free)"
            )
        lines.extend(_format_gpu_processes(state.gpu_processes))
    elif state.gpu_is_remote:
        # Remote inference but probe failed — be loud about WHY rather
        # than substituting local stats (the old bug). The agent should
        # see the probe error so it doesn't make claims about a GPU we
        # couldn't reach.
        reason = state.gpu_probe_error or "stats unavailable"
        lines.append(
            f"\nGPU (inference, remote @ {inference_host_label}): "
            f"unreachable — {reason}"
        )
    elif state.gpu_probe_error:
        lines.append(f"\nGPU: probe failed — {state.gpu_probe_error}")

    # Secondary local GPU — only relevant when inference is remote and
    # this box has its own card (e.g. ComfyUI here, Gemma on the remote).
    # Suppressed when same hostname/card as inference to avoid clutter.
    if state.local_gpu_name and state.gpu_is_remote:
        lines.append(f"\nGPU (local, this box): {state.local_gpu_name}")
        if state.local_gpu_vram_total_mb:
            l_used_gb = (state.local_gpu_vram_used_mb or 0) / 1024
            l_free_gb = (state.local_gpu_vram_free_mb or 0) / 1024
            l_total_gb = state.local_gpu_vram_total_mb / 1024
            lines.append(
                f"VRAM: {l_used_gb:.1f} / {l_total_gb:.1f} GB "
                f"({l_free_gb:.1f} GB free)"
            )
        lines.extend(_format_gpu_processes(state.local_gpu_processes))

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


# ---------------------------------------------------------------------------
# /events — SignalBus Persistence sprint
# ---------------------------------------------------------------------------

# Logical groups for the subcommands. Maps "skills" → ["skill_created", ...].
# Membership lives here (not in signal_bus.py) so the command surface can
# evolve independently of the bus.
_EVENT_GROUPS: dict[str, list[str]] = {
    "skills":  ["skill_created", "skill_refined"],
    "memory":  ["memory_updated"],
    "curator": ["curator_report", "curator_degraded"],
}


def _summarise_event_payload(signal_type: str, payload: dict[str, Any]) -> str:
    """One-line summary of an event payload for the /events recent feed.

    Best-effort: the payload schema varies by signal_type, so we pull a few
    common keys and fall back to the raw type name when none apply.
    """
    if not isinstance(payload, dict):
        return ""
    # Common high-signal keys, in priority order.
    for key in ("name", "skill_name", "path", "skill_path", "summary",
                "outcome", "message", "reason", "memory_kind", "target"):
        val = payload.get(key)
        if val:
            text = str(val)
            return text if len(text) <= 80 else text[:77] + "..."
    return ""


def cmd_events(arg: str = "") -> str:
    """Surface persisted SignalBus events.

    Subcommands:
      (empty) | recent   → last 20 events across all types
      skills             → recent skill_created / skill_refined
      memory             → recent memory_updated
      curator            → recent curator_report / curator_degraded
      show <id>          → full payload for a specific event

    Reads from telemetry.signal_events — the durable tail. Survives daemon
    restart, unlike the in-memory ``SignalBus.recent`` deque.
    """
    from prometheus.telemetry.tracker import get_telemetry_handle

    tel = get_telemetry_handle()
    if tel is None:
        return (
            "📡 /events: telemetry not wired.\n"
            "(daemon hasn't called set_telemetry_handle — restart required)"
        )

    parts = (arg or "").strip().split(maxsplit=1)
    sub = (parts[0] or "recent").lower() if parts else "recent"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "show":
        return _cmd_events_show(tel, rest)

    if sub in _EVENT_GROUPS:
        types = _EVENT_GROUPS[sub]
        header = f"📡 /events {sub} — recent"
        rows = tel.signal_events_since(signal_types=types, limit=20)
    elif sub == "recent":
        header = "📡 /events recent — last 20"
        rows = tel.signal_events_since(limit=20)
    else:
        valid = ", ".join(["recent", *sorted(_EVENT_GROUPS.keys()), "show <id>"])
        return f"📡 /events: unknown subcommand {sub!r}. Valid: {valid}."

    if not rows:
        return f"{header}\n\n_no events recorded in this window yet_"

    lines = [header, ""]
    for row in rows:
        ts = (row.get("timestamp") or "")[:19]  # YYYY-MM-DDTHH:MM:SS
        st = row.get("signal_type", "?")
        summary = _summarise_event_payload(st, row.get("payload") or {})
        tail = f"  {summary}" if summary else ""
        lines.append(f"  [{row.get('id'):>4}] {ts}  {st}{tail}")
    lines.append("")
    lines.append("Run `/events show <id>` for the full payload.")
    return "\n".join(lines)


def _cmd_events_show(tel: Any, rest: str) -> str:
    """Render the full row for a specific event id."""
    import json as _json

    raw = (rest or "").strip()
    if not raw.isdigit():
        return "📡 /events show: provide a numeric event id, e.g. `/events show 42`."
    event_id = int(raw)
    row = tel.signal_event_by_id(event_id)
    if row is None:
        return f"📡 /events show: no event with id={event_id}."

    try:
        payload_pretty = _json.dumps(row["payload"], indent=2, ensure_ascii=False)
    except Exception:
        payload_pretty = str(row["payload"])

    return (
        f"📡 Event #{row['id']}\n"
        f"  timestamp:  {row['timestamp']}\n"
        f"  type:       {row['signal_type']}\n"
        f"  source:     {row['source_subsystem']}\n"
        f"  read_at:    {row['read_at'] or '(unread)'}\n"
        f"\nPayload:\n{payload_pretty}"
    )


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


# ===========================================================================
# Shared slash-command dispatch table
# ===========================================================================
#
# A single, gateway-agnostic table of side-effect-free *formatter* commands,
# keyed by command name. A gateway builds a CommandContext (the handful of
# values these commands need beyond their argument text) and awaits
# run_formatter_command(); the returned string is whatever that surface sends.
#
# This is the "one table" the web (Beacon) slash-router dispatches through —
# see prometheus.web.slash_router. The Telegram and Slack gateways predate it
# and still register their own thin handlers around the same cmd_* functions;
# they can migrate onto this table later with no behaviour change (these
# handlers call the identical cmd_* used today).
#
# DELIBERATELY ABSENT: commands that mutate live daemon state bound to a gateway
# instance — /route, /steer, /queue, /approve, /benchmark, /press, /gepa,
# /symbiote, /audit, … . They are not reachable as pure formatters; surfacing
# them on the web requires threading daemon services into the web bridge and is
# tracked as a separate follow-up. The web router replies to those with an
# explicit "not on web yet" boundary rather than running them.


@dataclass
class CommandContext:
    """Gateway-agnostic inputs for a formatter command.

    A gateway fills what it has; every field has a safe default so a surface
    that lacks one (e.g. the web bridge has no uptime clock or cost tracker)
    degrades gracefully instead of failing. ``tool_registry`` is needed only by
    /status; ``config`` only by /doctor and /beacon; ``session`` /
    ``ensure_session`` only by the session-mutating commands.
    """

    model_name: str = ""
    model_provider: str = ""
    system_prompt: str = ""
    config: dict | None = None
    tool_registry: Any = None
    start_time: float = 0.0
    cost_tracker: Any = None
    # Session-mutating commands (/steer, /queue, /unqueue, /clearsteers):
    # ``session`` is the resolved active ChatSession or None (non-creating);
    # ``ensure_session`` is a no-arg callable that get_or_creates one (used by
    # /queue, which fires on a quiet chat once the first turn starts).
    session: Any = None
    ensure_session: Any = None


async def _fc_help(ctx: CommandContext, args: str) -> str:
    return cmd_help()


async def _fc_status(ctx: CommandContext, args: str) -> str:
    return cmd_status(
        ctx.model_name,
        ctx.model_provider,
        ctx.start_time,
        ctx.tool_registry,
        ctx.cost_tracker,
    )


async def _fc_model(ctx: CommandContext, args: str) -> str:
    return cmd_model(ctx.model_name, ctx.model_provider)


async def _fc_wiki(ctx: CommandContext, args: str) -> str:
    return cmd_wiki()


async def _fc_sentinel(ctx: CommandContext, args: str) -> str:
    return cmd_sentinel()


async def _fc_context(ctx: CommandContext, args: str) -> str:
    return cmd_context(ctx.system_prompt, ctx.model_name)


async def _fc_note(ctx: CommandContext, args: str) -> str:
    from prometheus.tools.builtin.wiki_compile import _memory_store

    if _memory_store is None:
        return "Memory store not initialized — is the daemon fully started?"
    return cmd_note(_memory_store, args)


async def _fc_skills(ctx: CommandContext, args: str) -> str:
    parts = args.split()
    if not parts:
        return cmd_skills()
    sub = parts[0].lower()
    name = " ".join(parts[1:]).strip()
    if sub == "list":
        return cmd_skills_auto_list()
    if sub == "show":
        return cmd_skills_show(name)
    if sub == "pin":
        return cmd_skills_pin(name)
    if sub == "unpin":
        return cmd_skills_unpin(name)
    if sub == "history":
        return cmd_skills_history(name)
    return (
        f"Unknown subcommand: {sub}\n"
        "Use: /skills [list | show <name> | pin <name> | unpin <name> | history <name>]"
    )


async def _fc_memory(ctx: CommandContext, args: str) -> str:
    parts = args.split()
    if not parts:
        return (
            "Memory commands:\n"
            "  /memory show           — MEMORY.md content\n"
            "  /memory show user      — USER.md content\n"
            "  /memory limits         — char ceilings + usage"
        )
    sub = parts[0].lower()
    tail = " ".join(parts[1:]).strip().lower()
    if sub == "show":
        target = "user" if tail == "user" else "memory"
        return cmd_memory_show(target=target)
    if sub == "limits":
        return cmd_memory_limits()
    return f"Unknown subcommand: {sub}\nUse: /memory [show [user] | limits]"


async def _fc_curator(ctx: CommandContext, args: str) -> str:
    parts = args.split()
    if not parts:
        return (
            "Curator commands:\n"
            "  /curator status        — last/next run, pinned skills\n"
            "  /curator show          — most recent REPORT.md\n"
            "  /curator run           — trigger an immediate pass\n"
            "  /curator run dry       — dry-run (no file moves)"
        )
    sub = parts[0].lower()
    if sub == "show":
        return cmd_curator_show()
    if sub == "status":
        return cmd_curator_status()
    if sub == "run":
        dry = len(parts) >= 2 and parts[1].lower().startswith("dry")
        return await cmd_curator_run(dry_run=dry)
    return f"Unknown subcommand: {sub}\nUse: /curator [show | status | run [dry]]"


async def _fc_events(ctx: CommandContext, args: str) -> str:
    return cmd_events(arg=args)


async def _fc_health(ctx: CommandContext, args: str) -> str:
    verbose = False
    since_hours = 24.0
    for tok in args.split():
        t = tok.strip().lower()
        if t == "verbose":
            verbose = True
            continue
        try:
            since_hours = float(t)
        except ValueError:
            pass
    return cmd_health(verbose=verbose, since_hours=since_hours)


async def _fc_notifications(ctx: CommandContext, args: str) -> str:
    return cmd_notifications(mode=args)


async def _fc_anatomy(ctx: CommandContext, args: str) -> str:
    return await cmd_anatomy()


async def _fc_doctor(ctx: CommandContext, args: str) -> str:
    return await cmd_doctor(ctx.config)


async def _fc_profile(ctx: CommandContext, args: str) -> str:
    # The web surface doesn't track an active profile across calls (that state
    # lives on the Telegram gateway instance), so the "current" marker is the
    # default. cmd_profile is descriptive — it never re-wires the running agent.
    return cmd_profile(arg=args.strip(), current="full")


async def _fc_beacon(ctx: CommandContext, args: str) -> str:
    return cmd_beacon(ctx.config or {})


# command name -> async handler(ctx, args) -> str
_FORMATTER_COMMANDS: dict[str, Any] = {
    "help": _fc_help,
    "status": _fc_status,
    "model": _fc_model,
    "wiki": _fc_wiki,
    "sentinel": _fc_sentinel,
    "context": _fc_context,
    "note": _fc_note,
    "skills": _fc_skills,
    "memory": _fc_memory,
    "curator": _fc_curator,
    "events": _fc_events,
    "health": _fc_health,
    "notifications": _fc_notifications,
    "anatomy": _fc_anatomy,
    "doctor": _fc_doctor,
    "profile": _fc_profile,
    "beacon": _fc_beacon,
}


def is_formatter_command(name: str) -> bool:
    """True if ``name`` is a side-effect-free command in the shared table."""
    return name.lower() in _FORMATTER_COMMANDS


def formatter_command_names() -> frozenset[str]:
    """The set of command names the shared table can serve."""
    return frozenset(_FORMATTER_COMMANDS)


async def run_formatter_command(
    name: str, args: str, ctx: CommandContext
) -> str | None:
    """Run a side-effect-free slash command by name.

    Returns the command's text result, or ``None`` if ``name`` is not a
    formatter command (the caller decides whether to fall through to the agent
    or reply with a boundary). A formatter that raises is caught and surfaced as
    an error string — a bad command never crashes the chat path.
    """
    handler = _FORMATTER_COMMANDS.get(name.lower())
    if handler is None:
        return None
    try:
        return await handler(ctx, args)
    except Exception as exc:  # noqa: BLE001 — fail loud to the user, not the loop
        log.exception("formatter command /%s failed", name)
        return f"⚠ /{name} failed: {exc}"


# ===========================================================================
# Session-mutating commands (steering + queued prompts)
# ===========================================================================
#
# These DO mutate state — they enqueue steers / follow-up prompts onto a live
# ChatSession the agent loop drains — so they're separate from the side-effect-
# free formatter table above. The logic is gateway-agnostic: each takes the
# resolved ChatSession (or None) plus arg text and returns the reply. The
# gateway owns session RESOLUTION (creating vs non-creating), because that
# differs per command and per surface.
#
# Telegram's _cmd_steer/_cmd_queue/_cmd_unqueue/_cmd_clear_steers call these;
# the web slash-router dispatches them through run_session_command() with a
# CommandContext carrying the session. One implementation, two surfaces.


def _preview(text: str, limit: int = 80) -> str:
    """Trim ``text`` to ``limit`` chars with an ellipsis (matches the gateways)."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cmd_steer(session: Any, text: str) -> str:
    """Inject mid-turn guidance into the running session.

    Arrives at the agent as a system-prompt addendum on the next model call
    (after the current tool batch). ``session`` is the resolved ChatSession or
    None (no active session — non-creating, since there's nothing to steer).
    """
    text = (text or "").strip()
    if not text:
        return (
            "/steer <text> — inject mid-turn guidance.\n"
            "Arrives after the next tool call. Example:\n"
            "/steer focus on Ubuntu, skip the Mac instructions"
        )
    if session is None:
        return (
            "No active session yet. Send a message first, then /steer while the "
            "agent is running."
        )
    if not session.enqueue_steer(text):
        return "Empty steer — nothing queued."
    return (
        f"📍 Steered: {_preview(text)}\n"
        f"   Arrives after the next tool call. "
        f"Pending: {len(session.queued_steers)}."
    )


def cmd_queue(session: Any, text: str) -> str:
    """Queue ``text`` as a fresh turn after the current one ends.

    ``session`` should be a get_or_created ChatSession (a queued prompt on a
    quiet chat fires when the user kicks off their first message). The caller
    creates it only when ``text`` is non-empty, so a bare ``/queue`` (usage)
    never materialises a session.
    """
    text = (text or "").strip()
    if not text:
        return (
            "/queue <text> — line up a follow-up turn.\n"
            "Runs after the current task ends."
        )
    if session is None:
        return "No active session."
    if not session.enqueue_prompt(text):
        return "Empty prompt — nothing queued."
    return (
        f"📥 Queued: {_preview(text)}\n"
        f"   Position: {len(session.queued_prompts)}. Fires when current turn ends."
    )


def cmd_unqueue(session: Any) -> str:
    """Drop the most recently queued prompt (LIFO)."""
    if session is None or not session.queued_prompts:
        return "No queued prompts to drop."
    dropped = session.queued_prompts.pop()
    return (
        f"🗑  Unqueued: {_preview(dropped)}\n"
        f"   Remaining: {len(session.queued_prompts)}."
    )


def cmd_clearsteers(session: Any) -> str:
    """Drop all pending steers without surfacing them."""
    if session is None:
        return "No active session."
    n = session.clear_steers()
    return f"🧹 Cleared {n} pending steer{'s' if n != 1 else ''}."


async def _sc_steer(ctx: CommandContext, args: str) -> str:
    return cmd_steer(ctx.session, args)


async def _sc_queue(ctx: CommandContext, args: str) -> str:
    # Create the session only when there's something to queue (parity with
    # Telegram, which doesn't materialise a session for a bare /queue usage).
    if args.strip() and ctx.ensure_session is not None:
        return cmd_queue(ctx.ensure_session(), args)
    return cmd_queue(ctx.session, args)


async def _sc_unqueue(ctx: CommandContext, args: str) -> str:
    return cmd_unqueue(ctx.session)


async def _sc_clearsteers(ctx: CommandContext, args: str) -> str:
    return cmd_clearsteers(ctx.session)


# command name -> async handler(ctx, args) -> str
_SESSION_COMMANDS: dict[str, Any] = {
    "steer": _sc_steer,
    "queue": _sc_queue,
    "unqueue": _sc_unqueue,
    "clearsteers": _sc_clearsteers,
}


def is_session_command(name: str) -> bool:
    """True if ``name`` is a session-mutating command in the shared table."""
    return name.lower() in _SESSION_COMMANDS


def session_command_names() -> frozenset[str]:
    """The set of session-mutating command names the shared table can serve."""
    return frozenset(_SESSION_COMMANDS)


async def run_session_command(
    name: str, args: str, ctx: CommandContext
) -> str | None:
    """Run a session-mutating slash command by name.

    Returns the command's text result, or ``None`` if ``name`` is not a session
    command. Like run_formatter_command, a handler that raises is caught and
    surfaced as an error string rather than crashing the chat path.
    """
    handler = _SESSION_COMMANDS.get(name.lower())
    if handler is None:
        return None
    try:
        return await handler(ctx, args)
    except Exception as exc:  # noqa: BLE001 — fail loud to the user, not the loop
        log.exception("session command /%s failed", name)
        return f"⚠ /{name} failed: {exc}"


# ===========================================================================
# SPRINT G1 — gateway parity: logic extracted from telegram.py handlers
# ===========================================================================
#
# Everything below was Telegram-embedded command logic, moved here verbatim
# (mechanical extraction — reply text preserved byte-for-byte, pinned by
# tests/test_gateway_command_pins.py). Each function is platform-independent:
# no adapter ``self``, no platform APIs. Subsystem objects (router via
# agent_loop, approval queue, GEPA engine, symbiote coordinator, printing
# press, escalation engine) are injected by the calling gateway.
#
# Conventions:
#   * ``prefix`` — the surface's slash-command prefix. Telegram passes the
#     default ``"/"``; Slack passes ``"/prometheus-"`` (its commands are
#     workspace-global and namespaced). Reply text mentioning other commands
#     is built with the prefix so each surface shows commands the user can
#     actually type.
#   * ``send`` — for multi-message flows (gepa run, symbiote, audit, press
#     install) the gateway injects ``async send(text) -> None`` bound to the
#     invoking chat/channel. Single-reply commands just return a string.


# Display labels for /claude, /gpt, etc. (moved from telegram.py — Sprint 22
# GRAFT-ROUTER-WIRE Phase 4).
PROVIDER_PRESET_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude (anthropic)",
    "gpt": "GPT (openai)",
    "gemini": "Gemini (google)",
    "xai": "Grok (xai)",
    # CLOUD EXPANSION (2026-07)
    "deepseek": "DeepSeek (deepseek)",
    "kimi": "Kimi (moonshot)",
    "glm": "GLM (z.ai)",
    "mimo": "MiMo (xiaomi)",
}


# ---------------------------------------------------------------------------
# Provider overrides — /claude /gpt /gemini /xai /grok /local /route
# ---------------------------------------------------------------------------


def cmd_provider_override(
    agent_loop: Any,
    prometheus_config: dict | None,
    session_key: str,
    preset_name: str,
    *,
    prefix: str = "/",
) -> tuple[str, bool]:
    """Set a per-session provider override (shared /claude, /gpt, … core).

    Validates router availability, ``overrides_enabled``, and the preset's
    API-key env var; records the override on the router keyed by
    ``session_key`` (``telegram:<chat_id>`` / ``slack:<channel_id>``).

    Returns ``(reply_text, applied)`` — ``applied`` is True only when the
    override was actually recorded, so the gateway can decide whether to
    dispatch an inline message through the new provider.
    """
    router = getattr(agent_loop, "_model_router", None)
    if router is None:
        return (
            "Routing is not enabled. Provider overrides require a "
            "configured router in prometheus.yaml.",
            False,
        )

    if not getattr(router.config, "overrides_enabled", True):
        log.warning(
            "Override command %s%s invoked for session %s but "
            "router.overrides.enabled is False — ignoring.",
            prefix, preset_name, session_key,
        )
        return (
            "Direct-mode provider overrides are disabled.\n"
            "Set router.overrides.enabled: true in config/prometheus.yaml "
            "and restart the daemon to enable.",
            False,
        )

    from prometheus.router.model_router import resolve_slash_command_target
    preset = resolve_slash_command_target(preset_name, prometheus_config or {})
    if preset is None:
        return (f"Unknown override preset '{preset_name}'.", False)

    # Early feedback if no credential is present — beats failing on the
    # user's next message with an opaque ValueError from the provider
    # registry. A subscription source (e.g. SuperGrok OAuth for xai) counts
    # as a credential and ALWAYS wins over the API key, mirroring the
    # registry's runtime precedence (_resolve_xai_credential).
    from prometheus.providers.credentials import credential_status

    api_key_env = preset.get("api_key_env", "")
    cred = credential_status(preset.get("provider", ""), api_key_env)
    if api_key_env and cred["mode"] is None:
        display = PROVIDER_PRESET_DISPLAY_NAMES.get(preset_name, preset_name)
        text = (
            f"{display} requires {api_key_env} to be set in the "
            f"environment.\n"
            f"Add it to ~/.config/prometheus/env and restart the daemon "
            f"(systemctl --user restart prometheus), then try {prefix}{preset_name} "
            f"again."
        )
        if cred["subscription_label"]:
            text += (
                f"\nOr use your {cred['subscription_label']} instead — "
                f"{cred['subscription_hint']}."
            )
        return (text, False)

    # Record the override. set_override raises ValueError if called with a
    # reserved session_id, but gateway session keys are never reserved.
    router.set_override(session_key, dict(preset))
    display = PROVIDER_PRESET_DISPLAY_NAMES.get(preset_name, preset_name)
    log.info(
        "Provider override for session %s → %s/%s (auth=%s)",
        session_key, preset.get("provider"), preset.get("model"), cred["mode"],
    )
    auth_line = f"Auth: {cred['detail']}\n" if cred["detail"] else ""
    return (
        f"Switched to {display}.\n"
        f"Model: {preset.get('model', '?')}\n"
        f"{auth_line}"
        f"Use {prefix}local to return to primary, {prefix}route to check.",
        True,
    )


def cmd_local_override(
    agent_loop: Any,
    session_key: str,
    model_name: str,
    model_provider: str,
) -> str:
    """Clear a per-session provider override (shared /local core).

    Silent no-op when no override was set (distinct reply text).
    """
    router = getattr(agent_loop, "_model_router", None)
    had_override = False
    if router is not None:
        had_override = router.get_override_for_session(session_key) is not None
        router.clear_override(session_key)
        if had_override:
            log.info(
                "Cleared provider override for session %s (back to primary)",
                session_key,
            )

    primary = f"{model_provider or '?'}/{model_name or '?'}"
    if had_override:
        return f"Back to primary ({primary})."
    return f"Already on primary ({primary}). No override was set."


def cmd_route(
    agent_loop: Any,
    session_key: str,
    model_name: str,
    model_provider: str,
    *,
    prefix: str = "/",
) -> str:
    """Show the current effective provider for this session (shared /route core).

    Reports one of three states:
      - override active: "{provider}/{model}  (override)"
      - no router:       "{primary_provider}/{primary_model}  (no router)"
      - primary:         "{primary_provider}/{primary_model}  (primary)"

    Followed by a list of available override commands.
    """
    router = getattr(agent_loop, "_model_router", None)
    lines = ["Route"]

    if router is None:
        lines.append(
            f"Active: {model_provider or '?'}/"
            f"{model_name or '?'}  (no router)"
        )
    else:
        override = router.get_override_for_session(session_key)
        if override is not None:
            cfg = override.provider_config
            lines.append(
                f"Active: {cfg.get('provider', '?')}/"
                f"{cfg.get('model', '?')}  (override)"
            )
            # Say HOW the override provider is authed (subscription vs API
            # key) when determinable; silent otherwise so mocked/env-less
            # configs don't print noise.
            from prometheus.providers.credentials import credential_status
            cred = credential_status(
                cfg.get("provider", ""), cfg.get("api_key_env", "")
            )
            if cred["detail"]:
                lines.append(f"Auth: {cred['detail']}")
            lines.append(f"Clear with: {prefix}local")
        else:
            lines.append(
                f"Active: {model_provider or '?'}/"
                f"{model_name or '?'}  (primary)"
            )

    lines.append("")
    lines.append("Override commands:")
    lines.append(f"  {prefix}claude  — Anthropic Claude")
    lines.append(f"  {prefix}gpt     — OpenAI GPT")
    lines.append(f"  {prefix}gemini  — Google Gemini")
    lines.append(f"  {prefix}xai     — xAI Grok")
    lines.append(f"  {prefix}grok    — alias for {prefix}xai")
    lines.append(f"  {prefix}deepseek — DeepSeek")
    lines.append(f"  {prefix}kimi    — Kimi (Moonshot)")
    lines.append(f"  {prefix}glm     — GLM (Z.ai)")
    lines.append(f"  {prefix}mimo    — MiMo (Xiaomi)")
    lines.append(f"  {prefix}local   — back to primary")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Approval queue — /approve /deny /pending
# ---------------------------------------------------------------------------


async def cmd_approve(queue: Any, request_id: str, *, prefix: str = "/") -> str:
    """Approve a pending tool request (shared /approve core)."""
    if not request_id:
        return f"Usage: {prefix}approve {{request_id}}"
    if queue is None:
        return "Approval queue not active."
    ok = await queue.approve(request_id)
    if ok:
        return f"Approved: {request_id}"
    return f"No pending request: {request_id}"


async def cmd_deny(queue: Any, request_id: str, *, prefix: str = "/") -> str:
    """Deny a pending tool request (shared /deny core)."""
    if not request_id:
        return f"Usage: {prefix}deny {{request_id}}"
    if queue is None:
        return "Approval queue not active."
    ok = await queue.deny(request_id)
    if ok:
        return f"Denied: {request_id}"
    return f"No pending request: {request_id}"


def cmd_pending(queue: Any) -> str:
    """List pending approval requests (shared /pending core)."""
    if queue is None:
        return "Approval queue not active."
    pending = queue.list_pending()
    if not pending:
        return "No pending requests."
    lines = ["Pending approval requests:"]
    for action in pending:
        lines.append(
            f"  {action.request_id}: {action.tool_name} — {action.description}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Teacher escalation — /escalations
# ---------------------------------------------------------------------------


def cmd_escalations(engine: Any) -> str:
    """Teacher-escalation counters + budget state (shared /escalations core)."""
    if engine is None:
        return "Teacher escalation: not available in this build."
    s = engine.stats()
    armed = (
        f"yes — {s['teacher']}" if s["armed"]
        else "no (escalation.teacher_model unset)"
    )
    lines = [
        "Teacher escalation",
        f"Armed: {armed}",
        f"Fired: {s['fired']}   Skills written: {s['skills_written']}",
        f"Teacher failures: {s['teacher_failed']}   Budget refusals: {s['refused_budget']}",
        f"Budget: {s['max_per_session']} per session (in-memory, resets on restart)",
    ]
    sessions = s.get("sessions") or {}
    if sessions:
        lines.append(
            "Used: " + ", ".join(
                f"{k}={v}" for k, v in sorted(sessions.items()))
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GEPA skill evolution — /gepa
# ---------------------------------------------------------------------------


async def gepa_run_with_approval(
    send: Any,
    queue: Any,
    engine: Any,
    chat_id: Any = None,
) -> None:
    """Background flow: request approval, then run a GEPA cycle if granted.

    ``send`` is an async callable delivering one message to the invoking
    chat. ``chat_id`` is forwarded to the approval queue for prompt delivery
    (the queue's transport decides where the /approve prompt lands).
    """
    from prometheus.permissions.approval_queue import ApprovalResult
    try:
        result = await queue.request_approval(
            tool_name="gepa",
            description="Run GEPA skill evolution cycle now",
            chat_id=chat_id,
        )
    except Exception as exc:
        await send(f"GEPA: approval failed: {exc}")
        return

    if result == ApprovalResult.DENIED:
        await send("GEPA run denied.")
        return
    if result == ApprovalResult.TIMEOUT:
        await send("GEPA run approval timed out.")
        return

    try:
        report = await engine.run_now()
    except Exception as exc:
        await send(f"GEPA cycle failed: {exc}")
        return

    if report is None:
        await send("GEPA: a cycle is already running (or returned no report).")
        return
    await send("GEPA cycle complete:\n" + report.to_telegram_summary())


async def cmd_gepa(
    engine: Any,
    queue: Any,
    sub: str,
    *,
    chat_id: Any = None,
    send: Any = None,
    prefix: str = "/",
) -> str | None:
    """Shared /gepa {status|run|history} core.

    Returns the immediate reply text. The ``run`` path additionally spawns
    ``gepa_run_with_approval`` in the background (requires ``send``).
    """
    import asyncio as _aio

    sub = (sub or "status").strip().lower() or "status"

    if sub == "status":
        if engine is None:
            return "GEPA: engine not active (set learning.gepa_enabled in config)."
        report = engine.last_report
        if report is None:
            return (
                "GEPA: no cycle has run yet. "
                f"Use {prefix}gepa run to trigger one manually."
            )
        return report.to_telegram_summary()

    if sub == "run":
        if engine is None:
            return "GEPA: engine not active (set learning.gepa_enabled in config)."
        if queue is None:
            return "GEPA: approval queue not active — cannot run on demand."
        _aio.create_task(
            gepa_run_with_approval(send, queue, engine, chat_id),
            name="gepa_run_with_approval",
        )
        return f"GEPA run pending approval. Watch for the {prefix}approve prompt."

    if sub == "history":
        from prometheus.config.paths import get_config_dir
        archive_dir = get_config_dir() / "skills" / "auto" / "archive"
        if not archive_dir.exists():
            return "GEPA history: no promotions yet."
        archives = sorted(
            archive_dir.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not archives:
            return "GEPA history: archive directory is empty."
        lines = ["GEPA promotion history:"]
        for path in archives[:15]:
            ts = path.stat().st_mtime
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            lines.append(f"  • {path.stem} ({stamp})")
        return "\n".join(lines)

    return f"Usage: {prefix}gepa [status | run | history]"


# ---------------------------------------------------------------------------
# SYMBIOTE — /symbiote (GitHub research → graft pipeline)
# ---------------------------------------------------------------------------


async def _symbiote_scout(send: Any, coordinator: Any, problem: str, prefix: str) -> None:
    """Run the Scout phase and reply with a candidate summary."""
    await send(f"SYMBIOTE: scouting GitHub for {problem!r}...")
    try:
        session = await coordinator.start_scout(problem)
    except RuntimeError as exc:
        await send(f"SYMBIOTE: {exc}")
        return
    except Exception as exc:
        log.exception("SYMBIOTE scout failed")
        await send(f"SYMBIOTE scout failed: {exc}")
        return
    if session.error:
        await send(f"SYMBIOTE scout: {session.error}")
        return
    report = session.scout_report or {}
    candidates = report.get("candidates") or []
    if not candidates:
        note = report.get("notes") or "no viable candidates"
        await send(
            f"SYMBIOTE: no candidates ({note}). Session {session.session_id[:8]}."
        )
        return
    lines = [
        f"SYMBIOTE session {session.session_id[:8]} — "
        f"{len(candidates)} candidate(s):",
    ]
    for c in candidates[:5]:
        lic = (c.get("license_check") or {}).get("spdx_id") or "?"
        lines.append(
            f"  • {c['full_name']} ({lic}, ★{c.get('stars', 0)}) "
            f"[{c.get('recommendation', '?')}] "
            f"score={float(c.get('relevance_score', 0)):.2f}"
        )
    lines.append("")
    lines.append(f"To proceed: {prefix}symbiote approve <full_name>")
    await send("\n".join(lines))


async def _symbiote_approve(
    send: Any, coordinator: Any, rest: str, *,
    approval_queue: Any, morph_engine: Any, chat_id: Any, prefix: str,
) -> None:
    """Queue an approval; on approval run harvest."""
    import asyncio as _aio

    if not rest:
        await send(f"Usage: {prefix}symbiote approve <owner/repo>")
        return
    active = coordinator.get_status()
    if active is None:
        await send(
            f"SYMBIOTE: no active session. Run {prefix}symbiote <problem> first."
        )
        return
    candidate_full_name = rest.split()[0]
    if approval_queue is None:
        await send("SYMBIOTE: approval queue not active — cannot start harvest.")
        return
    _aio.create_task(
        symbiote_run_with_approval(
            send, approval_queue, coordinator,
            phase="harvest",
            session_id=active.session_id,
            candidate=candidate_full_name,
            morph_engine=morph_engine,
            chat_id=chat_id,
        ),
        name="symbiote_harvest_approval",
    )
    await send(
        f"SYMBIOTE: harvest of {candidate_full_name} pending approval. "
        f"Watch for the {prefix}approve prompt."
    )


async def _symbiote_graft(
    send: Any, coordinator: Any, *,
    approval_queue: Any, morph_engine: Any, chat_id: Any, prefix: str,
) -> None:
    """Queue an approval; on approval run graft."""
    import asyncio as _aio

    active = coordinator.get_status()
    if active is None:
        await send("SYMBIOTE: no active session.")
        return
    from prometheus.symbiote.coordinator import SymbiotePhase
    if active.phase != SymbiotePhase.AWAITING_HARVEST_APPROVAL:
        await send(f"SYMBIOTE: cannot graft from phase {active.phase.value}.")
        return
    if approval_queue is None:
        await send("SYMBIOTE: approval queue not active — cannot graft.")
        return
    _aio.create_task(
        symbiote_run_with_approval(
            send, approval_queue, coordinator,
            phase="graft",
            session_id=active.session_id,
            morph_engine=morph_engine,
            chat_id=chat_id,
        ),
        name="symbiote_graft_approval",
    )
    await send(
        f"SYMBIOTE: graft pending approval. Watch for the {prefix}approve prompt."
    )


async def _symbiote_status(send: Any, coordinator: Any, rest: str) -> None:
    session_id = rest.split()[0] if rest else None
    session = coordinator.get_status(session_id)
    if session is None:
        await send("SYMBIOTE: no active session.")
        return
    await send(session.to_telegram_summary())


async def _symbiote_history(send: Any, coordinator: Any, rest: str) -> None:
    try:
        limit = int(rest.split()[0]) if rest else 10
    except ValueError:
        limit = 10
    history = coordinator.get_history(limit)
    if not history:
        await send("SYMBIOTE: no past sessions.")
        return
    lines = [f"SYMBIOTE history ({len(history)} session(s)):"]
    for s in history:
        lines.append(
            f"  • {s.session_id[:8]} {s.phase.value} — "
            f"{(s.problem_statement or '')[:60]}"
        )
    await send("\n".join(lines))


async def _symbiote_abort(send: Any, coordinator: Any) -> None:
    active = coordinator.get_status()
    if active is None:
        await send("SYMBIOTE: no active session.")
        return
    try:
        session = await coordinator.abort(active.session_id)
    except Exception as exc:
        await send(f"SYMBIOTE abort failed: {exc}")
        return
    await send(f"SYMBIOTE: session {session.session_id[:8]} aborted.")


async def _symbiote_morph(
    send: Any, coordinator: Any, *, morph_engine: Any, prefix: str,
) -> None:
    """Stage a candidate via MorphEngine and produce a MorphReport."""
    if morph_engine is None:
        await send(
            "SYMBIOTE: MorphEngine not active. "
            "Set symbiote.morph.enabled in config."
        )
        return
    active = coordinator.get_status()
    if active is None:
        await send("SYMBIOTE: no active session.")
        return
    from prometheus.symbiote.coordinator import SymbiotePhase
    if active.phase != SymbiotePhase.AWAITING_GRAFT_APPROVAL:
        await send(
            f"SYMBIOTE: cannot morph from phase {active.phase.value}. "
            f"Run {prefix}symbiote graft first."
        )
        return
    await send(
        "SYMBIOTE: staging candidate (full test run)... this may take a minute."
    )
    try:
        session = await coordinator.start_morph(active.session_id, morph_engine)
    except Exception as exc:
        log.exception("SYMBIOTE morph failed")
        await send(f"SYMBIOTE morph failed: {exc}")
        return
    if session.morph_report:
        report = coordinator._rebuild_morph_report(session.morph_report)
        await send(report.to_telegram_summary())
    elif session.error:
        await send(f"SYMBIOTE morph: {session.error}")


async def _symbiote_swap(
    send: Any, coordinator: Any, *,
    approval_queue: Any, morph_engine: Any, chat_id: Any, prefix: str,
) -> None:
    """Request approval; on approval run MorphEngine.execute_swap()."""
    import asyncio as _aio

    if morph_engine is None:
        await send("SYMBIOTE: MorphEngine not active.")
        return
    active = coordinator.get_status()
    if active is None:
        await send("SYMBIOTE: no active session.")
        return
    from prometheus.symbiote.coordinator import SymbiotePhase
    if active.phase != SymbiotePhase.AWAITING_SWAP_APPROVAL:
        await send(
            f"SYMBIOTE: cannot swap from phase {active.phase.value}. "
            f"Run {prefix}symbiote morph first."
        )
        return
    if approval_queue is None:
        await send("SYMBIOTE: approval queue not active — cannot swap.")
        return
    backup_id = active.backup_id or "?"
    warning = (
        "⚠️ This will:\n"
        "  1. Stop the daemon (~2-5s downtime)\n"
        "  2. Replace live code with the candidate\n"
        "  3. Restart the daemon\n"
        "  4. Auto-rollback if health check fails within 60s\n\n"
        f"Backup {backup_id} retained for manual rollback.\n\n"
        f"Watch for {prefix}approve prompt."
    )
    await send(warning)
    _aio.create_task(
        symbiote_run_with_approval(
            send, approval_queue, coordinator,
            phase="swap",
            session_id=active.session_id,
            morph_engine=morph_engine,
            chat_id=chat_id,
        ),
        name="symbiote_swap_approval",
    )


async def _symbiote_manual_backup(
    send: Any, rest: str, *, backup_vault: Any, chat_id: Any,
) -> None:
    """Create a manual backup snapshot. Trust Level 2 (no approval)."""
    if backup_vault is None:
        await send(
            "SYMBIOTE: BackupVault not active. Set symbiote.backup.enabled."
        )
        return
    description = rest.strip().strip('"').strip("'") or "manual backup via /symbiote"
    await send(
        "SYMBIOTE: creating backup (running test suite for status capture)..."
    )
    try:
        snap = await backup_vault.create_snapshot(
            description=description,
            source="manual",
            metadata={"chat_id": chat_id},
            capture_test_status=True,
        )
    except Exception as exc:
        log.exception("SYMBIOTE backup failed")
        await send(f"SYMBIOTE backup failed: {exc}")
        return
    await send(
        f"📦 Backup created: {snap.backup_id}\n"
        f"  files: {snap.file_count}, size: {snap.size_bytes/1024:.1f}KB\n"
        f"  tests: {snap.test_status}\n"
        f"  description: {snap.description}"
    )


async def _symbiote_backups(
    send: Any, rest: str, *, backup_vault: Any, prefix: str,
) -> None:
    """List available backups."""
    if backup_vault is None:
        await send("SYMBIOTE: BackupVault not active.")
        return
    try:
        limit = int(rest.split()[0]) if rest else 15
    except ValueError:
        limit = 15
    snaps = backup_vault.list_snapshots(limit=limit)
    if not snaps:
        await send("SYMBIOTE: no backups yet.")
        return
    lines = [f"📦 Backup Vault ({len(snaps)} snapshot(s)):"]
    for s in snaps:
        lines.append(
            f"  {s.backup_id} — {s.source} — "
            f"{s.size_bytes/1024:.1f}KB — tests: {s.test_status}"
        )
    lines.append("")
    lines.append(f"Use: {prefix}symbiote restore <backup_id>")
    await send("\n".join(lines))


async def _symbiote_restore(
    send: Any, rest: str, *,
    backup_vault: Any, approval_queue: Any, chat_id: Any, prefix: str,
) -> None:
    """Restore from a backup. Trust Level 1 — requires approval."""
    import asyncio as _aio

    if backup_vault is None:
        await send("SYMBIOTE: BackupVault not active.")
        return
    args = rest.split()
    dry_run = False
    backup_id: str | None = None
    for arg in args:
        if arg.lower() == "dry":
            dry_run = True
        else:
            backup_id = arg
    if backup_id is None:
        latest = backup_vault.get_latest()
        if latest is None:
            await send("SYMBIOTE: no backups available to restore.")
            return
        backup_id = latest.backup_id

    snap = backup_vault.get_snapshot(backup_id)
    if snap is None:
        await send(f"SYMBIOTE: unknown backup_id {backup_id!r}.")
        return

    if dry_run:
        await send(f"SYMBIOTE: dry-run restore of {backup_id}...")
        try:
            result = await backup_vault.restore_snapshot(backup_id, dry_run=True)
        except Exception as exc:
            await send(f"SYMBIOTE: dry-run failed: {exc}")
            return
        lines = [
            f"📦 Dry-run restore of {backup_id}:",
            f"  added: {len(result.files_added)} file(s)",
            f"  changed: {len(result.files_changed)} file(s)",
        ]
        if result.files_changed[:5]:
            lines.append("  example changed paths:")
            for p in result.files_changed[:5]:
                lines.append(f"    {p}")
        await send("\n".join(lines))
        return

    # Real restore — Trust Level 1.
    if approval_queue is None:
        await send("SYMBIOTE: approval queue not active — cannot restore.")
        return
    await send(
        f"⚠️ {prefix}symbiote restore {backup_id} will replace live source files. "
        "A pre-restore backup will be created automatically. "
        f"Watch for {prefix}approve prompt."
    )
    _aio.create_task(
        symbiote_restore_with_approval(
            send, approval_queue, backup_vault, backup_id, chat_id=chat_id,
        ),
        name="symbiote_restore_approval",
    )


async def symbiote_restore_with_approval(
    send: Any,
    queue: Any,
    vault: Any,
    backup_id: str,
    *,
    chat_id: Any = None,
) -> None:
    """Background flow: request approval, then restore the snapshot."""
    from prometheus.permissions.approval_queue import ApprovalResult
    try:
        result = await queue.request_approval(
            tool_name="symbiote_restore",
            description=f"SYMBIOTE: restore from {backup_id}",
            chat_id=chat_id,
        )
    except Exception as exc:
        await send(f"SYMBIOTE: approval failed: {exc}")
        return
    if result == ApprovalResult.DENIED:
        await send("SYMBIOTE: restore denied.")
        return
    if result == ApprovalResult.TIMEOUT:
        await send("SYMBIOTE: restore approval timed out.")
        return
    try:
        restore = await vault.restore_snapshot(
            backup_id, capture_test_status=False,
        )
    except Exception as exc:
        await send(f"SYMBIOTE: restore failed: {exc}")
        return
    if restore.error:
        await send(f"SYMBIOTE: restore error: {restore.error}")
        return
    await send(
        f"📦 Restore complete: {backup_id}\n"
        f"  files restored: {restore.files_restored}\n"
        f"  pre-restore backup: {restore.pre_restore_backup_id}"
    )


async def symbiote_run_with_approval(
    send: Any,
    queue: Any,
    coordinator: Any,
    *,
    phase: str,
    session_id: str,
    candidate: str | None = None,
    morph_engine: Any = None,
    chat_id: Any = None,
) -> None:
    """Background flow: queue.request_approval → run phase on APPROVED."""
    from prometheus.permissions.approval_queue import ApprovalResult
    descriptions = {
        "harvest": (
            f"SYMBIOTE: clone and analyze {candidate!r} "
            f"(session {session_id[:8]})"
        ),
        "graft": (
            f"SYMBIOTE: write adapted files + run tests "
            f"(session {session_id[:8]})"
        ),
        "swap": (
            f"SYMBIOTE: HOT SWAP — replace live src/prometheus with "
            f"candidate (session {session_id[:8]})"
        ),
    }
    try:
        result = await queue.request_approval(
            tool_name=f"symbiote_{phase}",
            description=descriptions.get(phase, f"SYMBIOTE {phase}"),
            chat_id=chat_id,
        )
    except Exception as exc:
        await send(f"SYMBIOTE: approval failed: {exc}")
        return
    if result == ApprovalResult.DENIED:
        await send(f"SYMBIOTE {phase} denied.")
        return
    if result == ApprovalResult.TIMEOUT:
        await send(f"SYMBIOTE {phase} approval timed out.")
        return

    try:
        if phase == "harvest":
            session = await coordinator.approve_scout(session_id, candidate)
        elif phase == "graft":
            session = await coordinator.approve_harvest(session_id)
        elif phase == "swap":
            if morph_engine is None:
                await send("SYMBIOTE swap: MorphEngine not active.")
                return
            session = await coordinator.approve_swap(session_id, morph_engine)
        else:
            await send(f"Unknown phase: {phase}")
            return
    except Exception as exc:
        log.exception("SYMBIOTE %s execution failed", phase)
        await send(f"SYMBIOTE {phase} failed: {exc}")
        return

    if session.error:
        await send(f"SYMBIOTE {phase}: {session.error}")
        return

    await send(session.to_telegram_summary())


async def cmd_symbiote(
    send: Any,
    body: str,
    *,
    approval_queue: Any = None,
    morph_engine: Any = None,
    backup_vault: Any = None,
    chat_id: Any = None,
    prefix: str = "/",
) -> None:
    """Shared /symbiote dispatcher.

    Forms:
      /symbiote <problem statement>      — start a scout
      /symbiote approve <full_name>      — request harvest approval
      /symbiote graft                    — request graft approval
      /symbiote status [session_id]      — show session state
      /symbiote abort                    — abort active session
      /symbiote history [N]              — last N sessions
      /symbiote morph | swap | backup | backups | restore  (Session B)
    """
    from prometheus.symbiote import get_coordinator
    coordinator = get_coordinator()
    if coordinator is None:
        await send("SYMBIOTE is not active. Set symbiote.enabled in config.")
        return

    body = (body or "").strip()

    # Detect subcommand by first token; otherwise treat whole body as problem.
    first_token = body.split(maxsplit=1)[0] if body else ""
    first_lower = first_token.lower()
    rest = body[len(first_token):].strip()

    known_subcommands = {
        # Session A
        "approve", "graft", "status", "abort", "history",
        # Session B
        "morph", "swap", "backup", "backups", "restore",
    }

    if not body:
        await _symbiote_status(send, coordinator, "")
        return

    if first_lower not in known_subcommands:
        # Treat whole body as the problem statement → start scout.
        await _symbiote_scout(send, coordinator, body, prefix)
        return

    if first_lower == "status":
        await _symbiote_status(send, coordinator, rest)
        return
    if first_lower == "history":
        await _symbiote_history(send, coordinator, rest)
        return
    if first_lower == "abort":
        await _symbiote_abort(send, coordinator)
        return
    if first_lower == "approve":
        await _symbiote_approve(
            send, coordinator, rest,
            approval_queue=approval_queue, morph_engine=morph_engine,
            chat_id=chat_id, prefix=prefix,
        )
        return
    if first_lower == "graft":
        await _symbiote_graft(
            send, coordinator,
            approval_queue=approval_queue, morph_engine=morph_engine,
            chat_id=chat_id, prefix=prefix,
        )
        return
    # ---- Session B subcommands ------------------------------------
    if first_lower == "morph":
        await _symbiote_morph(
            send, coordinator, morph_engine=morph_engine, prefix=prefix,
        )
        return
    if first_lower == "swap":
        await _symbiote_swap(
            send, coordinator,
            approval_queue=approval_queue, morph_engine=morph_engine,
            chat_id=chat_id, prefix=prefix,
        )
        return
    if first_lower == "backup":
        await _symbiote_manual_backup(
            send, rest, backup_vault=backup_vault, chat_id=chat_id,
        )
        return
    if first_lower == "backups":
        await _symbiote_backups(
            send, rest, backup_vault=backup_vault, prefix=prefix,
        )
        return
    if first_lower == "restore":
        await _symbiote_restore(
            send, rest,
            backup_vault=backup_vault, approval_queue=approval_queue,
            chat_id=chat_id, prefix=prefix,
        )
        return


# ---------------------------------------------------------------------------
# Web capability audit — /audit
# ---------------------------------------------------------------------------


async def _audit_show_last(send: Any, prefix: str) -> None:
    """Show summary of the most recent audit JSON, if any."""
    from prometheus.config.paths import get_config_dir

    audits_dir = get_config_dir() / "audits"
    if not audits_dir.is_dir():
        await send(f"No audits yet. Run `{prefix}audit run` to start one.")
        return
    json_files = sorted(audits_dir.glob("web_audit_*.json"))
    if not json_files:
        await send(f"No audits yet. Run `{prefix}audit run` to start one.")
        return
    latest = json_files[-1]
    try:
        import json as _json
        payload = _json.loads(latest.read_text())
    except Exception as exc:
        await send(f"Could not read latest audit: {exc}")
        return

    lines = [
        "🔬 Last Web Capability Audit",
        f"Date: {payload.get('timestamp', '?')}",
        f"Model: {payload.get('model', '?')}",
        (
            f"Result: {payload.get('passed', 0)}/"
            f"{payload.get('total_tests', 0)} passed "
            f"({payload.get('pass_rate', 0) * 100:.0f}%)"
        ),
        f"Duration: {payload.get('duration_seconds', 0):.0f}s",
        "",
        "By category:",
    ]
    for cat, stats in sorted((payload.get("categories") or {}).items()):
        n = stats["passed"] + stats["failed"]
        lines.append(f"  {cat}: {stats['passed']}/{n}")
    fb = payload.get("failure_breakdown") or {}
    if fb:
        lines.append("")
        lines.append("Failure breakdown:")
        for fc, n in sorted(fb.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {fc}: {n}")
    lines.append("")
    lines.append(f"Full report: {latest}")
    await send("\n".join(lines))


async def _audit_kick_off(send: Any, category: str | None) -> None:
    """Spawn the audit as a background subprocess and notify on completion."""
    import asyncio as _aio
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "web_capability_audit.py"
    if not script.is_file():
        await send(f"Audit script not found at {script}")
        return

    cmd: list[str] = ["python3", str(script)]
    if category:
        cmd += ["--category", category]
    label = f"category={category}" if category else "full audit"

    await send(
        f"🔬 Audit starting ({label}). "
        "I'll send a summary when it completes."
    )

    async def _runner() -> None:
        try:
            proc = await _aio.create_subprocess_exec(
                *cmd,
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.STDOUT,
                cwd=str(repo_root),
            )
            stdout, _ = await proc.communicate()
            rc = proc.returncode
            # Read the latest report (the script writes one even on partial runs)
            from prometheus.config.paths import get_config_dir
            audits_dir = get_config_dir() / "audits"
            latest_md = sorted(audits_dir.glob("web_audit_*.md"))[-1] \
                if audits_dir.is_dir() and any(
                    audits_dir.glob("web_audit_*.md")
                ) else None
            if latest_md is not None:
                msg = (
                    f"🔬 Audit complete (exit {rc}). "
                    f"Report: {latest_md}"
                )
            else:
                tail = (stdout or b"").decode("utf-8", errors="replace")[-1500:]
                msg = f"🔬 Audit finished (exit {rc}). Output tail:\n{tail}"
            await send(msg)
        except Exception as exc:
            await send(f"Audit failed to launch: {exc}")

    _aio.create_task(_runner())


async def cmd_audit(send: Any, body: str, *, prefix: str = "/") -> None:
    """Shared /audit dispatcher.

    Forms:
      /audit                  — show summary of the most recent audit
      /audit run              — start a full web capability audit
      /audit <category>       — start an audit for a single category
                                (search|fetch|youtube|download|research|graceful|railway)
    """
    body = (body or "").strip().lower()

    if not body:
        await _audit_show_last(send, prefix)
        return

    valid_categories = {
        "search", "fetch", "youtube", "download",
        "research", "graceful", "railway",
    }

    if body == "run":
        await _audit_kick_off(send, category=None)
        return

    if body in valid_categories:
        await _audit_kick_off(send, category=body)
        return

    await send(
        "Usage:\n"
        f"  {prefix}audit                — show last audit summary\n"
        f"  {prefix}audit run            — full audit (~30–60 min)\n"
        f"  {prefix}audit <category>     — single category\n"
        "Categories: search, fetch, youtube, download, research, graceful, railway"
    )


# ---------------------------------------------------------------------------
# Printing Press — /press (CLI library discovery + install)
# ---------------------------------------------------------------------------


async def _press_usage(send: Any, prefix: str) -> None:
    await send(
        "Printing Press — local CLI library\n\n"
        f"  {prefix}press list [category]  — list available CLIs\n"
        f"  {prefix}press search <query>   — fuzzy search\n"
        f"  {prefix}press install <name>   — install (queues approval)\n"
        f"  {prefix}press installed        — show what's installed\n"
        f"  {prefix}press update           — git pull the library clone\n"
    )


async def _press_list(
    send: Any, press: Any, *, category: str | None, prefix: str,
) -> None:
    records = press.list_available()
    if category:
        cat = category.lower()
        records = [r for r in records if cat in r.category.lower()]
    if not records:
        await send(
            f"No CLIs in the library{' for category ' + category if category else ''}."
        )
        return
    # Group by category for readability
    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r.category or "uncategorized"].append(r)
    lines = [f"Printing Press CLIs ({len(records)} available):"]
    for cat in sorted(by_cat):
        lines.append(f"\n[{cat}]")
        for r in sorted(by_cat[cat], key=lambda x: x.name):
            marker = "✓" if r.installed else "·"
            lines.append(f"  {marker} {r.name} — {r.description[:80]}")
    # Telegram caps messages around 4096 chars; trim if needed
    msg = "\n".join(lines)
    if len(msg) > 3800:
        msg = msg[:3800] + f"\n... (truncated; use {prefix}press search to narrow)"
    await send(msg)


async def _press_search(send: Any, press: Any, *, query: str, prefix: str) -> None:
    if not query:
        await send(f"Usage: {prefix}press search <query>")
        return
    records = press.search(query, limit=10)
    if not records:
        await send(f"No CLI matched '{query}'.")
        return
    lines = [f"Matches for '{query}':"]
    for r in records:
        marker = "✓" if r.installed else "·"
        lines.append(
            f"{marker} {r.name} ({r.category}) — {r.description[:140]}"
        )
    await send("\n".join(lines))


async def _press_installed(send: Any, press: Any, *, prefix: str) -> None:
    records = [r for r in press.list_available() if r.installed]
    if not records:
        await send(
            f"No Printing Press CLIs installed yet. Try {prefix}press list."
        )
        return
    lines = [f"Installed Printing Press CLIs ({len(records)}):"]
    for r in sorted(records, key=lambda x: x.name):
        lines.append(f"  ✓ {r.name} ({r.bin_name})")
    await send("\n".join(lines))


async def _press_update(send: Any, press: Any) -> None:
    ok, msg = await press.update_library()
    prefix_mark = "✓" if ok else "✗"
    await send(f"{prefix_mark} library update: {msg}")


async def _press_install(
    send: Any, press: Any, *, cli_name: str,
    approval_queue: Any, chat_id: Any, prefix: str,
) -> None:
    """Approval-queue install. Mirrors the /gepa run / /symbiote pattern."""
    import asyncio as _aio

    if approval_queue is None:
        await send("Approval queue is not active — cannot install.")
        return

    # Resolve up-front so we can show the user what they're approving
    records = press.search(cli_name, limit=1)
    if not records:
        await send(
            f"No CLI matching '{cli_name}' in the library. "
            f"Try {prefix}press search first."
        )
        return
    rec = records[0]
    if rec.installed:
        await send(f"{rec.name} ({rec.bin_name}) is already installed.")
        return

    async def _runner() -> None:
        from prometheus.permissions.approval_queue import ApprovalResult

        description = (
            f"go install {rec.install_module}@latest  "
            f"+ copy SKILL.md → ~/.prometheus/skills/{rec.skill_name}.md"
        )
        try:
            result = await approval_queue.request_approval(
                "printing_press_install", description, chat_id=chat_id,
            )
        except Exception as exc:
            await send(f"Approval queue error: {exc}")
            return
        if result == ApprovalResult.DENIED:
            await send(f"Install denied: {rec.name}.")
            return
        if result == ApprovalResult.TIMEOUT:
            await send(f"Install request timed out: {rec.name}.")
            return
        await send(f"Installing {rec.name}…")
        try:
            outcome = await press.install(rec.name)
        except Exception as exc:
            await send(f"Install failed: {exc}")
            return
        if outcome.success:
            lines = [
                f"✓ Installed {outcome.cli_name} ({outcome.bin_name})",
                f"  Skill: {'copied' if outcome.skill_installed else 'NOT copied'}",
            ]
            if not outcome.on_path:
                lines.append(
                    "  ⚠ Binary in ~/go/bin but not on $PATH — "
                    "add `export PATH=$HOME/go/bin:$PATH` to your shell."
                )
            await send("\n".join(lines))
        else:
            await send(f"✗ Install failed: {outcome.error}")

    _aio.create_task(_runner(), name=f"press_install_{rec.name}")
    await send(
        f"Install request queued for {rec.name}. Watch for {prefix}approve."
    )


async def cmd_press(
    send: Any,
    press: Any,
    body: str,
    *,
    approval_queue: Any = None,
    chat_id: Any = None,
    prefix: str = "/",
) -> None:
    """Shared /press dispatcher.

    Forms:
      /press                   — show usage
      /press list [category]   — list available CLIs (optionally filtered)
      /press search <query>    — fuzzy search by name / description
      /press install <name>    — request approval, then go install + skill copy
      /press installed         — list CLIs whose binary is on PATH or in ~/go/bin
      /press update            — git pull the library clone
    """
    if press is None or not press.is_available():
        await send(
            "Printing Press is not active. The library clone is missing "
            "(searched ~/printing-press-library/ and /tmp/printing-press-library/) "
            "or the feature is disabled in config."
        )
        return

    body = (body or "").strip()
    first_token = body.split(maxsplit=1)[0].lower() if body else ""
    rest = body[len(first_token):].strip() if first_token else ""

    if first_token == "" or first_token == "help":
        await _press_usage(send, prefix)
        return
    if first_token == "list":
        await _press_list(send, press, category=rest or None, prefix=prefix)
        return
    if first_token == "search":
        await _press_search(send, press, query=rest, prefix=prefix)
        return
    if first_token == "installed":
        await _press_installed(send, press, prefix=prefix)
        return
    if first_token == "update":
        await _press_update(send, press)
        return
    if first_token == "install":
        if not rest:
            await send(f"Usage: {prefix}press install <cli-name>")
            return
        await _press_install(
            send, press, cli_name=rest,
            approval_queue=approval_queue, chat_id=chat_id, prefix=prefix,
        )
        return
    await _press_usage(send, prefix)


# ---------------------------------------------------------------------------
# Voice reply mode — /voice (persistence is platform-independent; the TTS
# reply pipeline itself currently lives in the Telegram adapter)
# ---------------------------------------------------------------------------

_VOICE_MODES_FILE = "voice_modes.json"
_VALID_VOICE_MODES = ("auto", "on", "off")


def _voice_modes_path() -> str:
    """Path to the per-chat voice mode override JSON file."""
    from prometheus.config.paths import get_config_dir
    return str(get_config_dir() / _VOICE_MODES_FILE)


def load_voice_modes() -> dict[str, str]:
    """Read per-chat voice mode overrides. Returns {chat_key: mode}."""
    import json
    try:
        with open(_voice_modes_path()) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (FileNotFoundError, ValueError, OSError):
        pass
    return {}


def save_voice_modes(modes: dict[str, str]) -> None:
    """Persist per-chat voice mode overrides."""
    import json
    try:
        with open(_voice_modes_path(), "w") as f:
            json.dump(modes, f, indent=2)
    except OSError as exc:
        log.warning("Failed to persist voice modes: %s", exc)


def get_voice_mode(chat_key: str, default_mode: str = "auto") -> str:
    """Return the effective voice mode for a chat ('auto' | 'on' | 'off')."""
    overrides = load_voice_modes()
    mode = overrides.get(str(chat_key))
    if mode in _VALID_VOICE_MODES:
        return mode
    return default_mode if default_mode in _VALID_VOICE_MODES else "auto"


def set_voice_mode(chat_key: str, mode: str) -> None:
    """Persist a per-chat voice mode override."""
    if mode not in _VALID_VOICE_MODES:
        return
    modes = load_voice_modes()
    modes[str(chat_key)] = mode
    save_voice_modes(modes)


def cmd_voice(
    chat_key: str,
    arg: str,
    voice_config: dict | None = None,
    *,
    prefix: str = "/",
) -> str:
    """Shared /voice core — show or set the per-chat voice reply mode.

    Modes:
      auto  (default) — mirror input modality (voice in → voice out)
      on              — always reply with voice (TTS for every response)
      off             — always reply with text (disable voice replies)

    Persists across daemon restarts in ~/.prometheus/voice_modes.json.
    """
    cfg = voice_config or {}
    arg = (arg or "").strip().lower()

    if not arg:
        default = str(cfg.get("default_mode", "auto"))
        current = get_voice_mode(chat_key, default)
        engine = cfg.get("engine", "piper")
        model = cfg.get("model_path", "(unset)")
        return (
            f"Voice mode: {current}\n"
            f"  auto — mirror input modality (default)\n"
            f"  on   — always voice reply\n"
            f"  off  — always text reply\n"
            f"\nEngine: {engine}\nModel:  {model}\n"
            f"Set with: {prefix}voice [auto|on|off]"
        )

    if arg not in _VALID_VOICE_MODES:
        return f"Unknown voice mode: {arg}\nUse: {prefix}voice [auto|on|off]"

    set_voice_mode(chat_key, arg)
    descriptions = {
        "auto": "mirror input modality (voice in → voice out)",
        "on": "always reply with voice",
        "off": "always reply with text",
    }
    return f"Voice mode set to: {arg}\n  ({descriptions[arg]})"


# ---------------------------------------------------------------------------
# Telemetry formatters — /tools /pairs
# ---------------------------------------------------------------------------


def cmd_tools() -> str:
    """Tool-call telemetry dashboard, last 24h (shared /tools core)."""
    try:
        from prometheus.telemetry.dashboard import ToolDashboard
        dashboard = ToolDashboard()
        stats = dashboard.get_stats(hours=24)

        lines = ["Tool Call Stats (24h)\n"]
        lines.append(f"Total calls: {stats['total_calls']}")
        lines.append(f"Success rate: {stats['overall_success_rate']:.0%}")
        if stats.get("total_denials"):
            lines.append(f"Denied by policy: {stats['total_denials']}")

        if stats["most_called"]:
            lines.append("\nMost called:")
            for row in stats["most_called"][:5]:
                name, count = row["tool_name"], row["calls"]
                rate = stats["success_rate_by_tool"].get(name, 0)
                lines.append(f"  {name}: {count} calls ({rate:.0%} ok)")

        if stats["circuit_breaker_trips"]:
            lines.append(f"\nCircuit breaker trips: {stats['circuit_breaker_trips']}")
        if stats["adapter_repairs"]:
            lines.append(f"Adapter repairs: {stats['adapter_repairs']}")
        if stats["lucky_guesses"]:
            lines.append(f"Lucky guesses (deferred): {stats['lucky_guesses']}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Tool stats unavailable: {exc}"


def cmd_pairs() -> str:
    """Repair-pair flywheel stats (shared /pairs core)."""
    try:
        from prometheus.learning.pair_capture import PairStore, get_store
        store = get_store() or PairStore()
        stats = store.stats()

        lines = ["Training Pairs\n"]
        lines.append(f"Total: {stats['total']}")
        lines.append(
            f"Last 7d: {stats['last_7d']} (~{stats['per_day_7d']}/day)"
        )
        if stats["by_source"]:
            lines.append("\nBy source:")
            for src, n in sorted(
                stats["by_source"].items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {src}: {n}")
        if stats["by_tool"]:
            lines.append("\nBy tool:")
            for tool, n in list(stats["by_tool"].items())[:8]:
                lines.append(f"  {tool}: {n}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Pair stats unavailable: {exc}"
