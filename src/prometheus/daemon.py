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

from prometheus.config.paths import (
    get_config_dir,
    get_logs_dir,
    resolve_vault_root,
    resolve_wiki_root,
    set_vault_root,
    set_wiki_root,
)
from prometheus.context.environment import git_head_sha
from prometheus.engine.agent_loop import AgentLoop
from prometheus.config.shipped_defaults import resolve_max_tool_iterations, resolve_max_tool_iterations_cloud
from prometheus.gateway.archive_writer import ArchiveWriter
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.cron_scheduler import run_scheduler_loop
from prometheus.gateway.heartbeat import Heartbeat
from prometheus.gateway.platform_base import GatewaySubsystemRegistry
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
    always have the same tool set. WikiCompileTool and WikiQueryTool
    are unconditionally registered by ``create_tool_registry`` (see
    ``__main__.py:188-189``) — no daemon-specific addition needed.
    """
    if security_cfg is None:
        security_cfg = {}
    return create_tool_registry(security_cfg)


async def _detect_loaded_model_with_retry(
    provider: Any,
    *,
    attempts: int = 5,
    sleep=asyncio.sleep,
) -> str | None:
    """Probe ``detect_loaded_model`` with linear backoff before giving up.

    The model server frequently co-boots with the daemon, so a single probe can
    race it. Audit #7: a boot-time miss previously stranded a stale model name
    until the next restart. Retries ``attempts`` times (1s, 2s, … between tries)
    and returns the detected id, or None if every attempt fails. ``sleep`` is
    injectable so tests don't actually wait.
    """
    detected: str | None = None
    for attempt in range(1, attempts + 1):
        detected = await provider.detect_loaded_model()
        if detected:
            return detected
        if attempt < attempts:
            logger.info(
                "Model detection attempt %d/%d failed (server unreachable); "
                "retrying in %ds",
                attempt, attempts, attempt,
            )
            await sleep(attempt)
    return detected


def _wire_skill_creator(
    agent_loop: AgentLoop,
    provider: Any,
    *,
    model_name: str,
    learning_config: dict[str, Any],
    telemetry: Any = None,
) -> Any:
    """Build the daemon's SkillCreator and gate its post-task hook on config.

    Until 2026-08-03 the daemon constructed SkillCreator directly, so
    ``learning.auto_skill_creation`` had no reader anywhere and
    ``learning.skill_min_tool_calls`` never left prometheus.yaml — the
    template promised knobs the live path ignored (survey
    audits/20260803T215431Z-skillcreator-quality-gate-survey.md §1).

    ``auto_skill_creation: false`` skips ONLY the hook registration. The
    instance is still built and returned: teacher escalation drafts and
    record-a-skill trace uploads share it, and neither is "auto" creation.

    Returns None only when construction fails; every caller already
    tolerates a missing SkillCreator from the import-failure days.
    """
    try:
        from prometheus.learning.skill_creator import SkillCreator
        skill_creator = SkillCreator(
            provider,
            model=model_name,
            min_tool_calls=int(learning_config.get("skill_min_tool_calls", 3)),
            telemetry=telemetry,
        )
    except Exception as exc:
        logger.warning("SkillCreator not available: %s", exc)
        return None

    if not learning_config.get("auto_skill_creation", True):
        logger.info(
            "SkillCreator post-task hook disabled (learning.auto_skill_creation: "
            "false); instance kept for teacher escalation and record-a-skill"
        )
        return skill_creator

    agent_loop.add_post_task_hook(skill_creator.maybe_create)
    logger.info("SkillCreator wired to agent loop post-task hook")
    return skill_creator


def telegram_gateway_decision(
    gateway_config: dict[str, Any] | None,
    token: str,
) -> tuple[bool, str | None]:
    """Whether to start the Telegram gateway, and why not when the answer is no.

    EXTRACTED SO IT CAN BE TESTED AT ALL. It was an inline boolean inside
    ``run_daemon``, and ``run_daemon`` cannot be driven from a test — it builds
    the entire subsystem graph. A mutation that neutered the refusal branch
    (``if False and ...``) therefore survived the whole suite: the second layer
    (``PlatformConfig.chat_allowed``) still denied every message, so nothing
    went red while the daemon happily started an unrestricted gateway that
    silently ignored everyone. That is §3b's defence-in-depth variant — the
    better the layering, the more reliably a disabled control is masked by its
    neighbour.

    Returning the REASON rather than a bare bool is what makes the refusal
    observable. A silently dead gateway is indistinguishable from a working
    control (§2c).
    """
    from prometheus.config.shipped_defaults import (
        resolve_allowed_chat_ids,
        resolve_telegram_enabled,
    )

    if not token:
        return False, None                      # nothing configured; not a refusal
    if not resolve_telegram_enabled(gateway_config):
        return False, None                      # explicitly (or by default) off
    if not resolve_allowed_chat_ids(gateway_config):
        return False, (
            "Telegram gateway NOT started: gateway.telegram_enabled is on and a "
            "token is present, but gateway.allowed_chat_ids is empty or absent. "
            "An empty allowlist used to mean 'allow every chat', which exposes "
            "an agent with shell access to anyone who finds the bot. Add your "
            "chat id to gateway.allowed_chat_ids (find it with @userinfobot), "
            "or set gateway.telegram_enabled: false to silence this."
        )
    return True, None


#: The ONE sentence describing what a config_pin does. The boot WARNING, the
#: doctor line and the /api/status payload all render this exact string. An
#: operator who greps the log and an operator who reads a surface must not have
#: to reconcile two descriptions of one mechanism.
CONFIG_PIN_EFFECT = "Corrected in memory; the config file is left as written."

#: Deliberately `config_pins`, never bare "pins". Beacon already uses "pin" for
#: SKILL pins in the same UI, and a second unrelated meaning of the word on an
#: adjacent screen is a name collision waiting to be misread.
CONFIG_PINS_FILENAME = "config_pins.yaml"


def read_config_pins(pins_path: Path) -> dict:
    """The pinned values, or {} when no pin file exists. Read-only, no logging.

    Exists so doctor and the REST surfaces can answer "is anything pinned?"
    without importing the correction path or re-implementing the parse.
    """
    if not pins_path.is_file():
        return {}
    try:
        with pins_path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


#: Keys corrected at the most recent boot, and the pins in force. Populated by
#: apply_config_pins so a surface can report drift WITHOUT re-reading the file
#: and WITHOUT re-running the comparison — the surface must describe what
#: actually happened at boot, not recompute a fresh opinion (CROSS-CUTTING §12:
#: the runtime auto-detects, the record must follow).
CONFIG_PINS_STATE: dict = {"path": None, "pins": {}, "drifted": []}


def apply_config_pins(config: dict, pins_path: Path) -> list[str]:
    """Apply pinned config values IN MEMORY. Returns the dotted keys that drifted.

    ⚠ THIS DELIBERATELY DOES NOT WRITE THE CONFIG FILE.

    It used to. Correcting one pinned key rewrote the WHOLE file via
    ``yaml.dump``, which drops every comment and reformats everything else.
    That already happened on the live box: the shipped template carries 430
    comment lines and the deployed config carried 0 — including the blocks
    explaining that ``denied_paths`` does not cover bash, and that
    ``bash_confinement`` defaults to ``off`` so the floor is not in force
    until switched on. The file an operator opens to learn what a key means
    had been silently emptied of meaning by a routine correction.

    The write was never load-bearing. The daemon runs on ``config``, the dict
    mutated below; the disk write only made the file agree afterwards.

    The consequence, stated rather than hidden: the file and the running
    config now DISAGREE persistently whenever a pin is in force. That is more
    honest than before. The file shows what the operator wrote; the pin
    surfaces show what is enforced. Previously the file silently agreed, so
    the disagreement was invisible — which is how a pin restored
    ``model.model: gemma4-26b`` on every boot for six weeks while the rig
    served Qwen, and nobody could see it in the file.

    Removing this write also removes the SECOND writer of prometheus.yaml.
    ``SecurityGate._rewrite_config_grants`` documents its atomicity argument
    on the premise that "there is one writer process" — a premise this
    function was quietly falsifying.
    """
    CONFIG_PINS_STATE.update({"path": str(pins_path), "pins": {}, "drifted": []})
    if not pins_path.is_file():
        logger.info("config_pins: none active (no %s)", pins_path)
        return []
    pins = read_config_pins(pins_path)
    CONFIG_PINS_STATE["pins"] = dict(pins)

    drifted: list[str] = []
    for dotpath, expected in pins.items():
        if expected is None:
            continue
        parts = str(dotpath).split(".")
        val = config
        for p in parts:
            val = val.get(p, {}) if isinstance(val, dict) else None
        if val and str(val) != str(expected):
            logger.warning(
                "CONFIG DRIFT DETECTED: %s = %r (pinned: %r). %s",
                dotpath, val, expected, CONFIG_PIN_EFFECT,
            )
            obj = config
            for p in parts[:-1]:
                obj = obj.setdefault(p, {})
            obj[parts[-1]] = expected
            drifted.append(str(dotpath))

    # ALWAYS log, drift or not. Previously the only evidence a pin existed was
    # the WARNING above, which fires only WHEN it acts — so absence of a
    # warning was indistinguishable from absence of the mechanism, and a pin
    # restored the wrong model on every boot for six weeks unnoticed.
    logger.info(
        "config_pins: %d active from %s (%s); drift corrected this boot: %s",
        len(pins), pins_path, ", ".join(sorted(pins)) or "none",
        ", ".join(drifted) if drifted else "none",
    )
    CONFIG_PINS_STATE["drifted"] = list(drifted)
    return drifted


async def _model_identity_loop(
    provider: Any, config_model: str, interval: float
) -> None:
    """Keep asking the backend which model it is serving, and say so when it
    disagrees with the configured name.

    THE ASYMMETRY THIS CLOSES. ``budget.py`` already treats the backend as the
    source of truth for one property: the server-reported ``n_ctx`` overrides
    the configured ``effective_limit`` for the local model. The mechanism was
    written, agreed with, and applied — to the context WINDOW. It was never
    applied to the model's IDENTITY, which stayed a config string detected
    once at boot. So the same daemon believes the backend about how big the
    window is and believes the config about what is answering.

    ⚠ THE CONFIG VALUE IS A HINT. It is used before the first successful
    detection and whenever the backend is unreachable — never as an assertion
    that outranks the server. A local rig is meant to be swapped freely; a
    label that can only be corrected by a restart is model-ASSUMED, not
    model-agnostic.

    ⚠ EVERY OBSERVED DISAGREEMENT WARNS, not just the first. That is
    deliberate and it is the whole point: a config pin held
    ``model.model: gemma4-26b`` while the rig served Qwen for SIX WEEKS, and
    the reason nobody noticed is that the disagreement was silent. A warning
    that fires once at boot is a warning you scroll past; one that keeps
    firing is one you fix. The cost is a log line per interval while a
    disagreement persists, which is the correct price for the thing that cost
    six weeks.

    This OBSERVES and REPORTS. It does not rewrite the running label — see
    ``run_daemon``'s note on why that is deliberately a follow-on.
    """
    while True:
        try:
            await asyncio.sleep(interval)
            detected = await provider.detect_loaded_model()
            if detected is None:
                # Unreachable. The last known name stands — detect_loaded_model
                # already logged the failure, and inventing a fallback here
                # would be the config silently winning again.
                continue
            if config_model and detected != config_model:
                logger.warning(
                    "MODEL LABEL DISAGREES WITH THE BACKEND: config says %r, "
                    "the server is serving %r. The backend is authoritative; "
                    "the config value is a hint. Anything keyed on the "
                    "configured name (context model_overrides, the telemetry "
                    "`model` column, /api/status) is reporting the hint, not "
                    "the truth.", config_model, detected,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("model identity probe failed", exc_info=True)


def loop_ceiling_divergence(agent_loop: Any, loop_context: Any) -> list[tuple]:
    """Fields where the two loop constructions carry different ceilings.

    Returns ``[(field_name, agent_loop_value, loop_context_value), ...]`` --
    empty when they agree. Pure and module-level so the invariant can be TESTED
    rather than only executed at boot inside a 1600-line startup function.

    Compares the AgentLoop's stored values (which it copies into the LoopContext
    it builds per run, agent_loop.py:3944-3945) against the separately-built
    web/Beacon LoopContext. ``getattr`` with a sentinel rather than direct
    attribute access: a RENAMED private attribute must surface as a divergence,
    not as an AttributeError that takes the daemon down at boot.
    """
    _MISSING = object()
    pairs = (
        ("max_tool_iterations", "_max_tool_iterations"),
        ("max_tool_iterations_cloud", "_max_tool_iterations_cloud"),
    )
    out = []
    for ctx_field, loop_attr in pairs:
        a = getattr(agent_loop, loop_attr, _MISSING)
        b = getattr(loop_context, ctx_field, _MISSING)
        if a is _MISSING or b is _MISSING or a != b:
            out.append((
                ctx_field,
                None if a is _MISSING else a,
                None if b is _MISSING else b,
            ))
    return out


async def run_daemon(args: argparse.Namespace) -> None:
    """Main async entry point — start all subsystems."""
    config = load_config(args.config)
    model_config = config.get("model", {})
    gateway_config = config.get("gateway", {})
    security_config = config.get("security", {})

    # ── Wiki root ───────────────────────────────────────────────────────
    # Resolved ONCE here and pinned process-wide; every consumer reads it back
    # through get_wiki_root() and none keeps a fallback. Before this, nine
    # sites derived the path independently and three ignored
    # PROMETHEUS_CONFIG_DIR entirely (the web bridge and both Telegram
    # surfaces), which split the wiki in two under a non-default config dir.
    wiki_root = resolve_wiki_root(config)
    set_wiki_root(wiki_root)
    logger.info("Wiki root resolved to %s", wiki_root)

    # ── Brain vault root ────────────────────────────────────────────────
    # A SECOND, separate root: the brain-vault repo, read-only. Pinned the same
    # way and for the same reason as the wiki root above — the vault tools are
    # registered with no arguments and resolve it inside execute().
    # Deliberately NOT the Prometheus wiki: log both so a misconfiguration
    # that points one at the other is visible in the first ten lines of a boot
    # rather than inferred later from odd search results.
    vault_root = resolve_vault_root(config)
    set_vault_root(vault_root)
    logger.info("Brain vault root resolved to %s (read-only)", vault_root)
    if vault_root == wiki_root:
        logger.warning(
            "Brain vault root and Prometheus wiki root are THE SAME PATH (%s) "
            "— they are different corpora with different owners; check "
            "vault.root / wiki.root", vault_root,
        )

    # ── Env file (Onboarding Phase 0) ───────────────────────────────────
    # Under systemd the unit's EnvironmentFile= already populated these;
    # run bare, `prometheus daemon` loads the same file so tokens/keys
    # behave identically either way (setdefault — real env always wins).
    from prometheus.config.env_file import get_env_file_path, load_env_file
    _env_loaded = load_env_file()
    if _env_loaded:
        logger.info(
            "Loaded %d variable(s) from %s", _env_loaded, get_env_file_path()
        )

    # ── Boot-SHA staleness signal ───────────────────────────────────────
    # The repo HEAD at process start is the identity of the code THIS process
    # loaded. /api/status, /health, and the heartbeat compare it against the
    # live tree HEAD to surface "merged-but-dark" — new code on disk the
    # running process isn't executing. "unknown" off a git checkout.
    boot_sha = git_head_sha()
    logger.info("Boot SHA: %s", boot_sha)

    # ── Config drift guard (opt-in) ─────────────────────────────────────
    # Users can create ~/.prometheus/config_pins.yaml to pin critical
    # config values. If a pinned value drifts, the daemon corrects it
    # IN MEMORY. It does NOT rewrite the config file — see
    # apply_config_pins for why.
    #
    # Example ~/.prometheus/config_pins.yaml:
    #   model.provider: llama_cpp
    #   model.base_url: http://<gpu-host>:8080
    #
    # If the file doesn't exist, no checking happens — users who run
    # Ollama, Anthropic, etc. are unaffected.
    apply_config_pins(config, get_config_dir() / "config_pins.yaml")

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

    # Detect actual loaded model from the server (local providers only).
    # Retries a few times so a model server that co-boots with the daemon
    # isn't missed and stranded on a stale name (audit #7).
    config_model = model_config.get("model", "qwen3.5-32b")
    if hasattr(provider, "detect_loaded_model"):
        detected = await _detect_loaded_model_with_retry(provider)
        model_name = detected or config_model
        if detected:
            model_config["model"] = detected
            if config_model and detected != config_model:
                # Loud from the first observation, not merely recorded.
                logger.warning(
                    "MODEL LABEL DISAGREES WITH THE BACKEND at boot: config "
                    "says %r, the server is serving %r. Taking the server's "
                    "answer — the config value is a HINT, used before "
                    "detection and when the backend is unreachable.",
                    config_model, detected,
                )
        else:
            logger.warning(
                "Model detection failed after retries; falling back to the "
                "config HINT %s. This is the one case where the config wins, "
                "and only because there is nothing to ask.", config_model,
            )
        # Keep asking. Detection used to run exactly once, so swapping the
        # model on the backend left the label wrong indefinitely — a restart
        # was the only way to correct it, which is model-ASSUMED rather than
        # model-agnostic. 0 disables.
        identity_interval = float(
            model_config.get("identity_probe_interval_seconds", 300)
        )
        if identity_interval > 0:
            asyncio.create_task(
                _model_identity_loop(provider, config_model, identity_interval)
            )
    else:
        model_name = config_model
        logger.info("Cloud provider: %s, model: %s", model_config.get("provider"), model_name)

    # Thinking-suppression capability probe. MEASURED, never assumed.
    #
    # ``model.suppress_thinking`` is a chat-template kwarg, and a template
    # that does not know the key ignores it silently — 200 OK, normal-looking
    # completion, reasoning channel still running and still eating the output
    # budget. Whether it works was previously unreadable from the code, the
    # config and the logs alike, and on 2026-08-17 that produced a confident
    # wrong root-cause report. So the daemon now measures it at boot and says
    # so, once, whatever the answer.
    #
    # `off` skips it. `warn` (default) logs. `refuse` makes an unsupported
    # model a startup failure — for operators who would rather not run than
    # run with the reasoning channel silently live.
    # NOTE the provider is asked whether suppression is on, not the config.
    # A second `model_config.get("suppress_thinking", ...)` here would be a
    # second reader of one setting — and would have demanded a real template
    # key for it, quietly undoing #246, which left it commented-out precisely
    # because absent already means true.
    verify_mode = str(model_config.get("verify_thinking_suppression", "warn")).lower()
    if verify_mode != "off" and hasattr(provider, "verify_thinking_suppression"):
        status, detail = await provider.verify_thinking_suppression()
        line = "thinking suppression: %s — %s (model=%s)"
        if status == "unsupported":
            logger.error(line, status.upper(), detail, model_name)
            logger.error(
                "  The reasoning channel is LIVE despite suppress_thinking. "
                "Expect turns that spend their whole budget thinking and "
                "return empty content. Set model.suppress_thinking: false to "
                "stop claiming otherwise, or serve a model whose template "
                "honours it."
            )
            if verify_mode == "refuse":
                raise RuntimeError(
                    f"thinking suppression unsupported for {model_name} and "
                    f"model.verify_thinking_suppression is 'refuse': {detail}"
                )
        elif status == "unknown":
            # Never reported as a pass — see the probe's own docstring.
            logger.warning(line, status.upper(), detail, model_name)
        else:
            logger.info(line, status, detail, model_name)

    # Context size detection. The value is CONSUMED (passed to the compactor
    # below), not just logged — it used to be logged only, so a config
    # `effective_limit` that outlived a model swap silently won. That is how a
    # server reporting n_ctx=32768 came to be budgeted at 72000: prompts were
    # built 2.2x larger than the server could hold, leaving no room to
    # generate, and every turn returned empty.
    detected_ctx_size: int | None = None
    if hasattr(provider, "detect_context_size"):
        detected_ctx_size = await provider.detect_context_size()
        if detected_ctx_size:
            logger.info("Server context size: %d tokens", detected_ctx_size)

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
        from prometheus.telemetry.cost import CostTracker, set_cost_tracker_handle
        cost_tracker = CostTracker()
        # Register the process-wide handle so the telemetry usage seam feeds it
        # (audit: was instantiated + reported but never .record()'d → always $0).
        set_cost_tracker_handle(cost_tracker)

    # Telemetry — shared instance for AgentLoop and SENTINEL digest.
    # Wired BEFORE build_tool_registry so per-tool registration failures
    # (Phase 2 — see prometheus.tools.registration.try_register) land in
    # ``subsystem_runs`` and surface to /health on the very first startup.
    telemetry = ToolCallTelemetry()
    # Sprint 4 A3: expose to gateway/commands.py for the /health command.
    from prometheus.telemetry.tracker import set_telemetry_handle
    set_telemetry_handle(telemetry)

    # Repair-pair flywheel: every adapter repair / retry-success /
    # self-correction becomes a training pair in training.db. Local capture
    # defaults ON (recording-only); cloud_golden_capture defaults OFF.
    from prometheus.learning.pair_capture import configure as configure_pair_capture
    configure_pair_capture(config.get("training", {}))

    # Tool registry — same tools as CLI mode
    registry = build_tool_registry(security_cfg=security_config)

    # DynamicToolLoader — deferred loading support
    from prometheus.context.dynamic_tools import DynamicToolLoader
    tool_loader = DynamicToolLoader(registry, config.get("tools", {}).get("deferred_loading"))
    # FIRSTLIGHT FL-2b: the advertised baseline gets ONE visible line at
    # boot. Before this, a config with no tools: section advertised zero
    # tools and nothing anywhere said so — the only traces were telemetry
    # rows and per-call "Lucky guess" lines.
    #
    # FL-2u: this counts what the LOADER RESOLVES, not what the config
    # says. The first version read len(config[...]["always_loaded"]) and
    # went stale the moment absence stopped meaning zero — it reported 0
    # while the model was being offered 11 tools. A record that describes
    # the config instead of the runtime is CROSS-CUTTING §12, and the
    # upgrade harness caught it by reading this very line.
    _dl_cfg = config.get("tools", {}).get("deferred_loading") or {}
    logger.info(
        "Tool advertisement baseline: %d always_loaded of %d registered "
        "(deferred mode: %s)",
        len(tool_loader.schemas_for_run(True)),
        len(registry.list_tools()),
        _dl_cfg.get("enabled", "auto"),
    )

    # Sprint 15 wiring fix: daemon was missing adapter, security_gate,
    # model_router, and divergence_detector — all were built but not connected.
    # Phase 2: router now requires primary provider + adapter + model built first.
    adapter = create_adapter(model_config, config.get("adapter"))
    security_gate = create_security_gate(security_config, getattr(args, "config", None))
    model_router = create_model_router(config, provider, adapter, model_name)
    # Phase 3: wire the router back into the adapter's RetryEngine so it can
    # consult router.config.escalation_enabled at tool-retry exhaustion and
    # return RetryAction.ESCALATE instead of ABORT.
    if adapter is not None and hasattr(adapter, "retry"):
        adapter.retry.router = model_router

    # Emit one INFO line per /claude /gpt /gemini /xai slash command showing
    # the provider+model it'll route to. Lets the user grep journalctl for
    # "slash_commands" instead of source-archaeology when /claude returns
    # something unexpected.
    from prometheus.router.model_router import log_slash_command_wiring
    log_slash_command_wiring(config, logger)

    divergence_detector = create_divergence_detector(config)

    # Sprint 15 wiring fix: HookExecutor was built but never created in daemon
    hook_executor = None
    try:
        from prometheus.hooks.executor import HookExecutor, HookExecutionContext
        from prometheus.hooks.events import HookEvent
        from prometheus.hooks.loader import load_hook_registry
        # H3: build the registry from the config `hooks:` section instead of an
        # always-empty one. Absent/empty `hooks:` yields an empty registry (no
        # behavior change); a populated section now actually loads + fires.
        hook_registry = load_hook_registry(config.get("hooks", {}) or {})
        hook_executor = HookExecutor(
            registry=hook_registry,
            context=HookExecutionContext(
                cwd=Path.cwd(),
                provider=provider,
                default_model=model_name,
            ),
        )
        _hook_count = sum(
            len(hook_registry.get(ev)) for ev in HookEvent
        )
        if _hook_count:
            logger.info("Hooks: loaded %d hook(s) from config", _hook_count)
    except Exception:
        logger.warning("HookExecutor init failed — hooks disabled", exc_info=True)

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
                # force-search (IGNITION): hand over the grammar SOURCE so per-call
                # required/{tool:X} grammars derive via the same enforcer path.
                if hasattr(provider, "set_grammar_source"):
                    provider.set_grammar_source(adapter.enforcer, registry.to_api_schema())
                logger.info(
                    "GBNF grammar updated (%d tool schemas)",
                    len(registry.list_tools()),
                )

    _update_grammar()

    # SUNRISE: PeriodicNudge — self-reflection prompt every N turns.
    # Optional; ``None`` disables injection in agent loop.
    nudge: object | None = None
    try:
        from prometheus.learning.nudge import PeriodicNudge
        nudge = PeriodicNudge.from_config(args.config)
        if nudge is not None and getattr(nudge, "enabled", False):
            logger.info(
                "PeriodicNudge: enabled (interval=%d turns)",
                getattr(nudge, "interval", 15),
            )
        else:
            nudge = None
    except Exception as exc:
        logger.warning("PeriodicNudge not available: %s", exc)

    # SPRINT-2 WS2: file-mutation verifier — per-turn audit of claimed vs
    # actual filesystem changes. Opt-out via config, on by default.
    fmv: object | None = None
    try:
        from prometheus.hooks.file_mutation_verifier import make_default_verifier
        fmv = make_default_verifier(config)
        if fmv is not None and getattr(fmv, "enabled", False):
            logger.info("FileMutationVerifier: enabled (turn-end audit)")
        else:
            fmv = None
    except Exception as exc:
        logger.warning("FileMutationVerifier not available: %s", exc)

    # SPRINT-CONTEXT-COMPACTOR: assembly-time span compaction, behind
    # compaction.enabled (default false → from_config returns None and the
    # loop is untouched). SignalBus late-wired in the SENTINEL block.
    compactor: object | None = None
    try:
        from prometheus.context.compactor import ContextCompactor
        compactor = ContextCompactor.from_config(
            config, provider=provider, model=model_name, telemetry=telemetry,
            detected_limit=detected_ctx_size,
        )
        if compactor is not None:
            # Report the RESOLVED budget, not the configured one. This line
            # used to print `_effective_limit` — the raw config number — while
            # the budget actually in force came from limit_for(). During the
            # 2026-08-14 context outage it read "limit=72000" on a box whose
            # server had 32768, which is precisely the wrong thing to tell
            # someone diagnosing an over-budget failure.
            _resolved = compactor.limit_for(model_name)
            _source = (
                "detected" if _resolved == (detected_ctx_size or -1)
                else "config"
            )
            logger.info(
                "Context compactor ENABLED (limit=%d [%s], cloud_default=%d, "
                "threshold=%.0f%%)",
                _resolved,
                _source,
                compactor._cloud_default_limit,
                compactor._threshold_pct * 100,
            )
        else:
            logger.info("Context compactor disabled (compaction.enabled unset)")
    except Exception as exc:
        logger.warning("Context compactor not available: %s", exc)

    # Agent profile state (selector survey 2026-08-11 → wired). ONE holder
    # every surface shares: profiles.default seeds it, /profile and Beacon's
    # PUT /api/profiles/active mutate it, and BOTH loop constructions resolve
    # through it per run — before this, the filter had no caller and /profile
    # wrote a name nobody read. Degrades to None (= unfiltered advertisement)
    # if the store cannot load; a broken custom YAML must not kill the daemon.
    profile_state = None
    profile_store = None
    try:
        from prometheus.config.profiles import ActiveProfileState, get_profile_store
        profile_store = get_profile_store()
        profile_state = ActiveProfileState(
            profile_store, config.get("profiles", {}).get("default", "full"),
        )
        logger.info("Agent profile active: %s", profile_state.name)
    except Exception as exc:
        logger.warning("profile store unavailable — advertisement unfiltered: %s", exc)

    # Agent loop
    agent_loop = AgentLoop(
        provider=provider,
        model=model_name,
        profile_resolver=profile_state.get if profile_state else None,
        tool_registry=registry,
        adapter=adapter,
        permission_checker=security_gate,
        hook_executor=hook_executor,
        telemetry=telemetry,
        model_router=model_router,
        divergence_detector=divergence_detector,
        post_result_hooks=post_result_hooks or None,
        max_tool_iterations=resolve_max_tool_iterations(model_config),
        # The DAEMON path was dropping this while __main__.py passed it, so
        # LoopContext.max_tool_iterations_cloud stayed None and
        # _effective_max_tool_iterations fell back to the LOCAL cap for every
        # provider. Live evidence: a grok-4.5 turn stopped at "26/25" with
        # max_tool_iterations_cloud: 50 sitting correctly in the config —
        # the config was right, the daemon just never received it.
        max_tool_iterations_cloud=resolve_max_tool_iterations_cloud(model_config),
        tool_loader=tool_loader,
        nudge=nudge,
        file_mutation_verifier=fmv,
        tool_result_max=config.get("context", {}).get("tool_result_max", 4000),
        # Selector-survey 2026-08-11: the five below had config keys the CLI
        # threaded and the daemon did not — every daemon surface ran on
        # dataclass defaults that coincidentally EQUALLED the live config
        # values (8000/3/False/200/500), so editing the key changed the CLI
        # and silently did nothing here. Same class as max_tool_iterations_cloud
        # above; the parity guard now checks CLI-threaded config kwargs too.
        tool_results_turn_budget=config.get("context", {}).get("tool_results_turn_budget", 8000),
        microcompact_after_turns=config.get("context", {}).get("microcompact_after_turns", 3),
        microcompact_on_cloud=config.get("context", {}).get("microcompact_on_cloud", False),
        microcompact_keep_chars=config.get("context", {}).get("microcompact_keep_chars", 200),
        microcompact_keep_chars_no_lcm=config.get("context", {}).get("microcompact_keep_chars_no_lcm", 500),
        compactor=compactor,
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

    # SPRINT G1 — gateway-generic subsystem wiring. Every constructed gateway
    # adapter is registered here, and every gateway-facing subsystem is
    # attached through the registry (never to one adapter by name). Adapters
    # registered later replay all earlier attachments, so construction order
    # doesn't matter. Adding a gateway (Discord — G2) = construct + register.
    gateway_registry = GatewaySubsystemRegistry()
    if cost_tracker is not None:
        gateway_registry.attach("cost_tracker", cost_tracker)

    # Telegram adapter
    telegram: TelegramAdapter | None = None
    telegram_token = gateway_config.get("telegram_token", "") or os.environ.get("PROMETHEUS_TELEGRAM_TOKEN", "")
    from prometheus.config.shipped_defaults import resolve_allowed_chat_ids

    _tg_chat_ids = resolve_allowed_chat_ids(gateway_config)

    # ABSENCE IS NOT PERMISSION. This block used to read
    # `gateway_config.get("telegram_enabled", True)`, so a config that merely
    # omitted the key started the public gateway as soon as a token existed —
    # and the adjacent `allowed_chat_ids`, omitted for the same reason, meant
    # "allow every chat". One defect, two key names.
    #
    # The decision lives in telegram_gateway_decision() because an inline
    # boolean here is untestable: run_daemon cannot be driven from a test, and
    # a mutation neutering this branch survived the entire suite.
    _tg_start, _tg_refusal = telegram_gateway_decision(
        gateway_config, telegram_token)
    if _tg_refusal:
        logger.error("%s", _tg_refusal)
    if _tg_start:
        # SPRINT G3: failure-guarded like the Slack/Discord blocks below —
        # a bad token (or Telegram being unreachable) must not kill the
        # daemon; the other gateways and the web surface still come up.
        try:
            # The media/rate fields were previously left to their dataclass
            # defaults here — the same shape as the two-loop defect: a field
            # not passed at the construction site silently takes a default,
            # so the config keys described controls nobody was reading.
            from prometheus.config.shipped_defaults import resolve_media_allowlist

            _media_cfg = (gateway_config.get("media") or {})
            _rate_cfg = (gateway_config.get("rate_limits") or {})
            tg_config = PlatformConfig(
                platform=Platform.TELEGRAM,
                token=telegram_token,
                allowed_chat_ids=_tg_chat_ids,
                proxy_url=gateway_config.get("proxy_url"),
                max_file_size_mb=_media_cfg.get("max_file_size_mb", 20),
                media_cache_dir=_media_cfg.get("cache_dir"),
                messages_per_minute=_rate_cfg.get("messages_per_minute", 30),
                media_downloads_per_minute=_rate_cfg.get(
                    "media_downloads_per_minute", 10
                ),
                # `... or []` here collapsed ABSENT and EXPLICITLY-EMPTY into
                # the same value, and media_guard reads [] as "no restriction"
                # — so every config predating #141 had NO type filtering on
                # the one surface exposed to the public internet. The resolver
                # keeps the two apart: absent -> the shipped allowlist,
                # `[]` -> the operator's deliberate opt-out.
                allowed_image_types=resolve_media_allowlist(
                    _media_cfg, "allowed_image_types"),
                allowed_audio_types=resolve_media_allowlist(
                    _media_cfg, "allowed_audio_types"),
                allowed_document_types=resolve_media_allowlist(
                    _media_cfg, "allowed_document_types"),
            )
            # Cache quota + free-disk floor (CONVENIENCE guards, fail open).
            from prometheus.gateway.media_cache import configure_cache

            configure_cache(
                max_mb=_media_cfg.get("cache_max_mb", 512),
                free_disk_floor_mb=_media_cfg.get("free_disk_floor_mb", 1024),
            )
            from prometheus.context.prompt_assembler import build_runtime_system_prompt
            from prometheus.skills.loader import skills_for_prompt
            system_prompt = build_runtime_system_prompt(
                cwd=str(Path.cwd()), config=config,
                skills=skills_for_prompt(),
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
                detected_context_size=detected_ctx_size,
            )
            # Same late-wiring pattern as memory_recall: the /profile command
            # mutates THE holder the loops resolve, instead of a per-adapter
            # attribute nothing read.
            telegram.profile_state = profile_state
            gateway_registry.register_adapter(telegram)
            await telegram.start()
            archive.archive_event("telegram_started")
            logger.info("Telegram adapter started")
        except Exception as exc:
            # Redact: python-telegram-bot's InvalidToken message embeds the
            # token itself — never let a token value reach the log.
            logger.error(
                "Failed to start Telegram adapter: %s",
                str(exc).replace(telegram_token, "***"),
            )
            telegram = None

    if telegram is not None:
        # Sprint 15b GRAFT: wire approval queue if enabled.
        # NOTE: the queue's *prompt delivery* transport is still the Telegram
        # adapter (constructor arg — adapter-into-subsystem, the reverse
        # direction); the queue itself is attached to every gateway so
        # /approve, /deny, /pending work from any surface.
        approval_cfg = security_config.get("approval_queue", {})
        if approval_cfg.get("enabled", False):
            from prometheus.permissions.approval_queue import ApprovalQueue
            default_chat = (gateway_config.get("allowed_chat_ids") or [None])[0]
            approval_queue = ApprovalQueue(
                telegram_adapter=telegram,
                # LITERAL on purpose: test_config_defaults_equality (#221) statically
                # parses this line to compare it with the template, and a named
                # constant reads as <no-default> and defeats the guard. Bound to
                # DEFAULT_APPROVAL_TIMEOUT_SECONDS by
                # test_daemon_fallback_matches_the_named_default.
                timeout_seconds=approval_cfg.get("timeout_seconds", 1800),
                default_chat_id=default_chat,
            )
            security_gate._approval_queue = approval_queue
            gateway_registry.attach("_approval_queue", approval_queue)
            # Reverse attachment: /approve session|always derive a Grant and
            # record it on the gate, so the shared command core (all
            # gateways) reaches the decider through the queue handle.
            approval_queue._security_gate = security_gate
            logger.info("Approval queue wired to gateway adapters")

    # WEAVE-PRESS: Printing Press CLI registry
    press_cfg = config.get("printing_press", {}) or {}
    if press_cfg.get("enabled", False):
        try:
            from prometheus.tools.printing_press import PrintingPressRegistry

            press_registry = PrintingPressRegistry(
                library_path=press_cfg.get("library_path") or None,
            )
            if not press_registry.is_available():
                logger.info(
                    "Printing Press: enabled but no library clone found "
                    "(searched ~/printing-press-library/, /tmp/printing-press-library/)"
                )
            else:
                logger.info(
                    "Printing Press: library at %s (%d CLIs)",
                    press_registry.library_path,
                    len(press_registry.list_available()),
                )
                # Wire the skill registry hot-reload callback. The
                # ToolSearchTool holds the SkillRegistry instance; ask it
                # for a handle and bind the reload to it.
                try:
                    ts = registry.get("tool_search")
                    skill_reg = ts.get_skill_registry() if ts else None
                    if skill_reg is not None and hasattr(
                        skill_reg, "reload_user_skills"
                    ):
                        press_registry.set_reload_callback(
                            skill_reg.reload_user_skills
                        )
                        logger.info("Printing Press: skill hot-reload wired")
                except Exception:
                    logger.debug(
                        "Printing Press: failed to wire skill reload callback",
                        exc_info=True,
                    )
                # Library auto-update on startup (don't block startup on it)
                if press_cfg.get("auto_update_library", False):
                    asyncio.create_task(
                        press_registry.update_library(),
                        name="printing_press_update",
                    )
            # Attach to the agent loop so the bash command-not-found hook
            # can offer suggestions to user-initiated sessions.
            if agent_loop._tool_metadata is None:
                agent_loop._tool_metadata = {}
            agent_loop._tool_metadata["printing_press"] = press_registry
            # And to every gateway adapter for /press commands.
            gateway_registry.attach("_printing_press", press_registry)
        except Exception:
            logger.exception("Printing Press: failed to initialise")

    # Slack adapter
    # Sprint Polish: also reads the nested gateway.slack.* form so the config
    # mirrors gateway.telegram. Old flat keys (slack_bot_token, slack_app_token,
    # slack_channels, slack_enabled) still win when set.
    slack_adapter = None
    slack_nested = gateway_config.get("slack") if isinstance(gateway_config.get("slack"), dict) else {}
    slack_bot_token = (
        gateway_config.get("slack_bot_token", "")
        or (slack_nested or {}).get("bot_token", "")
        or os.environ.get("PROMETHEUS_SLACK_BOT_TOKEN", "")
    )
    slack_app_token = (
        gateway_config.get("slack_app_token", "")
        or (slack_nested or {}).get("app_token", "")
        or os.environ.get("PROMETHEUS_SLACK_APP_TOKEN", "")
    )
    slack_enabled = bool(
        gateway_config.get("slack_enabled", False)
        or (slack_nested or {}).get("enabled", False)
    )
    if slack_bot_token and slack_app_token and slack_enabled:
        try:
            from prometheus.gateway.slack import SlackAdapter
            slack_config = PlatformConfig(
                platform=Platform.SLACK,
                token=slack_bot_token,
                app_token=slack_app_token,
                allowed_channels=(
                    gateway_config.get("slack_channels")
                    or (slack_nested or {}).get("allowed_channels", [])
                    or []
                ),
            )
            if "system_prompt" not in dir():
                from prometheus.context.prompt_assembler import build_runtime_system_prompt
                from prometheus.skills.loader import skills_for_prompt
                system_prompt = build_runtime_system_prompt(
                    cwd=str(Path.cwd()), config=config,
                    skills=skills_for_prompt(),
                )
            slack_adapter = SlackAdapter(
                config=slack_config,
                agent_loop=agent_loop,
                tool_registry=registry,
                system_prompt=system_prompt,
                model_name=model_name,
                model_provider=model_config.get("provider", "llama_cpp"),
                session_manager=session_manager,
                prometheus_config=config,
            )
            slack_adapter.profile_state = profile_state
            # SPRINT G1: registering replays every subsystem attached so far
            # (cost tracker, approval queue, printing press, …) onto Slack.
            gateway_registry.register_adapter(slack_adapter)
            await slack_adapter.start()
            archive.archive_event("slack_started")
            logger.info("Slack adapter started (Socket Mode)")
        except ImportError:
            logger.warning(
                "Slack is enabled but slack-bolt is not installed. "
                "Install with: pip install 'prometheus[slack]'"
            )
        except Exception as exc:
            msg = str(exc).replace(slack_bot_token, "***").replace(slack_app_token, "***")
            logger.error("Failed to start Slack adapter: %s", msg)

    # Discord adapter (SPRINT G2) — mirrors the Slack optional-construction
    # block: nested gateway.discord.* config, token via config or env var.
    # No per-subsystem wiring here: register_adapter replays every subsystem
    # attached so far and future attach() calls reach it automatically.
    discord_adapter = None
    discord_nested = gateway_config.get("discord") if isinstance(gateway_config.get("discord"), dict) else {}
    discord_token = (
        (discord_nested or {}).get("token", "")
        or os.environ.get("PROMETHEUS_DISCORD_TOKEN", "")
    )
    discord_enabled = bool((discord_nested or {}).get("enabled", False))
    if discord_token and discord_enabled:
        try:
            from prometheus.gateway.discord import DiscordAdapter
            discord_config = PlatformConfig(
                platform=Platform.DISCORD,
                token=discord_token,
                allowed_guild_ids=[
                    int(g) for g in (discord_nested or {}).get("guild_ids", []) or []
                ],
                allowed_channel_ids=[
                    int(c) for c in (discord_nested or {}).get("channel_ids", []) or []
                ],
            )
            if "system_prompt" not in dir():
                from prometheus.context.prompt_assembler import build_runtime_system_prompt
                from prometheus.skills.loader import skills_for_prompt
                system_prompt = build_runtime_system_prompt(
                    cwd=str(Path.cwd()), config=config,
                    skills=skills_for_prompt(),
                )
            discord_adapter = DiscordAdapter(
                config=discord_config,
                agent_loop=agent_loop,
                tool_registry=registry,
                system_prompt=system_prompt,
                model_name=model_name,
                model_provider=model_config.get("provider", "llama_cpp"),
                session_manager=session_manager,
                prometheus_config=config,
            )
            discord_adapter.profile_state = profile_state
            # SPRINT G1 contract: registering replays every subsystem attached
            # so far (cost tracker, approval queue, printing press, …) onto
            # Discord.
            gateway_registry.register_adapter(discord_adapter)
            await discord_adapter.start()
            archive.archive_event("discord_started")
            logger.info("Discord adapter started")
        except ImportError:
            logger.warning(
                "Discord is enabled but discord.py is not installed. "
                "Install with: pip install 'oara-prometheus[discord]'"
            )
        except Exception as exc:
            logger.error(
                "Failed to start Discord adapter: %s",
                str(exc).replace(discord_token, "***"),
            )

    # Heartbeat — also watches background tasks and pushes proactive
    # finish/fail + progress notifications to the user (audit fix #3).
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tasks.store import TaskStore
    tasks_config = config.get("tasks", {})
    _notify_chat = gateway_config.get("briefing_chat_id") or (
        gateway_config.get("allowed_chat_ids") or [None]
    )[0]
    # Managed tasks: configure the singleton manager with durability (TaskStore)
    # and system-trust command vetting (same SecurityGate as cron/agent). The
    # SignalBus is wired later (after SENTINEL init) so task_completed/_failed
    # can drive re-engagement; notification stays in the heartbeat regardless.
    task_manager = get_task_manager()
    try:
        task_manager.store = TaskStore()
    except Exception:
        logger.warning(
            "TaskStore init failed — tasks will not survive restart", exc_info=True
        )
    task_manager.security_gate = security_gate
    task_manager.default_timeout_seconds = int(
        tasks_config.get("default_timeout_seconds", task_manager.default_timeout_seconds)
    )
    task_manager.poll_initial_interval = float(
        tasks_config.get("poll_initial_interval_seconds", task_manager.poll_initial_interval)
    )
    task_manager.poll_max_interval = float(
        tasks_config.get("poll_max_interval_seconds", task_manager.poll_max_interval)
    )
    heartbeat = Heartbeat(
        gateway=telegram,
        task_manager=task_manager,
        notify_chat_id=_notify_chat,
        boot_sha=boot_sha,
        # Optional: suppress the merged-but-dark nudge inside an operator-armed
        # maintenance window. Empty by default (feature off).
        maintenance_db=(config.get("heartbeat", {}) or {}).get("maintenance_db", ""),
    )
    # Event-loop lag watchdog. Started ONCE here rather than per-turn, so it
    # also samples while the daemon is idle — an idle stall is exactly the
    # case a per-turn timer cannot see, and Beacon terminates the socket at
    # 45s of silence regardless of whether a turn is running.
    from prometheus.engine import loop_watchdog as _loop_watchdog
    watchdog_task = asyncio.create_task(_loop_watchdog.watch(),
                                        name="event-loop-watchdog")

    heartbeat_task = asyncio.create_task(heartbeat.run_forever())
    tasks.append(heartbeat_task)

    # Cron scheduler (skip if --telegram-only). Wire a failure-notification
    # path so cron job failures push a Telegram message — the heartbeat
    # task-watcher only sees BackgroundTaskManager tasks, not cron
    # subprocesses, so without this a failing daily briefing is silent.
    if not args.telegram_only:
        from prometheus.gateway.cron_scheduler import (
            set_cron_notifier,
            set_cron_security_gate,
        )
        set_cron_notifier(telegram, _notify_chat)
        # Vet cron commands through the SAME SecurityGate as the agent, at system
        # (restricted) trust, before they run unattended (see cron_scheduler).
        set_cron_security_gate(security_gate)
        # own_signals=False: the daemon owns SIGTERM/SIGINT; the embedded
        # scheduler must not re-register them (FL-1 — last registration
        # wins, and this one was stealing the daemon's shutdown).
        cron_task = asyncio.create_task(run_scheduler_loop(own_signals=False))
        tasks.append(cron_task)
        logger.info(
            "Cron scheduler started (failure notifier %s)",
            "wired" if (telegram is not None and _notify_chat) else "disabled",
        )

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

    # PR fix/memory-lcm-full-rewire (2026-05-26) — wire LCM into the
    # session manager so ChatSession.add_result_messages can persist
    # conversation messages to LCM. session_manager was constructed
    # earlier (line ~342) when lcm_engine wasn't yet available; assign
    # here. Sessions are created lazily by the gateway on first message,
    # which happens after this point, so this ordering is safe.
    if lcm_engine is not None:
        session_manager.lcm_engine = lcm_engine
        # Same late-wiring for the loop itself: run_async reads the attribute
        # per call, so every telegram/CLI/bakeoff turn from here on carries it
        # into its LoopContext and the microcompactor's is_ingested check has
        # a real engine instead of the always-None dead branch.
        agent_loop.lcm_engine = lcm_engine

    # Memory extractor (optional, from Sprint 5)
    memory_recall = None  # set below when the store comes up; web path reads it
    try:
        from prometheus.memory.extractor import MemoryExtractor
        from prometheus.memory.recall import MemoryRecall, RecallConfig
        from prometheus.memory.store import MemoryStore
        from prometheus.memory.wiki_compiler import WikiCompiler
        from prometheus.tools.builtin.wiki_compile import set_wiki_compiler

        memory_store = MemoryStore()

        # Wiki compiler — auto-compiles after each extraction run
        wiki_compiler = WikiCompiler(store=memory_store, wiki_root=wiki_root)
        set_wiki_compiler(wiki_compiler, memory_store)
        logger.info("Wiki compiler initialised at %s", wiki_compiler.wiki_root)

        # PASSIVE RECALL (MEMORY-3 follow-up): relevant stored facts ride each
        # turn's request-only system prompt — the read half of the extraction
        # pipeline, which was write-only to the model since Sprint 5.
        # Late-assigned onto agent_loop (built long before the store; same
        # pattern as session_manager.lcm_engine above) — run_async reads the
        # attribute per call, so this reaches every subsequent turn.
        recall_cfg = RecallConfig.from_config(config)
        if recall_cfg.enabled:
            memory_recall = MemoryRecall(store=memory_store, config=recall_cfg)
            agent_loop.memory_recall = memory_recall
            logger.info(
                "Memory recall wired (max_facts=%d, max_chars=%d, min_confidence=%.2f)",
                recall_cfg.max_facts,
                recall_cfg.max_chars,
                recall_cfg.min_confidence,
            )
        else:
            logger.info("Memory recall disabled (memory.recall.enabled=false)")

        extractor = MemoryExtractor(
            store=memory_store,
            provider=provider,
            model=model_name,
            post_extract_callback=wiki_compiler.compile,
            telemetry=telemetry,
        )

        # Wire extractor into LCM for pre-compaction flush
        if lcm_engine is not None:
            lcm_engine.set_memory_extractor(extractor)
            logger.info("Memory extractor wired to LCM pre-compaction flush")

        # SUNRISE: name the task so it's visible in asyncio.all_tasks() and logs.
        extractor_task = asyncio.create_task(
            extractor.run_forever(),
            name="memory_extractor",
        )
        tasks.append(extractor_task)
        logger.info("Memory extractor started (task=memory_extractor)")
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
    skill_creator = _wire_skill_creator(
        agent_loop,
        provider,
        model_name=model_name,
        learning_config=config.get("learning", {}) or {},
        telemetry=telemetry,
    )

    # Teacher escalation (SPRINT-TEACHER-ESCALATION): cloud-teacher recovery
    # for failed local agent turns. Inert unless escalation.teacher_model is
    # set in config; from_config always returns an engine so /escalations can
    # report the unarmed state. SignalBus is wired later in the SENTINEL
    # block (same late-wire pattern as SkillCreator).
    escalation_engine = None
    try:
        from prometheus.escalation.teacher import TeacherEscalation
        escalation_engine = TeacherEscalation.from_config(
            config,
            telemetry=telemetry,
            skill_creator=skill_creator,
        )
        gateway_registry.attach("escalation_engine", escalation_engine)
        if escalation_engine.is_armed:
            logger.info(
                "Teacher escalation armed: %s (max %d/session)",
                escalation_engine.stats()["teacher"],
                escalation_engine.stats()["max_per_session"],
            )
        else:
            logger.info("Teacher escalation inert (escalation.teacher_model unset)")
    except Exception as exc:
        logger.warning("Teacher escalation not available: %s", exc)

    # Learning loop — SkillRefiner (refine existing skills when execution deviates beneficially)
    try:
        from prometheus.learning.skill_refiner import SkillRefiner
        skill_refiner = SkillRefiner.from_config(
            provider, args.config, telemetry=telemetry
        )
        if skill_refiner is not None:
            # Override model to match the running model_name (from_config can't know it)
            skill_refiner._model = model_name
            agent_loop.add_post_task_hook(skill_refiner.maybe_refine_recent)
            logger.info("SkillRefiner wired to agent loop post-task hook")
        else:
            logger.info("SkillRefiner: disabled by config (learning.skill_refinement_enabled)")
    except Exception as exc:
        logger.warning("SkillRefiner not available: %s", exc)

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

            # SignalBus Persistence sprint: pass the live telemetry handle so
            # emissions land in signal_events synchronously. Falls back to
            # get_telemetry_handle() if omitted, but explicit is clearer.
            signal_bus = SignalBus(telemetry=telemetry)

            # Leaf components
            wiki_linter = WikiLinter(wiki_root=wiki_root)
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
                    telemetry=telemetry,
                    wiki_root=wiki_root,
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
            # Sprint S1 (visible memory & skills): SkillCreator/SkillRefiner
            # also publish to the bus so the gateways and Beacon can surface
            # skill_created / skill_refined events. The setter pattern lets
            # us delay the wire until after SignalBus is constructed
            # (SkillCreator is wired earlier in the daemon, before this block).
            if skill_creator is not None:
                skill_creator.signal_bus = signal_bus
            if "skill_refiner" in dir() and skill_refiner is not None:
                skill_refiner.signal_bus = signal_bus
            # SPRINT-TEACHER-ESCALATION: golden traces + escalation events
            # flow through the bus once it exists (engine falls back to a
            # direct telemetry row before this point).
            if escalation_engine is not None:
                escalation_engine.signal_bus = signal_bus
            # SPRINT-CONTEXT-COMPACTOR: compaction events flow through the
            # bus once it exists (the compactor falls back to a direct
            # telemetry row before this point).
            if compactor is not None:
                compactor.signal_bus = signal_bus
            # MemoryTool (hermes_memory_tool) emits memory_updated on
            # MEMORY.md / USER.md writes. Module-level setter matches the
            # tools/builtin/sentinel_status.py pattern.
            try:
                from prometheus.memory.hermes_memory_tool import (
                    set_memory_signal_bus,
                )
                set_memory_signal_bus(signal_bus)
            except Exception:
                logger.debug("memory signal bus wiring skipped", exc_info=True)
            # Sprint S1 Stream 2 + SPRINT G1: every gateway subscribes to
            # skill_created / skill_refined / memory_updated / curator_report
            # for user-facing notifications (default quiet mode). setattr
            # invokes each adapter's signal_bus property setter, which does
            # the actual subscribe.
            gateway_registry.attach("signal_bus", signal_bus)
            # Managed tasks: wire the bus so task_completed/task_failed drive
            # re-engagement. The completion handler injects task results back
            # into the creating session via the shared inject_turn primitive
            # (notification stays in the heartbeat). Re-engagement requires the
            # bus, so it is off when SENTINEL is disabled; notification still works.
            task_manager.signal_bus = signal_bus
            try:
                from prometheus.tasks.completion_handler import TaskCompletionHandler

                task_completion = TaskCompletionHandler(
                    signal_bus=signal_bus,
                    inject_turn=(
                        getattr(telegram, "inject_turn", None)
                        if "telegram" in dir() and telegram is not None
                        else None
                    ),
                    config=tasks_config,
                )
                await task_completion.start()
                logger.info("TaskCompletionHandler subscribed (managed-task reengage)")
            except Exception:
                logger.warning("TaskCompletionHandler wiring failed", exc_info=True)

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

    # SUNRISE: GoldenTraceExporter — periodic JSONL export of golden traces.
    # Runs independent of SENTINEL but uses the bus if SENTINEL is up so
    # downstream consumers (e.g. GEPAEngine) can react to fresh exports.
    try:
        trajectory_cfg = config.get("trajectory_export", {})
        if trajectory_cfg.get("enabled", False):
            from prometheus.sentinel.golden_trace_exporter import GoldenTraceExporter
            # The conversation store is what makes an exported trace
            # trainable: telemetry records the tool call, the LCM store holds
            # the conversation that prompted it, and the export rejoins them
            # by session_id. Without it every row is skipped as untrainable.
            try:
                from prometheus.memory.lcm_conversation_store import (
                    LCMConversationStore,
                )
                trace_conv_store = LCMConversationStore()
            except Exception:
                logger.warning(
                    "GoldenTraceExporter: no conversation store — golden "
                    "traces cannot be paired with their context and will be "
                    "skipped", exc_info=True,
                )
                trace_conv_store = None
            golden_exporter = GoldenTraceExporter(
                telemetry=telemetry,
                signal_bus=signal_bus if "signal_bus" in dir() else None,
                config=trajectory_cfg,
                conversation_store=trace_conv_store,
            )
            exporter_task = await golden_exporter.start()
            if exporter_task is not None:
                tasks.append(exporter_task)
                logger.info("GoldenTraceExporter started")
    except Exception as exc:
        logger.warning("GoldenTraceExporter not available: %s", exc)

    # Sprint S1 (visible memory & skills): Curator — periodic consolidation
    # pass over ~/.prometheus/skills/auto/. Two-stage pipeline (deterministic
    # state transitions + LLM-suggested consolidations/prunings). Pinned
    # skills protected. Prunings move files to auto/.archive/ — never delete.
    # Pattern adapted from Hermes agent/curator.py. See
    # prometheus/learning/curator.py for the design notes and divergences.
    try:
        from prometheus.learning.curator import Curator, set_curator
        curator = Curator.from_config(
            provider,
            model=model_name,
            signal_bus=signal_bus if "signal_bus" in dir() else None,
            config=config,
            telemetry=telemetry,
        )
        if curator is not None:
            curator_task = await curator.start()
            if curator_task is not None:
                tasks.append(curator_task)
                logger.info(
                    "Curator: wired (interval=%ds, stale=%dd, archive=%dd)",
                    curator._interval,
                    curator._stale_after_days,
                    curator._archive_after_days,
                )
            # Register the singleton so /curator commands can reach it.
            set_curator(curator)
        else:
            logger.info("Curator: disabled by config (learning.curator_enabled)")
    except Exception as exc:
        logger.warning("Curator not available: %s", exc)

    # Coding-mode Docker sandboxes: sweep containers a crashed or killed run
    # left behind. The config has always described this as happening "on
    # daemon start"; until this call existed it described nothing, and stale
    # containers accumulated until someone noticed by hand.
    try:
        _coding_cfg = config.get("coding", {}) or {}
        if _coding_cfg.get("docker_cleanup_enabled", True):
            from prometheus.coding.sandbox import (
                cleanup_stale_docker_containers,
                docker_available,
            )

            if docker_available():
                _max_age = float(
                    _coding_cfg.get("docker_cleanup_max_age_hours", 24)
                )
                _removed = await asyncio.to_thread(
                    cleanup_stale_docker_containers, max_age_hours=_max_age
                )
                if _removed:
                    logger.info(
                        "Coding sandboxes: removed %d stale container(s) "
                        "older than %.0fh",
                        len(_removed), _max_age,
                    )
    except Exception as exc:
        # Never block startup on housekeeping.
        logger.warning("Coding sandbox cleanup skipped: %s", exc)

    # GRAFT-SYMBIOTE Session A: SymbioteCoordinator (Scout → Harvest → Graft).
    # Tools were registered in create_tool_registry; the coordinator is
    # exposed via prometheus.symbiote.set_coordinator() so the tools and
    # /symbiote Telegram command can find it.
    try:
        symbiote_cfg = config.get("symbiote", {}) or {}
        if symbiote_cfg.get("enabled", False):
            from pathlib import Path as _Path
            from prometheus.symbiote import set_coordinator
            from prometheus.symbiote.coordinator import SymbioteCoordinator
            from prometheus.symbiote.github_search import (
                GitHubClient,
                GitHubSearchTool,
            )
            from prometheus.symbiote.graft import GraftEngine
            from prometheus.symbiote.harvest import HarvestEngine
            from prometheus.symbiote.license_gate import LicenseGate
            from prometheus.symbiote.scout import ScoutEngine
            from prometheus.symbiote.code_scanner import DangerousCodeScanner

            sym_license_gate = LicenseGate()
            sym_scanner = DangerousCodeScanner()
            sym_gh_client = GitHubClient.from_config(symbiote_cfg)
            sym_search_tool = GitHubSearchTool(client=sym_gh_client)
            sym_scout = ScoutEngine(
                github_search=sym_search_tool,
                license_gate=sym_license_gate,
                provider=provider,
                model=symbiote_cfg.get("scout_model", model_name),
            )
            sandbox_dir = _Path(
                symbiote_cfg.get("sandbox_dir", "~/.prometheus/symbiote/sandbox/")
            ).expanduser()
            harvest_dir = _Path(
                symbiote_cfg.get("harvest_dir", "~/.prometheus/symbiote/harvests/")
            ).expanduser()
            sym_harvest = HarvestEngine(
                scanner=sym_scanner,
                license_gate=sym_license_gate,
                provider=provider,
                model=symbiote_cfg.get("harvest_model", model_name),
                max_repo_size_mb=int(symbiote_cfg.get("max_repo_size_mb", 100)),
                file_budget_max=int(symbiote_cfg.get("file_budget_max", 15)),
                file_budget_kb=int(symbiote_cfg.get("file_budget_kb", 50)),
                clone_timeout=int(symbiote_cfg.get("clone_timeout_seconds", 60)),
                sandbox_root=sandbox_dir,
                harvest_root=harvest_dir,
            )
            sym_graft = GraftEngine(scanner=sym_scanner, project_root=Path.cwd())
            sym_db_path = _Path(
                symbiote_cfg.get("sessions_db", "~/.prometheus/symbiote/sessions.db")
            ).expanduser()
            sym_db_path.parent.mkdir(parents=True, exist_ok=True)
            symbiote_coordinator = SymbioteCoordinator(
                scout_engine=sym_scout,
                harvest_engine=sym_harvest,
                graft_engine=sym_graft,
                db_path=sym_db_path,
            )
            set_coordinator(symbiote_coordinator)
            logger.info("SymbioteCoordinator wired (sessions=%s)", sym_db_path)

            # GRAFT-SYMBIOTE Session B: BackupVault + MorphEngine.
            # Attached to the Telegram adapter so /symbiote
            # backup/backups/restore/morph/swap can find them.
            backup_cfg = symbiote_cfg.get("backup", {}) or {}
            morph_cfg = symbiote_cfg.get("morph", {}) or {}
            sym_backup_vault = None
            if backup_cfg.get("enabled", True):
                from prometheus.symbiote.backup_vault import BackupVault
                sym_backup_vault = BackupVault(
                    project_root=Path.cwd(),
                    vault_root=_Path(
                        backup_cfg.get("vault_root", "~/.prometheus/symbiote/backups")
                    ).expanduser(),
                    max_backups=int(backup_cfg.get("max_backups", 10)),
                    include_identity=bool(backup_cfg.get("include_identity", True)),
                    include_config=bool(backup_cfg.get("include_config", True)),
                    exempt_from_retention=set(backup_cfg.get("exempt_from_retention") or []) or None,
                )
                gateway_registry.attach("_backup_vault", sym_backup_vault)
                logger.info("BackupVault wired (vault_root=%s)", sym_backup_vault._vault_root)

            if morph_cfg.get("enabled", False) and sym_backup_vault is not None:
                from prometheus.symbiote.morph import MorphEngine
                manager_override = morph_cfg.get("daemon_manager", "auto")
                manager_override = manager_override if manager_override != "auto" else None
                sym_morph_engine = MorphEngine(
                    backup_vault=sym_backup_vault,
                    project_root=Path.cwd(),
                    candidate_root=_Path(
                        morph_cfg.get("candidate_dir", "~/.prometheus/symbiote/candidate/")
                    ).expanduser(),
                    post_mortem_root=_Path(
                        morph_cfg.get("post_mortem_dir", "~/.prometheus/symbiote/post_mortem/")
                    ).expanduser(),
                    health_check_timeout=int(
                        morph_cfg.get("health_check_timeout_seconds", 60)
                    ),
                    health_check_interval=int(
                        morph_cfg.get("health_check_interval_seconds", 5)
                    ),
                    consecutive_passes_required=int(
                        morph_cfg.get("consecutive_passes_required", 3)
                    ),
                    daemon_manager_override=manager_override,
                    daemon_health_url=morph_cfg.get("daemon_health_url"),
                )
                gateway_registry.attach("_morph_engine", sym_morph_engine)
                logger.info(
                    "MorphEngine wired (manager_override=%s)", manager_override or "auto",
                )
    except Exception as exc:
        logger.warning("SymbioteCoordinator not available: %s", exc)

    # SUNRISE Session B: GEPA — idle-time skill evolution.
    # Requires SENTINEL (signal_bus) to be available. Subscribes to idle
    # signals; runs at most once per gepa_max_frequency_hours after
    # gepa_min_idle_minutes of continuous idle.
    gepa_engine = None
    try:
        learning_cfg = config.get("learning", {}) or {}
        if learning_cfg.get("gepa_enabled", False) and "signal_bus" in dir():
            from prometheus.learning.gepa import GEPAOptimizer
            from prometheus.sentinel.gepa_engine import GEPAEngine

            evals_cfg = config.get("evals", {}) or {}
            gepa_optimizer = GEPAOptimizer(
                provider=provider,
                judge_base_url=evals_cfg.get("judge_base_url"),
                telemetry=telemetry,
                config=learning_cfg,
            )
            gepa_engine = GEPAEngine(
                optimizer=gepa_optimizer,
                signal_bus=signal_bus,
                config=learning_cfg,
            )
            await gepa_engine.start()
            # Expose to every gateway adapter so /gepa commands can find it.
            gateway_registry.attach("_gepa_engine", gepa_engine)
            logger.info("GEPAEngine started")
    except Exception as exc:
        logger.warning("GEPAEngine not available: %s", exc)

    # Web bridge (Beacon dashboard backend)
    web_config = config.get("web", {})
    if web_config.get("enabled", False):
        # ── API token bootstrap (Onboarding Phase 0, item 2) ────────────
        # Web-on must not mean open-by-accident: with no token configured
        # anywhere, mint one, persist it to the env file, and print it
        # ONCE. An explicit empty PROMETHEUS_API_TOKEN= stays OPEN
        # (deliberate). Either way the auth state is logged every start.
        from prometheus.config.api_token import (
            describe_web_auth,
            ensure_api_token,
            format_minted_banner,
        )
        try:
            _token, _minted = ensure_api_token(config)
            if _minted:
                print(format_minted_banner(_token), flush=True)
                logger.warning(
                    "web auth: NEW API token generated and saved to %s "
                    "(printed once above — `prometheus token show` re-prints it)",
                    get_env_file_path(),
                )
        except Exception:
            logger.error("API token bootstrap failed", exc_info=True)
        logger.info(describe_web_auth(config))
        try:
            from prometheus.web.launcher import launch_web
            from prometheus.engine.agent_loop import LoopContext

            # Build system prompt if not already available
            if "system_prompt" not in dir():
                from prometheus.context.prompt_assembler import build_runtime_system_prompt
                from prometheus.skills.loader import skills_for_prompt
                system_prompt = build_runtime_system_prompt(
                    cwd=str(Path.cwd()), config=config,
                    skills=skills_for_prompt(),
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
                tool_result_max=config.get("context", {}).get("tool_result_max", 4000),
                # ...and the SEVENTH (selector-survey 2026-08-11) — a NEW
                # geometry the drift guard was structurally blind to: these
                # five had config keys that only __main__.py threaded, so BOTH
                # daemon constructions lacked them EQUALLY and the two-loop
                # comparison saw no drift. Masked further by coincidence: the
                # dataclass defaults equal today's live config values, so
                # behavior matched until the day someone edited the key. The
                # parity guard now also checks CLI-threaded config kwargs
                # against both daemon sites.
                tool_results_turn_budget=config.get("context", {}).get("tool_results_turn_budget", 8000),
                microcompact_after_turns=config.get("context", {}).get("microcompact_after_turns", 3),
                microcompact_on_cloud=config.get("context", {}).get("microcompact_on_cloud", False),
                microcompact_keep_chars=config.get("context", {}).get("microcompact_keep_chars", 200),
                microcompact_keep_chars_no_lcm=config.get("context", {}).get("microcompact_keep_chars_no_lcm", 500),
                # Phase 3.5: web bridge is its own session namespace.
                #
                # READ THIS BEFORE "FIXING" IT. This is a ROUTING NAMESPACE that
                # happens to sit in a field called session_id — it is NOT this
                # turn's conversation id, because ONE context is shared by every
                # web session (run_async builds a per-call context; the web path
                # cannot). Readers that want the conversation must use the
                # per-call argument: `session_id or context.session_id`, the
                # idiom already used for the FMV turn key, the ephemeral check
                # and the divergence task id, and named `effective_session_id`
                # in _run_loop.
                #
                # The literal is LOAD-BEARING and must not be changed to None or
                # to a real id: "web" is a member of _USER_SESSION_LITERALS
                # (permissions/checker.py), so origin_from_session_id("web") is
                # USER — while "web:" is NOT in _USER_SESSION_PREFIXES, making
                # origin_from_session_id("web:abc") SYSTEM. That classification
                # decides whether a human counts as present to sanction the next
                # tool call, so swapping this value silently demotes every
                # Beacon turn to the stricter class. Descriptive readers only.
                session_id="web",
                # Sprint 2 (OAra): the web path was the ONLY path without the
                # compactor — AgentLoop.run_async() threads it for telegram/CLI,
                # but this pre-built context never got it, so web/Beacon/Bridge
                # turns could never trigger assembly-time compaction even with
                # compaction.enabled=true. One line, months of config-dark.
                compactor=compactor,
                # PASSIVE RECALL: same compactor lesson — this pre-built
                # context bypasses AgentLoop.run_async, so thread it here or
                # web/Beacon turns silently never recall. None when the
                # extractor block failed or recall is disabled.
                memory_recall=memory_recall,
                # Agent profile: same holder as the AgentLoop path, resolved
                # per run — a Beacon profile switch reaches the next web turn
                # through this pre-built context too, not just telegram/CLI.
                profile_resolver=profile_state.get if profile_state else None,
                # THE SAME LESSON, THIRD TIME (2026-07-31). Everything below is
                # config that AgentLoop threads for telegram/CLI and that this
                # pre-built context silently dropped, so web/Beacon turns ran on
                # dataclass defaults no matter what the config said:
                #   * the iteration caps — Beacon was pinned to the LOCAL 25
                #     even on cloud models, the bug #124 fixed one call site up
                #   * tool_loader — without it deferred loading (#121) is inert
                #     on the web path and /api/tools/deferred cannot report the
                #     effective state ("daemon loop not wired").
                # If you add a LoopContext field that AgentLoop populates, add
                # it HERE too or Beacon quietly won't have it.
                max_tool_iterations=resolve_max_tool_iterations(model_config),
                max_tool_iterations_cloud=resolve_max_tool_iterations_cloud(model_config),
                tool_loader=tool_loader,
                # ...and the SAME LESSON, FOURTH TIME. `lsp.enabled: true` is
                # live, so this list holds the LSPDiagnosticsHook that appends
                # type errors to every write_file/edit_file result — on
                # telegram/CLI only. Beacon is the primary CODING surface and
                # was the one path that never saw a diagnostic. The hook is
                # stateless (orchestrator + delay; the orchestrator is already
                # a process-wide singleton that web turns reach through
                # LSPTool), so sharing it across concurrent web turns is safe.
                # Costs up to diagnostics_delay_ms per file write, same as the
                # other paths already pay.
                post_result_hooks=post_result_hooks or None,
                # ...and the FIFTH. This one needed the hook fixed first, not
                # just wiring: `fmv` is the one field here that is mutable
                # STATE rather than config, and it held a single flat
                # accumulator with no turn key. Sharing this context across
                # concurrent Beacon turns would have made the first turn to
                # finish report the other's writes as its own — so it was
                # carved out of the parity guard until the accumulator became
                # turn-scoped (run_loop now mints one key per invocation and
                # drops it in a `finally`). The same defect was already live
                # on telegram-vs-cron, which share this instance too.
                file_mutation_verifier=fmv,
                # ...and the SIXTH — the first that the field-level parity
                # guard could NOT have caught: the nudge was never a
                # LoopContext field at all, it was injected by
                # AgentLoop.run_async around the loop. Web/Beacon/Bridge
                # bypass run_async, so with learning.nudge_enabled: true in
                # the live config the self-reflection prompt reached
                # telegram/CLI and nothing else. See
                # tests/test_run_async_web_parity.py for the guard that
                # catches this whole class (run_async-only behaviour) rather
                # than just the field-level drift.
                nudge=nudge,
                # ...and the SEVENTH — the first that NEITHER guard could see,
                # because no path passed it: LoopContext.lcm_engine existed
                # only as a dataclass default, so the microcompactor's
                # is_ingested branch was dead code and microcompact_keep_chars
                # was unreachable on every surface. Shared-state analysis (the
                # fmv lesson, applied BEFORE wiring this time): the loop's one
                # consumer (_microcompact_old_results) is read-only —
                # is_ingested → SELECT 1 on a WAL sqlite connection created
                # with check_same_thread=False — synchronous (atomic under the
                # event loop) and holding no per-turn state, so concurrent web
                # turns sharing this instance cannot cross-talk. The same
                # engine already serves concurrent web writes via session
                # persistence (session_manager.lcm_engine) and the REST
                # history routes; this adds a point-read to that established
                # sharing regime. See tests/test_microcompact.py::
                # TestSharedEngineSafety.
                lcm_engine=lcm_engine,
            )

            # ── Startup agreement check: the two loops must carry the same
            # iteration ceilings ──────────────────────────────────────────
            #
            # run_daemon() builds the loop TWICE -- AgentLoop above (telegram,
            # CLI) and this LoopContext (web, Beacon) -- and the comment on the
            # max_tool_iterations lines above is a MANUAL instruction to keep
            # them in step. Manual discipline is what produced the bug that
            # comment memorialises: the daemon path once dropped
            # max_tool_iterations_cloud entirely, so a grok-4.5 turn stopped at
            # "26/25" while the config correctly said 50.
            #
            # Both objects are in hand exactly here, so the "they call the same
            # pure function on the same input, therefore they agree" inference
            # becomes a CHECKED INVARIANT. Unconditional -- not gated on a debug
            # flag or a config key -- because a check that only runs when
            # someone remembers to enable it has the same failure mode as the
            # comment it replaces.
            #
            # Loud, not fatal: a mismatch means one surface silently enforces a
            # different ceiling than the other, which is worth paging about, but
            # refusing to boot the web layer over it would take down the daemon
            # for a discrepancy the operator can see and fix in the log.
            _diverged = loop_ceiling_divergence(agent_loop, loop_context)
            if _diverged:
                logger.error(
                    "LOOP CEILING DIVERGENCE — the telegram/CLI loop and the "
                    "web/Beacon loop disagree, so the same request hits a "
                    "different limit depending on which surface it arrives on: "
                    "%s",
                    "; ".join(
                        f"{n}: AgentLoop={a!r} vs LoopContext={b!r}"
                        for n, a, b in _diverged
                    ),
                )
                _tel = telemetry
                if _tel is not None and hasattr(_tel, "record_silent_failure"):
                    try:
                        _tel.record_silent_failure(
                            subsystem="daemon",
                            operation="loop_ceiling_agreement",
                            exc=RuntimeError("loop ceiling divergence"),
                            context={
                                n: {"agent_loop": a, "loop_context": b}
                                for n, a, b in _diverged
                            },
                        )
                    except Exception:
                        logger.debug(
                            "ceiling divergence: telemetry write failed",
                            exc_info=True,
                        )
            else:
                logger.info(
                    "Loop ceilings agree across both constructions: "
                    "local=%s cloud=%s",
                    loop_context.max_tool_iterations,
                    loop_context.max_tool_iterations_cloud,
                )

            # Beacon D1 + selector survey: the store AND the active-profile
            # holder are built once, before the AgentLoop — the web layer gets
            # the same objects so GET /api/profiles lists the real profiles
            # and PUT /api/profiles/active mutates the state the loops
            # actually resolve, instead of a cosmetic app.state string.

            api_port = web_config.get("api_port", 8005)
            ws_port = web_config.get("ws_port", 8010)
            # Live recorder: give the web layer the daemon's SkillCreator
            # (signal_bus-wired) and the live SkillRegistry so trace uploads
            # persist through THE auto-skill write path and the running
            # agent picks new skills up via reload_user_skills().
            try:
                ts = registry.get("tool_search")
                web_skill_registry = ts.get_skill_registry() if ts else None
            except Exception:
                web_skill_registry = None

            web_task = asyncio.create_task(launch_web(
                config=config,
                boot_sha=boot_sha,
                signal_bus=signal_bus if "signal_bus" in dir() else None,
                session_mgr=session_manager,
                telemetry=telemetry,
                skill_registry=web_skill_registry,
                lcm_engine=lcm_engine if "lcm_engine" in dir() else None,
                agent_loop=agent_loop,
                approval_queue=approval_queue if "approval_queue" in dir() else None,
                loop_context=loop_context,
                profile_store=profile_store,
                profile_state=profile_state,
                skill_creator=skill_creator,
                # None when Telegram is disabled — /api/status then reports
                # gateway.wired=false rather than a health verdict.
                gateway_adapter=telegram,
                api_port=api_port,
                ws_port=ws_port,
            ))
            tasks.append(web_task)
            logger.info("Web bridge started (REST :%d, WS :%d)", api_port, ws_port)
        except Exception as exc:
            logger.warning("Web bridge not available: %s", exc)

    # Managed tasks durability: resume file_watch/poll watchers and reap
    # orphaned process tasks left "running" by a previous daemon process, so no
    # zombie rows survive a restart. Unconditional (runs even if SENTINEL/bus is
    # off) — reaping updates the durable store regardless of signal emission.
    try:
        await task_manager.resume_running()
    except Exception:
        logger.warning("Task resume_running failed", exc_info=True)

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

    if discord_adapter:
        await discord_adapter.stop()

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

    log_level = logging.DEBUG if args.debug else logging.INFO

    # ── Setup mode (Onboarding Phase 1) ─────────────────────────────────
    # No config anywhere → don't half-start on defaults (the old dead
    # end): boot the minimal pairing-only web server instead, so a client
    # (Beacon) can exchange a one-time code for the real API token while
    # the user finishes `prometheus setup`. Checked BEFORE the file
    # logger below — setup mode must not create ~/.prometheus/logs (or
    # any other ~/.prometheus state).
    from prometheus.web.setup_server import (
        SETUP_COMPLETE,
        find_config_file,
        run_setup_mode,
    )
    if args.config and not Path(args.config).expanduser().is_file():
        # An EXPLICIT --config that doesn't exist is a broken invocation —
        # fail loudly rather than silently falling back to the user config
        # (old behavior) or surprising the operator with setup mode.
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    if find_config_file(args.config) is None:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        result = run_setup_mode()
        # Phase 2: POST /api/setup/complete exits the serve loop with a
        # restart sentinel. RE-CHECK for config — present now → fall
        # through into the normal daemon boot IN THIS SAME PROCESS (no
        # systemd needed; under systemd a plain exit + Restart= also
        # works, but the fallthrough must work standalone).
        if result != SETUP_COMPLETE:
            sys.exit(result)
        if find_config_file(args.config) is None:
            print(
                "Setup reported complete but no config was found — "
                "run `prometheus setup` and restart the daemon.",
                file=sys.stderr,
            )
            sys.exit(1)
        logger.info(
            "Setup complete — starting the real daemon in this process."
        )

    # Logging (force=True: after a setup-mode fallthrough the root logger
    # already has the stdout-only handler installed above and basicConfig
    # would otherwise be a silent no-op — the file handler must attach).
    log_dir = get_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "daemon.log"),
        ],
        force=True,
    )
    # SPRINT G3: httpx logs every request URL at INFO — and the Telegram
    # Bot API puts the bot TOKEN in the URL path (…/bot<token>/getMe), so
    # httpx-at-INFO leaks the token into the daemon log on every Telegram
    # API call. WARNING keeps real failures visible without the URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(run_daemon(args))


if __name__ == "__main__":
    main()
