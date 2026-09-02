"""``oara doctor`` — first-line diagnostics for a Prometheus install.

Onboarding Phase 0, item 4: the :class:`prometheus.infra.doctor.Doctor`
class existed (daemon startup check + /doctor Telegram command) but had
no CLI entry point. This module gives it one, and adds the onboarding
checks a fresh install actually trips over:

- config found + parses (and WHICH config, per the documented search order)
- inference server reachable + model detected
- web port free / already served by Prometheus
- API token set (auth ENABLED vs OPEN)
- ~/.prometheus data dirs writable
- whisper available when voice is enabled

Output is human-readable ✓/✗ lines; the exit code is nonzero when any
check errors — "run oara doctor" is the eternal support answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

from prometheus.config.api_token import resolve_api_token
from prometheus.config.defaults import config_search_paths
from prometheus.config.paths import (
    get_config_dir,
    get_data_dir,
    get_logs_dir,
    get_workspace_dir,
)
from prometheus.infra.doctor import DiagnosticCheck

import logging

log = logging.getLogger(__name__)

_SYMBOLS = {"ok": "✓", "error": "✗", "warning": "!", "info": "·"}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_config_path(explicit: str | None = None) -> tuple[Path | None, list[Path]]:
    """Return ``(found_path_or_None, searched_paths)``.

    The candidate list comes from :func:`prometheus.config.defaults.config_search_paths`
    — the search order documented in the README and prometheus.yaml.default.
    This function keeps doctor's own contract on top of it: a ``None`` when
    nothing was found (doctor REPORTS absence rather than falling back), and
    the paths actually probed so the failure message can name them.

    ⚠ It used to inline its own ``parents[3]``. That was correct, and the
    identical expression in ``config/defaults.py`` was not — five hops instead
    of four, naming a file one directory above the repo root. Two independent
    hop counts is the shape that lets one of them be wrong for months; there is
    now one.
    """
    searched: list[Path] = []
    for candidate in config_search_paths(explicit):
        searched.append(candidate)
        if candidate.is_file():
            return candidate, searched
    return None, searched


# ---------------------------------------------------------------------------
# Individual checks (each returns a DiagnosticCheck)
# ---------------------------------------------------------------------------

def check_config(explicit: str | None = None) -> tuple[DiagnosticCheck, dict[str, Any]]:
    """Config exists and parses. Returns (check, parsed_config_or_{})."""
    path, searched = resolve_config_path(explicit)
    if path is None:
        return DiagnosticCheck(
            name="Config", category="platform", status="error",
            message="no prometheus.yaml found (searched: "
                    + ", ".join(str(p) for p in searched) + ")",
            fix="Run `oara setup` to create one.",
        ), {}
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return DiagnosticCheck(
            name="Config", category="platform", status="error",
            message=f"{path} has a YAML parse error: {exc}",
            fix=f"Fix the syntax in {path}, or re-run `oara setup`.",
        ), {}
    if not isinstance(config, dict):
        return DiagnosticCheck(
            name="Config", category="platform", status="error",
            message=f"{path} does not contain a YAML mapping",
            fix="Re-run `oara setup` to regenerate it.",
        ), {}
    return DiagnosticCheck(
        name="Config", category="platform", status="ok",
        message=f"loaded {path}",
    ), config


def check_inference(config: dict[str, Any], timeout: float = 5.0) -> tuple[DiagnosticCheck, DiagnosticCheck]:
    """Inference server reachable + a model detected. Returns two checks."""
    model_cfg = config.get("model", {}) or {}
    provider = model_cfg.get("provider", "llama_cpp")

    from prometheus.providers.registry import ProviderRegistry
    if ProviderRegistry.is_cloud(provider):
        key_env = model_cfg.get("api_key_env", "")
        has_key = bool(key_env and os.environ.get(key_env))
        reach = DiagnosticCheck(
            name="Inference", category="connectivity",
            status="ok" if has_key else "error",
            message=(f"cloud provider {provider} (key ${key_env} set)" if has_key
                     else f"cloud provider {provider} but ${key_env or '<api_key_env unset>'} is not set"),
            fix=None if has_key else f"Export {key_env or 'the provider API key'} "
                                     f"or add it to the env file.",
        )
        model = DiagnosticCheck(
            name="Model", category="model", status="ok" if model_cfg.get("model") else "warning",
            message=f"configured: {model_cfg.get('model') or '(none)'}",
            fix=None if model_cfg.get("model") else "Set model.model in prometheus.yaml.",
        )
        return reach, model

    base_url = (model_cfg.get("base_url") or "http://localhost:8080").rstrip("/")
    detected: list[str] = []
    error: str | None = None
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        detected = [str(m.get("id", "")) for m in resp.json().get("data", [])]
    except Exception as exc:
        error = str(exc)
        if provider == "ollama":
            try:
                resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
                resp.raise_for_status()
                detected = [str(m.get("name", "")) for m in resp.json().get("models", [])]
                error = None
            except Exception as exc2:
                error = str(exc2)

    if error is not None:
        reach = DiagnosticCheck(
            name="Inference", category="connectivity", status="error",
            message=f"{provider} not responding at {base_url}",
            fix="Start the inference server (or fix model.base_url), "
                "then re-run `oara doctor`.",
        )
        model = DiagnosticCheck(
            name="Model", category="model", status="error",
            message="no model detected (server unreachable)",
            fix="Load a model in llama.cpp / ollama.",
        )
        return reach, model

    reach = DiagnosticCheck(
        name="Inference", category="connectivity", status="ok",
        message=f"{provider} reachable at {base_url}",
    )
    if detected and detected[0]:
        model = DiagnosticCheck(
            name="Model", category="model", status="ok",
            message=f"detected: {detected[0]}"
                    + (f" (+{len(detected) - 1} more)" if len(detected) > 1 else ""),
        )
    else:
        model = DiagnosticCheck(
            name="Model", category="model", status="error",
            message="server reachable but no model loaded",
            fix="Load a model (e.g. `ollama pull <model>` or restart "
                "llama-server with -m <model>.gguf).",
        )
    return reach, model


def check_web_port(config: dict[str, Any], timeout: float = 3.0) -> DiagnosticCheck:
    """Web enabled → is the API port free, or already served by Prometheus?"""
    web_cfg = config.get("web", {}) or {}
    if not web_cfg.get("enabled", False):
        return DiagnosticCheck(
            name="Web", category="connectivity", status="warning",
            message="web.enabled is false — Beacon/REST API is OFF",
            fix="Set web.enabled: true (the setup wizard now does this by default).",
        )
    port = int(web_cfg.get("api_port", 8005))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        listening = sock.connect_ex(("127.0.0.1", port)) == 0
    if not listening:
        return DiagnosticCheck(
            name="Web", category="connectivity", status="ok",
            message=f"port {port} free (daemon not running — web API will bind on start)",
        )
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=timeout)
        if resp.status_code in (200, 401):
            suffix = " (auth required)" if resp.status_code == 401 else ""
            return DiagnosticCheck(
                name="Web", category="connectivity", status="ok",
                message=f"Prometheus web API serving on :{port}{suffix}",
            )
    except Exception:
        pass
    return DiagnosticCheck(
        name="Web", category="connectivity", status="error",
        message=f"port {port} is in use by something that isn't Prometheus",
        fix=f"Free port {port} or change web.api_port in prometheus.yaml.",
    )


def check_token(config: dict[str, Any]) -> DiagnosticCheck:
    """API token configured?"""
    token, source = resolve_api_token(config)
    if token:
        return DiagnosticCheck(
            name="API token", category="connectivity", status="ok",
            message=f"web auth ENABLED (token from {source})",
        )
    return DiagnosticCheck(
        name="API token", category="connectivity", status="warning",
        message="web auth OPEN — no PROMETHEUS_API_TOKEN set",
        fix="Run `oara token rotate` (the daemon also mints one "
            "automatically on first start with web enabled).",
    )


def check_dirs_writable() -> DiagnosticCheck:
    """~/.prometheus (or $PROMETHEUS_CONFIG_DIR) data dirs writable?"""
    failures: list[str] = []
    for label, path in (
        ("config", get_config_dir()),
        ("data", get_data_dir()),
        ("logs", get_logs_dir()),
        ("workspace", get_workspace_dir()),
    ):
        probe = path / f".doctor-probe-{uuid.uuid4().hex[:8]}"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            failures.append(f"{label} ({path})")
    if failures:
        return DiagnosticCheck(
            name="Data dirs", category="platform", status="error",
            message="not writable: " + ", ".join(failures),
            fix="Fix ownership/permissions on the listed directories.",
        )
    return DiagnosticCheck(
        name="Data dirs", category="platform", status="ok",
        message=f"writable ({get_config_dir()})",
    )


def _env_or_env_file(name: str) -> str:
    """A gateway secret the daemon would see: process env, then env file.

    ``oara doctor`` usually runs in a shell that has NOT sourced
    ``~/.config/prometheus/env`` — but the daemon loads it, so a token
    that lives only there still counts as present.
    """
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        from prometheus.config.env_file import get_env_file_path, parse_env_file
        return parse_env_file(get_env_file_path()).get(name, "")
    except Exception:
        return ""


def _library_installed(module: str) -> bool:
    """Importability probe without importing (no side effects)."""
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _gateway_check(
    label: str,
    *,
    enabled: bool,
    token_ok: bool,
    token_missing_msg: str,
    token_fix: str,
    library_module: str,
    library_label: str,
    library_fix: str,
) -> DiagnosticCheck:
    """One gateway's doctor line — the whisper-if-voice-enabled pattern:
    enabled-without-token = error with a fix hint; enabled-without-library
    = error suggesting the pip extra; disabled = info."""
    if not enabled:
        return DiagnosticCheck(
            name=label, category="connectivity", status="info",
            message="not enabled",
        )
    if not token_ok:
        return DiagnosticCheck(
            name=label, category="connectivity", status="error",
            message=token_missing_msg,
            fix=token_fix,
        )
    if not _library_installed(library_module):
        return DiagnosticCheck(
            name=label, category="connectivity", status="error",
            message=f"enabled with token(s) present, but {library_label} "
                    "is not installed",
            fix=library_fix,
        )
    return DiagnosticCheck(
        name=label, category="connectivity", status="ok",
        message=f"enabled — token present, {library_label} installed",
    )


def check_gateways(config: dict[str, Any]) -> list[DiagnosticCheck]:
    """SPRINT G3: per-gateway checks for Telegram, Slack, and Discord.

    Mirrors the daemon's construction blocks (daemon.py): the same
    config keys, the same env fallbacks, so what doctor reports is what
    the daemon would actually do.
    """
    gw = config.get("gateway", {}) or {}

    # Telegram — flat keys. This check is a DELIBERATE MIRROR of the daemon's
    # construction block, not an independent opinion: doctor exists to report
    # what the daemon would actually do, so when that default changes this one
    # changes with it. It read `default True` for the same reason it now reads
    # False — the daemon did.
    #
    # The daemon additionally REFUSES to start an enabled gateway whose
    # allowed_chat_ids is empty, so "enabled" here means enabled AND
    # allowlisted; reporting a gateway as up when the daemon would refuse it
    # is the failure this mirror exists to prevent (CROSS-CUTTING §12 — fix
    # the actor, then fix whatever writes down what it did).
    from prometheus.config.shipped_defaults import (
        resolve_allowed_chat_ids,
        resolve_telegram_enabled,
    )

    tg_token = gw.get("telegram_token", "") or _env_or_env_file(
        "PROMETHEUS_TELEGRAM_TOKEN")
    tg_flag = resolve_telegram_enabled(gw)
    tg_chat_ids = resolve_allowed_chat_ids(gw)
    telegram = _gateway_check(
        "Telegram gateway",
        # `enabled` is the operator's INTENT, deliberately — not "will it come
        # up". Folding the allowlist check in here would report an enabled
        # gateway as merely "not enabled", which is under-reporting: the
        # operator who turned it on and gets nothing needs to be told why, and
        # "not enabled" tells them the opposite of what they did. The
        # allowlist refusal is its own error line below.
        enabled=tg_flag,
        token_ok=bool(tg_token),
        token_missing_msg="enabled but no bot token (gateway.telegram_token "
                          "or PROMETHEUS_TELEGRAM_TOKEN)",
        token_fix="Get a token from @BotFather and add "
                  "PROMETHEUS_TELEGRAM_TOKEN to the env file — or run "
                  "`oara setup --gateway-only`.",
        library_module="telegram",
        library_label="python-telegram-bot",
        library_fix="pip install python-telegram-bot (a core dependency — "
                    "reinstall with `pip install oara-prometheus`).",
    )

    # The allowlist refusal, reported as its own failure rather than folded
    # into `enabled`. The daemon will NOT start an enabled Telegram gateway
    # whose allowed_chat_ids is empty — an empty allowlist used to mean "allow
    # every chat", which exposed an agent with shell access to anyone who
    # found the bot. Doctor exists to report what the daemon would actually
    # do, so it has to name this case; a gateway that is on, tokened, and
    # still refused is exactly the state an operator cannot diagnose alone.
    if tg_flag and tg_token and not tg_chat_ids and telegram.status == "ok":
        telegram = DiagnosticCheck(
            name="Telegram gateway", category="connectivity", status="error",
            message="enabled with a token, but gateway.allowed_chat_ids is "
                    "empty — the daemon refuses to start an unrestricted "
                    "Telegram gateway",
            fix="Add your chat id to gateway.allowed_chat_ids (get it from "
                "@userinfobot), or set gateway.telegram_enabled: false.",
        )

    # Slack — flat keys win over the nested gateway.slack.* form; needs
    # BOTH tokens (bot xoxb-... + app xapp-...).
    slack_nested = gw.get("slack") if isinstance(gw.get("slack"), dict) else {}
    slack_enabled = bool(
        gw.get("slack_enabled", False) or (slack_nested or {}).get("enabled", False)
    )
    slack_bot = (
        gw.get("slack_bot_token", "")
        or (slack_nested or {}).get("bot_token", "")
        or _env_or_env_file("PROMETHEUS_SLACK_BOT_TOKEN")
    )
    slack_app = (
        gw.get("slack_app_token", "")
        or (slack_nested or {}).get("app_token", "")
        or _env_or_env_file("PROMETHEUS_SLACK_APP_TOKEN")
    )
    missing_slack = [
        name for name, val in
        (("bot token (PROMETHEUS_SLACK_BOT_TOKEN)", slack_bot),
         ("app token (PROMETHEUS_SLACK_APP_TOKEN)", slack_app))
        if not val
    ]
    slack = _gateway_check(
        "Slack gateway",
        enabled=slack_enabled,
        token_ok=not missing_slack,
        token_missing_msg="enabled but missing " + " and ".join(
            missing_slack or ["tokens"]),
        token_fix="Slack needs BOTH tokens — create the app at "
                  "https://api.slack.com/apps (Socket Mode), then add "
                  "PROMETHEUS_SLACK_BOT_TOKEN and PROMETHEUS_SLACK_APP_TOKEN "
                  "to the env file.",
        library_module="slack_bolt",
        library_label="slack-bolt",
        library_fix="pip install 'oara-prometheus[slack]'",
    )

    # Discord — nested gateway.discord.* only (the shape the daemon reads).
    discord_nested = gw.get("discord") if isinstance(gw.get("discord"), dict) else {}
    discord_token = (
        (discord_nested or {}).get("token", "")
        or _env_or_env_file("PROMETHEUS_DISCORD_TOKEN")
    )
    discord = _gateway_check(
        "Discord gateway",
        enabled=bool((discord_nested or {}).get("enabled", False)),
        token_ok=bool(discord_token),
        token_missing_msg="enabled but no bot token (gateway.discord.token "
                          "or PROMETHEUS_DISCORD_TOKEN)",
        token_fix="Create a bot at https://discord.com/developers/applications "
                  "(enable the Message Content Intent) and add "
                  "PROMETHEUS_DISCORD_TOKEN to the env file.",
        library_module="discord",
        library_label="discord.py",
        library_fix="pip install 'oara-prometheus[discord]'",
    )
    return [telegram, slack, discord]


def check_advertised_tools(config: dict[str, Any]) -> DiagnosticCheck:
    """How many tools the model is actually OFFERED (FIRSTLIGHT FL-2u / A).

    Registered is not advertised (Standing-Principles §2d): the model
    receives ``schemas_for_run()``, not the registry. FL-2u's fix makes an
    absent ``always_loaded`` fall back to the shipped set, so this row
    should never read zero by accident — but an operator who writes
    ``always_loaded: []`` on purpose gets exactly what they asked for, and
    this is where that stays VISIBLE rather than being discovered as "the
    agent can't do anything".
    """
    try:
        from prometheus.__main__ import create_tool_registry
        from prometheus.context.dynamic_tools import DynamicToolLoader

        deferred = (config.get("tools", {}) or {}).get("deferred_loading")
        registry = create_tool_registry({})
        loader = DynamicToolLoader(registry, deferred)
        advertised = sorted(
            s.get("name") for s in loader.schemas_for_run(True)
        )
        total = len(registry.list_tools())
    except Exception as exc:  # diagnostics must never crash the doctor
        return DiagnosticCheck(
            name="Advertised tools", category="resources", status="warning",
            message=f"could not resolve the advertised set ({exc})",
            fix="Run `oara doctor --debug` and report this — the "
                "advertised set is what the model can actually call.",
        )

    if not advertised:
        return DiagnosticCheck(
            name="Advertised tools", category="resources", status="error",
            message=f"NONE — 0 of {total} registered tools are offered to "
                    f"the model, so it cannot call anything",
            fix="Remove `tools.deferred_loading.always_loaded: []` from your "
                "config to get the shipped set, or list the tools you want "
                "advertised.",
        )
    preview = ", ".join(advertised[:6])
    more = f" (+{len(advertised) - 6} more)" if len(advertised) > 6 else ""
    return DiagnosticCheck(
        name="Advertised tools", category="resources", status="ok",
        message=f"{len(advertised)} of {total} registered offered to the "
                f"model: {preview}{more}",
    )


def check_config_pins() -> DiagnosticCheck:
    """Is anything pinned, and did it override the config file at boot?

    This check exists because the mechanism previously had NO operator
    surface: a pin silently restored `model.model: gemma4-26b` on every boot
    for six weeks while the backend served Qwen, and the only evidence was one
    WARNING in the journal. An operator asking "is anything pinned?" had to
    know the file existed in order to look for it.

    Always `config_pins`, never bare "pins" — Beacon uses "pin" for SKILL pins
    on an adjacent screen.
    """
    from prometheus.config.paths import get_config_dir
    from prometheus.daemon import (
        CONFIG_PIN_EFFECT, CONFIG_PINS_FILENAME, read_config_pins)

    pins_path = get_config_dir() / CONFIG_PINS_FILENAME
    pins = read_config_pins(pins_path)
    if not pins:
        return DiagnosticCheck(
            name="config_pins", category="platform", status="ok",
            message=f"none active (no {pins_path})",
        )

    # Compare against the config ON DISK. A pinned key whose file value
    # differs is the state that silently overrides an operator's edit, and it
    # is the state worth being loud about.
    cfg_path, _ = resolve_config_path()
    on_disk: dict[str, Any] = {}
    if cfg_path and cfg_path.is_file():
        try:
            import yaml
            with cfg_path.open(encoding="utf-8") as fh:
                on_disk = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as exc:
            # doctor's whole job is to report the truth about this file. An
            # unreadable config reported as an empty one makes every pin below
            # look un-overridden.
            log.error(
                "doctor: UNREADABLE — cannot read %s (%s: %s); pin comparison "
                "below treats the on-disk config as EMPTY and is not reliable",
                cfg_path, type(exc).__name__, exc,
            )
            on_disk = {}

    overriding = []
    for dotted, expected in pins.items():
        val: Any = on_disk
        for part in str(dotted).split("."):
            val = val.get(part, {}) if isinstance(val, dict) else None
        if val and str(val) != str(expected):
            overriding.append(f"{dotted}: file={val!r} pinned={expected!r}")

    listing = ", ".join(sorted(pins))
    if overriding:
        return DiagnosticCheck(
            name="config_pins", category="platform", status="warning",
            message=(
                f"{len(pins)} active ({listing}); "
                f"{len(overriding)} overriding the config file — "
                + "; ".join(overriding)
                + f". {CONFIG_PIN_EFFECT}"
            ),
            fix=(
                f"The running value comes from {pins_path}, not the config "
                "file. Edit the pin, or delete it to let the config win."
            ),
        )
    return DiagnosticCheck(
        name="config_pins", category="platform", status="ok",
        message=f"{len(pins)} active ({listing}); config file agrees",
    )


def check_coding_sandbox(config: dict[str, Any]) -> DiagnosticCheck:
    """Which sandbox backend coding runs use, and whether it can start.

    ``create_sandbox()`` RAISES rather than silently degrading when the
    requested backend is unavailable, which is the right behaviour — but it
    raises at the moment a coding run starts, so an operator who selects
    ``bwrap`` or ``docker`` learns it is unusable only once they try to use
    it. Both carry a system dependency nothing else announces: bubblewrap is
    not a Python package, and Docker needs a reachable daemon.

    This reports the backend AND its availability, so the answer arrives at
    ``doctor`` time rather than mid-run.
    """
    coding = config.get("coding", {}) or {}
    backend = str(coding.get("sandbox_type", "process")).strip().lower()

    # Severity deliberately does NOT consult `coding.enabled`. That key is
    # documented in prometheus.yaml.default and defaults to false, but NOTHING
    # READS IT — `run_coding_task()` builds a sandbox regardless, so a coding
    # run can start whatever the flag says. Gating this row on it would report
    # "nothing invokes this yet" about a backend that is one command away from
    # failing, which is the sort of reassuring-but-false line doctor exists to
    # prevent.
    if backend == "process":
        # Available everywhere — but "available" is not "contains". Stating
        # the limit here is the whole point: ProcessSandbox confines the file
        # tools and nothing else, and a one-line shell redirect leaves it.
        return DiagnosticCheck(
            name="Coding sandbox", category="resources", status="ok",
            message="backend 'process' — confines file tools only, "
                    "a shell redirect escapes it",
        )

    detail: str
    available: bool
    if backend == "bwrap":
        try:
            from prometheus.coding.sandbox import BwrapSandbox

            if not BwrapSandbox.is_available():
                available, detail = False, (
                    "the bwrap binary is not on PATH (install bubblewrap)"
                )
            else:
                chk = BwrapSandbox.self_check()
                available = bool(chk.ok)
                detail = "namespace check passed" if chk.ok else chk.detail
        except Exception as exc:
            available, detail = False, f"could not be checked ({exc})"
    elif backend == "docker":
        try:
            from prometheus.coding.sandbox import docker_available

            available = docker_available()
            detail = ("Docker daemon reachable" if available
                      else "no reachable Docker daemon")
        except Exception as exc:
            available, detail = False, f"could not be checked ({exc})"
    else:
        return DiagnosticCheck(
            name="Coding sandbox", category="resources", status="error",
            message=f"unknown backend {backend!r}",
            fix="Set coding.sandbox_type to one of: process, bwrap, docker.",
        )

    if available:
        return DiagnosticCheck(
            name="Coding sandbox", category="resources", status="ok",
            message=f"backend {backend!r} — {detail}",
        )

    return DiagnosticCheck(
        name="Coding sandbox", category="resources", status="error",
        message=f"backend {backend!r} is NOT available — {detail}; "
                f"coding runs will fail to start",
        fix=f"Install/start the {backend} backend, or set "
            f"coding.sandbox_type: process — which starts anywhere but "
            f"confines file tools only.",
    )


def check_bash_floors(config: dict[str, Any]) -> list[DiagnosticCheck]:
    """The two kernel floors around bash: asked for, and actually happening.

    ``check_coding_sandbox`` above reports the sandbox around CODING RUNS.
    Nothing reported the floor around the bash TOOL — the surface the model
    reaches on every turn — so an operator could read a clean doctor run and
    a config saying ``required`` while bash ran with no floor at all.

    Two rows rather than one, because they fail independently and the fixes
    are different: the read floor needs a root-loaded AppArmor profile, the
    write floor needs bubblewrap and a kernel that grants an unprivileged
    namespace.

    ``dark`` is the row that matters. "required but unavailable" announces
    itself — every bash call fails — but the write floor's ``auto`` running
    unconfined is silent by design, and this is where that gets said.
    """
    from prometheus.config.shipped_defaults import resolve_workspace_root
    from prometheus.permissions import confinement as C

    sec = config.get("security", {}) or {}
    # resolve_workspace_root NEVER returns empty — it falls back to the
    # shipped root — so has_workspace is True on every real config and the
    # no-workspace row below is defensive, not expected. Passed anyway so
    # this does not silently become wrong if that resolver ever grows a
    # None path, which is the defect it was written to close.
    roots = resolve_workspace_root(sec)
    rep = C.floor_report(
        read_mode=sec.get("bash_confinement", "off"),
        write_mode=sec.get("bash_write_confinement", "auto"),
        has_workspace=bool(roots),
    )

    def _row(name: str, block: dict, off_message: str, fix_when_missing: str
             ) -> DiagnosticCheck:
        state = block["state"]
        mode = block["mode"]
        detail = block["detail"]
        if state == C.STATE_OFF:
            # Not an error: off is a legitimate, documented choice. It is
            # still worth a line, because "no row" and "no floor" look
            # identical on a report and only one of them is true.
            return DiagnosticCheck(
                name=name, category="platform", status="info",
                message=f"mode 'off' — {off_message}",
            )
        if state == C.STATE_NO_WORKSPACE:
            return DiagnosticCheck(
                name=name, category="platform", status="warning",
                message=f"mode {mode!r} but no workspace root is configured — "
                        "there is no boundary to enforce",
                fix="Set security.workspace_root, or set the floor to 'off' "
                    "so the config stops implying a boundary that is absent.",
            )
        if state == C.STATE_ACTIVE:
            return DiagnosticCheck(
                name=name, category="platform", status="ok",
                message=f"mode {mode!r} — ACTIVE ({detail})",
            )
        if state == C.STATE_REFUSING:
            return DiagnosticCheck(
                name=name, category="platform", status="error",
                message=f"mode 'required' but UNAVAILABLE — {detail}. "
                        "Every bash call is being refused.",
                fix=fix_when_missing,
            )
        if state == C.STATE_DARK:
            # WARNING, not error, and the distinction is the mode's own
            # promise rather than the outcome. `auto` is documented as
            # "confine where the mechanism exists, run where it does not" —
            # a host that cannot provide it is that mode working as written,
            # not a broken install. `required` is the mode that promises
            # enforcement, and it fails loudly above.
            #
            # Erroring here made `doctor` exit nonzero on every host without
            # bubblewrap — macOS has no equivalent at all, and CI caught it
            # on a config that had opted into nothing: the SHIPPED default
            # asked for the floor, not the operator. A permanent red row
            # nobody can clear is how a report gets ignored, which costs
            # more than this row is worth.
            return DiagnosticCheck(
                name=name, category="platform", status="warning",
                message=f"mode {mode!r} but UNAVAILABLE — {detail}. "
                        "bash is running WITHOUT this floor.",
                fix=fix_when_missing,
            )
        return DiagnosticCheck(
            name=name, category="platform", status="warning",
            message=f"mode {mode!r} — state could not be determined ({detail})",
        )

    read_row = _row(
        "Bash read floor", rep["bash_read_floor"],
        "bash may read ~/.ssh, ~/.gnupg and credential env files; the "
        "denied_paths list covers the path-declaring tools only",
        f"Load the profile: sudo apparmor_parser -r -W "
        f"/etc/apparmor.d/{C.PROFILE} — or set security.bash_confinement: "
        f"off to run without it knowingly.",
    )
    write_row = _row(
        "Bash write floor", rep["bash_write_floor"],
        "bash may write anywhere on the filesystem with no approval — the "
        "outside-workspace check reads a tool's file_path, and bash has none",
        "Install bubblewrap (sudo apt install bubblewrap) — or set "
        "security.bash_write_confinement: off to run without it knowingly.",
    )

    rows = [read_row, write_row]
    # Said once, on its own row, because it is the limit most likely to be
    # over-read from two green lines above: cron jobs, background tasks,
    # watch_dir predicates and command hooks each spawn /bin/bash at their
    # own call sites, and NEITHER floor reaches them.
    if any(r.status == "ok" for r in rows):
        rows.append(DiagnosticCheck(
            name="Bash floor scope", category="platform", status="info",
            message="the floors wrap the bash TOOL only — cron jobs, "
                    "background tasks, watch_dir predicates and command "
                    "hooks spawn their own shells and are not covered",
        ))
    return rows


def check_trajectory_export(config: dict[str, Any]) -> DiagnosticCheck:
    """Is golden-trace capture actually accumulating anything?

    A subsystem that is off ON PURPOSE and one that is off BY ACCIDENT look
    identical from outside, and the only thing distinguishing them is a
    comment in a yaml file that nobody greps. This is the row that tells
    them apart — and it does so by reporting the CONSEQUENCE rather than the
    flag, because the flag alone cannot: "disabled" next to 0 stranded
    traces is a deliberate choice, "disabled" next to 1400 is a leak.

    Silence here is the failure mode worth catching. Nothing breaks when
    export is off; the traces simply never accumulate, and the cost is only
    visible much later as an absent training corpus.
    """
    cfg = config.get("trajectory_export", {}) or {}
    enabled = bool(cfg.get("enabled", True))
    out_dir = Path(
        str(cfg.get("output_dir", "~/.prometheus/trajectories/"))
    ).expanduser()

    # How many golden rows sit past the export cursor. Best-effort: this is
    # a diagnostic, so a missing DB reports "unknown" rather than failing.
    stranded: int | None = None
    import sqlite3  # module-scope so the handler below can name sqlite3.Error

    try:
        from prometheus.sentinel.golden_trace_exporter import WATERMARK_FILENAME

        watermark = 0
        wm_path = out_dir / WATERMARK_FILENAME
        if wm_path.exists():
            watermark = int(json.loads(wm_path.read_text())["last_rowid"])

        db = Path("~/.prometheus/telemetry.db").expanduser()
        if db.exists():
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                stranded = conn.execute(
                    "SELECT COUNT(*) FROM tool_calls"
                    " WHERE is_golden = 1 AND rowid > ?",
                    (watermark,),
                ).fetchone()[0]
            finally:
                conn.close()
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        # `None` renders as "unknown" downstream, which is already the honest
        # answer — but WHY it is unknown was unreadable. Narrowed and named:
        # an unexpected exception type now propagates instead of quietly
        # becoming a diagnostic that says "unknown" forever.
        log.warning(
            "doctor: stranded-trace count unavailable (%s: %s); reporting "
            "unknown", type(exc).__name__, exc,
        )
        stranded = None

    if not enabled:
        waiting = (
            "an unknown number of" if stranded is None else f"{stranded}"
        )
        # Deliberate and harmless vs deliberate and forgotten differ only by
        # this count, so it leads the message.
        status = "ok" if stranded == 0 else "warning"
        return DiagnosticCheck(
            name="Trajectory export", category="resources", status=status,
            message=f"DISABLED — {waiting} golden trace(s) recorded and not "
                    f"being exported",
            fix="Set `trajectory_export.enabled: true` to resume writing "
                "fine-tuning JSONL. Ignore this row if you do not intend to "
                "fine-tune — the traces stay in telemetry.db either way.",
        )

    files = len(list(out_dir.glob("*.jsonl"))) if out_dir.is_dir() else 0
    return DiagnosticCheck(
        name="Trajectory export", category="resources", status="ok",
        message=f"enabled — {files} export file(s) in {out_dir}, "
                f"{'unknown' if stranded is None else stranded} trace(s) "
                f"awaiting the next cycle",
    )


def check_whisper(config: dict[str, Any]) -> DiagnosticCheck:
    """Whisper available when voice is enabled?"""
    whisper_cfg = config.get("whisper", {}) or {}
    voice_cfg = (config.get("gateway", {}) or {}).get("voice", {}) or {}
    voice_enabled = bool(whisper_cfg.get("enabled")) or bool(voice_cfg.get("enabled"))
    if not voice_enabled:
        return DiagnosticCheck(
            name="Whisper STT", category="resources", status="info",
            message="voice disabled — check skipped",
        )
    from prometheus.tools.builtin.whisper_stt import _detect_whisper_engine
    engine = _detect_whisper_engine()
    if engine:
        return DiagnosticCheck(
            name="Whisper STT", category="resources", status="ok",
            message=f"engine available: {engine}",
        )
    return DiagnosticCheck(
        name="Whisper STT", category="resources", status="error",
        message="voice is enabled but no Whisper engine was found",
        fix="Install one: `pip install 'oara-prometheus[voice]'` "
            "(faster-whisper) or the `whisper` CLI.",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_extended_checks(
    config: dict[str, Any],
    *,
    config_check: DiagnosticCheck,
    timeout: float = 5.0,
) -> list[DiagnosticCheck]:
    """The Phase 0 onboarding checks (config check computed by the caller)."""
    reach, model = check_inference(config, timeout=timeout)
    return [
        config_check,
        check_dirs_writable(),
        reach,
        model,
        check_web_port(config),
        check_token(config),
        *check_gateways(config),
        check_advertised_tools(config),
        check_coding_sandbox(config),
        *check_bash_floors(config),
        check_config_pins(),
        check_trajectory_export(config),
        check_whisper(config),
    ]


def run_anatomy_checks(config: dict[str, Any]) -> list[DiagnosticCheck]:
    """The pre-existing Doctor class checks (anatomy scan required).

    Best-effort: any failure here degrades to a single info line rather
    than masking the extended checks above.
    """
    try:
        from prometheus.infra.anatomy import AnatomyScanner
        from prometheus.infra.doctor import Doctor

        model_cfg = config.get("model", {}) or {}
        anatomy_cfg = config.get("anatomy", {}) or {}
        scanner = AnatomyScanner(
            llama_cpp_url=model_cfg.get("base_url", "http://localhost:8080"),
            ollama_url=model_cfg.get("fallback_url", "http://localhost:11434"),
            inference_engine=model_cfg.get("provider", "llama_cpp"),
            ssh_user=anatomy_cfg.get("ssh_user"),
            ssh_key=anatomy_cfg.get("ssh_key"),
        )

        async def _scan_and_diagnose():
            state = await scanner.scan()
            return await Doctor(config).diagnose(state)

        report = asyncio.run(_scan_and_diagnose())
        checks: list[DiagnosticCheck] = []
        for check in report.checks:
            # The extended checks above already cover these with the
            # documented search order / live probe — don't double-report
            # (the class's Config check is also repo-root-relative, which
            # is wrong for pip installs). "Telegram" is superseded by the
            # per-gateway checks (SPRINT G3).
            if check.name in ("Config", "Inference", "Model", "Whisper STT",
                              "Telegram"):
                continue
            # Missing SOUL.md/AGENTS.md doesn't break the loop — the fast
            # setup path deliberately skips identity. Warn, don't fail.
            if check.name == "Bootstrap" and check.status == "error":
                check = DiagnosticCheck(
                    name=check.name, category=check.category, status="warning",
                    message=check.message,
                    fix="Run `oara setup` (the rich wizard) to generate "
                        "identity files. Until then the agent runs with a "
                        "generic identity (SOUL.md/AGENTS.md shape the system "
                        "prompt only — tools and the loop are unaffected).",
                )
            checks.append(check)
        return checks
    except Exception as exc:
        return [DiagnosticCheck(
            name="Anatomy scan", category="platform", status="info",
            message=f"deep scan unavailable ({exc}) — extended checks above still apply",
        )]


def render_report(checks: list[DiagnosticCheck]) -> str:
    """Human-readable ✓/✗ report grouped like the Telegram /doctor output."""
    lines = ["oara doctor", ""]
    from prometheus.infra.doctor import DiagnosticReport
    for cat in DiagnosticReport.CATEGORY_ORDER:
        cat_checks = [c for c in checks if c.category == cat]
        if not cat_checks:
            continue
        lines.append(f"{DiagnosticReport.CATEGORY_LABELS.get(cat, cat)}:")
        for c in cat_checks:
            lines.append(f"  {_SYMBOLS.get(c.status, '?')} {c.name}: {c.message}")
            if c.fix and c.status in ("warning", "error"):
                lines.append(f"      fix: {c.fix.strip().splitlines()[0]}")
        lines.append("")
    errors = sum(1 for c in checks if c.status == "error")
    warnings = sum(1 for c in checks if c.status == "warning")
    if errors:
        lines.append(f"RESULT: {errors} error(s), {warnings} warning(s)")
    elif warnings:
        lines.append(f"RESULT: OK with {warnings} warning(s)")
    else:
        lines.append("RESULT: all checks passed")
    return "\n".join(lines)


def run_doctor_command(args: argparse.Namespace) -> int:
    """Entry point for ``oara doctor``. Exit 1 on any error check."""
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)  # probe noise

    config_check, config = check_config(getattr(args, "config", None))
    checks = run_extended_checks(
        config, config_check=config_check,
        timeout=float(getattr(args, "timeout", 5.0)),
    )
    if not getattr(args, "no_scan", False) and config:
        checks.extend(run_anatomy_checks(config))
    print(render_report(checks))
    return 1 if any(c.status == "error" for c in checks) else 0


def add_doctor_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``doctor`` subcommand."""
    p = subparsers.add_parser(
        "doctor", help="Diagnose the install: config, server, model, web, token, dirs",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to prometheus.yaml (default: standard search order)",
    )
    p.add_argument(
        "--no-scan", action="store_true",
        help="Skip the deep anatomy scan (faster; extended checks only)",
    )
    p.add_argument(
        "--timeout", type=float, default=5.0,
        help="Probe timeout in seconds (default 5)",
    )
