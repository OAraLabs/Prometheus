"""``prometheus doctor`` — first-line diagnostics for a Prometheus install.

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
check errors — "run prometheus doctor" is the eternal support answer.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

from prometheus.config.api_token import resolve_api_token
from prometheus.config.paths import (
    get_config_dir,
    get_data_dir,
    get_logs_dir,
    get_workspace_dir,
)
from prometheus.infra.doctor import DiagnosticCheck

_SYMBOLS = {"ok": "✓", "error": "✗", "warning": "!", "info": "·"}


# ---------------------------------------------------------------------------
# Config resolution (mirrors prometheus.__main__.load_config search order)
# ---------------------------------------------------------------------------

def resolve_config_path(explicit: str | None = None) -> tuple[Path | None, list[Path]]:
    """Return ``(found_path_or_None, searched_paths)``.

    Search order (documented in the README and prometheus.yaml.default):
    1. an explicit ``--config`` path
    2. the repo-local ``config/prometheus.yaml`` (checkout installs)
    3. ``$PROMETHEUS_CONFIG_DIR/prometheus.yaml`` (default ``~/.prometheus/``)
    """
    searched: list[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        return (p if p.is_file() else None), [p]
    repo_cfg = Path(__file__).resolve().parents[3] / "config" / "prometheus.yaml"
    searched.append(repo_cfg)
    if repo_cfg.is_file():
        return repo_cfg, searched
    user_cfg = get_config_dir() / "prometheus.yaml"
    searched.append(user_cfg)
    if user_cfg.is_file():
        return user_cfg, searched
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
            fix="Run `prometheus setup` to create one.",
        ), {}
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return DiagnosticCheck(
            name="Config", category="platform", status="error",
            message=f"{path} has a YAML parse error: {exc}",
            fix=f"Fix the syntax in {path}, or re-run `prometheus setup`.",
        ), {}
    if not isinstance(config, dict):
        return DiagnosticCheck(
            name="Config", category="platform", status="error",
            message=f"{path} does not contain a YAML mapping",
            fix="Re-run `prometheus setup` to regenerate it.",
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
                "then re-run `prometheus doctor`.",
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
        fix="Run `prometheus token rotate` (the daemon also mints one "
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
            # is wrong for pip installs).
            if check.name in ("Config", "Inference", "Model", "Whisper STT"):
                continue
            # Missing SOUL.md/AGENTS.md doesn't break the loop — the fast
            # setup path deliberately skips identity. Warn, don't fail.
            if check.name == "Bootstrap" and check.status == "error":
                check = DiagnosticCheck(
                    name=check.name, category=check.category, status="warning",
                    message=check.message,
                    fix="Run `prometheus setup` (the rich wizard) to generate "
                        "identity files.",
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
    lines = ["prometheus doctor", ""]
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
    """Entry point for ``prometheus doctor``. Exit 1 on any error check."""
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
