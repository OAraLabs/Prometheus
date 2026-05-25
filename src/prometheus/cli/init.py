"""``prometheus init`` — interactive setup for new installs.

The Hermes ``pip install hermes-agent && hermes init`` story is the
reference. We mirror it here: detect the user's local inference
servers (llama.cpp, Ollama, LM Studio, vLLM), ask three questions, and
write a working ``prometheus.yaml`` + env template that gets the user
to ``prometheus daemon`` in under five minutes.

This is *additive* — it doesn't replace the existing :mod:`setup_wizard`
which has a much richer surface (Telegram, Slack, profiles, identity).
``init`` is the fast path for first-time users.

Source: Novel code for Prometheus Polish & Platform sprint, WS4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prometheus.config.paths import get_config_dir

# ---------------------------------------------------------------------------
# Local inference-server detection
# ---------------------------------------------------------------------------
#
# Hermes GitHub issue #523 asked for "auto-detect Ollama / llama.cpp".
# Prometheus' Adapter Layer can offer per-engine strictness recommendations,
# which makes this work especially useful here.

KNOWN_LOCAL_SERVERS: list[dict[str, str]] = [
    {
        "name": "llama.cpp",
        "url": "http://localhost:8080",
        "models_path": "/v1/models",
        "provider": "llama_cpp",
    },
    {
        "name": "Ollama",
        "url": "http://localhost:11434",
        "models_path": "/api/tags",
        "provider": "ollama",
    },
    {
        "name": "LM Studio",
        "url": "http://localhost:1234",
        "models_path": "/v1/models",
        "provider": "lm_studio",
    },
    {
        "name": "vLLM",
        "url": "http://localhost:8000",
        "models_path": "/v1/models",
        "provider": "vllm",
    },
]


@dataclass(frozen=True)
class DetectedServer:
    """A reachable local inference endpoint."""

    name: str
    url: str
    provider: str
    models: list[str] = field(default_factory=list)
    response_ms: float = 0.0


def detect_local_servers(
    timeout: float = 1.0,
    candidates: list[dict[str, str]] | None = None,
) -> list[DetectedServer]:
    """Probe each known local server; return those that respond.

    Pure stdlib (urllib) so detection works even when ``httpx`` is not
    installed — :command:`prometheus init` should never error on a
    minimal install.

    The model-list parse is best-effort: Ollama uses ``{"models": [...]}``
    while OpenAI-compat servers use ``{"data": [{"id": ...}, ...]}``.
    Both shapes are handled; unknown shapes leave ``models`` empty.
    """
    import urllib.error
    import urllib.request

    found: list[DetectedServer] = []
    for spec in (candidates or KNOWN_LOCAL_SERVERS):
        url = spec["url"] + spec["models_path"]
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            continue
        except Exception:
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Parse model list from whichever JSON shape this server returns.
        models: list[str] = []
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                if isinstance(data.get("models"), list):  # Ollama
                    models = [
                        str(m.get("name") or m.get("model") or m)
                        for m in data["models"][:20]
                    ]
                elif isinstance(data.get("data"), list):  # OpenAI-compat
                    models = [
                        str(m.get("id") or m)
                        for m in data["data"][:20]
                    ]
        except (json.JSONDecodeError, TypeError):
            pass

        found.append(DetectedServer(
            name=spec["name"],
            url=spec["url"],
            provider=spec["provider"],
            models=models,
            response_ms=elapsed_ms,
        ))
    return found


# ---------------------------------------------------------------------------
# Config writing
# ---------------------------------------------------------------------------


def _default_config(server: DetectedServer | None, model: str | None) -> dict[str, Any]:
    """Build a minimal but functional prometheus.yaml."""
    return {
        "system": {
            "name": "Prometheus",
            "version": "0.1.0",
        },
        "model": {
            "provider": server.provider if server else "llama_cpp",
            "base_url": server.url if server else "http://localhost:8080",
            "model": model or (server.models[0] if (server and server.models) else "auto"),
            "grammar_enforcement": True,
            "max_tool_iterations": 25,
        },
        "context": {
            "effective_limit": 24000,
            "compression_trigger": 0.75,
            "reserved_output": 2000,
        },
        "security": {
            "permission_mode": "default",
            "workspace_root": "~",
        },
        "gateway": {
            "telegram_enabled": False,
            "telegram_token": "",
            "allowed_chat_ids": [],
            "cron_enabled": True,
            "slack": {
                "enabled": False,
                "bot_token": "",
                "app_token": "",
                "allowed_channels": [],
            },
        },
        "web": {
            "enabled": True,
            "api_port": 8005,
            "ws_port": 8010,
        },
        "learning": {
            "curator_enabled": True,
        },
    }


_ENV_TEMPLATE = """# Prometheus environment overrides
# Set any of these and they override the matching yaml field.
#
# Telegram (run prometheus init again or edit prometheus.yaml to enable):
#   PROMETHEUS_TELEGRAM_TOKEN=
#   PROMETHEUS_TELEGRAM_CHAT_IDS=
#
# Slack:
#   PROMETHEUS_SLACK_BOT_TOKEN=
#   PROMETHEUS_SLACK_APP_TOKEN=
#
# Cloud providers (optional):
#   ANTHROPIC_API_KEY=
#   OPENAI_API_KEY=
#   GOOGLE_API_KEY=
#
# Beacon dashboard API token (recommended if exposing beyond localhost):
#   PROMETHEUS_API_TOKEN=
"""


def write_config(
    config: dict[str, Any],
    target_path: Path,
    *,
    backup_existing: bool = True,
) -> Path | None:
    """Write *config* to *target_path*. Returns the backup path, if any."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if target_path.exists() and backup_existing:
        backup_path = target_path.with_suffix(
            target_path.suffix + f".backup-{int(time.time())}"
        )
        shutil.copy2(target_path, backup_path)
    with target_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)
    return backup_path


def write_env_template(target_path: Path) -> bool:
    """Write the env template only if it doesn't already exist. Returns True if written."""
    if target_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------


def _prompt(label: str, default: str = "") -> str:
    """Read a line of input with a default. Returns the trimmed value."""
    if default:
        full = f"{label} [{default}]: "
    else:
        full = f"{label}: "
    try:
        raw = input(full).strip()
    except EOFError:
        return default
    return raw or default


def _prompt_choice(label: str, options: list[str], default_index: int = 0) -> int:
    """Ask the user to pick one of *options*; return the chosen index."""
    print(f"\n{label}")
    for i, opt in enumerate(options):
        marker = "*" if i == default_index else " "
        print(f"  {marker} {i + 1}. {opt}")
    while True:
        raw = _prompt("Pick a number", str(default_index + 1))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  Please enter a valid number.")


def _print_detection_summary(servers: list[DetectedServer]) -> None:
    print()
    if not servers:
        print("Local inference: no servers detected on standard ports.")
        print("  Checked llama.cpp:8080, Ollama:11434, LM Studio:1234, vLLM:8000.")
        print("  You can still continue and add a cloud provider, or start a")
        print("  local server and rerun `prometheus init`.")
        return
    print(f"Local inference: {len(servers)} server(s) detected:")
    for s in servers:
        model_info = (
            f" ({len(s.models)} models, first: {s.models[0]})"
            if s.models else " (model list unavailable)"
        )
        print(f"  • {s.name:<10} @ {s.url:<28} {s.response_ms:.0f}ms{model_info}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def run_init(
    *,
    noninteractive: bool = False,
    target_dir: Path | None = None,
    timeout: float = 1.0,
) -> dict[str, Any]:
    """Run the init flow. Returns the config that was written."""
    cfg_dir = target_dir or get_config_dir()
    cfg_path = cfg_dir / "prometheus.yaml"
    env_path = cfg_dir / "env"

    print("┌─ Prometheus init ─────────────────────────────────────────────┐")
    print(f"  Config will be written to {cfg_path}")
    print("└───────────────────────────────────────────────────────────────┘")

    print("\nProbing for local inference servers …")
    servers = detect_local_servers(timeout=timeout)
    _print_detection_summary(servers)

    # Pick a model provider
    chosen_server: DetectedServer | None = None
    chosen_model: str | None = None
    if servers:
        if noninteractive:
            chosen_server = servers[0]
            chosen_model = chosen_server.models[0] if chosen_server.models else None
        else:
            labels = [f"{s.name} @ {s.url}" for s in servers]
            labels.append("None of these — I'll use a cloud provider")
            pick = _prompt_choice(
                "Which inference server should Prometheus use?",
                labels,
                default_index=0,
            )
            if pick < len(servers):
                chosen_server = servers[pick]
                # Model picker for that server
                if chosen_server.models:
                    if len(chosen_server.models) == 1 or noninteractive:
                        chosen_model = chosen_server.models[0]
                    else:
                        midx = _prompt_choice(
                            "Which model?", chosen_server.models, default_index=0,
                        )
                        chosen_model = chosen_server.models[midx]

    # Gateway choice (don't require Telegram up-front; user can enable later)
    if noninteractive:
        gateway_choice = "cli"
    else:
        gw_idx = _prompt_choice(
            "Which gateway should Prometheus respond on?",
            [
                "CLI only (run `prometheus` for an interactive shell)",
                "Telegram (you'll add a bot token via env later)",
                "Slack (you'll add bot+app tokens via env later)",
                "Both Telegram and Slack",
            ],
            default_index=0,
        )
        gateway_choice = ["cli", "telegram", "slack", "both"][gw_idx]

    # Build config
    config = _default_config(chosen_server, chosen_model)
    if gateway_choice in ("telegram", "both"):
        config["gateway"]["telegram_enabled"] = True
    if gateway_choice in ("slack", "both"):
        config["gateway"]["slack"]["enabled"] = True

    # Backup existing config if present
    backup = write_config(config, cfg_path, backup_existing=True)
    if backup is not None:
        print(f"\nExisting config backed up to {backup}")

    # Env template
    wrote_env = write_env_template(env_path)
    if wrote_env:
        print(f"Env template written to {env_path}")
    else:
        print(f"Env file already present at {env_path} — left untouched")

    # Friendly next-steps
    print()
    print("Setup complete. Next steps:")
    if gateway_choice == "cli":
        print("  1. Run: prometheus")
        print(
            "  2. Beacon dashboard available at http://localhost:8005 "
            "once daemon is up"
        )
    else:
        print(f"  1. Edit {env_path} to add gateway tokens")
        print("  2. Run: prometheus daemon")
        print(
            "  3. Beacon dashboard available at http://localhost:8005 "
            "once daemon is up"
        )
    print()
    return config


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``prometheus-init`` console-script entry point."""
    parser = argparse.ArgumentParser(
        prog="prometheus-init",
        description=(
            "Interactive first-run setup for Prometheus. Detects local "
            "inference servers and writes a working prometheus.yaml."
        ),
    )
    parser.add_argument(
        "--noninteractive",
        action="store_true",
        help="Skip prompts; pick the first detected server and CLI gateway.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Override the config directory (defaults to the standard location).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Per-server probe timeout in seconds (default 1.0).",
    )
    args = parser.parse_args(argv)

    try:
        run_init(
            noninteractive=args.noninteractive,
            target_dir=args.target_dir,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("\nInit aborted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
