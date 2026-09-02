"""The FAST setup path — probe, three questions, working config.

Detect the user's local inference servers (llama.cpp, Ollama, LM
Studio, vLLM), ask three questions, and write a working
``prometheus.yaml`` + env file that gets the user to
``oara daemon`` in under five minutes.

Onboarding Phase 0: this is no longer a competing wizard. The ONE
canonical entry point is ``oara setup`` (see
:mod:`prometheus.cli.setup`); this module implements its ``--fast`` /
``--noninteractive`` path. The ``prometheus-init`` console script is
kept as a thin forwarding alias.

Dead-end rule: **no path in here may write a config that is known
broken.** When nothing is detected we offer a remote URL, a cloud
provider (key written to the env file), or copy-paste install
instructions and a clean exit — never a default config pointing at a
server that isn't there.

Source: Novel code for Prometheus Polish & Platform sprint, WS4;
reworked in the onboarding overhaul (Phase 0).
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
from prometheus.config.shipped_defaults import (
    SHIPPED_ALWAYS_LOADED,
    SHIPPED_MAX_TOOL_ITERATIONS,
)

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

    The model-list parse doubles as the liveness check: a candidate only
    counts as detected when its models endpoint returns JSON in a known
    shape — Ollama's ``{"models": [...]}`` or the OpenAI-compat
    ``{"data": [...]}``. Anything else (an HTML dashboard squatting on
    :8080, a proxy error page) is skipped: writing a config pointing at
    a non-inference service is exactly the known-broken dead end the
    onboarding overhaul forbids.
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
        models: list[str] | None = None
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

        if models is None:
            # Responded, but not like an inference server — skip it.
            continue

        found.append(DetectedServer(
            name=spec["name"],
            url=spec["url"],
            provider=spec["provider"],
            models=models,
            response_ms=elapsed_ms,
        ))
    return found


def remote_server_candidates(url: str) -> list[dict[str, str]]:
    """Probe specs for ONE user-supplied base URL (both known API shapes).

    Shared by the interactive remote prompt below and the setup-mode
    ``GET /api/setup/detect?base_url=`` endpoint (Onboarding Phase 2) —
    one definition of "what counts as an inference server at this URL".
    """
    url = url.rstrip("/")
    return [
        {"name": "remote (OpenAI-compatible)", "url": url,
         "models_path": "/v1/models", "provider": "llama_cpp"},
        {"name": "remote (Ollama)", "url": url,
         "models_path": "/api/tags", "provider": "ollama"},
    ]


def probe_backend(
    provider: str, base_url: str, timeout: float = 3.0,
) -> DetectedServer | None:
    """Re-probe ONE chosen backend; ``None`` when unreachable/not-inference.

    The validation step behind ``POST /api/setup/configure`` — the same
    JSON-shape hardening as :func:`detect_local_servers` (an HTML page on
    the port does not count), aimed at exactly one provider+URL.
    """
    models_path = "/api/tags" if provider == "ollama" else "/v1/models"
    found = detect_local_servers(timeout=timeout, candidates=[{
        "name": provider, "url": base_url.rstrip("/"),
        "models_path": models_path, "provider": provider,
    }])
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Config writing
# ---------------------------------------------------------------------------


# SHIPPED_ALWAYS_LOADED (imported at the top of this module) is the tool
# set a fresh install advertises. It lives in
# prometheus.config.shipped_defaults so the READER (DynamicToolLoader) and
# this WRITER share ONE value: FL-2 made setup write it; FL-2u made the
# reader fall back to it, so an upgraded install whose config predates the
# key advertises correctly with no config migration.


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
            "max_tool_iterations": SHIPPED_MAX_TOOL_ITERATIONS,
        },
        "context": {
            "effective_limit": 24000,
            "compression_trigger": 0.75,
            "reserved_output": 2000,
        },
        # Only the two keys DynamicToolLoader itself reads (enabled,
        # always_loaded). The template also carries mcp_always_deferred /
        # search_mcp — read by mcp.bootstrap since 2026-08-28, but they
        # only matter once mcp_servers is populated, which init never does.
        "tools": {
            "deferred_loading": {
                "enabled": "auto",
                "always_loaded": list(SHIPPED_ALWAYS_LOADED),
            },
        },
        "security": {
            "permission_mode": "default",
            "workspace_root": "~/.prometheus/workspace",
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
            # SPRINT G3: discord is first-class in every written config —
            # exactly the nested shape the daemon's construction block reads.
            "discord": {
                "enabled": False,
                "token": "",
                "guild_ids": [],
                "channel_ids": [],
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


_ENV_TEMPLATE = """# Prometheus environment file — secrets live here, not in prometheus.yaml.
# Loaded by `oara daemon` at startup AND by the systemd unit
# (EnvironmentFile=). Uncomment and fill in what you need.
# Guided gateway setup: `oara setup --gateway-only`.
#
# Telegram — get a token by messaging @BotFather (/newbot); also set
# gateway.telegram_enabled: true in prometheus.yaml:
#   PROMETHEUS_TELEGRAM_TOKEN=
#   PROMETHEUS_TELEGRAM_CHAT_IDS=
#
# Slack — create an app at https://api.slack.com/apps, enable Socket Mode,
# install it; needs BOTH tokens (bot xoxb-... + app xapp-...) and
# gateway.slack.enabled: true in prometheus.yaml:
#   PROMETHEUS_SLACK_BOT_TOKEN=
#   PROMETHEUS_SLACK_APP_TOKEN=
#
# Discord — create an app at https://discord.com/developers/applications,
# add a Bot + enable its Message Content Intent, copy the bot token; also
# set gateway.discord.enabled: true in prometheus.yaml:
#   PROMETHEUS_DISCORD_TOKEN=
#
# Cloud providers (optional):
#   ANTHROPIC_API_KEY=
#   OPENAI_API_KEY=
#   GOOGLE_API_KEY=
#   DEEPSEEK_API_KEY=      # platform.deepseek.com — /deepseek override
#   MOONSHOT_API_KEY=      # platform.moonshot.ai — /kimi override
#   ZAI_API_KEY=           # z.ai API console — /glm override
#   MIMO_API_KEY=          # api.xiaomimimo.com console — /mimo override
#   QWEN_API_KEY=          # Alibaba Model Studio console — /qwen override
#
# Per-provider endpoint overrides (optional). Each cloud provider reads
# <PROVIDER>_BASE_URL — OPENAI_/GEMINI_/XAI_/ANTHROPIC_/DEEPSEEK_/MOONSHOT_/
# ZAI_/MIMO_/QWEN_BASE_URL. Set one when your key is scoped to a private or
# regional endpoint rather than the provider's shared public host. They live
# here rather than in prometheus.yaml because such a hostname often identifies
# the account (Alibaba workspace endpoints embed the workspace id).
#   QWEN_BASE_URL=         # e.g. https://ws-<id>.<region>.maas.aliyuncs.com/compatible-mode/v1
#
# Media generation (optional, paid — never auto-selected):
#   DASHSCOPE_API_KEY=     # Alibaba DashScope — image_generate backend=dashscope (WAN 2.5)
#   KLING_ACCESS_KEY=      # klingai.com API console — video_generate (Kling 3.0)
#   KLING_SECRET_KEY=
#
# Beacon dashboard API token. Leave UNSET and the daemon mints a secure
# one on first start (printed once; `oara token show` re-prints it).
# An explicit empty value (PROMETHEUS_API_TOKEN=) means deliberately open.
#   PROMETHEUS_API_TOKEN=
"""

# Cloud fast-path choices: provider → (api_key_env, default model, context limit)
# Appended entries keep the historical anthropic/openai numbering stable.
_CLOUD_FAST_PROVIDERS: dict[str, tuple[str, str, int]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-4-6", 100000),
    "openai": ("OPENAI_API_KEY", "gpt-4o", 64000),
    # CLOUD EXPANSION (2026-07)
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-v4-flash", 64000),
    "kimi": ("MOONSHOT_API_KEY", "kimi-k2.6", 64000),
    "glm": ("ZAI_API_KEY", "glm-5.2", 64000),
    "mimo": ("MIMO_API_KEY", "mimo-v2.5-pro", 64000),
    "qwen": ("QWEN_API_KEY", "qwen3.7-max", 64000),
}


def _cloud_default_config(provider: str, api_key_env: str, model: str) -> dict[str, Any]:
    """A valid cloud-provider config (no base_url — the provider knows)."""
    config = _default_config(None, model)
    config["model"] = {
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "max_tool_iterations": SHIPPED_MAX_TOOL_ITERATIONS,
    }
    config["context"]["effective_limit"] = _CLOUD_FAST_PROVIDERS[provider][2]
    return config


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
        print("  Already running a server on another port or machine?")
        print("    oara setup --probe-url http://host:port")
        return
    print(f"Local inference: {len(servers)} server(s) detected:")
    for s in servers:
        model_info = (
            f" ({len(s.models)} models, first: {s.models[0]})"
            if s.models else " (model list unavailable)"
        )
        print(f"  • {s.name:<10} @ {s.url:<28} {s.response_ms:.0f}ms{model_info}")


# ---------------------------------------------------------------------------
# Dead-end handling — no server detected
# ---------------------------------------------------------------------------


OLLAMA_INSTALL_INSTRUCTIONS = """\
To run a local model, install one of:

  Ollama (easiest):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3:8b        # or any model that fits your hardware

  llama.cpp (fastest):
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp && cmake -B build && cmake --build build -j
    ./build/bin/llama-server -m models/your-model.gguf -c 32768 --port 8080

Then run `oara setup` again — it will detect the server."""


def _print_install_instructions() -> None:
    print()
    print(OLLAMA_INSTALL_INSTRUCTIONS)
    print()
    print("No config was written — nothing usable was detected, and Prometheus")
    print("refuses to write a config that is known to be broken.")


def _prompt_remote_server(timeout: float) -> DetectedServer | None:
    """Ask for a remote base URL and probe it. None if nothing reachable."""
    while True:
        url = _prompt(
            "Base URL of your inference server (e.g. http://gpu-box:8080)"
        ).rstrip("/")
        if not url:
            return None
        found = detect_local_servers(
            timeout=max(timeout, 3.0), candidates=remote_server_candidates(url),
        )
        if found:
            s = found[0]
            print(f"  + Reachable: {s.name} @ {s.url}"
                  + (f" ({len(s.models)} models)" if s.models else ""))
            return s
        print(f"  x {url} did not respond on /v1/models or /api/tags.")
        retry = _prompt("Try a different URL? [y/N]", "N")
        if retry.lower() != "y":
            return None


def _prompt_cloud_provider(env_path: Path) -> dict[str, Any] | None:
    """Cloud fast path: pick anthropic/openai, get a key INTO the env file.

    Returns a complete, valid config dict — or None when no usable key
    could be established (in which case nothing is written).
    """
    import os

    providers = list(_CLOUD_FAST_PROVIDERS)
    idx = _prompt_choice(
        "Which cloud provider?",
        [f"{p} (key: ${_CLOUD_FAST_PROVIDERS[p][0]})" for p in providers],
        default_index=0,
    )
    provider = providers[idx]
    key_env, default_model, _limit = _CLOUD_FAST_PROVIDERS[provider]

    key = _prompt(
        f"Paste your {provider} API key (Enter to use ${key_env} from your environment)"
    )
    if key and len(key) > 10:
        write_env_template(env_path)  # make sure the commented template exists
        from prometheus.config.env_file import set_env_value
        set_env_value(key_env, key, env_path)
        os.environ.setdefault(key_env, key)
        print(f"  + Key saved to {env_path} (as {key_env})")
    elif not os.environ.get(key_env):
        print(f"  x No key pasted and ${key_env} is not set.")
        print(f"    Get one, add `{key_env}=...` to {env_path}, and re-run "
              "`oara setup --fast`.")
        return None
    else:
        print(f"  + Using ${key_env} from your environment "
              f"(remember to add it to {env_path} for the daemon).")

    model = _prompt("Model", default_model)
    return _cloud_default_config(provider, key_env, model)


def _cloud_key_source(key_env: str, env_path: Path) -> str | None:
    """Where ``key_env`` is set: ``"environment"``, ``"env file"``, or None.

    The env file is the one the daemon loads (``~/.config/prometheus/env``,
    or the target-dir copy), so a key present only there still counts —
    a re-run of setup must not demand the key be re-exported.
    """
    import os

    if os.environ.get(key_env):
        return "environment"
    try:
        from prometheus.config.env_file import parse_env_file

        if env_path.exists() and parse_env_file(env_path).get(key_env):
            return "env file"
    except Exception:  # unreadable env file: treat as absent, never crash setup
        pass
    return None


def _cloud_config_noninteractive(
    provider: str,
    api_key_env: str | None,
    env_path: Path,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Cloud config with NO prompts. The key comes from the environment or
    the env file — never from argv, which lands in shell history.

    Returns None (and writes nothing) when the key is nowhere: a cloud
    config without its key is exactly the known-broken config the
    dead-end rule exists to refuse. A key found only in the environment is
    copied into the env file, because ``oara daemon`` under systemd
    does not inherit this shell.
    """
    import os

    if provider not in _CLOUD_FAST_PROVIDERS:
        print(f"  x Unknown cloud provider {provider!r}. "
              f"Choose one of: {', '.join(_CLOUD_FAST_PROVIDERS)}")
        return None
    default_env, default_model, _limit = _CLOUD_FAST_PROVIDERS[provider]
    key_env = api_key_env or default_env
    source = _cloud_key_source(key_env, env_path)
    if source is None:
        print(f"  x {provider}: ${key_env} is not set, and it is not in {env_path}.")
        print(f"    Export {key_env} (or add `{key_env}=...` to {env_path}) and re-run.")
        print()
        print("No config was written — a cloud config without its key is known-broken,")
        print("and Prometheus refuses to write a config that cannot work.")
        return None
    if source == "environment":
        write_env_template(env_path)  # make sure the commented template exists
        from prometheus.config.env_file import set_env_value

        set_env_value(key_env, os.environ[key_env], env_path)
        print(f"  + {provider}: using ${key_env} from your environment "
              f"(copied to {env_path} so the daemon has it too)")
    else:
        print(f"  + {provider}: using ${key_env} from {env_path}")
    return _cloud_default_config(provider, key_env, model or default_model)


def _cloud_providers_with_keys(env_path: Path) -> list[str]:
    """Every fast-path cloud provider whose default key is set somewhere."""
    return [
        name for name, (key_env, _model, _limit) in _CLOUD_FAST_PROVIDERS.items()
        if _cloud_key_source(key_env, env_path) is not None
    ]


def _handle_no_server(
    env_path: Path, timeout: float,
) -> tuple[DetectedServer | None, dict[str, Any] | None]:
    """Interactive dead-end menu. Returns (remote_server, cloud_config).

    ``(None, None)`` means: exit cleanly, write nothing.
    """
    while True:
        choice = _prompt_choice(
            "No local inference server was found. What now?",
            [
                "Point Prometheus at a remote server URL",
                "Use a cloud provider (Anthropic / OpenAI / DeepSeek / "
                "Kimi / GLM / MiMo)",
                "Show install instructions for a local server and exit",
            ],
            default_index=2,
        )
        if choice == 0:
            server = _prompt_remote_server(timeout)
            if server is not None:
                return server, None
            continue  # nothing reachable — back to the menu
        if choice == 1:
            cloud = _prompt_cloud_provider(env_path)
            if cloud is not None:
                return None, cloud
            continue
        _print_install_instructions()
        return None, None


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def run_init(
    *,
    noninteractive: bool = False,
    target_dir: Path | None = None,
    timeout: float = 1.0,
    candidates: list[dict[str, str]] | None = None,
    probe_url: str | None = None,
    provider: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Run the fast setup flow. Returns the config that was written.

    ``provider`` (``--provider``) names a fast-path cloud provider and
    skips every menu: the config is written if that provider's key is in
    the environment or the env file, and refused otherwise. It wins over
    a detected local server, because it is explicit. Without it, a
    noninteractive run that finds no local server looks for exactly one
    cloud key in the environment / env file and uses that provider —
    ``ANTHROPIC_API_KEY=… oara setup --noninteractive`` on a box
    with no GPU is the cloud-only install. Several keys is ambiguous and
    refused with a ``--provider`` hint; none keeps the install-
    instructions dead end.

    Returns ``None`` when the flow exited cleanly WITHOUT writing a
    config (nothing detected and the user took — or noninteractive mode
    implies — the install-instructions path). ``candidates`` overrides
    the probe list for tests.

    ``probe_url`` (the ``--probe-url`` flag) replaces the four well-known
    localhost candidates with the two known API shapes at ONE user-supplied
    URL, via :func:`remote_server_candidates` — the same definition of
    "what counts as an inference server" the interactive remote prompt and
    the pairing wizard use. Without it, ``--noninteractive`` on a machine
    whose server is not on 8080/11434/1234/8000 (or in CI, ever) has
    nothing to detect and exits without writing a config — the FIRSTLIGHT
    fresh-install harness is what surfaced that gap. An explicit
    ``candidates`` still wins (tests).
    """
    from prometheus.config.env_file import get_env_file_path

    cfg_dir = target_dir or get_config_dir()
    cfg_path = cfg_dir / "prometheus.yaml"
    # The env file the daemon actually loads lives at
    # ~/.config/prometheus/env (see prometheus.config.env_file). An
    # explicit --target-dir confines EVERYTHING under it (tests rely on
    # this — no writes outside the target dir).
    env_path = (cfg_dir / "env") if target_dir else get_env_file_path()

    print("┌─ Prometheus setup (fast) ─────────────────────────────────────┐")
    print(f"  Config will be written to {cfg_path}")
    print("└───────────────────────────────────────────────────────────────┘")

    if candidates is None and probe_url:
        candidates = remote_server_candidates(probe_url)

    print("\nProbing for local inference servers …")
    servers = detect_local_servers(timeout=timeout, candidates=candidates)
    _print_detection_summary(servers)

    # Pick a model provider
    chosen_server: DetectedServer | None = None
    chosen_model: str | None = None
    cloud_config: dict[str, Any] | None = None
    if provider is not None:
        # Explicit cloud choice: no menus, wins over any detected server.
        if servers:
            print(f"\n--provider {provider}: using the cloud provider, not the "
                  f"detected local server(s).")
        cloud_config = _cloud_config_noninteractive(
            provider, api_key_env, env_path, model,
        )
        if cloud_config is None:
            return None
    elif servers:
        if noninteractive:
            chosen_server = servers[0]
            chosen_model = chosen_server.models[0] if chosen_server.models else None
        else:
            labels = [f"{s.name} @ {s.url}" for s in servers]
            labels.append("None of these — remote URL or cloud provider")
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
            else:
                chosen_server, cloud_config = _handle_no_server(env_path, timeout)
                if chosen_server is None and cloud_config is None:
                    return None
                if chosen_server is not None and chosen_server.models:
                    chosen_model = chosen_server.models[0]
    else:
        # Dead-end rule: never write a config pointing at a server that
        # isn't there.
        if noninteractive:
            keyed = _cloud_providers_with_keys(env_path)
            if len(keyed) == 1:
                print(f"\nNo local server, but ${_CLOUD_FAST_PROVIDERS[keyed[0]][0]} "
                      f"is set — using the {keyed[0]} cloud provider.")
                cloud_config = _cloud_config_noninteractive(
                    keyed[0], api_key_env, env_path, model,
                )
                if cloud_config is None:
                    return None
            elif len(keyed) > 1:
                print(f"\nNo local server, and keys for several cloud providers are "
                      f"set ({', '.join(keyed)}). Say which one:")
                print(f"  oara setup --noninteractive --provider {keyed[0]}")
                print()
                print("No config was written — the choice is yours, not a coin flip.")
                return None
            else:
                _print_install_instructions()
                return None
        else:
            chosen_server, cloud_config = _handle_no_server(env_path, timeout)
            if chosen_server is None and cloud_config is None:
                return None
            if chosen_server is not None and chosen_server.models:
                chosen_model = chosen_server.models[0]

    # Gateway choice (don't require tokens up-front; user can enable later.
    # SPRINT G3: all three gateways are options in the SAME prompt — the
    # fast path stays fast, `oara setup` has the guided per-gateway
    # flow with token validation.)
    if noninteractive:
        gateway_choice = "cli"
    else:
        gw_idx = _prompt_choice(
            "Which gateway should Prometheus respond on?",
            [
                "CLI only (run `prometheus` for an interactive shell)",
                "Telegram (you'll add a bot token via env later)",
                "Slack (you'll add bot+app tokens via env later)",
                "Discord (you'll add a bot token via env later)",
                "All three (Telegram + Slack + Discord — tokens via env later)",
            ],
            default_index=0,
        )
        gateway_choice = ["cli", "telegram", "slack", "discord", "all"][gw_idx]

    # Build config
    if cloud_config is not None:
        config = cloud_config
    else:
        config = _default_config(chosen_server, chosen_model)
    if gateway_choice in ("telegram", "all"):
        config["gateway"]["telegram_enabled"] = True
    if gateway_choice in ("slack", "all"):
        config["gateway"]["slack"]["enabled"] = True
    if gateway_choice in ("discord", "all"):
        config["gateway"]["discord"]["enabled"] = True

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

    # Onboarding Phase 1, item 1: "Connect a client (Beacon)" end block —
    # printed on the fast/noninteractive path too, so BOTH wizard paths
    # end with the address + token a client needs.
    from prometheus.config.api_token import format_connect_client_block
    print(format_connect_client_block(config), end="")

    # Friendly next-steps
    print()
    print("Setup complete. Next steps:")
    if gateway_choice == "cli":
        print("  1. Chat now:       prometheus")
        print("  2. Always-on:      oara daemon   (Beacon dashboard on "
              "http://localhost:8005;")
        print("     first start mints a web API token — printed once, "
              "`oara token show` re-prints)")
    else:
        print(f"  1. Edit {env_path} to add gateway tokens")
        print("  2. Run: oara daemon   (Beacon dashboard on "
              "http://localhost:8005)")
    print("  Guided gateway setup (Telegram / Slack / Discord, with token")
    print("  validation):       oara setup --gateway-only")
    print("  Health check anytime:  oara doctor")
    print()
    return config


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``prometheus-init`` console-script entry point.

    Thin forwarding alias: ``oara setup --fast`` is the canonical
    command now. This keeps existing muscle memory and scripts working.
    """
    parser = argparse.ArgumentParser(
        prog="prometheus-init",
        description=(
            "[alias for `oara setup --fast`] Detects local inference "
            "servers and writes a working prometheus.yaml."
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

    print("note: `prometheus-init` is now `oara setup --fast` — "
          "forwarding.\n")
    forward = ["--fast", "--timeout", str(args.timeout)]
    if args.noninteractive:
        forward.append("--noninteractive")
    if args.target_dir is not None:
        forward.extend(["--target-dir", str(args.target_dir)])

    from prometheus.cli.setup import main as setup_main
    try:
        return setup_main(forward)
    except KeyboardInterrupt:
        print("\nInit aborted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
