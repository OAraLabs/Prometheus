"""First-run setup wizard for Prometheus.

Walks the user through provider selection, gateway configuration,
directory creation, config writing, and a smoke test.

Usage:
    python3 -m prometheus --setup
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from prometheus.config.paths import get_config_dir
from prometheus.providers.registry import CLOUD_DEFAULTS, ProviderRegistry

# Config file lives in the repo's config/ directory
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "prometheus.yaml"

# Cloud provider model choices: (model_id, label, pricing)
CLOUD_PROVIDER_MODELS: dict[str, list[tuple[str, str, str]]] = {
    "openai": [
        ("gpt-4o", "Best quality", "$2.50/$10 per 1M tokens"),
        ("gpt-4o-mini", "Fast + cheap", "$0.15/$0.60 per 1M tokens"),
        ("o3-mini", "Reasoning", "$1.10/$4.40 per 1M tokens"),
    ],
    "anthropic": [
        ("claude-sonnet-4-6", "Best quality", "$3/$15 per 1M tokens"),
        ("claude-haiku-4-5-20251001", "Fast + cheap", "$0.80/$4 per 1M tokens"),
    ],
    "gemini": [
        ("gemini-2.5-flash", "Fast + cheap", "$0.15/$0.60 per 1M tokens"),
        ("gemini-2.5-pro", "Best quality", "$1.25/$10 per 1M tokens"),
    ],
    "xai": [
        ("grok-3", "Flagship", "$3/$15 per 1M tokens"),
        ("grok-3-mini", "Fast + cheap", "$0.30/$0.50 per 1M tokens"),
    ],
}

CLOUD_DEFAULT_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
}

# Effective context limits per provider for sane defaults
PROVIDER_EFFECTIVE_LIMITS: dict[str, int] = {
    "llama_cpp": 24000,
    "ollama": 24000,
    "openai": 64000,
    "gemini": 64000,
    "xai": 64000,
    "anthropic": 100000,
}


def _input(prompt: str, default: str = "") -> str:
    """Read a line from stdin with an optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        sys.exit(1)
    return value or default


def _ask_choice(prompt: str, options: list[str], default: int = 1) -> int:
    """Present numbered options and return the 1-based choice."""
    print(f"\n{prompt}\n")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    raw = _input(f"\nChoice", str(default))
    try:
        choice = int(raw)
        if 1 <= choice <= len(options):
            return choice
    except ValueError:
        pass
    print(f"Invalid choice, using default ({default}).")
    return default


class SetupWizard:
    """Interactive first-run setup. Runs once, creates config, validates."""

    def __init__(self, gateway_only: bool = False) -> None:
        self._gateway_only = gateway_only
        self._provider: str = "llama_cpp"
        self._base_url: str = "http://localhost:8080"
        self._model_name: str = ""
        self._api_key_env: str = ""
        self._gateway: str = "cli"
        self._telegram_token: str = ""
        self._telegram_chat_ids: list[int] = []
        self._telegram_bot_name: str = ""
        self._slack_bot_token: str = ""
        self._slack_app_token: str = ""
        self._slack_channels: list[str] = []
        self._slack_workspace: str = ""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Run the full wizard. Returns True if setup succeeded."""
        existing = self._load_existing_config()

        # Auto-detect Hermes/OpenClaw before setup
        if not existing and not self._gateway_only:
            self._offer_migration()

        if existing and not self._gateway_only:
            action = self._ask_rerun()
            if action == 4:
                print("Cancelled.")
                return False
            if action == 2:
                self._gateway_only = False
                self._prefill_from(existing)
                self._step_provider()
                self._merge_and_write(existing)
                return self._run_smoke_test()
            if action == 3:
                self._gateway_only = True
                self._prefill_from(existing)

        self._print_banner()

        # Identity setup — generate SOUL.md, AGENTS.md from templates
        if not self._gateway_only:
            self._setup_identity()

        if not self._gateway_only:
            if not self._check_prerequisites():
                return False
            self._step_provider()

        self._step_gateway()
        self._create_directories()

        if self._gateway_only and existing:
            self._merge_and_write(existing)
        else:
            self._write_config()

        passed = True
        if not self._gateway_only:
            passed = self._run_smoke_test()
            self._check_vision_hint()

        self._print_summary(passed)
        return passed

    # ------------------------------------------------------------------
    # Banner + prerequisites
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        GREEN = "\033[32m"
        RESET = "\033[0m"
        print(f"""
{GREEN}    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   🔥 P R O M E T H E U S                            ║
    ║                                                      ║
    ║      Sovereign AI Agent Harness                      ║
    ║      95 skills • 37 tools • Infinite potential       ║
    ║                                                      ║
    ║   "The model is the agent. The code is the harness." ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝{RESET}
""")

    def _check_prerequisites(self) -> bool:
        print("Checking prerequisites...")

        # Python version
        vi = sys.version_info
        if vi < (3, 11):
            print(f"  x Python {vi.major}.{vi.minor}.{vi.micro} — need 3.11+")
            print("\nPlease upgrade Python and try again.")
            return False
        print(f"  + Python {vi.major}.{vi.minor}.{vi.micro}")

        # Key dependencies
        missing: list[str] = []
        for mod in ("yaml", "httpx"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)

        if missing:
            print(f"  x Missing packages: {', '.join(missing)}")
            print('\n  Run: pip install -e ".[dev]"')
            return False
        print("  + All core dependencies installed")

        # Config dir
        config_dir = get_config_dir()
        print(f"  + Config directory: {config_dir}")
        print()
        return True

    # ------------------------------------------------------------------
    # Step 2: Model provider
    # ------------------------------------------------------------------

    def _step_provider(self) -> None:
        choice = _ask_choice(
            "Where is your LLM?",
            [
                "Local — llama.cpp (recommended for sovereignty)",
                "Local — Ollama",
                "Cloud — OpenAI (GPT-4o, o3-mini)",
                "Cloud — Anthropic (Claude Sonnet, Haiku)",
                "Cloud — Google Gemini (Flash, Pro)",
                "Cloud — xAI (Grok)",
                "I don't have one running yet",
            ],
            default=1,
        )

        if choice == 7:
            self._print_no_model_help()
            sys.exit(0)

        # Cloud providers
        cloud_map = {3: "openai", 4: "anthropic", 5: "gemini", 6: "xai"}
        if choice in cloud_map:
            self._step_cloud_provider(cloud_map[choice])
            return

        # Local providers
        self._provider = "llama_cpp" if choice == 1 else "ollama"
        default_url = (
            "http://localhost:8080" if choice == 1 else "http://localhost:11434"
        )

        while True:
            url = _input(f"\nEnter the URL", default_url)
            url = url.rstrip("/")
            print(f"\n  Testing connection to {url}...")

            model = self._test_provider(url)
            if model is not None:
                print(f"  + Connected")
                if model:
                    print(f"  Detected model: {model}")
                    self._model_name = model
                self._base_url = url
                return

            print(f"  x Could not connect to {url}")
            print("    - Is the server running?")
            print("    - Is the URL correct?")
            print("    - If remote, use the full URL (e.g., http://192.168.1.100:8080)")

            retry = _input("\nTry a different URL? [y/N]", "N")
            if retry.lower() != "y":
                print("\nSaving config with this URL — you can fix it later.")
                self._base_url = url
                return

    def _step_cloud_provider(self, provider: str) -> None:
        """Collect cloud provider config: API key, model, smoke test."""
        import os

        self._provider = provider
        default_env = CLOUD_DEFAULT_ENV_VARS[provider]

        print(f"\nEnter your API key, or press Enter to use env var ${default_env}:")
        key_input = _input(f"API key", default_env)

        if key_input == default_env or not key_input:
            # Use env var
            self._api_key_env = default_env
            api_key = os.environ.get(default_env, "")
            if not api_key:
                print(f"\n  ! ${default_env} is not set.")
                print(f"  Set it with: export {default_env}=your-key-here")
                print(f"  Config will reference ${default_env}.\n")
        elif len(key_input) > 20:
            # They pasted a raw key — store as env var reference
            self._api_key_env = default_env
            api_key = key_input
            print(f"\n  For security, add this to your shell profile:")
            print(f"  export {default_env}={key_input}")
            print(f"  Config will reference ${default_env}, not store the key.\n")
            os.environ[default_env] = key_input  # temp set for smoke test
        else:
            # Custom env var name
            self._api_key_env = key_input
            api_key = os.environ.get(key_input, "")

        # Model selection
        models = CLOUD_PROVIDER_MODELS[provider]
        print(f"\nWhich model?")
        for i, (name, desc, price) in enumerate(models, 1):
            print(f"  {i}. {name} ({desc}, {price})")
        raw = _input("Choice", "1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                self._model_name = models[idx][0]
            else:
                self._model_name = models[0][0]
        except ValueError:
            self._model_name = models[0][0]

        # Set base_url from defaults (cloud providers don't need user input)
        defaults = CLOUD_DEFAULTS.get(provider, {})
        self._base_url = defaults.get("base_url", "")

    def _test_provider(self, url: str) -> str | None:
        """Test connection to provider. Returns model name or None on failure."""
        # Try /v1/models (works for llama.cpp and Ollama OpenAI-compat)
        try:
            resp = httpx.get(f"{url}/v1/models", timeout=10.0)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                return models[0].get("id", "")
            return ""
        except Exception:
            pass

        # Try Ollama /api/tags
        if self._provider == "ollama":
            try:
                resp = httpx.get(f"{url}/api/tags", timeout=10.0)
                resp.raise_for_status()
                models = resp.json().get("models", [])
                if models:
                    return models[0].get("name", "")
                return ""
            except Exception:
                pass

        return None

    def _print_no_model_help(self) -> None:
        print(
            """
No problem. Here are your options:

  Local (free, sovereign):
    llama.cpp:
      git clone https://github.com/ggerganov/llama.cpp.git
      cd llama.cpp && make LLAMA_CUDA=1 -j$(nproc)
      ./llama-server -m models/your-model.gguf -c 32768 -ngl 99 --port 8080

    Ollama:
      curl -fsSL https://ollama.com/install.sh | sh
      ollama run qwen3.5:32b

  Cloud (API key required):
    export OPENAI_API_KEY=sk-...       # OpenAI
    export ANTHROPIC_API_KEY=sk-ant-...  # Anthropic
    export GEMINI_API_KEY=...          # Google Gemini
    export XAI_API_KEY=...             # xAI Grok

Run this wizard again after you're ready:
  python3 -m prometheus --setup
"""
        )

    # ------------------------------------------------------------------
    # Step 3: Gateway
    # ------------------------------------------------------------------

    def _step_gateway(self) -> None:
        choice = _ask_choice(
            "How do you want to talk to Prometheus?",
            [
                "Telegram",
                "Slack",
                "Both (Telegram + Slack)",
                "CLI only (no messaging gateway)",
            ],
            default=4,
        )

        if choice == 4:
            self._gateway = "cli"
            print(
                "\n  Got it — CLI mode. Add a gateway later with:"
                "\n    python3 -m prometheus --setup --gateway-only"
            )
            return

        if choice in (1, 3):
            self._setup_telegram()

        if choice in (2, 3):
            self._setup_slack()

        # Determine gateway value
        has_tg = bool(self._telegram_token)
        has_slack = bool(self._slack_bot_token and self._slack_app_token)
        if has_tg and has_slack:
            self._gateway = "both"
        elif has_tg:
            self._gateway = "telegram"
        elif has_slack:
            self._gateway = "slack"
        else:
            self._gateway = "cli"
            print("\n  No gateway configured. Running in CLI mode.")

    def _setup_telegram(self) -> None:
        """Collect and validate Telegram bot token."""
        token = _input("\nEnter your Telegram bot token (from @BotFather)")
        if not token:
            print("  No token provided. Skipping Telegram.")
            return

        print("  Testing token...")
        bot_name = self._test_telegram_token(token)
        if bot_name:
            print(f"  + Bot connected: @{bot_name}")
            self._telegram_token = token
            self._telegram_bot_name = bot_name
        else:
            print("  x Invalid token or could not reach Telegram API.")
            keep = _input("Save this token anyway? [y/N]", "N")
            if keep.lower() == "y":
                self._telegram_token = token
            else:
                print("  Skipping Telegram.")
                return

        # Optional: restrict to chat ID
        print(
            "\n  Optional: Restrict to your chat ID only?"
            "\n  Send /start to your bot, then enter your chat ID here."
            "\n  Leave blank to allow all users."
        )
        chat_id_str = _input("Chat ID", "")
        if chat_id_str:
            try:
                self._telegram_chat_ids = [int(chat_id_str)]
            except ValueError:
                print("  Invalid chat ID, skipping restriction.")

    def _setup_slack(self) -> None:
        """Collect and validate Slack bot + app tokens."""
        print(
            "\n  Slack setup requires two tokens from https://api.slack.com/apps"
            "\n"
            "\n  1. Create a new Slack App (From Scratch)"
            "\n  2. Add Bot Token Scopes: chat:write, app_mentions:read,"
            "\n     im:history, im:read, im:write, channels:history, commands"
            "\n  3. Enable Socket Mode (Settings > Socket Mode > Enable)"
            "\n  4. Install App to your workspace"
            "\n  5. Copy both tokens:"
        )
        bot_token = _input("\nBot Token (xoxb-...)")
        if not bot_token:
            print("  No bot token provided. Skipping Slack.")
            return

        app_token = _input("App Token (xapp-...)")
        if not app_token:
            print("  No app token provided. Skipping Slack.")
            return

        print("  Testing connection...")
        workspace = self._test_slack_token(bot_token)
        if workspace:
            print(f"  + Connected to workspace: {workspace}")
            self._slack_bot_token = bot_token
            self._slack_app_token = app_token
            self._slack_workspace = workspace
        else:
            print("  x Invalid token or could not reach Slack API.")
            keep = _input("Save these tokens anyway? [y/N]", "N")
            if keep.lower() == "y":
                self._slack_bot_token = bot_token
                self._slack_app_token = app_token
            else:
                print("  Skipping Slack.")
                return

        # Optional: restrict to channels
        print(
            "\n  Optional: Restrict to specific channels?"
            "\n  Enter channel IDs separated by commas, or leave blank for all."
        )
        channels_str = _input("Channels", "")
        if channels_str:
            self._slack_channels = [
                c.strip() for c in channels_str.split(",") if c.strip()
            ]

    def _test_slack_token(self, bot_token: str) -> str | None:
        """Test a Slack bot token via auth.test. Returns workspace name or None."""
        try:
            resp = httpx.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {bot_token}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                return data.get("team", "")
        except Exception:
            pass
        return None

    def _test_telegram_token(self, token: str) -> str | None:
        """Test a Telegram bot token via getMe. Returns bot username or None."""
        try:
            resp = httpx.get(
                f"https://api.telegram.org/bot{token}/getMe", timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                return data["result"].get("username", "")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Step 4: Create directories + write config
    # ------------------------------------------------------------------

    def _create_directories(self) -> None:
        print("\nCreating Prometheus data directories...")
        config_dir = get_config_dir()
        dirs = [
            config_dir / "workspace",
            config_dir / "wiki",
            config_dir / "sentinel",
            config_dir / "skills",
            config_dir / "data",
            config_dir / "data" / "sessions",
            config_dir / "logs",
            config_dir / "cache",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  + {d}")

    def _write_config(self) -> None:
        """Write a fresh config, preserving defaults from the template."""
        print(f"\nWriting config to {_REPO_CONFIG}...")

        # Load template defaults
        base = self._load_existing_config() or {}

        # Update wizard-touched fields
        self._apply_wizard_fields(base)

        self._save_config(base)

    def _merge_and_write(self, existing: dict[str, Any]) -> None:
        """Merge wizard fields into existing config."""
        print(f"\nUpdating config at {_REPO_CONFIG}...")
        self._apply_wizard_fields(existing)
        self._save_config(existing)

    def _apply_wizard_fields(self, cfg: dict[str, Any]) -> None:
        """Apply wizard answers to the config dict."""
        if not self._gateway_only:
            model = cfg.setdefault("model", {})
            model["provider"] = self._provider

            if ProviderRegistry.is_cloud(self._provider):
                # Cloud provider: store env var reference, remove base_url
                if self._api_key_env:
                    model["api_key_env"] = self._api_key_env
                model.pop("base_url", None)
                model.pop("fallback_url", None)
                model.pop("fallback_provider", None)
                # Set effective context limit for cloud
                context = cfg.setdefault("context", {})
                context["effective_limit"] = PROVIDER_EFFECTIVE_LIMITS.get(
                    self._provider, 64000
                )
            else:
                # Local provider: store URL
                model["base_url"] = self._base_url
                if self._provider == "ollama":
                    model["fallback_url"] = self._base_url
                model.pop("api_key_env", None)

            if self._model_name:
                model["model"] = self._model_name

        gateway = cfg.setdefault("gateway", {})

        # Telegram
        if self._gateway in ("telegram", "both"):
            gateway["telegram_enabled"] = True
            gateway["telegram_token"] = self._telegram_token
            gateway["allowed_chat_ids"] = self._telegram_chat_ids
        elif self._gateway == "cli":
            gateway["telegram_enabled"] = False

        # Slack
        if self._gateway in ("slack", "both"):
            gateway["slack_enabled"] = True
            gateway["slack_bot_token"] = self._slack_bot_token
            gateway["slack_app_token"] = self._slack_app_token
            gateway["slack_channels"] = self._slack_channels
        elif self._gateway == "cli":
            gateway["slack_enabled"] = False

    def _save_config(self, cfg: dict[str, Any]) -> None:
        """Write config dict to YAML file and update .gitignore."""
        _REPO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with _REPO_CONFIG.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

        # Print summary of what was written
        model = cfg.get("model", {})
        gw = cfg.get("gateway", {})
        provider = model.get("provider", "?")
        if ProviderRegistry.is_cloud(provider):
            env = model.get("api_key_env", "?")
            print(f"  + Model provider: {provider} (key: ${env})")
        else:
            print(f"  + Model provider: {provider} @ {model.get('base_url', '?')}")
        if model.get("model"):
            print(f"  + Model: {model['model']}")
        gateways_active: list[str] = []
        if gw.get("telegram_enabled"):
            token = gw.get("telegram_token", "")
            masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
            print(f"  + Gateway: Telegram (token: {masked})")
            gateways_active.append("telegram")
        if gw.get("slack_enabled"):
            token = gw.get("slack_bot_token", "")
            masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
            print(f"  + Gateway: Slack (bot token: {masked})")
            gateways_active.append("slack")
        if not gateways_active:
            print("  + Gateway: CLI only")
        print("  + Config saved")

        # Ensure prometheus.yaml is in .gitignore (it may contain secrets)
        self._ensure_gitignore()

    def _ensure_gitignore(self) -> None:
        """Add config/prometheus.yaml to .gitignore if not already there."""
        gitignore = _REPO_CONFIG.parents[1] / ".gitignore"
        entry = "config/prometheus.yaml"
        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            if entry in content:
                return
            if not content.endswith("\n"):
                content += "\n"
            content += f"\n# Prometheus config (contains bot tokens)\n{entry}\n"
            gitignore.write_text(content, encoding="utf-8")
        else:
            gitignore.write_text(f"# Prometheus config (contains bot tokens)\n{entry}\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 5: Smoke test
    # ------------------------------------------------------------------

    def _run_smoke_test(self) -> bool:
        """Send a simple prompt to the model and verify a response."""
        import os
        import time

        print("\nRunning smoke test...")
        print("  Testing LLM connection...")

        try:
            t0 = time.monotonic()

            headers: dict[str, str] = {"Content-Type": "application/json"}
            payload: dict[str, Any] = {
                "model": self._model_name or "local",
                "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                "max_tokens": 32,
                "stream": False,
            }

            if self._provider == "anthropic":
                # Anthropic uses a different API format
                api_key = os.environ.get(self._api_key_env, "")
                if not api_key:
                    print(f"  x ${self._api_key_env} not set — skipping smoke test.")
                    return True
                url = "https://api.anthropic.com/v1/messages"
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
                payload = {
                    "model": self._model_name or "claude-sonnet-4-6",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                }
                resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                elapsed = time.monotonic() - t0
                data = resp.json()
                content = data.get("content", [])
                text = content[0].get("text", "") if content else ""
            elif ProviderRegistry.is_cloud(self._provider):
                # OpenAI-compatible cloud provider
                api_key = os.environ.get(self._api_key_env, "")
                if not api_key:
                    print(f"  x ${self._api_key_env} not set — skipping smoke test.")
                    return True
                defaults = CLOUD_DEFAULTS.get(self._provider, {})
                base = self._base_url or defaults.get("base_url", "")
                base = base.rstrip("/")
                if base.endswith("/v1"):
                    url = f"{base}/chat/completions"
                else:
                    url = f"{base}/v1/chat/completions"
                headers["Authorization"] = f"Bearer {api_key}"
                resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                elapsed = time.monotonic() - t0
                data = resp.json()
                choices = data.get("choices", [])
                text = (choices[0].get("message", {}).get("content", "") or "").strip() if choices else ""
            else:
                # Local provider
                url = f"{self._base_url.rstrip('/')}/v1/chat/completions"
                resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                elapsed = time.monotonic() - t0
                data = resp.json()
                choices = data.get("choices", [])
                text = (choices[0].get("message", {}).get("content", "") or "").strip() if choices else ""

            print(f'  Sending test prompt: "What is 2+2?"')
            print(f"  Response: {text!r}")

            if "4" in text:
                print(f"  + Smoke test passed ({elapsed:.1f}s)")
                return True
            else:
                print(f"  ~ Smoke test got a response but '4' not found ({elapsed:.1f}s)")
                print("    The model is reachable — response quality may vary.")
                return True

        except Exception as exc:
            print(f"  x Smoke test failed: {exc}")
            print()
            print("  Config was saved. Fix the issue and test manually:")
            print('    python3 -m prometheus --once "What is 2+2?"')
            return False

    # ------------------------------------------------------------------
    # Identity generation (CLEAN-SLATE)
    # ------------------------------------------------------------------

    def _setup_identity(self) -> None:
        """Generate identity files if they don't exist yet."""
        from prometheus.cli.generate_identity import detect_hardware, generate_identity_files

        soul_path = get_config_dir() / "SOUL.md"
        if soul_path.exists():
            print(f"\n  Identity files already exist at {get_config_dir()}/")
            response = _input("  Regenerate identity? [y/N]", "n")
            if response.lower() != "y":
                return

        print("\n  Let's set up your Prometheus identity.\n")

        owner_name = _input("  Your name", "User")
        owner_desc = _input("  Brief description (e.g., 'engineer, builds robots') [optional]", "")

        print("\n  Detecting hardware...")
        hardware = detect_hardware()
        print(f"    Hostname: {hardware['hostname']}")
        print(f"    OS:       {hardware['os']} {hardware['arch']}")
        if hardware.get("gpu"):
            print(f"    GPU:      {hardware['gpu']}")
        else:
            print("    GPU:      not detected")

        print("\n  Hardware layout:")
        layout = _ask_choice("  How is your hardware set up?", [
            "Single machine (everything runs here)",
            "Split setup (brain + GPU on separate machines)",
        ], default=1)

        brain_name = None
        gpu_name = None
        hardware_layout = "single"
        if layout == 2:
            brain_name = _input("  Name for brain/storage machine", "Brain")
            gpu_name = _input("  Name for GPU/inference machine", "GPU")
            hardware_layout = "split"

        print("\n  Generating identity files...")
        results = generate_identity_files(
            owner_name=owner_name,
            hardware=hardware,
            hardware_layout=hardware_layout,
            gpu_machine_name=gpu_name,
            brain_machine_name=brain_name,
            owner_description=owner_desc,
            overwrite=True,
        )

        for filename, status in results.items():
            print(f"    {filename}: {status}")
        print()

    # ------------------------------------------------------------------
    # Vision hint (VISION-DETECT)
    # ------------------------------------------------------------------

    def _check_vision_hint(self) -> None:
        """After smoke test, check if the provider supports vision."""
        if self._provider != "llama_cpp":
            return
        try:
            from prometheus.providers.llama_cpp import LlamaCppProvider
            provider = LlamaCppProvider(base_url=self._base_url)
            has_vision = asyncio.run(provider.detect_vision())
            if has_vision:
                print("  Vision: enabled (multimodal)")
            else:
                print("  Vision: not available")
                model = (self._model_name or "").lower()
                vision_capable = ("gemma", "llava", "qwen-vl", "pixtral", "minicpm-v")
                if any(v in model for v in vision_capable):
                    print(
                        f"  Hint: {self._model_name} supports vision. "
                        "Restart llama.cpp with --mmproj to enable."
                    )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------

    def _print_summary(self, passed: bool) -> None:
        model_display = self._model_name or "(auto-detect)"
        provider_display = self._provider

        print()
        print("=" * 55)
        if passed:
            print("   + Prometheus is ready!")
        else:
            print("   ~ Prometheus config saved (smoke test failed)")
        print("=" * 55)
        print()
        print(f"  Model:    {model_display} via {provider_display}")

        # Gateway summary
        gw_parts: list[str] = []
        if self._gateway in ("telegram", "both"):
            if self._telegram_bot_name:
                gw_parts.append(f"Telegram (@{self._telegram_bot_name})")
            else:
                gw_parts.append("Telegram")
        if self._gateway in ("slack", "both"):
            if self._slack_workspace:
                gw_parts.append(f"Slack ({self._slack_workspace})")
            else:
                gw_parts.append("Slack")
        if not gw_parts:
            gw_parts.append("CLI only")
        print(f"  Gateway:  {' + '.join(gw_parts)}")

        print(f"  Config:   {_REPO_CONFIG}")
        print(f"  Data:     {get_config_dir()}")
        print()
        print("Start Prometheus:")
        print()
        print("  Interactive:  python3 -m prometheus")
        print('  One-shot:     python3 -m prometheus --once "your question"')
        print(f"  Daemon:       python3 scripts/daemon.py --config {_REPO_CONFIG}")
        if self._gateway in ("telegram", "both") and self._telegram_bot_name:
            print()
            print(f"Send /start to @{self._telegram_bot_name} to begin.")
        if self._gateway in ("slack", "both"):
            print()
            print("Mention @prometheus in a Slack channel to chat.")
        print()
        print("=" * 55)

    # ------------------------------------------------------------------
    # Re-run handling
    # ------------------------------------------------------------------

    def _offer_migration(self) -> None:
        """Detect Hermes/OpenClaw and offer migration before setup."""
        from prometheus.cli.migrate import detect_sources, run_migration

        sources = detect_sources()
        if not sources:
            return

        print("\n  Existing agent installations detected:\n")
        for name, path in sources.items():
            label = "Hermes Agent" if name == "hermes" else "OpenClaw"
            print(f"    {label}: {path}")
        print()

        for name, path in sources.items():
            label = "Hermes Agent" if name == "hermes" else "OpenClaw"
            resp = _input(
                f"  Import your {label} data (memories, skills, config)? [Y/n]", "Y"
            )
            if resp.lower() in ("y", ""):
                import argparse
                args = argparse.Namespace(
                    source_type=name,
                    source_path=str(path),
                    dry_run=False,
                    overwrite=False,
                    preset="user-data",
                    skill_conflict="skip",
                    yes=True,
                )
                run_migration(args)
                print()

    def _ask_rerun(self) -> int:
        """Ask what to do when config already exists. Returns 1-4."""
        return _ask_choice(
            f"Existing configuration found at {_REPO_CONFIG}",
            [
                "Start fresh (overwrite everything)",
                "Update model provider only",
                "Update gateway only",
                "Cancel",
            ],
            default=4,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_existing_config(self) -> dict[str, Any] | None:
        """Load existing config if it exists."""
        if _REPO_CONFIG.exists():
            try:
                with _REPO_CONFIG.open(encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or {}
            except Exception:
                return None
        return None

    def _prefill_from(self, cfg: dict[str, Any]) -> None:
        """Pre-fill wizard state from existing config."""
        model = cfg.get("model", {})
        self._provider = model.get("provider", "llama_cpp")
        self._base_url = model.get("base_url", "http://localhost:8080")
        self._model_name = model.get("model", "")
        self._api_key_env = model.get("api_key_env", "")

        gw = cfg.get("gateway", {})
        has_tg = gw.get("telegram_enabled", False)
        has_slack = gw.get("slack_enabled", False)

        if has_tg and has_slack:
            self._gateway = "both"
        elif has_tg:
            self._gateway = "telegram"
        elif has_slack:
            self._gateway = "slack"
        else:
            self._gateway = "cli"

        self._telegram_token = gw.get("telegram_token", "")
        self._telegram_chat_ids = gw.get("allowed_chat_ids", [])
        self._slack_bot_token = gw.get("slack_bot_token", "")
        self._slack_app_token = gw.get("slack_app_token", "")
        self._slack_channels = gw.get("slack_channels", [])
