"""Doctor — diagnoses Prometheus configuration against model capabilities.

Compares what IS running (from AnatomyScanner) against what SHOULD
be available (from model registry) and produces actionable diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

from prometheus.api.turn_errors import redact_url
from prometheus.config.paths import get_config_dir
from prometheus.infra.anatomy import AnatomyState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticCheck:
    name: str       # "Vision"
    category: str   # "platform", "connectivity", "model", "resources"
    status: str     # "ok", "warning", "error", "info"
    message: str    # "mmproj not loaded"
    fix: str | None = None  # "Add --mmproj to launch command"


@dataclass
class DiagnosticReport:
    model_name: str | None
    model_family: str | None  # matched registry display_name
    checks: list[DiagnosticCheck] = field(default_factory=list)
    timestamp: str = ""

    CATEGORY_ORDER = ["platform", "connectivity", "model", "resources"]
    CATEGORY_LABELS = {
        "platform": "Platform",
        "connectivity": "Connectivity",
        "model": "Model",
        "resources": "Resources",
    }

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warning" for c in self.checks)

    @property
    def has_errors(self) -> bool:
        return any(c.status == "error" for c in self.checks)

    def checks_by_category(self) -> dict[str, list[DiagnosticCheck]]:
        """Group checks by category in display order."""
        grouped: dict[str, list[DiagnosticCheck]] = {}
        for cat in self.CATEGORY_ORDER:
            cat_checks = [c for c in self.checks if c.category == cat]
            if cat_checks:
                grouped[cat] = cat_checks
        return grouped


# ---------------------------------------------------------------------------
# Registry matching
# ---------------------------------------------------------------------------

def match_model(model_name: str, registry: dict) -> dict | None:
    """Find registry entry for a loaded model.

    Tries each model family's match_patterns as case-insensitive
    substring matches against the model name/filename.
    Returns the first match, or None if unknown model.
    """
    if not model_name:
        return None
    model_lower = model_name.lower()
    for _family_id, family in registry.get("models", {}).items():
        for pattern in family.get("match_patterns", []):
            if pattern.lower() in model_lower:
                return family
    return None


def _mark_position(mark: yaml.Mark) -> str:
    """A PyYAML mark as ``name, line L, column C`` (1-based, like editors)."""
    return f"{mark.name}, line {mark.line + 1}, column {mark.column + 1}"


def yaml_error_summary(exc: BaseException) -> str:
    """WHAT KIND of YAML error, and WHERE — never any text from the document.

    Returns the exception's class name and its mark positions, nothing else.
    ``str(yaml.YAMLError)`` is unsafe to print because it quotes the document
    in two places:

    * the SNIPPET. ``MarkedYAMLError.__str__`` calls ``mark.get_snippet()``,
      which renders the offending line out of ``mark.buffer`` — the whole file,
      when it was loaded from a string — truncated at a fixed width rather than
      at a token boundary. prometheus.yaml holds credentials inline
      (``gateway.telegram_token``, ``model.api_key``), so an unterminated
      quote, a tab indent or an unclosed ``[`` on a token line printed part of
      the token.
    * the ``problem`` / ``context`` TEXT. These are not fixed messages: PyYAML
      interpolates what it read into them. An unquoted value that begins with a
      YAML indicator is read as a tag or an alias and quoted back — in full for
      ``!SECRET`` and ``!!SECRET`` ("could not determine a constructor for the
      tag ..."), up to the first ``:`` for ``*SECRET`` ("found undefined alias
      ..."), and the part between the bangs for ``!SECRET!x`` ("found undefined
      tag handle ..."). An earlier version of this helper kept both fields on
      the belief that PyYAML only ever quotes the single character it choked
      on. These shapes are the counterexample.

    Both doctors show this to a person — ``oara doctor`` prints it to stdout
    via ``render_report``, and the chat ``/doctor`` command sends it as a
    message — and stdout does not pass through ``install_log_redaction``.
    Redaction would not have caught it anyway: it matches whole tokens, and
    neither a truncated snippet nor a value quoted back as a tag is one.

    What survives is what the operator needs to find the line: the class name,
    which says what kind of failure it was, and both marks — where parsing
    failed (``problem_mark``) and where the construct it was inside started
    (``context_mark``). An unmarked ``YAMLError``, or any other exception, gets
    its class name only: we cannot know what an arbitrary ``__str__`` renders,
    and the default has to be the safe one.
    """
    summary = type(exc).__name__
    if not isinstance(exc, yaml.MarkedYAMLError):
        return summary
    if exc.problem_mark is not None:
        summary += f" at {_mark_position(exc.problem_mark)}"
    if exc.context_mark is not None:
        summary += (" (inside the construct that starts at "
                    f"{_mark_position(exc.context_mark)})")
    return summary


def check_router(config: dict) -> DiagnosticCheck | None:
    """The router: entries boot skipped, and why — or None when there is no
    router to speak of (no rules, no fallback, nothing refused).

    Reads the same parse the daemon boots with, so this row and the boot
    WARNINGs cannot disagree. A refused entry does not stop the daemon; it
    is skipped, which is exactly why it has to be visible somewhere other
    than a journal line.
    """
    from prometheus.router.model_router import read_router_config

    try:
        rc = read_router_config(config)
    except Exception as exc:  # noqa: BLE001 — a doctor row, never a crash
        return DiagnosticCheck(
            name="Router", category="platform", status="warning",
            message=f"could not read the router: section ({type(exc).__name__}: {exc})",
        )
    if rc.problems:
        return DiagnosticCheck(
            name="Router", category="platform", status="warning",
            message=f"{len(rc.problems)} skipped at boot: " + "; ".join(rc.problems),
            fix="Correct or remove each entry named above in the router: section "
                "of prometheus.yaml. Until then it is skipped and everything "
                "else runs.",
        )
    if not (rc.task_rules or rc.fallback_chain):
        return None
    return DiagnosticCheck(
        name="Router", category="platform", status="ok",
        message=f"{len(rc.task_rules)} rule(s), {len(rc.fallback_chain)} fallback entr"
                f"{'y' if len(rc.fallback_chain) == 1 else 'ies'}",
    )


#: The row both doctors show for the advertised tool set (``oara doctor``'s
#: extended check adds the count of tools actually offered to the same text).
ADVERTISED_TOOLS_ROW = "Advertised tools"


def always_loaded_case(config: dict | None) -> tuple[str, str, str | None]:
    """Does the advertised tool set follow the shipped default, or is it pinned?

    Returns ``(status, text, fix)``. Reads the same rule the loader boots with
    (``shipped_defaults.resolve_always_loaded``), so the row and the boot line
    cannot disagree. A pinned list that leaves out tools the shipped default
    offers is a warning: that is exactly how ``skill`` stayed out of every
    config setup wrote before 0.9.5.
    """
    from prometheus.config.shipped_defaults import (
        ALWAYS_LOADED_ABSENT,
        ALWAYS_LOADED_PINNED,
        SHIPPED_ALWAYS_LOADED,
        always_loaded_origin,
    )

    tools_cfg = (config or {}).get("tools")
    deferred = (tools_cfg or {}).get("deferred_loading") if isinstance(tools_cfg, dict) else None
    deferred = deferred if isinstance(deferred, dict) else {}
    origin = always_loaded_origin(deferred)
    shipped = len(SHIPPED_ALWAYS_LOADED)
    if origin == ALWAYS_LOADED_ABSENT:
        return "ok", f"follows the shipped default ({shipped} tools)", None
    if origin != ALWAYS_LOADED_PINNED:
        return ("ok", f"follows the shipped default ({shipped} tools): the config lists "
                "a set Prometheus shipped as its default, which counts as the default", None)
    configured = deferred["always_loaded"]
    remove = ("delete tools.deferred_loading.always_loaded to follow the shipped default")
    if not isinstance(configured, (list, tuple)):
        return ("warning", "pinned by tools.deferred_loading.always_loaded, which is not a list",
                f"Write a list of tool names, or {remove}.")
    missing = [name for name in SHIPPED_ALWAYS_LOADED if name not in configured]
    if missing:
        return ("warning",
                f"pinned by tools.deferred_loading.always_loaded ({len(configured)} tools); "
                f"not offered from the shipped default: {', '.join(missing)}",
                f"Add them to the list, or {remove}.")
    return ("ok", f"pinned by tools.deferred_loading.always_loaded ({len(configured)} tools, "
            "every shipped default among them)", None)


def check_always_loaded(config: dict | None) -> DiagnosticCheck:
    """The advertised tool set: following the shipped default, or pinned."""
    status, text, fix = always_loaded_case(config)
    return DiagnosticCheck(
        name=ADVERTISED_TOOLS_ROW, category="resources", status=status,
        message=text, fix=fix,
    )


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------

class Doctor:
    """Diagnoses Prometheus configuration against model capabilities."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.registry = self._load_registry()

    def _repo_root(self) -> Path:
        """Project root used to locate the bundled config files
        (``config/prometheus.yaml``, ``config/model_registry.yaml``).

        Defaults to the package's repo root, but is overridable via a
        ``repo_root`` attribute so tests can point the config checks at a
        fixture. Otherwise they depend on a gitignored ``config/prometheus.yaml``
        existing at the real repo root — absent in any linked git worktree (and
        in an installed package), which made ``test_full_diagnose_all_ok`` fail
        outside the daemon's checkout.
        """
        return getattr(self, "repo_root", None) or Path(__file__).resolve().parents[3]

    def _load_registry(self) -> dict:
        """Load model_registry.yaml from config directory."""
        doctor_cfg = self.config.get("doctor", {})
        registry_file = doctor_cfg.get("registry_file", "config/model_registry.yaml")

        # Try relative to project root first, then absolute
        candidates = [
            self._repo_root() / registry_file,
            Path(registry_file).expanduser(),
        ]
        for path in candidates:
            if path.exists():
                try:
                    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except Exception as exc:
                    log.warning("Failed to load model registry from %s: %s", path, exc)
                    return {}
        log.debug("Model registry not found at %s", candidates)
        return {}

    async def diagnose(self, state: AnatomyState) -> DiagnosticReport:
        """Run full diagnostic. Returns structured report."""
        checks: list[DiagnosticCheck | None] = []

        # ── Platform ──
        checks.append(self._check_python_version())
        checks.append(self._check_uv())
        checks.append(self._check_config_valid())
        checks.append(check_router(self.config))
        checks.append(check_always_loaded(self.config))
        checks.append(self._check_data_dir())
        checks.append(self._check_bootstrap_files())
        checks.append(self._check_dependencies())

        # ── Connectivity ──
        checks.append(await self._check_inference(state))
        checks.append(self._check_telegram_token())
        checks.append(self._check_cloud_keys())
        checks.append(self._check_tailscale(state))

        # ── Model ──
        checks.append(self._check_model_loaded(state))

        model_family = None
        if state.model_name or state.model_file:
            model_family = match_model(
                state.model_file or state.model_name or "",
                self.registry,
            )

        if model_family:
            checks.append(self._check_vision(state, model_family))
            checks.append(self._check_function_calling(state, model_family))

        # ── Resources ──
        checks.append(self._check_gpu(state))
        checks.append(self._check_disk(state))
        checks.append(self._check_whisper(state))

        # Filter out None checks
        return DiagnosticReport(
            model_name=state.model_name,
            model_family=model_family.get("display_name") if model_family else None,
            checks=[c for c in checks if c is not None],
            timestamp=state.scanned_at,
        )

    # ------------------------------------------------------------------
    # Platform checks
    # ------------------------------------------------------------------

    def _check_python_version(self) -> DiagnosticCheck:
        """Python 3.11+ required."""
        import sys
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 11):
            return DiagnosticCheck(
                name="Python", category="platform", status="ok",
                message=f"Python {version}",
            )
        return DiagnosticCheck(
            name="Python", category="platform", status="error",
            message=f"Python {version} — 3.11+ required",
            fix="Install Python 3.11 or newer: https://python.org/downloads/",
        )

    def _check_uv(self) -> DiagnosticCheck:
        """Is uv available? Prometheus runs via uv run."""
        import shutil
        if shutil.which("uv"):
            return DiagnosticCheck(
                name="uv", category="platform", status="ok",
                message="installed",
            )
        return DiagnosticCheck(
            name="uv", category="platform", status="warning",
            message="uv not found — needed for `uv run`",
            fix="Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
        )

    def _check_config_valid(self) -> DiagnosticCheck:
        """Does prometheus.yaml parse without errors?"""
        config_path = self._repo_root() / "config" / "prometheus.yaml"
        if not config_path.exists():
            return DiagnosticCheck(
                name="Config", category="platform", status="error",
                message="config/prometheus.yaml not found",
                fix="Run `python -m prometheus --setup` or copy prometheus.yaml.default",
            )
        try:
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            return DiagnosticCheck(
                name="Config", category="platform", status="ok",
                message="valid",
            )
        except yaml.YAMLError as exc:
            return DiagnosticCheck(
                name="Config", category="platform", status="error",
                # yaml_error_summary, NOT str(exc): the latter quotes the
                # file (a snippet, or a tag/alias value it read back), and
                # prometheus.yaml keeps credentials inline. See
                # yaml_error_summary.
                message=f"YAML parse error: {yaml_error_summary(exc)}",
                fix="Fix syntax errors in config/prometheus.yaml",
            )

    def _check_data_dir(self) -> DiagnosticCheck:
        """Does the Prometheus data directory exist?"""
        config_dir = get_config_dir()
        if config_dir.exists():
            return DiagnosticCheck(
                name="Data Dir", category="platform", status="ok",
                message=str(config_dir),
            )
        return DiagnosticCheck(
            name="Data Dir", category="platform", status="warning",
            message=f"{config_dir} does not exist",
            fix="Run `python -m prometheus --setup` or `mkdir -p ~/.prometheus`",
        )

    def _check_dependencies(self) -> DiagnosticCheck:
        """Can we import key packages?"""
        missing: list[str] = []
        for pkg in ["yaml", "httpx", "telegram", "pydantic"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            return DiagnosticCheck(
                name="Dependencies", category="platform", status="error",
                message=f"Missing packages: {', '.join(missing)}",
                fix="Run: pip install -e . (or uv pip install -e .)",
            )
        return DiagnosticCheck(
            name="Dependencies", category="platform", status="ok",
            message="all required packages installed",
        )

    def _check_bootstrap_files(self) -> DiagnosticCheck:
        """SOUL.md and AGENTS.md exist?"""
        config_dir = get_config_dir()
        soul = config_dir / "SOUL.md"
        agents = config_dir / "AGENTS.md"
        missing: list[str] = []
        if not soul.exists():
            missing.append("SOUL.md")
        if not agents.exists():
            missing.append("AGENTS.md")
        if missing:
            return DiagnosticCheck(
                name="Bootstrap", category="platform", status="error",
                message=f"Missing: {', '.join(missing)}",
                fix="Run `python -m prometheus --setup` to generate identity files.",
            )
        return DiagnosticCheck(
            name="Bootstrap", category="platform", status="ok",
            message="SOUL.md and AGENTS.md present",
        )

    # ------------------------------------------------------------------
    # Connectivity checks
    # ------------------------------------------------------------------

    async def _check_inference(self, state: AnatomyState) -> DiagnosticCheck:
        """Is the inference engine reachable?

        The URL is DISPLAYED through ``api.turn_errors.redact_url``, the same
        helper the message path uses: ``state.inference_url`` can carry
        ``user:pass@`` (a llama.cpp server behind an authenticating proxy) or a
        ``?api_key=``, and this row goes to a chat as ``/doctor`` output, which
        ``install_log_redaction`` does not cover. The REQUEST below still uses
        the real URL — redaction is display-only.
        """
        url = state.inference_url
        if not url:
            return DiagnosticCheck(
                name="Inference", category="connectivity", status="error",
                message="No inference URL configured",
                fix="Set model.base_url in prometheus.yaml.",
            )
        engine = state.inference_engine.replace("_", ".")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if "llama" in state.inference_engine:
                    resp = await client.get(f"{url}/v1/models")
                else:
                    resp = await client.get(f"{url}/api/tags")
                resp.raise_for_status()
            return DiagnosticCheck(
                name="Inference", category="connectivity", status="ok",
                message=f"{engine} reachable at {redact_url(url)}",
            )
        except Exception:
            return DiagnosticCheck(
                name="Inference", category="connectivity", status="error",
                message=f"{engine} not responding at {redact_url(url)}",
                fix="Check that the server is running on the target machine.",
            )

    def _check_telegram_token(self) -> DiagnosticCheck:
        """Is the Telegram bot token configured?

        Set/not-set ONLY — the value is never echoed, not even "masked". The
        previous form printed ``token[:4] + "..." + token[-4:]``, which is not
        a mask: for a Telegram token the head IS the bot id and the tail is
        real entropy from the secret half, so it published usable pieces of a
        live credential into a terminal, a Telegram chat and CI logs. This
        matches `_check_cloud_keys`, which has always reported presence only.
        """
        import os
        # Check env var (primary) then config
        token = os.environ.get("PROMETHEUS_TELEGRAM_TOKEN", "")
        if not token:
            gateway_cfg = self.config.get("gateway", {})
            token = gateway_cfg.get("telegram_token", "")
        if not token or token == "YOUR_BOT_TOKEN_HERE":
            return DiagnosticCheck(
                name="Telegram", category="connectivity", status="warning",
                message="bot token not configured",
                fix="Get a token from @BotFather and set PROMETHEUS_TELEGRAM_TOKEN env var.",
            )
        return DiagnosticCheck(
            name="Telegram", category="connectivity", status="ok",
            message="bot token set",
        )

    def _check_cloud_keys(self) -> DiagnosticCheck:
        """CLOUD EXPANSION (2026-07): one info line listing which optional
        cloud-service keys are present in the environment.

        Set/not-set ONLY — values are never echoed (not even masked). These
        services are dormant-until-keyed by design, so absence is "info",
        never a warning.
        """
        import os

        llm_keys = (
            ("DeepSeek", "DEEPSEEK_API_KEY"),
            ("Kimi", "MOONSHOT_API_KEY"),
            ("GLM", "ZAI_API_KEY"),
            ("MiMo", "MIMO_API_KEY"),
            ("Qwen", "QWEN_API_KEY"),
            # Presence only — a custom endpoint is the usual cause of a 403 on
            # a key that is otherwise valid, so it is worth seeing in /doctor.
            ("Qwen endpoint override", "QWEN_BASE_URL"),
            ("DashScope/WAN", "DASHSCOPE_API_KEY"),
        )
        parts = [
            f"{label} {'set' if os.environ.get(env) else 'not set'}"
            for label, env in llm_keys
        ]
        kling_ok = bool(
            os.environ.get("KLING_ACCESS_KEY")
            and os.environ.get("KLING_SECRET_KEY")
        )
        parts.append(f"Kling AK+SK {'set' if kling_ok else 'not set'}")
        return DiagnosticCheck(
            name="Cloud keys", category="connectivity", status="info",
            message=" · ".join(parts),
        )

    def _check_tailscale(self, state: AnatomyState) -> DiagnosticCheck | None:
        """Tailscale connectivity (if multi-machine setup)."""
        from urllib.parse import urlparse
        inf_host = urlparse(state.inference_url).hostname or ""
        is_remote = inf_host not in ("", "localhost", "127.0.0.1", "::1")

        if not is_remote:
            return None  # local setup, tailscale not relevant

        if not state.tailscale_ip:
            return DiagnosticCheck(
                name="Tailscale", category="connectivity", status="warning",
                message="Remote inference detected but Tailscale not found",
                fix="Install Tailscale for secure inter-machine networking.",
            )

        # Check if inference host is in peers and online
        for peer in state.tailscale_peers:
            if isinstance(peer, dict):
                if peer.get("ip") == inf_host and peer.get("online"):
                    return DiagnosticCheck(
                        name="Tailscale", category="connectivity", status="ok",
                        message=f"Connected to {peer['name']} ({inf_host})",
                    )
        return DiagnosticCheck(
            name="Tailscale", category="connectivity", status="ok",
            message=f"Tailscale active ({len(state.tailscale_peers)} peers)",
        )

    # ------------------------------------------------------------------
    # Model checks
    # ------------------------------------------------------------------

    def _check_model_loaded(self, state: AnatomyState) -> DiagnosticCheck:
        """Is a model actually loaded?"""
        if state.model_name:
            label = state.model_name
            if state.model_quantization:
                label += f" ({state.model_quantization})"
            return DiagnosticCheck(
                name="Model", category="model", status="ok",
                message=f"Loaded: {label}",
            )
        return DiagnosticCheck(
            name="Model", category="model", status="error",
            message="No model detected",
            fix="Load a model in llama.cpp or ollama.",
        )

    def _check_vision(
        self, state: AnatomyState, family: dict,
    ) -> DiagnosticCheck | None:
        """Vision: model supports it but is it enabled?"""
        vision_cap = family.get("capabilities", {}).get("vision", {})
        if not vision_cap.get("supported"):
            return DiagnosticCheck(
                name="Vision", category="model", status="info",
                message=f"{family['display_name']} does not support vision",
            )
        if state.vision_enabled:
            return DiagnosticCheck(
                name="Vision", category="model", status="ok",
                message="Vision enabled (mmproj loaded)",
            )
        return DiagnosticCheck(
            name="Vision", category="model", status="warning",
            message=f"{family['display_name']} supports vision but mmproj is not loaded",
            fix=vision_cap.get("setup_hint", "Load the mmproj file with --mmproj flag."),
        )

    def _check_function_calling(
        self, state: AnatomyState, family: dict,
    ) -> DiagnosticCheck | None:
        """Function calling available?"""
        fc_cap = family.get("capabilities", {}).get("function_calling", {})
        if not fc_cap.get("supported"):
            return None
        return DiagnosticCheck(
            name="Function Calling", category="model", status="ok",
            message=f"{family['display_name']} supports tool calling",
        )

    # ------------------------------------------------------------------
    # Resource checks
    # ------------------------------------------------------------------

    def _check_gpu(self, state: AnatomyState) -> DiagnosticCheck:
        """GPU detected and VRAM healthy?"""
        if state.gpu_name and state.gpu_vram_total_mb:
            free_gb = (state.gpu_vram_free_mb or 0) / 1024
            if free_gb < 0.5:
                return DiagnosticCheck(
                    name="GPU", category="resources", status="warning",
                    message=f"{state.gpu_name} — VRAM nearly full ({free_gb:.1f} GB free)",
                    fix="Consider a smaller quantization or unloading unused models.",
                )
            return DiagnosticCheck(
                name="GPU", category="resources", status="ok",
                message=f"{state.gpu_name} — {free_gb:.1f} GB VRAM free",
            )
        if state.gpu_name:
            return DiagnosticCheck(
                name="GPU", category="resources", status="info",
                message=f"{state.gpu_name} detected but VRAM stats unavailable",
                fix="Ensure nvidia-smi is accessible (locally or via SSH).",
            )
        return DiagnosticCheck(
            name="GPU", category="resources", status="warning",
            message="No GPU detected",
            fix="Running on CPU. For faster inference, configure a GPU machine.",
        )

    def _check_disk(self, state: AnatomyState) -> DiagnosticCheck:
        """Disk space adequate?"""
        if state.disk_free_gb < 10:
            return DiagnosticCheck(
                name="Disk", category="resources", status="warning",
                message=f"Low disk space: {state.disk_free_gb:.1f} GB free",
                fix="Free up space. GGUF models and wiki data can grow large.",
            )
        return DiagnosticCheck(
            name="Disk", category="resources", status="ok",
            message=f"{state.disk_free_gb:.1f} GB free",
        )

    def _check_whisper(self, state: AnatomyState) -> DiagnosticCheck:
        """Whisper STT status."""
        if state.whisper_model:
            return DiagnosticCheck(
                name="Whisper STT", category="resources", status="ok",
                message=f"Model: {state.whisper_model}",
            )
        return DiagnosticCheck(
            name="Whisper STT", category="resources", status="info",
            message="Not configured — voice input disabled",
            fix="Add whisper config to prometheus.yaml to enable voice messages.",
        )
