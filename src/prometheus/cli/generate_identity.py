"""Generate identity files (SOUL.md, AGENTS.md) from templates.

Called by the setup wizard during first-run. Can also be run standalone:
    python -m prometheus identity --regenerate
"""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache
from pathlib import Path

from prometheus.config.paths import get_config_dir


PROMETHEUS_HOME = get_config_dir()

TEMPLATE_NAMES = ("SOUL.md.template", "AGENTS.md.template")


class IdentityTemplateNotFound(FileNotFoundError):
    """A shipped identity template is missing from an installed package."""


@lru_cache(maxsize=None)
def identity_template_path(name: str) -> Path:
    """Absolute path to a shipped identity template.

    ⚠ THIS USED TO BE A MODULE CONSTANT, AND THE CONSTANT DID NOT SHIP.
    It pointed four parents up from this file, at ``<repo>/templates`` — one
    level ABOVE ``src/prometheus``, which is the only tree
    ``[tool.hatch.build.targets.wheel] packages`` carries. So every git
    checkout had these files by accident of the checkout, and every ``pip
    install`` had none: ``oara setup`` reached the identity step and died on
    ``FileNotFoundError`` partway through a first run. Same defect, same
    shape, and the same fix as the config template (see
    :mod:`prometheus.config.template`).

    Looks, in order:

    1. ``prometheus/templates/`` beside the package — where the wheel
       force-includes them;
    2. ``<repo>/templates/`` — a source checkout or editable install.

    Raises rather than returning ``None``: a caller handed ``None`` writes a
    half-personalised SOUL.md, which is worse than not writing one.
    """
    package_root = Path(__file__).resolve().parent.parent  # src/prometheus
    packaged = package_root / "templates" / name
    if packaged.is_file():
        return packaged

    # src/prometheus/cli/generate_identity.py -> repo root is four parents up.
    checkout = Path(__file__).resolve().parents[3] / "templates" / name
    if checkout.is_file():
        return checkout

    raise IdentityTemplateNotFound(
        f"{name} not found at {packaged} nor at {checkout}. The wheel "
        f"force-includes templates/ via "
        f"[tool.hatch.build.targets.wheel.force-include]; if that stanza was "
        f"removed, `oara setup` fails at the identity step on installed "
        f"packages while every checkout keeps working."
    )


def read_identity_template(name: str) -> str:
    """The raw text of a shipped identity template."""
    return identity_template_path(name).read_text(encoding="utf-8")


def detect_hardware() -> dict:
    """Auto-detect hardware configuration."""
    gpu = _detect_gpu()
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu": _detect_cpu(),
        "ram_gb": _detect_ram(),
        "gpu": gpu,
        "has_gpu": gpu is not None,
    }


def _detect_cpu() -> str:
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or "Unknown CPU"
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        return "Unknown CPU"
    except Exception:
        return "Unknown CPU"


def _detect_ram() -> int:
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip()) // (1024 ** 3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return kb // (1024 ** 2)
        return 0
    except Exception:
        return 0


def _detect_gpu() -> str | None:
    """Best-effort GPU name for the SOUL.md hardware block.

    ⚠ THIS RUNS DURING FIRST-RUN SETUP, so anything it raises aborts the
    install partway through. It caught only ``FileNotFoundError`` — the case
    where nvidia-smi is absent — and every other way nvidia-smi disappoints
    was an unhandled exception on a fresh machine:

    * **two GPUs.** ``--format=csv,noheader`` prints one line PER GPU, and
      this split the whole blob on "," — so ``parts[1]`` was
      ``"24564\nNVIDIA GeForce RTX 4090"`` and ``int()`` raised. A second
      card broke setup.
    * a hung driver → ``TimeoutExpired`` after 5s;
    * ``nvidia-smi`` present but not executable → ``PermissionError``;
    * a driver reporting ``[N/A]`` for memory → ``ValueError``.

    Now: first line only, and every failure degrades to less detail rather
    than to an exception — matching :func:`_detect_cpu` and :func:`_detect_ram`
    beside it, which have always caught broadly. A cosmetic field in a
    generated markdown file must never be able to fail an install.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # One line per GPU; the first is the one SOUL.md names.
            first = result.stdout.strip().splitlines()[0]
            parts = first.split(",")
            name = parts[0].strip()
            if not name:
                return None
            try:
                vram = int(parts[1].strip()) // 1024
            except (IndexError, ValueError):
                # Driver gave no usable number ("[N/A]", a changed column
                # set). The name alone is still true and still useful.
                return name
            return f"{name} ({vram}GB)"
    except Exception:
        pass
    if platform.system() == "Darwin" and "arm" in platform.machine():
        return "Apple Silicon (unified memory)"
    return None


def render_soul_md(
    owner_name: str,
    hardware: dict,
    hardware_layout: str = "single",
    gpu_machine_name: str | None = None,
    brain_machine_name: str | None = None,
    owner_description: str = "",
    vision_available: bool | None = None,
    agent_name: str = "Prometheus",
    persona: str = "",
) -> str:
    """Render SOUL.md from template with user's values.

    ``agent_name`` fills the template's ``{{AGENT_NAME}}`` slots (default
    keeps the historical "Prometheus" rendering byte-identical);
    ``persona`` (Onboarding Phase 2, Beacon identity step) appends a
    short "## Persona" section when non-empty.
    """
    template = read_identity_template("SOUL.md.template")

    if hardware_layout == "split":
        hw_lines = [
            f"- **{brain_machine_name or 'Brain'}**: storage, orchestration, Telegram gateway",
            f"- **{gpu_machine_name or 'GPU'}**: inference via llama.cpp, GPU-bound tasks",
        ]
        if hardware["gpu"]:
            hw_lines.append(f"- **GPU**: {hardware['gpu']}")
        hw_lines.append("- Connected via **Tailscale** mesh (or local network)")
        hw_lines.append("- Model loaded at startup (auto-detected from inference server)")
    else:
        hw_lines = [
            f"- **{hardware['hostname']}**: {hardware['os']} {hardware['arch']}",
        ]
        if hardware.get("cpu"):
            hw_lines.append(f"- **CPU**: {hardware['cpu']}")
        if hardware.get("ram_gb"):
            hw_lines.append(f"- **RAM**: {hardware['ram_gb']}GB")
        if hardware["gpu"]:
            hw_lines.append(f"- **GPU**: {hardware['gpu']}")
        else:
            hw_lines.append("- **GPU**: None (CPU inference or cloud API)")
        hw_lines.append("- Model loaded at startup (auto-detected from inference server)")

    if vision_available is True:
        vision_line = "multimodal image analysis (confirmed available)"
    elif vision_available is False:
        vision_line = "multimodal image analysis (not available \u2014 load mmproj to enable)"
    elif hardware.get("has_gpu"):
        vision_line = "multimodal image analysis via model's vision adapter (mmproj)"
    else:
        vision_line = "multimodal image analysis (if model supports vision)"
    voice_line = "speech-to-text via Whisper (local transcription of voice memos)"

    owner_desc = f" \u2014 {owner_description}" if owner_description else ""

    result = template.replace("{{AGENT_NAME}}", agent_name or "Prometheus")
    result = result.replace("{{OWNER_NAME}}", owner_name)
    result = result.replace("{{HARDWARE_SECTION}}", "\n".join(hw_lines))
    result = result.replace("{{VISION_LINE}}", vision_line)
    result = result.replace("{{VOICE_LINE}}", voice_line)
    result = result.replace("{{OWNER_DESCRIPTION}}", owner_desc)
    if persona.strip():
        result = result.rstrip("\n") + f"\n\n## Persona\n\n{persona.strip()}\n"
    return result


def render_agents_md() -> str:
    """Render AGENTS.md from template. No personalization needed."""
    return read_identity_template("AGENTS.md.template")


def generate_identity_files(
    owner_name: str,
    hardware: dict,
    hardware_layout: str = "single",
    gpu_machine_name: str | None = None,
    brain_machine_name: str | None = None,
    owner_description: str = "",
    overwrite: bool = False,
    dest: Path | None = None,
    agent_name: str = "Prometheus",
    persona: str = "",
) -> dict[str, str]:
    """Generate all identity files in ~/.prometheus/ (or dest).

    Returns a dict of filename -> status.
    MEMORY.md and USER.md are NEVER overwritten.
    """
    home = dest or PROMETHEUS_HOME
    home.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    soul_path = home / "SOUL.md"
    if soul_path.exists() and not overwrite:
        results["SOUL.md"] = "exists (skipped)"
    else:
        soul_path.write_text(render_soul_md(
            owner_name, hardware, hardware_layout,
            gpu_machine_name, brain_machine_name, owner_description,
            agent_name=agent_name, persona=persona,
        ))
        results["SOUL.md"] = "created"

    agents_path = home / "AGENTS.md"
    if agents_path.exists() and not overwrite:
        results["AGENTS.md"] = "exists (skipped)"
    else:
        agents_path.write_text(render_agents_md())
        results["AGENTS.md"] = "created"

    memory_path = home / "MEMORY.md"
    if not memory_path.exists():
        memory_path.write_text("# Memory\n\n<!-- Facts are added here by the agent -->\n")
        results["MEMORY.md"] = "created (empty)"
    else:
        results["MEMORY.md"] = "exists (preserved)"

    user_path = home / "USER.md"
    if not user_path.exists():
        user_path.write_text("# User Model\n\n<!-- Updated by the agent as it learns about you -->\n")
        results["USER.md"] = "created (empty)"
    else:
        results["USER.md"] = "exists (preserved)"

    return results
