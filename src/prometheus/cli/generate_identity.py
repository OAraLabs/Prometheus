"""Generate identity files (SOUL.md, AGENTS.md) from templates.

Called by the setup wizard during first-run. Can also be run standalone:
    python -m prometheus identity --regenerate
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from prometheus.config.paths import get_config_dir


TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates"
PROMETHEUS_HOME = get_config_dir()


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
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            vram = int(parts[1].strip()) // 1024
            return f"{name} ({vram}GB)"
    except FileNotFoundError:
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
) -> str:
    """Render SOUL.md from template with user's values."""
    template = (TEMPLATES_DIR / "SOUL.md.template").read_text()

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

    result = template.replace("{{OWNER_NAME}}", owner_name)
    result = result.replace("{{HARDWARE_SECTION}}", "\n".join(hw_lines))
    result = result.replace("{{VISION_LINE}}", vision_line)
    result = result.replace("{{VOICE_LINE}}", voice_line)
    result = result.replace("{{OWNER_DESCRIPTION}}", owner_desc)
    return result


def render_agents_md() -> str:
    """Render AGENTS.md from template. No personalization needed."""
    return (TEMPLATES_DIR / "AGENTS.md.template").read_text()


def generate_identity_files(
    owner_name: str,
    hardware: dict,
    hardware_layout: str = "single",
    gpu_machine_name: str | None = None,
    brain_machine_name: str | None = None,
    owner_description: str = "",
    overwrite: bool = False,
    dest: Path | None = None,
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
