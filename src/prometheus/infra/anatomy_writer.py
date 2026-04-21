"""AnatomyWriter — generates ANATOMY.md from infrastructure state + project configs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from prometheus.config.paths import get_config_dir
from prometheus.infra.anatomy import AnatomyState

log = logging.getLogger(__name__)


class AnatomyWriter:
    """Write and update ANATOMY.md from infrastructure state."""

    def __init__(self, anatomy_path: Path | None = None) -> None:
        self._path = anatomy_path or (get_config_dir() / "ANATOMY.md")

    @property
    def path(self) -> Path:
        return self._path

    def write(
        self,
        state: AnatomyState,
        project_summaries: list[dict] | None = None,
    ) -> str:
        """Generate full ANATOMY.md content and write to disk."""
        content = self._render(state, project_summaries or [])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(content, encoding="utf-8")
        log.info("ANATOMY.md written to %s", self._path)
        return content

    def update_active_section(self, state: AnatomyState) -> None:
        """Update only the Active Configuration section, preserving project configs."""
        if not self._path.exists():
            self.write(state)
            return

        text = self._path.read_text(encoding="utf-8")
        new_active = self._render_active(state)

        # Replace between "## Active Configuration" and the next "## " heading
        pattern = r"(## Active Configuration\n).*?(?=\n## |\Z)"
        replacement = f"## Active Configuration\n{new_active}"
        updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)

        if count == 0:
            # Section not found — append
            updated = text.rstrip() + f"\n\n## Active Configuration\n{new_active}\n"

        # Update timestamp
        updated = re.sub(
            r"Last scanned: .+",
            f"Last scanned: {state.scanned_at}",
            updated,
            count=1,
        )

        self._path.write_text(updated, encoding="utf-8")

    def render_mermaid(self, state: AnatomyState) -> str:
        """Generate Mermaid diagram of current architecture."""
        lines = ["graph LR"]

        lines.append('    User["Telegram"] --> Mini["Brain Node<br/>Daemon + Storage"]')

        if state.gpu_name:
            gpu_label = state.gpu_name.replace("NVIDIA ", "")
            model_label = _short_model(state.model_name) if state.model_name else "model"
            lines.append(
                f'    Mini -->|"Tailscale"| GPU["GPU Node<br/>{model_label}"]'
            )
            engine = state.inference_engine.replace("_", ".")
            port = state.inference_url.rsplit(":", 1)[-1].rstrip("/") if ":" in state.inference_url else "8080"
            lines.append(f'    GPU -->|"{engine} :{port}"| Mini')

            lines.append(f'    subgraph "4090 ({gpu_label})"')
            lines.append("        GPU")
            if state.vision_enabled:
                lines.append('        Vision["mmproj Vision"]')
            lines.append("    end")
        else:
            lines.append('    Mini -->|"local"| Model["Local Model"]')

        lines.append('    Mini --> Wiki[("Wiki + LCM<br/>SQLite")]')
        lines.append('    Mini --> Memory[("MEMORY.md<br/>USER.md")]')

        return "\n".join(lines)

    def render_summary(self, state: AnatomyState, project_names: list[str] | None = None) -> str:
        """Render compact summary for system prompt injection (~200-300 tokens)."""
        parts: list[str] = ["## Infrastructure"]

        # Hardware line
        hw = f"Running on {state.hostname}"
        if state.gpu_name:
            gpu_short = state.gpu_name.replace("NVIDIA ", "")
            hw += f" + GPU ({gpu_short})"
        hw += "."
        parts.append(hw)

        # Model line
        if state.model_name:
            model_line = f"Model: {_short_model(state.model_name)}"
            if state.model_quantization:
                model_line += f" ({state.model_quantization})"
            model_line += f" via {state.inference_engine.replace('_', '.')}."
            if state.vision_enabled:
                model_line += " Vision enabled."
            parts.append(model_line)

        # VRAM
        if state.gpu_vram_free_mb is not None and state.gpu_vram_total_mb:
            free_gb = state.gpu_vram_free_mb / 1024
            total_gb = state.gpu_vram_total_mb / 1024
            parts.append(f"VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total.")

        # Projects
        if project_names:
            parts.append(f"Configs: {', '.join(project_names)}.")

        parts.append("Use the anatomy tool for full details or to switch configurations.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Rendering internals
    # ------------------------------------------------------------------

    def _render(self, state: AnatomyState, project_summaries: list[dict]) -> str:
        sections = [
            f"# Anatomy \u2014 Infrastructure State\nLast scanned: {state.scanned_at}",
            f"## Active Configuration\n{self._render_active(state)}",
            f"## Architecture\n\n```mermaid\n{self.render_mermaid(state)}\n```",
        ]

        if project_summaries:
            proj_lines = ["## Project Configurations"]
            for proj in project_summaries:
                proj_lines.append(f"\n### {proj['name']} \u2014 {proj.get('description', '')}")
                for k, v in proj.items():
                    if k not in ("name", "description"):
                        proj_lines.append(f"- {k}: {v}")
            sections.append("\n".join(proj_lines))

        return "\n\n".join(sections) + "\n"

    def _render_active(self, state: AnatomyState) -> str:
        lines: list[str] = []

        # Hardware table
        lines.append("### Hardware")
        lines.append("| Machine | Role | CPU | RAM |")
        lines.append("|---------|------|-----|-----|")
        ram_str = f"{state.ram_total_gb:.0f}GB" if state.ram_total_gb else "?"
        lines.append(f"| {state.hostname} | Host | {state.cpu[:40]} | {ram_str} |")

        # GPU
        if state.gpu_name:
            lines.append("")
            lines.append("### GPU")
            lines.append(f"- **Name:** {state.gpu_name}")
            if state.gpu_vram_total_mb:
                used = state.gpu_vram_used_mb or 0
                free = state.gpu_vram_free_mb or 0
                total = state.gpu_vram_total_mb
                lines.append(f"- **VRAM:** {used}MB / {total}MB used ({free}MB free)")

        # Model
        lines.append("")
        lines.append("### Model")
        if state.model_name:
            lines.append(f"- **Loaded:** {state.model_name}")
            if state.model_file and state.model_file != state.model_name:
                lines.append(f"- **File:** {state.model_file}")
            if state.model_quantization:
                lines.append(f"- **Quantization:** {state.model_quantization}")
        else:
            lines.append("- **Loaded:** (none detected)")
        lines.append(f"- **Engine:** {state.inference_engine} ({state.inference_url})")
        lines.append(f"- **Vision:** {'Enabled' if state.vision_enabled else 'Disabled'}")
        if state.inference_features:
            lines.append(f"- **Features:** {', '.join(state.inference_features)}")

        # Services
        lines.append("")
        lines.append("### Services")
        lines.append("- Prometheus daemon: running")
        if state.whisper_model:
            lines.append(f"- Whisper STT: {state.whisper_model} model")
        if state.tailscale_ip:
            peer_count = len(state.tailscale_peers)
            lines.append(f"- Tailscale: {state.tailscale_ip} ({peer_count} peers)")
            for peer in state.tailscale_peers:
                if isinstance(peer, dict):
                    status = "online" if peer.get("online") else "offline"
                    lines.append(f"  - {peer['name']}: {peer.get('ip', '?')} ({status})")
                else:
                    lines.append(f"  - {peer}")

        # Storage
        if state.disk_total_gb:
            lines.append("")
            lines.append("### Storage")
            lines.append(f"- Disk: {state.disk_free_gb}GB free / {state.disk_total_gb}GB total")
            if state.prometheus_data_size_mb:
                lines.append(f"- ~/.prometheus: {state.prometheus_data_size_mb}MB")

        return "\n".join(lines)


def _short_model(name: str | None) -> str:
    """Shorten a model name for display."""
    if not name:
        return "unknown"
    # Strip path prefixes and common suffixes
    short = name.rsplit("/", 1)[-1]
    short = short.replace(".gguf", "")
    return short
