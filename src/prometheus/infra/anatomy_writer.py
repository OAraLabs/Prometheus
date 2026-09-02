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

        # Hardware line. Same trap as the full render: appending the
        # INFERENCE GPU to "Running on <host>" reads as "this host has that
        # card" — which is false whenever inference is remote. Name the local
        # card here (that is what "running on" means) and give the remote one
        # its own clause with the host attached.
        hw = f"Running on {state.hostname}"
        local_card = state.local_gpu_name or (
            state.gpu_name if not state.gpu_is_remote else None
        )
        if local_card:
            hw += f" + GPU ({local_card.replace('NVIDIA ', '')})"
        hw += "."
        if state.gpu_is_remote and state.gpu_name:
            hw += (
                f" Inference runs REMOTELY on "
                f"{state.gpu_inference_host or 'another host'} "
                f"({state.gpu_name.replace('NVIDIA ', '')})."
            )
        parts.append(hw)

        # Model line — "Local backend model", not "Model": this summary is read
        # by whichever model is serving the session, which under a /xai or
        # /claude override is NOT the local one (identity-confusion fix).
        if state.model_name:
            model_line = f"Local backend model: {_short_model(state.model_name)}"
            if state.model_quantization:
                model_line += f" ({state.model_quantization})"
            model_line += f" via {state.inference_engine.replace('_', '.')}."
            if state.vision_enabled:
                model_line += " Vision enabled."
            parts.append(model_line)

        # Backends beyond the primary, one clause each — the summary a chat
        # surface prints, so it stays one line.
        others = [b for b in state.backends if b.get("name") != "local"]
        if others:
            clauses = []
            for b in others:
                if not b.get("probed"):
                    clauses.append(f"{b.get('name')} not probed")
                elif b.get("ok"):
                    served = _short_model(b.get("model_path") or b.get("model") or "") or "?"
                    window = f", {int(b['n_ctx']) // 1024}k" if b.get("n_ctx") else ""
                    vision = ", vision" if b.get("vision") else ""
                    clauses.append(f"{b.get('name')} up ({served}{window}{vision})")
                else:
                    clauses.append(f"{b.get('name')} DOWN")
            parts.append("Backends: " + "; ".join(clauses) + ".")

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

        # GPU — this box may have TWO relevant cards, and the scanner has
        # always captured both. Rendering only `gpu_name` under a bare "GPU"
        # heading, directly beneath the local Hardware table, read as "this
        # machine has a 4090" when the 4090 is a SEPARATE box over Tailscale
        # and the local card is a 3090 Ti. anatomy.py's own field comment says
        # these exist because "without these fields the agent confidently
        # confuses the two" — they were populated and never consumed.
        #
        # This section lands in every system prompt, so each card is labelled
        # with WHICH MACHINE it is in, and a failed remote probe says so
        # rather than silently omitting the heading.
        gpu_lines: list[str] = []
        if state.gpu_name:
            if state.gpu_is_remote:
                where = f"REMOTE — {state.gpu_inference_host or 'inference host'}"
            else:
                where = "this machine"
            gpu_lines.append(f"- **Inference GPU ({where}):** {state.gpu_name}")
            if state.gpu_vram_total_mb:
                used = state.gpu_vram_used_mb or 0
                free = state.gpu_vram_free_mb or 0
                total = state.gpu_vram_total_mb
                gpu_lines.append(
                    f"  - VRAM: {used}MB / {total}MB used ({free}MB free)"
                )
        elif state.gpu_is_remote and state.gpu_inference_host:
            # Probe failed. Say so — an absent heading is indistinguishable
            # from "there is no GPU", which is a different fact.
            reason = state.gpu_probe_error or "probe failed"
            gpu_lines.append(
                f"- **Inference GPU (REMOTE — {state.gpu_inference_host}):** "
                f"not detected ({reason})"
            )

        if state.local_gpu_name:
            gpu_lines.append(
                f"- **Local GPU (this machine):** {state.local_gpu_name}"
            )
            if state.local_gpu_vram_total_mb:
                l_used = state.local_gpu_vram_used_mb or 0
                l_free = state.local_gpu_vram_free_mb or 0
                l_total = state.local_gpu_vram_total_mb
                gpu_lines.append(
                    f"  - VRAM: {l_used}MB / {l_total}MB used ({l_free}MB free)"
                )

        if gpu_lines:
            lines.append("")
            lines.append("### GPU")
            lines.extend(gpu_lines)

        # Model — labelled "local backend" deliberately. This section lands in
        # every system prompt; a cloud model serving an overridden session
        # (/xai, /claude, …) read the old "### Model / Loaded:" heading as ITS
        # OWN identity and answered "what model is this?" with the local GGUF.
        lines.append("")
        lines.append("### Local backend model (GPU node inventory)")
        lines.append(
            "- **Note:** auto-detected model loaded on the local GPU node — "
            "NOT necessarily the model serving this conversation (see the "
            "Environment section's `- Model:` line; per-session overrides "
            "may route chats to a cloud provider)."
        )
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

        # Every backend the registry knows — the same rows /api/backends and the
        # Models catalog render, so the agent reads what the operator sees.
        if state.backends:
            lines.append("")
            lines.append("### Backends (registry)")
            lines.append(
                "- **Note:** what each configured inference box was last seen "
                "serving. `/4090`-style commands point a chat at a box; the "
                "registry reports, it never restarts a server."
            )
            lines.append("")
            lines.append("| Backend | State | Provider | Serving | Window | Vision | Latency |")
            lines.append("|---------|-------|----------|---------|--------|--------|---------|")
            for row in state.backends:
                if not row.get("probed"):
                    state_txt = "not probed"
                elif row.get("ok"):
                    state_txt = "up" + (" (stale)" if row.get("stale") else "")
                else:
                    state_txt = f"DOWN — {row.get('error') or 'unknown error'}"
                served = _short_model(row.get("model_path") or row.get("model") or "") or "—"
                window = f"{int(row['n_ctx']) // 1024}k" if row.get("n_ctx") else "—"
                vision = {True: "yes", False: "no"}.get(row.get("vision"), "?")
                latency = f"{row['latency_ms']:.0f} ms" if row.get("latency_ms") is not None else "—"
                lines.append(
                    f"| {row.get('name')} | {state_txt} | {row.get('provider')} | {served} | {window} | {vision} | {latency} |"
                )
                if row.get("changed_at"):
                    lines.append(f"| | served model changed at {row['changed_at']} | | | | | |")

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
