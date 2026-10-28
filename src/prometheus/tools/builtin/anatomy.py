"""AnatomyTool — query and manage infrastructure configuration."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

# Module-level singletons set by daemon.py wiring
_scanner: object | None = None
_writer: object | None = None
_project_store: object | None = None
# The config's anatomy/host facts, for infra/model_view. None = not wired: the
# tool then withholds addresses, the safe reading.
_config: dict[str, Any] | None = None


def set_anatomy_components(
    scanner: object, writer: object, project_store: object,
    config: dict[str, Any] | None = None,
) -> None:
    """Wire anatomy components from daemon startup."""
    global _scanner, _writer, _project_store, _config
    _scanner = scanner
    _writer = writer
    _project_store = project_store
    _config = config


def _scanned_names(state: object | None) -> set[str]:
    """The machine names a scan saw: the inference host and each tailnet peer."""
    if state is None:
        return set()
    names = {getattr(state, "gpu_inference_host", None) or ""}
    for peer in getattr(state, "tailscale_peers", None) or []:
        if isinstance(peer, dict):
            names.add(str(peer.get("name") or ""))
        else:
            names.add(str(peer))
    return {n for n in names if n}


AnatomyAction = Literal["scan", "status", "projects", "switch", "diagram", "history"]


class AnatomyInput(BaseModel):
    action: AnatomyAction = Field(
        description=(
            "Action: 'scan' (full infra scan), 'status' (quick model+VRAM), "
            "'projects' (list configs), 'switch' (switch config), "
            "'diagram' (Mermaid architecture), 'history' (resource history)"
        )
    )
    project_name: str | None = Field(
        default=None,
        description="Project config name (for 'switch' action).",
    )


class AnatomyTool(BaseTool):
    """Query and manage infrastructure configuration."""

    name = "anatomy"
    description = (
        "Query Prometheus infrastructure state — hardware, loaded model, "
        "VRAM, services, project configurations, and architecture diagrams."
    )
    input_model = AnatomyInput
    find_me = ("show the hardware and the loaded model", "check GPU VRAM use")

    def is_read_only(self, arguments: BaseModel) -> bool:
        action = getattr(arguments, "action", "status")
        return action != "switch"

    async def execute(self, arguments: AnatomyInput, context: ToolExecutionContext) -> ToolResult:
        """Every answer goes to the model, so it passes through infra/model_view:
        no hosts, addresses, URLs or peer list unless anatomy.include_addresses
        is a literal true. The owner's /anatomy command is a separate renderer."""
        from prometheus.infra.model_view import for_model

        seen: set[str] = set()
        result = await self._run(arguments, seen)
        return dataclasses.replace(result, output=for_model(result.output, _config, extra_names=seen))

    async def _run(self, arguments: AnatomyInput, seen: set[str]) -> ToolResult:
        """The answer. Adds to *seen* every machine name it rendered, from the
        scan or from a project config, so the caller can withhold them."""
        if _scanner is None or _writer is None:
            return ToolResult(
                output="Anatomy system not initialized. Is the daemon running?",
                is_error=True,
            )

        from prometheus.infra.anatomy import AnatomyScanner
        from prometheus.infra.anatomy_writer import AnatomyWriter
        from prometheus.infra.project_configs import ProjectConfigStore

        scanner: AnatomyScanner = _scanner  # type: ignore[assignment]
        writer: AnatomyWriter = _writer  # type: ignore[assignment]
        store: ProjectConfigStore | None = _project_store  # type: ignore[assignment]

        if arguments.action == "scan":
            state = await scanner.scan()
            seen |= _scanned_names(state)
            summaries = store.summaries() if store else []
            writer.write(state, summaries)
            return ToolResult(output=f"Infrastructure scan complete.\n\n{writer.render_summary(state)}")

        if arguments.action == "status":
            state = await scanner.quick_scan()
            seen |= _scanned_names(state)
            return ToolResult(output=writer.render_summary(state))

        if arguments.action == "projects":
            if store is None:
                return ToolResult(output="Project config store not available.")
            projects = store.list_projects()
            if not projects:
                return ToolResult(output="No project configurations found.")
            lines: list[str] = []
            seen.update(str(m.machine) for p in projects for m in (p.models or []) if getattr(m, "machine", None))
            for p in projects:
                active_tag = " [ACTIVE]" if p.active else ""
                lines.append(f"**{p.name}**{active_tag} — {p.description}")
                if p.models:
                    for m in p.models:
                        lines.append(f"  - {m.name} ({m.role}, {m.engine}, ~{m.vram_estimate_gb}GB)")
                if p.notes:
                    lines.append(f"  Note: {p.notes}")
            return ToolResult(output="\n".join(lines))

        if arguments.action == "switch":
            if not arguments.project_name:
                return ToolResult(output="Provide project_name to switch.", is_error=True)
            if store is None:
                return ToolResult(output="Project config store not available.", is_error=True)
            cfg = store.get(arguments.project_name)
            if cfg is None:
                names = [p.name for p in store.list_projects()]
                return ToolResult(
                    output=f"Project '{arguments.project_name}' not found. Available: {', '.join(names)}",
                    is_error=True,
                )
            # Don't auto-execute — report what would need to change
            lines = [f"To switch to **{cfg.name}** ({cfg.description}):"]
            seen.update(str(m.machine) for m in cfg.models if getattr(m, "machine", None))
            for m in cfg.models:
                lines.append(f"- Load {m.name} on {m.machine} via {m.engine} (port {m.port}, ~{m.vram_estimate_gb}GB)")
                if m.extra_flags:
                    lines.append(f"  Flags: {' '.join(m.extra_flags)}")
            if cfg.notes:
                lines.append(f"\nNote: {cfg.notes}")
            lines.append("\nThis requires manual model loading or approval. Mark active after switching.")
            store.activate(cfg.name)
            return ToolResult(output="\n".join(lines))

        if arguments.action == "diagram":
            state = await scanner.quick_scan()
            seen |= _scanned_names(state)
            mermaid = writer.render_mermaid(state)
            return ToolResult(output=f"```mermaid\n{mermaid}\n```")

        if arguments.action == "history":
            from prometheus.config.paths import get_config_dir
            history_path = get_config_dir() / "anatomy" / "history.jsonl"
            if not history_path.exists():
                return ToolResult(output="No resource history recorded yet.")
            lines = history_path.read_text(encoding="utf-8").strip().splitlines()
            recent = lines[-10:] if len(lines) > 10 else lines
            return ToolResult(output="\n".join(recent))

        return ToolResult(output=f"Unknown action: {arguments.action}", is_error=True)
