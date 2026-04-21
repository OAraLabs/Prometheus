"""AnatomyTool — query and manage infrastructure configuration."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

# Module-level singletons set by daemon.py wiring
_scanner: object | None = None
_writer: object | None = None
_project_store: object | None = None


def set_anatomy_components(
    scanner: object, writer: object, project_store: object
) -> None:
    """Wire anatomy components from daemon startup."""
    global _scanner, _writer, _project_store
    _scanner = scanner
    _writer = writer
    _project_store = project_store


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

    def is_read_only(self, arguments: BaseModel) -> bool:
        action = getattr(arguments, "action", "status")
        return action != "switch"

    async def execute(self, arguments: AnatomyInput, context: ToolExecutionContext) -> ToolResult:
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
            summaries = store.summaries() if store else []
            writer.write(state, summaries)
            return ToolResult(output=f"Infrastructure scan complete.\n\n{writer.render_summary(state)}")

        if arguments.action == "status":
            state = await scanner.quick_scan()
            return ToolResult(output=writer.render_summary(state))

        if arguments.action == "projects":
            if store is None:
                return ToolResult(output="Project config store not available.")
            projects = store.list_projects()
            if not projects:
                return ToolResult(output="No project configurations found.")
            lines: list[str] = []
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
