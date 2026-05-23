"""Per-tool fault-tolerant registration helper.

Replaces the silently-failable ``try/except: pass`` blocks in
``src/prometheus/__main__.py::create_tool_registry`` flagged by
``docs/audits/ORPHAN-TOOLS-AUDIT.md`` (Phase 1).

Each registration becomes individually catchable, individually failable,
and individually visible to ``/health`` via the ``subsystem_runs`` table
(``subsystem="tool_registration"``, ``operation=<tool_name>``). The schema
is the one introduced in Sprint 4 A2 for autonomous-subsystem liveness —
no new table needed.

Why a separate module instead of adding to ``tools/base.py``:
``base.py`` owns the ``BaseTool`` / ``ToolRegistry`` primitives; this file
owns the *registration policy*. Keeping them apart means the primitives
stay independently testable and a future replacement of the policy
(e.g. a config-driven allowlist) doesn't churn ``base.py``.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from prometheus.telemetry.tracker import ToolCallTelemetry
    from prometheus.tools.base import BaseTool, ToolRegistry

log = logging.getLogger(__name__)


def try_register(
    registry: "ToolRegistry",
    display_name: str,
    module_path: str,
    class_name: str,
    *,
    factory: "Callable[[], BaseTool] | None" = None,
    telemetry: "ToolCallTelemetry | None" = None,
) -> bool:
    """Import ``module_path.class_name``, instantiate, and register.

    The function never raises. Three failure modes are surfaced uniformly:

    1. **Import fails** — module not installed, syntax error in the module,
       or one of its imports fails. Logged at WARN with full traceback;
       ``subsystem_runs`` row written with outcome=``"failed"``.
    2. **Class missing** — module imports but does not expose ``class_name``.
       Same surface.
    3. **Constructor / register raises** — instance can't be built or the
       registry rejects the registration. Same surface.

    Args:
        registry: the :class:`ToolRegistry` instance to register into.
        display_name: human-readable name used in logs and as the
            ``operation`` value in the telemetry row (e.g. ``"GitHubSearchTool"``).
            Distinct from the tool's runtime ``name`` so logs and telemetry
            are readable even when the tool itself decides to rename.
        module_path: import path of the module holding the tool class
            (e.g. ``"prometheus.tools.builtin.skill"``).
        class_name: name of the tool class within that module
            (e.g. ``"SkillTool"``).
        factory: optional callable returning a :class:`BaseTool` instance.
            When given, the class is *still* imported (so "module is broken"
            still fails fast) but the factory builds the instance. Use this
            for tools that need constructor args. Receives no arguments.
        telemetry: optional :class:`ToolCallTelemetry` handle. When omitted,
            falls back to :func:`get_telemetry_handle`; if that's also None,
            the registration still runs but no telemetry row is written.
            Telemetry writes are best-effort and never raise.

    Returns:
        ``True`` if the tool was successfully registered, ``False`` otherwise.
        Callers rarely need this — the WARN log + telemetry row are the
        primary failure surface — but the bool is handy in tests.
    """
    from prometheus.telemetry.tracker import get_telemetry_handle

    if telemetry is None:
        telemetry = get_telemetry_handle()

    started = time.time()
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = factory() if factory is not None else cls()
        registry.register(instance)
    except Exception as exc:
        duration_ms = (time.time() - started) * 1000.0
        log.warning(
            "tool_registration: %s FAILED — %s: %s",
            display_name, type(exc).__name__, exc,
            exc_info=True,
        )
        _record_run(
            telemetry, display_name, "failed", duration_ms,
            {
                "module_path": module_path,
                "class_name": class_name,
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc)[:500],
            },
        )
        return False

    duration_ms = (time.time() - started) * 1000.0
    log.info("tool_registration: %s registered", display_name)
    _record_run(
        telemetry, display_name, "success", duration_ms,
        {"module_path": module_path, "class_name": class_name},
    )
    return True


def _record_run(
    telemetry: "ToolCallTelemetry | None",
    tool_name: str,
    outcome: str,
    duration_ms: float,
    summary: dict[str, Any],
) -> None:
    """Best-effort write to ``subsystem_runs``. Never raises."""
    if telemetry is None:
        return
    try:
        telemetry.record_run(
            subsystem="tool_registration",
            operation=tool_name,
            outcome=outcome,
            duration_ms=duration_ms,
            summary=summary,
        )
    except Exception:
        log.debug(
            "tool_registration: telemetry write failed for %s",
            tool_name, exc_info=True,
        )
