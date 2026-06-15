"""Gym runner — executes a frozen task set against live Gemma, N runs per task.

Pipeline construction mirrors scripts/smoke_test_tool_calling.py (the proven
shape, same wiring as daemon.py): real provider from config, real ModelAdapter
(tier from config), real ToolRegistry, real SecurityGate, real run_loop.
Telemetry flows through a REAL ToolCallTelemetry instance pointed at gym.db
so induced failures never pollute live dashboards.

Determinism notes:
  - the task set is frozen (sha recorded per row);
  - the manifest changes exactly one variable;
  - each run gets a fresh adapter (adaptive-strictness state) and fresh
    workspace files;
  - the model itself samples — which is WHY runs_per_task ≥ 3 and the report
    refuses thin-sample verdicts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.gym.manifest import ExperimentManifest
from prometheus.gym.scoring import RunTranscript, score
from prometheus.gym.store import GymStore
from prometheus.gym.tasks import TaskSet, TaskSpec
from prometheus.telemetry.tracker import (
    ToolCallTelemetry,
    get_telemetry_handle,
    set_telemetry_handle,
)
from prometheus.tools.base import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

GYM_DB = "~/.prometheus/data/gym.db"
RUN_HARD_TIMEOUT_S = 240.0


# ---------------------------------------------------------------------------
# Pipeline construction (smoke-test pattern)
# ---------------------------------------------------------------------------


def build_pipeline(config: dict) -> dict[str, Any]:
    """Build provider + adapter factory + security gate from live config."""
    from prometheus.__main__ import create_adapter, create_security_gate
    from prometheus.providers.registry import ProviderRegistry

    model_cfg = config.get("model", {})
    provider = ProviderRegistry.create(model_cfg)
    security_gate = create_security_gate(config.get("security", {}))

    def adapter_factory():
        # Fresh per run: ModelAdapter carries adaptive-strictness state.
        return create_adapter(model_cfg, config.get("adapter"))

    return {
        "provider": provider,
        "adapter_factory": adapter_factory,
        "security_gate": security_gate,
        "model_name": model_cfg.get("model", "unknown"),
        "model_cfg": model_cfg,
    }


def preflight_endpoint(config: dict) -> None:
    """Refuse to start a gym run against an unreachable / wrong model endpoint.

    The gitignored ``config/prometheus.yaml`` does NOT travel into ``git
    worktree``s (``_PROMETHEUS_YAML`` resolves relative to the imported module),
    so a stub or default can silently point the runner at the wrong server —
    three wasted N-for-0 runs to date (s1 exp2/exp3, then series-2: 63×404
    against ollama's localhost:11434). The honest-report machinery flagged each
    after the fact; this catches it in one second, BEFORE any task runs, and
    names the resolved endpoint so the misconfiguration is loud, not silent.
    Raises RuntimeError on no-base_url / unreachable / non-200."""
    import httpx

    model_cfg = config.get("model", {})
    base_url = (model_cfg.get("base_url") or "").rstrip("/")
    provider = model_cfg.get("provider", "?")
    if not base_url:
        raise RuntimeError(
            f"gym preflight: no model.base_url in config (provider={provider!r}). "
            "Refusing to run against a silent default endpoint — in a worktree, "
            "copy the real config/prometheus.yaml in (it is gitignored and does "
            "not travel into `git worktree`s)."
        )
    url = f"{base_url}/v1/models"
    try:
        resp = httpx.get(url, timeout=5.0)
    except Exception as exc:  # noqa: BLE001 — any connect/timeout error refuses the run
        raise RuntimeError(
            f"gym preflight: model endpoint {url} is unreachable "
            f"({type(exc).__name__}: {exc}); provider={provider}. Refusing to run "
            "N-for-0 against a dead/wrong endpoint."
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"gym preflight: model endpoint {url} returned HTTP {resp.status_code} "
            f"(provider={provider}). Refusing to run — is the config pointing at "
            "the right server?"
        )
    log.info("gym preflight OK: %s (provider=%s) HTTP 200", url, provider)


def build_registry(workspace: Path, stub_tools: list[str]) -> ToolRegistry:
    """Real builtin tools; stub_tools get side-effect-free execute()."""
    from prometheus.tools.builtin import (
        BashTool,
        FileEditTool,
        FileReadTool,
        FileWriteTool,
        GlobTool,
        GrepTool,
    )
    from prometheus.tools.builtin.cron_list import CronListTool
    from prometheus.tools.builtin.lcm_expand_query import LCMExpandQueryTool
    from prometheus.tools.builtin.task_create import TaskCreateTool
    from prometheus.tools.builtin.task_list import TaskListTool

    registry = ToolRegistry()
    registry.register(BashTool(workspace=str(workspace)))
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GrepTool())
    registry.register(GlobTool())
    registry.register(CronListTool())
    registry.register(TaskListTool())
    registry.register(LCMExpandQueryTool())

    if "task_create" in stub_tools:
        class _GymTaskCreate(TaskCreateTool):
            """Real name/description/schema/validation; execution stubbed.

            The gym must never spawn real background tasks (they would run
            real commands and send real Telegram notifications)."""

            async def execute(self, arguments, context):  # noqa: ANN001
                return ToolResult(
                    output=(
                        f"Started managed task gym-stub ({arguments.type}): "
                        f"{arguments.description}. (gym: side effects stubbed)"
                    ),
                )

        registry.register(_GymTaskCreate())
    else:
        registry.register(TaskCreateTool())

    # Side-effecting tools the gym must never actually run (real network /
    # browser automation). Schema + validation stay real — argshape tasks
    # score the emitted CALL, not the tool output. Registered only when a
    # task asks for them via stub_tools, so the default registry is unchanged.
    if "download_file" in stub_tools:
        from prometheus.tools.builtin.download_file import DownloadFileTool

        class _GymDownloadFile(DownloadFileTool):
            async def execute(self, arguments, context):  # noqa: ANN001
                return ToolResult(output=f"(gym stub) would download {arguments.url}")

        registry.register(_GymDownloadFile())

    if "browser" in stub_tools:
        from prometheus.tools.builtin.browser import BrowserTool

        class _GymBrowser(BrowserTool):
            async def execute(self, arguments, context):  # noqa: ANN001
                return ToolResult(output=f"(gym stub) browser {arguments.action}")

        registry.register(_GymBrowser())

    try:
        from prometheus.tools.tool_search import ToolSearchTool

        ts = ToolSearchTool()
        ts.set_registry(registry)
        registry.register(ts)
    except ImportError:
        pass

    return registry


def _wire_lcm(provider: Any, tmp_dir: Path) -> None:
    """Real LCMEngine on a temp DB (empty is fine — the lcm regression task
    guards against FTS5 syntax errors, not retrieval quality)."""
    try:
        from prometheus.memory.lcm_engine import LCMEngine
        from prometheus.tools.builtin.lcm_grep import set_lcm_engine

        engine = LCMEngine(provider, db_path=tmp_dir / "gym-lcm.db")
        set_lcm_engine(engine)
    except Exception:
        log.exception("gym: LCM engine wiring failed — lcm tasks will error")


# ---------------------------------------------------------------------------
# Variable application (the ONE thing an experiment changes)
# ---------------------------------------------------------------------------


def apply_variable(
    manifest: ExperimentManifest,
    system_prompt: str,
    registry: ToolRegistry,
) -> str:
    """Apply the manifest's single variable. Returns the system prompt to use."""
    name, payload = manifest.variable_name, manifest.variable_payload
    if name == "none":
        return system_prompt
    if name == "system_prompt":
        return payload["text"]
    if name == "example_call":
        example = json.dumps({"name": payload["tool"], "arguments": payload["example"]})
        return (
            f"{system_prompt}\n\n"
            f"Example of a correct {payload['tool']} call:\n{example}"
        )
    if name == "tool_description":
        tool = registry.get(payload["tool"])
        if tool is None:
            raise ValueError(f"tool_description: unknown tool {payload['tool']!r}")
        tool.description = payload["text"]
        return system_prompt
    if name == "tool_error_honesty":
        from prometheus.tools.builtin.task_create import set_honest_mode_errors
        set_honest_mode_errors(True)
        return system_prompt
    if name == "adapter_unwrap":
        # applied per-run on the adapter instance in run_task_once
        return system_prompt
    raise ValueError(f"unimplemented variable {name!r}")


# ---------------------------------------------------------------------------
# Seed-message reconstruction (collapse-arc replay)
# ---------------------------------------------------------------------------


def build_seed_messages(seed: list[dict[str, Any]]) -> list[ConversationMessage]:
    messages: list[ConversationMessage] = []
    last_tool_id: str | None = None
    for step in seed:
        kind, value = next(iter(step.items()))
        if kind == "user":
            messages.append(ConversationMessage.from_user_text(str(value)))
        elif kind == "assistant_text":
            messages.append(
                ConversationMessage(role="assistant", content=[TextBlock(text=str(value))])
            )
        elif kind == "assistant_tool_call":
            last_tool_id = f"toolu_{uuid4().hex[:12]}"
            messages.append(
                ConversationMessage(
                    role="assistant",
                    content=[ToolUseBlock(
                        id=last_tool_id,
                        name=value["name"],
                        input=value.get("input", {}),
                    )],
                )
            )
        elif kind == "tool_result":
            if last_tool_id is None:
                raise ValueError("seed tool_result with no preceding assistant_tool_call")
            messages.append(
                ConversationMessage(
                    role="user",
                    content=[ToolResultBlock(
                        tool_use_id=last_tool_id,
                        content=value.get("content", ""),
                        is_error=bool(value.get("is_error", False)),
                    )],
                    provenance="user",
                    is_trusted=True,
                )
            )
    return messages


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    success: bool               # == execution_pass (the "did it work" axis)
    emission_pass: bool         # raw model emission satisfied the predicates
    execution_pass: bool        # post-adapter executed call satisfied them
    fail_reasons: list[str]     # execution-view failure reasons
    tools_called: list[str]
    latency_ms: float
    retries: int
    repairs: int
    dropped_malformed: int
    feedback_retries: int
    breaker_tripped: bool
    error: str = ""


async def run_task_once(
    pipeline: dict[str, Any],
    task: TaskSpec,
    system_prompt: str,
    workspace: Path,
    gym_tel: ToolCallTelemetry,
    manifest: ExperimentManifest,
) -> RunResult:
    # Fresh workspace files for this run
    for rel, content in task.setup_files.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    registry = build_registry(workspace, task.stub_tools)
    adapter = pipeline["adapter_factory"]()
    if manifest.variable_name == "adapter_unwrap":
        adapter.unwrap_tools = frozenset(manifest.variable_payload["tools"])
    system_prompt = apply_variable(manifest, system_prompt, registry)

    # GBNF grammar — same wiring as smoke/daemon
    provider = pipeline["provider"]
    model_cfg = pipeline["model_cfg"]
    if (
        model_cfg.get("grammar_enforcement", True)
        and hasattr(provider, "set_grammar")
        and adapter is not None
    ):
        grammar = adapter.generate_grammar(registry)
        if grammar:
            provider.set_grammar(grammar)

    t_start = time.time()
    tel_before = gym_tel._conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(retries),0), COALESCE(SUM(repairs),0), "
        "COALESCE(SUM(error_type = 'malformed_empty'),0) FROM tool_calls"
    ).fetchone()

    messages = build_seed_messages(task.seed) if task.seed else []
    messages.append(ConversationMessage.from_user_text(task.prompt))

    # Dual-scoring seam: record the raw-emitted vs actually-executed call per
    # tool_use_id (the repaired input is local to _execute_tool_call and never
    # written back into messages). The executed view is scored from this.
    observed: dict[str, dict[str, Any]] = {}

    def _observe(tool_use_id: str, raw: dict, executed: dict) -> None:
        observed[tool_use_id] = {"raw": raw, "executed": executed}

    context = LoopContext(
        provider=provider,
        model=pipeline["model_name"],
        system_prompt=system_prompt,
        max_tokens=1024,
        tool_registry=registry,
        permission_checker=pipeline["security_gate"],
        adapter=adapter,
        telemetry=gym_tel,
        cwd=workspace,
        max_tool_iterations=10,
        tool_timeout_seconds=45.0,
        session_id="system",
        tool_call_observer=_observe,
    )

    error = ""
    prev_handle = get_telemetry_handle()
    set_telemetry_handle(gym_tel)  # malformed-drop records land in gym.db
    t0 = time.monotonic()
    try:
        async def _drive() -> None:
            async for _ in run_loop(context, messages):
                pass

        await asyncio.wait_for(_drive(), timeout=RUN_HARD_TIMEOUT_S)
    except asyncio.TimeoutError:
        error = f"harness timeout after {RUN_HARD_TIMEOUT_S:.0f}s"
    except Exception as exc:  # harness-level crash — recorded, not raised
        error = f"{type(exc).__name__}: {exc}"
    finally:
        set_telemetry_handle(prev_handle)
    latency_ms = (time.monotonic() - t0) * 1000.0

    tel_after = gym_tel._conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(retries),0), COALESCE(SUM(repairs),0), "
        "COALESCE(SUM(error_type = 'malformed_empty'),0) FROM tool_calls"
    ).fetchone()
    retries = int(tel_after[1] - tel_before[1])
    repairs = int(tel_after[2] - tel_before[2])
    dropped = int(tel_after[3] - tel_before[3])

    transcript = RunTranscript.from_messages(messages, observed)
    transcript.dropped_malformed = dropped
    # timestamp filter not needed: gym_tel is per-experiment and single-threaded

    if error:
        return RunResult(
            success=False,
            emission_pass=False,
            execution_pass=False,
            fail_reasons=[error],
            tools_called=[e.exec_name for e in transcript.tool_events],
            latency_ms=latency_ms,
            retries=retries,
            repairs=repairs,
            dropped_malformed=dropped,
            feedback_retries=transcript.orchestrator_feedback,
            breaker_tripped=transcript.breaker_tripped,
            error=error,
        )

    from prometheus.gym.scoring import EMISSION, EXECUTION
    emit_ok, _emit_reasons = score(task.score, transcript, workspace, view=EMISSION)
    exec_ok, exec_reasons = score(task.score, transcript, workspace, view=EXECUTION)
    return RunResult(
        success=exec_ok,
        emission_pass=emit_ok,
        execution_pass=exec_ok,
        fail_reasons=exec_reasons,
        tools_called=[e.exec_name for e in transcript.tool_events],
        latency_ms=latency_ms,
        retries=retries,
        repairs=repairs,
        dropped_malformed=dropped,
        feedback_retries=transcript.orchestrator_feedback,
        breaker_tripped=transcript.breaker_tripped,
    )


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


async def run_experiment(
    manifest: ExperimentManifest,
    taskset: TaskSet,
    config: dict,
    *,
    store: GymStore | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    preflight_endpoint(config)  # refuse a 0-for-N run against a wrong endpoint
    pipeline = build_pipeline(config)
    store = store or GymStore(GYM_DB)
    gym_tel = ToolCallTelemetry(
        db_path=Path(os.path.expanduser("~/.prometheus/data/gym-telemetry.db"))
    )

    workspace = Path(taskset.workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    _wire_lcm(pipeline["provider"], workspace)

    totals = {"runs": 0, "passed": 0}
    for task in taskset.tasks:
        for run_idx in range(manifest.runs_per_task):
            result = await run_task_once(
                pipeline, task, taskset.system_prompt, workspace, gym_tel, manifest
            )
            totals["runs"] += 1
            totals["passed"] += int(result.success)
            store.record_run(
                series=manifest.series,
                experiment=manifest.experiment,
                task_id=task.id,
                run_idx=run_idx,
                model=pipeline["model_name"],
                category=task.category,
                success=int(result.success),
                emission_pass=int(result.emission_pass),
                execution_pass=int(result.execution_pass),
                fail_reasons="; ".join(result.fail_reasons)[:1000] or None,
                tools_called=json.dumps(result.tools_called),
                latency_ms=result.latency_ms,
                retries=result.retries,
                repairs=result.repairs,
                dropped_malformed=result.dropped_malformed,
                feedback_retries=result.feedback_retries,
                breaker_tripped=int(result.breaker_tripped),
                error=result.error or None,
                manifest_sha=manifest.sha256,
                taskset_sha=taskset.sha256,
            )
            if progress:
                # E/X = emission/execution; only when they differ does the
                # adapter show its value (E✗ X✓ = a repair saved the run).
                icon = "✅" if result.success else "❌"
                ex = (
                    f"E{'✓' if result.emission_pass else '✗'}"
                    f"X{'✓' if result.execution_pass else '✗'}"
                )
                extras = []
                if result.repairs:
                    extras.append(f"repairs={result.repairs}")
                if result.dropped_malformed:
                    extras.append(f"dropped={result.dropped_malformed}")
                if result.breaker_tripped:
                    extras.append("BREAKER")
                print(
                    f"  {icon} {ex} {task.id} run {run_idx + 1}/{manifest.runs_per_task} "
                    f"({result.latency_ms:.0f}ms){' [' + ', '.join(extras) + ']' if extras else ''}"
                )
                if not result.success:
                    print(f"     → {'; '.join(result.fail_reasons)[:160]}")

    if workspace.exists():
        shutil.rmtree(workspace)
    if manifest.variable_name == "tool_error_honesty":
        from prometheus.tools.builtin.task_create import set_honest_mode_errors
        set_honest_mode_errors(False)
    return totals
