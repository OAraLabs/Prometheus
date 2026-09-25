"""Run the ladder suite against one served model and record every run.

The pipeline is the gym's (``gym.runner.build_pipeline``): the real provider,
the real ModelAdapter at the tier the daemon would pick for this model, the
real SecurityGate and the real agent loop. What the ladder adds:

* a fresh sandbox per run (``fixtures.Sandbox``) and one fixed tool surface;
* per-run ``session_id`` so the loop's own telemetry rows join to the run;
* probes that RECORD what was actually served — model, quantization, KV
  cache — and never guess them;
* the judge pin: a judged task is graded by a pinned local model that is
  provably not the contestant.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.gym.ladder.fixtures import (
    Sandbox,
    build_ladder_registry,
    memory_prompt_section,
    seed_task,
)
from prometheus.gym.ladder.record import (
    final_round_used_reasoning_fallback,
    harvest_run_metrics,
    record_summary,
)
from prometheus.gym.ladder.suite import LadderSuite, LadderTask
from prometheus.gym.ladder.verdict import ERROR, Verdict, decide
from prometheus.gym.runner import (
    _probe_kv_cache,
    build_pipeline,
    build_seed_messages,
    preflight_endpoint,
)
from prometheus.gym.scoring import RunTranscript
from prometheus.telemetry.tracker import (
    ToolCallTelemetry,
    get_telemetry_handle,
    set_telemetry_handle,
)

log = logging.getLogger(__name__)

DEFAULT_TELEMETRY_DB = "~/.prometheus/ladder/telemetry.db"
TOOL_TIMEOUT_CAP_S = 120.0
# What run_loop raises when max_turns runs out (engine/agent_loop.py).
_ROUND_CAP_MESSAGE = "Exceeded maximum turn limit"

# The messages the agent LOOP writes as the turn's last assistant message when
# it stops a turn itself (engine/agent_loop.py, _make_assistant_msg call
# sites). They carry no provenance marker, so they are recognised by their
# fixed opening words — pinned against the engine source by
# tests/test_ladder.py so a reworded message fails a test instead of being
# silently scored as the model's answer. Engine code is out of scope for the
# ladder; marking these messages at the source is the durable fix.
LOOP_HALTS: tuple[tuple[str, str], ...] = (
    ("Tool iteration limit reached (", "tool_call_cap"),
    ("The model returned an empty response twice", "empty_response"),
    ("Circuit breaker tripped:", "circuit_breaker"),
    ("\u26a0\ufe0f Tool call failed ", "circuit_breaker"),
    ("Halted: no progress.", "repeat_halt"),
    ("Halted: the same tool has been running repeatedly", "divergence_halt"),
    ("TURN ENDED \u2014 ", "boundary_escape"),
    ("I emitted a tool call that could not be parsed", "parse_disagreement"),
    ("This conversation is too long for the local model", "context_overflow"),
)

# The loop's repeat guard answers a call that already failed twice with this,
# and writes NO tool_calls row for it.
REPEAT_BLOCKED_PREFIX = "BLOCKED: this exact "


class LadderAbort(RuntimeError):
    """Stop the whole ladder run: continuing would record noise (a dead
    endpoint, a sandbox that cannot be cleaned)."""


def loop_halt_kind(msg: ConversationMessage) -> str | None:
    """The halt kind if *msg* is a message the loop wrote itself, else None."""
    if msg.role != "assistant" or len(msg.content) != 1:
        return None
    block = msg.content[0]
    text = getattr(block, "text", None)
    if not isinstance(block, TextBlock) or not isinstance(text, str):
        return None
    for prefix, kind in LOOP_HALTS:
        if text.startswith(prefix):
            return kind
    return None


def count_tool_uses(messages: list[ConversationMessage]) -> int:
    return sum(isinstance(b, ToolUseBlock) for m in messages if m.role == "assistant"
               for b in m.content)


_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s'\"<>)]+", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")


def redact(text: str | None) -> str | None:
    """Hosts out of anything that persists: provider and judge errors quote
    the full request URL (httpx), and telemetry.db and the committed report
    must not carry an infrastructure identifier."""
    if not text:
        return text
    return _IPV4_RE.sub("<ip>", _URL_RE.sub("<url>", text))


async def endpoint_alive(pipeline: dict[str, Any], timeout_s: float = 45.0) -> tuple[bool, str]:
    """A one-token completion against the contestant. ``/health`` and
    ``/v1/models`` keep answering while a llama-server slot is wedged, so only
    a real generation proves the endpoint can still serve."""
    import httpx

    cfg = pipeline["model_cfg"]
    base = str(cfg.get("base_url", "")).rstrip("/")
    url = f"{base}/chat/completions" if re.search(r"/v\d+[a-z0-9]*$", base) else f"{base}/v1/chat/completions"
    payload = {"model": pipeline["model_name"], "max_tokens": 1, "stream": False,
               "messages": [{"role": "user", "content": "ok"}]}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, json=payload)
        return (r.status_code < 500, f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        return (False, type(exc).__name__)
LADDER_RECORD_VERSION = 1

# llama.cpp/HF quant tokens as they appear in GGUF file names.
_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4))(?=[.\-_]|$)",
    re.IGNORECASE,
)


class LadderPreflightError(RuntimeError):
    """Refused before any task ran — the run would not measure what it claims."""


@dataclass
class Contestant:
    provider: str
    base_url: str
    model: str = ""           # blank = ask the backend (llama.cpp serves one)
    quantization: str = ""    # declared; the probe's answer wins when it has one
    timeout: float = 300.0


@dataclass
class JudgePin:
    base_url: str
    model: str


# ---------------------------------------------------------------------------
# Probes — record what is served, never guess it
# ---------------------------------------------------------------------------


def quant_from_filename(name: str) -> str | None:
    base = os.path.basename(name or "")
    m = None
    for m in _QUANT_RE.finditer(base):
        pass  # the LAST quant-looking token is the file's quant
    return m.group(1).upper() if m else None


async def probe_identity(provider: str, base_url: str, model: str) -> dict[str, Any]:
    """What the endpoint says it is serving. Unknowns stay unknown."""
    import httpx

    base = base_url.rstrip("/")
    out: dict[str, Any] = {"served_model": None, "quantization": None,
                           "quantization_source": "unreported", "parameter_size": None}
    async with httpx.AsyncClient(timeout=10.0) as client:
        if provider == "ollama":
            r = await client.post(f"{base}/api/show", json={"model": model})
            if r.status_code == 404:
                raise LadderPreflightError(f"ollama does not have model {model!r}")
            r.raise_for_status()
            details = r.json().get("details") or {}
            out["served_model"] = model
            out["parameter_size"] = details.get("parameter_size")
            if details.get("quantization_level"):
                out["quantization"] = str(details["quantization_level"]).upper()
                out["quantization_source"] = "ollama:/api/show"
            return out
        r = await client.get(f"{base}/v1/models")
        r.raise_for_status()
        ids = [m.get("id", "") for m in r.json().get("data", [])]
        if provider == "llama_cpp":
            try:
                props = (await client.get(f"{base}/props")).json()
            except Exception:  # noqa: BLE001 — /props is optional
                props = {}
            path = props.get("model_path") or (ids[0] if ids else "")
            out["served_model"] = os.path.basename(path) or None
            q = quant_from_filename(path)
            if q:
                out["quantization"] = q
                out["quantization_source"] = "gguf-filename"
        else:
            out["served_model"] = model if model in ids else (ids[0] if len(ids) == 1 else None)
    return out


def _norm_model(name: str | None) -> str:
    n = os.path.basename((name or "").strip()).lower()
    n = re.sub(r"\.gguf$", "", n)
    return re.sub(r":latest$", "", n)


async def served_ids(base_url: str) -> list[str]:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{base_url.rstrip('/')}/v1/models")
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]


async def check_judge_pin(
    judge: JudgePin,
    *,
    contestant_model: str,
    contestant_served: str | None,
) -> None:
    """Refuse a judge that is unpinned, is the contestant, or is not served.

    The third check matters for llama.cpp: llama-server ignores the request's
    ``model`` field and answers with whatever it loaded, so a pin that names
    one model while the endpoint serves another would record a judge that
    never graded anything.
    """
    if not judge.model:
        raise LadderPreflightError("the judge must be pinned to a model (--judge-model)")
    pin = _norm_model(judge.model)
    for name in (contestant_model, contestant_served):
        if name and pin == _norm_model(name):
            raise LadderPreflightError(
                f"judge {judge.model!r} is the model under test — nothing grades itself"
            )
    try:
        ids = await served_ids(judge.base_url)
    except Exception as exc:  # noqa: BLE001
        raise LadderPreflightError(f"judge endpoint unreachable: {type(exc).__name__}: {exc}") from exc
    if not any(pin == _norm_model(i) for i in ids):
        raise LadderPreflightError(
            f"judge endpoint does not serve the pinned model {judge.model!r} "
            f"(it serves {[_norm_model(i) for i in ids]})"
        )


def harness_commit() -> str | None:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip() or None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src/prometheus/gym",
             "src/prometheus/engine", "src/prometheus/adapter", "scripts/ladder_run.py"],
            cwd=here.parents[3], capture_output=True, text=True, timeout=5,
        )
        if sha and dirty.returncode == 0 and dirty.stdout.strip():
            # Rows from modified or untracked harness code must not claim a
            # commit that does not contain it.
            sha += "-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return None


def bash_write_floor() -> str:
    """Whether bash's kernel write floor is in force on this machine.

    BashTool's default is ``auto``: enforced where bubblewrap works (Linux),
    unconfined where it does not (macOS). A correct ladder solution writes
    only inside the workspace, so verdicts do not depend on it — but rows
    from two platforms must say which regime they ran under.
    """
    try:
        from prometheus.permissions import confinement

        ok, _detail = confinement.write_preflight()
        return "active" if ok else "unavailable"
    except Exception:  # noqa: BLE001 — a probe never fails a run
        return "unknown"


def ladder_config(c: Contestant, workspace: Path) -> dict[str, Any]:
    """The whole config a ladder run uses — built here, not read from the
    machine's prometheus.yaml, so two boxes running the same rung run the
    same pipeline."""
    return {
        "model": {
            "provider": c.provider,
            "base_url": c.base_url,
            "model": c.model,
            "timeout": c.timeout,
            "grammar_enforcement": True,
        },
        "security": {"workspace_root": str(workspace)},
        "adapter": {},
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def expand(text: str, workspace: Path) -> str:
    """Tasks name files as ``{workspace}/...`` — absolute paths, as the gym's
    prompts do, without pinning the workdir. The SecurityGate treats a
    RELATIVE write path as unknown (it asks for approval, and a ladder run has
    no one to approve), so a task that only ever said "total.txt" would be
    measuring the gate, not the model."""
    return text.replace("{workspace}", str(workspace))


def expand_deep(value: Any, workspace: Path) -> Any:
    """``expand`` through seeds, whose tool calls and results nest dicts."""
    if isinstance(value, str):
        return expand(value, workspace)
    if isinstance(value, list):
        return [expand_deep(v, workspace) for v in value]
    if isinstance(value, dict):
        return {k: expand_deep(v, workspace) for k, v in value.items()}
    return value


def trace_of(t: RunTranscript, limit: int = 60) -> list[dict[str, Any]]:
    """A compact, bounded record of what the model did — enough to read a
    failure back out of telemetry.db without the transcript."""
    import json as _json

    out = []
    for e in t.tool_events[:limit]:
        try:
            args = _json.dumps(e.exec_input, default=str)
        except (TypeError, ValueError):
            args = str(e.exec_input)
        out.append({
            "tool": e.exec_name,
            "args": args[:200],
            "is_error": e.is_error,
            "repaired": e.repaired,
            "result": (e.result_content or "")[:200],
        })
    return out


def compose_system_prompt(base: str, workspace: Path) -> str:
    parts = [base.strip(), f"Your working directory is {workspace}. Relative paths resolve there."]
    memory = memory_prompt_section()
    if memory:
        parts.append(memory.strip())
    return "\n\n".join(parts)


async def run_task(
    task: LadderTask,
    suite: LadderSuite,
    pipeline: dict[str, Any],
    *,
    sandbox: Sandbox,
    tel: ToolCallTelemetry,
    judge: Any,
    run_label: str,
    run_idx: int,
    static: dict[str, Any],
    attribute_sessionless: bool = True,
) -> dict[str, Any]:
    try:
        sandbox.reset()
    except Exception as exc:  # noqa: BLE001
        raise LadderAbort(f"cannot give {task.id} a clean sandbox: {exc}") from exc
    provider = pipeline["provider"]
    lcm = seed_task(task, sandbox, provider)
    registry = build_ladder_registry(
        sandbox.workspace,
        web_pages=task.fixtures.get("web_pages"),
        live_web=task.web == "live",
    )
    adapter = pipeline["adapter_factory"]()
    if (
        pipeline["model_cfg"].get("grammar_enforcement", True)
        and hasattr(provider, "set_grammar")
        and adapter is not None
    ):
        grammar = adapter.generate_grammar(registry)
        if grammar:
            provider.set_grammar(grammar)
            if hasattr(provider, "set_grammar_source"):
                provider.set_grammar_source(adapter.enforcer, registry.to_api_schema())

    seed = expand_deep(task.seed, sandbox.workspace)
    messages = build_seed_messages(seed) if seed else []
    messages.append(ConversationMessage.from_user_text(expand(task.prompt, sandbox.workspace)))
    # Everything before this index was PUT there — seed and prompt. The
    # verdict reads only what the model produced after it.
    run_start = len(messages)
    session_id = f"ladder:{run_label}:{task.id}:{run_idx}:{uuid4().hex[:8]}"
    observed: dict[str, dict[str, Any]] = {}

    def _observe(tool_use_id: str, raw: dict, executed: dict) -> None:
        observed[tool_use_id] = {"raw": raw, "executed": executed}

    context = LoopContext(
        provider=provider,
        model=pipeline["model_name"],
        system_prompt=compose_system_prompt(suite.system_prompt, sandbox.workspace),
        max_tokens=task.max_tokens,
        tool_registry=registry,
        permission_checker=pipeline["security_gate"],
        adapter=adapter,
        telemetry=tel,
        cwd=sandbox.workspace,
        max_turns=task.max_rounds,
        max_tool_iterations=task.max_tool_calls,
        tool_timeout_seconds=min(TOOL_TIMEOUT_CAP_S, task.timeout_s),
        # "system" keeps the router's per-session overrides out (the reserved
        # id), exactly as the gym does; the per-run id below is what the
        # telemetry rows carry.
        session_id="system",
        tool_call_observer=_observe,
    )

    error, halted, stopped_by = "", "", "done"
    prev = get_telemetry_handle()
    set_telemetry_handle(tel)
    wall_start = time.time()
    t0 = time.monotonic()
    try:
        async def _drive() -> None:
            async for _ in run_loop(context, messages, session_id=session_id):
                pass

        await asyncio.wait_for(_drive(), timeout=task.timeout_s)
    except asyncio.TimeoutError:
        stopped_by = "timeout"
        halted = f"time budget exceeded ({task.timeout_s:.0f}s)"
    except RuntimeError as exc:
        if str(exc).startswith(_ROUND_CAP_MESSAGE):
            # The loop RAISES when max_turns runs out: the model did not
            # finish within its round budget — a fail, not a harness crash.
            stopped_by = "round_cap"
            halted = f"round budget exhausted ({task.max_rounds} rounds)"
        else:
            stopped_by, error = "error", f"{type(exc).__name__}: {exc}"
            log.warning("ladder run %s crashed: %s", task.id, redact(error))
    except Exception as exc:  # noqa: BLE001 — a crash is recorded, not raised
        stopped_by, error = "error", f"{type(exc).__name__}: {exc}"
        log.warning("ladder run %s crashed: %s", task.id, redact(error))
    finally:
        set_telemetry_handle(prev)
    duration_ms = (time.monotonic() - t0) * 1000.0
    window = (wall_start, time.time())

    abort: str | None = None
    if stopped_by == "timeout":
        # A slow model and a wedged endpoint look the same from here. Ask the
        # endpoint for one token: if it cannot answer, this row is not the
        # model's to own, and neither would any later one be.
        alive, detail = await endpoint_alive(pipeline)
        if not alive:
            stopped_by, halted = "error", ""
            error = f"endpoint unresponsive after the time budget ({detail})"
            abort = error

    # Scored over the RUN's messages only — seeded turns are context, not the
    # model's answer — and without the messages the loop wrote itself when it
    # stopped the turn: those are not the model's answer either.
    run_messages = messages[run_start:]
    halt_kinds = [k for k in (loop_halt_kind(m) for m in run_messages) if k]
    model_messages = [m for m in run_messages if not loop_halt_kind(m)]
    transcript = RunTranscript.from_messages(model_messages, observed)
    tool_uses = count_tool_uses(run_messages)
    if stopped_by == "done":
        if tool_uses > task.max_tool_calls or "tool_call_cap" in halt_kinds:
            stopped_by = "tool_call_cap"
            halted = f"tool-call budget exhausted ({task.max_tool_calls} calls)"
        elif halt_kinds:
            stopped_by = halt_kinds[-1]
            halted = f"the loop stopped the turn ({halt_kinds[-1]})"
    if stopped_by == "done" and final_round_used_reasoning_fallback(
        tel._conn, session_id, window
    ):
        # The provider returned the unfinished reasoning as the reply; a
        # right value mentioned along the way is not an answer.
        stopped_by = "reasoning_fallback"
        halted = "no answer: the final round's output budget went to reasoning"

    try:
        verdict = await decide(
            task, transcript, sandbox.workspace, judge=judge,
            harness_error=error, halted=halted, home=sandbox.home,
            prompt=expand(task.prompt, sandbox.workspace),
        )
    except Exception as exc:  # noqa: BLE001 — a verdict bug must not end the ladder
        log.warning("ladder verdict for %s raised: %s", task.id, exc, exc_info=True)
        verdict = Verdict(ERROR, None, None, task.verdict_source,
                          [f"harness error while deciding: {type(exc).__name__}: {exc}"])
    metrics = harvest_run_metrics(
        tel._conn, session_id, window=window if attribute_sessionless else None,
        transcript=transcript,
    )
    try:
        lcm.close()
    except Exception:  # noqa: BLE001
        log.debug("ladder: LCM close failed", exc_info=True)

    final = transcript.final_text or ""
    summary = {
        **static,
        "task_id": task.id,
        "task_class": task.task_class,
        "difficulty": task.difficulty,
        "run_idx": run_idx,
        "session_id": session_id,
        "verdict": verdict.verdict,
        "verdict_source": verdict.source,
        "success": verdict.success,
        "emission_pass": verdict.emission_pass,
        "answer_format_ok": verdict.answer_format_ok,
        "fail_reasons": [redact(r) for r in verdict.fail_reasons],
        "acceptance": _redact_acceptance(verdict.acceptance),
        "judge": verdict.judge,
        "error": redact(error) or None,
        "stopped_by": stopped_by,
        "loop_halts": halt_kinds,
        "duration_ms": round(duration_ms, 1),
        "web": task.web,
        "answer_changes_over_time": task.answer_changes_over_time,
        "budget": {"max_rounds": task.max_rounds, "max_tool_calls": task.max_tool_calls,
                   "max_tokens": task.max_tokens, "timeout_s": task.timeout_s},
        "tool_uses_emitted": tool_uses,
        "tools_called": [e.exec_name for e in transcript.tool_events],
        "trace": trace_of(transcript),
        "final_text_head": final[:300],
        # Whole, so a verdict can be audited against what the reader read (the
        # END of the reply). Local DB only; reports never render it.
        "final_text": final,
        **metrics,
    }
    record_summary(tel, summary)
    if abort:
        raise LadderAbort(abort)
    return summary


def _redact_acceptance(acc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not acc:
        return acc
    return {k: (redact(v) if isinstance(v, str) else v) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------


async def run_ladder(
    suite: LadderSuite,
    tasks: list[LadderTask],
    contestant: Contestant,
    *,
    run_label: str,
    judge_pin: JudgePin | None,
    telemetry_db: str | Path = DEFAULT_TELEMETRY_DB,
    runs_per_task: int = 1,
    workdir: str | Path | None = None,
    rung: str | None = None,
    expect_model_match: str | None = None,
    expect_adapter_tier: str | None = None,
    strict_quant: bool = False,
    progress: bool = True,
) -> list[dict[str, Any]]:
    if not tasks:
        raise LadderPreflightError("no tasks selected")
    from prometheus.config.paths import config_dir_path
    from prometheus.gym.ladder.fixtures import SandboxError

    # Resolved BEFORE the sandbox relocates the config dir.
    db = Path(os.path.expanduser(str(telemetry_db))).resolve()
    live_db = (config_dir_path() / "telemetry.db").resolve()
    attribute_sessionless = db != live_db
    # Resolved, so the gate, the tools and the prompt all name one spelling
    # of the path (macOS: /tmp is a symlink to /private/tmp).
    sandbox = Sandbox(Path(workdir or suite.workspace).expanduser().resolve())
    db_lock = None
    previous_env = sandbox.activate()
    try:
        try:
            sandbox.lock()
            db_lock = _lock_db(db)
            sandbox.reset()
        except SandboxError as exc:
            raise LadderPreflightError(str(exc)) from exc
        config = ladder_config(contestant, sandbox.workspace)
        try:
            preflight_endpoint(config)
        except RuntimeError as exc:
            raise LadderPreflightError(redact(str(exc)) or "endpoint preflight failed") from exc
        identity = await probe_identity(contestant.provider, contestant.base_url, contestant.model)
        model_name = contestant.model or identity["served_model"] or ""
        if not model_name:
            raise LadderPreflightError("could not determine the model name; pass --model")
        config["model"]["model"] = model_name
        if expect_model_match:
            # What the endpoint SAYS it serves decides. llama-server ignores
            # the request's model field, so the operator's --model string
            # proves nothing when a served name is available.
            served = identity["served_model"]
            candidate = _norm_model(served) if served else _norm_model(model_name)
            if not re.search(expect_model_match, candidate, re.IGNORECASE):
                raise LadderPreflightError(
                    f"rung {rung!r} expects a model matching {expect_model_match!r}, "
                    f"but the endpoint serves {served or model_name!r} — refusing to "
                    f"record rows under the wrong rung"
                )

        quant, quant_source = identity["quantization"], identity["quantization_source"]
        declared = contestant.quantization.upper() or None
        if strict_quant:
            if not quant:
                raise LadderPreflightError(
                    f"rung {rung!r} needs a probed quantization and the endpoint does not "
                    f"report one ({contestant.provider}) — a declared value is not evidence"
                )
            if declared and declared != quant:
                raise LadderPreflightError(
                    f"rung {rung!r} is {declared} but the endpoint serves {quant} "
                    f"({quant_source}) — refusing to file rows under the wrong quantization"
                )
        elif declared and quant and declared != quant:
            log.warning("declared quantization %s disagrees with the probe (%s: %s) — "
                        "recording the probe", declared, quant_source, quant)
        if not quant and declared:
            quant, quant_source = declared, "declared"

        pipeline = build_pipeline(config)
        # As the daemon does: cron_create vets commands through the SAME gate
        # the agent's calls go through, not a lazily-built default read from
        # whatever prometheus.yaml the config dir happens to hold.
        from prometheus.gateway.cron_scheduler import set_cron_security_gate

        set_cron_security_gate(pipeline["security_gate"])
        probe_adapter = pipeline["adapter_factory"]()
        if expect_adapter_tier and probe_adapter.tier != expect_adapter_tier:
            raise LadderPreflightError(
                f"rung {rung!r} expects adapter tier {expect_adapter_tier!r} but the daemon's "
                f"selection gives {probe_adapter.tier!r} for {model_name!r} — "
                f"config/model_registry.yaml and the rung disagree"
            )
        kv = await _probe_kv_cache(pipeline["provider"])
        bash_floor = bash_write_floor()
        thinking = await _probe_thinking(pipeline["provider"])
        if rung and thinking["status"] == "unsupported":
            raise LadderPreflightError(
                f"thinking suppression is requested but the served template ignores it "
                f"({thinking['detail']}) — reasoning would eat the output budget; refusing "
                f"to file rows under rung {rung!r}"
            )

        judge = None
        if judge_pin is not None:
            await check_judge_pin(judge_pin, contestant_model=model_name,
                                  contestant_served=identity["served_model"])
            from prometheus.evals.judge import PrometheusJudge

            judge = PrometheusJudge(base_url=judge_pin.base_url, model=judge_pin.model)
        elif any(t.judge for t in tasks):
            log.warning("no judge pinned — %d judged task(s) will be recorded UNSCORED",
                        sum(1 for t in tasks if t.judge))

        static = {
            "ladder_record_version": LADDER_RECORD_VERSION,
            "suite": suite.name,
            "suite_version": suite.version,
            "suite_sha": suite.sha256,
            "harness_commit": harness_commit(),
            "run_label": run_label,
            "rung": rung,
            "provider": contestant.provider,
            "model": model_name,
            "served_model": identity["served_model"],
            "parameter_size": identity["parameter_size"],
            "quantization": quant,
            "quantization_source": quant_source if quant else "unreported",
            "quantization_declared": declared,
            "adapter_tier": probe_adapter.tier,
            "adapter_strictness": probe_adapter._base_strictness.value,
            "kv_cache": {"k": kv.get("k"), "v": kv.get("v"), "source": kv.get("source")},
            "thinking_suppression": thinking["status"],
            "bash_write_floor": bash_floor,
            "judge_model": judge_pin.model if judge_pin else None,
            "live_web": any(t.web == "live" for t in tasks),
            "sessionless_attribution": "time-window" if attribute_sessionless
            else "off (live telemetry.db)",
        }

        tel = ToolCallTelemetry(db_path=db)
        if progress:
            print(f"  model {model_name} ({contestant.provider}), quant {quant or 'unreported'}"
                  f" [{static['quantization_source']}], adapter {static['adapter_tier']}/"
                  f"{static['adapter_strictness']}, thinking suppression "
                  f"{thinking['status']}, judge {static['judge_model'] or 'none'}")
            print(f"  {len(tasks)} task(s) × {runs_per_task} → {db}")

        rows: list[dict[str, Any]] = []
        try:
            for task in tasks:
                for run_idx in range(runs_per_task):
                    row = await run_task(
                        task, suite, pipeline, sandbox=sandbox, tel=tel, judge=judge,
                        run_label=run_label, run_idx=run_idx, static=static,
                        attribute_sessionless=attribute_sessionless,
                    )
                    rows.append(row)
                    if progress:
                        icon = {"pass": "✅", "fail": "❌", "unscored": "❔",
                                "error": "💥"}[row["verdict"]]
                        print(f"  {icon} {task.task_class:13s} {task.id:28s} "
                              f"rounds={row['rounds']} calls={row['tool_calls_ok']}/"
                              f"{row['tool_calls']} repairs={row['repairs']} "
                              f"{row['duration_ms'] / 1000:.1f}s")
                        if row["verdict"] != "pass":
                            print(f"       → {'; '.join(row['fail_reasons'])[:200]}")
        finally:
            tel.close()
        return rows
    finally:
        from prometheus.gateway.cron_scheduler import set_cron_security_gate

        set_cron_security_gate(None)
        if db_lock is not None:
            db_lock.close()
        sandbox.unlock()
        Sandbox.restore(previous_env)


def _lock_db(db: Path):  # noqa: ANN202
    """One ladder writer per telemetry DB at a time. Session-less rows are
    attributed to a run by time window; a second ladder run writing the same
    DB would put its rows inside the first one's window."""
    import fcntl

    from prometheus.gym.ladder.fixtures import SandboxError

    db.parent.mkdir(parents=True, exist_ok=True)
    fh = open(f"{db}.ladder.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SandboxError(f"another ladder run is writing {db} — use a different "
                           f"--telemetry-db, or wait") from None
    return fh


async def _probe_thinking(provider: Any) -> dict[str, str]:
    """Whether ``suppress_thinking`` actually suppresses thinking on this
    endpoint — measured by the provider's own probe, as the daemon does at
    boot. Providers without one (ollama) report ``not_probed``."""
    probe = getattr(provider, "verify_thinking_suppression", None)
    if probe is None:
        return {"status": "not_probed", "detail": f"{type(provider).__name__} has no probe"}
    try:
        status, detail = await probe()
        return {"status": str(status), "detail": redact(str(detail)) or ""}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unknown", "detail": f"{type(exc).__name__}"}
