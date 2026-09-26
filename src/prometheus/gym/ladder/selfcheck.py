"""Prove each task's verdict DISCRIMINATES — without running a model.

A check that passes on the untouched setup measures nothing, and a check that
fails on a correct solution measures the check. For every task with a
mechanical verdict, three states are replayed through the real predicates and
the real acceptance command:

* **unsolved** — the setup and fixtures as the agent found them, an empty
  final answer, no tool calls. Must FAIL.
* **wrong** — the reference solution's files, tools and cron state, but each
  ``reference.wrong_answers`` entry as the final text. Must be a FAIL — a
  wrong answer, not a format miss. This is what
  catches a check too loose to tell 714 from 741.
* **wrong files** — the reference solution with each ``reference.wrong_files``
  entry (``{path: content}``) written over it: a partial fix, a blind
  replace-all, the distractor value in the output file. Must FAIL. The file
  analogue of a wrong answer.
* **wrong cron state** — each ``reference.wrong_cron_jobs`` entry is a COMPLETE
  registry a near-miss run would leave (hour off by one, minute and hour
  swapped, the wrong weekdays, a sibling job deleted). Must FAIL.
* **format misses** — each ``reference.format_misses`` entry (a reply with no
  ``ANSWER:`` line whose answer cannot be isolated) must be classified exactly
  FORMAT MISS: not a pass, and — the point — not a wrong answer.
* **reference** — ``reference.answer`` / ``files`` / ``delete_files`` /
  ``cron_jobs`` / ``tools`` applied. Must PASS.
* **right answers** — the reference state with each ``reference.right_answers``
  entry as the final text: other correct phrasings (``12,600.0 seconds``,
  a markdown-bold value, a longer sentence) that a check must NOT reject.
  Must PASS. The guard against a check too strict to credit a right answer.

Judge-only tasks have nothing mechanical to prove; their rubric and reference
are validated at load time instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.gym.ladder.fixtures import (
    Sandbox,
    seed_cron_jobs,
    seed_memory_files,
    write_files,
)
from prometheus.gym.ladder.verdict import (
    FORMAT_MISS_REASON,
    check_predicates,
    run_acceptance,
)
from prometheus.gym.scoring import RunTranscript


def synthetic_transcript(answer: str, tools: list[str]) -> RunTranscript:
    """A finished run that called ``tools`` successfully and then said ``answer``."""
    messages: list[ConversationMessage] = [ConversationMessage.from_user_text("task")]
    for i, name in enumerate(tools):
        tid = f"toolu_ref{i}"
        messages.append(ConversationMessage(
            role="assistant", content=[ToolUseBlock(id=tid, name=name, input={})]))
        messages.append(ConversationMessage(
            role="user", content=[ToolResultBlock(tool_use_id=tid, content="ok", is_error=False)]))
    if answer:
        messages.append(ConversationMessage(role="assistant", content=[TextBlock(text=answer)]))
    return RunTranscript.from_messages(messages)


def _stage(task: Any, sandbox: Sandbox) -> None:
    sandbox.reset()
    write_files(sandbox.workspace, task.setup_files)
    fx = task.fixtures
    if fx.get("wiki_pages"):
        write_files(sandbox.wiki, fx["wiki_pages"])
    if fx.get("cron_jobs"):
        seed_cron_jobs(fx["cron_jobs"], sandbox.workspace)
    seed_memory_files(sandbox, fx.get("memory_md"), fx.get("user_md"))


def _apply_reference(task: Any, sandbox: Sandbox) -> None:
    ref = task.reference
    write_files(sandbox.workspace, ref.get("files") or {})
    for rel in ref.get("delete_files") or []:
        p = sandbox.workspace / rel
        if p.exists():
            p.unlink()
    if "cron_jobs" in ref:
        # The COMPLETE registry a correct run leaves behind.
        from prometheus.gateway.cron_service import save_cron_jobs

        save_cron_jobs([])
        seed_cron_jobs(ref["cron_jobs"] or [], sandbox.workspace)


def _verdict(task: Any, sandbox: Sandbox, t: RunTranscript) -> tuple[bool, list[str]]:
    ok, reasons = True, list[str]()
    if task.score:
        ok, reasons = check_predicates(task.score, t, sandbox.workspace)
    if task.acceptance:
        acc = run_acceptance(task.acceptance, task.acceptance_files, sandbox.workspace,
                             home=sandbox.home)
        if acc["status"] != "ran":
            return False, reasons + [f"acceptance could not run: {acc.get('error')}"]
        if not acc["passed"]:
            ok = False
            reasons = reasons + [f"acceptance exit {acc.get('exit_code')}: "
                                 f"{(acc.get('output_tail') or '')[-300:]}"]
    return ok, reasons


def _classify(task: Any, sandbox: Sandbox, t: RunTranscript) -> tuple[str, list[str]]:
    """pass / fail / format_miss — exactly as decide() would call it."""
    ok, reasons = _verdict(task, sandbox, t)
    if ok:
        return "pass", reasons
    if reasons and all(r.startswith(FORMAT_MISS_REASON) for r in reasons):
        return "format_miss", reasons
    return "fail", reasons


def selfcheck_task(task: Any, root: Path) -> list[str]:
    """Problems with *task*'s verdict. [] = it discriminates."""
    if not task.has_deterministic_check:
        return []
    problems: list[str] = []
    sandbox = Sandbox(root)
    previous = sandbox.activate()
    try:
        _stage(task, sandbox)
        ok, _ = _verdict(task, sandbox, synthetic_transcript("", []))
        if ok:
            problems.append("PASSES on the untouched setup with no answer — does not discriminate")

        ref = task.reference
        tools = list(ref.get("tools") or [])
        for wrong in ref.get("wrong_answers") or []:
            _stage(task, sandbox)
            _apply_reference(task, sandbox)
            kind, _ = _classify(task, sandbox, synthetic_transcript(wrong, tools))
            if kind == "pass":
                problems.append(f"PASSES on the wrong answer {wrong!r}")
            elif kind == "format_miss":
                problems.append(f"the wrong answer {wrong!r} is a FORMAT MISS, not a "
                                f"wrong answer — list it under format_misses")

        for miss in ref.get("format_misses") or []:
            _stage(task, sandbox)
            _apply_reference(task, sandbox)
            kind, reasons = _classify(task, sandbox, synthetic_transcript(miss, tools))
            if kind != "format_miss":
                problems.append(f"{miss!r} should be a FORMAT MISS but is {kind.upper()}: {reasons}")

        for i, wrong_files in enumerate(ref.get("wrong_files") or []):
            _stage(task, sandbox)
            _apply_reference(task, sandbox)
            write_files(sandbox.workspace, wrong_files)
            ok, _ = _verdict(task, sandbox, synthetic_transcript(ref.get("answer") or "", tools))
            if ok:
                problems.append(f"PASSES on wrong_files[{i}] ({sorted(wrong_files)})")

        for i, jobs in enumerate(ref.get("wrong_cron_jobs") or []):
            from prometheus.gateway.cron_service import save_cron_jobs

            _stage(task, sandbox)
            _apply_reference(task, sandbox)
            save_cron_jobs([])
            seed_cron_jobs(jobs or [], sandbox.workspace)
            ok, _ = _verdict(task, sandbox, synthetic_transcript(ref.get("answer") or "", tools))
            if ok:
                problems.append(f"PASSES on wrong_cron_jobs[{i}]")

        for right in ref.get("right_answers") or []:
            _stage(task, sandbox)
            _apply_reference(task, sandbox)
            ok, reasons = _verdict(task, sandbox, synthetic_transcript(right, tools))
            if not ok:
                problems.append(f"REJECTS the right answer {right!r}: {reasons}")

        _stage(task, sandbox)
        _apply_reference(task, sandbox)
        ok, reasons = _verdict(task, sandbox, synthetic_transcript(ref.get("answer") or "", tools))
        if not ok:
            problems.append(f"reference solution FAILS: {reasons}")
    finally:
        Sandbox.restore(previous)
    return problems
