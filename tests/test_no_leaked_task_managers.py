"""No test may leave a BackgroundTaskManager running. Structurally.

WHY THIS FILE EXISTS
--------------------
A manager that is not shut down leaves a live ``asyncio`` subprocess transport
when the test's loop closes. The GC finalises it later,
``BaseSubprocessTransport.__del__`` calls ``self._loop.call_soon(...)`` on a
closed loop, and ``RuntimeError: Event loop is closed`` is raised from inside a
finaliser. pytest attributes it to whichever test was running at GC time —
almost never the one that leaked it — so it presents as a 0-to-5-per-run flake
and puts failure-level annotations on CI jobs that passed.

The first fix was a drain fixture inside ``tests/test_tasks.py``. It worked and
it was the wrong shape: 25 more construction sites in 5 other files did not have
it, and the next site written would not either. **A fix that depends on each
test remembering is not a fix, it is a convention.**

So the drain moved to ``conftest.py`` and keys off a registry every manager
joins in ``__init__``. This file is the part that makes that stick:

* ``test_no_manager_was_collected_while_holding_processes`` — the guard
  proper. If a future test leaks a manager, or the conftest fixture is removed
  or stops working, this goes red and names it.
* ``test_a_manager_registers_itself`` — the guard's own precondition. If
  registration silently stopped happening, the guard above would pass on an
  empty registry while every manager leaked, which is the
  green-because-it-measures-nothing shape.
* ``test_asyncio_run_per_call_sites_are_converted`` — an async fixture's
  teardown cannot reach a loop that ``asyncio.run()`` already closed mid-test.
  Those sites had to be converted to ``pytest.mark.asyncio``, and this keeps
  them converted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from prometheus.tasks.manager import (
    BackgroundTaskManager,
    leaked_managers,
    live_managers,
)

TESTS_DIR = Path(__file__).resolve().parent


def test_a_manager_registers_itself():
    """The guard below is worthless if registration stops happening.

    Asserts the mechanism, not a count: a manager constructed here must appear
    in the registry while it is alive.
    """
    mgr = BackgroundTaskManager()
    assert mgr in live_managers(), (
        "BackgroundTaskManager.__init__ no longer registers in _LIVE_MANAGERS. "
        "The conftest drain keys off that registry, so every manager in the "
        "suite is now leaking and the emptiness guard would pass on an empty "
        "set while proving nothing."
    )


def test_no_manager_was_collected_while_holding_processes():
    """THE GUARD.

    Asserts on a record written by a ``weakref.finalize`` callback at COLLECTION
    time, not on a scan of the live registry. Two earlier versions of this guard
    scanned ``live_managers()`` and both passed under mutation, for two
    different reasons worth keeping written down:

    1. The first asserted that ``mgr._processes`` was non-empty.
       ``_watch_process`` pops an entry the moment its task completes, so a
       manager that ran short tasks and leaked every transport presents an empty
       dict — an operand that clears itself on the happy path can only report
       the healthy answer.
    2. The second asserted on flags held by the manager itself. But a WeakSet
       holds leaked managers only until Python collects them, and that
       collection is *exactly* what runs the transport's ``__del__`` and raises
       "Event loop is closed". By the time any later assertion looks, the
       evidence has already dropped out of the set.

    The fact is only observable at collection. Hence the finaliser, and hence
    the explicit ``gc.collect()`` here — without it this test would be as
    timing-dependent as the bug it guards against.
    """
    import gc

    gc.collect()
    leaked = leaked_managers()
    assert not leaked, (
        f"{len(leaked)} BackgroundTaskManager(s) were garbage-collected having "
        f"spawned a process and never been shut down: {leaked}. Each one leaves "
        f"a subprocess transport to be finalised against a closed event loop, "
        f"which surfaces as 'RuntimeError: Event loop is closed' attributed to "
        f"an unrelated test. Either a new test leaked one, or "
        f"tests/conftest.py::_drain_task_managers stopped reaching them — an "
        f"async fixture cannot reach a test that closed its own loop with "
        f"asyncio.run(). Convert that test to @pytest.mark.asyncio rather than "
        f"deleting this guard."
    )


def _files_building_managers() -> list[Path]:
    out = []
    for py in sorted(TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "BackgroundTaskManager"
            ):
                out.append(py)
                break
    return out


def test_asyncio_run_per_call_sites_are_converted():
    """A test that closes its own loop is unreachable by the drain.

    ``asyncio.run()`` creates a loop, runs, and closes it. By the time an async
    fixture's teardown executes, the transports created inside are already bound
    to a dead loop and nothing can reap them — which is why those sites were
    converted to ``@pytest.mark.asyncio``, where the fixture shares the test's
    loop.

    Checked over the AST, on files that actually build a manager, so a bare
    ``asyncio.run`` elsewhere in the suite is not swept up.
    """
    offenders: dict[str, list[int]] = {}
    for py in _files_building_managers():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
        ]
        if lines:
            offenders[py.name] = lines

    assert not offenders, (
        "These files build a BackgroundTaskManager AND call asyncio.run(), so "
        "they close their own loop mid-test and the conftest drain cannot reap "
        "what they created:\n  "
        + "\n  ".join(f"{name}: lines {lines}" for name, lines in offenders.items())
        + "\nConvert the test to @pytest.mark.asyncio and await directly."
    )


@pytest.mark.no_manager_drain
def test_the_detector_actually_detects(monkeypatch):
    """The guard's own precondition — leak one ON PURPOSE and see it recorded.

    Without this, removing ``weakref.finalize`` from ``__init__`` disables the
    evidence source and the guard above passes on an empty record while every
    manager in the suite leaks. That mutation was run and was NOT caught until
    this test existed: the green-because-it-measures-nothing shape, in the
    guard rather than in the code.

    The manager is never given a real process — ``spawned`` is set directly, so
    the test needs no subprocess and cannot itself leak one.
    """
    import gc

    from prometheus.tasks import manager as _m

    before = len(_m.leaked_managers())
    mgr = BackgroundTaskManager()
    mgr._leak_state.spawned = True      # pretend it ran something
    label = mgr._leak_state.label
    del mgr
    gc.collect()

    recorded = _m.leaked_managers()[before:]
    assert label in recorded, (
        "a manager was collected having spawned and not drained, and the "
        "finaliser did NOT record it. tests/test_no_leaked_task_managers.py's "
        "main guard reads that record, so it is now blind: it will pass while "
        "every manager in the suite leaks."
    )
    # Leave the record as we found it, or the guard sees our deliberate leak.
    _m._LEAKED_MANAGERS.remove(label)


@pytest.mark.asyncio
async def test_shutdown_reaps_a_process_that_is_still_running():
    """`shutdown()` must AWAIT the process, not just signal it.

    Covered separately because the suite's other managers run `echo`, which has
    already exited by teardown — so `shutdown`'s reap branch never executes
    there, and deleting `await process.wait()` left every test green. A control
    that is never reached by the cases around it needs its own case.
    """
    mgr = BackgroundTaskManager()
    task = await mgr.create_shell_task(
        command="sleep 120", description="still running", cwd="/tmp"
    )
    proc = mgr._processes.get(task.id)
    assert proc is not None and proc.returncode is None, (
        "the fixture task was not running — this test cannot reach the branch "
        "it exists to cover"
    )

    await mgr.shutdown()

    assert proc.returncode is not None, (
        "shutdown() returned while the process was still unreaped. Signalling "
        "without awaiting leaves the transport live, which is the entire "
        "defect: the loop closes and __del__ fires against it."
    )
    assert not mgr._processes, "shutdown() left entries in _processes"
