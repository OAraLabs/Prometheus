"""task_create.watch_dir is a declared directory path, resolved once at create.

Two defects, one line apart:

* The gate never saw it. `watch_dir` was undeclared, so `denied_paths` did not
  apply — a file_watch on ~/.ssh reported which key files exist. (Filename
  disclosure, not content: `_run_file_watch` sets `artifact_path` and never
  reads the file. Lesser than cron's cwd, which is execution.)
* It was never resolved. The value was stored and used VERBATIM — no
  `resolve()`, no `expanduser()` — so a relative dir re-resolved against
  whatever process cwd the daemon had at watch time, and `~/logs` was a
  LITERAL directory named `~`. file_watch tasks are persisted and RESUMED
  across restarts, which is what makes that load-bearing rather than cosmetic.

Scope: denied_paths only. No workspace lock — the historical sample is 0 of 0
(19 task_create calls, 17 with recorded args, none carrying a watch_dir), and
0-of-0 justifies nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.tasks.manager import BackgroundTaskManager


class TestResolutionIsSharedAndExpands:

    def test_relative_resolves_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert BackgroundTaskManager.resolve_task_path("logs") == str(
            (tmp_path / "logs").resolve()
        )

    def test_tilde_expands_to_home_not_a_literal_directory(self):
        """`~/x` was a literal dir named '~' under the process cwd, so the
        watcher waited forever on a path nobody meant."""
        got = BackgroundTaskManager.resolve_task_path("~/somedir")
        assert got == str(Path.home() / "somedir")
        assert "~" not in got

    def test_absolute_is_unchanged(self, tmp_path):
        assert BackgroundTaskManager.resolve_task_path(str(tmp_path)) == str(
            tmp_path.resolve()
        )

    def test_cwd_goes_through_the_same_resolver(self, tmp_path):
        """One resolver, not two — the cwd line is where it came from."""
        mgr = BackgroundTaskManager()
        rec = mgr._new_record(
            task_type="local_bash", description="d", cwd="~", command="true",
        )
        assert rec.cwd == str(Path.home())


class TestWatchDirIsStoredAbsolute:

    def _mgr(self):
        return BackgroundTaskManager()

    def test_relative_watch_dir_is_persisted_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "inbox").mkdir()
        mgr = self._mgr()
        rec = asyncio.run(mgr.create_file_watch_task(
            watch_dir="inbox", watch_pattern="*.done",
            description="d", cwd=tmp_path, timeout_seconds=1,
        ))
        assert rec.spec["dir"] == str((tmp_path / "inbox").resolve())
        assert Path(rec.spec["dir"]).is_absolute()

    def test_tilde_watch_dir_is_persisted_expanded(self, tmp_path):
        mgr = self._mgr()
        rec = asyncio.run(mgr.create_file_watch_task(
            watch_dir="~/somedir", watch_pattern="*.done",
            description="d", cwd=tmp_path, timeout_seconds=1,
        ))
        assert rec.spec["dir"] == str(Path.home() / "somedir")


class TestResumeWatchesTheCreatedDirectory:
    """The load-bearing half. Proven by dropping a file and seeing
    artifact_path — not by asserting on the stored string."""

    def test_resumed_task_watches_the_original_directory(self, tmp_path, monkeypatch):
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        async def scenario():
            monkeypatch.chdir(tmp_path)
            mgr = BackgroundTaskManager()
            rec = await mgr.create_file_watch_task(
                watch_dir="inbox", watch_pattern="*.done",
                description="d", cwd=tmp_path, timeout_seconds=20,
            )
            # Simulate the restart: cancel the live waiter, MOVE THE PROCESS
            # CWD (the whole point — a relative string would now resolve
            # somewhere else), then resume exactly as _resume_persisted does.
            mgr._waiters[rec.id].cancel()
            elsewhere = tmp_path / "elsewhere"
            elsewhere.mkdir()
            monkeypatch.chdir(elsewhere)
            mgr._waiters[rec.id] = asyncio.create_task(mgr._run_file_watch(rec.id))
            await asyncio.sleep(0.4)
            (inbox / "ready.done").write_text("x")
            for _ in range(60):
                if mgr._tasks[rec.id].status in ("completed", "failed"):
                    break
                await asyncio.sleep(0.2)
            return mgr._tasks[rec.id]

        task = asyncio.run(scenario())
        assert task.status == "completed", f"resume watched the wrong dir: {task.error}"
        assert task.artifact_path
        assert Path(task.artifact_path).resolve() == (inbox / "ready.done").resolve()


class TestInstrument:
    def test_file_watch_record_carries_the_resolved_dir(self, tmp_path):
        """0-of-0 sample: nothing recorded a watch_dir, so it could never
        grow. The resolved dir is now on the record itself."""
        mgr = BackgroundTaskManager()
        rec = asyncio.run(mgr.create_file_watch_task(
            watch_dir=str(tmp_path), watch_pattern="*.x",
            description="d", cwd=tmp_path, timeout_seconds=1,
        ))
        assert rec.spec["dir"] == str(tmp_path.resolve())


# --------------------------------------------------------------------------- #
# THE FAR SIDE — the real tool through the real dispatch path.
#
# #215's first mutation matrix left three alive because every test called the
# component directly. Same shape would repeat here: the classes above drive
# BackgroundTaskManager, and the gate does not live there — it lives in
# _execute_tool_call. Only these reach it the way production does.
# --------------------------------------------------------------------------- #


class TestTaskCreateThroughRealDispatch:

    def _ctx(self, tmp_path, denied, prompted):
        from prometheus.__main__ import create_tool_registry
        from prometheus.engine.agent_loop import LoopContext
        from prometheus.permissions.checker import SecurityGate

        gate = SecurityGate(denied_paths=[str(denied)])

        def _prompt(tool_name, description):
            prompted.append((tool_name, description))
            class R: approved = False; reason = "harness refuses"
            return R()

        return LoopContext(
            provider=None, model="v", system_prompt="", max_tokens=50,
            tool_registry=create_tool_registry({}, security_gate=gate),
            permission_checker=gate, cwd=tmp_path,
            session_id="cli:test", permission_prompt=_prompt,
        )

    def _call(self, ctx, watch_dir):
        from prometheus.engine.agent_loop import _execute_tool_call

        return asyncio.run(_execute_tool_call(ctx, "task_create", "t", {
            "type": "file_watch", "description": "d",
            "watch_dir": watch_dir, "watch_pattern": "*.done",
        }))

    def test_denied_watch_dir_is_refused(self, tmp_path):
        denied = tmp_path / "secrets"
        denied.mkdir()
        prompted: list = []
        res = self._call(self._ctx(tmp_path, denied, prompted), str(denied))
        assert res.is_error
        assert "denied" in (res.content or "").lower()

    def test_relative_watch_dir_proceeds_with_zero_prompts(self, tmp_path):
        """THE ADMISSION HALF — the check that matters at a 0-of-0 sample.
        A relative dir must resolve against the base, not become UNKNOWN."""
        denied = tmp_path / "secrets"
        denied.mkdir()
        (tmp_path / "inbox").mkdir()
        prompted: list = []
        res = self._call(self._ctx(tmp_path, denied, prompted), "inbox")
        assert not res.is_error, res.content
        assert prompted == [], f"a relative watch_dir prompted: {prompted}"

    def test_ordinary_absolute_watch_dir_proceeds(self, tmp_path):
        denied = tmp_path / "secrets"
        denied.mkdir()
        work = tmp_path / "work"
        work.mkdir()
        prompted: list = []
        res = self._call(self._ctx(tmp_path, denied, prompted), str(work))
        assert not res.is_error, res.content
        assert prompted == []


class TestGateAndWatcherAgreeOnTheDirectory:
    """A relative watch_dir must resolve to the SAME directory in both places.

    Found by outcome, not by review: bare `resolve()` anchors a relative path
    to the PROCESS cwd, while `gate_path_for` resolves the same argument
    against `context.cwd`. The gate cleared /task/cwd/inbox while the observer
    watched /daemon/checkout/inbox — a gate ruling on a directory nothing ever
    reads is the TOCTOU shape the cron work was about, one layer over.
    """

    def test_relative_dir_resolves_against_the_task_cwd_not_the_process(
        self, tmp_path, monkeypatch
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        task_cwd = tmp_path / "taskcwd"
        (task_cwd / "inbox").mkdir(parents=True)
        monkeypatch.chdir(elsewhere)          # process cwd != task cwd

        mgr = BackgroundTaskManager()
        rec = asyncio.run(mgr.create_file_watch_task(
            watch_dir="inbox", watch_pattern="*.done",
            description="d", cwd=task_cwd, timeout_seconds=1,
        ))
        assert rec.spec["dir"] == str((task_cwd / "inbox").resolve())
        assert "elsewhere" not in rec.spec["dir"]

    def test_the_gate_and_the_stored_dir_match(self, tmp_path, monkeypatch):
        """The two resolutions must produce one path — asserted against the
        gate's own function rather than a restatement of it."""
        from prometheus.__main__ import create_tool_registry
        from prometheus.permissions.tool_paths import gate_path_for

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        task_cwd = tmp_path / "taskcwd"
        (task_cwd / "inbox").mkdir(parents=True)
        monkeypatch.chdir(elsewhere)

        schema = {
            s["name"]: s for s in create_tool_registry({}, None).list_schemas()
        }["task_create"]
        gate_sees, unknown = gate_path_for(
            "task_create", {"watch_dir": "inbox"}, schema=schema, base=task_cwd,
        )
        assert unknown is None

        mgr = BackgroundTaskManager()
        rec = asyncio.run(mgr.create_file_watch_task(
            watch_dir="inbox", watch_pattern="*.done",
            description="d", cwd=task_cwd, timeout_seconds=1,
        ))
        assert rec.spec["dir"] == gate_sees, (
            "the gate ruled on one directory and the watcher stored another"
        )
