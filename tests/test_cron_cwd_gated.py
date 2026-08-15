"""A cron job's cwd is gated by denied_paths, at create AND at execute.

THE DEFECT: `vet_cron_command` evaluated `gate.evaluate("bash", command=...)`
with NO file_path, so `denied_paths` never saw the working directory — while
`execute_job` passed that same cwd to the subprocess. A gate that inspects a
command STRING cannot see danger delivered by LOCATION: `cat id_rsa`,
`tar -cf - .` and `grep -r . .` are unremarkable commands that become key
exfiltration when the cwd is ~/.ssh. And cron is unattended and persistent.

Scope, deliberately: denied_paths only. A workspace lock for cron cwds was
measured against ONE real sample (1 of 4 persisted jobs carries a cwd, 0
relative, 0 outside a root) and left unruled rather than shipped on
no-evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.gateway import cron_scheduler as cs


@pytest.fixture
def gated(monkeypatch, tmp_path):
    """A real SecurityGate whose denied_paths is a directory we control."""
    from prometheus.permissions.checker import SecurityGate

    denied = tmp_path / "secrets"
    denied.mkdir()
    gate = SecurityGate(denied_paths=[str(denied)])
    monkeypatch.setattr(cs, "_SECURITY_GATE", gate)
    return denied


class TestResolveCronCwd:
    """Resolve ONCE, at create, and persist the absolute value."""

    def test_relative_resolves_against_base(self, tmp_path):
        assert cs.resolve_cron_cwd("sub", base=tmp_path) == str(
            (tmp_path / "sub").resolve()
        )

    def test_absent_keeps_its_current_value_made_explicit(self, tmp_path):
        """Absent used to mean `Path(".")` — the DAEMON's process cwd, which
        moved when the daemon moved to the deploy clone. Same value, but now
        named, written down, and gateable. The default is NOT changed here."""
        assert cs.resolve_cron_cwd(None, base=tmp_path) == str(tmp_path.resolve())

    def test_absolute_is_unchanged(self, tmp_path):
        assert cs.resolve_cron_cwd(str(tmp_path)) == str(tmp_path.resolve())

    def test_result_is_always_absolute(self, tmp_path):
        for raw in (None, "", "x", "./x", str(tmp_path)):
            assert Path(cs.resolve_cron_cwd(raw, base=tmp_path)).is_absolute()


class TestGateSeesTheCwd:

    def test_denied_cwd_is_refused(self, gated):
        allowed, _resolved, reason = cs.normalize_and_vet_cron_job(
            "echo hi", str(gated),
        )
        assert not allowed
        assert "denied" in reason.lower()

    def test_innocuous_command_at_a_denied_cwd_is_still_refused(self, gated):
        """The whole point: the COMMAND is unremarkable. Only the location is
        dangerous, and the command-string check cannot see it."""
        allowed, _r, _reason = cs.normalize_and_vet_cron_job("cat id_rsa", str(gated))
        assert not allowed

    def test_subdirectory_of_a_denied_path_is_refused(self, gated):
        allowed, _r, _reason = cs.normalize_and_vet_cron_job(
            "echo hi", str(gated / "deeper"),
        )
        assert not allowed

    def test_relative_cwd_landing_in_a_denied_path_is_refused(self, gated, tmp_path):
        """Resolution happens BEFORE the gate, or a relative path walks in."""
        allowed, _r, _reason = cs.normalize_and_vet_cron_job(
            "echo hi", "secrets", base=tmp_path,
        )
        assert not allowed

    def test_ordinary_cwd_is_allowed(self, gated, tmp_path):
        allowed, resolved, _reason = cs.normalize_and_vet_cron_job(
            "echo hi", str(tmp_path / "work"),
        )
        assert allowed
        assert resolved == str((tmp_path / "work").resolve())

    def test_gate_unavailable_fails_closed(self, monkeypatch):
        monkeypatch.setattr(cs, "_SECURITY_GATE", None)
        monkeypatch.setattr(cs, "_get_security_gate", lambda: None)
        allowed, _r, reason = cs.normalize_and_vet_cron_job("echo hi", "/tmp")
        assert not allowed and "ungated" in reason


class TestExecuteSideIsTheTotalInvariant:
    """The half a create-only fix silently misses."""

    def _job(self, name, command, cwd):
        job = {"name": name, "command": command, "enabled": True}
        if cwd is not None:
            job["cwd"] = cwd
        return job

    def test_job_persisted_before_the_fix_is_refused_at_execute(
        self, gated, monkeypatch, tmp_path
    ):
        """A row already on disk with a denied cwd never passed a create check
        — this is the only thing that stops it."""
        import asyncio

        monkeypatch.setattr(cs, "get_data_dir", lambda: tmp_path)
        monkeypatch.setattr(cs, "mark_job_run", lambda *a, **k: None)
        entry = asyncio.run(cs.execute_job(self._job("legacy", "echo hi", str(gated))))
        assert entry["status"] == "blocked"
        assert entry["returncode"] == 126

    def test_blocked_entry_records_the_resolved_cwd(self, gated, monkeypatch, tmp_path):
        """Instrument fix: history rows carry the cwd so the sample can grow.
        The measurement that scoped this PR could not distinguish 'no cwd' from
        'cwd not recorded' — 16,460 history rows, zero cwd fields."""
        import asyncio

        monkeypatch.setattr(cs, "get_data_dir", lambda: tmp_path)
        monkeypatch.setattr(cs, "mark_job_run", lambda *a, **k: None)
        entry = asyncio.run(cs.execute_job(self._job("legacy", "echo hi", str(gated))))
        assert entry["cwd"] == str(gated.resolve())

    def test_keyless_job_still_runs(self, gated, monkeypatch, tmp_path):
        """THE REGRESSION HALF. Three live jobs (heartbeat watcher, vault lint,
        news briefing) carry no cwd key at all. Making absent explicit must not
        stop them running."""
        import asyncio

        monkeypatch.setattr(cs, "get_data_dir", lambda: tmp_path)
        monkeypatch.setattr(cs, "mark_job_run", lambda *a, **k: None)
        entry = asyncio.run(cs.execute_job(self._job("keyless", "echo alive", None)))
        assert entry["status"] == "success", entry.get("stderr")
        assert "alive" in entry["stdout"]
        assert Path(entry["cwd"]).is_absolute()

    def test_successful_run_records_the_cwd(self, gated, monkeypatch, tmp_path):
        import asyncio

        monkeypatch.setattr(cs, "get_data_dir", lambda: tmp_path)
        monkeypatch.setattr(cs, "mark_job_run", lambda *a, **k: None)
        work = tmp_path / "work"
        work.mkdir()
        entry = asyncio.run(cs.execute_job(self._job("ok", "pwd", str(work))))
        assert entry["status"] == "success"
        assert entry["cwd"] == str(work.resolve())
        assert str(work.resolve()) in entry["stdout"]


# --------------------------------------------------------------------------- #
# The FAR SIDE — drive the real creation paths, not the choke point directly.
#
# The first mutation matrix left M5/M6/M7 alive (tool persists the raw cwd /
# tool ignores the refusal / POST ignores the refusal) and the reason was not
# that they are unobservable — it is that every test above called
# normalize_and_vet_cron_job or execute_job DIRECTLY. Testing the component
# proves the component; only the far side proves the delivery. Widening a
# selector would have hidden that.
# --------------------------------------------------------------------------- #


class TestCronCreateToolIsGated:

    def _run(self, tmp_path, monkeypatch, cwd, command="echo hi"):
        import asyncio

        from prometheus.tools.base import ToolExecutionContext
        from prometheus.tools.builtin.cron_create import CronCreateTool

        reg = tmp_path / "cron_jobs.json"
        monkeypatch.setattr(
            "prometheus.gateway.cron_service.get_cron_registry_path", lambda: reg
        )
        tool = CronCreateTool()
        args = tool.input_model(
            name="j", schedule="*/5 * * * *", command=command, cwd=cwd
        )
        res = asyncio.run(
            tool.execute(args, ToolExecutionContext(cwd=tmp_path))
        )
        stored = json.loads(reg.read_text()) if reg.exists() else []
        jobs = stored if isinstance(stored, list) else stored.get("jobs", [])
        return res, jobs

    def test_denied_cwd_refused_and_nothing_persisted(
        self, gated, tmp_path, monkeypatch
    ):
        res, jobs = self._run(tmp_path, monkeypatch, str(gated))
        assert res.is_error
        assert "Refused" in res.output
        assert jobs == [], "a refused job must not reach the registry"

    def test_relative_cwd_is_persisted_ABSOLUTE(self, gated, tmp_path, monkeypatch):
        """Resolve once at create. Persisting the relative string would let
        create and execute disagree under different process cwds."""
        (tmp_path / "work").mkdir()
        res, jobs = self._run(tmp_path, monkeypatch, "work")
        assert not res.is_error, res.output
        assert len(jobs) == 1
        assert jobs[0]["cwd"] == str((tmp_path / "work").resolve())
        assert Path(jobs[0]["cwd"]).is_absolute()

    def test_absent_cwd_is_persisted_ABSOLUTE(self, gated, tmp_path, monkeypatch):
        res, jobs = self._run(tmp_path, monkeypatch, None)
        assert not res.is_error, res.output
        assert jobs[0]["cwd"] == str(tmp_path.resolve())


class TestCronRoutesAreGated:

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch, gated):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.web.server import create_app

        reg = tmp_path / "cron_jobs.json"
        monkeypatch.setattr(
            "prometheus.gateway.cron_service.get_cron_registry_path", lambda: reg
        )
        return TestClient(create_app({})), reg, gated

    def test_post_with_denied_cwd_is_refused(self, client):
        """The route a tool-only fix misses — and the one a remote client uses."""
        c, reg, denied = client
        r = c.post("/api/cron", json={
            "name": "x", "schedule": "*/5 * * * *",
            "command": "echo hi", "cwd": str(denied),
        })
        assert r.status_code == 400
        assert "SecurityGate" in r.json()["error"]
        assert not reg.exists() or json.loads(reg.read_text()) in ([], {})

    def test_post_persists_an_absolute_cwd(self, client, tmp_path):
        c, _reg, _d = client
        r = c.post("/api/cron", json={
            "name": "x", "schedule": "*/5 * * * *", "command": "echo hi",
        })
        assert r.status_code == 201
        assert Path(r.json()["job"]["cwd"]).is_absolute()

    def test_put_changing_ONLY_the_cwd_is_vetted(self, client):
        """The hole per-field vetting left: a PUT touching no `command` branch
        ran no check at all, so an already-approved command could be relocated
        into a denied path."""
        c, _reg, denied = client
        assert c.post("/api/cron", json={
            "name": "x", "schedule": "*/5 * * * *", "command": "echo hi",
        }).status_code == 201
        r = c.put("/api/cron/x", json={"cwd": str(denied)})
        assert r.status_code == 400
        assert "SecurityGate" in r.json()["error"]
        assert Path(c.get("/api/cron").json()[0]["cwd"]) != denied


class TestNoWorkspaceLockIsPinned:
    """A cwd OUTSIDE every workspace root is ALLOWED, with zero prompts.

    UNRULED, NOT INTENDED — see PR #215. The no-workspace-lock decision was
    made because the evidence for a lock was one sample (1 of 4 persisted jobs
    carries a cwd, 0 relative, 0 outside a root), not because an unconfined
    cron cwd is known-good. Will left it unruled deliberately.

    Right now that ruling is honoured only by ACCIDENT of two facts: this call
    site evaluates the tool name ``"bash"``, and ``_APPROVE_TOOLS`` happens to
    contain only ``write_file``/``edit_file``. Either could change without
    anyone connecting it to cron — add ``bash`` to ``_APPROVE_TOOLS`` for an
    unrelated reason, or rename the tool this site evaluates, and cron jobs
    outside the workspace would start prompting (and, unattended, refusing)
    with no one having decided that.

    So the behaviour is pinned here rather than left to fall out. If this goes
    red, nothing is necessarily broken — but a decision has been made by
    accident and needs to be made on purpose.
    """

    def test_cwd_outside_every_workspace_root_is_allowed(self, monkeypatch, tmp_path):
        from prometheus.permissions.checker import SecurityGate

        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        gate = SecurityGate(workspace_root=[str(workspace)], denied_paths=[])
        monkeypatch.setattr(cs, "_SECURITY_GATE", gate)

        allowed, resolved, reason = cs.normalize_and_vet_cron_job(
            "echo hi", str(outside),
        )
        assert allowed, (
            f"a cron cwd outside every workspace root was refused ({reason}). "
            "The no-workspace-lock ruling is UNRULED, NOT INTENDED — see PR "
            "#215. Check whether 'bash' entered _APPROVE_TOOLS or this site's "
            "evaluated tool name changed."
        )
        assert resolved == str(outside.resolve())

    def test_the_mechanism_that_makes_it_so(self):
        """Names the two facts the row above depends on, so a red test points
        at the cause instead of the symptom."""
        from prometheus.permissions.checker import _APPROVE_TOOLS

        assert "bash" not in _APPROVE_TOOLS, (
            "'bash' entered _APPROVE_TOOLS — cron cwds outside the workspace "
            "will now prompt, and unattended jobs will refuse. That may be "
            "correct, but it is a ruling nobody made: see PR #215."
        )

    def test_denied_still_wins_outside_the_workspace(self, monkeypatch, tmp_path):
        """No workspace lock does NOT mean no confinement — denied_paths is
        unconditional and applies wherever the cwd lands."""
        from prometheus.permissions.checker import SecurityGate

        denied = tmp_path / "secrets"
        denied.mkdir()
        gate = SecurityGate(workspace_root=[str(tmp_path / "ws")],
                            denied_paths=[str(denied)])
        monkeypatch.setattr(cs, "_SECURITY_GATE", gate)
        allowed, _r, _reason = cs.normalize_and_vet_cron_job("echo hi", str(denied))
        assert not allowed
