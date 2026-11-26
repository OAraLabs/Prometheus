"""Post-merge acceptance: runs on the INTEGRATED tree, records separately.

Real git repos; the board is a real orphan ref. The operator's working
tree must never be touched. The command comes from the roadmap on the
board — never from the caller — and runs through the coding sandbox
(floors + env allowlist), not a bare shell with the daemon's environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.coding.acceptance import (  # noqa: E402
    AcceptanceError,
    _write_board_record,
    run_post_merge_acceptance,
)


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _put_roadmap(repo: Path, mission: str, items: list[dict]) -> None:
    """Write roadmap.json onto the board with the module's own plumbing."""
    _write_board_record(repo, board_ref="refs/blackboard", mission=mission,
                        path="roadmap.json", content=json.dumps({"items": items}))


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "test@example.invalid")
    _git(r, "config", "user.name", "Test")
    (r / "check.sh").write_text("#!/bin/sh\nexit 0\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    _git(r, "branch", "bb/m1/integrate")
    _put_roadmap(r, "m1", [
        {"id": "m1.i1", "acceptance": {"command": "sh check.sh", "expected_exit_code": 0}},
        {"id": "m1.i2", "acceptance": {"command": "exit 3", "expected_exit_code": 0}},
        {"id": "m1.i3", "acceptance": {"command": "exit 2", "expected_exit_code": 2}},
        {"id": "m1.i4", "acceptance": {"command": "test -f merged_only.txt && sh check.sh"}},
        {"id": "m1.i5", "acceptance": {"command": "sleep 5"}},
        {"id": "m1.a", "acceptance": {"command": "true"}},
        {"id": "m1.b", "acceptance": {"command": "true"}},
        {"id": "m1.env", "acceptance": {"command": "test -z \"$BB_TEST_SECRET\""}},
        {"id": "m1.prose", "acceptance": {"note": "looks good to me"}},
    ])
    return r


def _board_record(repo: Path, mission: str, path: str) -> dict:
    ref = f"refs/blackboard/{mission}"
    out = _git(repo, "show", f"{ref}:{path}")
    return json.loads(out)


def test_green_acceptance_records_passed_on_the_integrated_tree(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i1",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m1",
    )
    assert rec["kind"] == "merged"
    assert rec["passed"] is True
    assert rec["exit_code"] == 0
    assert rec["expected_exit_code"] == 0
    assert rec["source"] == "daemon-run"
    assert rec["command"] == "sh check.sh"          # from the roadmap
    assert rec["tree"] == _git(repo, "rev-parse", "bb/m1/integrate")
    # the record is ON THE BOARD, readable without a checkout
    on_board = _board_record(repo, "m1", "results/merged/m1.i1/m1.json")
    assert on_board["passed"] is True
    # the output is IN the board tree next to the record — a referenced
    # blob, not a dangling hash-object git gc would prune
    assert rec["output_ref"] == "results/merged/m1.i1/m1.output.txt"
    assert _git(repo, "show", f"refs/blackboard/m1:{rec['output_ref']}") is not None


def test_a_failing_command_is_a_result_not_an_exception(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i2",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m2",
    )
    assert rec["passed"] is False
    assert rec["exit_code"] == 3
    assert _board_record(repo, "m1", "results/merged/m1.i2/m2.json")["passed"] is False


def test_expected_nonzero_exit_is_green_when_matched(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i3",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m3",
    )
    assert rec["passed"] is True


def test_the_command_runs_on_the_merge_result_not_the_working_tree(repo, tmp_path):
    # the integration branch has a file the working tree (main) does not
    _git(repo, "checkout", "-q", "bb/m1/integrate")
    (repo / "merged_only.txt").write_text("merged\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "merge work")
    _git(repo, "checkout", "-q", "main")
    assert not (repo / "merged_only.txt").exists()
    # the operator's working tree is dirty — must be untouched
    (repo / "check.sh").write_text("#!/bin/sh\nexit 9\n")
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i4",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m4",
    )
    # the integrated check.sh exits 0 (the committed one), not the dirty 9
    assert rec["passed"] is True, rec
    assert (repo / "check.sh").read_text() == "#!/bin/sh\nexit 9\n"  # untouched
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_branch_local_and_merged_results_are_separate_stores(repo):
    # a branch-local record written by hand (the run harness's job, §7)
    branch_rec = json.dumps({"item_id": "m1.i1", "kind": "branch", "run_id": "r1",
                             "exit_code": 0, "passed": True})
    _write_board_record(repo, board_ref="refs/blackboard", mission="m1",
                        path="results/branch/m1.i1/r1.json", content=branch_rec)
    run_post_merge_acceptance(
        repo, item_id="m1.i2",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m9",
    )
    b = _board_record(repo, "m1", "results/branch/m1.i1/r1.json")
    m = _board_record(repo, "m1", "results/merged/m1.i2/m9.json")
    assert b["kind"] == "branch" and b["passed"] is True
    assert m["kind"] == "merged" and m["passed"] is False
    # green branch + red merge stays two facts; neither overwrote the other


def test_unknown_ref_raises(repo):
    with pytest.raises(AcceptanceError, match="unknown integration ref"):
        run_post_merge_acceptance(repo, item_id="m1.i1",
                                  integration_ref="bb/nope/integrate", mission="m1")


def test_not_a_repo_raises(tmp_path):
    with pytest.raises(AcceptanceError, match="not a git repository"):
        run_post_merge_acceptance(tmp_path, item_id="x",
                                  integration_ref="main", mission="m1")


def test_timeout_is_a_result(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i5",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m5",
        timeout_seconds=0.5,
    )
    assert rec["passed"] is False
    assert rec["exit_code"] is None
    assert "timed out" in _git(repo, "show", f"refs/blackboard/m1:{rec['output_ref']}")


def test_the_board_history_is_append_only(repo):
    run_post_merge_acceptance(repo, item_id="m1.a",
                              integration_ref="bb/m1/integrate", mission="m1", merge_id="a")
    run_post_merge_acceptance(repo, item_id="m1.b",
                              integration_ref="bb/m1/integrate", mission="m1", merge_id="b")
    log = _git(repo, "log", "--format=%s", "refs/blackboard/m1").splitlines()
    # roadmap write + two acceptance writes
    assert len(log) == 3
    assert _board_record(repo, "m1", "results/merged/m1.a/a.json")["item_id"] == "m1.a"
    assert _board_record(repo, "m1", "results/merged/m1.b/b.json")["item_id"] == "m1.b"


# --- review #2 fixes -------------------------------------------------------


def test_the_command_runs_without_the_daemons_environment(repo, monkeypatch):
    """The sandbox env allowlist, not os.environ: an API-key-shaped secret
    in the daemon's environment must NOT reach the acceptance command."""
    monkeypatch.setenv("BB_TEST_SECRET", "sk-do-not-leak")
    rec = run_post_merge_acceptance(
        repo, item_id="m1.env",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="env",
    )
    assert rec["passed"] is True, rec


def test_the_caller_cannot_supply_a_command(repo):
    """The signature has no acceptance_command — the roadmap is the only
    source. An item whose acceptance is prose is a named error."""
    with pytest.raises(TypeError):
        run_post_merge_acceptance(repo, item_id="m1.i1", acceptance_command="true",
                                  integration_ref="bb/m1/integrate", mission="m1")
    with pytest.raises(AcceptanceError, match="no acceptance command"):
        run_post_merge_acceptance(repo, item_id="m1.prose",
                                  integration_ref="bb/m1/integrate", mission="m1")


def test_an_unknown_item_is_a_named_error(repo):
    with pytest.raises(AcceptanceError, match="not on the roadmap"):
        run_post_merge_acceptance(repo, item_id="m1.nope",
                                  integration_ref="bb/m1/integrate", mission="m1")


def test_empty_mission_is_refused(repo):
    """The bare refs/blackboard would block every refs/blackboard/<mission>."""
    with pytest.raises(AcceptanceError, match="mission id required"):
        run_post_merge_acceptance(repo, item_id="m1.i1",
                                  integration_ref="bb/m1/integrate", mission="")
    with pytest.raises(AcceptanceError, match="mission id required"):
        _write_board_record(repo, board_ref="refs/blackboard", mission="",
                            path="x.json", content="{}")
    # nothing was written at the bare ref
    assert subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "-q",
                           "refs/blackboard"], capture_output=True).returncode != 0


def test_a_concurrent_board_write_is_not_lost(repo):
    """CAS: a writer that moves the ref between read and update-ref forces
    a re-read + retry; both records survive."""
    import prometheus.coding.acceptance as acc

    real_git = acc._git
    # deterministic race: right after the module reads the ref's tip
    # (rev-parse), a concurrent writer lands a branch result — the CAS
    # update-ref must fail on the stale old value, re-read and retry.
    calls = {"n": 0}

    def racing_revparse(r, *args, **kw):
        res = real_git(r, *args, **kw)
        if args[:1] == ("rev-parse",) and "refs/blackboard/m1" in args and calls["n"] == 0:
            calls["n"] += 1
            _write_board_record(r, board_ref="refs/blackboard", mission="m1",
                                path="results/branch/m1.i1/race.json",
                                content=json.dumps({"kind": "branch", "run_id": "race"}))
        return res

    import unittest.mock as mock
    with mock.patch.object(acc, "_git", racing_revparse):
        rec = run_post_merge_acceptance(repo, item_id="m1.a",
                                        integration_ref="bb/m1/integrate",
                                        mission="m1", merge_id="cas")
    assert rec["passed"] is True
    # BOTH the racing branch record and the merged record are at the tip
    assert _board_record(repo, "m1", "results/branch/m1.i1/race.json")["run_id"] == "race"
    assert _board_record(repo, "m1", "results/merged/m1.a/cas.json")["passed"] is True


def test_awkward_paths_round_trip_through_z_listing(repo):
    """ls-tree -r -z: a path with a newline/quote survives a second write."""
    awkward = 'results/branch/m1.x/we"ird\nname.json'
    _write_board_record(repo, board_ref="refs/blackboard", mission="m1",
                        path=awkward, content='{"ok": true}')
    # a second write must preserve the awkward entry (newline-split parsing
    # would corrupt or drop it)
    _write_board_record(repo, board_ref="refs/blackboard", mission="m1",
                        path="results/branch/m1.x/plain.json", content="{}")
    out = subprocess.run(["git", "-C", str(repo), "show",
                          f"refs/blackboard/m1:{awkward}"],
                         capture_output=True, text=True, check=True)
    assert out.stdout == '{"ok": true}'

def test_the_configured_backend_is_used_when_none_passed(repo):
    """Review #2 nit: the acceptance sandbox uses coding.sandbox_type —
    the configured backend, same as Loop Manager's acceptance — not a
    hardcoded process default."""
    import prometheus.coding.sandbox as sb

    seen = {}
    real_create = sb.create_sandbox

    def spy_create(root, **kw):
        seen["backend"] = kw.get("backend")
        return real_create(root, **kw)

    class _Load:
        def section(self, name):
            return {"sandbox_type": "bwrap"} if name == "coding" else {}

    import unittest.mock as mock
    with mock.patch.object(sb, "create_sandbox", spy_create), \
         mock.patch("prometheus.config.load.load_config_file", return_value=_Load()):
        # bwrap is unavailable on this host -> create_sandbox raises; the
        # point is the backend it was ASKED for, not availability.
        try:
            run_post_merge_acceptance(repo, item_id="m1.a",
                                      integration_ref="bb/m1/integrate",
                                      mission="m1", merge_id="backend")
        except Exception:
            pass
    assert seen.get("backend") == "bwrap"
