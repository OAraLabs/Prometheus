"""Post-merge acceptance: runs on the INTEGRATED tree, records separately.

Real git repos; the board is a real orphan ref. The operator's working
tree must never be touched.
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
    run_post_merge_acceptance,
)


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


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
    return r


def _board_record(repo: Path, mission: str, path: str) -> dict:
    ref = f"refs/blackboard/{mission}"
    out = _git(repo, "show", f"{ref}:{path}")
    return json.loads(out)


def test_green_acceptance_records_passed_on_the_integrated_tree(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i1", acceptance_command="sh check.sh",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m1",
    )
    assert rec["kind"] == "merged"
    assert rec["passed"] is True
    assert rec["exit_code"] == 0
    assert rec["expected_exit_code"] == 0
    assert rec["source"] == "daemon-run"
    assert rec["tree"] == _git(repo, "rev-parse", "bb/m1/integrate")
    # the record is ON THE BOARD, readable without a checkout
    on_board = _board_record(repo, "m1", "results/merged/m1.i1/m1.json")
    assert on_board["passed"] is True
    # the output blob resolves
    assert _git(repo, "cat-file", "-p", rec["output_ref"]) is not None


def test_a_failing_command_is_a_result_not_an_exception(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i2", acceptance_command="exit 3",
        expected_exit_code=0, integration_ref="bb/m1/integrate",
        mission="m1", merge_id="m2",
    )
    assert rec["passed"] is False
    assert rec["exit_code"] == 3
    assert _board_record(repo, "m1", "results/merged/m1.i2/m2.json")["passed"] is False


def test_expected_nonzero_exit_is_green_when_matched(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i3", acceptance_command="exit 2",
        expected_exit_code=2, integration_ref="bb/m1/integrate",
        mission="m1", merge_id="m3",
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
        acceptance_command="test -f merged_only.txt && sh check.sh",
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
    # write it via the same plumbing the module uses
    from prometheus.coding.acceptance import _write_board_record
    _write_board_record(repo, board_ref="refs/blackboard", mission="m1",
                        path="results/branch/m1.i1/r1.json", content=branch_rec)
    run_post_merge_acceptance(
        repo, item_id="m1.i1", acceptance_command="exit 1",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m9",
    )
    b = _board_record(repo, "m1", "results/branch/m1.i1/r1.json")
    m = _board_record(repo, "m1", "results/merged/m1.i1/m9.json")
    assert b["kind"] == "branch" and b["passed"] is True
    assert m["kind"] == "merged" and m["passed"] is False
    # green branch + red merge stays two facts; neither overwrote the other


def test_unknown_ref_raises(repo):
    with pytest.raises(AcceptanceError, match="unknown integration ref"):
        run_post_merge_acceptance(repo, item_id="x", acceptance_command="true",
                                  integration_ref="bb/nope/integrate", mission="m1")


def test_not_a_repo_raises(tmp_path):
    with pytest.raises(AcceptanceError, match="not a git repository"):
        run_post_merge_acceptance(tmp_path, item_id="x", acceptance_command="true",
                                  integration_ref="main")


def test_timeout_is_a_result(repo):
    rec = run_post_merge_acceptance(
        repo, item_id="m1.i5", acceptance_command="sleep 5",
        integration_ref="bb/m1/integrate", mission="m1", merge_id="m5",
        timeout_seconds=0.5,
    )
    assert rec["passed"] is False
    assert rec["exit_code"] is None
    assert "timed out" in _git(repo, "cat-file", "-p", rec["output_ref"])


def test_the_board_history_is_append_only(repo):
    run_post_merge_acceptance(repo, item_id="m1.a", acceptance_command="true",
                              integration_ref="bb/m1/integrate", mission="m1", merge_id="a")
    run_post_merge_acceptance(repo, item_id="m1.b", acceptance_command="true",
                              integration_ref="bb/m1/integrate", mission="m1", merge_id="b")
    log = _git(repo, "log", "--format=%s", "refs/blackboard/m1").splitlines()
    assert len(log) == 2
    # both records readable at the tip
    assert _board_record(repo, "m1", "results/merged/m1.a/a.json")["item_id"] == "m1.a"
    assert _board_record(repo, "m1", "results/merged/m1.b/b.json")["item_id"] == "m1.b"