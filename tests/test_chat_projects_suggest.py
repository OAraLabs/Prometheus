"""``projects/suggest.py``: which project a conversation's tool calls point at (docs/design/chat-projects.md, 3.6).

Sections A and B are the desktop's own smoke cases (``src/main/session-projects-smoke.ts`` sections 2 and 3, the same
cases the phone's ``SessionProjectRulesTests`` run), one assertion each, so the three implementations answer alike.
Section C is the daemon's one deliberate addition: a path inside a chat project's folder belongs to that project, and
the projects root is a container.
"""

from __future__ import annotations

import json

import pytest

from prometheus.projects.suggest import (
    KnownFolder,
    project_of_path,
    projects_in_tool_input,
    suggest,
    tally,
)


def key(p: str):
    hit = project_of_path(p)
    return hit[0] if hit else None


def label(p: str):
    hit = project_of_path(p)
    return hit[1] if hit else None


# --------------------------------------------------------------------------- A. the path rule (smoke section 2)


@pytest.mark.parametrize(("path", "expected"), [
    ("/Users/op/projects/beacon-desktop/src/main/index.ts", "beacon-desktop"),   # container skipped
    ("/home/op/prometheus-deploy/src/prometheus/__main__.py", "prometheus"),     # -deploy folds
    ("~/projects/beacon-desktop/.claude/worktrees/foo/x.ts", "beacon-desktop"),  # a worktree inside the repo
    ("/home/op/beacon-wt-3/src/a.ts", "beacon"),                                 # a -wt-N checkout folds
    ("/home/op/.prometheus/workspace/Officefootballpool/app.py", "officefootballpool"),
    ("/home/op/.prometheus/data/cron_jobs.json", None),                          # the agent's home
    ("/home/op/.prometheus/files/notes.md", None),                               # the outbox
    ("/home/op/.config/oara/middleware.env", None),                              # other dot-directories
    ("/home/op/android-sdk/platform-tools/adb", None),                           # a toolchain
    ("/home/op/jdk-17/bin/java", None),                                          # a jdk
    ("/Users/op/Downloads/x.zip", None),
    ("/home/op/notes.md", None),                                                 # a file directly in home
    ("/home/op/projects/README.md", None),                                       # a file directly in a container
    ("/home/op", None), ("/home/op/", None),                                     # home itself
    ("/home/op/pilot", "pilot"), ("~/projects/reachout", "reachout"),            # a bare project directory
    ("/home/op/projects", None),                                                 # a bare container
    ("/tmp/beacon/x.ts", None), ("/etc/hosts", None),                            # outside home
    ("relative/path/x.ts", None),
])
def test_path_to_project(path, expected):
    assert key(path) == expected


def test_the_label_keeps_the_folders_own_case():
    assert label("/home/op/Prometheus/src/a.py") == "Prometheus"


def test_a_shell_command_yields_every_project_it_names():
    bash = json.dumps({"command": "cd /home/op/pilot && git status && cat ~/projects/beacon-desktop/package.json | head"})
    assert sorted(k for k, _ in projects_in_tool_input(bash)) == ["beacon-desktop", "pilot"]


def test_one_call_naming_a_project_twice_counts_it_once():
    assert len(projects_in_tool_input(json.dumps({"command": "ls /home/op/pilot/a /home/op/pilot/b"}))) == 1


def test_a_path_stops_at_an_escaped_newline_in_json_text():
    quoted = json.dumps({"path": "/home/op/reachout/src/app.ts", "content": "line\n/home/op/other/x"})
    assert sorted(k for k, _ in projects_in_tool_input(quoted)) == ["other", "reachout"]


# --------------------------------------------------------------------------- B. the suggestion (smoke section 3)

B = "/home/op/projects/beacon-desktop/src/a.ts"
P = "/home/op/Prometheus/src/b.py"
N = "/home/op/.prometheus/data/x.json"


def call(path: str) -> str:
    return json.dumps({"path": path})


def test_three_of_four_project_calls_is_75_percent():
    hint = suggest(tally([call(B), call(B), call(B), call(P), call(N), call(N)]))
    assert hint is not None and hint.key == "beacon-desktop" and hint.calls == 4
    assert abs(hint.share - 0.75) < 1e-9


def test_fewer_than_three_project_calls_is_no_suggestion():
    assert suggest(tally([call(B), call(B)])) is None


def test_exactly_sixty_percent_is_suggested():
    hint = suggest(tally([call(B), call(B), call(B), call(P), call(P)]))
    assert hint is not None and hint.key == "beacon-desktop"


def test_fifty_fifty_is_no_suggestion():
    assert suggest(tally([call(B), call(B), call(B), call(P), call(P), call(P), call(B), call(P)])) is None


def test_a_split_with_no_majority_is_no_suggestion():
    assert suggest(tally([call(B), call(B), call(P), call(P), call("/home/op/pilot/x")])) is None


def test_a_tie_at_one_hundred_percent_is_no_suggestion():
    both = json.dumps({"command": f"cp {B} {P}"})
    assert suggest(tally([both, both, both])) is None


def test_case_variants_are_one_project_and_the_commoner_spelling_labels_it():
    hint = suggest(tally([call("/home/op/reachout/a"), call("/home/op/ReachOut/b"), call("/home/op/ReachOut/c")]))
    assert hint is not None and hint.key == "reachout" and hint.label == "ReachOut" and hint.share == 1


# --------------------------------------------------------------------------- C. the daemon's own folders

ROOT = "/home/op/prometheus-projects"
KITCHEN = KnownFolder(id="cp_aaaaaaaaaaaa", name="Kitchen remodel", folder=f"{ROOT}/kitchen-remodel")
REPO = KnownFolder(id="cp_bbbbbbbbbbbb", name="Beacon", folder="/home/op/projects/beacon-desktop")


def test_a_path_inside_a_project_folder_is_that_project():
    assert project_of_path(f"{ROOT}/kitchen-remodel/quote.pdf", known=[KITCHEN], root=ROOT) == (KITCHEN.id, KITCHEN.name)


def test_a_project_on_an_existing_folder_wins_over_the_folder_rule():
    assert project_of_path(f"{REPO.folder}/src/a.ts", known=[REPO], root=ROOT) == (REPO.id, REPO.name)


def test_the_root_is_a_container():
    assert project_of_path(f"{ROOT}/garden/plan.md", known=[], root=ROOT) == ("garden", "garden")
    assert project_of_path(ROOT, known=[], root=ROOT) is None


def test_a_folder_that_only_starts_like_a_project_folder_is_not_it():
    assert project_of_path(f"{ROOT}/kitchen-remodel-old/x", known=[KITCHEN], root=ROOT) == (
        "kitchen-remodel-old", "kitchen-remodel-old")


def test_suggestions_name_the_project_id():
    calls = [call(f"{KITCHEN.folder}/a"), call(f"{KITCHEN.folder}/b"), call(f"{KITCHEN.folder}/c")]
    hint = suggest(tally(calls, known=[KITCHEN], root=ROOT))
    assert hint is not None and hint.key == KITCHEN.id and hint.label == KITCHEN.name


def test_a_project_folder_outside_home_is_still_found():
    outside = KnownFolder(id="cp_cccccccccccc", name="Shared", folder="/srv/projects/shared")
    calls = [call("/srv/projects/shared/a"), call("/srv/projects/shared/b"), call("/srv/projects/shared/c")]
    hint = suggest(tally(calls, known=[outside], root="/srv/projects"))
    assert hint is not None and hint.key == outside.id
    assert projects_in_tool_input(json.dumps({"path": "/srv/projects/garden/x"}), root="/srv/projects") == [
        ("garden", "garden")]
