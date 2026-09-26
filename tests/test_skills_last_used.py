"""/skills says when a skill was last USED: its last recorded load, not its file date.

The list read the file's modification time as "last used", but refinement, a
curator pass or a copy rewrites a skill that nobody loaded, and loading one
never touches it. C1 counts every successful ``skill`` load in telemetry
(``ToolCallTelemetry.skill_load_stats``); the gateways' /skills list and the
``/api/skills/list`` rows it mirrors now read that. A skill with no load on
record says so, and where no counter is wired the file date is labelled for
what it is: modified, not used.
"""

from __future__ import annotations

import time

import pytest

from prometheus.gateway import commands

DAY = 86400.0


class _Telemetry:
    def __init__(self, stats: dict) -> None:
        self.stats = stats

    def skill_load_stats(self) -> dict:
        return self.stats


@pytest.fixture
def auto(tmp_path, monkeypatch):
    from prometheus.config import paths

    monkeypatch.setattr(paths, "get_config_dir", lambda: tmp_path)
    d = tmp_path / "skills" / "auto"
    d.mkdir(parents=True)
    return d


def _skill(auto_dir, stem: str, name: str | None = None) -> None:
    (auto_dir / f"{stem}.md").write_text(f"---\nname: {name or stem}\ndescription: d\n---\n# {stem}\n")


def _with(monkeypatch, tel) -> None:
    monkeypatch.setattr("prometheus.telemetry.tracker.get_telemetry_handle", lambda: tel)


def _line(out: str, stem: str) -> str:
    [line] = [ln for ln in out.splitlines() if ln.strip().split(" ")[0] == stem]
    return line


def test_last_used_is_the_last_recorded_load(auto, monkeypatch):
    _skill(auto, "deploy-stack")
    loaded_at = time.time() - 2 * DAY - 60
    _with(monkeypatch, _Telemetry({"deploy-stack": {"loads": 3, "last_loaded_at": loaded_at,
                                                    "source": "auto", "file": "deploy-stack"}}))
    line = _line(commands.cmd_skills_auto_list(), "deploy-stack")
    day = time.strftime("%Y-%m-%d", time.localtime(loaded_at))
    assert f"last used {day} (2d ago), 3 loads" in line


def test_a_skill_nobody_loaded_says_so_however_new_its_file(auto, monkeypatch):
    _skill(auto, "fresh-file")  # mtime: now
    _with(monkeypatch, _Telemetry({}))
    line = _line(commands.cmd_skills_auto_list(), "fresh-file")
    assert "no load recorded" in line and "last used" not in line


def test_one_load_is_singular(auto, monkeypatch):
    _skill(auto, "once")
    _with(monkeypatch, _Telemetry({"once": {"loads": 1, "last_loaded_at": time.time(),
                                            "source": "auto", "file": "once"}}))
    assert "(0d ago), 1 load" in _line(commands.cmd_skills_auto_list(), "once")


def test_loads_are_matched_by_the_served_name_not_the_file_name(auto, monkeypatch):
    _skill(auto, "deploy-stack-2", name="deploy-stack")
    _with(monkeypatch, _Telemetry({"deploy-stack": {"loads": 5, "last_loaded_at": time.time() - DAY,
                                                    "source": "auto", "file": "deploy-stack-2"}}))
    assert "5 loads" in _line(commands.cmd_skills_auto_list(), "deploy-stack-2")


def test_without_a_counter_the_file_date_is_called_modified(auto, monkeypatch):
    _skill(auto, "anything")
    _with(monkeypatch, None)
    out = commands.cmd_skills_auto_list()
    line = _line(out, "anything")
    assert "modified" in line and "last used" not in line and "d ago" in line
    assert "load counts unavailable" in out


def test_the_api_rows_carry_the_same_counts(auto, monkeypatch):
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    _skill(auto, "used")
    _skill(auto, "unused")
    app = create_app({})
    loaded_at = time.time() - DAY
    app.state.telemetry = _Telemetry({"used": {"loads": 2, "last_loaded_at": loaded_at,
                                               "source": "auto", "file": "used"}})
    rows = {r["name"]: r for r in TestClient(app).get("/api/skills/list").json()}
    assert rows["used"]["loads"] == 2 and rows["used"]["last_loaded_at"] == pytest.approx(loaded_at)
    assert rows["unused"]["loads"] == 0 and rows["unused"]["last_loaded_at"] is None
    assert "last_modified" in rows["used"]  # unchanged for clients that read it
