"""Nothing written to the env file can start a line of its own.

The env file is line-based, and its readers once split it with ``str.splitlines()``, which breaks on far more than
``\\n``: carriage return, vertical tab, form feed, the information separators U+001C-U+001E, NEL (U+0085) and the
Unicode line and paragraph separators (U+2028, U+2029). A check that refused only ASCII controls let those through,
and a value carrying one read back as a second assignment. Every writer now refuses any control or line-breaking
character, the writer itself refuses one as a last line, and the reader splits on ``\\n`` only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prometheus.config import env_file

#: Every character ``str.splitlines()`` breaks on, plus C0/C1 controls and DEL.
BREAKS = ["\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]
CONTROLS = ["\x00", "\x1b", "\x7f", "\x9b"]
BAD = BREAKS + CONTROLS
IDS = [f"U+{ord(c):04X}" for c in BAD]


def _value(ch: str) -> str:
    return f"abc{ch}INJECTED_VAR=planted"


def test_every_character_splitlines_breaks_on_is_covered():
    probe = "".join(f"a{chr(cp)}" for cp in range(0x110000) if len(f"a{chr(cp)}b".splitlines()) > 1)
    covered = {ch for ch in probe if ch != "a"}
    assert covered <= set(BAD)


@pytest.mark.parametrize("ch", BAD, ids=IDS)
def test_the_writer_refuses_and_writes_nothing(tmp_path: Path, ch: str):
    path = tmp_path / "env"
    env_file.set_env_value("SOME_KEY", "before", path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        env_file.set_env_value("SOME_KEY", _value(ch), path)
    assert path.read_bytes() == before


def test_the_writer_refuses_a_bad_variable_name(tmp_path: Path):
    with pytest.raises(ValueError):
        env_file.set_env_value("A B", "x", tmp_path / "env")


@pytest.mark.parametrize("ch", [c for c in BREAKS if c != "\n"], ids=[i for c, i in zip(BAD, IDS) if c in BREAKS and c != "\n"])
def test_the_reader_never_makes_a_second_assignment_from_one_line(tmp_path: Path, ch: str):
    path = tmp_path / "env"
    path.write_text(f"SOME_KEY=abc{ch}INJECTED_VAR=planted\nOTHER=1\n", encoding="utf-8")
    values = env_file.parse_env_file(path)
    assert "INJECTED_VAR" not in values
    assert values["OTHER"] == "1"


@pytest.mark.parametrize("ch", [" ", "\x85", "\x1c"], ids=["U+2028", "U+0085", "U+001C"])
def test_a_rewrite_does_not_turn_a_separator_into_a_newline(tmp_path: Path, ch: str):
    path = tmp_path / "env"
    path.write_text(f"SOME_KEY=abc{ch}INJECTED_VAR=planted\n", encoding="utf-8")
    env_file.set_env_value("OTHER", "1", path)
    env_file.remove_env_value("OTHER", path)
    assert "\nINJECTED_VAR" not in path.read_text(encoding="utf-8")
    assert "INJECTED_VAR" not in env_file.parse_env_file(path)


@pytest.mark.parametrize("ch", BAD, ids=IDS)
def test_the_setup_routes_refuse(ch: str):
    from prometheus.web.setup_server import _unsafe_secret

    assert _unsafe_secret(_value(ch)) is not None


@pytest.mark.parametrize("ch", BAD, ids=IDS)
def test_the_model_route_refuses(ch: str):
    from prometheus.web.model_settings import ModelRefusal, clean_key

    with pytest.raises(ModelRefusal):
        clean_key(_value(ch))


@pytest.mark.parametrize("ch", BAD, ids=IDS)
def test_the_provider_keys_route_refuses_and_writes_nothing(monkeypatch, ch: str):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.delenv("INJECTED_VAR", raising=False)
    c = TestClient(create_app({"model": {"provider": "llama_cpp", "model": "m"}}))
    r = c.put("/api/providers/keys/deepseek", json={"values": {"DEEPSEEK_API_KEY": _value(ch)}})
    assert r.status_code == 400
    path = Path(os.environ["PROMETHEUS_ENV_FILE"])
    assert not path.exists() or "INJECTED_VAR" not in path.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" not in os.environ and "INJECTED_VAR" not in os.environ
    assert "planted" not in r.text


@pytest.mark.parametrize("value", ["abc\ud800", "abc\\", "'abc", '"abc'],
                         ids=["lone-surrogate", "trailing-backslash", "leading-single-quote", "leading-double-quote"])
def test_values_an_older_reader_would_run_on_or_cannot_encode_are_refused(tmp_path: Path, value: str):
    path = tmp_path / "env"
    env_file.set_env_value("API_TOKEN_LIKE", "kept", path)
    before = path.read_bytes()
    assert env_file.env_value_problem(value) is not None
    with pytest.raises(ValueError):
        env_file.set_env_value("SOME_KEY", value, path)
    assert path.read_bytes() == before


def test_a_failed_write_leaves_the_file_whole(tmp_path: Path, monkeypatch):
    path = tmp_path / "env"
    env_file.set_env_value("API_TOKEN_LIKE", "kept", path)
    assert (path.stat().st_mode & 0o777) == 0o600

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(env_file.os, "replace", boom)
    with pytest.raises(OSError):
        env_file.set_env_value("OTHER", "x", path)
    with pytest.raises(OSError):
        env_file.remove_env_value("API_TOKEN_LIKE", path)
    assert env_file.parse_env_file(path) == {"API_TOKEN_LIKE": "kept"}
    assert [p.name for p in tmp_path.iterdir()] == ["env"]


def test_the_provider_keys_route_refuses_a_lone_surrogate_and_keeps_the_file(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    path = Path(os.environ["PROMETHEUS_ENV_FILE"])
    env_file.set_env_value("API_TOKEN_LIKE", "kept", path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    c = TestClient(create_app({"model": {"provider": "llama_cpp", "model": "m"}}))
    r = c.put("/api/providers/keys/deepseek",
              content=b'{"values": {"DEEPSEEK_API_KEY": "abc\\ud800"}}', headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert env_file.parse_env_file(path) == {"API_TOKEN_LIKE": "kept"}


def test_a_space_before_a_quote_is_refused_too():
    assert env_file.env_value_problem(' "abc') is not None
    assert env_file.env_value_problem("abc\\ ") is not None


def test_a_folder_that_takes_no_new_file_falls_back_to_an_in_place_write(tmp_path: Path, monkeypatch):
    import errno
    import tempfile

    path = tmp_path / "env"
    env_file.set_env_value("API_TOKEN_LIKE", "kept", path)

    def refused(*a, **k):
        raise PermissionError(errno.EACCES, "read-only folder")

    monkeypatch.setattr(tempfile, "mkstemp", refused)
    env_file.set_env_value("OTHER", "1", path)
    assert env_file.parse_env_file(path) == {"API_TOKEN_LIKE": "kept", "OTHER": "1"}
    assert (path.stat().st_mode & 0o777) == 0o600


def test_a_file_mounted_on_its_own_falls_back_to_an_in_place_write(tmp_path: Path, monkeypatch):
    import errno

    path = tmp_path / "env"
    env_file.set_env_value("API_TOKEN_LIKE", "kept", path)

    def busy(*a, **k):
        raise OSError(errno.EBUSY, "device or resource busy")

    monkeypatch.setattr(env_file.os, "replace", busy)
    env_file.remove_env_value("API_TOKEN_LIKE", path)
    assert env_file.parse_env_file(path) == {}
    assert [p.name for p in tmp_path.iterdir()] == ["env"]


def test_refusals_say_why():
    from prometheus.web.setup_server import _unsafe_secret

    assert "quote" in _unsafe_secret('"abc"')


@pytest.mark.parametrize("ch", BAD, ids=IDS)
def test_plan_values_refuse_every_line_break_and_control_character(monkeypatch, ch: str):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    for var in ("QWEN_API_KEY", "QWEN_BASE_URL"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    c = TestClient(create_app({"model": {"provider": "llama_cpp", "model": "m"}}))
    bodies = [
        {"values": {"QWEN_API_KEY": "sk-ok"}, "plan": "workspace", "workspace": {"id": f"ws-a{ch}b", "region": "us-east-1"}},
        {"values": {"QWEN_API_KEY": "sk-ok"}, "plan": "workspace",
         "workspace": {"id": "ws-ab", "region": f"us-east-1{ch}"}},
        {"values": {"QWEN_API_KEY": "sk-ok"}, "plan": f"coding-plan{ch}"},
        {"values": {"QWEN_API_KEY": _value(ch)}, "plan": "coding-plan"},
    ]
    path = Path(os.environ["PROMETHEUS_ENV_FILE"])
    for body in bodies:
        r = c.put("/api/providers/keys/qwen", json=body)
        assert r.status_code == 400, body
        assert not path.exists() or path.read_text(encoding="utf-8") == ""
        assert "QWEN_BASE_URL" not in os.environ


def test_every_listed_plan_endpoint_is_a_value_the_env_file_holds():
    from prometheus.providers import plans

    for service_plans in plans.PROVIDER_PLANS.values():
        for plan in service_plans:
            assert env_file.env_value_problem(plan.endpoint) is None, plan.id
    for region in plans.WORKSPACE_REGIONS:
        assert env_file.env_value_problem(plans.workspace_endpoint("ws-a1", region)) is None
