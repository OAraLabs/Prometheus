"""A loud warning when the web API token sits where a model's shell can read it, even with the bash read floor on.

0.9.17 keeps the token's default home, ``~/.config/prometheus/env``, out of a floored shell's reach: Seatbelt,
AppArmor and the gate all deny ``~/.config/*/*env``. Three settings put it where they don't reach:

- ``web.api_token`` in prometheus.yaml (reads of the config stay allowed);
- ``PROMETHEUS_API_TOKEN_FILE`` naming a file outside that pattern;
- an env file moved out of the pattern by ``PROMETHEUS_ENV_FILE`` or ``XDG_CONFIG_HOME``, holding the token.

Any of them, with the read floor on, is reported in the startup log, in ``/api/status`` (``warnings``) and by
``oara doctor``, saying to move the token to ``~/.config/prometheus/env``. With no read floor (off, or a container,
where the documented secret is ``/run/secrets/<name>``) every location is readable alike — the operator's choice —
so nothing is said. The warning never contains the token or the token file's path.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from prometheus.config import api_token as T
from prometheus.permissions import confinement as C

SECRET = "tok-" + "q" * 40
DAEMON = Path(__file__).resolve().parent.parent / "src" / "prometheus" / "daemon.py"


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = (tmp_path / "home").resolve()
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    for name in ("PROMETHEUS_API_TOKEN", "PROMETHEUS_API_TOKEN_FILE", "PROMETHEUS_ENV_FILE", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    return h


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _warn(cfg: dict | None = None, *, floor: bool = True) -> str | None:
    return T.token_location_warning(cfg or {}, read_floor_active=floor)


# ── what counts as exposed ───────────────────────────────────────────────────

def test_a_token_in_prometheus_yaml_is_reported(home):
    msg = _warn({"web": {"api_token": SECRET}})
    assert msg and "web.api_token" in msg and "~/.config/prometheus/env" in msg, msg
    assert SECRET not in msg


def test_a_token_file_outside_the_floored_pattern_is_reported(home, monkeypatch, tmp_path):
    f = _write(tmp_path / "secrets" / "api_token", SECRET + "\n")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(f))
    msg = _warn()
    assert msg and "PROMETHEUS_API_TOKEN_FILE" in msg and "~/.config/prometheus/env" in msg, msg
    assert str(f) not in msg and "secrets" not in msg and SECRET not in msg, "the file's path is secret-shaped"


def test_a_token_file_inside_the_floored_pattern_is_fine(home, monkeypatch):
    f = _write(home / ".config" / "prometheus" / "api.env", SECRET + "\n")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(f))
    assert _warn() is None


def test_the_default_env_file_is_fine(home):
    _write(home / ".config" / "prometheus" / "env", f"PROMETHEUS_API_TOKEN={SECRET}\n")
    assert _warn() is None


@pytest.mark.parametrize("variable", ["PROMETHEUS_ENV_FILE", "XDG_CONFIG_HOME"])
def test_an_env_file_moved_outside_the_pattern_and_holding_the_token_is_reported(home, monkeypatch, tmp_path,
                                                                                variable):
    elsewhere = tmp_path / "elsewhere"
    env = elsewhere / "env" if variable == "PROMETHEUS_ENV_FILE" else elsewhere / "prometheus" / "env"
    _write(env, f"PROMETHEUS_API_TOKEN={SECRET}\n")
    monkeypatch.setenv(variable, str(env if variable == "PROMETHEUS_ENV_FILE" else elsewhere))
    msg = _warn()
    assert msg and f"({variable} moved it)" in msg, msg
    assert str(elsewhere) not in msg and SECRET not in msg


def test_a_moved_env_file_without_the_token_is_fine(home, monkeypatch, tmp_path):
    env = _write(tmp_path / "elsewhere" / "env", "OPENAI_API_KEY=x\n")
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env))
    assert _warn() is None


def test_an_explicit_empty_token_is_not_a_location(home):
    assert _warn({"web": {"api_token": ""}}) is None, "deliberately open is reported by the auth row, not this one"


def test_every_reason_is_named(home, monkeypatch, tmp_path):
    f = _write(tmp_path / "secrets" / "api_token", SECRET + "\n")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(f))
    msg = _warn({"web": {"api_token": SECRET}})
    assert msg and "web.api_token" in msg and "PROMETHEUS_API_TOKEN_FILE" in msg, msg


def test_nothing_is_said_without_a_read_floor(home, monkeypatch, tmp_path):
    # A container: the documented secret is /run/secrets/<name>, and with no read floor every place is readable alike.
    f = _write(tmp_path / "run" / "secrets" / "api_token", SECRET + "\n")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(f))
    assert _warn({"web": {"api_token": SECRET}}, floor=False) is None


# ── at startup ───────────────────────────────────────────────────────────────

def _floor(state: str):
    def report(**_kwargs):
        return {"bash_read_floor": {"state": state}, "bash_write_floor": {"state": C.STATE_OFF}, "dark": False}
    return report


def test_the_startup_check_logs_it_loudly(home, monkeypatch, caplog):
    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_ACTIVE))
    with caplog.at_level(logging.WARNING):
        asyncio.run(T.warn_if_token_shell_readable({"web": {"api_token": SECRET}}))
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("SECURITY" in m and "web.api_token" in m for m in msgs), msgs
    assert not any(SECRET in m for m in msgs)


def test_the_startup_check_is_quiet_without_a_read_floor(home, monkeypatch, caplog):
    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_OFF))
    with caplog.at_level(logging.WARNING):
        asyncio.run(T.warn_if_token_shell_readable({"web": {"api_token": SECRET}}))
    assert not [r for r in caplog.records if "SECURITY" in r.getMessage()]


def test_the_startup_check_never_raises(home, monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("probe broke")
    monkeypatch.setattr(C, "floor_report", boom)
    asyncio.run(T.warn_if_token_shell_readable({"web": {"api_token": SECRET}}))


def test_the_daemon_runs_the_startup_check():
    tree = ast.parse(DAEMON.read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_daemon")
    assert "warn_if_token_shell_readable" in ast.unparse(run), "run_daemon must schedule the token-location check"


# ── oara doctor ──────────────────────────────────────────────────────────────

def test_doctor_has_a_warning_row_with_the_fix(home, monkeypatch):
    from prometheus.cli.doctor import check_token_location

    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_ACTIVE))
    row = check_token_location({"web": {"api_token": SECRET}})
    assert row.status == "warning" and "web.api_token" in row.message and row.fix, row
    assert "~/.config/prometheus/env" in row.fix and SECRET not in row.message + row.fix


def test_doctor_is_ok_when_the_token_is_in_the_floored_env_file(home, monkeypatch):
    from prometheus.cli.doctor import check_token_location

    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_ACTIVE))
    assert check_token_location({}).status == "ok"


def test_doctor_says_why_it_did_not_judge_without_a_read_floor(home, monkeypatch):
    from prometheus.cli.doctor import check_token_location

    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_OFF))
    row = check_token_location({"web": {"api_token": SECRET}})
    assert row.status == "info" and "read floor" in row.message, row


def test_the_doctor_row_is_in_the_default_check_list():
    import inspect

    from prometheus.cli import doctor

    assert "check_token_location" in inspect.getsource(doctor.run_extended_checks)


# ── /api/status (Beacon reads it) ────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")

from tests.support.real_app import BOUNDARY_DOUBLE, build_real_app  # noqa: E402


def _status(h) -> dict:
    r = h.client.get("/api/status", headers=h.auth())
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_status_lists_the_warning(monkeypatch, tmp_path):
    h = build_real_app()
    f = _write(tmp_path / "secrets" / "api_token", h.auth()["Authorization"].split(" ", 1)[1] + "\n")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(f))
    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_ACTIVE))
    with h.client:
        body = _status(h)
    items = [w for w in body["warnings"] if w["code"] == T.TOKEN_LOCATION_WARNING_CODE]
    assert items and items[0]["level"] == "warning" and "PROMETHEUS_API_TOKEN_FILE" in items[0]["message"], body
    assert str(f) not in str(items) and "~/.config/prometheus/env" in items[0]["fix"]


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_status_has_no_token_warning_without_a_read_floor(monkeypatch, tmp_path):
    h = build_real_app()
    f = _write(tmp_path / "secrets" / "api_token", h.auth()["Authorization"].split(" ", 1)[1] + "\n")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(f))
    monkeypatch.setattr(C, "floor_report", _floor(C.STATE_OFF))
    with h.client:
        body = _status(h)
    assert not [w for w in body["warnings"] if w["code"] == T.TOKEN_LOCATION_WARNING_CODE], body["warnings"]
