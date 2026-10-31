"""scripts/distill/teacher_gen.py under a supervisor (gen_runner.py) that restarts it across days and runs two
of it at once: the settings fingerprint survives a date change, a sandbox root has one user, and a crash
has its own exit code. No model, no network.
"""

from __future__ import annotations

import datetime as real_dt
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("distill_teacher_gen", "teacher_gen.py")
SENTINEL = "<<TODAY>>"


@pytest.fixture(autouse=True)
def _restore_state_env(monkeypatch):
    for var in ("PROMETHEUS_LOGS_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR", "PROMETHEUS_ENV_FILE",
                *tg.PROXY_ENV, "NO_PROXY", "no_proxy"):
        old = os.environ.get(var)
        monkeypatch.setenv(var, old if old is not None else "")
        if old is None:
            monkeypatch.delenv(var)


def _on(monkeypatch: pytest.MonkeyPatch, day: str) -> None:
    """teacher_gen's ``dt`` with ``date.today()`` fixed to *day*."""
    fixed = real_dt.date.fromisoformat(day)

    class _Date(real_dt.date):
        @classmethod
        def today(cls) -> real_dt.date:  # type: ignore[override]
            return fixed

    shim = types.SimpleNamespace(**{k: getattr(real_dt, k) for k in dir(real_dt) if not k.startswith("_")})
    shim.date = _Date
    monkeypatch.setattr(tg, "dt", shim)


def _settings(tmp_path: Path) -> Any:
    pack = tmp_path / "request_pack.json"
    pack.write_text(json.dumps({"system_prompt": f"You are the assistant. Today's date is {SENTINEL}.",
                                "date_sentinel": SENTINEL}))
    a = tg.parse_args(["run", "--out", str(tmp_path / "out"), "--builtin-smoke", "--pack", str(pack),
                       "--sandbox-root", str(tmp_path / "sbx"), "--injection-manifest", str(tmp_path / "m.json"),
                       "--eval-tools", str(tmp_path / "e.jsonl")])
    return tg.settings_from_args(a)


def test_a_resume_on_a_later_day_is_the_same_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _on(monkeypatch, "2026-10-31")
    first = _settings(tmp_path)
    assert "2026-10-31" in first.system_prompt(tmp_path) and SENTINEL not in first.system_prompt(tmp_path)
    tg.Store(first.out, first.name, first.fingerprint(), first.coding_cap)
    # gen_runner restarts the same --out after the 03:00 eval on the next calendar day: not refused
    _on(monkeypatch, "2026-11-01")
    second = _settings(tmp_path)
    assert second.fingerprint() == first.fingerprint()
    tg.Store(second.out, second.name, second.fingerprint(), second.coding_cap)
    # each task's prompt carries the day it runs on, as at runtime
    assert "Today's date is 2026-11-01." in second.system_prompt(tmp_path)
    # another pack (another template) is still another run
    (tmp_path / "request_pack.json").write_text(json.dumps({"system_prompt": f"Other. {SENTINEL}",
                                                            "date_sentinel": SENTINEL}))
    third = tg.settings_from_args(tg.parse_args(["run", "--out", str(tmp_path / "out"), "--builtin-smoke", "--pack",
                                                 str(tmp_path / "request_pack.json"), "--injection-manifest", "m",
                                                 "--eval-tools", "e"]))
    with pytest.raises(tg.SettingsMismatch, match="system_prompt_sha"):
        tg.Store(third.out, third.name, third.fingerprint(), third.coding_cap)
    # teacher_ab fixes the date into the prompt itself (both arms byte-identical)
    assert tg.load_pack_prompt(str(tmp_path / "request_pack.json")) == "Other. 2026-11-01"


def test_a_sandbox_root_has_one_user(tmp_path: Path) -> None:
    root = tmp_path / "sbx"
    fd = tg.lock_sandbox_root(root)
    try:
        with pytest.raises(tg.SandboxBusy, match="in use by another teacher_gen run"):
            tg.lock_sandbox_root(root)
        # a run on the same root is refused before anything runs (exit 2), never sharing the workspaces
        rc = tg.main(["run", "--out", str(tmp_path / "out"), "--builtin-smoke", "--sandbox-root", str(root),
                      "--teacher-url", "http://127.0.0.1:1", "--injection-manifest", str(tmp_path / "m.json"),
                      "--eval-tools", str(tmp_path / "e.jsonl")])
        assert rc == tg.EXIT_USAGE
        assert not (tmp_path / "out").exists()
        other = tg.lock_sandbox_root(tmp_path / "sbx-other")   # another root is free
        os.close(other)
    finally:
        os.close(fd)
    os.close(tg.lock_sandbox_root(root))                       # released with its holder


def test_a_crash_exits_6_never_1(tmp_path: Path) -> None:
    # an unexpected exception mid-run (a full disk while appending, a bug in a new code path): Python would
    # exit 1, which gen_runner.py would read as "kept nothing" and never restart
    script = (f"import sys, importlib.util\n"
              f"spec = importlib.util.spec_from_file_location('tgx', {str(_DIR / 'teacher_gen.py')!r})\n"
              f"T = importlib.util.module_from_spec(spec); sys.modules['tgx'] = T; spec.loader.exec_module(T)\n"
              f"def boom(a, **kw):\n    raise OSError(28, 'No space left on device')\n"
              f"T.cmd_run = boom\n"
              f"sys.exit(T.cli(['run', '--out', {str(tmp_path / 'out')!r}, '--builtin-smoke', '--sandbox-root', "
              f"{str(tmp_path / 'sbx')!r}, '--injection-manifest', 'm', '--eval-tools', 'e']))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROMETHEUS_", "QWEN_"))}
    env["HOME"] = str(tmp_path / "home")
    cp = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert cp.returncode == tg.EXIT_CRASH == 6, cp.stderr[-2000:]
    assert "Traceback" in cp.stderr and "No space left on device" in cp.stderr
    assert "teacher_gen crashed (exit 6)" in cp.stderr
    # the command line goes through cli(), and a usage error is still 2, not a crash
    assert "sys.exit(cli())" in (_DIR / "teacher_gen.py").read_text()
    assert tg.cli(["run", "--out", str(tmp_path / "o2"), "--builtin-smoke", "--sandbox-root", str(tmp_path / "s2"),
                   "--teacher-url", "https://example.invalid", "--injection-manifest", "m",
                   "--eval-tools", "e"]) == tg.EXIT_USAGE
