"""Voice is the `voice` extra, not a base dependency — and the base install
must import and boot with none of it present.

WHY THIS EXISTS
---------------
#22 added piper-tts, sounddevice and scipy to the BASE dependencies "for voice
mode", inside a PR titled as a cron fix. Every install from then on carried
voice's native stack whether or not it ever spoke. piper-tts requires
onnxruntime, which ships no sdist and no macOS x86_64 wheel after 1.23.2, and
1.23.2 stops at cp313. `uv pip compile pyproject.toml --python-platform
x86_64-apple-darwin` under Python 3.14 answered "No solution found"; under
<=3.13 it resolved only by backtracking onto that abandoned line; uv.lock's
onnxruntime (1.26.0) has no Intel-mac wheel at all. The base install depended
on a package it never imports.

"Never imports" is the load-bearing fact, and it is what makes the move safe:
piper runs as a CLI subprocess (`shutil.which("piper")`), and sounddevice /
scipy / numpy are imported inside the mic and playback functions. So this
file pins two halves that must stay together:

  * the DECLARATION — voice's stack lives in `voice` (and `full`, so an
    existing `[full]` install loses nothing), never in the base list;
  * the ABSENCE — with that whole stack genuinely unimportable, every
    `prometheus` module still imports, a real `oara daemon` still boots, and
    each voice entry point says `pip install 'oara-prometheus[voice]'`
    instead of crashing or going quiet.

The declaration tests were red on origin/main. The absence tests were GREEN
on origin/main — nothing imported the stack at module level then either.
They are guards: they are what stops the next top-level `import numpy` from
turning a slimmer base install into a crashing one.

HOW "ABSENT" IS MADE REAL
-------------------------
The dev environment has the voice stack installed, so absence is simulated in
a child process by a meta-path finder that refuses every module the base
install no longer carries (the closure diff, before vs after, on macOS and
Linux). The finder also records WHICH module asked: an import-time request
from `prometheus.*` code is the defect; a third-party library in this dev
environment reaching for numpy is not, because a real install of that
library's own extra brings numpy with it. `packaging` and `google.protobuf`
also left the base closure but are shared with pytest and other extras here,
so they are attributed rather than blocked: only a request from `prometheus.*`
code is refused.

`test_the_detector_catches_a_top_level_voice_import` is the mutation check:
it replays the exact regression shape through the same finder. Without it, a
finder that silently blocked nothing would make every absence test green.
"""

from __future__ import annotations

import logging
import os
import signal
import site
import socket
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tests.test_daemon_shutdown import _free_port, _ModelsHandler

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# What `voice` must declare. piper-tts is TTS, sounddevice + scipy are the
# CLI's mic capture and playback, faster-whisper is STT.
VOICE_OUTPUT = {"piper-tts", "sounddevice", "scipy"}
VOICE_DISTS = VOICE_OUTPUT | {"faster-whisper"}
# Never a base requirement, directly or by name: onnxruntime is the one with
# no Intel-mac wheel.
NEVER_BASE = VOICE_DISTS | {"onnxruntime"}

# Import names of everything the base install stopped carrying
# (`uv pip compile` before/after, x86_64 + aarch64 macOS and x86_64 Linux),
# plus the STT side of the extra, which was never base.
BLOCKED = sorted({
    "piper", "onnxruntime", "sounddevice", "_sounddevice", "scipy", "numpy",
    "pathvalidate", "flatbuffers", "coloredlogs", "humanfriendly", "sympy",
    "mpmath", "faster_whisper", "ctranslate2", "av",
})
ATTRIBUTED_ONLY = ["packaging", "google.protobuf"]

_CHILD = textwrap.dedent('''\
    import importlib.abc, json, os, pkgutil, runpy, sys, traceback

    BLOCKED = set(json.loads(os.environ["VOICE_BLOCKED"]))
    ATTRIBUTED_ONLY = json.loads(os.environ["VOICE_ATTRIBUTED_ONLY"])
    VIOLATIONS = []

    def _importer():
        f = sys._getframe(2)
        while f is not None and (
            "importlib" in f.f_code.co_filename
            or f.f_code.co_filename == __file__
        ):
            f = f.f_back
        return f.f_globals.get("__name__", "?") if f is not None else "?"

    class VoiceAbsent(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            top = name.partition(".")[0]
            attributed = any(name == p or name.startswith(p + ".")
                             for p in ATTRIBUTED_ONLY)
            if top not in BLOCKED and not attributed:
                return None
            who = _importer()
            if who.split(".")[0] == "prometheus":
                VIOLATIONS.append(f"{who} imports {name}")
            elif attributed:
                return None
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    sys.meta_path.insert(0, VoiceAbsent())

    # The finder must bite, or every assertion downstream is vacuous.
    for probe in ("piper", "onnxruntime", "sounddevice", "scipy", "numpy"):
        try:
            __import__(probe)
        except ModuleNotFoundError:
            continue
        sys.exit(f"BLOCKER INERT: {probe} imported")

    mode = sys.argv[1]
    if mode == "walk":
        import prometheus
        imported, errors = [], {}
        onerror = lambda name: errors.setdefault(name, "walk_packages failed")
        for info in pkgutil.walk_packages(prometheus.__path__, "prometheus.",
                                          onerror=onerror):
            try:
                __import__(info.name)
                imported.append(info.name)
            except BaseException as exc:
                errors[info.name] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"imported": imported, "errors": errors,
                          "violations": VIOLATIONS}))
    elif mode == "probe":
        # The regression, replayed: a prometheus.* module that imports
        # the voice stack at load time.
        ns = {"__name__": "prometheus._voice_probe"}
        try:
            exec(compile("import sounddevice", "<probe>", "exec"), ns)
        except ModuleNotFoundError:
            pass
        print(json.dumps({"violations": VIOLATIONS}))
    elif mode == "daemon":
        sys.argv = ["oara", "daemon"]
        runpy.run_module("prometheus", run_name="__main__", alter_sys=True)
''')


def _child_env(home: Path | None = None) -> dict[str, str]:
    import json

    env = {
        "PATH": os.pathsep.join(
            d for d in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep)
            if not (Path(d) / "piper").exists()
        ),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "VOICE_BLOCKED": json.dumps(BLOCKED),
        "VOICE_ATTRIBUTED_ONLY": json.dumps(ATTRIBUTED_ONLY),
        # A foreign HOME moves user site-packages; see test_daemon_shutdown.
        "PYTHONUSERBASE": site.getuserbase(),
        "HOME": str(home) if home else os.environ.get("HOME", "/tmp"),
    }
    for key in ("VIRTUAL_ENV", "PYTHONPATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _run_child(tmp_path: Path, mode: str) -> dict:
    import json

    script = tmp_path / "voice_absent.py"
    script.write_text(_CHILD, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), mode],
        env=_child_env(), cwd=tmp_path, capture_output=True, text=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ── the declaration ─────────────────────────────────────────────────────


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _names(specs: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(s).name) for s in specs}


def test_base_dependencies_carry_no_voice_stack():
    leaked = _names(_project()["dependencies"]) & NEVER_BASE
    assert not leaked, (
        f"{sorted(leaked)} in the BASE dependencies — every install then "
        f"carries voice's native stack (onnxruntime has no Intel-mac wheel "
        f"after 1.23.2). Voice belongs in the `voice` extra."
    )


def test_voice_extra_carries_the_whole_voice_stack():
    extras = _project()["optional-dependencies"]
    missing = VOICE_DISTS - _names(extras["voice"])
    assert not missing, f"`voice` extra is missing {sorted(missing)}"


def test_full_keeps_what_left_the_base():
    """`[full]` installs had voice OUTPUT through the base list until 0.9.1.
    Moving it out must not quietly take it away from them."""
    extras = _project()["optional-dependencies"]
    missing = VOICE_OUTPUT - _names(extras["full"])
    assert not missing, f"`full` extra is missing {sorted(missing)}"


# ── the absence ─────────────────────────────────────────────────────────


def test_the_detector_catches_a_top_level_voice_import(tmp_path):
    """Mutation check: the finder must see the regression it exists for."""
    result = _run_child(tmp_path, "probe")
    assert result["violations"] == [
        "prometheus._voice_probe imports sounddevice"
    ]


def test_every_module_imports_with_the_voice_stack_absent(tmp_path):
    result = _run_child(tmp_path, "walk")
    # Violations, not import errors, are the verdict: a module that needs a
    # third-party extra which itself uses numpy fails here too, and that is
    # the extra's business. Only prometheus code asking is the defect.
    assert not result["violations"], (
        "prometheus code imports the voice stack at import time — a base "
        "install (no [voice] extra) would crash here:\n  "
        + "\n  ".join(result["violations"])
    )
    # The walk must have actually reached the code in question.
    for must in ("prometheus.daemon", "prometheus.__main__",
                 "prometheus.cli.voice", "prometheus.gateway.telegram",
                 "prometheus.tools.builtin.tts",
                 "prometheus.tools.builtin.whisper_stt"):
        assert must in result["imported"], (
            f"{must} did not import: {result['errors'].get(must)}"
        )


def test_daemon_boots_with_the_voice_stack_absent(tmp_path):
    """A REAL `oara daemon`, voice CONFIGURED, piper off PATH and the whole
    stack unimportable: the web API must come up. Same harness shape as
    test_daemon_shutdown — loopback stub model, isolated HOME."""
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    try:
        api_port, ws_port = _free_port(), _free_port()
        home = tmp_path / "home"
        (home / ".prometheus").mkdir(parents=True)
        (home / ".prometheus" / "prometheus.yaml").write_text(textwrap.dedent(f"""\
            model:
              provider: llama_cpp
              base_url: http://127.0.0.1:{stub.server_port}
              model: fl1-model
            gateway:
              telegram_enabled: false
              voice:
                engine: piper
                model_path: {tmp_path / "voice.onnx"}
                default_mode: auto
            web:
              enabled: true
              api_port: {api_port}
              ws_port: {ws_port}
            tools:
              deferred_loading:
                enabled: auto
                always_loaded: [bash, read_file]
            """), encoding="utf-8")
        script = tmp_path / "voice_absent.py"
        script.write_text(_CHILD, encoding="utf-8")
        log_path = tmp_path / "daemon.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, str(script), "daemon"],
                cwd=tmp_path, env=_child_env(home),
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.time() + 60
                while time.time() < deadline:
                    if proc.poll() is not None:
                        pytest.fail(
                            f"daemon exited rc={proc.returncode} before "
                            f"booting with the voice stack absent:\n"
                            f"{log_path.read_text()[-3000:]}")
                    try:
                        with socket.create_connection(
                                ("127.0.0.1", api_port), timeout=1):
                            break
                    except OSError:
                        time.sleep(0.5)
                else:
                    pytest.fail("daemon web port never opened within 60s:\n"
                                + log_path.read_text()[-3000:])
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=15)
            finally:
                if proc.poll() is None:
                    proc.kill()
    finally:
        stub.shutdown()


# ── every voice entry point names the fix ───────────────────────────────

FIX = "oara-prometheus[voice]"


@pytest.fixture
def no_piper(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


@pytest.fixture
def fresh_cli_voice(monkeypatch):
    from prometheus.cli import voice
    monkeypatch.setattr(voice, "_WARNED", set())
    return voice


async def test_cli_speak_without_piper_names_the_extra(
        no_piper, fresh_cli_voice, capsys):
    out = await fresh_cli_voice.synthesize_wav("hello", {"model_path": "/m.onnx"})
    assert out is None
    assert FIX in capsys.readouterr().out


async def test_cli_mic_without_sounddevice_names_the_extra(
        fresh_cli_voice, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert await fresh_cli_voice.record_push_to_talk() is None
    assert FIX in capsys.readouterr().out


def test_cli_playback_without_audio_deps_names_the_extra(
        no_piper, fresh_cli_voice, monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    wav = tmp_path / "reply.wav"
    wav.write_bytes(b"RIFF")
    assert fresh_cli_voice.play_wav(wav) is False
    assert FIX in capsys.readouterr().out


async def test_telegram_voice_reply_without_piper_warns_once(
        no_piper, caplog, tmp_path):
    """Configured voice + no piper is an install problem: WARNING, once,
    naming the extra — not the debug line an upgrade used to vanish into.
    Unconfigured voice stays quiet."""
    from prometheus.gateway.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    out = str(tmp_path / "o.wav")
    with caplog.at_level(logging.DEBUG, logger="prometheus.gateway.telegram"):
        assert await adapter._run_piper("hi", out, None) is False
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert await adapter._run_piper("hi", out, "/m.onnx") is False
        assert await adapter._run_piper("hi", out, "/m.onnx") is False
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1 and FIX in warned[0].getMessage()


@pytest.mark.parametrize("engine", [None, "piper"], ids=["auto", "named"])
async def test_tts_tool_without_piper_names_the_extra(no_piper, engine, tmp_path):
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.tts import TTSInput, TTSTool

    result = await TTSTool().execute(
        TTSInput(text="hello", engine=engine), ToolExecutionContext(cwd=tmp_path),
    )
    assert result.is_error and FIX in result.output
