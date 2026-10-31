"""scripts/distill/daemon_switch.py against a fake daemon and temp files.

The fake daemon boots the way the real one does: it reads the config and the
pins file and applies the pins with the daemon's own ``apply_config_pins``,
and answers ``GET /api/model`` with the real ``describe``. ``POST /api/restart``
boots it again. The same server plays the local model server (``/llm/...``)
and a Max endpoint (``/max/...``) that must never be called.
"""

import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "distill"))

import daemon_switch as DS  # noqa: E402

TOKEN = "tok-not-for-logs-7f3a"
QWEN_KEY = "sk-sp-not-for-logs-91c2"
LOCAL_MODEL = "/models/local-27b-iq2.gguf"

CONFIG = """\
# Prometheus config (test copy). Comments are part of what must survive.
system:
  name: test
model:
  # Env var holding this provider's API key (cloud providers only; local
  # llama_cpp/ollama need none).
  # api_key_env: ""
  provider: llama_cpp             # llama_cpp, ollama, openai, anthropic,
                                    # deepseek, kimi, glm, mimo
  base_url: {llm} # Inference server URL: the local model server
                                  # (test drive; was another box)
  model: ''                            # HINT ONLY: blank means ask the backend
  grammar_enforcement: true             # GBNF for llama.cpp tool calls
  suppress_thinking: true
context:
  effective_limit: 72000
learning:
  curator_enabled: false   # a background caller of the primary, off
web:
  enabled: true
  api_port: 8005
backends:
  mini_gpu:
    provider: llama_cpp
    base_url: {llm}
"""

PINS = """\
# Config drift pins: enforced on every daemon startup.

model.provider: llama_cpp
model.base_url: {llm}   # test drive; was another box
# model.model: old-name.gguf
"""


class FakeDaemon:
    def __init__(self, config: Path, pins: Path, clock: dict):
        self.config, self.pins, self.clock = config, pins, clock
        self.requests: list[tuple[str, str]] = []
        self.restarts = 0
        self.state = "idle"
        self.restart_ok = True
        self.pending = False
        self.boot_override: dict = {}
        self.llm_ok = True
        self.llm_model = LOCAL_MODEL
        self.boot()
        self.boot_t = -1000.0

    def boot(self):
        from prometheus.daemon import CONFIG_PINS_STATE, apply_config_pins

        doc = yaml.safe_load(self.config.read_text()) or {}
        drifted = apply_config_pins(doc, self.pins)
        doc["model"].update(self.boot_override)
        self.running = doc["model"]
        self.pins_in_force = dict(CONFIG_PINS_STATE["pins"])
        self.drifted = drifted
        self.boot_t = self.clock["t"]

    def view(self):
        from prometheus.web.model_settings import describe

        running = describe(self.running)
        saved = describe((yaml.safe_load(self.config.read_text()) or {}).get("model"))
        return {"running": running, "saved": saved, "pending": self.pending or saved != running,
                "applied": "live", "key": {"env": running["api_key_env"], "present": True}
                if running["api_key_env"] else None,
                "providers": [{"provider": "qwen", "api_key_env": "QWEN_API_KEY", "key_present": True}],
                "warnings": []}


def serve(fake: FakeDaemon):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self):
            return self.headers.get("Authorization") == "Bearer " + TOKEN

        def do_GET(self):
            fake.requests.append(("GET", self.path))
            p = self.path
            if p == "/health":
                return self._send(200, {"status": "ok", "uptime_seconds": fake.clock["t"] - fake.boot_t,
                                        "config_pins": {"active": len(fake.pins_in_force)}})
            if p.startswith("/api/") and not self._authed():
                return self._send(401, {"error": "unauthorized"})
            if p == "/api/model":
                return self._send(200, fake.view())
            if p == "/api/status":
                return self._send(200, {"state": fake.state, "model": fake.running.get("model") or "detected",
                                        "provider": fake.running.get("provider"),
                                        "config_pins": {"pins": fake.pins_in_force,
                                                        "drift_corrected_at_boot": fake.drifted}})
            if p == "/llm/health":
                return self._send(200 if fake.llm_ok else 503, {"status": "ok" if fake.llm_ok else "loading"})
            if p == "/llm/props":
                return self._send(200, {"model_path": fake.llm_model})
            if p == "/llm/v1/models":
                return self._send(200, {"data": [{"id": fake.llm_model}]})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            fake.requests.append(("POST", self.path))
            if self.path == "/api/restart":
                if not self._authed():
                    return self._send(401, {"error": "unauthorized"})
                if self.headers.get("Origin"):
                    return self._send(400, {"error": "browser_not_allowed"})
                if not fake.restart_ok:
                    return self._send(503, {"error": "restart_unavailable"})
                fake.restarts += 1
                fake.boot()
                return self._send(202, {"restarting": True})
            return self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def clock(monkeypatch):
    c = {"t": 0.0, "sleeps": 0}

    def sleep(s):
        c["t"] += s
        c["sleeps"] += 1

    monkeypatch.setattr(DS, "_sleep", sleep)
    monkeypatch.setattr(DS, "_now", lambda: c["t"])
    return c


@pytest.fixture
def rig(tmp_path, monkeypatch, clock):
    for name in ("PROMETHEUS_API_TOKEN", "PROMETHEUS_API_TOKEN_FILE", "PROMETHEUS_PROVIDER",
                 "PROMETHEUS_MODEL", "QWEN_BASE_URL", "QWEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(DS, "systemctl_user", lambda verb, unit: pytest.fail(f"systemctl {verb} {unit}"))
    # The switch mechanics are tested on a daemon whose background callers of the primary all have a config
    # switch (this daemon's memory extractor has none yet: test_to_max_refuses_while_background_callers_*).
    real_background = DS.BACKGROUND_CALLERS
    monkeypatch.setattr(DS, "BACKGROUND_CALLERS", tuple(c for c in real_background if c[1]))
    cfg, pins, env = tmp_path / "prometheus.yaml", tmp_path / "config_pins.yaml", tmp_path / "env"
    cfg.write_text("placeholder\n")
    pins.write_text("placeholder: 1\n")
    fake = FakeDaemon.__new__(FakeDaemon)
    fake.config, fake.pins, fake.clock = cfg, pins, clock
    srv = serve(fake)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    cfg.write_text(CONFIG.format(llm=base + "/llm"))
    pins.write_text(PINS.format(llm=base + "/llm"))
    os.chmod(cfg, 0o600)
    os.chmod(pins, 0o664)
    env.write_text(f"PROMETHEUS_API_TOKEN={TOKEN}\nQWEN_API_KEY={QWEN_KEY}\n"
                   f"export QWEN_BASE_URL={base}/max/compatible-mode/v1\n# a comment\n")
    os.chmod(env, 0o600)
    FakeDaemon.__init__(fake, cfg, pins, clock)
    r = type("Rig", (), {})()
    r.tmp, r.cfg, r.pins, r.env, r.fake, r.base = tmp_path, cfg, pins, env, fake, base
    r.real_background = real_background
    r.state = tmp_path / "state"
    r.notified = tmp_path / "notified.txt"
    r.cfg0, r.pins0 = cfg.read_bytes(), pins.read_bytes()
    r.cfg_st, r.pins_st = cfg.stat(), pins.stat()

    def run(*cmd, execute=True, extra=()):
        argv = [*cmd, "--config", str(cfg), "--pins", str(pins), "--env-file", str(env),
                "--state-dir", str(r.state), "--api", base, "--notify-cmd", f"cat > {r.notified}",
                "--backend-wait", "30", "--idle-wait", "60", *extra]
        return DS.main(argv + (["--execute"] if execute else []))

    r.run = run
    yield r
    srv.shutdown()
    assert not [q for q in fake.requests if q[1].startswith("/max")], "something called the Max endpoint"
    assert not [q for q in fake.requests if "completion" in q[1]], "something sent inference"


def no_secrets(r, out: str):
    assert TOKEN not in out and QWEN_KEY not in out
    for p in r.state.rglob("*"):
        if p.is_file():
            text = p.read_bytes()
            assert TOKEN.encode() not in text and QWEN_KEY.encode() not in text, p


# ── dry run, capture ────────────────────────────────────────────────────────


def test_dry_run_is_the_default_and_changes_nothing(rig, capsys):
    assert rig.run("capture", execute=False) == 0
    assert rig.run("to-max", execute=False) == 0
    assert rig.run("restore", execute=False) == 0
    out = capsys.readouterr().out
    assert "[dry run]" in out and "model.provider" in out
    assert not rig.state.exists()
    assert rig.cfg.read_bytes() == rig.cfg0 and rig.pins.read_bytes() == rig.pins0
    assert rig.fake.restarts == 0
    assert not [q for q in rig.fake.requests if q[0] == "POST"]
    no_secrets(rig, out)


def test_capture_records_the_exact_before_state(rig, capsys):
    assert rig.run("capture") == 0
    active = (rig.state / DS.ACTIVE_FILE).read_text().strip()
    d = rig.state / active
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    for name in (DS.CONFIG_COPY, DS.PINS_COPY, DS.MANIFEST):
        assert stat.S_IMODE((d / name).stat().st_mode) == 0o600
    assert (d / DS.CONFIG_COPY).read_bytes() == rig.cfg0
    assert (d / DS.PINS_COPY).read_bytes() == rig.pins0
    m = json.loads((d / DS.MANIFEST).read_text())
    assert m["files"]["config"]["sha256"] == DS.sha256(rig.cfg0)
    assert m["files"]["pins"]["mode"] == 0o664 and m["files"]["config"]["mode"] == 0o600
    assert m["runtime"]["model"]["running"]["provider"] == "llama_cpp"
    assert m["runtime"]["status"]["config_pins"]["drift_corrected_at_boot"] == []
    assert m["env_file"]["names"]["QWEN_BASE_URL"] == {"set": True}
    assert m["env_file"]["names"]["PROMETHEUS_PROVIDER"] == {"set": False}
    assert m["backend"]["model_path"] == LOCAL_MODEL and m["backend"]["health"] == 200
    no_secrets(rig, capsys.readouterr().out)


def test_capture_refuses_a_pending_change_or_a_config_already_on_max(rig):
    rig.fake.pending = True
    assert rig.run("capture") == 1
    assert not (rig.state / DS.ACTIVE_FILE).exists()
    assert "pending" in rig.notified.read_text() or "waiting" in rig.notified.read_text()
    rig.fake.pending = False
    text = rig.cfg.read_text().replace("provider: llama_cpp  ", "provider: qwen       ", 1)
    rig.cfg.write_text(text)
    rig.pins.write_text("model.provider: qwen\n")
    assert rig.run("capture") == 1  # recording Max as "before" would make restore keep Max


# ── the round trip ──────────────────────────────────────────────────────────


def test_to_max_then_restore_round_trip_is_byte_identical(rig, capsys):
    assert rig.run("capture") == 0
    assert rig.run("to-max") == 0
    doc = yaml.safe_load(rig.cfg.read_text())
    assert doc["model"]["provider"] == "qwen" and doc["model"]["model"] == "qwen3.8-max"
    assert doc["model"]["api_key_env"] == "QWEN_API_KEY" and doc["model"]["base_url"] == ""
    assert doc["backends"]["mini_gpu"]["base_url"] == rig.base + "/llm"  # a deeper key is not touched
    old, new = rig.cfg0.decode().splitlines(), rig.cfg.read_text().splitlines()
    assert len(new) == len(old) + 1
    assert sum(1 for line in old if line not in new) == 3  # provider, base_url, model lines
    assert "  # api_key_env: \"\"" in new and "# Inference server URL" in rig.cfg.read_text()
    # the daemon came back on Max with no drift, after one restart through its own API
    assert rig.fake.restarts == 1
    assert rig.fake.running["provider"] == "qwen" and rig.fake.drifted == []
    assert stat.S_IMODE(rig.pins.stat().st_mode) == 0o664
    assert stat.S_IMODE(rig.cfg.stat().st_mode) == 0o600
    sw = json.loads(next(rig.state.glob("*/" + DS.SWITCHED)).read_text())
    assert sw["status"] == "verified"

    assert rig.run("restore") == 0
    assert rig.cfg.read_bytes() == rig.cfg0 and rig.pins.read_bytes() == rig.pins0
    for path, st in ((rig.cfg, rig.cfg_st), (rig.pins, rig.pins_st)):
        assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(st.st_mode)
        assert path.stat().st_mtime_ns == st.st_mtime_ns
    assert rig.fake.restarts == 2 and rig.fake.running["provider"] == "llama_cpp"
    assert not (rig.state / DS.ACTIVE_FILE).exists()
    assert list(rig.state.glob("*/" + DS.RESTORED))
    assert not rig.notified.exists()
    no_secrets(rig, capsys.readouterr().out)
    # nothing left to do: a second restore (the dead-man after a manual restore) is a no-op
    assert rig.run("restore") == 0 and rig.fake.restarts == 2


def test_the_edited_pins_and_config_boot_on_the_endpoint_from_the_env(rig, monkeypatch):
    """The real pin logic and the real base-URL resolver: no 8092, no drift."""
    from prometheus.daemon import apply_config_pins
    from prometheus.providers.registry import _resolve_base_url

    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    doc = yaml.safe_load(rig.cfg.read_text())
    assert apply_config_pins(doc, rig.pins) == []
    assert yaml.safe_load(rig.pins.read_text()) == {"model.provider": "qwen", "model.base_url": ""}
    monkeypatch.setenv("QWEN_BASE_URL", "https://max.example/compatible-mode/v1")
    assert _resolve_base_url(doc["model"], "qwen") == "https://max.example/compatible-mode/v1"


def test_without_a_pins_file_only_the_config_changes(rig):
    rig.pins.unlink()
    rig.fake.boot()
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    assert not rig.pins.exists() and rig.fake.running["provider"] == "qwen"
    assert rig.run("restore") == 0
    assert not rig.pins.exists() and rig.cfg.read_bytes() == rig.cfg0


def test_a_pins_file_created_during_the_window_is_left_alone(rig):
    rig.pins.unlink()
    rig.fake.boot()
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    rig.pins.write_text("# created during the window\ncontext.effective_limit: 72000\n")
    assert rig.run("restore") == 0
    assert rig.pins.read_text().startswith("# created during the window") and rig.cfg.read_bytes() == rig.cfg0


def test_hook_reruns_are_idempotent(rig):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    captures = sorted(p.name for p in rig.state.iterdir() if p.is_dir())
    assert rig.run("capture") == 0  # already switched: the before-state is kept, not replaced
    assert rig.run("to-max") == 0
    assert sorted(p.name for p in rig.state.iterdir() if p.is_dir()) == captures
    assert rig.fake.restarts == 1


# ── refusals ────────────────────────────────────────────────────────────────


def test_to_max_refuses_without_a_capture(rig):
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 and rig.fake.restarts == 0
    assert "no capture" in rig.notified.read_text()


def test_restore_refuses_switched_files_without_a_capture_and_is_a_noop_otherwise(rig):
    assert rig.run("restore") == 0  # nothing switched, nothing captured
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    switched = rig.cfg.read_bytes()
    (rig.state / DS.ACTIVE_FILE).unlink()  # the capture is gone
    assert rig.run("restore") == 1
    assert rig.cfg.read_bytes() == switched and rig.fake.running["provider"] == "qwen"
    assert "no active capture" in rig.notified.read_text()


def test_restore_refuses_a_damaged_capture(rig):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    copy = next(rig.state.glob("*/" + DS.CONFIG_COPY))
    os.chmod(copy, 0o600)
    copy.write_bytes(copy.read_bytes() + b"# tampered\n")
    switched = rig.cfg.read_bytes()
    assert rig.run("restore") == 1
    assert rig.cfg.read_bytes() == switched


def test_to_max_refuses_files_changed_since_capture(rig):
    assert rig.run("capture") == 0
    edited = rig.cfg.read_text() + "extra_section:\n  added: by someone else\n"
    rig.cfg.write_text(edited)
    assert rig.run("to-max") == 1
    assert rig.cfg.read_text() == edited and rig.fake.restarts == 0
    assert "changed since capture" in rig.notified.read_text()


@pytest.mark.parametrize("old,new,says", [
    ("qwen3.8-max", "qwen3.8-plus", "changed again"),                      # a key to-max changed
    ("grammar_enforcement: true", "grammar_enforcement: false", "touches"),  # the line next to one it changed
])
def test_restore_never_writes_over_an_edit_made_during_the_window(rig, old, new, says):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    edited = rig.cfg.read_text().replace(old, new, 1)
    assert edited != rig.cfg.read_text()
    rig.cfg.write_text(edited)
    assert rig.run("restore") == 1
    assert rig.cfg.read_text() == edited and rig.fake.running["provider"] == "qwen"
    msg = rig.notified.read_text()
    assert "REFUSED" in msg and says in msg and len(msg.splitlines()) <= 5


# What the daemon's SecurityGate writes on an "always allow": the whole config again (temp file + rename).
GRANT = "security:\n  grants:\n  - id: g-window\n    tool_name: bash\n    kind: always\n"


def add_grant(path: Path) -> None:
    tmp = path.with_name(path.name + ".grant.tmp")
    tmp.write_bytes(path.read_bytes() + GRANT.encode())
    os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp, path)


def test_restore_keeps_a_grant_the_daemon_wrote_during_the_window(rig, capsys):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    add_grant(rig.cfg)
    assert rig.run("restore") == 0
    assert rig.cfg.read_bytes() == rig.cfg0 + GRANT.encode()  # to-max's lines out, the grant kept
    assert rig.pins.read_bytes() == rig.pins0
    assert rig.fake.running["provider"] == "llama_cpp" and rig.fake.restarts == 2
    assert not (rig.state / DS.ACTIVE_FILE).exists() and not rig.notified.exists()
    assert "that edit is kept" in capsys.readouterr().out


def test_to_max_never_writes_over_a_grant_written_while_it_waited_for_an_idle_daemon(rig, monkeypatch):
    assert rig.run("capture") == 0
    rig.fake.state = "thinking"  # a turn in progress, which ends by persisting an "always allow"
    tick = DS._sleep

    def sleep(s):
        if rig.fake.state == "thinking":
            add_grant(rig.cfg)
            rig.fake.state = "idle"
        tick(s)

    monkeypatch.setattr(DS, "_sleep", sleep)
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 + GRANT.encode() and rig.pins.read_bytes() == rig.pins0
    assert rig.fake.restarts == 0 and not list(rig.state.glob("*/" + DS.SWITCHED))
    assert "changed while this run waited" in rig.notified.read_text()
    # setups' next attempt runs capture again, then to-max: the grant is part of the before-state now
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    assert GRANT in rig.cfg.read_text() and rig.fake.running["provider"] == "qwen"
    assert rig.run("restore") == 0 and rig.cfg.read_bytes() == rig.cfg0 + GRANT.encode()


def test_restore_after_a_switch_that_never_wrote_leaves_a_later_edit_alone(rig):
    # The window was abandoned after to-max refused: the capture is stale but holds no edit of to-max's.
    assert rig.run("capture") == 0
    add_grant(rig.cfg)
    assert rig.run("restore") == 0
    assert rig.cfg.read_bytes() == rig.cfg0 + GRANT.encode() and rig.fake.restarts == 0
    assert not (rig.state / DS.ACTIVE_FILE).exists()


def test_a_write_that_lands_after_the_check_is_not_written_over(rig, monkeypatch):
    assert rig.run("capture") == 0
    real = DS._write_switched

    def write_switched(cap, record):
        real(cap, record)
        if record["status"] == "writing" and GRANT not in rig.cfg.read_text():
            add_grant(rig.cfg)  # between to-max's last look and its rename

    monkeypatch.setattr(DS, "_write_switched", write_switched)
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 + GRANT.encode() and rig.pins.read_bytes() == rig.pins0
    assert rig.fake.restarts == 0  # nothing was switched, so the rollback restarts nothing
    assert not list(rig.state.glob("*/" + DS.SWITCHED))
    assert list(rig.state.glob("*/" + DS.SWITCHED + ".rolled-back"))


def test_restore_never_writes_over_a_grant_written_while_it_waited_for_the_model_server(rig, monkeypatch):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    rig.fake.llm_ok = False  # the model server is still loading
    tick = DS._sleep

    def sleep(s):
        if not rig.fake.llm_ok:
            add_grant(rig.cfg)
            rig.fake.llm_ok = True
        tick(s)

    monkeypatch.setattr(DS, "_sleep", sleep)
    assert rig.run("restore") == 0
    assert rig.cfg.read_bytes() == rig.cfg0 + GRANT.encode()
    assert rig.fake.running["provider"] == "llama_cpp"


def test_replace_file_is_a_compare_and_swap(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"a")
    with pytest.raises(DS.Changed):
        DS.replace_file(p, b"b", 0o600, expect_sha=DS.sha256(b"x"))
    assert p.read_bytes() == b"a" and [x.name for x in tmp_path.iterdir()] == ["f"]
    DS.replace_file(p, b"b", 0o600, expect_sha=DS.sha256(b"a"))
    assert p.read_bytes() == b"b"


def test_restore_checks_the_boot_of_the_captured_files_not_a_runtime_that_predates_a_pin(rig):
    # A pin added without a restart: the running daemon has not applied it; the files (and a restore) do.
    rig.pins.write_text(rig.pins.read_text() + "context.effective_limit: 64000\n")
    rig.pins0 = rig.pins.read_bytes()
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    assert rig.run("restore") == 0
    assert rig.fake.drifted == ["context.effective_limit"]
    assert not (rig.state / DS.ACTIVE_FILE).exists() and rig.pins.read_bytes() == rig.pins0


def test_a_pins_file_with_no_model_pin_is_left_alone(rig):
    rig.pins.write_text("# pins\ncontext.effective_limit: 72000\n")
    rig.fake.boot()
    before, st0 = rig.pins.read_bytes(), rig.pins.stat()
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    st1 = rig.pins.stat()
    assert (st1.st_ino, st1.st_mtime_ns) == (st0.st_ino, st0.st_mtime_ns)  # not rewritten
    assert rig.run("capture") == 0  # a setups retry runs capture again: already switched
    assert rig.run("to-max") == 0 and rig.fake.restarts == 1  # and to-max: no second restart
    assert rig.run("restore") == 0 and rig.fake.restarts == 2
    assert rig.pins.read_bytes() == before and rig.cfg.read_bytes() == rig.cfg0


@pytest.mark.parametrize("cut", ["before the restart", "between the two writes"])
def test_to_max_finishes_a_switch_that_was_cut_off_after_writing(rig, monkeypatch, cut):
    assert rig.run("capture") == 0
    real_restart, real_replace = DS.restart_daemon, DS.replace_file

    def killed(*a, **kw):
        raise KeyboardInterrupt  # e.g. the ssh hook's timeout ends the process

    def replace(path, *a, **kw):
        if path == rig.pins:
            killed()
        return real_replace(path, *a, **kw)

    if cut == "before the restart":
        monkeypatch.setattr(DS, "restart_daemon", killed)
    else:
        monkeypatch.setattr(DS, "replace_file", replace)
    with pytest.raises(KeyboardInterrupt):
        rig.run("to-max")
    assert yaml.safe_load(rig.cfg.read_text())["model"]["provider"] == "qwen" and rig.fake.restarts == 0
    monkeypatch.setattr(DS, "restart_daemon", real_restart)
    monkeypatch.setattr(DS, "replace_file", real_replace)
    assert rig.run("to-max") == 0  # its own pending change is not a refusal
    assert rig.fake.restarts == 1 and rig.fake.running["provider"] == "qwen"
    assert json.loads(next(rig.state.glob("*/" + DS.SWITCHED)).read_text())["status"] == "verified"
    assert rig.run("restore") == 0 and rig.cfg.read_bytes() == rig.cfg0 and rig.pins.read_bytes() == rig.pins0


def test_a_dry_run_restore_during_a_window_only_notes_a_model_server_not_up_yet(rig, capsys):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    switched = rig.cfg.read_bytes()
    rig.fake.llm_ok = False  # the window's own model is on the GPU; the apply starts this one first
    capsys.readouterr()
    assert rig.run("restore", execute=False) == 0
    out = capsys.readouterr().out
    assert "[dry run] note:" in out and "not healthy" in out
    assert rig.cfg.read_bytes() == switched and rig.fake.restarts == 1


def test_to_max_refuses_while_background_callers_would_use_max(rig, monkeypatch, capsys):
    monkeypatch.setattr(DS, "BACKGROUND_CALLERS", rig.real_background)  # this daemon, as it is
    assert rig.run("to-max", execute=False) == 1  # before any capture
    assert rig.run("capture") == 0
    assert rig.run("to-max", execute=False) == 1
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 and rig.pins.read_bytes() == rig.pins0 and rig.fake.restarts == 0
    assert not list(rig.state.glob("*/" + DS.SWITCHED))
    out = capsys.readouterr().out
    assert "memory extractor" in out and "interactive use only" in out
    assert "memory extractor" in rig.notified.read_text()


def test_a_background_caller_with_a_switch_must_be_off_for_the_window(rig):
    rig.cfg.write_text(rig.cfg.read_text().replace("curator_enabled: false", "curator_enabled: true"))
    rig.cfg0 = rig.cfg.read_bytes()
    assert rig.run("capture") == 0
    assert rig.run("to-max") == 1 and rig.cfg.read_bytes() == rig.cfg0
    assert "curator" in rig.notified.read_text()
    assert rig.run("to-max", extra=["--set", "learning.curator_enabled=false"]) == 0
    assert yaml.safe_load(rig.cfg.read_text())["learning"]["curator_enabled"] is False
    assert rig.run("restore") == 0 and rig.cfg.read_bytes() == rig.cfg0


def test_a_pin_that_turns_a_background_caller_on_counts():
    cfg = "model:\n  provider: llama_cpp\nlearning:\n  curator_enabled: false\n"
    assert any("curator" in b for b in DS.max_plan(cfg, "learning.curator_enabled: true\n")["background"])
    plan = DS.max_plan(cfg, "learning.curator_enabled: true\n", ["learning.curator_enabled=false"])
    assert not any("curator" in b for b in plan["background"])  # the pin is edited in step


@pytest.mark.parametrize("abbrev", ["--exec", "--exe", "--ex", "--execut"])
def test_an_abbreviation_never_stands_in_for_execute(rig, abbrev):
    with pytest.raises(SystemExit):
        rig.run("capture", execute=False, extra=[abbrev])
    assert not rig.state.exists() and rig.fake.restarts == 0


def test_to_max_refuses_when_the_env_file_lacks_the_endpoint(rig):
    assert rig.run("capture") == 0
    rig.env.write_text(f"PROMETHEUS_API_TOKEN={TOKEN}\nQWEN_API_KEY={QWEN_KEY}\nQWEN_BASE_URL=\n")
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 and rig.fake.restarts == 0


def test_to_max_waits_for_an_idle_daemon_and_refuses_a_busy_one(rig, clock):
    assert rig.run("capture") == 0
    rig.fake.state = "thinking"
    assert rig.run("to-max") == 1
    assert clock["t"] >= 60 and rig.cfg.read_bytes() == rig.cfg0 and rig.fake.restarts == 0


def test_a_to_max_that_does_not_verify_is_rolled_back(rig):
    assert rig.run("capture") == 0
    rig.fake.boot_override = {"provider": "llama_cpp"}  # e.g. something else re-pins it at boot
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 and rig.pins.read_bytes() == rig.pins0
    assert rig.fake.restarts == 2  # the switch, then the rollback
    assert "rolled back" in rig.notified.read_text()
    rig.fake.boot_override = {}
    rig.fake.boot()
    assert rig.run("restore") == 0 and rig.fake.restarts == 2  # nothing left to write or restart


def test_restore_waits_for_the_model_server_before_pointing_back(rig, clock):
    assert rig.run("capture") == 0 and rig.run("to-max") == 0
    switched = rig.cfg.read_bytes()
    rig.fake.llm_ok = False  # the local model server is not up yet
    assert rig.run("restore") == 1
    assert clock["t"] >= 30 and rig.cfg.read_bytes() == switched and rig.fake.running["provider"] == "qwen"
    rig.fake.llm_ok, rig.fake.llm_model = True, "/models/teacher-27b-q4.gguf"  # the wrong model on the port
    assert rig.run("restore") == 1
    assert rig.cfg.read_bytes() == switched
    rig.fake.llm_model = LOCAL_MODEL
    assert rig.run("restore") == 0 and rig.cfg.read_bytes() == rig.cfg0


def test_restart_only_through_the_api_never_a_systemd_stop(rig, monkeypatch):
    calls = []

    def fake_systemctl(verb, unit):
        calls.append((verb, unit))
        out = "active" if verb == "is-active" else ""
        return DS.subprocess.CompletedProcess([verb], 0, out + "\n", "")

    monkeypatch.setattr(DS, "systemctl_user", fake_systemctl)
    assert rig.run("capture") == 0
    rig.fake.restart_ok = False  # a daemon that cannot restart itself
    assert rig.run("to-max") == 1
    assert rig.cfg.read_bytes() == rig.cfg0 and calls == []  # no systemd fallback


@pytest.mark.parametrize("verb", ["stop", "restart", "kill", "try-restart"])
def test_the_systemctl_helper_cannot_stop_anything(verb):
    with pytest.raises(ValueError):
        DS.systemctl_user(verb, DS.DAEMON_UNIT)


def test_restart_starts_a_daemon_that_is_down_but_never_kills_a_hung_one(rig, monkeypatch, clock):
    calls = []
    state = {"unit": "active"}

    def fake_systemctl(verb, unit):
        calls.append((verb, unit))
        if verb == "start":
            state["unit"] = "active"
            rig.fake.restart_ok = True
            srv_up["up"] = True
            rig.fake.boot()
        return DS.subprocess.CompletedProcess([verb], 0, state["unit"] + "\n", "")

    srv_up = {"up": False}
    real_http = DS.http

    def http(method, url, token=None, timeout=10.0):
        if not srv_up["up"] and url.startswith(rig.base) and "/llm" not in url:
            return 0, None
        return real_http(method, url, token, timeout)

    monkeypatch.setattr(DS, "systemctl_user", fake_systemctl)
    monkeypatch.setattr(DS, "http", http)
    ctx = DS.Ctx(config=rig.cfg, pins=rig.pins, env_file=rig.env, state_dir=rig.state, api=rig.base,
                 execute=True, boot_wait=30)
    doc = yaml.safe_load(rig.cfg.read_text())
    with pytest.raises(DS.Refusal, match="never stops the daemon"):
        DS.restart_daemon(ctx, doc)  # API silent while the unit is active: hands off
    assert calls == [("is-active", DS.DAEMON_UNIT)]
    state["unit"] = "inactive"
    DS.restart_daemon(ctx, doc)
    assert calls[-1] == ("start", DS.DAEMON_UNIT)
    assert all(v in ("is-active", "start") for v, _ in calls)


# ── the text editor ─────────────────────────────────────────────────────────


def test_set_scalar_keeps_comments_and_every_other_line():
    text = "a: 1\nmodel:\n  # note\n  provider: x   # keep me\n  url: 'it''s # not a comment'  # real\n"
    out = DS.set_scalar(text, ("model", "provider"), "qwen")
    assert out == text.replace("provider: x   # keep me", "provider: qwen   # keep me")
    out = DS.set_scalar(text, ("model", "url"), "")
    assert "  url: ''  # real\n" in out
    out = DS.set_scalar(text, ("model", "api_key_env"), "QWEN_API_KEY", insert_before="provider")
    assert out.splitlines()[3] == "  api_key_env: QWEN_API_KEY"


@pytest.mark.parametrize("text,path", [
    ("model:\n  provider: a\n  provider: b\n", ("model", "provider")),     # duplicate key
    ("model: {provider: a}\n", ("model", "provider")),                     # flow mapping
    ("model:\n  provider:\n    nested: 1\n", ("model", "provider")),      # not a scalar
    ("model:\n  provider: a\nmodel:\n  provider: b\n", ("model", "provider")),
    ("model.provider: a\n", ("model.base_url",)),                          # a pin that is not there
])
def test_set_scalar_refuses_what_it_cannot_edit_exactly(text, path):
    with pytest.raises(ValueError):
        DS.set_scalar(text, path, "qwen")


def test_max_plan_refuses_an_inline_key():
    with pytest.raises(DS.Refusal, match="api_key"):
        DS.max_plan("model:\n  provider: llama_cpp\n  api_key: abc\n", None)


def test_changed_lines_blank_secret_values():
    lines = DS.changed_lines("model:\n  api_key: abc\n", "model:\n  api_key: ''\n")
    assert lines and all("abc" not in x for x in lines)


def test_set_pauses_an_existing_key_for_the_window_and_restore_puts_it_back(rig):
    text = rig.cfg.read_text() + "sentinel:\n  enabled: true\n  synthesis_enabled: true   # idle-time job\n"
    rig.cfg.write_text(text)
    rig.cfg0 = rig.cfg.read_bytes()
    assert rig.run("capture") == 0
    assert rig.run("to-max", extra=["--set", "sentinel.synthesis_enabled=false"]) == 0
    doc = yaml.safe_load(rig.cfg.read_text())
    assert doc["sentinel"] == {"enabled": True, "synthesis_enabled": False}
    assert "  synthesis_enabled: false   # idle-time job" in rig.cfg.read_text()
    assert rig.run("restore") == 0 and rig.cfg.read_bytes() == rig.cfg0


@pytest.mark.parametrize("item", ["model.provider=x", "nosuch.key=1", "sentinel=1", "context.effective_limit=[1]"])
def test_set_refuses_what_it_should_not_touch(item):
    cfg = "model:\n  provider: llama_cpp\ncontext:\n  effective_limit: 72000\n"
    with pytest.raises(DS.Refusal):
        DS.max_plan(cfg, None, [item])


def test_unrelated_pre_existing_drift_does_not_block_the_switch():
    cfg = "model:\n  provider: llama_cpp\n  base_url: http://127.0.0.1:9\ncontext:\n  effective_limit: 1\n"
    pins = "model.provider: llama_cpp\ncontext.effective_limit: 2\n"
    plan = DS.max_plan(cfg, pins)
    assert plan["drift"] == ["context.effective_limit"] and plan["pin_edits"] == ["model.provider"]
