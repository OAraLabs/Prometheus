"""Pause everything (Beacon iOS Piece 17, PR 3): the push alert and the doctor row.

Held here:

  * a NEW pause pushes one alert, "Prometheus paused", saying who paused, to every live owner device with no live
    socket (the spec: "A pause nobody knows about looks like a broken agent"); no category, so a lock screen offers
    nothing to act on; never the reason (it stays on ``/api/status``); never to a scoped or revoked device; a second
    ``PUT /api/pause`` (the same pause) pushes nothing; a store that cannot name its owners sends nothing (fail
    closed); a push that fails does not fail the pause;
  * ``oara doctor`` says whether everything is paused: ok when not, a warning with since when (never who or why)
    when paused, and an error with what to do when the pause file is unreadable (which counts as paused);
  * ``/curator run`` refuses while paused instead of running its non-model pass.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from prometheus.engine import pause


@pytest.fixture
def state(tmp_path):
    s = pause.PauseState(tmp_path / "pause.json")
    pause.install(s)
    yield s
    pause.install(None)


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, **kwargs):
        from prometheus.push.apns import SendResult

        self.sent.append(kwargs)
        return SendResult(outcome="ok", status=200)


def _store(tmp_path):
    from prometheus.config.device_store import DeviceStore

    store = DeviceStore(tmp_path / "devices.db")
    away = store.mint_owner("owner phone", "ios", by="test")
    watching = store.mint_owner("owner tablet", "ios", by="test")
    scoped = store.mint("scoped phone", "ios")
    gone = store.mint_owner("old owner phone", "ios", by="test")
    for d, tok in ((away, "tok-away"), (watching, "tok-watching"), (scoped, "tok-scoped"), (gone, "tok-gone")):
        assert store.set_push(d["id"], tok, "sandbox", "ai.test.beacon")
    store.revoke(gone["id"])
    return store, watching["id"]


def _dispatcher(store, sender, live_ids=()):
    from prometheus.push.dispatcher import PushDispatcher

    bridge = types.SimpleNamespace(_ws_identity={object(): types.SimpleNamespace(id=i) for i in live_ids})
    return PushDispatcher(store, sender, bridge)


# ── the push ─────────────────────────────────────────────────────────────────

def test_a_pause_alerts_owner_devices_with_no_live_socket(tmp_path):
    store, watching = _store(tmp_path)
    sender = _Sender()
    info = pause.PauseInfo(since=1_700_000_000.0, by="owner phone", reason="private words")
    asyncio.run(_dispatcher(store, sender, live_ids=[watching]).push_paused(info))
    assert [s["apns_token"] for s in sender.sent] == ["tok-away"]
    body = sender.sent[0]["payload"]
    assert body["aps"]["alert"]["title"] == "Prometheus paused"
    assert "owner phone" in body["aps"]["alert"]["body"]
    assert "category" not in body["aps"]
    assert body["paused_since"] == 1_700_000_000.0
    assert "private words" not in str(body)


def test_a_store_that_cannot_name_its_owners_sends_no_pause_alert(tmp_path):
    class _Store:
        def push_targets(self):
            return [types.SimpleNamespace(id="d1", apns_token="tok", environment="sandbox", bundle_id="",
                                          push_failures=0)]

        def owner_device_ids(self):
            raise RuntimeError("database is locked")

    sender = _Sender()
    asyncio.run(_dispatcher(_Store(), sender).push_paused(pause.PauseInfo(since=1.0, by="owner")))
    assert sender.sent == []


@pytest.fixture
def w(tmp_path, state):
    pytest.importorskip("fastapi")
    from tests.support.device_world import World

    return World(tmp_path)


class _Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[pause.PauseInfo] = []
        self.fail = fail

    async def push_paused(self, info):
        self.calls.append(info)
        if self.fail:
            raise RuntimeError("apns down")


def test_pausing_pushes_once_and_pausing_again_pushes_nothing(w):
    rec = _Recorder()
    w.bridge.push_dispatcher = rec
    assert w.call("op", "PUT", "/api/pause", json={"reason": "away"}).status_code == 200
    assert w.call("op", "PUT", "/api/pause", json={}).status_code == 200
    assert len(rec.calls) == 1 and rec.calls[0].reason == "away"


def test_a_push_that_fails_does_not_fail_the_pause(w):
    w.bridge.push_dispatcher = _Recorder(fail=True)
    r = w.call("op", "PUT", "/api/pause", json={})
    assert r.status_code == 200 and r.json()["paused"] is True
    assert pause.current() is not None


def test_pausing_from_the_owners_telegram_pushes_too(w):
    from prometheus.web.pause_routes import pause_from_chat

    rec = _Recorder()
    w.bridge.push_dispatcher = rec
    asyncio.run(pause_from_chat(by="telegram owner", reason=""))
    assert len(rec.calls) == 1


# ── the doctor row ───────────────────────────────────────────────────────────

def test_the_doctor_says_not_paused(state):
    from prometheus.infra.doctor import check_pause

    row = check_pause()
    assert row.status == "ok" and row.name == "Pause"


def test_the_doctor_says_paused_and_since_when_but_not_who_or_why(state):
    from prometheus.infra.doctor import check_pause

    state.pause(by="owner phone", reason="private words")
    row = check_pause()
    assert row.status == "warning" and "paused" in row.message.lower()
    assert "owner phone" not in row.message and "private words" not in row.message
    assert row.fix and "/resume" in row.fix


def test_the_doctor_says_an_unreadable_pause_file_is_an_error_that_counts_as_paused(tmp_path, monkeypatch):
    from prometheus.infra.doctor import check_pause

    pause.install(None)
    path = tmp_path / "pause.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(pause.CHILD_ENV, str(path))
    row = check_pause()
    assert row.status == "error"
    assert "unreadable" in row.message.lower() and "paused" in row.message.lower()
    assert row.fix and "/resume" in row.fix


def test_the_doctor_runs_the_pause_row():
    import inspect

    from prometheus.infra.doctor import Doctor

    assert "check_pause()" in inspect.getsource(Doctor.diagnose)


# ── /curator run ─────────────────────────────────────────────────────────────

def test_curator_run_refuses_while_paused(state, monkeypatch):
    from prometheus.gateway import commands
    from prometheus.learning import curator

    ran: list[bool] = []

    class _Curator:
        async def run_once(self, *, dry_run=False):
            ran.append(dry_run)
            raise RuntimeError("stop here")

    monkeypatch.setattr(curator, "get_curator", lambda: _Curator())
    state.pause(by="owner", reason="")
    reply = asyncio.run(commands.cmd_curator_run())
    assert ran == [] and "paused" in reply.lower()


# ── review fixes (#920): the doctor row everywhere, odd times, the archive step, Apple, the cap ─────────────────

def test_the_pause_row_is_in_the_checks_that_always_run(tmp_path, monkeypatch):
    """Review of #920: the row came only from the deep scan, so `oara doctor --no-scan`, a missing config or a failed
    scan showed no Pause row and exited 0 with an unreadable pause file."""
    import inspect

    from prometheus.cli import doctor

    ok = doctor.DiagnosticCheck
    monkeypatch.setattr(doctor, "check_inference", lambda *a, **k: (ok("Inference", "connectivity", "ok", "x"),
                                                                    ok("Model", "model", "ok", "x")))
    monkeypatch.setattr(doctor, "check_web_port", lambda *a, **k: ok("Web", "connectivity", "ok", "x"))
    pause.install(None)
    path = tmp_path / "pause.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(pause.CHILD_ENV, str(path))
    rows = doctor.run_extended_checks({}, config_check=ok("Config", "platform", "ok", "x"))
    [row] = [r for r in rows if r.name == "Pause"]
    assert row.status == "error"
    assert '"Pause"' in inspect.getsource(doctor.run_anatomy_checks)        # not reported twice


def test_the_not_paused_row_names_the_file_it_read(state):
    from prometheus.infra.doctor import check_pause

    assert str(state.path) in check_pause().message


@pytest.mark.parametrize("since", ["Infinity", "NaN", "-Infinity", "1e300", "-5", "1" + "0" * 400])
def test_an_out_of_range_since_is_unreadable_not_a_crash(tmp_path, since):
    from prometheus.infra.doctor import check_pause

    path = tmp_path / "pause.json"
    path.write_text('{"paused": true, "since": ' + since + ', "by": "owner"}', encoding="utf-8")
    info = pause.read_file(path)
    assert info is not None and info.by == "unreadable"
    s = pause.PauseState(path)                                       # a daemon booting on it: paused, no raise
    pause.install(s)
    try:
        assert s.current is not None and check_pause().status == "error"
    finally:
        pause.install(None)


def test_a_pause_landing_during_the_curators_review_archives_nothing(state, tmp_path, monkeypatch):
    from prometheus.learning.curator import Curator
    from prometheus.learning.skill_state import SkillStateStore

    auto = tmp_path / "skills" / "auto"
    auto.mkdir(parents=True)
    (auto / "a.md").write_text("---\nname: a\n---\n# A\n")
    c = Curator(object(), auto_dir=auto, reports_dir=tmp_path / "curator",
                state_store=SkillStateStore(tmp_path / "_state.json"), interval_seconds=60, telemetry=None)

    async def _review(skills):
        state.pause(by="owner", reason="")                            # the pause lands while the model reviews
        return [], [{"name": "a", "reason": "stale"}], "{}"

    archived: list = []
    monkeypatch.setattr(c, "_review_via_llm", _review)
    monkeypatch.setattr(c, "_apply_prunings", lambda *a, **k: archived.append(a) or ([], [], []))
    asyncio.run(c.run_once())
    assert archived == [] and (auto / "a.md").exists()


def test_the_alert_does_not_carry_a_chat_id_to_apple(tmp_path):
    store, _ = _store(tmp_path)
    sender = _Sender()
    info = pause.PauseInfo(since=1.0, by="owner in Telegram (telegram:123456789)")
    asyncio.run(_dispatcher(store, sender).push_paused(info))
    assert sender.sent
    for s in sender.sent:
        assert "123456789" not in str(s["payload"]) and "owner in Telegram" in s["payload"]["aps"]["alert"]["body"]


def test_one_slow_device_does_not_keep_the_alert_from_the_others(tmp_path):
    from prometheus.config.device_store import DeviceStore

    store = DeviceStore(tmp_path / "devices.db")
    for name, tok in (("phone", "tok-1"), ("tablet", "tok-2")):
        d = store.mint_owner(name, "ios", by="test")
        assert store.set_push(d["id"], tok, "sandbox", "ai.test.beacon")
    tried: list[str] = []

    class _Slow(_Sender):
        async def send(self, **kwargs):
            tried.append(kwargs["apns_token"])
            if len(tried) == 1:
                await asyncio.Event().wait()                          # the first device never answers
            return await super().send(**kwargs)

    async def go():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.5):
                await _dispatcher(store, _Slow()).push_paused(pause.PauseInfo(since=1.0, by="owner"))

    asyncio.run(go())
    assert sorted(tried) == ["tok-1", "tok-2"]


def test_a_hanging_push_is_cut_off_and_the_pause_answers(w, monkeypatch):
    import time as _time

    from prometheus.web import pause_routes

    class _Hang:
        async def push_paused(self, info):
            await asyncio.Event().wait()

    monkeypatch.setattr(pause_routes, "PUSH_ALERT_TIMEOUT_S", 0.2)
    w.bridge.push_dispatcher = _Hang()
    started = _time.monotonic()
    r = w.call("op", "PUT", "/api/pause", json={})
    assert r.status_code == 200 and _time.monotonic() - started < 3
