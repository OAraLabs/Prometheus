"""A Live Activity for a job the phone did not start (Beacon iOS "Piece 18").

A Live Activity used to exist only when the app started one, in the foreground, for a turn sent from that phone.
A coding run, a background task, a desktop task, or a turn sent from the desktop never got one, and those are the
long jobs. The daemon now starts one by push (iOS 17.2 push-to-start) on the owner's devices once a job has run
for 60 seconds and is still running, updates it at most every 10 seconds, and ends it with how the job ended.

Held here, with a fake sender and an injected clock:

  * a job is started on a phone once it has run 60 s, once per device, never before, and never for a job that
    ended first;
  * owner devices get every kind; a scoped device only a turn in a session it owns;
  * no start for a turn the phone already shows (it registered an activity token for that session);
  * updates are throttled to one per job per 10 s, and every start and update carries ``stale-date`` (now + 45 s),
    so a daemon that dies mid-job shows as stale instead of looking alive;
  * the end says how the job ended (done, failed, stopped, paused) and drops the job's tokens;
  * a turn's session id and a job id never share a token row, even ``coding:<id>`` against coding job ``<id>``;
  * every body names the node (``node_pub``) and stays under APNs' 4 KB with long titles;
  * the content-state keys are one constant, and the phone's fixture is generated from it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import types
from pathlib import Path

import pytest

from prometheus.config.device_store import DeviceStore
from prometheus.push import job_activity
from prometheus.push.dispatcher import PushDispatcher
from prometheus.push.job_activity import (
    ATTRIBUTES_TYPE,
    CONTENT_STATE_KEYS,
    END_PHASES,
    JOB_KINDS,
    START_AFTER_SECONDS,
    STALE_AFTER_SECONDS,
    TITLE_CAP,
    DesktopSource,
    Job,
    JobActivityWatcher,
    TaskSource,
    TurnSource,
)

PUB = base64.b64encode(bytes(range(32))).decode()
BUNDLE = "ai.example.app"
REPO = Path(__file__).resolve().parent.parent


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.outcomes: dict[str, str] = {}
        self.config = types.SimpleNamespace(topic=BUNDLE)

    async def send(self, **kw):
        from prometheus.push.apns import SendResult

        self.sent.append(kw)
        outcome = self.outcomes.get(kw["apns_token"], "ok")
        return SendResult(outcome=outcome, status={"ok": 200, "unregistered": 410, "failed": 500}[outcome])

    def events(self, event: str | None = None) -> list[dict]:
        return [s for s in self.sent if event is None or s["payload"]["aps"].get("event") == event]


class _Source:
    """A fake job source: what is running now, and how a job ended."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.jobs: list[Job] = []
        self.outcomes: dict[str, str] = {}
        self.raises = False

    def running(self):
        if self.raises:
            raise RuntimeError("source unavailable")
        return list(self.jobs)

    def outcome(self, job_id: str):
        return self.outcomes.get(job_id)


class _World:
    def __init__(self, tmp_path) -> None:
        self.store = DeviceStore(tmp_path / "devices.db")
        self.owner = self.store.mint_owner("owner phone", "ios", by="same-mac-pairing")
        self.scoped = self.store.mint("scoped phone", "ios")
        for d, n in ((self.owner, "o"), (self.scoped, "s")):
            self.store.set_push(d["id"], f"alert-{n}", "sandbox", BUNDLE)
            assert self.store.set_activity_start_token(d["id"], f"start-{n}")
        self.store.claim_session("web:mine", self.scoped["id"])
        self.clock = {"t": 1000.0}
        self.sender = _Sender()
        self.dispatcher = PushDispatcher(self.store, self.sender, bridge=None,
                                         now=lambda: self.clock["t"], node_pub=lambda: PUB)
        self.sources = {k: _Source(k) for k in JOB_KINDS}
        self.watcher = JobActivityWatcher(self.dispatcher, self.store, sources=list(self.sources.values()),
                                          now=lambda: self.clock["t"])

    def at(self, t: float) -> None:
        self.clock["t"] = t
        asyncio.run(self.watcher.tick())

    def add(self, kind: str, job_id: str, *, session_id: str | None = None, title: str = "A job",
            started_at: float = 1000.0, detail: str = "", steps: int = 0) -> Job:
        job = Job(kind=kind, id=job_id, title=title, started_at=started_at, session_id=session_id,
                  detail=detail, steps=steps)
        self.sources[kind].jobs.append(job)
        return job

    def end(self, kind: str, job_id: str, phase: str) -> None:
        src = self.sources[kind]
        src.jobs = [j for j in src.jobs if j.id != job_id]
        src.outcomes[job_id] = phase


@pytest.fixture
def w(tmp_path) -> _World:
    return _World(tmp_path)


def _tokens(sent: list[dict]) -> list[str]:
    return [s["apns_token"] for s in sent]


# ── start ─────────────────────────────────────────────────────────────────────

def test_a_job_that_has_run_60_seconds_starts_one_activity_per_owner_device(w):
    w.add("coding", "c1", session_id="coding:c1", title="Coding run", detail="edit_file", steps=3)
    w.at(1000 + START_AFTER_SECONDS - 1)
    assert w.sender.sent == [], "59 s is not a minute"
    w.at(1000 + START_AFTER_SECONDS)
    [start] = w.sender.events("start")
    assert start["apns_token"] == "start-o", "the owner device's push-to-start token, and only the owner's"
    assert start["push_type"] == "liveactivity"
    assert start["topic"] == f"{BUNDLE}.push-type.liveactivity"
    assert start["priority"] == 10
    aps = start["payload"]["aps"]
    assert aps["attributes-type"] == ATTRIBUTES_TYPE == "JobActivityAttributes"
    assert aps["attributes"] == {"kind": "coding", "id": "c1", "title": "Coding run",
                                 "session_id": "coding:c1", "node_pub": PUB}
    assert list(aps["content-state"]) == list(CONTENT_STATE_KEYS)
    assert aps["content-state"] == {"phase": "running", "detail": "edit_file", "elapsedS": 60, "steps": 3}
    assert aps["stale-date"] == int(w.clock["t"] + STALE_AFTER_SECONDS)
    assert aps["timestamp"] == int(w.clock["t"])
    assert start["payload"]["node_pub"] == PUB, "every push body names the node (_named)"

    for t in (1070, 1080, 1500):
        w.at(t)
    assert len(w.sender.events("start")) == 1, "one start per job per device, ever"


def test_a_job_that_ends_before_60_seconds_starts_nothing(w):
    w.add("task", "t1")
    w.at(1030)
    w.end("task", "t1", "done")
    w.at(1070)
    w.at(1080)
    assert w.sender.sent == []


def test_a_device_that_registers_late_still_gets_its_one_start(w):
    w.store.clear_activity_start_token(w.owner["id"])
    w.add("task", "t1")
    w.at(1060)
    assert w.sender.sent == []
    w.store.set_activity_start_token(w.owner["id"], "start-o2")
    w.at(1070)
    w.at(1080)
    assert _tokens(w.sender.events("start")) == ["start-o2"]


def test_a_scoped_device_gets_a_start_only_for_a_turn_in_its_own_session(w):
    w.add("turn", "web:mine", session_id="web:mine")
    w.add("turn", "web:theirs", session_id="web:theirs")
    w.add("task", "t1")
    w.add("coding", "c1", session_id="coding:c1")
    w.add("desktop", "d1", session_id="web:mine")
    w.at(1060)
    by_job = {}
    for s in w.sender.events("start"):
        by_job.setdefault(s["payload"]["aps"]["attributes"]["id"], set()).add(s["apns_token"])
    assert by_job == {
        "web:mine": {"start-o", "start-s"},
        "web:theirs": {"start-o"},
        "t1": {"start-o"},
        "c1": {"start-o"},
        "d1": {"start-o"},
    }


def test_a_revoked_device_gets_nothing(w):
    w.store.revoke(w.owner["id"])
    w.add("task", "t1")
    w.at(1060)
    assert w.sender.sent == []


def test_no_start_for_a_turn_the_phone_already_shows(w):
    w.store.set_activity_token(w.owner["id"], "web:mine", "in-app-activity")
    w.add("turn", "web:mine", session_id="web:mine")
    w.at(1060)
    assert _tokens(w.sender.events("start")) == ["start-s"], "the owner phone started its own; the other did not"


def test_no_start_for_a_job_the_phone_already_registered(w):
    w.store.set_job_activity_token(w.owner["id"], "coding", "c1", "in-app-job")
    w.add("coding", "c1")
    w.at(1060)
    assert w.sender.events("start") == []


def test_a_410_on_a_start_token_drops_that_token_and_nothing_else(w):
    w.sender.outcomes["start-o"] = "unregistered"
    w.add("task", "t1")
    w.at(1060)
    assert [t.id for t, _ in w.store.activity_start_targets()] == [w.scoped["id"]]
    assert {t.id for t in w.store.push_targets()} == {w.owner["id"], w.scoped["id"]}, \
        "the device's alert registration is a different token and stays"


# ── updates ───────────────────────────────────────────────────────────────────

def test_updates_are_throttled_to_one_per_job_per_10_seconds_and_carry_stale_date(w):
    w.add("coding", "c1", detail="bash", steps=1)
    w.at(1060)
    w.store.set_job_activity_token(w.owner["id"], "coding", "c1", "upd-o")
    for t in (1061, 1065, 1069.9):
        w.at(t)
    assert w.sender.events("update") == [], "inside 10 s of the start"
    w.at(1070)
    w.at(1075)
    w.at(1080)
    updates = w.sender.events("update")
    assert len(updates) == 2
    for u in updates:
        assert u["apns_token"] == "upd-o"
        assert u["push_type"] == "liveactivity" and u["topic"] == f"{BUNDLE}.push-type.liveactivity"
        aps = u["payload"]["aps"]
        assert list(aps["content-state"]) == list(CONTENT_STATE_KEYS)
        assert aps["stale-date"] == aps["timestamp"] + STALE_AFTER_SECONDS
        assert u["payload"]["node_pub"] == PUB
    assert updates[-1]["payload"]["aps"]["content-state"]["elapsedS"] == 80


def test_a_job_the_phone_started_itself_is_updated_before_a_minute(w):
    w.store.set_job_activity_token(w.owner["id"], "task", "t1", "upd-o")
    w.add("task", "t1")
    w.at(1010)
    w.at(1020)
    assert _tokens(w.sender.events("update")) == ["upd-o", "upd-o"]
    assert w.sender.events("start") == []


# ── end ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phase", ["done", "failed", "stopped", "paused"])
def test_the_end_says_how_the_job_ended_and_drops_its_tokens(w, phase):
    assert phase in END_PHASES
    w.add("task", "t1")
    w.at(1060)
    w.store.set_job_activity_token(w.owner["id"], "task", "t1", "upd-o")
    w.end("task", "t1", phase)
    w.at(1070)
    [end] = w.sender.events("end")
    assert end["apns_token"] == "upd-o"
    aps = end["payload"]["aps"]
    assert aps["content-state"]["phase"] == phase
    assert list(aps["content-state"]) == list(CONTENT_STATE_KEYS)
    assert "dismissal-date" in aps
    assert end["payload"]["node_pub"] == PUB
    assert w.store.job_activity_targets("task", "t1") == []
    w.at(1080)
    assert len(w.sender.events("end")) == 1


def test_a_job_whose_source_is_unreadable_is_not_ended(w):
    w.add("task", "t1")
    w.at(1060)
    w.store.set_job_activity_token(w.owner["id"], "task", "t1", "upd-o")
    w.sources["task"].raises = True
    w.at(1070)
    assert w.sender.events("end") == [], "a source that failed to answer is not a job that ended"
    w.sources["task"].raises = False
    w.at(1080)
    assert len(w.sender.events("update")) == 1 and w.sender.events("end") == []


def test_turn_updates_and_end_stay_with_the_bridge_path(w):
    """A turn's updates already ride agent_progress and its end rides chat_done; the watcher only starts it."""
    w.add("turn", "web:mine", session_id="web:mine")
    w.at(1060)
    w.store.set_activity_token(w.owner["id"], "web:mine", "turn-upd")
    w.at(1070)
    w.sources["turn"].jobs = []
    w.at(1080)
    assert w.sender.events("update") == [] and w.sender.events("end") == []


# ── kinds apart ───────────────────────────────────────────────────────────────

def test_a_turns_session_id_and_a_job_id_never_share_a_token(w):
    dev = w.owner["id"]
    w.store.set_activity_token(dev, "coding:c1", "turn-tok")
    w.store.set_job_activity_token(dev, "coding", "c1", "coding-tok")
    w.store.set_job_activity_token(dev, "task", "c1", "task-tok")
    assert [tok for _, tok in w.store.activity_targets("coding:c1")] == ["turn-tok"]
    assert [tok for _, tok in w.store.job_activity_targets("coding", "c1")] == ["coding-tok"]
    assert [tok for _, tok in w.store.job_activity_targets("task", "c1")] == ["task-tok"]
    assert w.store.job_activity_targets("coding", "coding:c1") == []
    w.store.clear_job_activity_tokens("coding", "c1")
    assert [tok for _, tok in w.store.activity_targets("coding:c1")] == ["turn-tok"]
    assert [tok for _, tok in w.store.job_activity_targets("task", "c1")] == ["task-tok"]


def test_the_end_of_a_coding_job_reaches_only_that_jobs_token(w):
    dev = w.owner["id"]
    w.add("coding", "c1", session_id="coding:c1")
    w.at(1060)
    w.store.set_activity_token(dev, "coding:c1", "turn-tok")
    w.store.set_job_activity_token(dev, "coding", "c1", "coding-tok")
    w.end("coding", "c1", "done")
    w.at(1070)
    assert _tokens(w.sender.events("end")) == ["coding-tok"]
    assert [tok for _, tok in w.store.activity_targets("coding:c1")] == ["turn-tok"]


def test_the_start_token_never_appears_in_the_device_listing(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    store.set_activity_start_token(d["id"], "secret-start-token")
    assert "secret-start-token" not in repr(store.list_devices())


def test_an_existing_database_gains_the_start_tokens_safely(tmp_path):
    first = DeviceStore(tmp_path / "devices.db")
    d = first.mint("phone", "ios")
    first.set_push(d["id"], "alert", "sandbox", BUNDLE)
    first.close()
    again = DeviceStore(tmp_path / "devices.db")
    assert again.activity_start_targets() == []
    assert again.set_activity_start_token(d["id"], "start")
    assert again.set_activity_start_token("no-such-device", "start") is False
    [(target, token)] = again.activity_start_targets()
    assert (target.id, token, target.environment, target.bundle_id) == (d["id"], "start", "sandbox", BUNDLE)
    again.close()
    third = DeviceStore(tmp_path / "devices.db")
    assert [tok for _, tok in third.activity_start_targets()] == ["start"], "durable"
    assert third.clear_activity_start_token(d["id"])
    assert third.activity_start_targets() == []


# ── size and node ─────────────────────────────────────────────────────────────

def test_long_titles_are_capped_and_every_body_stays_under_4kb(w):
    w.add("coding", "i" * 5000, session_id="s" * 5000, title="t" * 20000 + "\nsecond line", detail="d" * 20000)
    w.at(1060)
    w.store.set_job_activity_token(w.owner["id"], "coding", "i" * 5000, "upd-o")
    w.at(1070)
    w.end("coding", "i" * 5000, "done")
    w.at(1080)
    assert {s["payload"]["aps"]["event"] for s in w.sender.sent} == {"start", "update", "end"}
    for s in w.sender.sent:
        assert len(json.dumps(s["payload"]).encode()) < 4096
    attrs = w.sender.events("start")[0]["payload"]["aps"]["attributes"]
    assert len(attrs["title"]) <= TITLE_CAP


def test_no_node_key_means_no_node_pub_anywhere(w):
    w.dispatcher._node_pub = lambda: None
    w.add("task", "t1")
    w.at(1060)
    [start] = w.sender.events("start")
    assert "node_pub" not in start["payload"]
    assert "node_pub" not in start["payload"]["aps"]["attributes"]


# ── the sources: what each kind shows ────────────────────────────────────────

def test_the_turn_source_shows_the_tool_name_and_never_its_arguments():
    progress = {"phase": "tool", "tool_name": "bash", "round": 2, "chars": 10, "tool_calls": 4,
                "inputs": {"command": "cat secrets"}}
    bridge = types.SimpleNamespace(running_turns=lambda: [("web:a", 900.0, progress)])
    [job] = list(TurnSource(bridge, known_tool=lambda name: name == "bash").running())
    assert (job.kind, job.id, job.session_id, job.started_at) == ("turn", "web:a", "web:a", 900.0)
    assert (job.phase, job.detail, job.steps) == ("tool", "bash", 4)
    assert "secrets" not in repr(job)


def _task(task_id, *, status="running", session_id=None, description="run the tests", started_at=900.0,
          task_type="local_bash", metadata=None):
    return types.SimpleNamespace(id=task_id, status=status, session_id=session_id, description=description,
                                 started_at=started_at, created_at=started_at - 1, type=task_type,
                                 metadata=metadata or {}, command="echo secret-arg", prompt="secret prompt")


def test_the_task_source_separates_tasks_from_coding_runs():
    records = {"t1": _task("t1"), "c1": _task("c1", session_id="coding:c1",
                                                 description="coding task c1: fix it", task_type="local_agent")}
    manager = types.SimpleNamespace(running_tasks=lambda: list(records.values()),
                                    get_task=lambda i: records.get(i))
    tasks = list(TaskSource(manager, kind="task").running())
    coding = list(TaskSource(manager, kind="coding").running())
    assert [j.id for j in tasks] == ["t1"] and [j.id for j in coding] == ["c1"]
    assert tasks[0].title == "Background task: shell command"
    assert coding[0].title == "Coding run"
    assert coding[0].session_id == "coding:c1"
    assert "secret" not in repr(tasks + coding), "never the command or the prompt"
    assert "run the tests" not in repr(tasks + coding) and "fix it" not in repr(coding), "never the description"


@pytest.mark.parametrize("status, phase", [("completed", "done"), ("failed", "failed"),
                                           ("blocked", "failed"), ("killed", "stopped")])
def test_the_task_source_says_how_a_task_ended(status, phase):
    rec = _task("t1", status=status)
    manager = types.SimpleNamespace(running_tasks=lambda: [], get_task=lambda i: rec)
    assert TaskSource(manager, kind="task").outcome("t1") == phase


def test_a_paused_coding_run_shows_paused():
    rec = _task("c1", session_id="coding:c1", task_type="local_agent")
    manager = types.SimpleNamespace(running_tasks=lambda: [rec], get_task=lambda i: rec)
    [job] = list(TaskSource(manager, kind="coding", is_paused=lambda record: True).running())
    assert job.phase == "paused"
    [job] = list(TaskSource(manager, kind="coding", is_paused=lambda record: False).running())
    assert job.phase == "running"


def test_the_coding_source_shows_the_last_tool_and_counts_rounds():
    rec = _task("c1", session_id="coding:c1", task_type="local_agent")
    manager = types.SimpleNamespace(running_tasks=lambda: [rec], get_task=lambda i: rec)
    source = TaskSource(manager, kind="coding", known_tool=lambda name: name in ("edit_file", "bash"))
    sig = lambda kind, payload: types.SimpleNamespace(kind=kind, payload=payload)  # noqa: E731
    asyncio.run(source.on_signal(sig("coding_round", {"session_id": "coding:c1", "round_index": 0})))
    asyncio.run(source.on_signal(sig("coding_round", {"session_id": "coding:c1", "round_index": 1})))
    asyncio.run(source.on_signal(sig("coding_tool", {"session_id": "coding:c1", "tool_name": "edit_file",
                                                     "success": True})))
    asyncio.run(source.on_signal(sig("coding_tool", {"session_id": "coding:other", "tool_name": "bash"})))
    [job] = list(source.running())
    assert (job.detail, job.steps) == ("edit_file", 2)


def _desktop(task_id, *, status="running", outcome=None, steps=2):
    return types.SimpleNamespace(task_id=task_id, session_id="web:a", goal="type my password into the bank",
                                 app="Bank", status=status, outcome=outcome, steps=steps, started_at=900.0)


def test_the_desktop_source_carries_no_content():
    """Like a desktop task's approval push: not the goal, not the app."""
    rec = _desktop("d1")
    runner = types.SimpleNamespace(running=lambda: [rec], get=lambda i: rec)
    [job] = list(DesktopSource(runner).running())
    assert (job.kind, job.id, job.session_id, job.steps) == ("desktop", "d1", "web:a", 2)
    assert "password" not in repr(job) and "Bank" not in repr(job)


@pytest.mark.parametrize("outcome, phase", [("done", "done"), ("abstained", "done"), ("stopped", "stopped"),
                                            ("refused", "failed"), ("failed", "failed"), ("limit", "failed")])
def test_the_desktop_source_says_how_a_task_ended(outcome, phase):
    rec = _desktop("d1", status="ended", outcome=outcome)
    runner = types.SimpleNamespace(running=lambda: [], get=lambda i: rec)
    assert DesktopSource(runner).outcome("d1") == phase


# ── a turn's own updates (the bridge path) carry the job keys and stale-date ──

class _TurnStore:
    def activity_targets(self, session_id):
        return [(types.SimpleNamespace(id="d1", apns_token="a", environment="sandbox", bundle_id=BUNDLE,
                                       push_failures=0), "turn-tok")]

    def clear_activity_token(self, device_id, session_id):
        pass


def test_a_turn_update_carries_the_job_keys_and_a_stale_date():
    sender = _Sender()
    d = PushDispatcher(_TurnStore(), sender, bridge=None, now=lambda: 5000.0, node_pub=lambda: PUB)
    d.known_tool = lambda name: name == "bash"
    asyncio.run(d.on_agent_progress("web:a", {"phase": "tool", "tool_name": "bash", "round": 2,
                                              "elapsed_s": 70.4, "tool_calls": 3, "chars": 9}))
    aps = sender.sent[0]["payload"]["aps"]
    assert aps["stale-date"] == 5000 + STALE_AFTER_SECONDS
    state = aps["content-state"]
    assert set(CONTENT_STATE_KEYS) <= set(state)
    assert (state["phase"], state["detail"], state["steps"]) == ("tool", "bash", 3)


@pytest.mark.parametrize("payload, phase", [
    ({}, "done"),
    ({"interrupted": True}, "stopped"),
    ({"stop_reason": "max_turns_exhausted"}, "paused"),
    ({"failed": True}, "failed"),
])
def test_a_turns_end_says_how_it_ended(payload, phase):
    sender = _Sender()
    d = PushDispatcher(_TurnStore(), sender, bridge=None, now=lambda: 5000.0, node_pub=lambda: PUB)
    asyncio.run(d.on_chat_done("web:a", {"session_id": "web:a", **payload}))
    state = sender.sent[0]["payload"]["aps"]["content-state"]
    assert set(CONTENT_STATE_KEYS) <= set(state)
    assert state["phase"] == "done", "the shipped per-turn decode reads phase; it stays exactly done"
    assert state["outcome"] == phase


# ── the phone's fixture ───────────────────────────────────────────────────────

def test_the_phones_fixture_is_generated_from_the_constants():
    on_disk = json.loads((REPO / "docs" / "reference" / "job-activity.json").read_text(encoding="utf-8"))
    assert on_disk == job_activity.fixture(), \
        "docs/reference/job-activity.json drifted: regenerate with uv run python scripts/gen_reference.py"
    assert on_disk["content_state_keys"] == list(CONTENT_STATE_KEYS)
    assert on_disk["attributes_type"] == ATTRIBUTES_TYPE
    assert on_disk["kinds"] == list(JOB_KINDS)
    assert on_disk["end_phases"] == list(END_PHASES)
    assert on_disk["turn_end_outcome_key"] == "outcome"


def test_the_fixture_follows_a_changed_constant(monkeypatch):
    monkeypatch.setattr(job_activity, "CONTENT_STATE_KEYS", (*CONTENT_STATE_KEYS, "extra"))
    assert job_activity.fixture()["content_state_keys"][-1] == "extra"


# ── review fixes ──────────────────────────────────────────────────────────────

DISTINCT = "ZQX-distinctive-goal-7781"


def test_h1_a_description_never_reaches_any_body(tmp_path):
    """A task's description and a coding goal are user or model text: no push carries them."""
    w = _World(tmp_path)
    records = {
        "t1": _task("t1", description=f"{DISTINCT} task", started_at=1000.0),
        "c1": _task("c1", session_id="coding:c1", description=f"coding task c1: {DISTINCT}",
                    task_type="local_agent", started_at=1000.0),
    }
    manager = types.SimpleNamespace(running_tasks=lambda: [r for r in records.values() if r.status == "running"],
                                    get_task=lambda i: records.get(i))
    w.watcher = JobActivityWatcher(w.dispatcher, w.store, sources=[
        TaskSource(manager, kind="task"), TaskSource(manager, kind="coding", is_paused=lambda r: False)],
        now=lambda: w.clock["t"])
    w.at(1060)
    for kind, job_id in (("task", "t1"), ("coding", "c1")):
        w.store.set_job_activity_token(w.owner["id"], kind, job_id, f"upd-{job_id}")
    w.at(1070)
    for r in records.values():
        r.status = "completed"
    w.at(1080)
    sig = lambda kind, payload: types.SimpleNamespace(kind=kind, payload=payload)  # noqa: E731
    asyncio.run(w.dispatcher.on_signal(sig("task_completed", {"task_id": "t1", "status": "completed",
                                                              "description": f"{DISTINCT} task",
                                                              "session_id": "web:mine"})))
    assert {s["payload"]["aps"].get("event") for s in w.sender.sent} >= {"start", "update", "end"}
    for sent in w.sender.sent:
        assert DISTINCT not in json.dumps(sent["payload"]), sent["payload"]


def _started_then_ended(w, kind="task", job_id="t1", phase="done"):
    w.add(kind, job_id, session_id=job_id if kind == "turn" else None)
    w.at(1060)
    w.end(kind, job_id, phase)
    w.at(1070)


def test_m1_a_job_that_ended_before_the_phone_registered_is_ended_at_once(w):
    _started_then_ended(w, phase="failed")
    assert w.sender.events("end") == []
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late-tok")) is True
    [end] = w.sender.events("end")
    assert end["apns_token"] == "late-tok"
    assert end["payload"]["aps"]["content-state"]["phase"] == "failed"
    assert end["payload"]["node_pub"] == PUB
    assert w.store.job_activity_targets("task", "t1") == []


def test_m1_the_tombstone_lasts_about_ten_minutes(w):
    _started_then_ended(w)
    w.clock["t"] = 1070 + job_activity.TOMBSTONE_SECONDS + 1
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late-tok")) is False
    assert w.sender.events("end") == []


def test_m1_a_running_job_is_not_ended(w):
    w.add("task", "t1")
    w.at(1060)
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "tok")) is False


def test_m1_a_turn_whose_start_was_sent_and_which_ended_is_ended_at_once(w):
    w.add("turn", "web:mine", session_id="web:mine")
    w.at(1060)
    asyncio.run(w.dispatcher.on_chat_done("web:mine", {"session_id": "web:mine", "interrupted": True,
                                                        "turn_started_at": 1000.0}))
    w.sources["turn"].jobs = []
    w.at(1070)
    assert asyncio.run(w.watcher.end_if_ended(w.scoped["id"], "turn", "web:mine", "late-tok")) is True
    [end] = w.sender.events("end")
    state = end["payload"]["aps"]["content-state"]
    assert (state["phase"], state["outcome"]) == ("done", "stopped"), "the same shape as the chat_done end"


def test_m1_a_turn_tombstone_is_only_for_a_device_sent_its_start_and_not_while_a_new_turn_runs(w):
    w.add("turn", "web:theirs", session_id="web:theirs")
    w.at(1060)
    w.sources["turn"].jobs = []
    w.at(1070)
    assert asyncio.run(w.watcher.end_if_ended(w.scoped["id"], "turn", "web:theirs", "x")) is False, \
        "the scoped device was never sent that start"
    w.add("turn", "web:theirs", session_id="web:theirs", started_at=1072)
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "turn", "web:theirs", "x")) is False, \
        "a new turn runs in that session: the token is for it"


def test_m2_a_turn_activity_on_a_device_without_alert_push_is_reached(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    store.set_activity_token(d["id"], "web:a", "act")
    assert [tok for _, tok in store.activity_targets("web:a")] == ["act"]
    sender = _Sender()
    disp = PushDispatcher(store, sender, bridge=None, now=lambda: 5000.0, node_pub=lambda: PUB)
    asyncio.run(disp.on_agent_progress("web:a", {"phase": "thinking"}))
    asyncio.run(disp.on_chat_done("web:a", {"session_id": "web:a"}))
    assert [s["payload"]["aps"]["event"] for s in sender.sent] == ["update", "end"]
    assert all(s["topic"] == f"{BUNDLE}.push-type.liveactivity" for s in sender.sent)


def test_m2_a_410_on_a_turn_activity_token_drops_that_token_not_the_alert_registration(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    store.set_push(d["id"], "alert", "sandbox", BUNDLE)
    store.set_activity_token(d["id"], "web:a", "act")
    sender = _Sender()
    sender.outcomes["act"] = "unregistered"
    disp = PushDispatcher(store, sender, bridge=None, now=lambda: 5000.0, node_pub=lambda: PUB)
    asyncio.run(disp.on_agent_progress("web:a", {"phase": "thinking"}))
    assert store.activity_targets("web:a") == []
    assert [t.id for t in store.push_targets()] == [d["id"]]


def test_l1_the_boot_sweep_ends_every_leftover_job_activity(w):
    w.store.set_job_activity_token(w.owner["id"], "coding", "old-1", "left-1")
    w.store.set_job_activity_token(w.scoped["id"], "task", "old-2", "left-2")
    w.store.set_activity_token(w.owner["id"], "web:mine", "turn-tok")
    asyncio.run(w.watcher.sweep())
    ends = w.sender.events("end")
    assert sorted(_tokens(ends)) == ["left-1", "left-2"]
    assert all(e["payload"]["aps"]["content-state"]["phase"] == "stopped" for e in ends)
    assert w.store.job_activity_targets("coding", "old-1") == [] and w.store.job_activity_targets("task", "old-2") == []
    assert [tok for _, tok in w.store.activity_targets("web:mine")] == ["turn-tok"], "a turn's token is not a job's"


def test_l2_back_to_back_turns_in_one_session_are_separate_jobs(w):
    w.add("turn", "web:mine", session_id="web:mine", started_at=1000)
    w.at(1060)
    w.sources["turn"].jobs = []
    w.add("turn", "web:mine", session_id="web:mine", started_at=1062)
    w.at(1065)
    w.at(1125)
    starts = [s for s in w.sender.events("start") if s["apns_token"] == "start-o"]
    assert len(starts) == 2


def test_l3_detail_is_a_tool_name_only_when_the_tool_is_registered():
    progress = {"phase": "tool", "tool_name": "rm -rf / please", "tool_calls": 1}
    bridge = types.SimpleNamespace(running_turns=lambda: [("web:a", 900.0, progress)])
    [job] = list(TurnSource(bridge, known_tool=lambda name: name == "bash").running())
    assert job.detail == job_activity.GENERIC_TOOL_DETAIL
    [job] = list(TurnSource(bridge).running())
    assert job.detail == job_activity.GENERIC_TOOL_DETAIL, "no registry to ask: never the raw name"
    sender = _Sender()
    d = PushDispatcher(_TurnStore(), sender, bridge=None, now=lambda: 5000.0, node_pub=lambda: PUB)
    asyncio.run(d.on_agent_progress("web:a", {"phase": "tool", "tool_name": "rm -rf / please"}))
    assert sender.sent[0]["payload"]["aps"]["content-state"]["detail"] == job_activity.GENERIC_TOOL_DETAIL


def test_l4_one_bad_job_does_not_stop_the_rest(w, monkeypatch):
    real = w.store.job_activity_targets

    def flaky(kind, job_id):
        if job_id == "bad":
            raise RuntimeError("boom")
        return real(kind, job_id)

    monkeypatch.setattr(w.store, "job_activity_targets", flaky)
    w.add("task", "bad")
    w.add("task", "good")
    w.at(1060)
    assert [s["payload"]["aps"]["attributes"]["id"] for s in w.sender.events("start")] == ["good"]
    w.store.set_job_activity_token(w.owner["id"], "task", "good", "upd-good")
    w.end("task", "bad", "done")
    w.end("task", "good", "done")
    w.at(1070)
    assert _tokens(w.sender.events("end")) == ["upd-good"]


# ── round-2 review ────────────────────────────────────────────────────────────

def test_a_turn_that_ends_between_ticks_is_tombstoned_when_it_ends(w):
    """Start at 60 s, end at 61 s, token at 63 s: no tick in between, and the end is still sent."""
    w.add("turn", "web:mine", session_id="web:mine", started_at=1000.0)
    w.at(1060)
    w.clock["t"] = 1061
    asyncio.run(w.dispatcher.on_chat_done("web:mine", {"session_id": "web:mine", "turn_started_at": 1000.0,
                                                        "stop_reason": "max_turns_exhausted"}))
    w.sources["turn"].jobs = []
    w.clock["t"] = 1063
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "turn", "web:mine", "late-tok")) is True
    [end] = [e for e in w.sender.events("end") if e["apns_token"] == "late-tok"]
    state = end["payload"]["aps"]["content-state"]
    assert (state["phase"], state["outcome"]) == ("done", "paused")
    w.at(1070)
    assert len([e for e in w.sender.events("end") if e["apns_token"] == "late-tok"]) == 1


def test_a_later_turn_that_was_never_started_does_not_overwrite_the_tombstone(w):
    w.add("turn", "web:mine", session_id="web:mine", started_at=1000.0)
    w.at(1060)
    asyncio.run(w.dispatcher.on_chat_done("web:mine", {"session_id": "web:mine", "turn_started_at": 1000.0,
                                                        "interrupted": True}))
    w.sources["turn"].jobs = []
    w.at(1061)
    w.add("turn", "web:mine", session_id="web:mine", started_at=1062.0)
    w.at(1063)
    asyncio.run(w.dispatcher.on_chat_done("web:mine", {"session_id": "web:mine", "turn_started_at": 1062.0}))
    w.sources["turn"].jobs = []
    w.at(1064)
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "turn", "web:mine", "late-tok")) is True
    [end] = w.sender.events("end")
    assert end["payload"]["aps"]["content-state"]["outcome"] == "stopped"


def test_a_tombstone_answers_each_device_once(w):
    _started_then_ended(w)
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late-1")) is True
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late-2")) is False
    assert _tokens(w.sender.events("end")) == ["late-1"]


class _AliveSource(_Source):
    def __init__(self, kind: str, alive: set[str]) -> None:
        super().__init__(kind)
        self._alive = alive

    def alive(self, job_id: str) -> bool:
        return job_id in self._alive


def test_the_sweep_skips_a_job_that_is_running_again_after_resume(w):
    w.sources["task"] = _AliveSource("task", {"resumed"})
    w.watcher = JobActivityWatcher(w.dispatcher, w.store, sources=list(w.sources.values()),
                                   now=lambda: w.clock["t"])
    w.store.set_job_activity_token(w.owner["id"], "task", "resumed", "keep")
    w.store.set_job_activity_token(w.owner["id"], "task", "gone", "end-me")
    asyncio.run(w.watcher.sweep())
    assert _tokens(w.sender.events("end")) == ["end-me"]
    assert [tok for _, tok in w.store.job_activity_targets("task", "resumed")] == ["keep"]
    assert w.store.job_activity_targets("task", "gone") == []


def test_the_sweep_deletes_only_the_rows_it_sent_to(w):
    from prometheus.push.dispatcher import FAILURE_MUTE_THRESHOLD

    for _ in range(FAILURE_MUTE_THRESHOLD):
        w.store.record_push_failure(w.scoped["id"])
    w.store.set_job_activity_token(w.owner["id"], "task", "a", "sent")
    w.store.set_job_activity_token(w.scoped["id"], "task", "b", "muted")
    asyncio.run(w.watcher.sweep())
    assert _tokens(w.sender.events("end")) == ["sent"]
    assert w.store.job_activity_targets("task", "a") == []
    assert [tok for _, tok in w.store.job_activity_targets("task", "b")] == ["muted"]


def test_the_sweep_is_capped(w):
    for i in range(job_activity.SWEEP_CAP + 5):
        w.store.set_job_activity_token(w.owner["id"], "task", f"j{i}", f"tok{i}")
    asyncio.run(w.watcher.sweep())
    assert len(w.sender.events("end")) == job_activity.SWEEP_CAP
    assert len(w.store.all_job_activity_tokens()) == 5


def test_the_task_source_says_a_resumable_task_is_alive():
    records = {"r": _task("r", status="running", task_type="file_watch"), "d": _task("d", status="completed")}
    manager = types.SimpleNamespace(running_tasks=lambda: [], get_task=lambda i: records.get(i))
    source = TaskSource(manager, kind="task")
    assert source.alive("r") is True and source.alive("d") is False and source.alive("nope") is False


def test_the_first_tick_does_not_wait_for_the_sweep(w, monkeypatch):
    order: list[str] = []

    async def slow_sweep():
        await asyncio.sleep(0.2)
        order.append("sweep")

    async def tick():
        order.append("tick")
        raise asyncio.CancelledError

    w.watcher._tick_seconds = 0.01
    monkeypatch.setattr(w.watcher, "sweep", slow_sweep)
    monkeypatch.setattr(w.watcher, "tick", tick)

    async def go():
        with pytest.raises(asyncio.CancelledError):
            await w.watcher.run()
        await asyncio.sleep(0.3)

    asyncio.run(go())
    assert order[0] == "tick"


# ── round-3 review ────────────────────────────────────────────────────────────

def test_a_tick_between_chat_done_and_the_bridges_cleanup_does_not_restart_the_turn(w):
    """chat_done fires while the bridge still lists the turn (its finally has not run yet)."""
    w.add("turn", "web:mine", session_id="web:mine", started_at=1000.0)
    w.at(1060)
    starts = len(w.sender.events("start"))
    asyncio.run(w.dispatcher.on_chat_done("web:mine", {"session_id": "web:mine", "turn_started_at": 1000.0,
                                                        "stop_reason": "max_turns_exhausted"}))
    w.at(1061)  # a tick in that window
    assert len(w.sender.events("start")) == starts, "no second push-to-start for a turn that ended"
    w.sources["turn"].jobs.clear()
    w.at(1070)
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "turn", "web:mine", "late")) is True
    [end] = [e for e in w.sender.events("end") if e["apns_token"] == "late"]
    assert end["payload"]["aps"]["content-state"]["outcome"] == "paused", "the real outcome, not stopped"


def test_the_ended_marker_does_not_block_the_next_turn(w):
    w.add("turn", "web:mine", session_id="web:mine", started_at=1000.0)
    w.at(1060)
    asyncio.run(w.dispatcher.on_chat_done("web:mine", {"session_id": "web:mine", "turn_started_at": 1000.0}))
    w.sources["turn"].jobs.clear()
    w.add("turn", "web:mine", session_id="web:mine", started_at=1062.0)
    w.at(1125)
    assert len([s for s in w.sender.events("start") if s["apns_token"] == "start-o"]) == 2


def test_a_failed_late_end_can_be_retried(w):
    _started_then_ended(w)
    w.sender.outcomes["late"] = "failed"
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late")) is False
    w.sender.outcomes.pop("late")
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late")) is True
    assert asyncio.run(w.watcher.end_if_ended(w.owner["id"], "task", "t1", "late")) is False
