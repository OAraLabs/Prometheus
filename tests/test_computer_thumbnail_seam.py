"""PR 6b — the SEAM: what the wiring does, not what the rules decide.

``test_computer_thumbnails.py`` proves ``decide_capture`` returns the right
answer. That is not the same as proving the runner and the action log USE it
correctly, and the gap between those two is where this feature can silently
fail: a rule that is right but never called, a frame emitted to the wrong
channel, a picture written to disk when persist is off.

Every assertion here is about a boundary:
  * what reached the SignalBus (and what provably did NOT);
  * what reached the sink;
  * the seq the image frame carries vs the step frame's;
  * whether the driver's capture was called AT ALL;
  * what landed on disk, and with what mode.

RECURRING §4h: verify the consumed layer, not the changed one.
"""

from __future__ import annotations

import asyncio
import base64
import os
import stat
from types import SimpleNamespace

import pytest

from prometheus.computer.livestream import ComputerLiveStream
from prometheus.computer.thumbnails import ThumbnailConfig
from prometheus.computer.types import WindowCapture

SESSION = "telegram:1"
TASK = "a1b2c3"
NAMES = ("gedit", "gnome-text-editor")
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"p" * 64).decode()


# ── fakes at the boundaries ────────────────────────────────────────────────

class _FakeBus:
    """Records every SignalBus emission. The point is what does NOT appear."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    async def emit(self, signal) -> None:
        self.seen.append((signal.kind, dict(signal.payload)))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.seen]

    def payload(self, kind: str) -> dict:
        return next(p for k, p in self.seen if k == kind)


class _FakeSink:
    """A ThumbnailSink that records viewer counts and every frame sent."""

    def __init__(self, viewers: int = 1, raise_on_send: bool = False) -> None:
        self._viewers = viewers
        self.raise_on_send = raise_on_send
        self.sent: list[dict] = []
        self.viewer_calls: list[str] = []

    def viewer_count(self, session_id: str) -> int:
        self.viewer_calls.append(session_id)
        return self._viewers

    async def send_thumbnail(self, session_id: str, frame: dict) -> None:
        if self.raise_on_send:
            raise RuntimeError("socket gone")
        self.sent.append(dict(frame))


def _task(**over):
    t = SimpleNamespace(session_id=SESSION, task_id=TASK, app="gedit")
    for k, v in over.items():
        setattr(t, k, v)
    return t


def _capture(*, roles=("push button",), app_name="gedit", data=PNG,
             mime="image/png", degraded=False, truncated=False,
             frame_valid=None) -> WindowCapture:
    return WindowCapture(
        target="local", app="gedit", pid=1, window_id=2, app_name=app_name,
        roles=tuple(roles), degraded=degraded, truncated=truncated,
        frame_valid=frame_valid, image_mime=mime, image_base64=data,
        image_width=480, image_height=300)


def _stream(tmp_path, *, sink=None, config=None, data_dir=None):
    bus = _FakeBus()
    cfg = config if config is not None else ThumbnailConfig()
    live = ComputerLiveStream(
        bus, bridge=None, keep_per_session=200,
        thumbnail_sink=sink if sink is not None else _FakeSink(),
        thumb_config=cfg,
        thumbnail_data_dir=str(data_dir) if data_dir else None)
    return bus, live


async def _step(live, task, **kw):
    await live.step(task, status="executed", verb="click",
                    description="press Save", **kw)


# ── the happy path ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_clean_capture_is_sent_and_the_step_says_sent(tmp_path):
    bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)

    assert bus.payload("computer_step")["thumbnail"] == "sent"
    assert bus.payload("computer_step")["thumbnail_skip_reason"] is None
    assert len(live._thumb_sink.sent) == 1


@pytest.mark.asyncio
async def test_the_image_frame_carries_the_same_seq_as_its_step(tmp_path):
    """The seq is what lets a client put the picture beside its action. A
    mismatch would be invisible in either stream on its own."""
    bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    step_seq = bus.payload("computer_step")["seq"]
    frame = live._thumb_sink.sent[0]
    assert frame["seq"] == step_seq


@pytest.mark.asyncio
async def test_the_thumbnail_never_reaches_the_signal_bus(tmp_path):
    """THE privacy invariant. SignalBus persists everything it is given, so a
    picture reaching it would be a screenshot in signal_events, backfilled to
    any client and sitting in a database on disk."""
    bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)

    assert "computer_step_thumbnail" not in bus.kinds()
    # And no emission carries image bytes under any kind.
    for kind, payload in bus.seen:
        assert "data_base64" not in payload, kind
        assert PNG not in str(payload.values()), kind


# ── the skips, observed at the boundary ────────────────────────────────────

@pytest.mark.asyncio
async def test_a_password_field_skips_and_sends_nothing(tmp_path):
    bus, live = _stream(tmp_path)
    await _step(live, _task(),
                capture=_capture(roles=("push button", "password text")),
                capture_app_names=NAMES)
    assert bus.payload("computer_step")["thumbnail"] == "skipped"
    assert bus.payload("computer_step")["thumbnail_skip_reason"] == "password_field"
    assert live._thumb_sink.sent == []


@pytest.mark.asyncio
async def test_a_window_that_changed_app_skips(tmp_path):
    """D19 at the seam: another app came to the front, so the picture is not
    the window the person approved."""
    bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(app_name="firefox"),
                capture_app_names=NAMES)
    assert bus.payload("computer_step")["thumbnail_skip_reason"] == "window_changed"
    assert live._thumb_sink.sent == []


@pytest.mark.asyncio
async def test_the_app_matches_by_any_name_the_binding_knows_it_by(tmp_path):
    """The driver reports an executable name; the person picked a display
    name. Both are in the binding's set, so neither may skip."""
    bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(app_name="gnome-text-editor"),
                capture_app_names=NAMES)
    assert bus.payload("computer_step")["thumbnail"] == "sent"


@pytest.mark.asyncio
async def test_no_step_frame_carries_a_thumbnail_when_none_applies(tmp_path):
    """A refused step took no action, so no decision was made: thumbnail is
    null, NOT "skipped" — those are different statements to a person."""
    bus, live = _stream(tmp_path)
    await live.step(_task(), status="refused", reason="not covered")
    payload = bus.payload("computer_step")
    assert payload["thumbnail"] is None
    assert payload["thumbnail_skip_reason"] is None
    assert live._thumb_sink.sent == []


@pytest.mark.asyncio
async def test_the_runner_skip_reason_is_reported_verbatim(tmp_path):
    """The runner decides no_viewer before any capture; livestream must pass
    that reason through rather than inventing its own."""
    bus, live = _stream(tmp_path)
    await live.step(_task(), status="executed", verb="click",
                    description="press Save", thumbnail_skip="no_viewer")
    assert bus.payload("computer_step")["thumbnail"] == "skipped"
    assert bus.payload("computer_step")["thumbnail_skip_reason"] == "no_viewer"
    assert live._thumb_sink.sent == []


@pytest.mark.asyncio
async def test_a_capture_that_failed_at_the_driver_says_capture_failed(tmp_path):
    bus, live = _stream(tmp_path)
    await live.step(_task(), status="executed", verb="click",
                    description="press Save", thumbnail_skip="capture_failed")
    assert bus.payload("computer_step")["thumbnail_skip_reason"] == "capture_failed"


# ── the viewer gate: no viewer means no capture call ───────────────────────

@pytest.mark.asyncio
async def test_thumbnail_viewers_is_zero_when_the_feature_is_off(tmp_path):
    """One check covers both: off means no viewers, which is what stops the
    runner calling the driver at all."""
    _bus, live = _stream(tmp_path, config=ThumbnailConfig(enabled=False))
    assert live.thumbnail_viewers(SESSION) == 0


@pytest.mark.asyncio
async def test_thumbnail_viewers_asks_the_sink(tmp_path):
    sink = _FakeSink(viewers=2)
    _bus, live = _stream(tmp_path, sink=sink)
    assert live.thumbnail_viewers(SESSION) == 2
    assert sink.viewer_calls == [SESSION]


@pytest.mark.asyncio
async def test_a_sink_that_raises_means_no_viewers(tmp_path):
    """Fail closed: an eligibility question that cannot be answered must not
    be answered yes — that would capture a desktop picture nobody asked for."""
    class _Boom(_FakeSink):
        def viewer_count(self, session_id):
            raise RuntimeError("store gone")

    _bus, live = _stream(tmp_path, sink=_Boom())
    assert live.thumbnail_viewers(SESSION) == 0


@pytest.mark.asyncio
async def test_the_runner_gate_never_captures_without_a_viewer():
    """The runner's own gate, tested directly: _thumb_plan decides whether
    driver.capture is called at all. A task nobody watches must not put a
    screenshot through the driver."""
    from prometheus.computer.task import ComputerTaskRunner

    runner = object.__new__(ComputerTaskRunner)
    # No viewers → NONE (no decision made), not SKIP: nothing was refused,
    # there was simply nobody to capture for.
    runner.live = SimpleNamespace(
        thumbnail_plan=lambda s, **kw: ("none", None))
    assert runner._thumb_plan(SESSION, status="executed",
                              after_stop=False) == ("none", None)

    runner.live = SimpleNamespace(
        thumbnail_plan=lambda s, **kw: ("capture", None))
    assert runner._thumb_plan(SESSION, status="executed",
                              after_stop=False) == ("capture", None)

    # No log wired at all → NONE, not an AttributeError.
    runner.live = None
    assert runner._thumb_plan(SESSION, status="executed",
                              after_stop=False) == ("none", None)


def test_the_runner_gate_treats_a_missing_plan_as_none():
    """A live object predating this PR has no thumbnail_plan. That must read
    as "no picture", not crash a desktop task."""
    from prometheus.computer.task import ComputerTaskRunner

    runner = object.__new__(ComputerTaskRunner)
    runner.live = SimpleNamespace()
    assert runner._thumb_plan(SESSION, status="executed",
                              after_stop=False) == ("none", None)


def test_a_plan_that_raises_reads_as_none():
    """Fail closed: a plan that cannot be answered must not become a capture."""
    from prometheus.computer.task import ComputerTaskRunner

    runner = object.__new__(ComputerTaskRunner)
    runner.live = SimpleNamespace(thumbnail_plan=_boom)
    assert runner._thumb_plan(SESSION, status="executed",
                              after_stop=False) == ("none", None)


def _boom(*_a, **_kw):
    raise RuntimeError("sink gone")


# ── a send that fails is a dropped picture, never a task error ─────────────

@pytest.mark.asyncio
async def test_a_failed_send_reports_skipped_not_sent(tmp_path):
    """The viewer left between the count and the send. Reporting "sent" here
    would tell a person a picture arrived that did not."""
    bus, live = _stream(tmp_path, sink=_FakeSink(raise_on_send=True))
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    payload = bus.payload("computer_step")
    assert payload["thumbnail"] == "skipped"
    assert payload["thumbnail_skip_reason"] == "no_viewer"


# ── retention: memory only, and only for as long as the task ───────────────

@pytest.mark.asyncio
async def test_the_latest_thumbnail_is_kept_for_a_reconnecting_device(tmp_path):
    _bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    assert live.last_thumbnail(TASK) is not None
    assert live.last_thumbnail(TASK)["data_base64"] == PNG


@pytest.mark.asyncio
async def test_the_in_memory_thumbnail_is_dropped_when_the_task_ends(tmp_path):
    """Holding it past the end would be a cache of desktop pixels that
    outlives the thing that justified capturing them."""
    _bus, live = _stream(tmp_path)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    task = _task(outcome="done", reason="the goal's one action ran", steps=1,
                 approvals=0, in_flight_at_stop=False,
                 started_at=0.0, ended_at=1.0)
    await live.task_ended(task, summary="done")
    assert live.last_thumbnail(TASK) is None


# ── persist: off by default, and 0600 when on ──────────────────────────────

@pytest.mark.asyncio
async def test_persist_off_writes_nothing_to_disk(tmp_path):
    """The default. A screenshot on disk is the one irreversible thing this
    feature can do, so the default must provably not do it."""
    data = tmp_path / "data"
    _bus, live = _stream(tmp_path, data_dir=data)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    assert not data.exists()


@pytest.mark.asyncio
async def test_persist_on_writes_0600_under_the_thumbnails_tree(tmp_path):
    data = tmp_path / "data"
    _bus, live = _stream(
        tmp_path, config=ThumbnailConfig(persist=True), data_dir=data)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)

    path = data / "computer" / "thumbnails" / TASK / "1.png"
    assert path.exists(), f"expected {path}"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"owner-only, got {oct(mode)}"
    assert path.read_bytes() == base64.b64decode(PNG)


@pytest.mark.asyncio
async def test_persist_needs_both_the_flag_and_a_data_dir(tmp_path):
    """persist=True with no data dir is a no-op, not a write to CWD."""
    _bus, live = _stream(tmp_path, config=ThumbnailConfig(persist=True),
                         data_dir=None)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    assert live._thumb_sink.sent  # the picture still went out live
    assert not (tmp_path / "computer").exists()


@pytest.mark.asyncio
async def test_persisted_thumbnails_are_pruned_when_the_task_ends(tmp_path):
    """THE claim the docstring made and the code did not keep. A screenshot
    that outlives the log entry that justified capturing it is a desktop
    picture on disk with nothing tracking it — so the files must go when the
    task's log goes."""
    data = tmp_path / "data"
    _bus, live = _stream(
        tmp_path, config=ThumbnailConfig(persist=True), data_dir=data)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)

    task_dir = data / "computer" / "thumbnails" / TASK
    assert task_dir.exists() and list(task_dir.iterdir()), "a file was written"

    await live.task_ended(_ended_task(), summary="done")
    assert not task_dir.exists(), "the task's thumbnails outlived its log"


@pytest.mark.asyncio
async def test_prune_removes_only_the_sessions_own_tasks(tmp_path):
    """A prune deletes what THIS process recorded — never a scan of the tree,
    so a foreign or stale directory is not deleted by inference."""
    data = tmp_path / "data"
    _bus, live = _stream(
        tmp_path, config=ThumbnailConfig(persist=True), data_dir=data)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)

    other = data / "computer" / "thumbnails" / "ffff9999"
    other.mkdir(parents=True, exist_ok=True)
    (other / "1.png").write_bytes(b"someone else's")

    await live.task_ended(_ended_task(), summary="done")
    assert not (data / "computer" / "thumbnails" / TASK).exists()
    assert (other / "1.png").exists(), "a task this stream never wrote was deleted"


@pytest.mark.asyncio
async def test_prune_with_persist_off_writes_and_deletes_nothing(tmp_path):
    data = tmp_path / "data"
    _bus, live = _stream(tmp_path, data_dir=data)
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    await live.task_ended(_ended_task(), summary="done")
    assert not data.exists(), "nothing was written, so nothing to prune"


@pytest.mark.asyncio
async def test_prune_survives_a_task_it_never_wrote(tmp_path):
    """A task that ended with no thumbnails (feature off mid-run, or every
    step skipped) must not raise in the prune path."""
    data = tmp_path / "data"
    _bus, live = _stream(
        tmp_path, config=ThumbnailConfig(persist=True), data_dir=data)
    await live.task_ended(_ended_task(task_id="never-wrote"), summary="done")


@pytest.mark.asyncio
async def test_a_recorded_task_id_cannot_prune_outside_the_tree(tmp_path):
    """Belt and braces: the recorded id goes back through persist_path, which
    validates it, so a traversal id cannot point the rmtree at a parent."""
    from prometheus.computer.thumbnails import persist_path
    assert persist_path(str(tmp_path), "../../etc", 1, "image/png") is None


def _ended_task(task_id=TASK, **over):
    t = _task(task_id=task_id, outcome="done",
              reason="the goal's one action ran", steps=1, approvals=0,
              in_flight_at_stop=False, started_at=0.0, ended_at=1.0)
    for k, v in over.items():
        setattr(t, k, v)
    return t


@pytest.mark.asyncio
async def test_a_persist_failure_does_not_stop_the_live_send(tmp_path):
    """Losing the archived copy is not losing the feature. The live picture
    is the point; the file is an opt-in extra."""
    data = tmp_path / "data"
    _bus, live = _stream(
        tmp_path, config=ThumbnailConfig(persist=True), data_dir=data)
    # An unwritable tree: make the path a FILE where a dir must go.
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "computer").write_text("blocked")
    await _step(live, _task(), capture=_capture(), capture_app_names=NAMES)
    assert _bus.payload("computer_step")["thumbnail"] == "sent"
    assert len(live._thumb_sink.sent) == 1
