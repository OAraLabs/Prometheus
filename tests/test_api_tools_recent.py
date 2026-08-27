"""GET /api/tools/recent — the Tool Feed's backfill, read off `tool_calls`.

Why the route exists: the live `tool_call_start`/`tool_call_end` frames are
pure WS fan-out (`WebSocketServer.broadcast` persists nothing and returns
early with no clients attached), and no `tool_call_*` signal is ever emitted
onto the SignalBus, so `signal_events` — and therefore /api/events/recent —
never contains one. A client that was not connected when a tool ran had no
way to learn it happened, while /api/telemetry happily reported the lifetime
total from the very table this route now exposes per-call.

Pinned here: newest-first order, the cap, the tool filter, success/error
round-tripping, and `parsed_tool_call` → `inputs` decoding (including the
nullable and malformed cases, which must not take the route down).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402

from prometheus.telemetry.tracker import (  # noqa: E402
    SYNTHETIC_TOOL_NAME,
    ToolCallTelemetry,
)
from prometheus.web.server import create_app  # noqa: E402


def _seed(tmp_path: Path) -> ToolCallTelemetry:
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    tel.record(
        model="local/default", tool_name="bash", success=True, latency_ms=12.5,
        parsed_tool_call=json.dumps({"name": "bash", "input": {"command": "ls"}}),
    )
    tel.record(
        model="claude-opus-5", tool_name="read_file", success=False,
        latency_ms=3.0, error_type="ValidationError", error_detail="bad path",
        parsed_tool_call=json.dumps({"name": "read_file", "input": {"path": "/x"}}),
    )
    tel.record(model="local/default", tool_name="grep", success=True)
    return tel


def _client(tel: ToolCallTelemetry, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from prometheus.telemetry import tracker as _tracker
    monkeypatch.setattr(_tracker, "_telemetry_singleton", tel)
    app = create_app(config={}, signal_bus=None, session_mgr=None, telemetry=tel)
    return TestClient(app)


class TestApiToolsRecent:
    def test_returns_per_call_rows_newest_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _client(_seed(tmp_path), monkeypatch) as client:
            resp = client.get("/api/tools/recent?limit=10")
            assert resp.status_code == 200
            rows = resp.json()
            assert [r["tool_name"] for r in rows] == ["grep", "read_file", "bash"], (
                "newest-first: the LAST recorded call must lead"
            )
            for row in rows:
                assert set(row.keys()) >= {
                    "call_id", "timestamp", "tool_name", "success",
                    "latency_ms", "error_type", "inputs",
                }

    def test_success_error_and_inputs_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _client(_seed(tmp_path), monkeypatch) as client:
            rows = client.get("/api/tools/recent").json()
            by_name = {r["tool_name"]: r for r in rows}
            assert by_name["bash"]["success"] is True
            assert by_name["bash"]["latency_ms"] == 12.5
            # parsed_tool_call is {"name","input"}; only the input half is inputs.
            assert by_name["bash"]["inputs"] == {"command": "ls"}
            assert by_name["read_file"]["success"] is False
            assert by_name["read_file"]["error_type"] == "ValidationError"
            assert by_name["read_file"]["error_detail"] == "bad path"
            # A call with no parsed_tool_call decodes to None, not a crash.
            assert by_name["grep"]["inputs"] is None

    def test_limit_is_capped_and_filter_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _client(_seed(tmp_path), monkeypatch) as client:
            assert len(client.get("/api/tools/recent?limit=1").json()) == 1
            filtered = client.get("/api/tools/recent?tool=bash").json()
            assert [r["tool_name"] for r in filtered] == ["bash"]

    def test_over_large_limit_clamps_to_500(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 200 alone would not prove the cap — count the rows that come back."""
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        for _ in range(520):
            tel.record(model="m", tool_name="bash", success=True)
        with _client(tel, monkeypatch) as client:
            rows = client.get("/api/tools/recent?limit=99999").json()
            assert len(rows) == 500, f"limit must clamp to 500 (got {len(rows)})"

    def test_malformed_parsed_tool_call_does_not_break_the_route(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        tel.record(
            model="m", tool_name="bash", success=True,
            parsed_tool_call="{not json",
        )
        # A bare JSON string (not the {"name","input"} object) must also survive.
        tel.record(model="m", tool_name="grep", success=True, parsed_tool_call='"nope"')
        with _client(tel, monkeypatch) as client:
            resp = client.get("/api/tools/recent")
            assert resp.status_code == 200
            assert all(r["inputs"] is None for r in resp.json())

    def test_synthetic_loop_rows_are_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_loop_transition` is the agent loop's per-turn echo, not a tool call.

        It is ~48% of `tool_calls` on a live daemon, and every other per-tool reader
        in tracker.py drops it. The route shipped without the exclusion and served a
        feed that was half loop echoes — this is the regression pin.
        """
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        for _ in range(5):
            tel.record(model="m", tool_name=SYNTHETIC_TOOL_NAME, success=True)
        tel.record(model="m", tool_name="bash", success=True)
        with _client(tel, monkeypatch) as client:
            rows = client.get("/api/tools/recent?limit=100").json()
            assert [r["tool_name"] for r in rows] == ["bash"], (
                f"only real tool calls survive (got {[r['tool_name'] for r in rows]})"
            )

    def test_exclusion_beats_an_explicit_tool_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asking for the synthetic tool by name still returns nothing — excluded means excluded."""
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        tel.record(model="m", tool_name=SYNTHETIC_TOOL_NAME, success=True)
        tel.record(model="m", tool_name="bash", success=True)
        with _client(tel, monkeypatch) as client:
            assert client.get(f"/api/tools/recent?tool={SYNTHETIC_TOOL_NAME}").json() == []
            assert len(client.get("/api/tools/recent?tool=bash").json()) == 1

    def test_empty_table_returns_empty_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        with _client(tel, monkeypatch) as client:
            assert client.get("/api/tools/recent").json() == []
