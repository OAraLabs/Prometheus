"""silent_failures must record the SERVER's words, not just the client's.

Piece 4 of the 2026-08-17 arc. The table did its job that day — it captured the
400 twice, from two subsystems. What it could not do was diagnose it:
``exception_msg`` held httpx's "Client error '400 Bad Request' for url ...",
which names the status and nothing about the cause. The server's body said
``Failed to tokenize prompt`` — a media-marker rejection, NOT a context
overflow — and that word reached journald only.

From the DB alone the two failures were indistinguishable, and overflow is the
wrong guess to leave lying around: it sends the next reader to the context
budget instead of to the tool result that poisoned the prompt.

Tested by OUTCOME: given only a row, can a reader tell the two apart?
"""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry, _response_body

TOKENIZE = json.dumps(
    {"error": {"code": 400, "message": "Failed to tokenize prompt",
               "type": "invalid_request_error"}}
)
OVERFLOW = json.dumps(
    {"error": {"code": 400,
               "message": "request (34413 tokens) exceeds the available "
                          "context size (32768 tokens), try increasing it",
               "type": "invalid_request_error"}}
)


def _http_400(body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend:8080/v1/chat/completions")
    response = httpx.Response(400, request=request, content=body.encode())
    return httpx.HTTPStatusError(
        "Client error '400 Bad Request' for url "
        "'http://backend:8080/v1/chat/completions'",
        request=request, response=response,
    )


def _rows(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM silent_failures ORDER BY timestamp")]
    finally:
        con.close()


def test_the_two_400s_are_distinguishable_from_the_db_alone(tmp_path):
    """The outcome that was missing. Both are 400s with identical client text."""
    db = tmp_path / "t.db"
    tel = ToolCallTelemetry(db_path=str(db))
    tel.record_silent_failure("agent_loop", "loop_round", _http_400(TOKENIZE))
    tel.record_silent_failure("agent_loop", "loop_round", _http_400(OVERFLOW))

    rows = _rows(db)
    assert len(rows) == 2
    # The client's summary is identical for both — this is the defect.
    assert rows[0]["exception_msg"] == rows[1]["exception_msg"]
    # The server's body tells them apart.
    assert "Failed to tokenize prompt" in rows[0]["response_body"]
    assert "exceeds the available context size" in rows[1]["response_body"]
    assert rows[0]["response_body"] != rows[1]["response_body"]


def test_non_http_exception_records_null_and_does_not_crash(tmp_path):
    db = tmp_path / "t.db"
    tel = ToolCallTelemetry(db_path=str(db))
    tel.record_silent_failure("curator", "run_once", ValueError("plain"))
    row = _rows(db)[0]
    assert row["response_body"] is None
    assert row["exception_type"] == "ValueError"


def test_unread_streaming_response_does_not_raise(tmp_path):
    """``.text`` on an unread httpx stream raises ResponseNotRead.

    A telemetry helper must not turn a recorded failure into an unrecorded one,
    so the extractor swallows that and the ROW STILL LANDS.
    """
    request = httpx.Request("POST", "http://backend:8080/v1/chat/completions")
    streamed = httpx.Response(
        400, request=request,
        stream=httpx.SyncByteStream(),  # never read
    )
    exc = httpx.HTTPStatusError("boom", request=request, response=streamed)
    with pytest.raises(httpx.ResponseNotRead):
        _ = streamed.text                      # the hazard is real
    assert _response_body(exc) is None         # ...and handled

    db = tmp_path / "t.db"
    tel = ToolCallTelemetry(db_path=str(db))
    tel.record_silent_failure("web_bridge", "_run_agent", exc)
    assert len(_rows(db)) == 1                 # the row still landed


def test_bytes_body_is_decoded(tmp_path):
    request = httpx.Request("GET", "http://x/")
    resp = httpx.Response(500, request=request, content=b"\xff raw bytes")
    exc = httpx.HTTPStatusError("boom", request=request, response=resp)
    body = _response_body(exc)
    assert isinstance(body, str) and "raw bytes" in body


def test_body_is_capped(tmp_path):
    request = httpx.Request("GET", "http://x/")
    resp = httpx.Response(500, request=request, content=b"x" * 20_000)
    exc = httpx.HTTPStatusError("boom", request=request, response=resp)
    assert len(_response_body(exc)) == 4000


def test_migration_adds_the_column_to_an_existing_db(tmp_path):
    """Existing telemetry.db files must gain the column, not error."""
    db = tmp_path / "old.db"
    ToolCallTelemetry(db_path=str(db))                 # create
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE silent_failures DROP COLUMN response_body")
    con.commit(); con.close()
    cols = lambda: {r[1] for r in sqlite3.connect(db).execute(
        "PRAGMA table_info(silent_failures)")}
    assert "response_body" not in cols()
    ToolCallTelemetry(db_path=str(db))                 # re-open migrates
    assert "response_body" in cols()
