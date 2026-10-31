"""Golden trace v2: the request a golden call answered, stored as sent (telemetry/golden_request.py).

v1 kept the call and rebuilt its input half from LCM at export: no system
prompt, no earlier tool calls or results, one tool schema, at most 12
messages. Pinned here: a golden call stores its round's request (system
prompt, every message with tool calls and results, the full tool list, the
route) content-addressed and redacted; a non-golden call stores nothing; a
capture failure keeps the call row; the export writes v2 lines that a v1
reader (the distillation harvester's access pattern) still reads, and v1 rows
still export as v1; both carry the trace labels.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.providers.base import ApiMessageRequest
from prometheus.telemetry import golden_request as gr
from prometheus.telemetry.tracker import ToolCallTelemetry

# Token shapes the capture redactor knows (security/log_redaction.py).
GH = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
SYSTEM = "You are Prometheus. Tools follow."
TOOLS = [
    {"name": "bash", "description": "run a command",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {"name": "read_file", "description": "read a file",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
]


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)


def _history(n_rounds: int = 1, secret: str = "") -> list[ConversationMessage]:
    msgs = [ConversationMessage.from_user_text("list the repo, then read the README " + secret)]
    for i in range(n_rounds):
        msgs.append(ConversationMessage(role="assistant", content=[
            TextBlock(text=f"step {i}"),
            ToolUseBlock(id=f"tu{i}", name="bash", input={"command": f"ls {i} {secret}"})]))
        msgs.append(ConversationMessage(role="user", content=[
            ToolResultBlock(tool_use_id=f"tu{i}", content=f"README.md\nsrc {secret}")]))
    return msgs


def _capture(messages, *, system=SYSTEM, provider="qwen", route=None) -> gr.GoldenRequestCapture:
    return gr.GoldenRequestCapture(
        request=ApiMessageRequest(model="qwen3.8-max", messages=messages, system_prompt=system,
                                  tools=TOOLS, max_tokens=900),
        provider=provider,
        route=route or {"key": "qwen", "provider": "qwen", "degraded": False},
    )


def _golden(tel, capture, *, tool="read_file", provider="qwen", tool_use_id="tuX", **kw):
    tel.record(model="qwen3.8-max", tool_name=tool, success=True, retries=0,
               raw_model_output="reading it", provider=provider,
               parsed_tool_call=json.dumps({"name": tool, "input": {"path": "README.md"}}),
               session_id="beacon:s1", tool_schema=json.dumps({"name": tool}),
               tool_use_id=tool_use_id, golden_request=capture, **kw)


def _export(tel, tmp_path: Path, resolver=None) -> list[dict]:
    res = tel.export_new_golden_traces(output_dir=tmp_path / "traj", context_resolver=resolver)
    assert res is not None and res.path is not None
    return [json.loads(line) for line in res.path.read_text().splitlines()]


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #


def test_a_golden_call_stores_its_request_as_sent(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history(2)))
    request_id = tel._conn.execute("SELECT request_id FROM tool_calls").fetchone()[0]
    req = gr.load_request(tel._conn, request_id)
    tel.close()
    assert req is not None and req["trace_format"] == 2 and req["wire"] == "openai_chat"
    assert req["system"] == SYSTEM
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"], roles
    assert req["messages"][1]["tool_calls"][0]["function"]["name"] == "bash"
    assert req["messages"][2]["tool_call_id"] == "tu0"
    assert [t["function"]["name"] for t in req["tools"]] == ["bash", "read_file"]
    assert req["tool_set_hash"] and req["requested_model"] == "qwen3.8-max"
    assert req["provider"] == "qwen" and req["route"]["key"] == "qwen"
    assert req["max_tokens"] == 900 and req["build_version"]


def test_there_is_no_message_cap(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history(40)))
    rid = tel._conn.execute("SELECT request_id FROM tool_calls").fetchone()[0]
    assert len(gr.load_request(tel._conn, rid)["messages"]) == 81
    tel.close()


def test_a_shared_prefix_is_stored_once(tmp_path):
    """Round n re-sends round n-1's messages: each distinct message is one blob."""
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history(3)), tool_use_id="a")
    _golden(tel, _capture(_history(3)), tool_use_id="b")       # same round, second call
    _golden(tel, _capture(_history(4)), tool_use_id="c")       # next round
    c = tel._conn
    ids = [r[0] for r in c.execute("SELECT request_id FROM tool_calls ORDER BY rowid")]
    assert ids[0] == ids[1] != ids[2]
    assert c.execute("SELECT COUNT(*) FROM golden_requests").fetchone()[0] == 2
    # 1 user + 4×(assistant, tool) messages, plus the system prompt and the tool list.
    assert c.execute("SELECT COUNT(*) FROM golden_blobs WHERE kind = 'message'").fetchone()[0] == 9
    assert c.execute("SELECT COUNT(*) FROM golden_blobs WHERE kind != 'message'").fetchone()[0] == 2
    tel.close()


@pytest.mark.parametrize("kw", [
    {"provider": "llama_cpp"},                 # local: never golden
    {"trainable": False},                      # a group member's turn
])
def test_a_call_that_is_not_golden_stores_no_request(tmp_path, kw):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    provider = kw.pop("provider", "qwen")
    _golden(tel, _capture(_history()), provider=provider, **kw)
    c = tel._conn
    assert c.execute("SELECT is_golden, request_id FROM tool_calls").fetchone() == (0, None)
    assert c.execute("SELECT COUNT(*) FROM golden_requests").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM golden_blobs").fetchone()[0] == 0
    tel.close()


def test_a_failed_capture_keeps_the_call_as_v1(tmp_path, monkeypatch):
    tel = ToolCallTelemetry(tmp_path / "t.db")

    def boom(_capture):
        raise RuntimeError("serializer bug")

    monkeypatch.setattr(gr, "serialize", boom)
    _golden(tel, _capture(_history()))
    c = tel._conn
    assert c.execute("SELECT is_golden, request_id, tool_ok FROM tool_calls").fetchone() \
        == (1, None, 1)
    assert c.execute("SELECT subsystem, operation FROM silent_failures").fetchall() == [
        ("telemetry", "golden_request")]
    tel.close()


def test_a_failed_row_write_does_not_strand_the_request_parts(tmp_path, monkeypatch):
    """The row's INSERT failing rolls the request's parts back too; a later call
    of the same round must write them again, not trust a cache of rolled-back rows."""
    tel = ToolCallTelemetry(tmp_path / "t.db")
    cap = _capture(_history(2))
    real = tel._conn

    class _FailRowInsert:
        def __getattr__(self, name):
            return getattr(real, name)

        def execute(self, sql, *a):
            if sql.lstrip().startswith("INSERT INTO tool_calls"):
                raise sqlite3.OperationalError("database is locked")
            return real.execute(sql, *a)

    tel._conn = _FailRowInsert()  # type: ignore[assignment]
    _golden(tel, cap)
    tel._conn = real
    assert real.execute("SELECT COUNT(*) FROM golden_blobs").fetchone()[0] == 0
    _golden(tel, cap)
    rid = real.execute("SELECT request_id FROM tool_calls").fetchone()[0]
    assert gr.load_request(real, rid) is not None
    tel.close()


def test_secrets_never_land_in_the_request_store_or_the_export(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    msgs = _history(2, secret=GH)
    # A token after an escaped newline inside a tool call's JSON arguments.
    msgs.append(ConversationMessage(role="assistant", content=[
        ToolUseBlock(id="tuS", name="bash", input={"command": f"echo hi\n{GH}"})]))
    msgs.append(ConversationMessage(role="user", content=[
        ToolResultBlock(tool_use_id="tuS", content="ok")]))
    _golden(tel, _capture(msgs, system=f"{SYSTEM} key={GH}"))
    dump = "\n".join(str(r) for t in ("golden_blobs", "golden_requests", "tool_calls")
                     for r in tel._conn.execute(f"SELECT * FROM {t}"))
    assert GH not in dump and GH[:12] not in dump
    lines = _export(tel, tmp_path)
    tel.close()
    assert GH not in json.dumps(lines)


def test_images_are_stored_as_a_digest_not_their_bytes(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    data = "iVBORw0KGgo" + "A" * 400
    msgs = [ConversationMessage(role="user", content=[
        TextBlock(text="what is this"), ImageBlock(media_type="image/png", data=data)])]
    _golden(tel, _capture(msgs))
    body = "\n".join(r[0] for r in tel._conn.execute("SELECT body FROM golden_blobs"))
    tel.close()
    assert data not in body and "omitted:sha256:" in body


def test_the_scrub_covers_the_request_store(tmp_path):
    from prometheus.security.scrub_capture_stores import TELEMETRY_TARGETS

    assert ("golden_blobs", "hash", ("body",)) in TELEMETRY_TARGETS


# --------------------------------------------------------------------------- #
# export: v2 lines, v1 lines, labels
# --------------------------------------------------------------------------- #


def test_a_v2_export_line_is_the_full_request_plus_the_call(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history(2)), tool_use_id="tu9")
    [line] = _export(tel, tmp_path)          # no resolver needed: the request is stored
    tel.close()
    msgs = line["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert [m["role"] for m in msgs[1:-1]] == ["user", "assistant", "tool", "assistant", "tool"]
    target = msgs[-1]
    assert target["role"] == "assistant" and target["content"] == "reading it"
    assert target["tool_calls"] == [{"id": "tu9", "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": "README.md"})}}]
    assert [t["function"]["name"] for t in line["tools"]] == ["bash", "read_file"]
    meta = line["_meta"]
    assert meta["trace_format"] == 2
    assert meta["provider"] == "qwen" and meta["route"]["key"] == "qwen"
    assert meta["tool_name"] == "read_file" and meta["session_id"] == "beacon:s1"
    assert meta["system_prompt_hash"] and meta["tools_hash"] and meta["request_id"]
    assert meta["labels"] == {"tool_ok": 1, "turn_outcome": None, "judged_ok": None,
                              "judged_by": None, "turn_signal": None}


def test_a_row_without_a_request_still_exports_as_v1_with_labels(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, None)
    tel._conn.execute("UPDATE tool_calls SET turn_outcome = 'stopped', judged_ok = 0,"
                      " judged_by = 'rubric-v2'")
    tel._conn.commit()
    [line] = _export(tel, tmp_path, resolver=lambda _t: [{"role": "user", "content": "read it"}])
    tel.close()
    assert line["messages"][0] == {"role": "user", "content": "read it"}
    assert line["_meta"]["trace_format"] == 1
    assert line["tools"] == [{"type": "function", "function": {"name": "read_file"}}]
    assert line["_meta"]["labels"]["turn_outcome"] == "stopped"
    assert line["_meta"]["labels"]["judged_ok"] == 0


def test_the_export_carries_the_turns_user_signal_apart_from_its_end_state(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    tel._conn.execute("INSERT INTO turns (turn_id, session_id, started_at, ended_at, outcome)"
                      " VALUES ('beacon:s1:t', 'beacon:s1', 1, 2, 'abandoned')")
    _golden(tel, _capture(_history()), turn_id="beacon:s1:t")
    tel._conn.execute("UPDATE tool_calls SET turn_outcome = 'completed'")
    tel._conn.commit()
    [line] = _export(tel, tmp_path)
    tel.close()
    assert line["_meta"]["labels"]["turn_outcome"] == "completed"
    assert line["_meta"]["labels"]["turn_signal"] == "abandoned"


def test_a_cli_dump_never_takes_the_incremental_exports_name(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history()))
    path = tel.export_golden_traces(output_dir=tmp_path / "traj")
    tel.close()
    assert path.name.startswith("golden_dump_")
    assert not list((tmp_path / "traj").glob("golden_traces_*.jsonl"))


# --------------------------------------------------------------------------- #
# readers: v1 and v2 in one shape; the harvester's access pattern
# --------------------------------------------------------------------------- #

V1_LINE = {
    "messages": [
        {"role": "user", "content": "list the repo"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_7", "type": "function",
             "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
    ],
    "tools": [{"type": "function", "function": {"name": "bash"}}],
    "_meta": {"model": "qwen3.8-max", "tool_name": "bash", "timestamp": 1.0,
              "session_id": "beacon:s1"},
}


def _v2_line(tmp_path) -> dict:
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history(2)))
    [line] = _export(tel, tmp_path)
    tel.close()
    return line


def test_read_example_reads_v1_and_v2_alike(tmp_path):
    v1 = gr.read_example(V1_LINE)
    assert v1["trace_format"] == 1 and v1["system"] is None and v1["labels"] == {}
    assert v1["context"] == [V1_LINE["messages"][0]]
    v2 = gr.read_example(_v2_line(tmp_path))
    assert v2["trace_format"] == 2 and v2["system"] == SYSTEM
    assert v2["context"][0]["role"] == "user" and v2["target"]["tool_calls"]
    assert v2["labels"]["tool_ok"] == 1
    with pytest.raises(ValueError):
        gr.read_example({"messages": [{"role": "user", "content": "x"}]})


@pytest.mark.parametrize("which", ["v1", "v2"])
def test_a_v1_reader_reads_both_formats(tmp_path, which):
    """The distillation harvester (scripts/distill/build.py, feat/distill-flywheel)
    reads exactly these: messages[-1].tool_calls[0].function, the context's
    role/content, _meta.tool_name/session_id/model, a non-empty tools list.
    Replayed here so a v2 line can never stop being readable by it."""
    line = V1_LINE if which == "v1" else _v2_line(tmp_path)
    meta = line.get("_meta") or {}
    msgs = line["messages"]
    assert line.get("tools") and msgs[-1].get("tool_calls")
    call = (msgs[-1].get("tool_calls") or [{}])[0].get("function", {})
    args = json.loads(call.get("arguments") or "{}")
    assert isinstance(args, dict) and call["name"] == meta["tool_name"]
    ctx = msgs[:-1]
    json.dumps([[m["role"], m.get("content")] for m in ctx])
    users = [m.get("content") for m in ctx if m.get("role") == "user"]
    assert users and isinstance(users[-1], str)
    assert meta.get("session_id") and meta.get("model")
    for m in ctx:
        for tc in m.get("tool_calls") or []:
            assert (tc.get("function") or {}).get("name")


# --------------------------------------------------------------------------- #
# end to end through the loop
# --------------------------------------------------------------------------- #


def test_a_cloud_turn_through_the_loop_stores_the_rounds_request(tmp_path):
    from tests.test_fallback_attribution import _run, _Scripted

    con = _run(tmp_path, primary=_Scripted("bash", provider_name="qwen"))
    [(rid, golden)] = con.execute("SELECT request_id, is_golden FROM tool_calls"
                                  " WHERE tool_name = 'bash'").fetchall()
    assert golden == 1 and rid
    req = gr.load_request(con, rid)
    assert req["system"] and "qwen3.8-max" in req["system"]
    assert [m["role"] for m in req["messages"]] == ["user"]
    assert req["messages"][0]["content"] == "go"
    assert [t["function"]["name"] for t in req["tools"]] == ["bash"]
    assert req["provider"] == "qwen" and req["route"]["degraded"] is False


def test_a_degraded_round_stores_no_request(tmp_path):
    """The fallback served it: not golden, so nothing is stored (T14 holds)."""
    from tests.test_fallback_attribution import _ExpiredCloud, _run, _Scripted

    con = _run(tmp_path, primary=_ExpiredCloud(), fallback_provider=_Scripted("bash"))
    assert con.execute("SELECT COUNT(*) FROM golden_requests").fetchone()[0] == 0


def test_the_round_request_does_not_leak_past_the_run(tmp_path):
    from prometheus.engine import agent_loop
    from tests.test_fallback_attribution import _run, _Scripted

    _run(tmp_path, primary=_Scripted("bash", provider_name="qwen"))
    assert agent_loop._ROUND_REQUEST.get() is None


def test_sqlite_reads_of_a_missing_part_return_none(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    _golden(tel, _capture(_history()))
    rid = tel._conn.execute("SELECT request_id FROM tool_calls").fetchone()[0]
    tel._conn.execute("DELETE FROM golden_blobs WHERE kind = 'tools'")
    tel._conn.commit()
    assert gr.load_request(tel._conn, rid) is None
    assert gr.load_request(tel._conn, "nope") is None
    assert gr.load_request(sqlite3.connect(":memory:"), rid) is None
    tel.close()
