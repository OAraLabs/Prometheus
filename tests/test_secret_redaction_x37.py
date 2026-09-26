"""X.37 — token shapes are redacted before LCM and telemetry rows exist, and before
the learning loop sends anything to a model.

The skill-usage audit (docs/audits/SKILL-USAGE.md) found GitHub tokens pasted into
chats sitting verbatim in ``lcm.db`` — in messages and in the summaries built from
them — and therefore in every nightly snapshot and backup. Capture-time redaction
already covered ``tool_calls``, ``silent_failures``, training pairs and golden
exports (test_capture_redaction.py). It did not cover LCM, the other telemetry
JSON columns, or the prompts the learning loop builds from conversation text.

The live conversation is NOT redacted: a user who pastes a token for the agent to
use must still reach the model with it. Only what is kept, and what the learning
loop sends, is.

Every fake token below is assembled at runtime, so no literal key-shaped string
sits in this file (.githooks/pre-commit scans whole staged files).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.security import REDACTED, redact_secrets


def _run_of(prefix: str, n: int, alphabet: str = "aB3xY9") -> str:
    return prefix + (alphabet * n)[:n]


GH_CLASSIC = _run_of("gh" + "p_", 36)
GH_OAUTH = _run_of("gh" + "o_", 36)
GH_FINE = _run_of("github" + "_pat_", 82, "aB3x_Y9")
AWS_KEY_ID = "AK" + "IA" + "Q7ZX" * 4
AWS_SECRET_LINE = "aws_secret" + "_access_key = " + _run_of("", 40, "aB3/x+Y9")
GITLAB = _run_of("gl" + "pat-", 20, "aB3x-Y9")
HUGGINGFACE = _run_of("h" + "f_", 34)
STRIPE_LIVE = _run_of("sk" + "_live_", 24)
STRIPE_RESTRICTED = _run_of("rk" + "_live_", 24)
NPM = _run_of("np" + "m_", 36)
PYPI = _run_of("py" + "pi-AgEIcHlwaS5vcmc", 60, "aB3x-Y9_")
GOOGLE_OAUTH = _run_of("ya" + "29.", 60, "aB3x-Y9_")
SENDGRID = "S" + "G." + _run_of("", 22, "aB3x-Y9_") + "." + _run_of("", 43, "aB3x-Y9_")
JWT = ("ey" + "J" + _run_of("", 30) + "." + "ey" + "J" + _run_of("", 40) + "." + _run_of("", 43, "aB3x-Y9_"))
GROQ = _run_of("gs" + "k_", 52)
REPLICATE = _run_of("r" + "8_", 37)
TAVILY = _run_of("tv" + "ly-", 32)
PERPLEXITY = _run_of("pp" + "lx-", 48)
LINEAR = _run_of("lin" + "_api_", 40)
DIGITALOCEAN = _run_of("dop" + "_v1_", 64, "0a1b2c3d4e5f")
TAILSCALE = _run_of("ts" + "key-auth-", 40, "aB3x-Y9")
PEM = ("-----BEGIN RSA PRIV" + "ATE KEY-----\n" + _run_of("", 64) + "\n" + _run_of("", 64)
       + "\n-----END RSA PRIV" + "ATE KEY-----")

SHAPES = {
    "github classic": GH_CLASSIC, "github oauth": GH_OAUTH, "github fine-grained": GH_FINE,
    "aws key id": AWS_KEY_ID, "gitlab": GITLAB, "hugging face": HUGGINGFACE,
    "stripe secret": STRIPE_LIVE, "stripe restricted": STRIPE_RESTRICTED, "npm": NPM,
    "pypi": PYPI, "google oauth": GOOGLE_OAUTH, "sendgrid": SENDGRID, "jwt": JWT,
    "groq": GROQ, "replicate": REPLICATE, "tavily": TAVILY, "perplexity": PERPLEXITY,
    "linear": LINEAR, "digitalocean": DIGITALOCEAN, "tailscale": TAILSCALE,
}


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_every_shape_is_redacted(kind):
    secret = SHAPES[kind]
    out = redact_secrets(f"here: {secret} (end)")
    assert secret not in out, kind
    assert REDACTED in out and out.endswith("(end)")


def test_a_private_key_block_is_redacted_whole():
    out = redact_secrets(f"key follows\n{PEM}\ndone")
    assert "PRIV" + "ATE KEY" not in out
    assert out.startswith("key follows") and out.endswith("done")


def test_an_aws_secret_is_redacted_by_its_label():
    out = redact_secrets(AWS_SECRET_LINE)
    assert AWS_SECRET_LINE.split("= ")[1] not in out and REDACTED in out


@pytest.mark.parametrize("benign", [
    "commit 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a landed",
    "uuid 123e4567-e89b-12d3-a456-426614174000",
    "sha256 828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "call hf_hub_download(repo_id, filename)",
    "pip install scikit-learn  # not sk-learn",
    "the gh" + "p_ prefix alone is not a token",
    "a JWT starts with eyJ and has three parts",
    "npm_config_cache and my_api_key_name are variable names",
    "rk_live and sk_test are words here",
    "widgets: 7\ngears: 3\nsprockets: 11",
    # shapes the older, log-line patterns matched in kept conversation text (X.37)
    "psql postgres://app:devpass@localhost:5432/app",
    "redis://:devpass@127.0.0.1:6379/0",
    "git push https://x-access-token:${GH_TOKEN}@github.com/o/r",
    "switch to xai-grok-4-fast-reasoning for this",
    "cache key 1727712345:" + "3f2a9c" * 5 + "3f",
])
def test_benign_text_is_untouched(benign):
    assert redact_secrets(benign) == benign


def test_a_remote_database_password_is_redacted():
    out = redact_secrets("postgresql://postgres:Remote0Pass9@db.example.net:5432/postgres")
    assert "Remote0Pass9" not in out and out == f"postgresql://postgres:{REDACTED}@db.example.net:5432/postgres"
    # loopback only by exact host: a name that merely starts with localhost is remote
    assert "Remote0Pass9" not in redact_secrets("postgres://u:Remote0Pass9@localhost.example.net/x")


def test_a_flat_xai_key_is_redacted():
    key = _run_of("xa" + "i-", 60)
    assert redact_secrets(f"key {key}") == f"key {REDACTED}"


# ---------------------------------------------------------------------------
# JSON held as text
# ---------------------------------------------------------------------------

def test_a_token_after_an_escaped_newline_is_found_in_json_text():
    """In raw JSON the character before the token is the n of \\n, so a \\b-anchored
    pattern over the raw text misses it. redact_json_text reads the decoded string."""
    from prometheus.security import redact_json_text

    raw = json.dumps({"text": "line one\n" + GH_CLASSIC})
    assert redact_secrets(raw) == raw  # the miss this helper exists for
    assert json.loads(redact_json_text(raw))["text"] == "line one\n" + REDACTED


def test_json_text_keeps_its_encoding_and_falls_back_to_text():
    from prometheus.security import redact_json_text

    raw = json.dumps({"t": "café " + NPM}, ensure_ascii=False)
    out = redact_json_text(raw)
    assert "café" in out and NPM not in out and json.loads(out)["t"] == "café " + REDACTED
    assert redact_json_text("not json " + NPM) == "not json " + REDACTED
    clean = '{"a": [1, 2.50, "x"],   "b": null}'
    assert redact_json_text(clean) is clean


# ---------------------------------------------------------------------------
# LCM: messages and summaries are redacted before the row exists
# ---------------------------------------------------------------------------

def _store(tmp_path):
    from prometheus.memory.lcm_conversation_store import LCMConversationStore

    return LCMConversationStore(tmp_path / "lcm.db")


def _msg(text: str, **kw):
    from prometheus.memory.lcm_types import MessagePart

    return MessagePart(role="user", content=text, session_id="telegram:1",
                       content_json=json.dumps([{"type": "text", "text": text}]), **kw)


def test_an_lcm_message_is_stored_redacted_and_the_callers_copy_is_not(tmp_path):
    store = _store(tmp_path)
    msg = _msg(f"here is my token {GH_CLASSIC} please use it")
    store.insert_message(msg)
    content, content_json = store._conn.execute(
        "SELECT content, content_json FROM lcm_messages").fetchone()
    assert GH_CLASSIC not in content and REDACTED in content
    assert json.loads(content_json)[0]["text"] == f"here is my token {REDACTED} please use it"
    # the full-text index holds only the redacted text
    hits = store._conn.execute(
        "SELECT count(*) FROM lcm_messages_fts WHERE lcm_messages_fts MATCH ?",
        ('"' + GH_CLASSIC + '"',)).fetchone()[0]
    assert hits == 0
    # the live conversation keeps what the user sent
    assert GH_CLASSIC in msg.content


def test_a_clean_lcm_message_is_stored_byte_identical(tmp_path):
    store = _store(tmp_path)
    raw_json = '[{"type": "text", "text": "widgets: 7"}, {"type":"tool_use","id":"t1","name":"bash","input":{}}]'
    msg = _msg("widgets: 7")
    msg.content_json = raw_json
    store.insert_message(msg)
    assert store._conn.execute("SELECT content_json FROM lcm_messages").fetchone()[0] == raw_json


def test_content_json_stays_valid_when_a_token_sits_next_to_escapes(tmp_path):
    store = _store(tmp_path)
    text = f'curl "https://x.test/?token={GH_OAUTH}"\nthen "quote" and \\ backslash'
    store.insert_message(_msg(text))
    content_json = store._conn.execute("SELECT content_json FROM lcm_messages").fetchone()[0]
    [block] = json.loads(content_json)
    assert GH_OAUTH not in block["text"] and '"quote"' in block["text"] and "\\ backslash" in block["text"]


def test_an_lcm_summary_is_stored_redacted(tmp_path):
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.memory.lcm_types import SummaryNode

    summaries = LCMSummaryStore(tmp_path / "lcm.db")
    summaries.insert_summary(SummaryNode(summary_text=f"The user shared {GH_FINE} for the repo."))
    text = summaries._conn.execute("SELECT summary_text FROM lcm_summaries").fetchone()[0]
    assert GH_FINE not in text and REDACTED in text


def test_a_session_title_is_stored_redacted(tmp_path):
    store = _store(tmp_path)
    store.set_session_title("telegram:1", f"Push with {GH_CLASSIC}")
    title = store._conn.execute("SELECT title FROM session_titles").fetchone()[0]
    assert GH_CLASSIC not in title and title == f"Push with {REDACTED}"


def test_divergence_checkpoints_are_stored_redacted(tmp_path):
    """The divergence detector writes the goal and a conversation snapshot into lcm.db."""
    from prometheus.coordinator.divergence import Checkpoint, CheckpointStore

    CheckpointStore(tmp_path / "lcm.db").save(Checkpoint(
        task_id="t", step_number=5, goal_description=f"deploy using {GITLAB}", goal_hash="h",
        messages_snapshot=[{"role": "user", "content": f"token:\n{GH_OAUTH}"}],
        tool_calls=[{"name": "bash", "input": {"command": f"export HF_TOKEN={HUGGINGFACE}"}}]))
    goal, messages, calls = sqlite3.connect(tmp_path / "lcm.db").execute(
        "SELECT goal_description, messages_json, tool_calls_json FROM checkpoints").fetchone()
    assert GITLAB not in goal and GH_OAUTH not in messages and HUGGINGFACE not in calls
    assert json.loads(messages)[0]["content"] == f"token:\n{REDACTED}"
    assert json.loads(calls)[0]["input"]["command"] == f"export HF_TOKEN={REDACTED}"


# ---------------------------------------------------------------------------
# Telemetry: run summaries and signal payloads
# ---------------------------------------------------------------------------

def test_a_parsed_tool_call_is_redacted_value_by_value(tmp_path):
    from prometheus.telemetry.tracker import ToolCallTelemetry

    tel = ToolCallTelemetry(tmp_path / "t.db")
    call = json.dumps({"name": "bash", "input": {"command": "cat <<EOF > .env\n" + GH_CLASSIC + "\nEOF"}})
    tel.record(model="m", tool_name="bash", success=True, parsed_tool_call=call)
    [stored] = tel._conn.execute("SELECT parsed_tool_call FROM tool_calls").fetchone()
    assert GH_CLASSIC not in stored
    assert json.loads(stored)["input"]["command"] == f"cat <<EOF > .env\n{REDACTED}\nEOF"


def test_run_summaries_and_signal_payloads_are_redacted(tmp_path):
    from prometheus.telemetry.tracker import ToolCallTelemetry

    tel = ToolCallTelemetry(tmp_path / "t.db")
    tel.record_run("skill_creator", "generate_skill", "success",
                   summary={"note": f"used {HUGGINGFACE}", "n": 3})
    tel.record_signal_event("skill_created", {"trigger_task": f"deploy with {GH_CLASSIC}"},
                            "skill_creator")
    [summary_json] = tel._conn.execute("SELECT summary_json FROM subsystem_runs").fetchone()
    [payload] = tel._conn.execute("SELECT payload FROM signal_events").fetchone()
    assert HUGGINGFACE not in summary_json and json.loads(summary_json)["n"] == 3
    assert GH_CLASSIC not in payload and REDACTED in json.loads(payload)["trigger_task"]


def test_a_circuit_breaker_sample_is_redacted_before_it_is_cut(tmp_path):
    """Truncating first could leave a token too short to match, stored in part."""
    from prometheus.telemetry.tracker import ToolCallTelemetry

    tel = ToolCallTelemetry(tmp_path / "t.db")
    tel.record_diagnosis("m", "full", "bash", "malformed", False, "x" * 479 + " " + GH_CLASSIC,
                         False, "none", golden_reference=json.dumps({"input": {"t": f"a\n{NPM}"}}))
    sample, golden = tel._conn.execute(
        "SELECT raw_sample, golden_reference FROM circuit_breaker_diagnostics").fetchone()
    assert GH_CLASSIC[:12] not in sample and sample.endswith(REDACTED)
    assert NPM not in golden and json.loads(golden)["input"]["t"] == f"a\n{REDACTED}"


# ---------------------------------------------------------------------------
# The learning loop: what it sends to a model
# ---------------------------------------------------------------------------

class _Capture:
    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        from prometheus.providers.base import ApiTextDeltaEvent

        self.requests.append(request)
        yield ApiTextDeltaEvent(text="ok")

    def sent_text(self) -> str:
        return json.dumps([m.model_dump() if hasattr(m, "model_dump") else m
                           for r in self.requests for m in r.messages], default=str)


def test_the_envelope_redacts_the_prompt_it_sends():
    from prometheus.learning.llm_envelope import LLMCallEnvelope

    provider = _Capture()
    asyncio.run(LLMCallEnvelope("skill_creator").call(
        provider=provider, model="m", prompt=f"trace: bash(git push https://{GH_CLASSIC}@x)"))
    assert GH_CLASSIC not in provider.sent_text() and REDACTED in provider.sent_text()


def test_the_main_loop_stream_is_left_alone():
    """The agent's own turn is not the learning loop: a pasted token must reach the model."""
    from prometheus.engine.messages import ConversationMessage
    from prometheus.learning.llm_envelope import LLMCallEnvelope
    from prometheus.providers.base import ApiMessageRequest

    provider = _Capture()
    request = ApiMessageRequest(model="m", max_tokens=16,
                                messages=[ConversationMessage.from_user_text(f"use {GH_CLASSIC}")])

    async def go():
        async for _ in LLMCallEnvelope("agent_loop").stream(provider=provider, request=request):
            pass

    asyncio.run(go())
    assert GH_CLASSIC in provider.sent_text()


def test_the_live_compactor_opts_out_and_its_prompt_keeps_the_token():
    """The context compactor summarises the live conversation: its summary replaces
    turns the model is still working from, so a token the user gave must survive."""
    from prometheus.context.compactor import ContextCompactor
    from prometheus.learning.llm_envelope import LLMCallEnvelope

    compactor = ContextCompactor(provider=_Capture(), model="m", effective_limit=8000)
    assert compactor._envelope._redact_prompts is False
    provider = _Capture()
    asyncio.run(LLMCallEnvelope("context_compactor", redact_prompts=False).call(
        provider=provider, model="m", prompt=f"summarise: use {GH_CLASSIC}"))
    assert GH_CLASSIC in provider.sent_text()


def test_gepa_redacts_its_prompts():
    from prometheus.learning.gepa import GEPAOptimizer

    provider = _Capture()
    gepa = GEPAOptimizer.__new__(GEPAOptimizer)
    gepa._provider, gepa._model = provider, "m"
    asyncio.run(gepa._call_provider(f"variant of a skill that uses {NPM}"))
    assert NPM not in provider.sent_text()


def test_the_lcm_summarizer_redacts_its_prompt():
    from prometheus.memory.lcm_summarize import LCMSummarizer
    from prometheus.memory.lcm_types import MessagePart

    provider = _Capture()
    summarizer = LCMSummarizer(provider, model="m")
    asyncio.run(summarizer.summarize_messages([MessagePart(role="user", content=f"my key is {GROQ}")]))
    assert GROQ not in provider.sent_text()


def test_the_vision_digest_redacts_its_text_parts():
    from prometheus.learning.video_ingest.vision_digest import _call_vision, _text_part

    provider = _Capture()
    asyncio.run(_call_vision(provider, "m", [_text_part(f"screen shows {LINEAR}")], 16))
    assert LINEAR not in provider.sent_text()


def test_the_vision_digest_leaves_image_data_alone():
    """Redacting inside base64 image data would corrupt the image."""
    from prometheus.learning.video_ingest.vision_digest import _call_vision

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD" + AWS_KEY_ID}}
    provider = _Capture()
    asyncio.run(_call_vision(provider, "m", [image], 16))
    assert provider.requests[0].messages[0]["content"][0] == image


# ---------------------------------------------------------------------------
# scrub_capture_stores.py covers LCM messages and summaries
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "scrub_capture_stores.py"


def _seed_old_lcm(tmp_path: Path) -> Path:
    """An LCM database with rows written before X.37 — inserted with raw SQL, as
    the pre-X.37 store would have left them, FTS rows included."""
    from prometheus.memory.lcm_summary_store import LCMSummaryStore

    db = tmp_path / "lcm.db"
    store = _store(tmp_path)
    LCMSummaryStore(db)  # creates the summaries tables
    conn = store._conn
    text = f"token {GH_CLASSIC} and more"
    conn.execute(
        "INSERT INTO lcm_messages (id, session_id, turn_index, role, content, content_json,"
        " token_count, timestamp) VALUES ('m1', 's', 0, 'user', ?, ?, 1, 1.0)",
        (text, json.dumps([{"type": "text", "text": text}])))
    conn.execute("INSERT INTO lcm_messages_fts (rowid, content) VALUES "
                 "((SELECT rowid FROM lcm_messages WHERE id='m1'), ?)", (text,))
    conn.execute(
        "INSERT INTO lcm_messages (id, session_id, turn_index, role, content, content_json,"
        " token_count, timestamp) VALUES ('m2', 's', 1, 'user', 'clean', '[]', 1, 2.0)")
    conn.execute("INSERT INTO lcm_summaries (id, parent_ids, source_message_ids, summary_text,"
                 " depth, token_count, created_at, is_leaf) VALUES ('s1', '[]', '[]', ?, 0, 1, 1.0, 1)",
                 (f"summary mentions {GH_OAUTH}",))
    conn.execute("INSERT INTO lcm_summaries_fts (rowid, summary_text) VALUES "
                 "((SELECT rowid FROM lcm_summaries WHERE id='s1'), ?)", (f"summary mentions {GH_OAUTH}",))
    conn.commit()
    return db


def _env(tmp_path: Path) -> dict[str, str]:
    """Every default path lands in tmp_path, never in the real ~/.prometheus."""
    return {**os.environ, "PROMETHEUS_HOME": str(tmp_path / "home")}


def _scrub(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    none = tmp_path / "none"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--telemetry", str(none / "t.db"), "--training",
         str(none / "tr.db"), "--trajectories", str(none), "--lcm", str(tmp_path / "lcm.db"), *args],
        capture_output=True, text=True, timeout=120, env=_env(tmp_path))


def test_scrub_dry_run_counts_lcm_rows_and_touches_nothing(tmp_path):
    db = _seed_old_lcm(tmp_path)
    before = db.read_bytes()
    r = _scrub(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "lcm_messages.content                            1 would change" in r.stdout
    assert "lcm_messages.content_json                       1 would change" in r.stdout
    assert "lcm_summaries.summary_text                      1 would change" in r.stdout
    assert GH_CLASSIC not in r.stdout and GH_OAUTH not in r.stdout
    assert db.read_bytes() == before
    assert not list(tmp_path.glob("*.pre-scrub-*"))


def test_scrub_apply_rewrites_lcm_rebuilds_its_index_and_is_idempotent(tmp_path):
    db = _seed_old_lcm(tmp_path)
    r = _scrub(tmp_path, "--apply")
    assert r.returncode == 0, r.stderr
    assert "3 row(s)/line(s) rewritten" in r.stdout
    [backup] = list(tmp_path.glob("lcm.db.pre-scrub-*"))
    conn = sqlite3.connect(db)
    content, content_json = conn.execute(
        "SELECT content, content_json FROM lcm_messages WHERE id='m1'").fetchone()
    assert GH_CLASSIC not in content and GH_CLASSIC not in content_json
    assert json.loads(content_json)[0]["text"] == f"token {REDACTED} and more"
    assert GH_OAUTH not in conn.execute("SELECT summary_text FROM lcm_summaries").fetchone()[0]
    for fts, col in (("lcm_messages_fts", "content"), ("lcm_summaries_fts", "summary_text")):
        hits = conn.execute(f"SELECT count(*) FROM {fts} WHERE {fts} MATCH ?",
                            ('"' + (GH_CLASSIC if col == "content" else GH_OAUTH) + '"',)).fetchone()[0]
        assert hits == 0, fts
        assert conn.execute(f"SELECT count(*) FROM {fts} WHERE {fts} MATCH 'redacted'").fetchone()[0] >= 1
    assert GH_CLASSIC in sqlite3.connect(backup).execute(
        "SELECT content FROM lcm_messages WHERE id='m1'").fetchone()[0]
    r2 = _scrub(tmp_path, "--apply")
    assert r2.returncode == 0 and "0 row(s)/line(s) rewritten" in r2.stdout


def test_scrub_covers_the_telemetry_json_columns(tmp_path):
    from prometheus.telemetry.tracker import ToolCallTelemetry

    tel_db = tmp_path / "t.db"
    ToolCallTelemetry(tel_db)
    conn = sqlite3.connect(tel_db)
    conn.execute("INSERT INTO subsystem_runs (id, timestamp, subsystem, outcome, summary_json)"
                 " VALUES ('r1', 1.0, 'x', 'success', ?)", (json.dumps({"k": f"v {GH_CLASSIC}"}),))
    conn.execute("INSERT INTO signal_events (timestamp, signal_type, payload, source_subsystem)"
                 " VALUES ('t', 'skill_created', ?, 'skill_creator')",
                 (json.dumps({"trigger_task": f"use {GH_OAUTH}"}),))
    conn.commit()
    r = subprocess.run([sys.executable, str(SCRIPT), "--telemetry", str(tel_db), "--training",
                        str(tmp_path / "no.db"), "--trajectories", str(tmp_path / "none"),
                        "--lcm", str(tmp_path / "no-lcm.db"), "--apply"],
                       capture_output=True, text=True, timeout=120, env=_env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "subsystem_runs.summary_json" in r.stdout and "signal_events.payload" in r.stdout
    summary, = conn.execute("SELECT summary_json FROM subsystem_runs").fetchone()
    payload, = conn.execute("SELECT payload FROM signal_events").fetchone()
    assert GH_CLASSIC not in summary and json.loads(summary)["k"] == f"v {REDACTED}"
    assert GH_OAUTH not in payload
