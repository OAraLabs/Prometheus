"""The parity harness's own contract (WP-1.2) — fast, hermetic, no daemon.

The replay itself runs in ``.github/workflows/parity.yml`` against a real
daemon process. These tests pin the pieces a WRONG replay would hide behind:

* the normalizer — every rewrite is a place a regression could hide, so each
  rule's reach is pinned in both directions (what it hides, what it leaves);
* the replay matcher — a divergent request must be flagged, never absorbed;
* the differ — a changed row must land in the category a reviewer reads;
* the committed traces — each must still exercise its own subject, and none
  may carry a private identifier (the repo is public).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from parity import compare as cmp  # noqa: E402
from parity import normalize as norm  # noqa: E402
from parity import traces  # noqa: E402
from parity.model_server import Exchange, ModelServer, ServerState  # noqa: E402
from parity.runner import RunOutput  # noqa: E402
from parity.scenarios import BY_NAME, SCENARIOS, Evidence  # noqa: E402

FIXTURES = traces.trace_dir(REPO)


# ── the normalizer ──────────────────────────────────────────────────────────

ENV = ("# Environment\n- OS: Linux 7.0.0-31-generic\n- Architecture: x86_64\n"
       "- Shell: bash\n- Working directory: /tmp/prometheus-parity/cwd\n"
       "- Date: 2026-09-24\n- Python: 3.11.15\n- Model: qwen (provider: llama_cpp)")


def test_request_rules_hide_only_the_host_lines():
    out = norm.normalize_request({"messages": [{"role": "system", "content": ENV}]})
    text = out["messages"][0]["content"]
    for hidden in ("7.0.0-31", "2026-09-24", "3.11.15"):
        assert hidden not in text
    # Everything the harness PINS (the shell) or the daemon decides stays visible.
    for kept in ("x86_64", "- Shell: bash", "/tmp/prometheus-parity/cwd",
                 "- Model: qwen (provider: llama_cpp)"):
        assert kept in text


def test_request_normalization_is_idempotent():
    body = {"messages": [{"content": ENV + " toolu_a16616d8428c /coding/c1-1790299243"}]}
    once = norm.normalize_request(body)
    assert norm.normalize_request(once) == once


def test_daemon_minted_tool_ids_become_per_request_ordinals():
    body = {"a": "toolu_a16616d8428c", "b": "toolu_a16616d8428c", "c": "toolu_14a20ecb36dd"}
    assert norm.normalize_request(body) == {"a": "<toolu:1>", "b": "<toolu:1>", "c": "<toolu:2>"}
    # A SERVER-assigned id is recorded and replayed verbatim — never rewritten.
    assert norm.normalize_request({"id": "call_fl_9bb696c8"}) == {"id": "call_fl_9bb696c8"}


def test_observable_ordinals_keep_references_linked():
    u1, u2 = "b2dd20c26e5a4ad59502bb8eb0d5d893", "0c85488d-435d-4da7-9ad3-b7eac084340f"
    out = norm.normalize_observables({"steps": [{"row": u1, "ref": f"see {u1}", "other": u2}],
                                      "stores": {}})
    step = out["steps"][0]
    assert step == {"row": "<uuid:1>", "ref": "see <uuid:1>", "other": "<uuid:2>"}


def test_field_rules_keep_null_and_reach_json_in_text_columns():
    obs = {"steps": [], "stores": {"home/.prometheus/telemetry.db": {"sqlite": {"t": {
        "columns": ["timestamp", "latency_ms", "summary_json", "outcome"],
        "rows": [[1790298963.7, None, '{"duration_ms": 845.7, "span": 4}', "success"]],
    }}}}}
    row = norm.normalize_observables(obs)["stores"]["home/.prometheus/telemetry.db"]["sqlite"]["t"]["rows"][0]
    assert row[0] == "<time>"
    assert row[1] is None                       # absence is still compared
    assert row[2] == {"$json": {"duration_ms": "<ms>", "span": 4}}
    assert row[3] == "success"


def test_field_rules_do_not_touch_counts_or_outcomes():
    # A regression in HOW MANY or WHICH is exactly what must stay visible.
    obs = {"steps": [{"repairs": 1, "success": 0, "error_type": "permission_denied",
                      "input_tokens": 42, "decision": "deny"}], "stores": {}}
    assert norm.normalize_observables(obs)["steps"][0] == obs["steps"][0]


def test_every_rule_is_documented():
    rules = norm.all_rules()
    assert len({r.name for r in rules}) == len(rules)
    for r in rules:
        assert r.replaces and r.why and r.cost, r.name


def test_response_binding_rebinds_a_path_split_across_stream_chunks():
    chunks = ["root /t/coding/cpar", "ity01-179", "02992", "43/calc.py done"]
    body = "\n\n".join("data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
                       for c in chunks) + "\n\ndata: [DONE]\n\n"
    out = norm.bind_response(body, "Repository root: /t/coding/cparity01-1790299300.")
    got = [json.loads(ln[6:])["choices"][0]["delta"]["content"]
           for ln in out.split("\n") if ln.startswith("data: {")]
    assert "".join(got) == "root /t/coding/cparity01-1790299300/calc.py done"
    assert [len(c) for c in got] == [len(c) for c in chunks]   # granularity unchanged


# ── the replay matcher (a real HTTP round trip) ─────────────────────────────

def _completion(content: str) -> str:
    return ("data: " + json.dumps({"choices": [{"delta": {"content": content}}]})
            + "\n\ndata: [DONE]\n\n")


def _post(port: int, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@pytest.fixture
def replay_server():
    recorded = [
        Exchange("POST", "/v1/chat/completions", 200, "text/event-stream",
                 _completion("first"), request=norm.normalize_request({"q": 1}), upstream="primary"),
        Exchange("POST", "/v1/chat/completions", 200, "text/event-stream",
                 _completion("second"), request=norm.normalize_request({"q": 2}), upstream="primary"),
        Exchange("GET", "/props", 200, "application/json", '{"n_ctx": 1}', upstream="primary"),
    ]
    state = ServerState(mode="replay", recorded=recorded)
    server = ModelServer(state)
    port = server.start(["primary"])["primary"]
    yield state, port
    server.stop()


def test_exact_requests_match_in_any_order(replay_server):
    state, port = replay_server
    assert "second" in _post(port, {"q": 2})[1]
    assert "first" in _post(port, {"q": 1})[1]
    assert [s.matched for s in state.served] == [True, True]


def test_a_divergent_request_is_served_but_flagged(replay_server):
    state, port = replay_server
    code, body = _post(port, {"q": 99})
    assert code == 200 and "first" in body          # the turn continues...
    assert state.served[0].matched is False         # ...and the divergence is on record


def test_a_request_beyond_the_recording_is_refused_not_retried(replay_server):
    state, port = replay_server
    _post(port, {"q": 1})
    _post(port, {"q": 2})
    code, _ = _post(port, {"q": 3})
    assert code == 400                              # a 5xx would be retried and look like a hang
    assert state.served[-1].recorded_index is None


def test_probes_are_answered_without_being_consumed(replay_server):
    state, port = replay_server
    for _ in range(3):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as r:
            assert json.loads(r.read()) == {"n_ctx": 1}
    assert state.consumed == set()


# ── the differ ──────────────────────────────────────────────────────────────

def _run(steps: list, stores: dict) -> RunOutput:
    return RunOutput(scenario="t", mode="replay", steps=steps, stores=stores, exchanges=[],
                     served=[], unconsumed=[], turns=[], wall_offset_ns=0, rss=[],
                     boot_seconds=1.0, shutdown="clean", daemon_log=Path("/dev/null"))


def _tool_store(success: int) -> dict:
    return {"home/.prometheus/telemetry.db": {"sqlite": {"tool_calls": {
        "columns": ["tool_name", "success"], "rows": [["read_file", success]]}}}}


def test_identical_runs_are_parity():
    run = _run([{"op": "chat", "reply": "hi"}], _tool_store(1))
    expected = cmp.expected_from(run, Path("/"))
    assert cmp.compare(run, expected, Path("/")).exit_code == 0


def test_a_changed_tool_row_lands_under_tool_calls():
    expected = cmp.expected_from(_run([], _tool_store(1)), Path("/"))
    res = cmp.compare(_run([], _tool_store(0)), expected, Path("/"))
    assert res.exit_code == 1 and list(res.diffs) == ["tool_calls"]


def test_a_changed_reply_lands_under_final_reply():
    expected = cmp.expected_from(_run([{"op": "chat", "reply": "a"}], {}), Path("/"))
    res = cmp.compare(_run([{"op": "chat", "reply": "b"}], {}), expected, Path("/"))
    assert list(res.diffs) == ["final_reply"]


def test_an_unexpected_new_store_is_a_diff_not_ignored():
    expected = cmp.expected_from(_run([], {}), Path("/"))
    res = cmp.compare(_run([], {"home/.prometheus/new.db": {"sqlite": {}}}), expected, Path("/"))
    assert res.exit_code == 1 and "other" in res.diffs


def test_a_harness_error_is_never_reported_as_parity():
    run = _run([], {})
    run.errors.append("daemon did not answer")
    assert cmp.compare(run, cmp.expected_from(_run([], {}), Path("/")), Path("/")).exit_code == 2


# ── the committed traces ────────────────────────────────────────────────────

REQUIRED = {"plain_chat", "tool_calls", "repaired_tool_call", "gate_blocked",
            "checkpoint_undo", "compaction", "coding_run", "linked_workspace", "model_switch"}


def test_every_required_scenario_is_recorded():
    assert REQUIRED <= set(traces.available(REPO))
    assert {s.name for s in SCENARIOS} == set(traces.available(REPO))


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_each_trace_still_exercises_its_subject(name):
    """A trace labelled gate_blocked with no DENY in it proves nothing."""
    trace = traces.load_trace(REPO, name)
    expected = traces.load_expected(REPO, name)
    completions = [e for e in trace["exchanges"]
                   if e["method"] == "POST" and e["path"] == "/v1/chat/completions"]
    ev = Evidence(stores=expected["stores"], steps=expected["steps"],
                  requests=[e["request"] for e in completions],
                  upstream_labels=[e["upstream"] for e in completions])
    assert BY_NAME[name].require(ev) == []


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_committed_files_are_fixed_points_of_the_current_rules(name):
    """A rule added without `rebaseline` would make replay and recording
    disagree for a reason that is not the daemon's."""
    expected = traces.load_expected(REPO, name)
    assert norm.normalize_observables(expected) == expected
    for ex in traces.load_trace(REPO, name)["exchanges"]:
        if ex.get("request") is not None:
            assert norm.normalize_request(ex["request"]) == ex["request"]


# The hook's OWN patterns, resolved by bash from .githooks/pre-commit — never
# transcribed here (see tests/test_sdist_contents.py for why a transcription
# is a different regex that happens to share a spelling).
from tests.test_sdist_contents import _ere_to_python, _hook_checks  # noqa: E402

# What the hook does not look for but a recording could leak: a real
# address, a real user's home, a tailnet name. The harness's own isolated
# HOME (/tmp/prometheus-parity/home) is synthetic and the one exemption.
EXTRA = [
    ("IPv4 address", re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("home path", re.compile(r"(?<!/prometheus-parity)/(?:home|Users)/[A-Za-z0-9._-]+")),
    ("tailnet name", re.compile(r"\.ts\.net\b|tailscale", re.I)),
]


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_committed_traces_carry_no_private_identifier(path):
    text = path.read_text(encoding="utf-8")
    hits = [(label, m.group(0)) for label, rx in EXTRA for m in rx.finditer(text)]
    checks, placeholder, allowlist = _hook_checks()
    for label, pattern, skip_ph, _scope, icase in checks:
        rx = re.compile(_ere_to_python(pattern), re.IGNORECASE if icase == "1" else 0)
        for line in text.splitlines():
            if not rx.search(line) or not rx.search(re.sub(_ere_to_python(allowlist), "", line)):
                continue
            if skip_ph == "1" and re.search(_ere_to_python(placeholder), line, re.I):
                continue
            hits.append((label, line.strip()[:120]))
    assert hits == []
