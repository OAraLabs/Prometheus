"""The local tool-calling and context gate, a–f, through the real daemon.

The daemon boots from config as a child process (tests/support/gate_daemon.py)
and only the model is replaced: tests/support/scripted_llama.py keeps the
4090's llama-server constraints (32k window and its 400, calls only to offered
tools, ``tool_calls`` reported on a cut-off call, one prompt cache). Every
check reads what the CLIENT received over the WebSocket, as Beacon would, or,
for (e) and (f), what the model server received.

  a  sub-agent spawn            d  no silent turn: every ending reaches Beacon
  b  deferred tool, same turn   e  fixed overhead, per component
  c  32k coding session         f  cloud unchanged

Written before any fix. What 0.9.12 fails is ``xfail(strict=True)``, so a fix
that makes a check pass fails the run until its marker comes off. To read the
gate against another release, set ``LOCAL_GATE_CHECKOUT`` to its checkout and
run with ``--runxfail``.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.support.gate_daemon import CLOUD_KEY, GateDaemon, build_and_boot
from tests.support.scripted_llama import ScriptedLlamaServer
from tests.support.served_template import recorded_props

pytestmark = pytest.mark.integration

#: The compaction line on a 32,768 window before 0.9.12: 80% of the window,
#: which binds while the reply room is 4,096 and the largest recent result is
#: under 2,458 tokens. A pass while the request is under it is "early".
LINE_32K = 26_214


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    """One daemon, a local model stand-in, and a cloud stand-in a session can switch to."""
    tmp = tmp_path_factory.mktemp("gate")
    ws = str(tmp / "workspace")
    with ScriptedLlamaServer(props=recorded_props(), workspace=ws) as model, \
            ScriptedLlamaServer(n_ctx=1_000_000, native_tool_grammar=False, workspace=ws) as cloud_model:
        daemon = build_and_boot(tmp, model.url, cloud_url=cloud_model.url)
        try:
            yield daemon, model, cloud_model
        finally:
            daemon.stop()


@pytest.fixture(scope="module")
def local(gate):
    return gate[0], gate[1]


@pytest.fixture(scope="module")
def coding_turn(local):
    """The 32k multi-file session, run once; (c)'s checks read it."""
    daemon, model = local
    before = len(model.exchanges)
    turn = daemon.turn(
        "[gate:coding] Read every module in src/ledger/ (nine files, one at a time), then add a one-line "
        "docstring to compute_total in src/ledger/billing.py. Tell me what you changed.", timeout=420)
    return turn, model.exchanges[before:]


def _explain(turn, daemon: GateDaemon) -> str:
    return (f"tools started: {turn.tools_started}\nstreamed: {turn.streamed[:300]!r}\n"
            f"final stored: {turn.final_stored_text[:300]!r}\n--- daemon log tail ---\n{daemon.log_tail(25)}")


# ---------------------------------------------------------------- (a)


@pytest.mark.xfail(strict=True, reason="gate a on 0.9.12: `agent` is deferred and cannot be called on a local turn")
def test_a_a_sub_agent_is_spawned_and_its_answer_reaches_the_client(local):
    daemon, model = local
    turn = daemon.turn("[gate:agent] Use a sub-agent to list the files in your workspace.")
    assert "agent" in turn.tools_started, _explain(turn, daemon)
    assert model.of("sub"), "the sub-agent never sent the model a request"
    assert "The sub-agent" in turn.streamed or "Done." in turn.streamed, _explain(turn, daemon)


# ---------------------------------------------------------------- (b)


@pytest.mark.xfail(strict=True, reason="gate b on 0.9.12: a selected tool never joins the run's tools")
@pytest.mark.parametrize("script,tool", [
    ("cron", "cron_create"), ("wiki", "wiki_query"), ("memory", "lcm_grep"), ("mcp", "mcp__docs__lookup"),
])
def test_b_a_deferred_tool_is_found_and_called_in_the_same_turn(local, script, tool):
    daemon, _ = local
    turn = daemon.turn(f"[gate:{script}] Please take care of this for me.")
    assert tool in turn.tools_started, _explain(turn, daemon)
    assert turn.selects() <= 1, f"{turn.selects()} tool_search selects\n" + _explain(turn, daemon)
    assert turn.streamed.strip(), _explain(turn, daemon)


# ---------------------------------------------------------------- (c)


def _passes(turn) -> list[dict]:
    """Compaction passes the client was told about (a cached reuse is not a pass)."""
    return [p for p in turn.of("context_compaction") if p.get("stage") in ("start", "noop")]


@pytest.mark.xfail(strict=True, reason=(
    "gate c on 0.9.12: the 8192 output cap moves the projected-overflow line to 24,576 minus the largest "
    "recent result, so the session compacts while it is still under the 26,214 line"))
def test_c_a_long_coding_session_does_not_compact_early(local, coding_turn):
    daemon, _ = local
    turn, _ = coding_turn
    # Early = under the line, or forced by the reply room ("projected"): with
    # each result under 2,458 tokens, a 4096 room never forces a pass below the
    # 80% line, so a projected pass means the output cap took the room. (A pass
    # that only trims results sends the client no frame; the summary passes do.)
    early = [p for p in _passes(turn)
             if int(p.get("tokens_before") or 0) < LINE_32K or p.get("trigger") == "projected"
             or int(p.get("ceiling") or LINE_32K) < LINE_32K]
    assert not early, f"compaction started early (line {LINE_32K}): " + json.dumps(early)[:600]


def test_c_the_session_finishes_its_task_after_compacting(local, coding_turn):
    daemon, model = local
    turn, exchanges = coding_turn
    if not _passes(turn):
        pytest.skip("the session never reached the compaction line; raise LEDGER_MODULE_CHARS")
    assert "edit_file" in turn.tools_started, _explain(turn, daemon)
    billing = (daemon.workspace / "src" / "ledger" / "billing.py").read_text(encoding="utf-8")
    assert '"""Sum one ledger' in billing, "the edit did not land"
    assert "docstring" in turn.streamed, _explain(turn, daemon)


def test_c_the_window_is_never_overrun(local, coding_turn):
    _, model = local
    over = [e.prompt_tokens for e in model.exchanges if e.status == 400]
    assert not over, f"the daemon sent prompts over the 32k window: {over}"


# ---------------------------------------------------------------- (d)


@pytest.mark.parametrize("script,message,expect", [
    ("repeat", "[gate:repeat] Find the gate files.", "Halted: no progress"),
    ("empty", "[gate:empty] Say something.", "empty response twice"),
    ("preflight", "[gate:hello] " + "very long paste " * 9000, "too long"),
])
def test_d_every_ending_reaches_the_client_with_its_reason(local, script, message, expect):
    daemon, _ = local
    turn = daemon.turn(message)
    assert turn.finished, _explain(turn, daemon)
    assert expect in turn.final_stored_text, "precondition: the ending under test did not happen\n" + _explain(turn, daemon)
    assert expect in turn.streamed, "the stored ending never reached the client\n" + _explain(turn, daemon)


# ---------------------------------------------------------------- (e)

#: PROPOSED (audit 2026-10-16, pending approval): the most Prometheus itself may
#: add to round 0 of a local turn, in server tokens: base prompt, environment,
#: documents, skills, formatter text, framing and the shipped tool schemas,
#: i.e. everything but the operator's own files and saved-file listing.
PROMETHEUS_OWNED_TARGET = 5_000


def _prometheus_owned_tokens(model, ex, operator_texts: list[str]) -> int:
    """Round 0 in server tokens, less the operator's own content.

    Bootstrap files and project instructions go in verbatim and are cut by
    content. The memory section re-renders its files (one bullet per line),
    so it is cut by position: it runs from ``# Memory`` to ``# Saved Files``,
    and the saved-file listing from there to the tool list the formatter
    appends (prompt_assembler.py's section order)."""
    system = ex.system
    for text in operator_texts:
        body = text.strip()
        if body and body in system:
            system = system.replace(body, "", 1)
    for start, end in (("\n# Memory", "\n# Saved Files"), ("\n# Saved Files", "\nYou have access to these tools:")):
        i = system.find(start)
        if i < 0:
            continue
        j = system.find(end, i + len(start))
        system = system[:i] + (system[j:] if j >= 0 else "")
    tools = sum(len(json.dumps(t["function"], ensure_ascii=False)) for t in ex.request.get("tools") or [])
    return model.tokens(len(system)) + model.tokens(tools)


def _sections(system: str) -> list[tuple[str, str]]:
    parts, cur, title = [], [], "(start)"
    for line in system.split("\n"):
        if line.startswith("# ") or line.startswith("<!-- Bootstrap:") or line.startswith("You have access to these tools:"):
            if cur:
                parts.append((title, "\n".join(cur)))
            title, cur = line.strip()[:60], [line]
        else:
            cur.append(line)
    if cur:
        parts.append((title, "\n".join(cur)))
    return parts


def test_e_round_zero_overhead_is_measured_per_component(local):
    daemon, model = local
    daemon.turn("[gate:hello] Hi.")
    ex = next(e for e in model.of("hello") if e.status == 200)
    rows = [(t, model.tokens(len(s))) for t, s in _sections(ex.system)]
    tools = model.tokens(sum(len(json.dumps(t["function"], ensure_ascii=False)) for t in ex.request.get("tools") or []))
    measured = sum(n for _, n in rows) + tools
    # The components account for the request the server counted (framing aside).
    assert abs(ex.prompt_tokens - measured) <= 600, (ex.prompt_tokens, measured)


@pytest.mark.xfail(strict=True, reason="gate e on 0.9.12: no per-component overhead reaches the client")
def test_e_the_client_can_read_the_overhead_breakdown(local):
    daemon, _ = local
    status = daemon.get("/api/status")
    ctx = status.get("context") or {}
    assert ctx.get("overhead_components"), "GET /api/status reports no per-component prompt overhead"


@pytest.mark.xfail(strict=True, reason="gate e on 0.9.12: Prometheus's own round-0 overhead is over the proposed target")
def test_e_prometheus_owned_overhead_is_within_target(local):
    daemon, model = local
    ex = next(e for e in model.of("hello") if e.status == 200) if model.of("hello") else None
    if ex is None:
        daemon.turn("[gate:hello] Hi.")
        ex = next(e for e in model.of("hello") if e.status == 200)
    owned = _prometheus_owned_tokens(model, ex, daemon.operator_texts)
    assert owned <= PROMETHEUS_OWNED_TARGET, (
        f"Prometheus-owned round-0 overhead {owned} > {PROMETHEUS_OWNED_TARGET} (of {ex.prompt_tokens} in all)")


# ---------------------------------------------------------------- (f)


def test_f_a_cloud_turn_offers_every_tool_and_keeps_its_prefix(gate):
    daemon, _, model = gate
    sid = "beacon:gate-cloud"
    daemon.use_model(sid, CLOUD_KEY)
    turn = daemon.turn("[gate:cloud] Read two of the ledger files.", sid=sid)
    assert turn.streamed.strip(), _explain(turn, daemon)
    rounds = [e for e in model.of("cloud") if e.status == 200]
    assert len(rounds) >= 3, f"expected 3 rounds, saw {len(rounds)}"
    first = rounds[0]
    assert "tool_search" in first.offered and "agent" in first.offered and len(first.offered) >= 40, first.offered
    for later in rounds[1:]:
        assert later.system == first.system, "the cloud system prompt changed mid-turn"
        assert later.request.get("tools") == first.request.get("tools"), "the cloud tool list changed mid-turn"
        assert later.cached_tokens >= first.prompt_tokens - 50, "the cached prefix did not survive the round"
    assert not re.search(r"You have access to these tools:", first.system), "a cloud prompt carries the local tool prose"
