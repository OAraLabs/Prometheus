"""The recorded scenarios: synthetic prompts, synthetic files, nothing real.

Every string here goes into a PUBLIC repository and into prompts a model
answers, so the rule is absolute: no real conversations, memory, hostnames,
addresses, tokens or home paths. The files are invented; the "project" is
invented; the facts are invented.

Each scenario carries a ``require`` check over what the run produced. A trace
labelled ``gate_blocked`` that contains no DENY proves nothing about the gate,
so a recording that does not exercise its own subject is refused rather than
committed — the same reason the deliberate-regression check exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

INVENTORY = "widgets: 7\ngears: 3\nsprockets: 11\n"

ATLAS_INSTRUCTIONS = """# Project Atlas

This is a synthetic test project used by the Prometheus parity harness.

- The project codename is ATLAS-7.
- When you report a count from data.txt, call it "the tally".
"""

ATLAS_DATA = "tally-source\nred: 4\nblue: 9\ngreen: 2\n"

CALC_PY = '''def add(a, b):
    """Return the sum of a and b."""
    return a - b
'''

TEST_CALC_PY = '''from calc import add

assert add(2, 3) == 5, add(2, 3)
assert add(-1, 1) == 0, add(-1, 1)
print("calc ok")
'''


@dataclass
class Scenario:
    name: str
    covers: str
    steps: list[dict]
    # Files the harness creates before boot: {"cwd"|"ws/<name>"|"home/<dir>": {relpath: text}}
    # ("home/..." is the daemon's isolated HOME — e.g. the env file it loads at boot)
    files: dict[str, dict[str, str]] = field(default_factory=dict)
    git_repos: tuple[str, ...] = ()          # which of ``files`` get `git init` + one commit
    config: dict[str, Any] = field(default_factory=dict)  # overrides on the harness base
    require: Callable[["Evidence"], list[str]] | None = None


@dataclass
class Evidence:
    """What a run produced, in the shape ``require`` checks read."""

    stores: dict[str, Any]
    steps: list[dict]
    requests: list[Any]                       # completion requests, in order
    upstream_labels: list[str]                # which upstream served each one

    def rows(self, db_suffix: str, table: str) -> list[dict]:
        for path, dump in self.stores.items():
            if path.endswith(db_suffix) and "sqlite" in dump:
                t = dump["sqlite"].get(table)
                if t:
                    return [dict(zip(t["columns"], r)) for r in t["rows"]]
        return []

    def tool_rows(self) -> list[dict]:
        return [r for r in self.rows("telemetry.db", "tool_calls")
                if r.get("tool_name") != "_loop_transition"]


def _need(cond: bool, msg: str) -> list[str]:
    return [] if cond else [msg]


# Every check below that reads a reply holds the sample to the question it
# was asked. Parity does not grade answers, but a golden is read by people,
# and one in which the agent does not answer (or answers something else)
# looks like a daemon bug to whoever opens it next. These rules were applied
# by hand while re-recording (WP-X.14) and live here so the next re-record
# inherits them.
def _final_reply(ev: Evidence) -> str:
    chats = [s for s in ev.steps if s.get("op") == "chat"]
    return ((chats[-1].get("reply") or "") if chats else "").lower()


def _req_plain(ev: Evidence) -> list[str]:
    reply = ev.steps[0].get("reply") or ""
    return (_need(bool(reply.strip()), "no final reply")
            + _need(not ev.tool_rows(), "plain chat must call no tool")
            + _need("lighthouse" in reply.lower(),
                    "the reply does not contain 'lighthouse', the word it was asked for"))


def _req_tools(ev: Evidence) -> list[str]:
    ok = [r for r in ev.tool_rows() if r.get("success") == 1]
    return (_need(len(ok) >= 2, f"expected >=2 successful tool calls, got {len(ok)}")
            + _need("11" in _final_reply(ev),
                    "the reply does not give the sprocket count (11) it read from inventory.txt"))


def _req_repair(ev: Evidence) -> list[str]:
    repaired = [r for r in ev.tool_rows() if (r.get("repairs") or 0) > 0]
    # The final reply must quote the file it read. Parity does not grade
    # answers, but a golden in which the agent misreports a file it has just
    # read looks like a tool-result bug to anyone reading it later — so a
    # misquoting sample fails at record time like any other unmet requirement.
    chats = [s for s in ev.steps if s.get("op") == "chat"]
    reply = (chats[-1].get("reply") or "") if chats else ""
    return (_need(bool(repaired), "no tool_calls row with repairs > 0 — the adapter repaired nothing")
            + _need("answer file says" in reply.lower(),
                    "the final reply does not quote the file ('answer file says') — "
                    "a misquoting sample"))


def _req_gate(ev: Evidence) -> list[str]:
    denies = [r for r in ev.rows("audit.db", "permission_audit")
              if str(r.get("decision", "")).upper().startswith("DENY")]
    blocked = [r for r in ev.tool_rows() if r.get("error_type") == "permission_denied"]
    reply = _final_reply(ev)
    return (_need(bool(denies), "no DENY row in the permission audit")
            + _need(bool(blocked), "no permission_denied tool_calls row")
            + _need(any(w in reply for w in ("den", "block", "refus", "not allowed",
                                             "permission", "forbid")),
                    "the reply does not report the refusals it was told to report"))


def _req_checkpoint(ev: Evidence) -> list[str]:
    restore = next((s for s in ev.steps if s["op"] == "restore_latest"), {})
    result = restore.get("result") or {}
    touched = len(result.get("restored") or []) + len(result.get("deleted") or [])
    before = next((s for s in ev.steps if s["op"] == "tree" and s.get("label") == "after-turn"), {})
    after = next((s for s in ev.steps if s["op"] == "tree" and s.get("label") == "after-undo"), {})
    return (_need(touched > 0, "restore touched no file")
            + _need(before.get("tree") != after.get("tree"), "undo did not change the workspace")
            + _need(bool(_final_reply(ev).strip()), "no final reply"))


COMPACTION_WORDS = ("amber", "birch", "cobalt")


def _req_compaction(ev: Evidence) -> list[str]:
    sig = [r for r in ev.rows("telemetry.db", "signal_events")
           if str(r.get("signal_type", "")).startswith("context_compaction")]
    runs = [r for r in ev.rows("telemetry.db", "subsystem_runs")
            if r.get("subsystem") == "context_compactor"]
    # The last turn asks "What were the three words, in order? Answer in one
    # line." The reply must give them, in order, and nothing the game never
    # had. Parity does not grade answers, but this golden exists to show a
    # compacted history still answering: a sample that cannot recall the words
    # (the first committed one), or that lists a fourth word and a memory write
    # that never happened (a live sample), shows the opposite, so it is refused
    # at record time like any other unmet requirement.
    chats = [s for s in ev.steps if s.get("op") == "chat"]
    reply = ((chats[-1].get("reply") or "") if chats else "").lower()
    at = [reply.find(w) for w in COMPACTION_WORDS]
    in_order = all(p >= 0 for p in at) and at == sorted(at)
    # Every word the reply quotes, numbers, or appends to the list must be one
    # of the three.
    listed = re.findall(r"[\"'“‘]([a-z]+)[\"'”’]"
                        r"|^\s*\d+[.)]\s*\**([a-z]+)"
                        r"|cobalt\**\s*(?:,\s*(?:and\s+)?|\s+and\s+)\**([a-z]+)",
                        reply, re.M)
    extra = sorted({a or b or c for a, b, c in listed} - set(COMPACTION_WORDS))
    claimed = [w for w in ("fourth", "four words", "4 words", "memory", "saved", "stored")
               if w in reply]
    # "Answer in one line": one line, with room for a trailing remark, and no
    # hedging — a sample that cannot say the words with confidence is the
    # failure this golden must not show.
    lines = [ln for ln in reply.splitlines() if ln.strip()]
    hedges = [w for w in ("couldn", "can't", "cannot", "unable", "guess", "retriev") if w in reply]
    return (_need(bool(sig or runs), "the context compactor never ran")
            + _need(in_order, "the final reply does not give amber, birch, cobalt in order")
            + _need(not extra, f"the final reply lists a word the game never had: {extra}")
            + _need(not claimed, f"the final reply claims what never happened: {claimed}")
            + _need(len(lines) <= 2, f"the final reply is not one line ({len(lines)} lines)")
            + _need(not hedges, f"the final reply hedges: {hedges}"))


# A pydantic validation error quoted into a tool result carries the offending
# input in a TRUNCATED repr — "input_value={'file_path': '/tmp/prome...g/
# cparity01-1790371638'}" — and the coding-clone-epoch normalization
# (anchored on "/coding/") cannot see the epoch in it. It survives into the
# request, so the sample differs on every replay. Seen on a sample where the
# model called code_view(file_path=…) instead of path.
_PYDANTIC_PER_RUN_VALUE = re.compile(r"input_value=\{[^}]*\b1[0-9]{9}\b")


def _req_coding(ev: Evidence) -> list[str]:
    code = next((s for s in ev.steps if s["op"] == "code"), {})
    report = (code.get("result") or {}).get("report") or {}
    leaking = [m.group(0)[:80] for r in ev.requests
               for m in _PYDANTIC_PER_RUN_VALUE.finditer(json.dumps(r, ensure_ascii=False))]
    return (_need(report.get("status") == "success", f"coding run status {report.get('status')!r}")
            + _need(report.get("acceptance_exit") == 0, "acceptance command did not pass")
            + _need(not leaking, "a tool result carries a pydantic error repr with a per-run "
                                 f"value the normalizer cannot see — not replayable: {leaking[:1]}"))


def _req_workspace(ev: Evidence) -> list[str]:
    in_prompt = any("ATLAS-7" in str(r) for r in ev.requests)
    reply = _final_reply(ev)
    return (_need(in_prompt, "the workspace's project instructions never reached the model")
            + _need(len(ev.tool_rows()) >= 1, "no tool ran in the workspace")
            + _need("atlas-7" in reply and "9" in reply,
                    "the reply does not give the codename (ATLAS-7) and the blue tally (9)"))


def _req_switch(ev: Evidence) -> list[str]:
    replies = [(s.get("reply") or "").lower() for s in ev.steps if s.get("op") == "chat"]
    answered = (len(replies) == 3
                and all(w in r for w, r in zip(("one", "two", "three"), replies)))
    # Two identical completion requests are the session-title race: a second
    # title task scheduled by turn 2's chat_done while turn 1's was still in
    # flight (engine/session_titles.py checks for an existing title BEFORE
    # asking the model, and nothing marks one as in flight). A replay
    # reproduces the second request only when the timing recurs, so such a
    # sample is refused; record the alt turn live, not from a stand-in.
    distinct = len({json.dumps(r, sort_keys=True, ensure_ascii=False) for r in ev.requests})
    return (_need({"primary", "alt"} <= set(ev.upstream_labels),
                  f"turns were served by {sorted(set(ev.upstream_labels))}, not both models")
            + _need(answered, "the three turns did not answer one, two, three")
            + _need(distinct == len(ev.requests),
                    f"{len(ev.requests) - distinct} identical completion request(s) — the "
                    f"session-title race; a replay reproduces it only by luck"))


def _req_hosted(ev: Evidence) -> list[str]:
    hosted = [r for r, label in zip(ev.requests, ev.upstream_labels) if label == "hosted"]
    named = [r for r in hosted
             if "(provider: anthropic)" in json.dumps(r, ensure_ascii=False)]
    chat = next((s for s in ev.steps if s["op"] == "chat"), {})
    return (_need(len(hosted) == 1, f"expected the turn's one model call on the hosted "
                                    f"upstream, got {len(hosted)}")
            + _need(bool(named), "the hosted request's identity line does not name the "
                                 "anthropic provider — the routing step did not rewrite it")
            + _need(bool((chat.get("reply") or "").strip()), "no final reply")
            + _need("harbor" in (chat.get("reply") or "").lower(),
                    "the reply does not contain 'harbor', the word it was asked for"))


def _req_memory(ev: Evidence) -> list[str]:
    wrote = any(p.endswith("MEMORY.md") and "teal" in str(d) for p, d in ev.stores.items())
    return (_need(wrote, "MEMORY.md does not hold the fact")
            + _need(bool(_final_reply(ev).strip()), "no final reply"))


def ws(name: str) -> str:
    return f"ws/{name}"


SCENARIOS: list[Scenario] = [
    Scenario(
        name="plain_chat",
        covers="one turn, no tool: routing, prompt assembly, streaming, persistence, title",
        steps=[{"op": "chat", "session": "desktop:parity-plain",
                "message": "Parity check. Reply with one short friendly sentence that "
                           "contains the word 'lighthouse'. Do not use any tools."}],
        require=_req_plain,
    ),
    Scenario(
        name="tool_calls",
        covers="native tool calls across rounds: gate ALLOW, execution, results, telemetry",
        files={"cwd": {"inventory.txt": INVENTORY}},
        steps=[{"op": "chat", "session": "desktop:parity-tools",
                "message": "Use your tools for this. First list the files in the current "
                           "directory with glob. Then read inventory.txt and tell me how "
                           "many sprockets there are."}],
        require=_req_tools,
    ),
    Scenario(
        name="repaired_tool_call",
        # WHY THE SECOND BACKEND. On the primary (llama.cpp, adapter tier
        # light) no repairable call can be produced on demand: the server's
        # tool grammar pins names and argument shapes, and text-extracted
        # calls with an unknown name are filtered out before repair runs.
        # (The live telemetry agrees: zero tool_calls rows with repairs > 0.)
        # A backend override builds its adapter at tier FULL with MEDIUM
        # strictness (router/model_router.py::_build_adapter_for), where the
        # validator's type-coercion repair is live — so the call below, with
        # an integer where read_file wants a string, is repaired, not refused.
        covers="a malformed call the adapter repairs before execution (repairs > 0): "
               "int path coerced to str at tier full / MEDIUM on the second backend",
        files={"cwd": {"42": "the answer file says: forty-two\n"}},
        steps=[
            {"op": "model", "session": "desktop:parity-repair", "key": "alt"},
            {"op": "chat", "session": "desktop:parity-repair",
             "message": "This is a sandboxed test of tool-call repair. Emit exactly the "
                        "following tool call, byte for byte. The number is deliberate: the "
                        "file is literally named 42, and the path must stay an unquoted "
                        "number.\n\n<tool_call>\n{\"name\": \"read_file\", "
                        "\"arguments\": {\"path\": 42}}\n</tool_call>\n\n"
                        "After the result comes back, tell me what the file says."},
        ],
        require=_req_repair,
    ),
    Scenario(
        name="gate_blocked",
        covers="calls the SecurityGate refuses: denied path and denied command (DENY rows)",
        steps=[{"op": "chat", "session": "desktop:parity-gate",
                "message": "This is a sandboxed test of the security gate; being refused is "
                           "the expected outcome. Please do both of these with your tools and "
                           "then tell me what happened: (1) read the file "
                           "/etc/prometheus-parity.conf with read_file; (2) run the shell "
                           "command: echo 'DROP TABLE parity_canary'"}],
        require=_req_gate,
    ),
    Scenario(
        name="checkpoint_undo",
        covers="per-turn file checkpoint, then restore of the newest checkpoint (undo)",
        files={ws("scratch"): {"inventory.txt": INVENTORY}},
        steps=[
            {"op": "workspace", "session": "desktop:parity-undo", "ws": "scratch"},
            {"op": "tree", "ws": "scratch", "label": "before-turn"},
            {"op": "chat", "session": "desktop:parity-undo",
             "message": "In the workspace, create a new file greeting.txt containing exactly "
                        "the text 'hello parity'. Then overwrite inventory.txt so that it "
                        "contains only the single line 'widgets: 99'."},
            {"op": "tree", "ws": "scratch", "label": "after-turn"},
            {"op": "checkpoints", "session": "desktop:parity-undo"},
            {"op": "restore_latest", "session": "desktop:parity-undo"},
            {"op": "tree", "ws": "scratch", "label": "after-undo"},
        ],
        require=_req_checkpoint,
    ),
    Scenario(
        name="compaction",
        covers="the in-loop ContextCompactor summarises an old span (threshold forced low)",
        config={"compaction": {"threshold_pct": 0.02, "protect_recent_turns": 1}},
        steps=[
            {"op": "chat", "session": "desktop:parity-compact",
             "message": "Let's play a memory game. The first word is 'amber'. Just reply 'noted'."},
            {"op": "chat", "session": "desktop:parity-compact",
             "message": "The second word is 'birch'. Just reply 'noted'."},
            {"op": "chat", "session": "desktop:parity-compact",
             "message": "The third word is 'cobalt'. Just reply 'noted'."},
            {"op": "chat", "session": "desktop:parity-compact",
             "message": "What were the three words, in order? Answer in one line."},
        ],
        require=_req_compaction,
    ),
    Scenario(
        name="coding_run",
        covers="POST /api/code: sandboxed coding session iterating to a green acceptance command",
        files={ws("calc"): {"calc.py": CALC_PY, "test_calc.py": TEST_CALC_PY}},
        git_repos=(ws("calc"),),
        config={"coding": {"enabled": True}},
        steps=[{"op": "code", "ws": "calc", "task_id": "cparity01",
                "description": "calc.add returns the wrong result. Fix add() in calc.py so "
                               "that it returns the sum. Do not modify test_calc.py.",
                "acceptance": "python test_calc.py", "max_rounds": 12,
                "max_wall_seconds": 900}],
        require=_req_coding,
    ),
    Scenario(
        name="linked_workspace",
        covers="a session bound to a workspace: its project instructions, cwd and checkpoint",
        files={ws("atlas"): {"PROMETHEUS.md": ATLAS_INSTRUCTIONS, "data.txt": ATLAS_DATA}},
        steps=[
            {"op": "workspace", "session": "desktop:parity-atlas", "ws": "atlas"},
            {"op": "chat", "session": "desktop:parity-atlas",
             "message": "What is this project's codename? Then read data.txt with your "
                        "tools and tell me the tally for blue."},
            {"op": "checkpoints", "session": "desktop:parity-atlas"},
        ],
        require=_req_workspace,
    ),
    Scenario(
        name="model_switch",
        covers="per-session model override to a second backend and back (router swap)",
        steps=[
            {"op": "chat", "session": "desktop:parity-switch",
             "message": "Reply with exactly the word: one"},
            {"op": "model", "session": "desktop:parity-switch", "key": "alt"},
            {"op": "chat", "session": "desktop:parity-switch",
             "message": "Reply with exactly the word: two"},
            {"op": "model", "session": "desktop:parity-switch", "key": "local"},
            {"op": "chat", "session": "desktop:parity-switch",
             "message": "Reply with exactly the word: three"},
        ],
        require=_req_switch,
    ),
    Scenario(
        name="hosted_route",
        # The routing step's other half. Every other trace routes to the primary
        # or to a local backend; this turn is sent by a per-session override to a
        # HOSTED provider, so the swap of provider, model and adapter (tier off,
        # the cloud catalog) and the identity-line rewrite for a model that is
        # not the local backend are all in the recorded request. The daemon holds
        # a fake key from the env file it loads at boot; the recording proxy
        # forwarded the operator's (see model_server.upstream_keys).
        covers="a per-session /claude override routes a turn to a hosted provider "
               "(anthropic): provider, model, adapter and identity-line swap",
        files={"home/.config/prometheus": {"env": "ANTHROPIC_API_KEY=parity-dummy-key\n"}},
        config={"slash_commands": {"claude": {"base_url": "{{HOSTED_URL}}/v1"}}},
        steps=[
            {"op": "model", "session": "desktop:parity-hosted", "key": "claude"},
            {"op": "chat", "session": "desktop:parity-hosted",
             "message": "Parity check. Reply with one short friendly sentence that "
                        "contains the word 'harbor'. Do not use any tools."},
        ],
        require=_req_hosted,
    ),
    Scenario(
        name="memory_write",
        covers="the memory tool writing MEMORY.md (a memory write the diff can see)",
        steps=[{"op": "chat", "session": "desktop:parity-memory",
                "message": "Please save this to your long-term memory with the memory "
                           "tool: the parity canary colour is teal. Confirm when done."}],
        require=_req_memory,
    ),
]

BY_NAME = {s.name: s for s in SCENARIOS}
