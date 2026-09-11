"""`POST /api/chat` must reach the model. It has not since 2026-06-27.

THE DEFECT (#305)
-----------------
The handler built its call as::

    result = await agent_loop.run_async(
        system_prompt=system_prompt,
        messages=session.get_messages(),
        tools=app.state.skill_registry.list_schemas() if app.state.skill_registry else None,
    )

`list_schemas()` is a **ToolRegistry** method. `SkillRegistry` has
`register` / `get` / `list_skills` / `reload_user_skills` — and no
`list_schemas`. Two registries confused at one call site, copied from
`gateway/discord.py` where `self.tool_registry` is the right object.

Python evaluates arguments before the call, so every request died on
`AttributeError: 'SkillRegistry' object has no attribute 'list_schemas'`
before `run_async` was entered. Dated to `66fd820`, 2026-06-27 (#71).

THE ARGUMENT IS REMOVED, NOT CORRECTED
---------------------------------------
`run_async` accepts `tools` and NEVER READS IT — it appears in the signature
and nowhere else in the body. The loop resolves its own catalog from
`context.tool_loader` / `context.tool_registry`, which is exactly why
`/api/chat/send` and every gateway adapter work.

Passing the "right" registry would have been a no-op that read like a fix and
taught the next reader that this route selects its own tools. It does not.

⚠ RULE 8 — WHY NOTHING CAUGHT THIS FOR TWO AND A HALF MONTHS
--------------------------------------------------------------
No test covered this route at all, and the one client (Beacon) calls
`/api/chat/send`, which dispatches through the WebSocket bridge on a different
path. A route with no test and no caller is not "working"; it is unobserved.
These tests are the first thing to exercise it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.skills.registry import SkillRegistry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _RecordingLoop:
    """Stands in for AgentLoop; records what the route passed it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_async(self, **kwargs):
        # SNAPSHOT the message list. `session.get_messages()` hands back a live
        # reference, so storing it verbatim makes every recorded call point at
        # the same growing list — two calls then look identical and a history
        # assertion compares a list against itself. (It did.)
        recorded = dict(kwargs)
        if "messages" in recorded:
            recorded["messages"] = list(recorded["messages"])
        self.calls.append(recorded)
        return SimpleNamespace(
            text="the answer",
            turns=1,
            messages=[],
            usage=SimpleNamespace(input_tokens=7, output_tokens=11),
        )


@pytest.fixture
def client_and_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    loop = _RecordingLoop()
    app = create_app(
        {"gateway": {"system_prompt": "sys"}},
        session_mgr=SessionManager(),
        # THE REAL SkillRegistry — the object whose missing method was the bug.
        # A mock would answer `list_schemas` happily and prove nothing.
        skill_registry=SkillRegistry(),
        agent_loop=loop,
    )
    return TestClient(app), loop


def test_the_route_reaches_the_model(client_and_loop):
    """THE defect: it never got this far."""
    client, loop = client_and_loop
    response = client.post("/api/chat", json={"session_id": "s1", "content": "hi"})

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["text"] == "the answer"
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 11}
    assert loop.calls, "run_async was never called"


def test_the_failure_was_not_a_generic_500(client_and_loop):
    """Pin the exact symptom, so a different 500 cannot pass as fixed."""
    client, _ = client_and_loop
    response = client.post("/api/chat", json={"session_id": "s1", "content": "hi"})
    assert "list_schemas" not in response.text
    assert "SkillRegistry" not in response.text


def test_the_route_does_not_pass_a_tools_argument(client_and_loop):
    """`run_async` never reads `tools`; passing one is a no-op that misleads.

    Asserting on what the route PASSED, not just that it returned 200 — a
    `tools=` that happened to evaluate would restore the 200 while leaving the
    misleading call shape in place.
    """
    client, loop = client_and_loop
    client.post("/api/chat", json={"session_id": "s1", "content": "hi"})

    assert loop.calls
    assert "tools" not in loop.calls[0], (
        f"the route still passes tools={loop.calls[0].get('tools')!r}; "
        f"run_async accepts that parameter and never reads it"
    )
    assert set(loop.calls[0]) == {"system_prompt", "messages"}


def test_the_skill_registry_still_has_no_list_schemas():
    """The premise, pinned.

    If `SkillRegistry` ever grows `list_schemas`, this test fails and whoever
    added it has to decide deliberately whether this route should use it —
    rather than the old call site quietly starting to "work" and re-entrenching
    the registry confusion.
    """
    assert not hasattr(SkillRegistry, "list_schemas"), (
        "SkillRegistry gained list_schemas — re-read #305 before assuming the "
        "old /api/chat call site was correct after all"
    )


def test_the_session_still_records_the_turn(client_and_loop):
    """The route's other job must survive the change.

    (First draft of this test had NO assertion at all — it called the route
    twice and ended. It passed, and measured nothing. Rule 8 applies to the
    tests written in this batch as much as to the ones inherited.)
    """
    client, loop = client_and_loop
    client.post("/api/chat", json={"session_id": "s1", "content": "hello there"})
    client.post("/api/chat", json={"session_id": "s1", "content": "again"})

    assert len(loop.calls) == 2
    first_history = loop.calls[0]["messages"]
    second_history = loop.calls[1]["messages"]
    assert len(second_history) > len(first_history), (
        f"the second turn did not see the first — the session is not recording "
        f"({len(first_history)} then {len(second_history)})"
    )
    assert [m.text for m in first_history] == ["hello there"]
    assert [m.text for m in second_history] == ["hello there", "again"]


def test_bad_input_still_400s(client_and_loop):
    client, _ = client_and_loop
    assert client.post("/api/chat", json={"content": "no session"}).status_code == 400
    assert client.post("/api/chat", json={"session_id": "s1"}).status_code == 400
