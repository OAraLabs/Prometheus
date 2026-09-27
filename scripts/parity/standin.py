"""A replay stand-in for one side of a re-record.

A re-record takes the primary live and answers the other sides (the alt, the
hosted API) from the committed exchanges, so a recording costs no hosted call
and repeats the scripted replies those sides exist to give. The stand-in is the
harness's own replay-mode ModelServer with a stricter matcher: a completion is
answered only by a committed exchange of the same path and backend whose
normalized request is byte-identical, and anything else is refused with HTTP 400
(the daemon sees a failed call, and the recording fails its step). Nothing is
ever answered with a reply recorded for a different request.

One tolerance exists, for C2 (#593), which edits the system prompt and the tool
list of every request: on the sides named in ``tolerate``, a request is also
tried with C2's enumerated edits removed (``c2_reverted``), and served only when
that reverted form is byte-identical to a committed request. Any other
difference is still refused.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from parity.model_server import Exchange, ModelServer, ServerState
from parity.normalize import fingerprint

# ---------------------------------------------------------------------------
# C2 (#593): its edits to a model request, enumerated. Each constant is pinned
# to the live code by tests/test_parity_standin.py.
# ---------------------------------------------------------------------------

# 1. The skills section: an instruction now leads it, and its header is plain.
C2_SKILLS_INTRO = (
    "A skill is saved step-by-step instructions for one kind of task. When one fits "
    "the task in front of you, load it with the skill tool before you start, and follow it."
)
C2_CORE_HEADER = "## Core skills"
PRE_C2_CORE_HEADER = "## Core skills (always available)"
_SKILLS_HEAD = "# Available Skills\n\n"

# 2. The skill tool's description (tool arrays, and the prompt's tool list).
C2_SKILL_DESCRIPTION = (
    "Load a skill: saved step-by-step instructions for one kind of task (a procedure, "
    "a house convention, a known fix). Call it with the skill's name before starting a "
    "task that a skill covers, then follow what it returns. The system prompt names the "
    "core skills; find others with tool_search."
)
PRE_C2_SKILL_DESCRIPTION = "Read a builtin or user-defined skill by name."

# 3. `skill` joins always_loaded: the prompt's tool list gains its line
#    (adapter/formatter.py renders `- {name}: {description}`) and a local tool
#    array gains its entry, which is the tool count C2 shifts (11 -> 12).
C2_TOOL_LINE = f"- skill: {C2_SKILL_DESCRIPTION}"


def _revert_prompt(text: str) -> str:
    """Edits 1 and 3 in one system-prompt string, each at most once."""
    new_head = f"{_SKILLS_HEAD}{C2_SKILLS_INTRO}\n{C2_CORE_HEADER}\n"
    if new_head in text:
        text = text.replace(new_head, f"{_SKILLS_HEAD}{PRE_C2_CORE_HEADER}\n", 1)
    if f"{C2_TOOL_LINE}\n" in text:
        text = text.replace(f"{C2_TOOL_LINE}\n", "", 1)
    elif text.endswith(f"\n{C2_TOOL_LINE}"):
        text = text[: -len(f"\n{C2_TOOL_LINE}")]
    return text


def _tool_name_and_desc(tool: Any) -> tuple[Any, Any, dict | None]:
    """(name, description, the dict holding the description) for either wire shape."""
    if not isinstance(tool, dict):
        return None, None, None
    if isinstance(tool.get("function"), dict):          # OpenAI-style
        fn = tool["function"]
        return fn.get("name"), fn.get("description"), fn
    return tool.get("name"), tool.get("description"), tool   # Anthropic-style


def c2_reverted(norm: Any) -> list[Any]:
    """The normalized request with C2's enumerated edits removed, in the forms a
    pre-C2 request could have had: the skill tool's entry described the old way
    (a tool array that already carried it), or the entry absent (the tool count
    C2 shifts). Only the system prompt and the tool array are touched."""
    if not isinstance(norm, dict):
        return []
    base = copy.deepcopy(norm)
    msgs = base.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = _revert_prompt(m["content"])
    system = base.get("system")
    if isinstance(system, str):
        base["system"] = _revert_prompt(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = _revert_prompt(block["text"])
    tools = base.get("tools")
    if not isinstance(tools, list):
        return [base]
    described = copy.deepcopy(base)
    for tool in described["tools"]:
        name, desc, holder = _tool_name_and_desc(tool)
        if name == "skill" and desc == C2_SKILL_DESCRIPTION and holder is not None:
            holder["description"] = PRE_C2_SKILL_DESCRIPTION
    absent = copy.deepcopy(base)
    absent["tools"] = [t for t in absent["tools"]
                       if not (_tool_name_and_desc(t)[0] == "skill"
                               and _tool_name_and_desc(t)[1] == C2_SKILL_DESCRIPTION)]
    return [described, absent]


# ---------------------------------------------------------------------------
# The stand-in
# ---------------------------------------------------------------------------

@dataclass
class StandInState(ServerState):
    """Replay state whose matcher never answers a request it does not match."""

    # upstream label -> fn(normalized request) -> alternative forms to try
    tolerate: dict[str, Callable[[Any], Iterable[Any]]] = field(default_factory=dict)

    def match(self, path: str, norm: Any, upstream: str) -> tuple[int | None, bool]:
        exact = [fingerprint(norm)]
        fn = self.tolerate.get(upstream)
        tolerated = [fingerprint(v) for v in fn(norm)] if fn else []
        with self.lock:
            candidates = [i for i, ex in enumerate(self.recorded)
                          if ex.method == "POST" and ex.path == path
                          and ex.upstream == upstream and i not in self.consumed]
            for fps in (exact, tolerated):          # an exact match always wins
                for i in candidates:
                    if self.recorded[i].fp in fps:
                        self.consumed.add(i)
                        return i, True
        return None, False


class StandIn:
    """The committed exchanges of the given sides, served on local ports."""

    def __init__(self, exchanges: Iterable[Exchange], labels: list[str], *,
                 tolerate: dict[str, Callable[[Any], Iterable[Any]]] | None = None) -> None:
        self.labels = labels
        self.state = StandInState(mode="replay",
                                  recorded=[e for e in exchanges if e.upstream in labels],
                                  tolerate=dict(tolerate or {}))
        self._server = ModelServer(self.state)
        self.ports: dict[str, int] = {}

    def start(self) -> dict[str, int]:
        self.ports = self._server.start(self.labels)
        return self.ports

    def stop(self) -> None:
        self._server.stop()

    def url(self, label: str) -> str:
        return f"http://127.0.0.1:{self.ports[label]}"

    def summary(self) -> list[tuple[str, bool, int | None]]:
        """(upstream, answered, committed index) per completion it received."""
        return [(s.upstream, s.matched, s.recorded_index) for s in self.state.served]
