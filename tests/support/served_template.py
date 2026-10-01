"""The production server's served chat template, as a tool-calling verdict.

The ``tool_calls`` parity golden recorded the 4090's ``/props`` on 2026-09-25:
Qwen3.8-27B's template and llama-server's own analysis of it
(``chat_template_caps``). It renders a tools list and tool calls as Qwen XML.
Every first-run ladder rung ships a template of that kind — each GGUF's
``tokenizer.chat_template`` was read and classified native, ``qwen-xml`` on
2026-09-30 — so this recording stands in for the template a rung's server
would serve, without a server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PARITY = Path(__file__).resolve().parents[1] / "fixtures" / "parity"


def recorded_props() -> dict[str, Any]:
    trace = json.loads((PARITY / "tool_calls.trace.json").read_text())
    for exchange in trace["exchanges"]:
        if exchange.get("path") == "/props":
            props: dict[str, Any] = json.loads(exchange["body"])
            return props
    raise AssertionError("the tool_calls golden records no /props exchange")


def recorded_tool_template() -> Any:
    """``classify_template`` over the recorded /props — what ``detect_tool_template`` returns."""
    from prometheus.adapter.tier import classify_template

    props = recorded_props()
    return classify_template(chat_template=props["chat_template"], caps=props["chat_template_caps"])
