"""Model-backed suggestion generator for the Documents Editor (suggest mode).

ONE-SHOT and span-bounded: the document is shown to the model ONCE (the caller
caps its size) and the model returns redline edits as JSON. This is NOT an agent
loop and does NOT inject the document every turn — the bakeoff fixed-overhead
finding forbids whole-document-per-turn. This module only turns
(document, instruction) into raw ``{find, replace, reason}`` dicts; the service
(:meth:`prometheus.documents.service.DocumentsService.suggest`) validates each
``find`` for uniqueness and is the only thing that may ever apply them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

SUGGEST_SYSTEM = (
    "You are a precise document-editing assistant. The user gives you a "
    "document and an instruction. Propose a small set of concrete edits as a "
    'JSON object: {"edits": [{"find": "...", "replace": "...", "reason": "..."}]}\n'
    "RULES:\n"
    "- `find` MUST be text copied VERBATIM from the document, long enough to "
    "occur EXACTLY ONCE (include surrounding words to disambiguate).\n"
    "- `replace` is the replacement text; `reason` is one short sentence.\n"
    "- Change only what the instruction asks. Output JSON only, no prose."
)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of the first balanced JSON object in *text*.

    Tolerant of ``` fences and leading/trailing prose, because local models
    do not reliably honour "JSON only".
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except (ValueError, TypeError):
                        break
        start = text.find("{", start + 1)
    return {}


def parse_suggestions(text: str) -> list[dict]:
    """Turn raw model text into a list of ``{find, replace, reason}`` dicts."""
    obj = _extract_json_object(text)
    edits = obj.get("edits", [])
    if not isinstance(edits, list):
        return []
    out: list[dict] = []
    for e in edits:
        if isinstance(e, dict) and e.get("find"):
            out.append(
                {
                    "find": str(e.get("find", "")),
                    "replace": str(e.get("replace", "")),
                    "reason": str(e.get("reason", "")),
                }
            )
    return out


async def generate_suggestions(
    provider: Any,
    model: str,
    content: str,
    instruction: str,
    *,
    max_tokens: int = 2048,
) -> list[dict]:
    """One-shot model call → raw ``{find, replace, reason}`` dicts (no apply)."""
    from prometheus.engine.messages import ConversationMessage
    from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest

    user = f"INSTRUCTION:\n{instruction}\n\nDOCUMENT:\n{content}"
    req = ApiMessageRequest(
        model=model,
        messages=[ConversationMessage.from_user_text(user)],
        system_prompt=SUGGEST_SYSTEM,
        max_tokens=max_tokens,
        suppress_thinking=True,
    )
    text = ""
    async for event in provider.stream_message(req):
        if isinstance(event, ApiMessageCompleteEvent):
            text = event.message.text
    return parse_suggestions(text)
