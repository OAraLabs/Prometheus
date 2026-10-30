#!/usr/bin/env python3
"""Render chat examples with the student model's own chat template.

The student's Hugging Face repo ships ``tokenizer.json`` and a Jinja chat
template (``chat_template.jinja``, or ``chat_template`` inside
``tokenizer_config.json``). Everything here renders with that template and
counts with that tokenizer, so a training row is the exact text the student
sees at inference, not an approximation of it.

Rendering follows the Hugging Face ``apply_chat_template`` conventions
(sandboxed Jinja, ``trim_blocks`` + ``lstrip_blocks``, ``tojson`` with
``ensure_ascii=False``) plus the one thing llama-server does before it renders:
an OpenAI-shape tool call carries ``arguments`` as a JSON *string*, and the
template iterates ``arguments | items``, so the string is parsed to an object
first. Whether the result matches llama-server byte for byte is what
:func:`apply_template` checks (``build.py spot-check``).

A training example is one request plus the assistant turn that answered it.
:func:`render_example` renders the request with the generation prompt
(``add_generation_prompt=True``, thinking off: the same text llama-server
builds for a live request), renders the whole conversation, checks the first
is a prefix of the second, and marks the rest (the assistant's text and tool
calls through ``<|im_end|>``) as the only span that carries loss.

Tokenization is per segment: ``ids = enc(prompt) + enc(completion) +
enc(tail)``. That is how the tokens exist at inference (the server tokenizes
the prompt; the model produces the completion), so the loss span is exact by
construction. ``boundary_stable`` records whether tokenizing the joined text
in one call gives the same ids; a trainer must use the segment recipe either
way.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

#: chat_template_kwargs LlamaCppProvider sends when thinking is suppressed
#: (the live default): both keys, so any template family reads its own.
THINKING_OFF = {"thinking": False, "enable_thinking": False}
END_OF_TURN = "<|im_end|>"


class RenderError(ValueError):
    """An example the template cannot render the way training needs."""


class Encoder(Protocol):
    def encode_ids(self, text: str) -> list[int]: ...


class HFTokenizer:
    """``tokenizers.Tokenizer`` behind the :class:`Encoder` protocol.

    Special tokens written in the text (``<|im_start|>``...) encode as their
    single ids, and nothing is added around the text: the template already
    wrote every special token the model sees.
    """

    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        self.path = Path(path)
        self._tok = Tokenizer.from_file(str(self.path))

    def encode_ids(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)


def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    # The transformers definition of the filter (chat_template_utils).
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def _raise(message: str) -> None:
    raise RenderError(f"template refused the messages: {message}")


def load_template(source: str) -> Any:
    """Compile a chat template in the sandbox transformers uses."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(source)


def read_template_source(model_dir: str | Path) -> str:
    """``chat_template.jinja`` when the repo ships one, else the string in
    ``tokenizer_config.json`` (older repos)."""
    d = Path(model_dir).expanduser()
    jinja = d / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    tpl = cfg.get("chat_template")
    if isinstance(tpl, list):  # named templates: the default one
        tpl = next((t.get("template") for t in tpl if t.get("name") == "default"), None)
    if not isinstance(tpl, str) or not tpl:
        raise RenderError(f"no chat template in {d}")
    return tpl


@dataclass
class Student:
    """The student's template and tokenizer, loaded once."""

    template: Any
    encoder: Encoder
    source: str = ""

    @classmethod
    def load(cls, model_dir: str | Path) -> "Student":
        d = Path(model_dir).expanduser()
        src = read_template_source(d)
        return cls(template=load_template(src), encoder=HFTokenizer(d / "tokenizer.json"),
                   source=str(d))

    def count(self, text: str) -> int:
        return len(self.encoder.encode_ids(text))


def prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages as llama-server hands them to the template.

    Tool-call ``arguments`` that are a JSON string become the object they
    encode (the template iterates them with ``| items``). A string that does
    not parse is left as is, and the template then refuses it, which is the
    honest outcome for a call whose arguments were never valid JSON.
    """
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args.strip() else {}
                except ValueError:
                    pass
    return out


def render(template: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
           *, add_generation_prompt: bool, template_kwargs: dict[str, Any] | None = None) -> str:
    """One template render. Empty ``tools`` is passed as None, as llama-server
    does for a request that offers none."""
    kwargs = dict(THINKING_OFF if template_kwargs is None else template_kwargs)
    try:
        return template.render(messages=prepare_messages(messages), tools=tools or None,
                               add_generation_prompt=add_generation_prompt, **kwargs)
    except RenderError:
        raise
    except Exception as exc:  # jinja errors carry template text only
        raise RenderError(f"{type(exc).__name__} while rendering") from exc


@dataclass
class Rendered:
    """One rendered training example."""

    text: str
    prompt_chars: int
    loss_char_spans: list[tuple[int, int]]
    loss_token_spans: list[tuple[int, int]]
    n_tokens: int
    prompt_tokens: int
    boundary_stable: bool
    segment_tokens: list[int] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "loss_spans_char": [list(s) for s in self.loss_char_spans],
            "loss_spans_tok": [list(s) for s in self.loss_token_spans],
            "n_tokens": self.n_tokens,
            "prompt_tokens": self.prompt_tokens,
            "boundary_stable": self.boundary_stable,
            "tokenize": "segments",
        }


def render_prompt(student: Student, messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]] | None,
                  template_kwargs: dict[str, Any] | None = None) -> str:
    """The text llama-server builds for this request: the generation prompt
    included, thinking off unless *template_kwargs* says otherwise."""
    return render(student.template, messages, tools, add_generation_prompt=True,
                  template_kwargs=template_kwargs)


def render_example(student: Student, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None,
                   template_kwargs: dict[str, Any] | None = None,
                   *, check_boundary: bool = True) -> Rendered:
    """Render request + final assistant turn, with the assistant-only loss span.

    ``messages[-1]`` is the assistant turn being learned; everything before it
    is the request. Only that last turn carries loss: earlier assistant turns
    are context the student did not produce at this step (and in a long
    conversation some were written by other models).
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise RenderError("the last message must be the assistant turn to learn")
    prompt = render(student.template, messages[:-1], tools, add_generation_prompt=True,
                    template_kwargs=template_kwargs)
    full = render(student.template, messages, tools, add_generation_prompt=False,
                  template_kwargs=template_kwargs)
    if not full.startswith(prompt):
        raise RenderError("the request's render is not a prefix of the full render")
    completion = full[len(prompt):]
    end = completion.rfind(END_OF_TURN)
    if end < 0:
        raise RenderError("the assistant turn has no end-of-turn token")
    end += len(END_OF_TURN)
    seg_text = (prompt, completion[:end], completion[end:])
    seg_ids = [student.encoder.encode_ids(s) if s else [] for s in seg_text]
    p, c, t = (len(x) for x in seg_ids)
    stable = True
    if check_boundary:
        stable = student.encoder.encode_ids(full) == seg_ids[0] + seg_ids[1] + seg_ids[2]
    return Rendered(
        text=full,
        prompt_chars=len(prompt),
        loss_char_spans=[(len(prompt), len(prompt) + end)],
        loss_token_spans=[(p, p + c)],
        n_tokens=p + c + t,
        prompt_tokens=p,
        boundary_stable=stable,
        segment_tokens=[p, c, t],
    )


# ---------------------------------------------------------------------------
# Fitting a long example into the training window
# ---------------------------------------------------------------------------


def last_query_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the last user message that is a real query (the template's
    ``last_query_index``: not a bare ``<tool_response>`` wrapper)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "user":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        stripped = text.strip()
        if not (stripped.startswith("<tool_response>") and stripped.endswith("</tool_response>")):
            return i
    return None


def drop_points(messages: list[dict[str, Any]]) -> list[int]:
    """How many leading non-system messages may be dropped, smallest first.

    A cut always lands so the first kept message is a user query (never an
    orphaned tool result or an assistant turn whose request is gone), and
    never past the last user query: the request the example answers stays.
    """
    start = 1 if messages and messages[0].get("role") == "system" else 0
    limit = last_query_index(messages)
    if limit is None:
        return [0]
    points = [0]
    for i in range(start + 1, limit + 1):
        if messages[i].get("role") == "user":
            points.append(i - start)
    return points


def turn_steps(messages: list[dict[str, Any]]) -> list[int]:
    """Indexes of the assistant steps between the last user query and the
    final turn (a long agentic turn's earlier calls, each followed by its tool
    results)."""
    q = last_query_index(messages)
    if q is None:
        return []
    return [i for i in range(q + 1, len(messages) - 1) if messages[i].get("role") == "assistant"]


@dataclass
class Fit:
    messages: list[dict[str, Any]]
    dropped: int  # messages left out, in all
    rendered: Rendered
    dropped_steps: int = 0  # of them, earlier steps of the current turn


def fit_to_budget(student: Student, messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]] | None, budget: int,
                  template_kwargs: dict[str, Any] | None = None,
                  *, first: Rendered | None = None, split_turns: bool = True) -> Fit | None:
    """Drop the oldest context until the example fits *budget* tokens.

    The system prompt and the tools are always kept (they are the request's
    head, rendered first), and so are the last user query and the turn being
    learned. Cuts come in order: earlier conversation first (each cut starts
    on a user query), then, with *split_turns*, the oldest steps of the
    current turn (an assistant call with its tool results), which turns one
    long agentic turn into a window over its latest steps. Binary search over
    the cuts, so a long history costs a few renders, not one per message.
    None when even the shortest request is over budget.
    """
    start = 1 if messages and messages[0].get("role") == "system" else 0
    head = messages[:start]

    def at(kept: list[dict[str, Any]]) -> Rendered:
        return render_example(student, kept, tools, template_kwargs, check_boundary=False)

    whole = first or at(messages)
    if whole.n_tokens <= budget:
        return Fit(messages, 0, whole)
    cuts: list[tuple[list[dict[str, Any]], int, int]] = [
        (head + messages[start + k:], k, 0) for k in drop_points(messages)[1:]]
    q = last_query_index(messages)
    steps = turn_steps(messages) if split_turns else []
    if q is not None:
        for n in range(1, len(steps) + 1):
            nxt = steps[n] if n < len(steps) else len(messages) - 1
            cuts.append((head + [messages[q]] + messages[nxt:], (q - start) + (nxt - q - 1), n))
    lo, hi = 0, len(cuts) - 1
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if at(cuts[mid][0]).n_tokens <= budget:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    if best is None:
        return None
    kept, dropped, dropped_steps = cuts[best]
    # Re-render once with the boundary check, so the stored flag is real.
    return Fit(kept, dropped, render_example(student, kept, tools, template_kwargs), dropped_steps)


# ---------------------------------------------------------------------------
# llama-server spot check
# ---------------------------------------------------------------------------


def apply_template_body(payload: dict[str, Any], *, add_generation_prompt: bool = True
                        ) -> dict[str, Any]:
    """The ``POST /apply-template`` body for a chat-completions payload.

    llama-server parses it with the same code as ``/v1/chat/completions``
    (messages, tools, tool_choice, chat_template_kwargs) and returns the
    rendered prompt instead of generating.
    """
    body: dict[str, Any] = {"messages": payload["messages"],
                            "add_generation_prompt": add_generation_prompt}
    for key in ("tools", "tool_choice", "chat_template_kwargs"):
        if payload.get(key) is not None:
            body[key] = payload[key]
    return body


def apply_template(base_url: str, payload: dict[str, Any], *,
                   post: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
                   timeout: float = 120.0, add_generation_prompt: bool = True) -> str:
    """The prompt llama-server renders for *payload* (``POST /apply-template``).

    Note llama-server takes a trailing assistant message as a prefill (it
    renders the rest with the generation prompt and appends that message's
    text), so a conversation is compared in full only when it ends on a user
    or tool message."""
    url = base_url.rstrip("/") + "/apply-template"
    body = apply_template_body(payload, add_generation_prompt=add_generation_prompt)
    if post is None:
        import httpx

        def post(u: str, b: dict[str, Any]) -> dict[str, Any]:
            resp = httpx.post(u, json=b, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

    data = post(url, body)
    prompt = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(prompt, str):
        raise RenderError("apply-template reply has no string 'prompt'")
    return prompt


def first_mismatch(a: str, b: str) -> int | None:
    """Offset of the first differing character, or None when equal."""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
