"""Tests for scripts/distill/build.py and render.py, the Phase 3 dataset builder.

Synthetic fixtures only: a compact Qwen-style chat template, a regex tokenizer
that treats the chat special tokens as single tokens, a hand-made request pack
and made-up traces with neutral names. No network, no GPU, no real traces.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import os
import re
import sqlite3
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("distill_build", "build.py")
R = build.R
score = _load("distill_score", "score.py")

# ── fixtures ───────────────────────────────────────────────────────────────

# A compact template with the Qwen3.5 shapes that matter here: tools block in
# the system turn, <think></think> on assistant turns after the last query,
# XML tool calls iterating ``arguments | items``, tool results as user turns,
# and a thinking-off generation prompt. PyTemplate below renders the same text
# in plain Python, so the builder tests run where jinja2 is not installed (the
# dev environment); test_py_template_matches_the_jinja_template keeps the two
# identical wherever jinja2 is available.
TEMPLATE = r"""
{%- if tools %}
{{- '<|im_start|>system\n# Tools\n<tools>' }}
{%- for tool in tools %}{{- '\n' + (tool | tojson) }}{%- endfor %}
{{- '\n</tools>' }}
{%- if messages[0].role == 'system' %}{{- '\n\n' + ((messages[0].content or '') | trim) }}{%- endif %}
{{- '<|im_end|>\n' }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\n' + ((messages[0].content or '') | trim) + '<|im_end|>\n' }}
{%- endif %}
{%- set ns = namespace(last=-1) %}
{%- for m in messages %}{%- if m.role == 'user' %}{%- set ns.last = loop.index0 %}{%- endif %}{%- endfor %}
{%- if ns.last < 0 %}{{- raise_exception('No user query found in messages.') }}{%- endif %}
{%- for m in messages %}
{%- if m.role == 'user' %}
{{- '<|im_start|>user\n' + ((m.content or '') | trim) + '<|im_end|>\n' }}
{%- elif m.role == 'assistant' %}
{{- '<|im_start|>assistant\n' }}
{%- if loop.index0 > ns.last %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}
{%- set c = (m.content or '') | trim %}
{{- c }}
{%- for tc in (m.tool_calls or []) %}
{%- if loop.first and not c %}{{- '<tool_call>\n' }}{%- else %}{{- '\n\n<tool_call>\n' }}{%- endif %}
{{- '<function=' + tc.function.name + '>\n' }}
{%- for k, v in tc.function.arguments | items %}
{{- '<parameter=' + k + '>\n' + (v if v is string else (v | tojson)) + '\n</parameter>\n' }}
{%- endfor %}
{{- '</function>\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- elif m.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + (m.content or '') + '\n</tool_response><|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}
{%- else %}{{- '<think>\n' }}{%- endif %}
{%- endif %}
"""


class PyTemplate:
    """TEMPLATE, rendered in plain Python (same ``render`` keywords as jinja)."""

    def render(self, *, messages: list[dict[str, Any]], tools: Any,
               add_generation_prompt: bool, enable_thinking: Any = None, **_: Any) -> str:
        out: list[str] = []
        first = messages[0]
        if tools:
            s = "<|im_start|>system\n# Tools\n<tools>" + "".join(
                "\n" + json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"
            if first["role"] == "system":
                s += "\n\n" + (first.get("content") or "").strip()
            out.append(s + "<|im_end|>\n")
        elif first["role"] == "system":
            out.append("<|im_start|>system\n" + (first.get("content") or "").strip()
                       + "<|im_end|>\n")
        last = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        if last < 0:
            raise ValueError("No user query found in messages.")
        for i, m in enumerate(messages):
            if m["role"] == "user":
                out.append("<|im_start|>user\n" + (m.get("content") or "").strip()
                           + "<|im_end|>\n")
            elif m["role"] == "assistant":
                out.append("<|im_start|>assistant\n")
                if i > last:
                    out.append("<think>\n\n</think>\n\n")
                c = (m.get("content") or "").strip()
                out.append(c)
                for j, tc in enumerate(m.get("tool_calls") or []):
                    out.append("<tool_call>\n" if j == 0 and not c else "\n\n<tool_call>\n")
                    fn = tc["function"]
                    out.append("<function=" + fn["name"] + ">\n")
                    for k, v in fn["arguments"].items():  # a str here raises, as in jinja
                        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                        out.append("<parameter=" + k + ">\n" + val + "\n</parameter>\n")
                    out.append("</function>\n</tool_call>")
                out.append("<|im_end|>\n")
            elif m["role"] == "tool":
                out.append("<|im_start|>user\n<tool_response>\n" + (m.get("content") or "")
                           + "\n</tool_response><|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
            out.append("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
        return "".join(out)


class FakeEncoder:
    """Special tokens are single ids; other text splits into words, runs of
    whitespace and single symbols (so a boundary inside whitespace can merge)."""

    SPECIAL = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")
    PIECE = re.compile(r"\w+|\s+|[^\w\s]")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.calls = 0

    def encode_ids(self, text: str) -> list[int]:
        self.calls += 1
        ids: list[int] = []
        for part in self.SPECIAL.split(text):
            if not part:
                continue
            pieces = [part] if self.SPECIAL.fullmatch(part) else self.PIECE.findall(part)
            ids += [self.vocab.setdefault(p, len(self.vocab)) for p in pieces]
        return ids


def make_student() -> Any:
    return R.Student(template=PyTemplate(), encoder=FakeEncoder())


def schema(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": required}}


TOOLS = [
    schema("bash", "Run a shell command.", {"command": {"type": "string"}}, ["command"]),
    schema("read_file", "Read a file.", {"path": {"type": "string"}}, ["path"]),
    schema("web_search", "Search the web.", {"query": {"type": "string"}}, ["query"]),
    schema("memory", "Store a fact.", {"fact": {"type": "string"}}, ["fact"]),
]


def make_pack() -> dict[str, Any]:
    return {
        "schema": build.PACK_SCHEMA, "prometheus_commit": "test",
        "date_sentinel": build.DATE_SENTINEL,
        "system_prompt": f"You are a test agent.\nToday's date: {build.DATE_SENTINEL}",
        "tools": {"full": TOOLS, "stubs": sorted(TOOLS, key=lambda t: t["name"])},
        "stub_names": [], "runtime_catalog": "stubs", "approx": list(build.PACK_APPROX),
    }


def openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOLS]


def call(name: str, args: dict[str, Any], cid: str = "call_1") -> dict[str, Any]:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
         "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
         "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]


def word(i: int) -> str:
    return WORDS[i % len(WORDS)] + ("" if i < len(WORDS) else WORDS[i // len(WORDS) % len(WORDS)])


def trace(task: str, tool: str, args: dict[str, Any], *, model: str = "qwen3.8-max",
          ts: float = 1_790_000_000.0, session: str = "beacon:s1", prose: str = "",
          ctx_extra: list[dict[str, Any]] | None = None, **meta: Any) -> dict[str, Any]:
    target: dict[str, Any] = {"role": "assistant", "content": prose,
                              "tool_calls": [call(tool, args)]}
    ctx = [{"role": "user", "content": f"opening note {task}"},
           {"role": "assistant", "content": f"sure, about {task}"},
           *(ctx_extra or []),
           {"role": "user", "content": f"please handle the {task} job"}]
    return {"messages": [*ctx, target],
            "tools": [{"type": "function", "function": TOOLS[0]}],
            "_meta": {"model": model, "tool_name": tool, "timestamp": ts,
                      "session_id": session, **meta}}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def make_builder(tmp_path: Path, *, spool: bool = True, lcm: Any = None,
                 verdicts: dict[str, str] | None = None, index: Any = None,
                 verdict_checks: dict[str, Any] | None = None, **settings: Any) -> Any:
    s = build.Settings(train_range=(1, 100000), eval_min=1, min_presence=0, **settings)
    return build.Builder(student=make_student(), pack=make_pack(), catalog="stubs", settings=s,
                         index=index or {}, verdicts=verdicts or {}, lcm=lcm,
                         spool=(tmp_path / "spool.jsonl") if spool else None,
                         verdict_checks=verdict_checks)


def feed(b: Any, path: Path, kind: str = "golden") -> None:
    for t in build.iter_traces([path], kind, b.stats):
        b.add(t)


# ── render + masks ─────────────────────────────────────────────────────────


def test_render_example_masks_only_the_assistant_turn() -> None:
    st = make_student()
    msgs = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": "Listing.",
             "tool_calls": [call("bash", {"command": "ls -la", "n": 2})]}]
    r = R.render_example(st, msgs, openai_tools())
    prompt = R.render_prompt(st, msgs[:-1], openai_tools())
    assert r.text.startswith(prompt) and r.prompt_chars == len(prompt)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    (a, b), = r.loss_char_spans
    learned = r.text[a:b]
    assert learned.startswith("Listing.") and learned.endswith("<|im_end|>")
    # the JSON-string arguments were rendered as parameters, not as one string
    assert "<parameter=command>\nls -la\n</parameter>" in learned
    assert "<parameter=n>\n2\n</parameter>" in learned
    assert r.text[b:] == "\n"
    # token spans follow the segment recipe exactly
    enc = st.encoder
    p, c, t = (enc.encode_ids(x) for x in (r.text[:a], r.text[a:b], r.text[b:]))
    assert r.loss_token_spans == [(len(p), len(p) + len(c))]
    assert r.n_tokens == len(p) + len(c) + len(t)
    assert r.prompt_tokens == len(p) and r.boundary_stable


def test_render_example_refuses_a_request_without_the_turn_to_learn() -> None:
    with pytest.raises(R.RenderError):
        R.render_example(make_student(), [{"role": "user", "content": "hi"}], None)


def test_chat_mode_renders_without_a_tools_block() -> None:
    st = make_student()
    r = R.render_example(st, [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "Hello."}], None)
    assert "<tools>" not in r.text and r.text[slice(*r.loss_char_spans[0])] == "Hello.<|im_end|>"


def test_py_template_matches_the_jinja_template() -> None:
    pytest.importorskip("jinja2")
    jinja = R.load_template(TEMPLATE)
    convs = [long_conversation(2), [{"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "Hello ü."}]]
    for msgs in convs:
        for tools in (openai_tools(), None):
            for gen in (True, False):
                body = msgs[:-1] if gen else msgs
                kw = dict(messages=R.prepare_messages(body), tools=tools,
                          add_generation_prompt=gen, **R.THINKING_OFF)
                assert jinja.render(**kw) == PyTemplate().render(**kw)


def test_unparseable_arguments_are_refused_not_rendered_as_text() -> None:
    bad = {"role": "assistant", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{oops"}}]}
    with pytest.raises(R.RenderError):
        R.render_example(make_student(), [{"role": "user", "content": "go"}, bad], openai_tools())


def test_real_student_template_keeps_the_prefix_property_when_present() -> None:
    pytest.importorskip("jinja2")
    d = Path(__file__).resolve().parent.parent / "data" / "distill" / "models" / "qwen3.5-9b-tokenizer"
    if not (d / "chat_template.jinja").is_file():
        pytest.skip("student template not fetched (gitignored data/distill/models)")
    st = R.Student(template=R.load_template(R.read_template_source(d)), encoder=FakeEncoder())
    msgs = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": "", "tool_calls": [call("bash", {"command": "ls"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
            {"role": "assistant", "content": "Reading it.",
             "tool_calls": [call("read_file", {"path": "a.py"})]}]
    r = R.render_example(st, msgs, openai_tools())
    learned = r.text[slice(*r.loss_char_spans[0])]
    assert learned.startswith("Reading it.\n\n<tool_call>\n<function=read_file>")
    assert learned.endswith("</tool_call><|im_end|>")
    assert r.text.index("<tools>") < r.text.index("Be brief.")


# ── truncation ─────────────────────────────────────────────────────────────


def long_conversation(n: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "System rules stay."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {word(i)} " + "filler " * 30})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            call("bash", {"command": f"echo {word(i)}"}, f"c{i}")]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "output " * 20})
        msgs.append({"role": "assistant", "content": f"done with {word(i)}"})
    msgs.append({"role": "user", "content": "now the final request"})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        call("read_file", {"path": "final.py"}, "cz")]})
    return msgs


def test_truncation_drops_oldest_context_and_keeps_system_tools_and_query() -> None:
    st = make_student()
    msgs = long_conversation(12)
    whole = R.render_example(st, msgs, openai_tools())
    minimal = R.render_example(st, [msgs[0], *msgs[-2:]], openai_tools())
    budget = minimal.n_tokens + (whole.n_tokens - minimal.n_tokens) // 3
    fit = R.fit_to_budget(st, msgs, openai_tools(), budget)
    assert fit is not None and fit.dropped > 0
    assert fit.rendered.n_tokens <= budget
    kept = fit.messages
    assert kept[0] == msgs[0] and kept[-2:] == msgs[-2:]
    assert kept[1]["role"] == "user"  # never starts on an orphaned tool result
    text = fit.rendered.text
    assert text.startswith("<|im_start|>system\n# Tools\n<tools>")
    assert "System rules stay." in text and '"name": "read_file"' in text
    # the smallest cut that fits: one cut point less would be over budget
    points = R.drop_points(msgs)
    i = points.index(fit.dropped)
    longer = [msgs[0], *msgs[1 + points[i - 1]:]]
    assert R.render_example(st, longer, openai_tools()).n_tokens > budget


def test_truncation_gives_up_when_the_request_itself_is_too_long() -> None:
    st = make_student()
    msgs = long_conversation(3)
    minimal = R.render_example(st, [msgs[0], *msgs[-2:]], openai_tools())
    assert R.fit_to_budget(st, msgs, openai_tools(), minimal.n_tokens - 1) is None


def test_a_long_turn_becomes_a_window_over_its_latest_steps() -> None:
    st = make_student()
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "Rules."},
                                  {"role": "user", "content": "earlier chat"},
                                  {"role": "assistant", "content": "earlier reply"},
                                  {"role": "user", "content": "build the thing"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            call("bash", {"command": f"step {word(i)}"}, f"s{i}")]})
        msgs.append({"role": "tool", "tool_call_id": f"s{i}", "content": "log line " * 80})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        call("read_file", {"path": "out.txt"}, "last")]})
    keep_last = [msgs[0], msgs[3], *msgs[-3:]]
    budget = R.render_example(st, keep_last, openai_tools()).n_tokens + 5
    fit = R.fit_to_budget(st, msgs, openai_tools(), budget)
    assert fit is not None and fit.messages == keep_last
    assert fit.dropped_steps == 3 and fit.dropped == len(msgs) - len(keep_last)
    assert fit.rendered.n_tokens <= budget
    assert R.fit_to_budget(st, msgs, openai_tools(), budget, split_turns=False) is None


def test_drop_points_land_on_user_queries_only() -> None:
    msgs = long_conversation(4)
    for k in R.drop_points(msgs):
        assert msgs[1 + k]["role"] == "user"
    assert max(R.drop_points(msgs)) == R.last_query_index(msgs) - 1


def test_head_cache_is_exact() -> None:
    st = make_student()
    text = R.render_example(st, long_conversation(2), openai_tools()).text
    plain = FakeEncoder()
    cached = build.HeadCachedEncoder(FakeEncoder())
    plain.vocab = cached.inner.vocab  # one shared vocabulary
    assert cached.encode_ids(text) == plain.encode_ids(text)
    calls = cached.inner.calls
    assert cached.encode_ids(text) == plain.encode_ids(text)
    assert cached.inner.calls == calls + 1  # the head came from the cache


# ── category caps ──────────────────────────────────────────────────────────


def ex(i: int, category: str, task: str, prefix: str | None = None, **kw: Any) -> Any:
    return build.Example(
        id=f"e{i}", trace_id=f"f:{i}", kind="golden", source="max-golden", category=category,
        task_id=task, h_exact=f"x{i}", h_prompt_call=f"pc{i}", label="tool-success",
        flags={"chat_mode": False, "error_recovery": False, "compacted": False},
        context_source="stored", n_tokens_original=10, n_tokens=10, dropped_messages=0,
        fits=True, longctx=False, boundary_stable=True, prefix=prefix or f"p-{task}",
        expect_tool="bash", **kw)


def test_category_cap_keeps_the_largest_set_under_the_cap() -> None:
    rows = ([ex(i, "coding", f"c{i % 7}") for i in range(86)]
            + [ex(100 + i, "daemon", f"d{i}") for i in range(8)]
            + [ex(200 + i, "other", f"o{i}") for i in range(5)]
            + [ex(300, "memory", "m0")])
    kept, report = build.cap_categories(rows, 0.35, "seed")
    counts = {c: r["kept"] for c, r in report.items()}
    assert counts == {"coding": 7, "daemon": 7, "memory": 1, "other": 5}
    assert all(k / len(kept) <= 0.35 for k in counts.values())
    assert report["coding"]["dropped"] == 79 and report["daemon"]["dropped"] == 1
    # round-robin: the 7 coding rows kept come from 7 different tasks
    assert len({e.task_id for e in kept if e.category == "coding"}) == 7


def test_category_cap_is_a_no_op_when_nothing_dominates() -> None:
    rows = [ex(i, c, f"t{i}") for i, c in enumerate(["a", "b", "c"] * 4)]
    kept, report = build.cap_categories(rows, 0.35, "seed")
    assert len(kept) == 12 and all(r["dropped"] == 0 for r in report.values())


# ── class caps (teacher_gen writes every verified row; build caps the dataset) ──


def _class_rows() -> list[Any]:
    # 10 reported_blocked rows (5 tasks), 10 no-tool answers (one also blocked), 30 plain rows
    rows = [ex(i, "other", f"b{i % 5}", classes=["reported_blocked"]) for i in range(10)]
    rows += [ex(100 + i, "memory", f"n{i}", classes=["no_tool_answer"]) for i in range(9)]
    rows.append(ex(110, "memory", "nb", classes=["no_tool_answer", "reported_blocked"]))
    rows += [ex(200 + i, ["a", "b", "c"][i % 3], f"p{i}") for i in range(30)]
    return rows


def test_class_caps_keep_each_class_under_its_share_of_train() -> None:
    rows = _class_rows()
    caps = {"reported_blocked": 0.15, "no_tool_answer": 0.20}
    kept, report = build.cap_classes(rows, caps, "seed")
    n = len(kept)
    for c, cap in caps.items():
        assert sum(1 for e in kept if c in e.classes) <= cap * n
        assert report[c]["before"] - report[c]["dropped"] == report[c]["kept"]
    assert report["reported_blocked"]["dropped"] > 0 and report["no_tool_answer"]["dropped"] > 0
    # plain rows are never dropped by a class cap
    assert sum(1 for e in kept if not e.classes) == 30
    # round-robin: the blocked rows kept come from as many tasks as possible
    blocked = [e for e in kept if "reported_blocked" in e.classes]
    assert len({e.task_id for e in blocked}) == min(len(blocked), 6)
    # deterministic by seed; another seed may keep other rows, never another count
    again, _ = build.cap_classes(_class_rows(), caps, "seed")
    assert [e.id for e in again] == [e.id for e in kept]
    other, _ = build.cap_classes(_class_rows(), caps, "other-seed")
    assert len(other) == len(kept)


def test_class_and_category_caps_hold_together_and_report_what_each_dropped() -> None:
    # coding dominates: the category cap's drops raise the classes' shares, so the
    # class caps run again until neither drops a row
    rows = ([ex(i, "coding", f"c{i}") for i in range(40)]
            + [ex(100 + i, "other", f"o{i}", classes=["reported_blocked"]) for i in range(8)]
            + [ex(200 + i, "memory", f"m{i}", classes=["no_tool_answer"]) for i in range(10)]
            + [ex(300 + i, "daemon", f"d{i}") for i in range(12)])
    caps = {"reported_blocked": 0.15, "no_tool_answer": 0.20}
    kept, cats, classes = build.cap_train(rows, 0.35, caps, "seed")
    n = len(kept)
    counts = collections.Counter(e.category for e in kept)
    assert max(counts.values()) <= 0.35 * n
    for c, cap in caps.items():
        k = sum(1 for e in kept if c in e.classes)
        assert k <= cap * n and classes[c]["kept"] == k and classes[c]["share"] == round(k / n, 4)
        assert classes[c]["dropped"] == classes[c]["dropped_by_class_cap"] + classes[c]["dropped_by_category_cap"]
    assert cats["coding"]["dropped"] > 0 and cats["coding"]["kept"] == counts["coding"]
    assert classes["rows_dropped"]["class_cap"] + classes["rows_dropped"]["category_cap"] == len(rows) - n
    assert classes["rows_dropped"]["class_cap"] > 0
    # without classes it is the category cap alone
    plain = [ex(i, "coding", f"c{i % 7}") for i in range(86)] + [ex(100 + i, "daemon", f"d{i}") for i in range(8)]
    alone, _ = build.cap_categories(plain, 0.35, "seed")
    both, _, _ = build.cap_train(plain, 0.35, caps, "seed")
    assert [e.id for e in both] == [e.id for e in alone]


def test_val_keeps_the_share_of_each_class_train_kept() -> None:
    val = ([ex(i, "other", f"b{i}", classes=["reported_blocked"]) for i in range(8)]
           + [ex(100 + i, "other", f"p{i}") for i in range(8)])
    kept, report = build.apply_class_ratios(val, {"reported_blocked": 0.25, "no_tool_answer": 1.0}, "seed")
    assert report["reported_blocked"] == {"before": 8, "kept": 2, "dropped": 6}
    assert report["no_tool_answer"] == {"before": 0, "kept": 0, "dropped": 0}
    assert len(kept) == 10 and sum(1 for e in kept if not e.classes) == 8


def test_val_does_not_count_category_cap_drops_twice_for_a_class() -> None:
    # every 5th coding row is reported_blocked and coding dominates: in train the
    # category cap drops blocked rows along with coding, never the class cap. Val's
    # category ratio already drops those; the class ratio must not drop them again.
    def rows(off: int) -> list[Any]:
        r = [ex(off + i, "coding", f"c{i}", classes=["reported_blocked"] if i % 5 == 0 else [])
             for i in range(100)]
        return r + [ex(off + 100 + i, ["a", "b", "c"][i % 3], f"p{i}") for i in range(50)]

    caps = {"reported_blocked": 0.15, "no_tool_answer": 0.20}
    train, cats, classes = build.cap_train(rows(0), 0.35, caps, "seed")
    blocked = classes["reported_blocked"]
    assert blocked["dropped_by_category_cap"] > 0 and blocked["dropped_by_class_cap"] == 0
    ratios = build.class_ratios(classes, caps)
    assert ratios == {"reported_blocked": 1.0}
    val, _ = build.apply_ratios(rows(1000), {c: 1 - r["dropped"] / r["before"] for c, r in cats.items()}, "seed")
    val, report = build.apply_class_ratios(val, ratios, "seed")
    assert report["reported_blocked"]["dropped"] == 0
    share = sum(1 for e in val if "reported_blocked" in e.classes) / len(val)
    assert abs(share - blocked["share"]) < 0.01


def test_builder_caps_reported_blocked_and_no_tool_rows_from_their_meta_tags(tmp_path: Path) -> None:
    rows = [gym("plain" + word(t), "bash", {"command": f"make p{t}"}, category=["coding", "other", "memory/wiki"][t % 3])
            for t in range(12)]
    rows += [gym("blocked" + word(t), "bash", {"command": f"ls b{t}"}, category="other",
                 judge_outcome="reported_blocked") for t in range(6)]
    for t in range(6):
        r = gym("prose" + word(t), "bash", {}, category="sentinel/daemon", no_tool_answer=True, judge_outcome="done")
        r["messages"][-1] = {"role": "assistant", "content": f"No tool needed: answer {word(t)}."}
        r["_meta"].pop("tool_name")
        rows.append(r)
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "gym.jsonl", rows), "extra")
    tagged = collections.Counter(c for e in b.examples for c in e.classes)
    assert tagged == {"reported_blocked": 6, "no_tool_answer": 6}
    info = b.split()
    gates = b.gates()
    train = b.final["train"]
    for c, cap in build.CLASS_CAPS.items():
        assert sum(1 for e in train if c in e.classes) <= cap * len(train)
    assert gates["class_caps"]["pass"] and gates["class_caps"]["cap"] == build.CLASS_CAPS
    dropped = info["train_class_caps"]
    assert dropped["reported_blocked"]["dropped"] > 0 and dropped["no_tool_answer"]["dropped"] > 0
    assert dropped["rows_dropped"]["class_cap"] == len(b.examples) - len(train)
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    assert report["caps"]["class_caps"]["train"]["reported_blocked"]["cap"] == 0.15
    assert report["settings"]["class_caps"] == build.CLASS_CAPS
    # deterministic: the same inputs and seed keep the same rows
    (tmp_path / "again").mkdir()
    b2 = make_builder(tmp_path / "again", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b2, tmp_path / "gym.jsonl", "extra")
    b2.split()
    assert [e.id for e in b2.final["train"]] == [e.id for e in train]


# ── task split and leakage ─────────────────────────────────────────────────


def test_tasks_sharing_a_prefix_land_in_one_group_and_one_split() -> None:
    rows = []
    for t in range(60):
        for k in range(3):
            rows.append(ex(t * 10 + k, "coding", f"task{t}", prefix=f"p{t}"))
    # tasks 0..4 share one opening context, so they are one group
    for e in rows:
        if e.task_id in {f"task{t}" for t in range(5)}:
            e.prefix = "shared-opening"
    build.assign_groups(rows)
    assert len({e.group for e in rows if e.task_id in {f"task{t}" for t in range(5)}}) == 1
    plan = build.plan_splits(rows, seed="s", eval_task_frac=0.1, val_frac=0.05,
                             max_group_share=0.5)
    split_of = {e.group: plan.groups[e.group] for e in rows}
    assert len({split_of[e.group] for e in rows if e.prefix == "shared-opening"}) == 1
    for e in rows:
        e.split = plan.groups[e.group]
    splits = {s: [e for e in rows if e.split == s] for s in build.SPLITS}
    assert all(splits.values())
    assert build.leakage(splits)["pass"]
    # deterministic for a seed
    again = build.plan_splits(rows, seed="s", eval_task_frac=0.1, val_frac=0.05,
                              max_group_share=0.5)
    assert again.groups == plan.groups


def test_groups_too_big_to_hold_out_stay_in_train() -> None:
    rows = [ex(i, "coding", "huge", prefix="hp") for i in range(50)]
    rows += [ex(100 + i, "coding", f"small{i}", prefix=f"sp{i}") for i in range(10)]
    build.assign_groups(rows)
    plan = build.plan_splits(rows, seed="s", eval_task_frac=0.5, val_frac=0.1,
                             max_group_share=0.1)
    assert plan.groups["huge"] == "train" and plan.ineligible_groups == 1


def test_leakage_check_catches_a_shared_prefix_task_and_hash() -> None:
    a, b = ex(1, "coding", "t1", prefix="same"), ex(2, "coding", "t2", prefix="same")
    a.group, b.group = "g1", "g2"
    res = build.leakage({"train": [a], "val": [b], "eval-tools": []})
    assert not res["pass"] and res["overlaps"]["train|val"]["prefix"] == 1
    c = ex(3, "coding", "t1", prefix="other")
    c.group = "g3"
    res = build.leakage({"train": [a], "val": [], "eval-tools": [c]})
    assert not res["pass"] and res["overlaps"]["train|eval-tools"]["task"] == 1


def test_prefix_hash_reads_the_first_messages_after_the_system_prompt() -> None:
    base = [{"role": "system", "content": "s1"}, {"role": "user", "content": "Open 12"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    other_sys = [{"role": "system", "content": "s2"}, *base[1:]]
    assert build.prefix_hash(base, 2) == build.prefix_hash(other_sys, 2)
    # normalized like the task proxy: digits masked, case folded
    renumbered = [base[0], {"role": "user", "content": "open 99"}, *base[2:]]
    assert build.prefix_hash(base, 2) == build.prefix_hash(renumbered, 2)
    assert build.prefix_hash(base, 3) != build.prefix_hash(base[:3] + [
        {"role": "user", "content": "different"}], 3)


def test_a_task_inside_another_examples_window_shares_its_group(tmp_path: Path) -> None:
    earlier = [{"role": "user", "content": "please handle the alpha job"},
               {"role": "assistant", "content": "done with alpha"}]
    rows = [trace("alpha", "bash", {"command": "make alpha"}),
            # a later task whose window still holds alpha's query (another opening)
            trace("charlie", "bash", {"command": "make charlie"}, ctx_extra=earlier),
            # the same words in another conversation are not alpha's context
            trace("delta", "bash", {"command": "make delta"}, ctx_extra=earlier,
                  session="beacon:s2")]
    b = make_builder(tmp_path)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", rows))
    a, c, d = b.examples
    assert a.prefix != c.prefix  # the prefix alone would not join them
    build.assign_groups(b.examples)
    assert a.group == c.group and d.group != a.group
    # and the leakage check sees context shared across splits
    res = build.leakage({"train": [c], "val": [], "eval-tools": [a]})
    assert not res["pass"] and res["overlaps"]["train|eval-tools"]["context"] == 1
    assert build.leakage({"train": [d], "val": [], "eval-tools": [a]})["overlaps"][
        "train|eval-tools"]["context"] == 0


def test_an_example_that_cannot_fit_joins_no_group_through_its_context(tmp_path: Path) -> None:
    earlier = [{"role": "user", "content": "please handle the alpha job"},
               {"role": "assistant", "content": "done with alpha"}]
    long_one = trace("kilo", "bash", {"command": "make kilo"}, ctx_extra=earlier)
    long_one["messages"][-2]["content"] += " details" * 400  # its own query is too long
    b = make_builder(tmp_path, max_tokens=600)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                        [trace("alpha", "bash", {"command": "make alpha"}), long_one]))
    (a,), (u,) = b.examples, b.unfittable
    assert u.ctx_keys == [] and a.ctx_keys
    build.assign_groups(b.examples + b.unfittable)
    assert a.group != u.group


# ── the builder end to end ─────────────────────────────────────────────────


def corpus(tmp_path: Path) -> Path:
    rows = []
    for t in range(40):
        task = word(t)
        for k, cmd in enumerate(["git status", "git log", "git diff"]):
            rows.append(trace(task, "bash", {"command": f"{cmd} {task}"}, ts=1_790_000_000 + t * 100 + k))
    for t in range(40, 52):
        rows.append(trace(word(t), "bash", {"command": f"systemctl status {word(t)}"}))
    for t in range(52, 64):
        rows.append(trace(word(t), "web_search", {"query": f"news {word(t)}"}))
    for t in range(64, 76):
        rows.append(trace(word(t), "memory", {"fact": f"likes {word(t)}"}))
    rows.append(trace(word(0), "bash", {"command": f"git status {word(0)}"}))  # duplicate
    rows.append(trace("flashjob", "bash", {"command": "ls"}, model="other-model"))
    return write_jsonl(tmp_path / "golden_traces_1_2.jsonl", rows)


def test_builder_writes_schema_valid_sets_with_caps_and_no_leakage(tmp_path: Path) -> None:
    b = make_builder(tmp_path, eval_task_frac=0.15, val_frac=0.05, holdout_max_group_share=0.5)
    feed(b, corpus(tmp_path))
    assert b.stats["excluded:duplicate"] == 1 and b.stats["excluded:non_teacher_model"] == 1
    info = b.split()
    gates = b.gates()
    assert gates["leakage"]["pass"]
    assert gates["category_cap"]["pass"], gates["category_cap"]
    assert info["train_caps"]["coding"]["dropped"] > 0
    (tmp_path / "out").mkdir()
    counts = b.write(tmp_path / "out")
    assert counts["train.jsonl"] == len(b.final["train"]) > 0
    # eval-tools and val-cases parse with score.py's own case parser
    cases = score.load_cases([tmp_path / "out" / "eval-tools.jsonl",
                              tmp_path / "out" / "val-cases.jsonl"])
    assert {c.set for c in cases} == {"eval-tools", "val"}
    for c in cases:
        assert c.messages[0]["role"] == "system" and c.messages[-1]["role"] == "user"
        assert c.expect["tool"] in {"bash", "web_search", "memory"}
        assert c.raw["meta"]["approximate"] is True
        # what score.py would send is the rebuilt request itself
        payload = score.build_payload(c, score.RequestSettings())
        assert payload["messages"] == c.messages and len(payload["tools"]) == len(TOOLS)
    train = [json.loads(x) for x in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    for row in train:
        a, z = row["loss_spans_char"][0]
        assert row["text"][a:z].endswith("<|im_end|>")
        assert row["messages"][0]["role"] == "system" and "Today's date: 2026-" in \
            row["messages"][0]["content"]
        assert row["approximate"] is True and "stored_context" in row["approx"]
    # nothing trace-derived in the report
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    blob = json.dumps(report)
    assert "please handle" not in blob and "git status" not in blob
    assert "golden_traces" not in blob and "beacon" not in blob


def test_verdicts_override_and_extra_traces_need_one(tmp_path: Path) -> None:
    golden = write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "bash", {"command": "make alpha"}),
        trace("bravo", "bash", {"command": "make bravo"}),
    ])
    gym = write_jsonl(tmp_path / "gym.jsonl", [
        trace("charlie", "web_search", {"query": "charlie"}, session="gym:1",
              source="max-gym-2026-11-01", verdict="pass"),
        trace("delta", "web_search", {"query": "delta"}, session="gym:2",
              source="max-gym-2026-11-01"),
        trace("alpha", "bash", {"command": "make alpha"}, source="max-gym-2026-11-01",
              verdict="pass"),  # dupe of a golden trace
    ])
    b = make_builder(tmp_path, verdicts={"golden_traces_1_1.jsonl:2": "fail"})
    feed(b, golden)
    feed(b, gym, "extra")
    labels = {e.trace_id: (e.label, e.source, e.category) for e in b.examples}
    assert labels["golden_traces_1_1.jsonl:1"][0] == "tool-success"
    assert "golden_traces_1_1.jsonl:2" not in labels
    assert labels["gym.jsonl:1"] == ("judge-pass", "max-gym-2026-11-01", "other")
    assert b.stats["excluded:judge_fail"] == 1 and b.stats["excluded:unverified"] == 1
    assert b.stats["excluded:duplicate"] == 1


def _judge_rows(index: list[dict[str, Any]], outcomes: list[tuple[str, str | None]]
                ) -> list[dict[str, Any]]:
    """Rows exactly as judge.py writes them: its pass pool and its own _row()."""
    judge = _load("distill_judge", "judge.py")
    runner = types.SimpleNamespace(
        config=types.SimpleNamespace(view="outcome"), client=types.SimpleNamespace(model="judge"),
        clock=lambda: datetime(2026, 10, 31, 6, 0, tzinfo=timezone.utc))

    def reply(v: str) -> dict[str, Any]:
        return judge.parse_verdict(json.dumps({"verdict": v, "reasons": ["r"],
                                               "rubric": {k: 1 for k in judge.RUBRIC_KEYS}}))

    rows = []
    for item, (status, v) in zip(judge.pass_pool(index), outcomes):
        fields = {"agrees": v == "pass", **reply(v)} if status == "ok" and v else {"error": "x"}
        rows.append(judge.JudgeRun._row(runner, item, status=status, **fields))
    fail_side = {"id": "fail-1", "label": "fail", "category": "coding",
                 "source": {"telemetry_rowid": 9}}
    rows.append(judge.JudgeRun._row(runner, fail_side, status="ok", agrees=True, **reply("fail")))
    return rows


def test_judge_sample_rows_are_applied_by_their_source_line(tmp_path: Path) -> None:
    ts = 1_790_000_000.0
    golden = write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace(word(i), "bash", {"command": f"make {word(i)}"}, ts=ts + i) for i in range(4)])
    index = [{"file": golden.name, "model": "qwen3.8-max", "h_prompt_call": f"k{i}",
              "category": "coding", "rowid": 100 + i, "tool": "bash", "ts": ts + i}
             for i in range(4)]
    rows = _judge_rows(index, [("ok", "fail"), ("unparseable", None), ("ok", "pass"),
                               ("error", None)])
    assert rows[1]["label"] == "pass" and "verdict" not in rows[1]  # a sampled label, no verdict
    vfile = write_jsonl(tmp_path / "judge-sample.jsonl", rows)
    stats: collections.Counter = collections.Counter()
    checks: dict[str, Any] = {}
    verdicts = build.load_verdicts([str(vfile)], stats, checks)
    # judge.py's line is 0-based; the trace id's line is 1-based
    assert verdicts == {"golden_traces_1_1.jsonl:1": "fail", "golden_traces_1_1.jsonl:3": "pass"}
    assert checks == {"golden_traces_1_1.jsonl:1": 100, "golden_traces_1_1.jsonl:3": 102}
    assert stats["verdict_rows_not_ok"] == 2 and stats["verdict_rows_fail_side"] == 1
    assert stats["verdict_rows_unusable"] == 0
    idx_path = write_jsonl(tmp_path / "index.jsonl", index)
    b = make_builder(tmp_path, verdicts=verdicts, verdict_checks=checks,
                     index=build.load_index(str(idx_path)))
    feed(b, golden)
    assert {e.trace_id: e.label for e in b.examples} == {
        "golden_traces_1_1.jsonl:2": "tool-success", "golden_traces_1_1.jsonl:3": "judge-pass",
        "golden_traces_1_1.jsonl:4": "tool-success"}
    assert b.stats["excluded:judge_fail"] == 1
    b.split()
    assert b.gates()["verdicts"]["pass"]

    # the corpus moved since judging: the verdict is not applied, and the gate fails
    index[0]["rowid"] = 555
    moved = make_builder(tmp_path, verdicts=verdicts, verdict_checks=checks,
                         index=build.load_index(str(write_jsonl(tmp_path / "i2.jsonl", index))))
    feed(moved, golden)
    assert moved.stats["verdict_index_mismatch"] == 1 and moved.stats["excluded:judge_fail"] == 0
    moved.split()
    gate = moved.gates()["verdicts"]
    assert not gate["pass"] and gate["index_mismatch"] == 1


def test_unusable_or_unmatched_verdicts_fail_the_gate(tmp_path: Path) -> None:
    vfile = write_jsonl(tmp_path / "v.jsonl", [{"trace_id": "golden_traces_1_1.jsonl:9",
                                                "verdict": "pass"}, {"verdict": "pass"}])
    stats: collections.Counter = collections.Counter()
    verdicts = build.load_verdicts([str(vfile)], stats)
    b = make_builder(tmp_path, verdicts=verdicts)
    b.stats.update(stats)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                        [trace("alpha", "bash", {"command": "make alpha"})]))
    b.split()
    gate = b.gates()["verdicts"]
    assert not gate["pass"] and gate["unmatched"] == 1 and gate["rows_unusable"] == 1


def test_build_refuses_input_files_that_share_a_name(tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    golden = write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [trace("a", "bash", {"command": "x"})])
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        write_jsonl(tmp_path / d / "traces.jsonl", [trace(d, "bash", {"command": d})])
    rc = build.main(["build", "--traces", str(golden), "--extra-traces",
                     str(tmp_path / "*" / "traces.jsonl"), "--pack", str(tmp_path / "no-pack"),
                     "--tokenizer-dir", "unused", "--out", str(tmp_path / "out"), "--dry-run"])
    assert rc == 2 and "traces.jsonl" in capsys.readouterr().err
    assert build.duplicate_names([golden, tmp_path / "a" / "traces.jsonl"]) == []


def test_calls_that_fail_against_todays_tools_are_excluded(tmp_path: Path) -> None:
    b = make_builder(tmp_path)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "retired_tool", {"x": "y"}),
        trace("bravo", "read_file", {"offset": "3"}),   # misses required "path"
        trace("charlie", "read_file", {"path": "ok.txt"}),
    ]))
    assert b.stats["excluded:tool_not_offered_now"] == 1
    assert b.stats["excluded:required_argument_missing_now"] == 1
    assert [e.trace_id for e in b.examples] == ["golden_traces_1_1.jsonl:3"]


def test_chat_mode_compacted_and_recovery_flags(tmp_path: Path) -> None:
    chat = trace("echo", "bash", {}, source="max-gym-2026-11-01", verdict="pass",
                 suppress_tools=True)
    chat["messages"][-1] = {"role": "assistant", "content": "Just an answer."}
    compacted = trace("foxtrot", "bash", {"command": "make foxtrot"}, ctx_extra=[
        {"role": "user", "content": "[Compacted summary of turns 1-9] earlier work"}],
        source="max-gym-2026-11-01", verdict="pass", error_recovery=True)
    recovering = trace("golf", "bash", {"command": "make golf"}, source="max-gym-2026-11-01",
                       verdict="pass")
    recovering["messages"][-1:-1] = [
        {"role": "assistant", "content": "", "tool_calls": [call("bash", {"command": "mk"}, "c0")]},
        {"role": "tool", "tool_call_id": "c0", "content": "x", "is_error": True}]
    b = make_builder(tmp_path)
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [chat, compacted, recovering]), "extra")
    by_id = {e.trace_id: e for e in b.examples}
    c = by_id["gym.jsonl:1"]
    assert c.flags["chat_mode"] and c.category == "chat-mode" and c.expect_tool is None
    f = by_id["gym.jsonl:2"]
    # _meta (or telemetry) alone says nothing about the context that is trained:
    # no failed result in the request, so no error/recovery example
    assert f.flags["compacted"] and not f.flags["error_recovery"] and f.flags["error_signal"]
    g = by_id["gym.jsonl:3"]
    assert g.flags["error_recovery"]
    heavy = b.read_heavy(c)
    assert heavy["tools"] == [] and "<tools>" not in heavy["render"]["text"]
    assert heavy["expect"] == {"tool": None}


def test_long_examples_are_truncated_and_kept_as_long_context_candidates(tmp_path: Path) -> None:
    filler = [{"role": "user", "content": "background " + "words " * 200},
              {"role": "assistant", "content": "noted " * 100}] * 3
    rows = [trace("golf", "bash", {"command": "make golf"}, ctx_extra=filler)]
    b = make_builder(tmp_path)
    first = None
    for t in build.iter_traces([write_jsonl(tmp_path / "golden_traces_1_1.jsonl", rows)],
                               "golden", b.stats):
        req = b.rebuilder.rebuild(t, build.trace_keys(t.raw))
        first = R.render_example(b.student, req.messages, req.tools, req.template_kwargs)
    assert first is not None
    b.s.max_tokens = first.n_tokens - 50
    feed(b, tmp_path / "golden_traces_1_1.jsonl")
    (e,) = b.examples
    assert e.longctx and e.fits and e.dropped_messages > 0 and e.n_tokens <= b.s.max_tokens
    heavy = b.read_heavy(e)
    assert heavy["original_messages"][1:-1] != heavy["messages"][1:-1]
    assert heavy["messages"][0]["role"] == "system"


def test_fail_traces_become_hard_cases_only_outside_train(tmp_path: Path) -> None:
    b = make_builder(tmp_path, eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                        [trace(word(i), "bash", {"command": f"make {word(i)}"}) for i in range(6)]))
    fails = write_jsonl(tmp_path / "fail.jsonl", [
        trace("hotel", "read_file", {"path": "missing.txt"}),   # its own task
        trace(word(1), "bash", {"command": "make broken"}),     # a train task
    ])
    feed(b, fails, "fail")
    info = b.split()
    assert info["fail_used"] == 1 and info["fail_dropped_task_in_train_or_val"] == 1
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    (case,) = score.load_cases([out / "eval-tools.jsonl"])
    assert case.raw["meta"]["hard_case"] and case.expect["tool"] == "read_file"
    assert case.expect["args_match"] == "subset"
    assert build.leakage(b.final)["pass"]


def test_a_failed_call_missing_a_required_argument_is_still_a_hard_case(tmp_path: Path) -> None:
    b = make_builder(tmp_path)
    feed(b, write_jsonl(tmp_path / "fail.jsonl", [
        trace("hotel", "read_file", {"offset": "3"}),   # why it failed: no "path"
        trace("india", "retired_tool", {"x": "y"}),     # the case could not offer it
    ]), "fail")
    assert [e.trace_id for e in b.fails] == ["fail.jsonl:1"]
    assert b.stats["excluded:tool_not_offered_now"] == 1
    assert b.stats["excluded:required_argument_missing_now"] == 0
    assert b.read_heavy(b.fails[0])["expect"]["tool"] == "read_file"


def test_a_failed_call_whose_request_cannot_fit_is_not_a_case(tmp_path: Path) -> None:
    b = make_builder(tmp_path, max_tokens=40)
    feed(b, write_jsonl(tmp_path / "fail.jsonl",
                        [trace("hotel", "read_file", {"path": "missing.txt"})]), "fail")
    assert b.fails == [] and b.stats["excluded:fail_unfittable"] == 1


# ── structured context from a conversation-store snapshot ─────────────────


def make_lcm(path: Path, *, old_history: int = 0) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE lcm_messages (id TEXT PRIMARY KEY, session_id TEXT, turn_index INT,"
                " role TEXT, content TEXT, content_json TEXT, token_count INT, timestamp REAL,"
                " compacted INT DEFAULT 0, provenance TEXT DEFAULT 'user', is_trusted INT DEFAULT 1,"
                " author TEXT)")
    rows = [
        ("m1", 1, "user", "earlier question", [{"type": "text", "text": "earlier question"}], 100),
        ("m2", 2, "assistant", "earlier answer", [{"type": "text", "text": "earlier answer"}], 101),
        ("m3", 3, "user", "fix the thing", [{"type": "text", "text": "fix the thing"}], 200),
        ("m4", 4, "assistant", "Looking.", [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "git status"}}], 260),
        ("m5", 5, "user", "", [
            {"type": "tool_result", "tool_use_id": "t1", "content": "fatal: oops", "is_error": True}], 260),
        ("m6", 6, "assistant", "", [
            {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "a.py"}}], 260),
    ]
    for i in range(old_history):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"old {role} message " + "lorem ipsum " * 300
        rows.insert(i, (f"o{i}", -100 + i, role, text, [{"type": "text", "text": text}], 10 + i))
    for mid, ti, role, text, blocks, ts in rows:
        con.execute("INSERT INTO lcm_messages (id, session_id, turn_index, role, content,"
                    " content_json, token_count, timestamp) VALUES (?,?,?,?,?,?,0,?)",
                    (mid, "beacon:s1", ti, role, text, json.dumps(blocks), ts))
    con.commit()
    con.close()
    return path


def test_lcm_snapshot_restores_earlier_calls_and_results(tmp_path: Path) -> None:
    snap = build.LcmSnapshot(str(make_lcm(tmp_path / "lcm.db")))
    t = {"messages": [{"role": "user", "content": "earlier question"},
                      {"role": "assistant", "content": "earlier answer"},
                      {"role": "user", "content": "fix the thing"},
                      {"role": "assistant", "content": "",
                       "tool_calls": [call("read_file", {"path": "a.py"})]}],
         "tools": [], "_meta": {"model": "qwen3.8-max", "tool_name": "read_file",
                                "timestamp": 250.0, "session_id": "beacon:s1"}}
    b = make_builder(tmp_path, lcm=snap)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [t]))
    (e,) = b.examples
    assert e.context_source == "lcm" and e.flags["error_recovery"]
    msgs = b.read_heavy(e)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant",
                                        "tool", "assistant"]
    assert msgs[4]["tool_calls"][0]["function"]["name"] == "bash"
    assert msgs[5]["content"] == "fatal: oops"
    assert msgs[-1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert b.lcm_match["exact"] == 1
    # the snapshot is opened read-only
    with pytest.raises(sqlite3.OperationalError):
        snap.con.execute("DELETE FROM lcm_messages")


def test_lcm_history_beyond_the_prewindow_is_counted_not_rendered(tmp_path: Path) -> None:
    snap = build.LcmSnapshot(str(make_lcm(tmp_path / "lcm.db", old_history=6)))
    t = {"messages": [{"role": "user", "content": "fix the thing"},
                      {"role": "assistant", "content": "",
                       "tool_calls": [call("read_file", {"path": "a.py"})]}],
         "tools": [], "_meta": {"model": "qwen3.8-max", "tool_name": "read_file",
                                "timestamp": 250.0, "session_id": "beacon:s1"}}
    b = make_builder(tmp_path, lcm=snap, max_tokens=1000, prewindow_chars_per_token=1)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [t]))
    (e,) = b.examples
    assert e.context_source == "lcm" and e.omitted_messages == 6 and e.longctx
    assert e.n_tokens_original > e.n_tokens and e.lcm_target == "m6"
    msgs = b.read_heavy(e)["messages"]
    assert not any("lorem ipsum" in (m.get("content") or "") for m in msgs[:2])
    assert msgs[1]["role"] == "user"


def lcm_db(path: Path, rows: list[tuple[str, int, str, str, list[dict[str, Any]], float]]) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE lcm_messages (id TEXT PRIMARY KEY, session_id TEXT, turn_index INT,"
                " role TEXT, content TEXT, content_json TEXT, token_count INT, timestamp REAL,"
                " compacted INT DEFAULT 0, provenance TEXT DEFAULT 'user', is_trusted INT DEFAULT 1,"
                " author TEXT)")
    for mid, ti, role, text, blocks, ts in rows:
        con.execute("INSERT INTO lcm_messages (id, session_id, turn_index, role, content,"
                    " content_json, token_count, timestamp) VALUES (?,?,?,?,?,?,0,?)",
                    (mid, "beacon:s1", ti, role, text, json.dumps(blocks), ts))
    con.commit()
    con.close()
    return path


def use(tool: str, args: dict[str, Any], tid: str) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "id": tid, "name": tool, "input": args}]


def result(tid: str, text: str, err: bool = False) -> list[dict[str, Any]]:
    return [{"type": "tool_result", "tool_use_id": tid, "content": text, "is_error": err}]


def golden_call(tool: str, args: dict[str, Any], ts: float) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "run the checks"},
                         {"role": "assistant", "content": "", "tool_calls": [call(tool, args)]}],
            "tools": [], "_meta": {"model": "qwen3.8-max", "tool_name": tool,
                                   "timestamp": ts, "session_id": "beacon:s1"}}


def test_a_turn_longer_than_the_window_keeps_its_query_and_latest_steps(tmp_path: Path) -> None:
    rows: list[Any] = [("q", 1, "user", "run the checks", [{"type": "text", "text": "run the checks"}], 100)]
    for i in range(8):
        rows.append((f"a{i}", 2 + 2 * i, "assistant", "", use("bash", {"command": f"check {word(i)}"},
                                                                f"t{i}"), 300))
        rows.append((f"r{i}", 3 + 2 * i, "user", "", result(f"t{i}", "output " * 400), 300))
    rows.append(("last", 40, "assistant", "", use("read_file", {"path": "report.txt"}, "tz"), 300))
    snap = build.LcmSnapshot(str(lcm_db(tmp_path / "lcm.db", rows)))
    b = make_builder(tmp_path, lcm=snap, max_tokens=4000, prewindow_chars_per_token=2)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                        [golden_call("read_file", {"path": "report.txt"}, 250.0)]))
    (e,) = b.examples
    msgs = b.read_heavy(e)["messages"]
    assert msgs[1] == {"role": "user", "content": "run the checks"}
    assert msgs[2]["role"] == "assistant" and msgs[-1]["tool_calls"][0]["function"]["name"] == \
        "read_file"
    assert e.omitted_messages > 0 and e.n_tokens <= 4000


def test_the_same_call_twice_in_a_turn_maps_to_both_rows(tmp_path: Path) -> None:
    rows: list[Any] = [("q", 1, "user", "run the checks", [{"type": "text", "text": "run the checks"}], 100),
                       ("a1", 2, "assistant", "", use("bash", {"command": "make test"}, "t1"), 300),
                       ("r1", 3, "user", "", result("t1", "1 failed"), 300),
                       ("a2", 4, "assistant", "", use("bash", {"command": "make test"}, "t2"), 300)]
    snap = build.LcmSnapshot(str(lcm_db(tmp_path / "lcm.db", rows)))
    b = make_builder(tmp_path, lcm=snap)
    t1 = golden_call("bash", {"command": "make test"}, 200.0)
    t2 = golden_call("bash", {"command": "make test"}, 210.0)
    # a distinct trace with the same call (Phase 1's prompt+call dedupe would
    # otherwise drop it before the snapshot is consulted)
    t2["messages"][0]["content"] = "run the checks once more"
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [t1, t2]))
    assert [e.lcm_target for e in b.examples] == ["a1", "a2"]
    assert len(b.read_heavy(b.examples[1])["messages"]) == 5  # system, query, a1, r1, a2


def test_lcm_falls_back_to_the_stored_window_when_the_call_is_not_found(tmp_path: Path) -> None:
    snap = build.LcmSnapshot(str(make_lcm(tmp_path / "lcm.db")))
    b = make_builder(tmp_path, lcm=snap)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                        [trace("india", "bash", {"command": "make india"}, ts=250.0)]))
    (e,) = b.examples
    assert e.context_source == "stored" and b.lcm_match["unmatched"] == 1


def test_a_row_with_a_failed_parallel_call_is_not_trained(tmp_path: Path) -> None:
    q = ("q", 1, "user", "run the checks", [{"type": "text", "text": "run the checks"}], 100)
    rows: list[Any] = [
        q, ("a1", 2, "assistant", "", use("bash", {"command": "make test"}, "t1")
            + use("read_file", {"path": "a.py"}, "t2"), 300),
        ("r1", 3, "user", "", result("t1", "ok") + result("t2", "no such file", err=True), 300),
        ("a2", 4, "assistant", "", use("read_file", {"path": "b.py"}, "t3"), 300)]
    snap = build.LcmSnapshot(str(lcm_db(tmp_path / "lcm.db", rows)))
    b = make_builder(tmp_path, lcm=snap)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        golden_call("bash", {"command": "make test"}, 250.0),     # its sibling failed
        golden_call("read_file", {"path": "b.py"}, 260.0)]))
    assert b.stats["excluded:failed_sibling_call"] == 1
    (e,) = b.examples
    assert e.lcm_target == "a2" and e.flags["error_recovery"]
    # the failed call is a hard case, expecting its own tool, not the first call's
    feed(b, write_jsonl(tmp_path / "fail.jsonl",
                        [golden_call("read_file", {"path": "a.py"}, 255.0)]), "fail")
    (f,) = b.fails
    assert f.lcm_target == "a1" and f.expect_tool == "read_file"
    assert b.read_heavy(f)["expect"]["tool"] == "read_file"


def test_a_step_filed_under_a_later_query_joins_that_querys_task(tmp_path: Path) -> None:
    # The second query was persisted before the turn's rows, which then got
    # lower indexes: the rebuilt context of the call ends on the first query.
    rows: list[Any] = [
        ("q1", 1, "user", "check the build", [{"type": "text", "text": "check the build"}], 100),
        ("a1", 2, "assistant", "", use("bash", {"command": "make"}, "t1"), 300),
        ("r1", 3, "user", "", result("t1", "ok"), 300),
        ("a2", 4, "assistant", "", use("read_file", {"path": "log.txt"}, "t2"), 300),
        ("r2", 5, "user", "", result("t2", "ok"), 300),
        ("q2", 9, "user", "also tidy the docs", [{"type": "text", "text": "also tidy the docs"}],
         150)]
    snap = build.LcmSnapshot(str(lcm_db(tmp_path / "lcm.db", rows)))
    first = golden_call("bash", {"command": "make"}, 200.0)
    first["messages"][0]["content"] = "check the build"
    second = golden_call("read_file", {"path": "log.txt"}, 210.0)
    second["messages"][:1] = [{"role": "user", "content": "check the build"},
                              {"role": "assistant", "content": "on it"},
                              {"role": "user", "content": "also tidy the docs"}]
    b = make_builder(tmp_path, lcm=snap)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [first, second]))
    e1, e2 = b.examples
    assert e2.lcm_target == "a2" and e1.task_id != e2.task_id and e1.prefix != e2.prefix
    assert b.stats["lcm_query_differs_from_prompt"] == 1
    build.assign_groups(b.examples)
    assert e1.group == e2.group


def test_error_recovery_needs_the_failed_step_in_the_kept_window(tmp_path: Path) -> None:
    def db(name: str, failed_step: int) -> Any:
        rows: list[Any] = [("q", 1, "user", "run the checks",
                            [{"type": "text", "text": "run the checks"}], 100)]
        for i in range(8):
            rows.append((f"a{i}", 2 + 2 * i, "assistant", "",
                         use("bash", {"command": f"check {word(i)}"}, f"t{i}"), 300))
            rows.append((f"r{i}", 3 + 2 * i, "user", "",
                         result(f"t{i}", "output " * 400, err=i == failed_step), 300))
        rows.append(("last", 40, "assistant", "", use("read_file", {"path": "r.txt"}, "tz"), 300))
        return build.LcmSnapshot(str(lcm_db(tmp_path / name, rows)))

    for name, failed_step, expected in (("early.db", 0, False), ("late.db", 7, True)):
        b = make_builder(tmp_path, lcm=db(name, failed_step), max_tokens=4000)
        feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                            [golden_call("read_file", {"path": "r.txt"}, 250.0)]))
        (e,) = b.examples
        assert e.dropped_steps > 0 and e.flags["error_signal"]
        assert e.flags["error_recovery"] is expected


def test_requests_ending_on_an_assistant_or_with_unanswered_calls_are_excluded(
        tmp_path: Path) -> None:
    q = ("q", 1, "user", "run the checks", [{"type": "text", "text": "run the checks"}], 100)
    rows: list[Any] = [
        q, ("a1", 2, "assistant", "", use("bash", {"command": "make a"}, "t1"), 300),  # no result
        ("a2", 3, "assistant", "", use("read_file", {"path": "x"}, "t2"), 300),
        ("r2", 4, "user", "", result("t2", "ok"), 300),
        ("a3", 5, "assistant", "", use("bash", {"command": "make c"}, "t3"), 300)]
    snap = build.LcmSnapshot(str(lcm_db(tmp_path / "lcm.db", rows)))
    b = make_builder(tmp_path, lcm=snap)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        golden_call("read_file", {"path": "x"}, 250.0),
        golden_call("bash", {"command": "make c"}, 260.0)]))
    assert b.examples == []
    assert b.stats["excluded:target_after_assistant"] == 1
    assert b.stats["excluded:unanswered_tool_call"] == 1
    ok = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
          {"role": "assistant", "tool_calls": [call("bash", {"command": "ls"}, "c1")]},
          {"role": "tool", "tool_call_id": "c1", "content": "a"},
          {"role": "assistant", "content": "done"}]
    assert build.request_shape_problem(ok) is None


# ── llama-server spot check ────────────────────────────────────────────────


def test_apply_template_client_sends_the_chat_payload() -> None:
    seen: dict[str, Any] = {}

    def post(url: str, body: dict[str, Any]) -> dict[str, Any]:
        seen.update(url=url, body=body)
        return {"prompt": "rendered"}

    payload = {"messages": [{"role": "user", "content": "hi"}], "tools": openai_tools(),
               "tool_choice": "auto", "chat_template_kwargs": dict(R.THINKING_OFF),
               "max_tokens": 5, "stream": True}
    assert R.apply_template("http://127.0.0.1:8080/", payload, post=post) == "rendered"
    assert seen["url"] == "http://127.0.0.1:8080/apply-template"
    assert set(seen["body"]) == {"messages", "tools", "tool_choice", "chat_template_kwargs",
                                 "add_generation_prompt"}
    assert seen["body"]["add_generation_prompt"] is True
    with pytest.raises(R.RenderError):
        R.apply_template("http://x", payload, post=lambda u, b: {"nope": 1})
    assert R.first_mismatch("abc", "abc") is None and R.first_mismatch("abc", "abd") == 2
    assert R.first_mismatch("ab", "abc") == 2


def test_spot_check_compares_byte_for_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    b = make_builder(tmp_path, eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl",
                        [trace(word(i), "bash", {"command": f"make {word(i)}"}) for i in range(3)]))
    b.split()
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    st = make_student()
    monkeypatch.setattr(R.Student, "load", classmethod(lambda cls, d: st))

    def honest(url: str, body: dict[str, Any]) -> dict[str, Any]:
        msgs = body["messages"]
        if not body["add_generation_prompt"]:
            # a full render is only ever asked for a conversation ending on tool results
            assert msgs[-1]["role"] == "tool"
        return {"prompt": R.render(st.template, msgs, body.get("tools"),
                                   add_generation_prompt=body["add_generation_prompt"],
                                   template_kwargs=body.get("chat_template_kwargs"))}

    def spot(rows: Path, post: Any) -> tuple[int, dict[str, Any]]:
        args = build.parse_args(["spot-check", "--server", "http://127.0.0.1:1", "--rows",
                                 str(rows), "--tokenizer-dir", "unused", "--n", "5"])
        rc = build.cmd_spot_check(args, post=post)
        return rc, json.loads(capsys.readouterr().out)

    rc, res = spot(out / "train.jsonl", honest)
    assert rc == 0 and res["spot_check"] == "PASS" and len(res["rows"]) == 3
    assert all(r["completion_match"] is True for r in res["rows"])

    def drifted(url: str, body: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": honest(url, body)["prompt"].replace('"name": ', '"name":')}

    rc, res = spot(out / "train.jsonl", drifted)
    assert rc == 1 and res["spot_check"] == "FAIL"
    assert all("prompt_first_mismatch" in r for r in res["rows"])
    assert all("prompt_ours" not in r for r in res["rows"])  # no text unless --show-diff

    # The server renders the loss-bearing turn (a call's parameters) its own way.
    def completion_drift(url: str, body: dict[str, Any]) -> dict[str, Any]:
        text = honest(url, body)["prompt"]
        return {"prompt": text if body["add_generation_prompt"]
                else text.replace("<parameter=command>", "<parameter=cmd>")}

    rc, res = spot(out / "train.jsonl", completion_drift)
    assert rc == 1 and all(r["prompt_match"] and r["completion_match"] is False
                           for r in res["rows"])

    # train.jsonl built with another template: the server and today's render
    # agree, but the stored text that would be trained does not.
    rows = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    for r in rows:
        r["text"] = r["text"].replace("Today's date", "Todays date")
    stale = write_jsonl(tmp_path / "stale.jsonl", rows)
    rc, res = spot(stale, honest)
    assert rc == 1 and all(r["prompt_match"] and not r["stored_match"] for r in res["rows"])


def test_a_build_without_the_spot_check_is_incomplete_not_pass(tmp_path: Path) -> None:
    b = make_builder(tmp_path, eval_task_frac=0.15, val_frac=0.05, holdout_max_group_share=0.5,
                     val_share_range=(0.0, 1.0))
    feed(b, corpus(tmp_path))
    b.split()
    gates = b.gates()
    assert all(v["pass"] for v in gates.values() if isinstance(v, dict) and "pass" in v)
    assert gates["spot_check"]["status"] == "NOT RUN" and gates["overall"] == "INCOMPLETE"

    # --spot-check-server: the check runs on the written rows and decides the verdict
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)

    def server(url: str, body: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": R.render(b.student.template, body["messages"], body.get("tools"),
                                   add_generation_prompt=body["add_generation_prompt"],
                                   template_kwargs=body.get("chat_template_kwargs"))}

    def down(url: str, body: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("refused")

    gates["spot_check"] = build.spot_check_gate("http://x", out / "train.jsonl", b.student,
                                                post=server)
    assert gates["spot_check"]["status"] == "PASS" and build.overall_verdict(gates) == "PASS"
    assert set(gates["spot_check"]) == {"status", "rows", "mismatched", "pass"}  # no ids
    gates["spot_check"] = build.spot_check_gate("http://x", out / "train.jsonl", b.student,
                                                post=down)
    assert gates["spot_check"]["status"] == "ERROR" and build.overall_verdict(gates) == "FAIL"


# ── the request pack from Prometheus's own builders ───────────────────────


def test_pack_uses_the_prompt_builder_without_memory_and_every_tool(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "SOUL.md").write_text("SOUL-MARKER: you are the test agent.\n")
    (cfg_dir / "MEMORY.md").write_text("MEMORY-MARKER: a private fact.\n")
    config = tmp_path / "prometheus.yaml"
    config.write_text("security: {}\ntools:\n  deferred_loading:\n    enabled: auto\n")
    pack = build.build_pack(str(config), str(tmp_path))
    sp = pack["system_prompt"]
    assert build.DATE_SENTINEL in sp and "SOUL-MARKER" in sp and "MEMORY-MARKER" not in sp
    names = {t["name"] for t in pack["tools"]["full"]}
    assert {"bash", "read_file", "write_file", "edit_file"} <= names
    assert {t["name"] for t in pack["tools"]["stubs"]} == names
    assert pack["runtime_catalog"] == "stubs"
    # the patches were undone
    from prometheus.context import prompt_assembler as pa

    assert pa.get_environment_info(cwd=str(tmp_path)).date != build.DATE_SENTINEL


def test_runtime_catalog_follows_deferred_loading() -> None:
    assert build.runtime_catalog({}) == "stubs"
    assert build.runtime_catalog({"tools": {"deferred_loading": {"enabled": False}}}) == "full"
    assert build.runtime_catalog(
        {"tools": {"deferred_loading": {"enabled": "auto", "local_catalog": "names"}}}) == "full"


# ── data policy: only listed train sources train; Max rows are eval-only ──

GYM = "qwen38-27b-gym"
GYM_MODEL = "qwen3.8-27b"
VERIFIED = {"verified": True, "tool_success": True, "sandboxed": True, "side_effects": False,
            "judge": {"verdict": "pass", "model": GYM_MODEL}}


def gym(task: str, tool: str, args: dict[str, Any], *, model: str = GYM_MODEL,
        session: str | None = None, verification: Any = VERIFIED, **meta: Any
        ) -> dict[str, Any]:
    """A teacher_gen.py row: golden-trace shape, source tag, verification record."""
    if verification is not None:
        meta["verification"] = verification
    meta.setdefault("source", GYM)
    return trace(task, tool, args, model=model, session=session or f"gym:{task}", **meta)


def gym_chat(task: str, answer: str = "Here is the plain answer.", **kw: Any) -> dict[str, Any]:
    """A chat-mode gym row: no tools offered, a prose answer, no tool name."""
    row = gym(task, "bash", {}, suppress_tools=True, category="chat-mode", **kw)
    row["messages"][-1] = {"role": "assistant", "content": answer}
    row["tools"] = []
    row["_meta"].pop("tool_name")
    return row


def policy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cats = ["coding", "other", "memory/wiki", "sentinel/daemon"]
    rows = []
    for t in range(40):
        task = "gym" + word(t)
        rows.append(gym(task, "bash", {"command": f"make {task}"}, category=cats[t % 4]))
        rows.append(gym(task, "read_file", {"path": f"{task}.txt"}, category=cats[t % 4]))
    rows += [gym_chat("chat" + word(t)) for t in range(8)]
    return corpus(tmp_path), write_jsonl(tmp_path / "gym.jsonl", rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_only_listed_train_sources_train_and_max_rows_are_eval_only(tmp_path: Path) -> None:
    golden, gym_file = policy_inputs(tmp_path)
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.15, val_frac=0.05,
                     holdout_max_group_share=0.5, val_share_range=(0.0, 1.0))
    feed(b, golden)
    feed(b, gym_file, "extra")
    max_rows = [e for e in b.examples if e.source == "max-golden"]
    assert max_rows and all(e.max_derived and not e.train_ok for e in max_rows)
    info = b.split()
    gates = b.gates()
    for split in ("train", "val"):
        assert all(e.source == GYM and e.train_ok and not e.max_derived and e.model == GYM_MODEL
                   for e in b.final[split])
    assert b.final["train"] and b.final["val"]
    # every Max pool row is held-out scoring material, and some gym tasks are held out too
    ev = b.final["eval-tools"]
    assert sum(e.source == "max-golden" for e in ev) == len(max_rows)
    assert any(e.source == GYM for e in ev)
    assert info["plan"].pinned_groups > 0 and info["plan"].pinned_train_rows == 0
    assert gates["data_policy"]["pass"] and gates["data_policy"]["train_sources"] == [GYM]
    assert gates["data_policy"]["max_derived_in_train_val"] == 0
    # leakage is checked across sources, and passes
    assert gates["leakage"]["pass"]
    assert set(gates["leakage"]["sources"]["eval-tools"]) >= {"max-golden", GYM}
    assert gates["leakage"]["sources"]["train"] == [GYM]
    assert gates["train_size"]["value"] == len(b.final["train"])
    assert gates["train_size"]["by_source"] == {GYM: len(b.final["train"])}
    assert gates["category_cap"]["pass"] and gates["presence"]["chat_mode"] >= 1

    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    for row in read_jsonl(out / "train.jsonl") + read_jsonl(out / "val.jsonl"):
        assert (row["source"], row["model"], row["max_derived"]) == (GYM, GYM_MODEL, False)
    cases = score.load_cases([out / "eval-tools.jsonl", out / "val-cases.jsonl"])
    max_cases = [c for c in cases if c.raw["meta"]["source"] == "max-golden"]
    assert max_cases and {c.set for c in max_cases} == {"eval-tools"}
    assert all(c.raw["meta"]["max_derived"] and c.raw["meta"]["model"] == "qwen3.8-max"
               for c in max_cases)

    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    # pool.tasks counts the pool's own tasks, beside its rows (Max ones included);
    # the tasks the hold-out quotas came from (pinned groups left out) are apart
    pool_tasks = {e.task_id for e in b.examples}
    assert report["pool"]["rows"] == len(b.examples)
    assert report["pool"]["tasks"] == len(pool_tasks)
    assert pool_tasks >= {e.task_id for e in max_rows}
    pinned = {e.group for e in b.examples if not e.train_ok}
    free = {e.task_id for e in b.examples if e.group not in pinned}
    assert report["eval_tools"]["splittable_tasks"] == info["plan"].total_tasks == len(free)
    assert info["plan"].total_tasks < report["pool"]["tasks"]
    dp = report["data_policy"]
    assert dp["train_sources"] == [GYM] and dp["rule"] == build.POLICY_RULE
    mg, gs = dp["sources"]["max-golden"], dp["sources"][GYM]
    assert not mg["listed"] and mg["train"] == mg["val"] == 0 and mg["eval-tools"] == len(max_rows)
    assert mg["excluded_by_reason"] == {"duplicate": 1, "non_teacher_model": 1}
    assert gs["listed"] and gs["input"] == gs["accepted"] == gs["train_eligible"] == 88
    assert gs["models"] == {GYM_MODEL: 88} and gs["verification"] == {"pass": 88}
    assert gs["chat_mode"] == 8 and gs["max_derived"] == 0
    assert gs["train"] + gs["val"] + gs["eval-tools"] <= 88
    # the guards' key: train/val hold only the local teacher's rows
    assert report["settings"]["teacher_model"] == GYM_MODEL == dp["teacher_model"]
    assert report["settings"]["train_sources"] == [GYM]
    blob = json.dumps(report)
    assert "please handle" not in blob and "gym.jsonl" not in blob and "gym:" not in blob


def test_the_trainers_guard_clears_policy_rows_and_still_refuses_max(tmp_path: Path) -> None:
    if not (_DIR / "train_qlora.py").is_file():
        pytest.skip("train_qlora.py not in this checkout")
    T = _load("distill_train_qlora", "train_qlora.py")
    golden, gym_file = policy_inputs(tmp_path)
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.15, val_frac=0.05,
                     holdout_max_group_share=0.5)
    feed(b, golden)
    feed(b, gym_file, "extra")
    info = b.split()
    gates = b.gates()
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    (out / T.BUILD_REPORT).write_text(json.dumps(report))
    teacher = T.builder_teacher_is_max(out / "train.jsonl")
    assert teacher is False  # train/val hold only the local teacher's rows
    rows = read_jsonl(out / "train.jsonl")
    assert rows and not any(T.is_max_derived(r, teacher) for r in rows)
    cases = read_jsonl(out / "eval-tools.jsonl")
    assert any(T.is_max_derived(c, teacher) for c in cases)


def test_a_group_holding_a_max_row_goes_to_eval_tools_whole(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "bash", {"command": "make alpha"})]))
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "read_file", {"path": "alpha.txt"}),   # Max's task: held out with it
        *[gym("gym" + word(i), "bash", {"command": f"make {i}"}) for i in range(4)]]), "extra")
    info = b.split()
    by_id = {e.trace_id: e.split for e in b.examples}
    assert by_id["golden_traces_1_1.jsonl:1"] == by_id["gym.jsonl:1"] == "eval-tools"
    assert {by_id[f"gym.jsonl:{i}"] for i in range(2, 6)} == {"train"}
    assert info["plan"].pinned_groups == 1 and info["plan"].pinned_train_rows == 1
    assert build.leakage(b.final)["pass"]
    # without a policy the same rows all train (the library default)
    (tmp_path / "legacy").mkdir()
    b2 = make_builder(tmp_path / "legacy", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b2, tmp_path / "golden_traces_1_1.jsonl")
    b2.split()
    assert [e.source for e in b2.final["train"]] == ["max-golden"]


def test_a_failed_max_call_pins_its_group_under_a_policy(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("hotel", "bash", {"command": "make hotel"}),
        gym("india", "bash", {"command": "make india"})]), "extra")
    feed(b, write_jsonl(tmp_path / "fail.jsonl", [
        trace("hotel", "read_file", {"path": "missing.txt"})]), "fail")
    info = b.split()
    assert info["fail_used"] == 1 and info["fail_dropped_task_in_train_or_val"] == 0
    assert [e.trace_id for e in b.final["train"]] == ["gym.jsonl:2"]
    assert {e.trace_id for e in b.final["eval-tools"]} == {"gym.jsonl:1", "fail.jsonl:1"}


def test_train_source_rows_need_a_non_max_model_and_a_verification(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM, "*-gym-*"))
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        # a gym tag on a golden row still needs its verification: no tool-success fallback
        trace("kilo", "bash", {"command": "make kilo"}, model=GYM_MODEL, source=GYM)]))
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "bash", {"command": "a"}, model="qwen3.8-max"),       # Max under a gym tag
        gym("bravo", "bash", {"command": "b"}, model=""),                  # no model
        gym("charlie", "bash", {"command": "c"}, verification=None),       # unverified
        # a bare verdict is not a verification record: unverified
        gym("delta", "bash", {"command": "d"}, verification=None, verdict="pass"),
        # a tag naming Max that a broad pattern matches is still Max-derived: eval-only
        trace("echo", "bash", {"command": "e"}, source="max-gym-2026-11-01", verdict="pass"),
        # an untagged row of another model is nobody's teacher row
        trace("foxtrot", "bash", {"command": "f"}, model=GYM_MODEL, session="gym:f",
              verification=VERIFIED),
    ]), "extra")
    st = b.stats
    assert st["excluded:train_source_names_max"] == 1 and st["excluded:train_source_no_model"] == 1
    assert st["excluded:unverified"] == 3 and st["excluded:non_teacher_model"] == 1
    (e,) = b.examples
    assert e.trace_id == "gym.jsonl:5"
    assert not e.train_ok and e.max_derived and e.source == "max-gym-2026-11-01"


@pytest.mark.parametrize("record,decision,basis", [
    (None, None, "absent"),
    ("yes", False, "unreadable"),
    ({}, False, "no_decision"),
    ({"verified": False}, False, "not_verified"),
    ({"status": "failed"}, False, "not_verified"),
    ({"verified": "true"}, False, "no_decision"),          # not a bool, no status word
    # (read as a Max row's record: parts may be absent; a train-source row
    # needs every part, test_a_train_source_record_needs_every_part)
    ({"verified": True}, True, "verified"),
    ({"status": "pass"}, True, "verified"),
    ({"verdict": "PASS", "tool_success": True}, True, "tool-success"),
    ({"verified": True, "tool_success": False}, False, "tool_failed"),
    ({"verified": True, "judge": {"verdict": "fail"}}, False, "judge_not_pass"),
    ({"verified": True, "judge_verdict": "pass"}, True, "judge-pass"),
    ({"verified": True, "judge": "pass"}, True, "judge-pass"),
    ({"verified": True, "judge": {"verdict": "pass", "model": "qwen3.8-max"}}, False,
     "judge_is_max"),
    ({"verified": True, "judge_model": "Qwen3.8-MAX", "judge_verdict": "pass"}, False,
     "judge_is_max"),
    ({"verified": True, "judge": {"verdict": "pass", "model": "maximal-judge"}}, True,
     "judge-pass"),
    ({"verified": True, "sandboxed": False}, False, "not_sandboxed"),
    ({"verified": True, "side_effects": True}, False, "side_effects"),
    ({"verified": True, "real_side_effects": True}, False, "side_effects"),
    # a veto is any value but the safe one, not just the opposite bool
    ({"verified": True, "side_effects": ["wrote ~/.ssh/config"]}, False, "side_effects"),
    ({"verified": True, "side_effects": "true"}, False, "side_effects"),
    ({"verified": True, "side_effects": "false"}, False, "side_effects"),
    ({"verified": True, "side_effects": 1}, False, "side_effects"),
    ({"verified": True, "side_effects": 0}, False, "side_effects"),
    ({"verified": True, "side_effects": None}, False, "side_effects"),
    ({"verified": True, "real_side_effects": {"http": "POST"}}, False, "side_effects"),
    ({"verified": True, "side_effects": False, "real_side_effects": ["sent mail"]}, False,
     "side_effects"),
    ({"verified": True, "side_effects": []}, True, "verified"),
    ({"verified": True, "side_effects": {}, "real_side_effects": ""}, True, "verified"),
    ({"verified": True, "sandboxed": "false"}, False, "not_sandboxed"),
    ({"verified": True, "sandboxed": "true"}, False, "not_sandboxed"),
    ({"verified": True, "sandboxed": 0}, False, "not_sandboxed"),
    ({"verified": True, "sandboxed": 1}, False, "not_sandboxed"),
    ({"verified": True, "tool_success": "false"}, False, "tool_failed"),
    ({"verified": True, "tool_success": "true"}, False, "tool_failed"),
    ({"verified": True, "tool_success": 0}, False, "tool_failed"),
    ({"verified": True, "tool_success": None}, False, "tool_failed"),
    # a judge that never gave a verdict
    ({"verified": True, "judge": {"model": "qwen3.8-27b", "error": "timeout"}}, False,
     "judge_no_verdict"),
    ({"verified": True, "judge": {"verdict": "pass", "error": "timeout"}}, False,
     "judge_no_verdict"),
    ({"verified": True, "judge_model": "qwen3.8-27b"}, False, "judge_no_verdict"),
    ({"verified": True, "judge": True}, False, "judge_no_verdict"),
    ({"verified": True, "judge": {"verdict": "pass", "pass": False}}, False, "judge_not_pass"),
    ({"verified": True, "judge": {"verdict": "pass", "pass": "true"}}, False, "judge_not_pass"),
    ({"verified": True, "judge": {"verdict": "pass", "pass": True}}, True, "judge-pass"),
])
def test_the_verification_record_is_read_fail_closed(record: Any, decision: bool | None,
                                                     basis: str) -> None:
    meta = {} if record is None else {"verification": record}
    v = build.verification_of(meta)
    assert (v.decision, v.basis) == (decision, basis)



# The operator's rule for a train-source row (OPEN-QUESTIONS #6): tool success +
# a local judge, sandboxed tools only, no real-world side effects.
@pytest.mark.parametrize("record,basis", [
    ({k: v for k, v in VERIFIED.items() if k != "tool_success"}, "no_tool_success"),
    ({k: v for k, v in VERIFIED.items() if k != "judge"}, "no_judge"),
    ({**VERIFIED, "judge": "pass"}, "judge_no_model"),
    ({**VERIFIED, "judge": {"verdict": "pass", "model": " "}}, "judge_no_model"),
    ({**VERIFIED, "judge": {"verdict": "pass", "model": "qwen2.5:7b"}}, "judge_not_allowed"),
    ({**VERIFIED, "judge": {"verdict": "pass", "model": "qwen3.8-plus"}}, "judge_not_allowed"),
    ({**VERIFIED, "judge": {"verdict": "pass", "model": "qwen3.8-max"}}, "judge_is_max"),
    ({**VERIFIED, "judge": {"verdict": "pass", "model": GYM_MODEL,
                            "served_model": "qwen3.8-max"}}, "judge_is_max"),
    ({k: v for k, v in VERIFIED.items() if k != "sandboxed"}, "no_sandbox_record"),
    ({k: v for k, v in VERIFIED.items() if k != "side_effects"}, "no_side_effects_record"),
    # the reviewer's three: each would have trained
    ({"verified": True}, "no_tool_success"),
    ({"verified": True, "tool_success": True}, "no_judge"),
    ({"verified": True, "judge": "pass"}, "no_tool_success"),
])
def test_a_train_source_record_needs_every_part(record: dict[str, Any], basis: str) -> None:
    v = build.verification_of({"verification": record}, strict=True,
                              judge_models=build.DEFAULT_TRAIN_MODELS)
    assert (v.decision, v.basis) == (False, basis)


def test_a_complete_train_source_record_passes_strictly() -> None:
    models = build.DEFAULT_TRAIN_MODELS
    for rec in (VERIFIED,
                {**VERIFIED, "side_effects": [], "judge": {"verdict": "PASS", "pass": True,
                                                           "model": "Qwen3.8-27B"}},
                {k: v for k, v in VERIFIED.items() if k != "side_effects"}
                | {"real_side_effects": False}):
        v = build.verification_of({"verification": rec}, strict=True, judge_models=models)
        assert (v.decision, v.basis) == (True, "judge-pass")
    other = {**VERIFIED, "judge": {"verdict": "pass", "model": "qwen2.5:7b"}}
    assert build.verification_of({"verification": other}, strict=True,
                                 judge_models=("qwen2.5:*",)).decision is True


def test_partial_records_on_train_source_rows_never_train(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "bash", {"command": "a"}, verification={"verified": True}),
        gym("bravo", "bash", {"command": "b"},
            verification={"verified": True, "tool_success": True}),
        gym("charlie", "bash", {"command": "c"}, verification={"verified": True, "judge": "pass"}),
        gym("delta", "bash", {"command": "d"},
            verification={**VERIFIED, "side_effects": ["wrote ~/.ssh/config"]}),
        gym("echo", "bash", {"command": "e"}),
    ]), "extra")
    st = b.stats
    assert st["excluded:verification_failed"] == 4
    assert st["verification:no_tool_success"] == 2 and st["verification:no_judge"] == 1
    assert st["verification:side_effects"] == 1 and st["verification:pass"] == 1
    assert [e.trace_id for e in b.examples] == ["gym.jsonl:5"]
    b.split()
    assert [e.trace_id for e in b.final["train"]] == ["gym.jsonl:5"]
    assert b.gates()["data_policy"]["pass"]


def test_a_verdict_never_verifies_a_train_source_row_and_max_never_judges(
        tmp_path: Path) -> None:
    rows = [gym("alpha", "bash", {"command": "a"}, verification=None),
            gym("bravo", "bash", {"command": "b"}, verification=None, verdict="pass"),
            gym("charlie", "bash", {"command": "c"}, verification=None),
            gym("delta", "bash", {"command": "d"})]
    gym_file = write_jsonl(tmp_path / "gym.jsonl", rows)
    vfile = write_jsonl(tmp_path / "v.jsonl", [
        # judge.py rows (0-based line): alpha judged by the hosted Max route
        {"source": {"file": "gym.jsonl", "line": 0}, "status": "ok", "verdict": "pass",
         "label": "pass", "model": "qwen3.8-max"},
        # charlie judged locally, but a verdict is no verification record
        {"source": {"file": "gym.jsonl", "line": 2}, "status": "ok", "verdict": "pass",
         "label": "pass", "model": GYM_MODEL},
        # a Max fail still drops its row (dropping is always safe)
        {"trace_id": "gym.jsonl:4", "verdict": "fail", "judge_model": "Qwen3.8-MAX"},
    ])
    stats: collections.Counter = collections.Counter()
    verdicts = build.load_verdicts([str(vfile)], stats)
    assert verdicts == {"gym.jsonl:3": "pass", "gym.jsonl:4": "fail"}
    assert stats["verdict_rows_max_judge"] == 2
    b = make_builder(tmp_path, train_sources=(GYM,), verdicts=verdicts)
    b.stats.update(stats)
    feed(b, gym_file, "extra")
    assert b.examples == []
    assert b.stats["excluded:unverified"] == 3 and b.stats["excluded:judge_fail"] == 1
    b.split()
    gate = b.gates()["verdicts"]
    assert not gate["pass"] and gate["max_judge"] == 2 and gate["unmatched"] == 0
    # a local judge's verdict on a Max row still labels it (eval-only material)
    (tmp_path / "m").mkdir()
    golden = write_jsonl(tmp_path / "m" / "golden_traces_1_1.jsonl",
                         [trace("kilo", "bash", {"command": "make kilo"})])
    m = make_builder(tmp_path / "m", train_sources=(GYM,),
                     verdicts={"golden_traces_1_1.jsonl:1": "pass"})
    feed(m, golden)
    (k,) = m.examples
    assert k.label == "judge-pass" and k.max_derived and not k.train_ok


def test_train_source_rows_must_come_from_the_local_teacher(tmp_path: Path) -> None:
    rows = [gym("alpha", "bash", {"command": "a"}, model="qwen3.8-plus"),  # hosted, same key
            gym("bravo", "bash", {"command": "b"}, model="Qwen3.8-27B"),   # the teacher, any case
            gym("charlie", "bash", {"command": "c"}, served_model="qwen3.8-max"),
            gym("delta", "bash", {"command": "d"}, model="qwen3.8-27b-q4_k_m")]
    gym_file = write_jsonl(tmp_path / "gym.jsonl", rows)
    b = make_builder(tmp_path, train_sources=(GYM,))
    feed(b, gym_file, "extra")
    assert b.stats["excluded:train_source_wrong_model"] == 1
    assert b.stats["excluded:train_source_names_max"] == 1
    assert [e.trace_id for e in b.examples] == ["gym.jsonl:2", "gym.jsonl:4"]
    # the operator can widen the list, never to Max (the CLI refuses that)
    (tmp_path / "w").mkdir()
    w = make_builder(tmp_path / "w", train_sources=(GYM,),
                     train_models=("qwen3.8-27b*", "qwen3.8-plus"))
    feed(w, gym_file, "extra")
    assert [e.trace_id for e in w.examples] == ["gym.jsonl:1", "gym.jsonl:2", "gym.jsonl:4"]
    # and a wrong-model row that reached train/val fails the data_policy gate
    b.split()
    stray = ex(90, "other", "s0", max_derived=False, model="qwen3.8-plus")
    stray.source = GYM
    b.final["train"].append(stray)
    dp = b.gates()["data_policy"]
    assert not dp["pass"] and dp["unlisted_model_in_train_val"] == 1
    assert dp["max_derived_in_train_val"] == 0


def test_a_row_its_producer_marked_max_derived_never_trains(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "bash", {"command": "a"}, max_derived=True),   # seeded from a Max chat
        gym("bravo", "bash", {"command": "b"}, max_derived="false"),
        gym("charlie", "bash", {"command": "c"}, max_derived=None),
        gym("delta", "bash", {"command": "d"}, verification={**VERIFIED, "max_derived": 1}),
        gym("echo", "bash", {"command": "e"}, max_derived=False),
        gym("foxtrot", "bash", {"command": "f"}),
    ]), "extra")
    assert b.stats["excluded:train_source_max_derived"] == 4
    assert [e.trace_id for e in b.examples] == ["gym.jsonl:5", "gym.jsonl:6"]
    b.split()
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    assert {r["max_derived"] for r in read_jsonl(out / "train.jsonl")} == {False}


def test_the_flywheels_guard_clears_a_clean_policy_build_and_refuses_max(
        tmp_path: Path) -> None:
    if not (_DIR.parent.parent / "src" / "prometheus" / "jobs" / "flywheel_weekly.py").is_file():
        pytest.skip("flywheel_weekly.py not in this checkout")
    fw = pytest.importorskip("prometheus.jobs.flywheel_weekly")
    cfg = fw.parse_config(None)  # no max_data approval recorded

    def built(root: Path, *, policy: bool, models: tuple[str, ...] = (GYM_MODEL,)
              ) -> tuple[str, list[Path]]:
        root.mkdir()
        golden, gym_file = policy_inputs(root)
        rows = read_jsonl(gym_file)
        for i, r in enumerate(rows):
            r["_meta"]["model"] = models[i % len(models)]
        write_jsonl(gym_file, rows)
        b = make_builder(root, train_sources=(GYM,) if policy else None, eval_task_frac=0.15,
                         val_frac=0.05, holdout_max_group_share=0.5)
        feed(b, golden)
        feed(b, gym_file, "extra")
        info = b.split()
        out = root / "out"
        out.mkdir()
        b.write(out)
        report = b.report(info, b.gates(), catalog="stubs", head_tokens={}, inputs={},
                          elapsed=0.0)
        return report["settings"]["teacher_model"], [out / "train.jsonl", out / "val.jsonl"]

    teacher, paths = built(tmp_path / "gym", policy=True)
    assert teacher == GYM_MODEL
    g = fw.check_training_data(paths, cfg, teacher)
    assert g.rows > 0 and g.max_rows == 0 and g.refusals == []
    # without the key every row counted as Max-derived: clean data could never train
    assert fw.check_training_data(paths, cfg, None).max_rows == g.rows
    # a build that trains on Max rows (no policy) names the Max model: refused
    teacher, paths = built(tmp_path / "legacy", policy=False)
    assert teacher == build.TEACHER_MODEL
    g = fw.check_training_data(paths, cfg, teacher)
    assert g.rows > 0 and g.max_rows == g.rows and g.refusals
    # two teacher models cannot be named as one: the build fails closed
    teacher, paths = built(tmp_path / "mixed", policy=True, models=(GYM_MODEL, "qwen3.8-27b-q8"))
    assert teacher == build.TEACHER_MODEL


def test_a_failing_record_or_verdict_excludes_whatever_the_other_says(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,),
                     verdicts={"gym.jsonl:2": "fail", "gym.jsonl:3": "pass"})
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "bash", {"command": "a"}, verdict="pass",
            verification={"verified": True, "tool_success": False}),
        gym("bravo", "bash", {"command": "b"}),                          # judged fail
        gym("charlie", "bash", {"command": "c"},
            verification={"verified": True, "judge": {"verdict": "fail"}}),  # verdict pass
        gym("delta", "bash", {"command": "d"}),
    ]), "extra")
    assert b.stats["excluded:verification_failed"] == 2 and b.stats["excluded:judge_fail"] == 1
    assert b.stats["verification:tool_failed"] == 1 and b.stats["verification:judge_not_pass"] == 1
    (d,) = b.examples
    assert d.trace_id == "gym.jsonl:4" and d.label == "judge-pass"


def test_a_failed_attempt_does_not_take_a_verified_retrys_dedupe_key(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,))
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "bash", {"command": "ls"}, verification={"verified": False}),
        gym("alpha", "bash", {"command": "ls"}),           # the retry, same first call
        gym("alpha", "bash", {"command": "ls"}),           # a true duplicate of the retry
    ]), "extra")
    assert [e.trace_id for e in b.examples] == ["gym.jsonl:2"]
    assert b.stats["excluded:verification_failed"] == 1 and b.stats["excluded:duplicate"] == 1
    # a Max row still claims its key when read: a verdict on it is a verdict on the call
    m = make_builder(tmp_path, verdicts={"golden_traces_1_1.jsonl:1": "fail"})
    feed(m, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "bash", {"command": "ls"}), trace("alpha", "bash", {"command": "ls"})]))
    assert m.examples == [] and m.stats["excluded:judge_fail"] == 1
    assert m.stats["excluded:duplicate"] == 1


def test_generator_task_ids_join_rows_into_one_group(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM, "other-gym"))
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym("alpha", "bash", {"command": "a"}, task_id="t-7"),
        gym("bravo", "bash", {"command": "b"}, task_id="t-7"),      # another prompt, same task
        gym("charlie", "bash", {"command": "c"}, task_id="t-8"),
        gym("delta", "bash", {"command": "d"}, task_id="t-7", source="other-gym"),  # other source
    ]), "extra")
    build.assign_groups(b.examples)
    a, bb, c, d = b.examples
    assert a.task_id != bb.task_id and a.group == bb.group
    assert c.group != a.group and d.group != a.group
    res = build.leakage({"train": [a], "val": [], "eval-tools": [bb]})
    assert not res["pass"] and res["overlaps"]["train|eval-tools"]["source_task"] == 1


def test_chat_mode_gym_rows_render_without_tools_and_count_for_presence(tmp_path: Path) -> None:
    prose = gym("bravo", "bash", {}, category="other")   # tools offered, answered in prose
    prose["messages"][-1] = {"role": "assistant", "content": "Nothing to run; it is done."}
    prose["_meta"].pop("tool_name")
    called = gym_chat("charlie")
    called["messages"][-1]["tool_calls"] = [call("bash", {"command": "ls"})]
    uncategorized = gym_chat("delta")
    uncategorized["_meta"].pop("category")
    plain = gym("echo", "bash", {}, category="other")     # tools offered, no category, prose
    plain["messages"][-1] = {"role": "assistant", "content": "Done already."}
    plain["_meta"].pop("category")
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "gym.jsonl",
                        [gym_chat("alpha"), prose, called, uncategorized, plain]), "extra")
    assert b.stats["excluded:call_in_chat_mode"] == 1
    by_id = {e.trace_id: e for e in b.examples}
    chat, fa, unc, pl = (by_id[f"gym.jsonl:{i}"] for i in (1, 2, 4, 5))
    assert chat.flags["chat_mode"] and not chat.flags["final_answer"] and chat.expect_tool is None
    assert unc.flags["chat_mode"] and unc.category == "chat-mode"
    assert not fa.flags["chat_mode"] and fa.flags["final_answer"]
    assert not pl.flags["chat_mode"] and pl.category == "other"  # not chat mode: tools offered
    heavy = b.read_heavy(chat)
    text = heavy["render"]["text"]
    # the runtime's chat-mode request: no tools, no tool_choice; the plain system turn
    assert heavy["tools"] == [] and heavy["tool_choice"] is None
    assert text.startswith("<|im_start|>system\nYou are a test agent.") and "<tools>" not in text
    a, z = heavy["render"]["loss_spans_char"][0]
    assert text[a:z] == "Here is the plain answer.<|im_end|>"
    assert text[:a].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text == R.render(b.student.template, heavy["messages"], None,
                            add_generation_prompt=False)
    fa_heavy = b.read_heavy(fa)
    assert "<tools>" in fa_heavy["render"]["text"] and fa_heavy["tool_choice"] == "auto"
    assert fa_heavy["expect"] == {"tool": None}
    b.split()
    gates = b.gates()
    assert gates["presence"]["chat_mode"] == 2 and gates["data_policy"]["pass"]


def test_size_and_presence_gates_count_the_train_sources_only(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,))
    b.s.train_range, b.s.min_presence = (3, 3), 1
    gym_rows = [ex(i, "other", f"g{i}", max_derived=False, model=GYM_MODEL) for i in range(3)]
    for e in gym_rows:
        e.source = GYM
    gym_rows[0].flags["compacted"] = True
    leaked = [ex(10 + i, "coding", f"m{i}") for i in range(2)]  # Max rows, never train_ok
    for e in leaked:
        e.train_ok = False
        e.flags.update(chat_mode=True, error_recovery=True)
    b.final = {"train": gym_rows + leaked, "val": [], "eval-tools": []}
    gates = b.gates()
    assert gates["train_size"]["value"] == 3 and gates["train_size"]["pass"]
    assert gates["train_size"]["by_source"] == {GYM: 3}
    assert gates["presence"]["chat_mode"] == 0 and gates["presence"]["compacted"] == 1
    assert not gates["presence"]["pass"]
    # and Max rows in train/val fail the data_policy gate (and so the build)
    assert not gates["data_policy"]["pass"]
    assert gates["data_policy"]["max_derived_in_train_val"] == 2
    assert gates["data_policy"]["outside_train_sources_in_train_val"] == 2
    assert gates["overall"] == "FAIL"
    b.final["train"] = gym_rows
    assert b.gates()["data_policy"]["pass"]


def test_spot_sample_holds_a_chat_row_and_a_tool_row() -> None:
    rows = [{"id": f"r{i}", "flags": {"chat_mode": False}} for i in range(20)]
    rows.append({"id": "chat", "flags": {"chat_mode": True}})
    picked = build.spot_sample(rows, 3, "spot-v1")
    assert len(picked) == 3 and sum(build.is_chat_row(r) for r in picked) == 1
    assert build.spot_sample(rows[:2], 5, "s") == sorted(
        rows[:2], key=lambda r: build.seeded("s", r["id"]))
    only_chat = [{"id": "c", "suppress_tools": True}]
    assert build.spot_sample(only_chat, 5, "s") == only_chat
    assert build.spot_sample(rows, 0, "s") == []


def test_spot_check_sends_a_chat_row_the_way_the_runtime_does(tmp_path: Path) -> None:
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "gym.jsonl", [
        gym_chat("alpha"), gym("bravo", "bash", {"command": "make bravo"})]), "extra")
    b.split()
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    bodies: list[dict[str, Any]] = []

    def honest(url: str, body: dict[str, Any]) -> dict[str, Any]:
        bodies.append(body)
        return {"prompt": R.render(b.student.template, body["messages"], body.get("tools"),
                                   add_generation_prompt=body["add_generation_prompt"],
                                   template_kwargs=body.get("chat_template_kwargs"))}

    res = build.spot_check_rows("http://x", out / "train.jsonl", b.student, n=2, post=honest)
    assert len(res) == 2 and all(r["match"] for r in res)
    assert {r["completion_match"] is True for r in res} == {True, False}  # a chat row has no call
    chat_bodies = [x for x in bodies if "tools" not in x]
    assert chat_bodies and all("tool_choice" not in x for x in chat_bodies)


def test_cli_build_applies_the_default_train_source_policy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    golden, gym_file = policy_inputs(tmp_path)
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(make_pack()))
    monkeypatch.setattr(R.Student, "load", classmethod(lambda cls, d: make_student()))

    def run(*extra: str) -> int:
        return build.main(["build", "--traces", str(golden), "--extra-traces", str(gym_file),
                           "--pack", str(pack), "--tokenizer-dir", "unused",
                           "--out", str(tmp_path / "out"), "--dry-run", "--eval-min", "1",
                           "--train-range", "1", "1000", "--val-share-range", "0", "1",
                           "--eval-task-frac", "0.15", "--holdout-max-group-share", "0.5",
                           "--min-presence", "0", *extra])

    assert run() == 1  # INCOMPLETE: no spot check server
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["settings"]["train_sources"] == list(build.DEFAULT_TRAIN_SOURCES)
    assert report["settings"]["max_model"] == build.TEACHER_MODEL
    assert report["settings"]["train_models"] == list(build.DEFAULT_TRAIN_MODELS)
    assert report["settings"]["teacher_model"] == GYM_MODEL
    g = report["gates"]
    assert g["data_policy"]["pass"] and g["train_size"]["by_source"] == {
        GYM: report["splits"]["train"]["rows"]}
    assert report["splits"]["train"]["source"] == {GYM: report["splits"]["train"]["rows"]}
    assert report["data_policy"]["sources"]["max-golden"]["train"] == 0
    assert g["overall"] == "INCOMPLETE"
    capsys.readouterr()
    # the old flag name still sets the Max model; a source naming Max is refused
    assert run("--teacher-model", "some-other-max") == 1
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["settings"]["max_model"] == "some-other-max"
    assert report["data_policy"]["sources"]["max-golden"]["excluded_by_reason"][
        "non_teacher_model"] > 0
    capsys.readouterr()
    for flag, bad in (("--train-sources", ["max-golden"]), ("--train-sources", ["max-gym-*"]),
                      ("--train-sources", [GYM, "Qwen3.8-MAX"]),
                      ("--train-models", ["qwen3.8-max"]), ("--train-models", ["qwen3.8-*", "MAX"]),
                      ("--judge-models", ["qwen3.8-max"])):
        assert run(flag, *bad) == 2
        assert f"{flag} may not name Max" in capsys.readouterr().err
    for flag in ("--train-sources", "--train-models", "--judge-models"):
        assert run(flag, " ") == 2
    # a train model the gym rows do not match: nothing trains
    assert run("--train-models", "other-27b*") == 1
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["splits"]["train"]["rows"] == 0
    assert report["data_policy"]["sources"][GYM]["excluded_by_reason"] == {
        "train_source_wrong_model": 88}
    assert report["settings"]["teacher_model"] == build.TEACHER_MODEL  # fails closed


# ── Q6b: the explicit, recorded allow for Max-derived rows to train ────────

AP = build.AP
REF = "operator Q6b 2026-10-30"


def allow_builder(root: Path, **kw: Any) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    settings = {"train_sources": (GYM,), "eval_task_frac": 0.15, "val_frac": 0.05,
                "holdout_max_group_share": 0.5, "val_share_range": (0.0, 1.0),
                "allow_max_derived": True, "max_derived_approval": REF, **kw}
    return make_builder(root, **settings)


def test_q6b_dataset_manifest_name_is_the_policy_modules() -> None:
    assert build.DATASET_MANIFEST == AP.DATASET_MANIFEST
    assert build.REPORT_SCHEMA == AP.REPORT_SCHEMA


def test_q6b_allow_needs_both_flags_and_a_tagged_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    golden, gym_file = policy_inputs(tmp_path)
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(make_pack()))
    monkeypatch.setattr(R.Student, "load", classmethod(lambda cls, d: make_student()))

    def run(out: Path, *extra: str) -> int:
        return build.main(["build", "--traces", str(golden), "--extra-traces", str(gym_file),
                           "--pack", str(pack), "--tokenizer-dir", "unused", "--out", str(out),
                           "--dry-run", "--eval-min", "1", "--train-range", "1", "1000",
                           "--val-share-range", "0", "1", "--eval-task-frac", "0.15",
                           "--holdout-max-group-share", "0.5", "--min-presence", "0", *extra])

    plain, tagged = tmp_path / "phase3", tmp_path / "phase3-max-derived"
    sessions = tmp_path / "exclude-sessions.txt"
    sessions.write_text("# the 2026-10-13 Token-Plan incident session\nincident:s9\n")
    excl = ("--exclude-sessions", str(sessions))
    for extra in (["--allow-max-derived"], ["--max-derived-approval", REF]):
        assert run(tagged, *extra, *excl) == 2
        assert "go together" in capsys.readouterr().err
    assert run(tagged, "--allow-max-derived", "--max-derived-approval", "  ", *excl) == 2
    assert "not blank" in capsys.readouterr().err
    assert run(plain, "--allow-max-derived", "--max-derived-approval", REF, *excl) == 2
    assert "phase3-max-derived" in capsys.readouterr().err  # the tagged name it should use
    assert run(tagged, "--allow-max-derived", "--max-derived-approval", REF, *excl,
               "--report", str(tmp_path / "report.json")) == 2
    # the incident session's exclusion list is required with the allow, and not empty
    assert run(tagged, "--allow-max-derived", "--max-derived-approval", REF) == 2
    assert "--exclude-sessions" in capsys.readouterr().err
    empty = tmp_path / "none.txt"
    empty.write_text("# nothing listed\n\n")
    assert run(tagged, "--allow-max-derived", "--max-derived-approval", REF,
               "--exclude-sessions", str(empty)) == 2
    assert "lists no session" in capsys.readouterr().err
    assert not plain.exists() and not tagged.exists()
    # without the allow Max rows stay eval-only
    assert run(plain) == 1
    report = json.loads((plain / "report.json").read_text())
    assert report["max_derived"] is False and report["tags"] == [] and not report["private"]
    assert report["holds_max_rows"] is True  # eval-tools is Max material
    assert report["data_policy"]["max_derived_allow"]["allowed"] is False
    assert report["splits"]["train"]["max_derived"] == 0
    assert report["data_policy"]["share"]["by_split"]["train"]["max"]["rows"] == 0
    capsys.readouterr()
    # with both: logged, recorded verbatim, and the dataset is tagged
    assert run(tagged, "--allow-max-derived", "--max-derived-approval", REF, *excl) == 1  # INCOMPLETE
    err = capsys.readouterr().err
    assert "ALLOW (Q6b)" in err and REF in err
    report = json.loads((tagged / "report.json").read_text())
    assert report["max_derived"] is True and report["tags"] == ["max-derived"]
    assert report["settings"]["max_derived_approval"] == REF
    assert report["settings"]["allow_max_derived"] is True
    allow = report["data_policy"]["max_derived_allow"]
    assert allow["allowed"] and allow["approval"] == REF
    assert allow["flags"] == [AP.BUILD_ALLOW_FLAG, AP.BUILD_APPROVAL_FLAG]
    assert allow["excluded_sessions_listed"] == 1 and allow["excluded_session_rows"] == 0
    assert report["settings"]["exclude_sessions"] == 1 and "incident:s9" not in json.dumps(report)
    assert allow["rows_in_train_val"] == allow["covered_in_train_val"] > 0
    assert report["gates"]["data_policy"]["pass"]
    assert report["settings"]["teacher_model"] == build.TEACHER_MODEL  # the guards fail closed
    assert AP.dataset_approval(report) == REF
    assert not (tagged / build.DATASET_MANIFEST).exists()  # a dry run writes the report only


def test_q6b_max_rows_train_under_the_same_gates_with_shares(tmp_path: Path) -> None:
    golden, gym_file = policy_inputs(tmp_path)
    b = allow_builder(tmp_path / "b")
    feed(b, golden)
    feed(b, gym_file, "extra")
    assert b.max_allowed
    info = b.split()
    gates = b.gates()
    train_val = b.final["train"] + b.final["val"]
    max_tv = [e for e in train_val if e.max_derived]
    gym_tv = [e for e in train_val if e.source == GYM]
    assert max_tv and gym_tv and len(max_tv) + len(gym_tv) == len(train_val)
    assert all(e.approval == REF and e.model == build.TEACHER_MODEL and e.train_ok
               and e.kind == "golden" for e in max_tv)
    assert all(not e.max_derived and e.approval == "" for e in gym_tv)
    # the same gates: leakage across splits, the category cap over the mix, data policy
    assert gates["leakage"]["pass"] and gates["category_cap"]["pass"]
    dp = gates["data_policy"]
    assert dp["pass"] and dp["max_derived_allowed"] and dp["max_derived_approval"] == REF
    assert dp["max_derived_covered_in_train_val"] == len(max_tv) == dp["max_derived_in_train_val"]
    assert dp["max_derived_uncovered_in_train_val"] == 0
    assert dp["outside_train_sources_in_train_val"] == 0
    assert gates["train_size"]["by_source"]["max-golden"] > 0  # Max rows count toward size
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    assert report["max_derived"] is True and report["tags"] == ["max-derived"]
    assert report["private"] is True and report["holds_max_rows"] is True
    pol = report["data_policy"]
    assert pol["rule"].startswith(build.POLICY_RULE) and build.MAX_ALLOW_RULE in pol["rule"]
    share = pol["share"]
    assert share["approval"] == REF and share["max_derived_allowed"] is True
    for split, rows in b.final.items():
        s = share["by_split"][split]
        assert s["rows"] == len(rows)
        assert sum(s[c]["rows"] for c in build.SHARE_CLASSES) == len(rows)
        if rows:
            assert abs(sum(s[c]["pct"] for c in build.SHARE_CLASSES) - 100.0) < 0.05
    tr = share["by_split"]["train"]
    assert tr["max"]["rows"] == sum(1 for e in b.final["train"] if e.max_derived) > 0
    assert tr[GYM]["rows"] == sum(1 for e in b.final["train"] if e.source == GYM) > 0
    mg, gs = pol["sources"]["max-golden"], pol["sources"][GYM]
    assert mg["class"] == "max" and gs["class"] == GYM
    assert mg["share"]["train"] == tr["max"]["pct"] and gs["share"]["train"] == tr[GYM]["pct"]
    assert mg["max_derived_allowed"] == mg["accepted"] and gs["max_derived_allowed"] == 0
    blob = json.dumps(report)
    assert "please handle" not in blob and "git status" not in blob
    # written rows carry the flag and the approval; the trainer reads both
    out = tmp_path / "phase3-max-derived"
    out.mkdir()
    written = b.write(out)
    (out / "report.json").write_text(json.dumps(report))
    for row in read_jsonl(out / "train.jsonl") + read_jsonl(out / "val.jsonl"):
        if row["source"] == GYM:
            assert row["max_derived"] is False and row["max_derived_approval"] is None
        else:
            assert row["max_derived"] is True and row["max_derived_approval"] == REF
    m = b.dataset_manifest(out, written, report, out / "report.json")
    assert m["schema"] == AP.DATASET_MANIFEST_SCHEMA and m["max_derived"] is True
    assert m["tags"] == ["max-derived"] and m["max_derived_allow"]["approval"] == REF
    assert m["files"]["train.jsonl"]["rows"] == len(b.final["train"])
    assert len(m["files"]["train.jsonl"]["sha256"]) == 64 and m["report"] == "report.json"
    if (_DIR / "train_qlora.py").is_file():
        T = _load("distill_train_qlora", "train_qlora.py")
        assert T.builder_teacher_is_max(out / "train.jsonl") is True
        assert T.dataset_record(out / "train.jsonl")["approval"] == REF


def test_q6b_the_max_only_holdout_and_listed_tasks_stay_eval_tools(tmp_path: Path) -> None:
    golden, gym_file = policy_inputs(tmp_path)
    # what a Max-only build (the Phase 3 build) holds out
    (tmp_path / "a").mkdir()
    b0 = make_builder(tmp_path / "a", eval_task_frac=0.15, val_frac=0.05,
                      holdout_max_group_share=0.5)
    feed(b0, golden)
    b0.split()
    held0 = {e.task_id for e in b0.examples if e.split == "eval-tools"}
    assert held0
    b1 = allow_builder(tmp_path / "b")
    feed(b1, golden)
    feed(b1, gym_file, "extra")
    info = b1.split()
    tv = {e.task_id for k in ("train", "val") for e in b1.final[k]}
    held1 = {e.task_id for e in b1.examples if e.split == "eval-tools"}
    assert held0 <= held1 and not held0 & tv  # still eval data, never trained
    assert info["plan"].max_holdout_tasks == len(held0)
    assert any(e.source == GYM for e in b1.final["eval-tools"])  # the quota comes from gym
    assert any(e.max_derived for e in b1.final["train"])
    # an earlier eval set's tasks (--eval-holdout) stay eval-tools too
    max_t = next(e.task_id for e in b1.final["train"] if e.max_derived)
    gym_t = next(e.task_id for e in b1.final["train"] if e.source == GYM)
    cases = write_jsonl(tmp_path / "eval-tools.jsonl", [{"meta": {"task_id": max_t}},
                                                         {"task_id": gym_t}, {"meta": {}}])
    tasks = build.read_holdout_tasks(cases)
    assert tasks == {max_t, gym_t}
    with pytest.raises(ValueError):
        build.read_holdout_tasks(write_jsonl(tmp_path / "none.jsonl", [{"meta": {}}]))
    b2 = allow_builder(tmp_path / "c")
    b2.eval_holdout_tasks = frozenset(tasks)
    feed(b2, golden)
    feed(b2, gym_file, "extra")
    info2 = b2.split()
    split_of = {e.task_id: e.split for e in b2.examples}
    assert split_of[max_t] == split_of[gym_t] == "eval-tools"
    assert info2["plan"].listed_holdout_groups == 2
    report = b2.report(info2, b2.gates(), catalog="stubs", head_tokens={}, inputs={}, elapsed=0)
    assert report["eval_tools"]["eval_holdout_groups"] == 2
    assert report["eval_tools"]["eval_holdout_tasks_listed"] == 2
    assert max_t not in json.dumps(report)  # counts only


def test_q6b_an_uncovered_max_row_fails_the_data_policy_gate(tmp_path: Path) -> None:
    b = allow_builder(tmp_path / "b")
    gym_rows = [ex(i, "other", f"g{i}", max_derived=False, model=GYM_MODEL) for i in range(2)]
    for e in gym_rows:
        e.source = GYM
    covered = ex(10, "coding", "m0", model=build.TEACHER_MODEL, approval=REF)
    for bad, field_, value in ((None, None, None), ("approval", "approval", ""),
                               ("approval", "approval", "someone else"),
                               ("model", "model", GYM_MODEL), ("kind", "kind", "fail"),
                               ("train_ok", "train_ok", False)):
        row = ex(11, "coding", "m1", model=build.TEACHER_MODEL, approval=REF)
        if field_:
            setattr(row, field_, value)
        b.final = {"train": gym_rows + [covered, row], "val": [], "eval-tools": []}
        dp = b.gates()["data_policy"]
        assert dp["max_derived_in_train_val"] == 2
        if bad is None:
            assert dp["pass"] and dp["max_derived_covered_in_train_val"] == 2
        else:
            assert not dp["pass"] and dp["max_derived_uncovered_in_train_val"] == 1, bad
    # the same covered row without the allow fails
    b.s.allow_max_derived = False
    b.final = {"train": gym_rows + [covered], "val": [], "eval-tools": []}
    dp = b.gates()["data_policy"]
    assert not dp["pass"] and dp["max_derived_uncovered_in_train_val"] == 1
    assert dp["max_derived_allowed"] is False and dp["max_derived_approval"] is None


def test_q6b_failing_verdicts_unverified_extras_and_fails_never_train(tmp_path: Path) -> None:
    b = allow_builder(tmp_path / "b", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                      verdicts={"golden_traces_1_1.jsonl:1": "fail",
                                "golden_traces_1_1.jsonl:2": "pass"})
    feed(b, write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "bash", {"command": "make alpha"}),   # judged fail: excluded
        trace("bravo", "bash", {"command": "make bravo"}),   # judged pass
        trace("charlie", "bash", {"command": "make charlie"}),   # tool-success label
        trace("delta", "bash", {"command": "make delta"},
              verification={"verified": True, "judge": {"verdict": "fail", "model": GYM_MODEL}}),
    ]))
    tool_ok = {"verified": True, "tool_success": True}
    feed(b, write_jsonl(tmp_path / "maxgym.jsonl", [
        trace("echo", "bash", {"command": "make echo"}, source="max-gym-2026-11-01",
              session="gym:echo", verification=VERIFIED),                      # trains
        trace("foxtrot", "bash", {"command": "make foxtrot"}, source="max-gym-2026-11-01",
              session="gym:foxtrot", verdict="pass"),                          # eval-only
        trace("golf", "bash", {"command": "make golf"}, source="max-gym-2026-11-01",
              session="gym:golf", verification={"verified": True, "tool_success": False}),
        trace("hotel", "bash", {"command": "make hotel"}, source="max-gym-2026-11-01",
              session="gym:hotel", verification={**tool_ok, "judge": {
                  "verdict": "pass", "model": "qwen3.8-max"}}),                # Max judge
    ]), "extra")
    feed(b, write_jsonl(tmp_path / "fail.jsonl", [
        trace("india", "read_file", {"path": "missing.txt"})]), "fail")
    assert b.stats["excluded:judge_fail"] == 1
    assert b.stats["excluded:verification_failed"] == 3  # delta, golf, hotel
    assert b.stats["max_derived_eval_only:no_record"] == 1  # foxtrot: a verdict alone
    by_id = {e.trace_id: e for e in b.examples + b.fails}
    assert by_id["golden_traces_1_1.jsonl:2"].label == "judge-pass"
    assert by_id["golden_traces_1_1.jsonl:3"].label == "tool-success"
    assert by_id["maxgym.jsonl:1"].train_ok and by_id["maxgym.jsonl:1"].approval == REF
    assert not by_id["maxgym.jsonl:2"].train_ok and by_id["maxgym.jsonl:2"].approval == ""
    assert not by_id["fail.jsonl:1"].train_ok and by_id["fail.jsonl:1"].approval == ""
    b.split()
    trained = {e.trace_id for k in ("train", "val") for e in b.final[k]}
    assert trained == {"golden_traces_1_1.jsonl:2", "golden_traces_1_1.jsonl:3", "maxgym.jsonl:1"}
    assert by_id["maxgym.jsonl:2"].split == "eval-tools"  # an eval-only row pins its group
    assert "fail.jsonl:1" not in trained
    assert b.gates()["data_policy"]["pass"]


class _TinyBackend:
    """The least train_qlora.Trainer needs: one float of weights, no torch."""

    def __init__(self) -> None:
        self.w = 0.0

    def load(self, max_seq: int, revision: Any) -> None:
        pass

    def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
        return 2.0 / (1.0 + self.w)

    def step(self, lr: float) -> None:
        self.w += lr

    def zero_grad(self) -> None:
        pass

    def eval_loss(self, ids: Any, spans: Any) -> float:
        return 1.0

    def save(self, d: Path) -> None:
        (d / "adapter").mkdir(parents=True)
        (d / "adapter" / "w.json").write_text(json.dumps({"w": self.w}))

    def restore(self, d: Path) -> None:
        self.w = json.loads((d / "adapter" / "w.json").read_text())["w"]

    def rng_state(self) -> Any:
        return 0

    def set_rng_state(self, state: Any) -> None:
        pass

    def save_rng(self, path: Path, state: Any) -> None:
        path.write_text("0")

    def load_rng(self, path: Path) -> Any:
        return 0

    def save_adapter(self, d: Path) -> None:
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"\0")

    def fingerprint(self) -> float:
        return self.w

    def peak_memory_gb(self) -> float | None:
        return None

    def free(self) -> None:
        pass

    def versions(self) -> dict[str, Any]:
        return {"fake": "1"}


def test_q6b_tags_flow_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed dataset (Max + local 27B gym) built under the allow -> a run on it
    -> its adapter -> a quantized model beside it: every level is tagged and the
    publish guard refuses every one. The same chain on a gym-only build is
    untagged, and its model publishes (its dataset still holds Max eval rows)."""
    if not (_DIR / "train_qlora.py").is_file():
        pytest.skip("train_qlora.py not in this checkout")
    T = _load("distill_train_qlora", "train_qlora.py")
    golden, gym_file = policy_inputs(tmp_path)
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(make_pack()))
    monkeypatch.setattr(R.Student, "load", classmethod(lambda cls, d: make_student()))
    monkeypatch.setattr(T, "hf_encoder", lambda d: FakeEncoder())

    def no_template(d: Any) -> Any:
        raise ImportError("jinja2")

    monkeypatch.setattr(T, "template_refitter", no_template)
    monkeypatch.setattr(T, "resolve_revision", lambda repo, rev: "b" * 40)
    tok = tmp_path / "tok"
    tok.mkdir()

    def build_set(out: Path, *extra: str) -> None:
        rc = build.main(["build", "--traces", str(golden), "--extra-traces", str(gym_file),
                         "--pack", str(pack), "--tokenizer-dir", "unused", "--out", str(out),
                         "--eval-min", "1", "--train-range", "1", "1000",
                         "--val-share-range", "0", "1", "--eval-task-frac", "0.15",
                         "--holdout-max-group-share", "0.5", "--min-presence", "0", *extra])
        assert rc == 1  # INCOMPLETE (no spot-check server); everything is written

    def train(data: Path, run: Path, *extra: str) -> int:
        return T.main(["--data", str(data / "train.jsonl"), "--tokenizer-dir", str(tok),
                       "--out", str(run), "--log-dir", str(run.parent / (run.name + "-logs")),
                       "--grad-accum", "4", "--save-every", "1000", *extra],
                      backend_factory=lambda c: _TinyBackend())

    def quantized(adapter: Path, q: Path) -> Path:
        (q / "merged").mkdir(parents=True)
        (q / "merged" / "model.safetensors").write_bytes(b"\0")
        (q / "student-Q4_K_M.gguf").write_bytes(b"GGUF")
        (q / "quantize-manifest.json").write_text(json.dumps({
            "schema": AP.QUANTIZE_MANIFEST_SCHEMA, "inputs": {"adapter": {"dir": str(adapter)}},
            "max_derived": {"calibration_rows": 4, "calibration_max_derived": 0}}))
        return q

    # ── the mixed, approved chain ──
    ds = tmp_path / "phase3-max-derived"
    sessions = tmp_path / "exclude-sessions.txt"
    sessions.write_text("incident:s9\n")
    build_set(ds, "--allow-max-derived", "--max-derived-approval", REF,
              "--exclude-sessions", str(sessions))
    rep = json.loads((ds / "report.json").read_text())
    man = json.loads((ds / build.DATASET_MANIFEST).read_text())
    for doc in (rep, man):
        assert doc["max_derived"] is True and doc["tags"] == ["max-derived"] and doc["private"]
    assert man["max_derived_allow"]["approval"] == REF
    assert man["share"]["by_split"]["train"]["max"]["rows"] > 0
    assert man["share"]["by_split"]["train"][GYM]["rows"] > 0  # a mixed dataset
    assert set(man["files"]) >= {"train.jsonl", "val.jsonl", "eval-tools.jsonl"}
    run = tmp_path / "runs" / "9b-r32-max-derived"
    assert train(ds, run) == 3  # the dataset's approval alone is not enough
    untagged = tmp_path / "runs" / "9b-r32"
    assert train(ds, untagged, T.APPROVAL_FLAG) == 2 and not untagged.exists()  # name it
    assert train(ds, run, T.APPROVAL_FLAG) == 0
    adapter = run / "adapter-max-derived"
    assert (adapter / "adapter_config.json").is_file() and not (run / "adapter").exists()
    rm = json.loads((run / "manifest.json").read_text())
    assert rm["max_derived"] is True and rm["tags"] == ["max-derived"]
    assert rm["approvals"]["dataset"]["approval"] == REF and rm["data"]["max_derived_rows"] > 0
    q = quantized(adapter, tmp_path / "quant" / "9b-r32-q4")
    levels = [ds, ds / "report.json", ds / build.DATASET_MANIFEST, ds / "train.jsonl",
              ds / "eval-tools.jsonl", run, run / "manifest.json", adapter,
              adapter / "adapter_model.safetensors", q, q / "quantize-manifest.json",
              q / "student-Q4_K_M.gguf", q / "merged"]
    for level in levels:
        assert AP.is_max_derived(level) or AP.provenance(level).holds_max_rows, level
        with pytest.raises(AP.NotPublishable):
            AP.assert_publishable(level)
    # the quantized model is Max-derived through its adapter, not its name
    assert AP.provenance(q / "student-Q4_K_M.gguf").max_derived is True
    assert AP.provenance(rm).max_derived is True  # a manifest dict, read the same way

    # ── --report elsewhere: the dataset manifest in --out carries the approval ──
    ds2 = tmp_path / "phase3b-max-derived"
    rep2 = tmp_path / "reports-max-derived" / "report.json"
    rep2.parent.mkdir()
    build_set(ds2, "--allow-max-derived", "--max-derived-approval", REF,
              "--exclude-sessions", str(sessions), "--report", str(rep2))
    assert rep2.is_file() and not (ds2 / "report.json").exists()
    assert T.dataset_record(ds2 / "train.jsonl")["approval"] == REF
    assert T.builder_teacher_is_max(ds2 / "train.jsonl") is True
    run2 = tmp_path / "runs" / "9b-b-max-derived"
    assert train(ds2, run2, T.APPROVAL_FLAG) == 0
    assert json.loads((run2 / "manifest.json").read_text())["approvals"]["dataset"][
        "recorded_in"] == str(ds2 / build.DATASET_MANIFEST)

    # ── the gym-only chain (no allow) ──
    clean = tmp_path / "phase3"
    build_set(clean)
    crep = json.loads((clean / "report.json").read_text())
    assert crep["max_derived"] is False and crep["holds_max_rows"] is True
    crun = tmp_path / "runs" / "9b-gym"
    assert train(clean, crun) == 0
    cadapter = crun / "adapter"
    assert (cadapter / "adapter_config.json").is_file()
    assert json.loads((crun / "manifest.json").read_text())["tags"] == []
    cq = quantized(cadapter, tmp_path / "quant" / "9b-gym-q4")
    for level in (clean / "train.jsonl", crun, cadapter, cq, cq / "student-Q4_K_M.gguf"):
        assert AP.provenance(level).max_derived is False, level
        AP.assert_publishable(level)
    for level in (clean, clean / "report.json", clean / "eval-tools.jsonl"):
        with pytest.raises(AP.NotPublishable):  # Max eval rows: still private
            AP.assert_publishable(level)


# ── Q6b reviewer findings: never-train tags, strict gym records, the PAYG
#    judge, the incident session ──

MAXGYM = "max-gym-2026-11-01"


def _payg_judge() -> dict[str, Any]:
    """A judge object exactly as judge.py writes it for the pay-per-token Max judge."""
    judge = _load("distill_judge", "judge.py")
    rec = judge.verification_record({"verdict": "pass"}, model="qwen3.8-max",
                                    served_model="qwen3.8-max", provider="max-payg")
    return dict(rec["judge"])


def test_q6b_a_max_model_smoke_or_relaxed_row_never_trains(tmp_path: Path) -> None:
    rows = [
        # a Max-mode teacher_gen --builtin-smoke row, fully verified
        trace("alpha", "bash", {"command": "make alpha"}, source="builtin-smoke",
              session="gym:alpha", verification=VERIFIED),
        # a relaxed row from the Max model, its record claiming every part
        trace("bravo", "bash", {"command": "make bravo"}, source="qwen38-27b-gym-relaxed",
              session="gym:bravo", verification=VERIFIED),
        trace("charlie", "bash", {"command": "make charlie"}, source=MAXGYM,
              session="gym:charlie", verification=VERIFIED),
    ]
    for allow in (True, False):
        name = f"smoke-{'allow' if allow else 'plain'}.jsonl"
        b = allow_builder(tmp_path / f"b-{name}", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                          allow_max_derived=allow, max_derived_approval=REF if allow else None)
        feed(b, write_jsonl(tmp_path / name, rows), "extra")
        assert b.stats["excluded:never_train_source"] == 1  # neither train nor eval
        by_id = {e.trace_id: e for e in b.examples}
        assert f"{name}:1" not in by_id
        assert not by_id[f"{name}:2"].train_ok and by_id[f"{name}:2"].approval == ""
        b.split()
        trained = {e.trace_id for k in ("train", "val") for e in b.final[k]}
        assert trained == ({f"{name}:3"} if allow else set())
        assert b.gates()["data_policy"]["pass"]
    assert b.max_row_trains(build.Trace("x:1", "extra", "x", 1, rows[1])) == (False,
                                                                             "relaxed_source")
    # the 27B's own smoke rows keep their reason
    (tmp_path / "s").mkdir()
    s = make_builder(tmp_path / "s", train_sources=(GYM,))
    feed(s, write_jsonl(tmp_path / "s.jsonl", [gym("delta", "bash", {"command": "d"},
                                                   source="builtin-smoke")]), "extra")
    assert s.stats["excluded:non_teacher_model"] == 1 and not s.examples


def test_q6b_a_max_gym_row_trains_only_on_the_strict_gym_record(tmp_path: Path) -> None:
    payg = _payg_judge()
    no = {k: v for k, v in VERIFIED.items()}
    rows = [
        ("tool-success only", {"verified": True, "tool_success": True}, "no_judge"),
        ("no sandbox record", {k: v for k, v in no.items() if k != "sandboxed"},
         "no_sandbox_record"),
        ("no side-effects record", {k: v for k, v in no.items() if k != "side_effects"},
         "no_side_effects_record"),
        ("a judge not listed", {**no, "judge": {"verdict": "pass", "model": "qwen2.5:7b"}},
         "judge_not_allowed"),
        ("the local judge", VERIFIED, None),
        ("the PAYG Max judge", {**no, "judge": payg}, None),
    ]
    traces = [trace(word(i), "bash", {"command": f"make {word(i)}"}, source=MAXGYM,
                    session=f"gym:{word(i)}", verification=rec)
              for i, (_, rec, _) in enumerate(rows)]
    # a Max judge that is not the PAYG judge (the Token Plan key) verifies nothing
    traces.append(trace("tp", "bash", {"command": "make tp"}, source=MAXGYM, session="gym:tp",
                        verification={**no, "judge": {"verdict": "pass", "model": "qwen3.8-max"}}))
    b = allow_builder(tmp_path / "b", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "maxgym.jsonl", traces), "extra")
    assert b.excluded == [("maxgym.jsonl:7", "verification_failed")]
    assert b.stats["verification:judge_is_max"] == 1
    by_id = {e.trace_id: e for e in b.examples}
    for i, (what, _rec, why) in enumerate(rows):
        e = by_id[f"maxgym.jsonl:{i + 1}"]
        assert e.train_ok is (why is None), what
        if why:
            assert b.stats[f"max_derived_eval_only:{why}"] == 1, what
    assert by_id["maxgym.jsonl:6"].label == "judge-pass"
    assert b.stats["verification:payg_max_judge"] == 1
    b.split()
    trained = {e.trace_id for k in ("train", "val") for e in b.final[k]}
    assert trained == {"maxgym.jsonl:5", "maxgym.jsonl:6"}
    assert b.gates()["data_policy"]["pass"]
    # the local 27B's own gym rows still need a --judge-models judge: PAYG Max is not one
    (tmp_path / "g").mkdir()
    g = make_builder(tmp_path / "g", train_sources=(GYM,))
    feed(g, write_jsonl(tmp_path / "gym.jsonl", [
        gym("echo", "bash", {"command": "e"}, verification={**no, "judge": payg})]), "extra")
    assert g.examples == [] and g.stats["verification:judge_is_max"] == 1


def test_payg_max_judge_verdicts_are_applied_and_do_not_fail_the_gate(tmp_path: Path) -> None:
    judge = _load("distill_judge", "judge.py")
    ts = 1_790_000_000.0
    golden = write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace(word(i), "bash", {"command": f"make {word(i)}"}, ts=ts + i) for i in range(3)])
    index = [{"file": golden.name, "model": "qwen3.8-max", "h_prompt_call": f"k{i}",
              "category": "coding", "rowid": 100 + i, "tool": "bash", "ts": ts + i}
             for i in range(3)]

    def rows_from(client: Any) -> list[dict[str, Any]]:
        runner = types.SimpleNamespace(
            config=types.SimpleNamespace(view="outcome"), client=client,
            clock=lambda: datetime(2026, 10, 31, 6, 0, tzinfo=timezone.utc))
        out = []
        for item, v in zip(judge.pass_pool(index), ("pass", "fail", "pass")):
            parsed = judge.parse_verdict(json.dumps({
                "verdict": v, "reasons": ["r"], "rubric": {k: 1 for k in judge.RUBRIC_KEYS}}))
            out.append(judge.JudgeRun._row(runner, item, status="ok", agrees=v == "pass",
                                           **parsed))
        return out

    payg_client = types.SimpleNamespace(model="qwen3.8-max", provider=judge.PaygJudgeClient.provider,
                                        judge_key=judge.PaygJudgeClient.judge_key, region="intl")
    rows = rows_from(payg_client)
    assert {(r["provider"], r["judge_key"], r["model"]) for r in rows} == {
        ("max-payg", "payg", "qwen3.8-max")}
    stats: collections.Counter = collections.Counter()
    checks: dict[str, Any] = {}
    verdicts = build.load_verdicts([str(write_jsonl(tmp_path / "payg.jsonl", rows))], stats,
                                   checks)
    assert verdicts == {"golden_traces_1_1.jsonl:1": "pass", "golden_traces_1_1.jsonl:2": "fail",
                        "golden_traces_1_1.jsonl:3": "pass"}
    assert stats["verdict_rows_payg_max_judge"] == 3 and stats["verdict_rows_max_judge"] == 0
    b = make_builder(tmp_path, verdicts=verdicts, verdict_checks=checks,
                     index=build.load_index(str(write_jsonl(tmp_path / "index.jsonl", index))))
    b.stats.update(stats)
    feed(b, golden)
    assert {e.trace_id: e.label for e in b.examples} == {
        "golden_traces_1_1.jsonl:1": "judge-pass", "golden_traces_1_1.jsonl:3": "judge-pass"}
    assert b.stats["excluded:judge_fail"] == 1
    b.split()
    gate = b.gates()["verdicts"]
    assert gate["pass"] and gate["payg_max_judge"] == 3 and gate["max_judge"] == 0
    # the same rows without the PAYG provider and key (any other Max route) still fail it
    for client in (types.SimpleNamespace(model="qwen3.8-max", provider="local", judge_key="none"),
                   types.SimpleNamespace(model="qwen3.8-max", provider="max-payg",
                                         judge_key="token-plan")):
        st: collections.Counter = collections.Counter()
        v = build.load_verdicts([str(write_jsonl(tmp_path / "other.jsonl", rows_from(client)))],
                                st)
        assert st["verdict_rows_max_judge"] == 3 and st["verdict_rows_payg_max_judge"] == 0
        assert v == {"golden_traces_1_1.jsonl:2": "fail"}  # a fail still drops its row
    # the record mirror: a Max row's PAYG-judged record verifies it; nothing else naming Max does
    payg = {"verification": {"verified": True, "judge": _payg_judge()}}
    assert build.verification_of(payg) == build.Verification(False, "judge_is_max")
    v = build.verification_of(payg, payg_judge=True)
    assert (v.decision, v.basis, v.payg_judge) == (True, "judge-pass", True)
    tp = {"verification": {"verified": True, "judge": {**_payg_judge(), "judge_key": "sk-sp"}}}
    assert build.verification_of(tp, payg_judge=True).basis == "judge_is_max"
    lookalike = {"verification": {"verified": True, "judge": {
        "verdict": "pass", "model": "qwen3.8-27b", "provider": "max-payg",
        "served_model": "qwen3.8-max"}}}
    assert build.verification_of(lookalike, payg_judge=True).basis == "judge_is_max"


def test_q6b_listed_sessions_never_enter_the_dataset(tmp_path: Path) -> None:
    golden = write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "bash", {"command": "make alpha"}, session="beacon:incident"),
        trace("bravo", "bash", {"command": "make bravo"}, session="beacon:incident"),
        trace("charlie", "bash", {"command": "make charlie"}, session="beacon:ok"),
    ])
    b = allow_builder(tmp_path / "b", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                      exclude_sessions=("beacon:incident",),
                      verdicts={"golden_traces_1_1.jsonl:1": "pass"})  # Phase 2 judged it
    feed(b, golden)
    assert b.stats["excluded:excluded_session"] == 2
    assert b.stats["verdicts_on_excluded_sessions"] == 1
    assert [e.trace_id for e in b.examples] == ["golden_traces_1_1.jsonl:3"]
    info = b.split()
    assert {e.trace_id for k in ("train", "val") for e in b.final[k]} == {
        "golden_traces_1_1.jsonl:3"}
    gates = b.gates()
    assert gates["verdicts"]["pass"] and gates["verdicts"]["unmatched"] == 0
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    allow = report["data_policy"]["max_derived_allow"]
    assert allow["excluded_sessions_listed"] == 1 and allow["excluded_session_rows"] == 2
    assert report["settings"]["exclude_sessions"] == 1
    assert "beacon:incident" not in json.dumps(report)  # counts only, never the ids
    lst = tmp_path / "sessions.txt"
    lst.write_text("# incident\nbeacon:incident  # 2026-10-13\n\n")
    assert build.read_session_list(lst) == {"beacon:incident"}


def test_q6b_gym_rows_built_from_a_listed_sessions_prompts_never_enter(tmp_path: Path) -> None:
    # a gym row carries no session id: it is tied to the listed session through its task group
    # ("tg-" + the task proxy of the real prompt it was built from), a recombined partner's group or
    # a padding turn's group
    golden = write_jsonl(tmp_path / "golden_traces_1_1.jsonl", [
        trace("alpha", "bash", {"command": "make alpha"}, session="beacon:incident"),
        trace("charlie", "bash", {"command": "make charlie"}, session="beacon:ok"),
    ])
    lcm_db = tmp_path / "lcm.db"
    con = sqlite3.connect(lcm_db)
    con.execute("CREATE TABLE lcm_messages (session_id TEXT, role TEXT, content TEXT, content_json TEXT, "
                "turn_index INTEGER)")
    con.execute("INSERT INTO lcm_messages VALUES ('beacon:incident', 'user', 'resume the old job', NULL, 9)")
    con.commit()
    con.close()
    task = build.h(build.norm("please handle the alpha job"))
    pad = build.h(build.norm("opening note alpha"))
    lcm_only = build.h(build.norm("resume the old job"))
    other = build.h(build.norm("please handle the charlie job"))
    gym_file = write_jsonl(tmp_path / "gym.jsonl", [
        gym("delta", "bash", {"command": "make delta"}, task_id="tg-" + task),
        gym("echo", "bash", {"command": "make echo"}, task_id="tg-" + other, partner_task_id="tg-" + lcm_only),
        gym("foxtrot", "bash", {"command": "make foxtrot"}, task_id="tg-" + other,
            compaction_pad={"task_groups": ["tg-" + other, "tg-" + pad]}),
        gym("golf", "bash", {"command": "make golf"}, task_id="tg-" + other),
    ])
    lcm = build.LcmSnapshot(str(lcm_db))
    prompts = build.excluded_session_prompts({"beacon:incident"}, [golden, gym_file], lcm)
    assert {task, pad, lcm_only} <= prompts and other not in prompts
    assert build.excluded_session_prompts({"beacon:incident"}, [golden], None) == {task, pad}
    s = build.Settings(train_range=(1, 100000), eval_min=1, min_presence=0, train_sources=(GYM,),
                       eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                       exclude_sessions=("beacon:incident",))
    b = build.Builder(student=make_student(), pack=make_pack(), catalog="stubs", settings=s, index={},
                      verdicts={}, lcm=None, spool=tmp_path / "spool.jsonl", excluded_prompts=prompts)
    feed(b, golden)
    feed(b, gym_file, "extra")
    assert b.stats["excluded:excluded_session"] == 1
    assert b.stats["excluded:excluded_session_task"] == 3
    assert [e.trace_id for e in b.examples if e.source == GYM] == ["gym.jsonl:4"]
    info = b.split()
    report = b.report(info, b.gates(), catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    allow = report["data_policy"]["max_derived_allow"]
    assert allow["excluded_session_task_rows"] == 3 and allow["excluded_session_prompts"] == len(prompts)
    assert task not in json.dumps(report) and "beacon:incident" not in json.dumps(report)


# ── the flywheel's pool hands build.py the producer's own tag ──


def _flywheel() -> Any:
    if not (_DIR.parent.parent / "src" / "prometheus" / "jobs" / "flywheel_weekly.py").is_file():
        pytest.skip("flywheel_weekly.py not in this checkout")
    return pytest.importorskip("prometheus.jobs.flywheel_weekly")


def pooled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*rows* as the weekly flywheel pools them: harvested as extra traces
    (tag_trace), passed by its verify, re-written by pooled_meta."""
    fw = _flywheel()
    cfg = fw.parse_config(None)
    out = []
    for row in rows:
        meta = dict(row["_meta"])
        tag = fw.tag_trace(meta, cfg, {})
        harvested = {**tag, "kind": "extra", "source": fw.source_label("extra", meta)}
        basis, why = fw.verification(harvested, cfg.student_models)
        assert basis, why
        out.append({**row, "_meta": fw.pooled_meta(meta, harvested, basis, "2026-10-31")})
    return out


def test_a_pooled_max_smoke_row_is_still_never_trained(tmp_path: Path) -> None:
    """A Max-mode teacher_gen --builtin-smoke row (hand-written task text) the
    flywheel verified and pooled keeps the tag build.py refuses it by, under
    the Q6b allow or without it."""
    for judge in (VERIFIED["judge"], _payg_judge()):
        rows = pooled([trace("alpha", "bash", {"command": "make alpha"}, source="builtin-smoke",
                             session="gym:alpha", max_derived=True,
                             verification={**VERIFIED, "judge": judge})])
        assert rows[0]["_meta"]["source"] == "builtin-smoke"
        assert rows[0]["_meta"]["flywheel"]["tag"] == (
            "flywheel/teacher/qwen3.8-max@none/max-derived")
        for allow in (True, False):
            root = tmp_path / f"{judge['model']}-{allow}"
            b = allow_builder(root, eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                              allow_max_derived=allow, max_derived_approval=REF if allow else None)
            feed(b, write_jsonl(root / "pool.jsonl", rows), "extra")
            assert b.excluded == [("pool.jsonl:1", "never_train_source")]
            assert not b.examples


def test_pooled_payg_gym_rows_keep_build_pys_payg_rules(tmp_path: Path) -> None:
    """teacher_gen --teacher max-payg rows the flywheel pooled are still
    max-payg-gym rows to build.py: refused outright without the Q6b allow (no
    eval-tools rows that pin the local 27B's groups, no early dedupe claim),
    and under it admitted as PAYG gym rows. Fed pooled or fed directly, the
    builder decides the same."""
    payg = _payg_judge()
    tasks = [word(i) for i in range(12)]
    local = [gym(t, "bash", {"command": f"make {t}"}) for t in tasks]
    paid = [trace(t, "bash", {"command": f"make {t}"}, source="max-payg-gym",
                  session=f"gym:{t}", max_derived=True,
                  verification={**VERIFIED, "judge": payg if i % 2 else VERIFIED["judge"]})
            for i, t in enumerate(tasks[:6])]
    pool = pooled(paid)
    assert {r["_meta"]["source"] for r in pool} == {"max-payg-gym"}

    def built(root: Path, rows: list[dict[str, Any]], allow: bool) -> Any:
        b = allow_builder(root, category_cap=1.0, allow_max_derived=allow,
                          max_derived_approval=REF if allow else None)
        # max-payg-gym* sorts before qwen38-27b-gym*: the PAYG file comes first
        feed(b, write_jsonl(root / "max-payg-gym.jsonl", rows), "extra")
        feed(b, write_jsonl(root / "qwen38-27b-gym.jsonl", local), "extra")
        b.split()
        return b

    def splits(b: Any) -> dict[str, list[str]]:
        return {k: sorted(e.trace_id for e in b.final.get(k, []))
                for k in ("train", "val", "eval-tools")}

    alone = splits(built(tmp_path / "alone", [], False))     # the local rows by themselves
    assert alone["train"]
    for allow in (False, True):
        direct = built(tmp_path / f"direct-{allow}", paid, allow)
        via_pool = built(tmp_path / f"pool-{allow}", pool, allow)
        assert sorted(via_pool.excluded) == sorted(direct.excluded)
        assert splits(via_pool) == splits(direct)
        if not allow:
            assert via_pool.stats["excluded:max_payg_gym_needs_allow"] == len(paid)
            assert {e.source for e in via_pool.examples} == {GYM}
            # refused rows pin no group and claim no key: the local rows split as alone
            assert splits(via_pool) == alone
        else:
            assert via_pool.stats["excluded:max_payg_gym_needs_allow"] == 0
            assert {e.source for e in via_pool.examples if e.max_derived} == {"max-payg-gym"}


def test_the_flywheels_train_share_is_build_pys(tmp_path: Path) -> None:
    """A Q6b set mixes Max and local 27B rows: build.py names the Max model as
    its teacher, the flywheel's guard counts every row as Max-derived, and its
    share of the rows still matches build.py's data_policy.share."""
    fw = _flywheel()
    golden, gym_file = policy_inputs(tmp_path)
    b = allow_builder(tmp_path / "b")
    feed(b, golden)
    feed(b, gym_file, "extra")
    info = b.split()
    report = b.report(info, b.gates(), catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    out = tmp_path / "phase3-max-derived"
    out.mkdir()
    b.write(out)
    teacher = report["settings"]["teacher_model"]
    assert teacher == build.TEACHER_MODEL
    sessions = write_jsonl(tmp_path / "sessions.txt", [])
    sessions.write_text("beacon:incident\n")
    cfg = fw.parse_config({"max_derived": {"approved": True, "approval_ref": REF,
                                           "exclude_sessions_file": str(sessions)}})
    g = fw.check_training_data([out / "train.jsonl", out / "val.jsonl"], cfg, teacher)
    by = report["data_policy"]["share"]["by_split"]
    assert g.share == {c: by["train"][c]["rows"] + by["val"][c]["rows"]
                       for c in build.SHARE_CLASSES}
    assert g.share["max"] > 0 and g.share[GYM] > 0
    assert g.max_rows == g.rows and g.refusals == []
