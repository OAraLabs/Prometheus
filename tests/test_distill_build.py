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
