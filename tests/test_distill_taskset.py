"""Tests for scripts/distill/taskset_gen.py, the gym taskset generator.

Synthetic fixtures only (tests/fixtures/distill_taskset/): invented generic
prompts, a temp git repo, a temp wiki, a stand-in word list, a temp
conversation-store snapshot and a hand-made holdout / injection manifest.
Personal-data cases are assembled here at run time. No real prompts, no
network, no GPU.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"
FIX = Path(__file__).resolve().parent / "fixtures" / "distill_taskset"


def _load(name: str, filename: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("distill_taskset_gen", "taskset_gen.py")
BES = tg.BES
PROMPTS = json.loads((FIX / "prompts.json").read_text(encoding="utf-8"))
HOST = json.loads((FIX / "host.json").read_text(encoding="utf-8"))
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


# ── personal-data cases, assembled at run time (no secret-shaped literals) ──

def _pii_prompts() -> dict[str, str]:
    key = "sk-" + "Ab1" * 8
    gh = "ghp_" + "x9Y" * 8
    jwt = "eyJ" + "hbGciOi" + "JIUzI1" + "." + "eyJzdWIi" + "OiIxMjM0"
    hook = "https://hooks." + "slack.com/services/" + "T0AB12CD3/B0CD34EF5/" + "Xy12Ab34Cd56Ef78Gh90Ij12"
    q = "q8Zr2" + "LmN4tYp"
    return {
        "webhook_url": f"Post the daily summary to {hook} every morning.",
        "access_token_url": "Fetch https://api.example.com/v1/items?access" + f"_token={q} and list them.",
        "key_query_url": f"Fetch https://api.example.com/v1/items?key={q} and list them.",
        "sig_query_url": f"Download https://files.example.com/a/report.csv?sig={q} for me.",
        "email": "Send the weekly report to ops" + "@" + "example.com today.",
        "user_at_host": "Copy src/app/util.py over to deploy" + "@" + "buildbox please.",
        "ip": "Restart the service on 192.0.2." + "44 tonight please.",
        "api_key": f"Use the key {key} when you call the provider.",
        "api_key_gh": f"Clone it with the token {gh} and build it.",
        "jwt": f"Decode this session value {jwt} and explain it.",
        "credential": "Log in with password" + ": " + "hunter22 and check the queue.",
        "phone": "Text the on-call number 555-" + "123-4567 when it finishes.",
        "mac": "Wake the box with MAC 00:1a:2b:" + "3c:4d:5e tonight please.",
        "tailnet_host": "Ping the box at node-a.tail" + "1234.ts.net from here please.",
    }


# ── the synthetic host ──────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    # fixed dates: the fixture commit (and every sha slot value) is the same on every run
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, env=env)


def _golden_line(text: str, surface: str, n: int) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "An earlier message in the same chat."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{n}", "type": "function",
                 "function": {"name": "bash", "arguments": json.dumps({"command": f"echo step {n}"})}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
        "_meta": {"model": "teacher-model", "tool_name": "bash", "timestamp": 1.0 + n,
                  "session_id": f"{surface}:session-{n % 3}"},
    }


def _lcm_db(path: Path, rows: list[tuple[str, str, str, str | None, str]]) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE lcm_messages (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, turn_index INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL, content TEXT NOT NULL, content_json TEXT, token_count INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL, compacted INTEGER NOT NULL DEFAULT 0,
            provenance TEXT NOT NULL DEFAULT 'user', is_trusted INTEGER NOT NULL DEFAULT 1, author TEXT);
    """)
    for i, (sid, role, content, cj, prov) in enumerate(rows):
        con.execute("INSERT INTO lcm_messages (id, session_id, turn_index, role, content, content_json, timestamp,"
                    " provenance) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (f"m{i}", sid, i, role, content, cj, float(i), prov))
    con.commit()
    con.close()


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("git") is None:
        pytest.skip("needs git")
    root = tmp_path_factory.mktemp("taskset-host")
    repo = root / "repo"
    for rel, content in HOST["repo_files"].items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    wiki = root / "wiki"
    for rel, content in HOST["wiki"].items():
        p = wiki / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    (root / "cron_jobs.json").write_text(json.dumps(HOST["cron_jobs"]), encoding="utf-8")
    (root / "tools.json").write_text(json.dumps({"tools": [
        {"type": "function", "function": {"name": n}} for n in HOST["tools"]]}), encoding="utf-8")
    (root / "words.txt").write_text("\n".join(HOST["dictionary"]) + "\n", encoding="utf-8")

    traces = root / "traces"
    traces.mkdir()
    golden = [dict(g) for g in PROMPTS["golden"]]
    golden += [{"text": t, "surface": "beacon", "category": "other", "traces": 1} for t in _pii_prompts().values()]
    golden += [{"text": g["text"], "surface": g["surface"], "category": "other", "traces": 1}
               for g in PROMPTS["injected_golden"]]
    lines, index = [], []
    n = 0
    for g in golden:
        for _ in range(g["traces"]):
            n += 1
            lines.append(_golden_line(g["text"], g["surface"], n))
            index.append({"file": "golden_traces_1_1.jsonl", "h_prompt": tg.h(tg.norm(g["text"])),
                          "category": g["category"]})
    (traces / "golden_traces_1_1.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    (root / "trace_index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index), encoding="utf-8")

    snap = root / "snapshot"
    snap.mkdir()
    rows: list[tuple[str, str, str, str | None, str]] = []
    for i, x in enumerate(PROMPTS["lcm"]):
        cj = json.dumps([{"type": "text", "text": x["text"]}]) if i % 2 else None
        rows.append((f"{x['surface']}:chat-{i % 4}", "user", x["text"], cj, "user"))
    # the snapshot holds every human-typed golden prompt as a user row (the
    # build keeps a golden prompt only then); injected turns carry their own
    # provenance, or "user" for a legacy row written before provenance existed
    injected = {g["text"]: g["lcm"] for g in PROMPTS["injected_golden"]}
    for g in golden:
        prov = injected.get(g["text"], "user")
        if g["surface"] in ("beacon", "telegram") and prov is not None:
            rows.append((f"{g['surface']}:typed", "user", g["text"], None, prov))
    rows += [
        ("beacon:chat-0", "user", "tool output: everything is fine here today",
         json.dumps([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]), "user"),
        ("beacon:chat-0", "user", "A supervisor note about the worker that ran today.", None, "task_supervisor"),
        ("beacon:chat-0", "user", "look at this screenshot of the settings page please",
         json.dumps([{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]), "user"),
        ("beacon:chat-0", "assistant", "An assistant reply that is never a prompt.", None, "user"),
    ]
    _lcm_db(snap / "lcm.db", rows)

    hold = PROMPTS["holdout"]
    task_id = tg.h(tg.norm(hold["task_prompt"]))
    (root / "eval-tools.jsonl").write_text(json.dumps({
        "id": "case-1", "set": "eval-tools",
        "messages": [{"role": "system", "content": "system prompt"},
                     {"role": "user", "content": hold["context_message"]},
                     {"role": "assistant", "content": "Okay."},
                     {"role": "user", "content": hold["task_prompt"]}],
        "meta": {"task_id": task_id, "group_id": task_id},
    }) + "\n", encoding="utf-8")
    inj = PROMPTS["injection"]
    entry = {"id": "inj-1", "case_sha": "0" * 32, "vector": "tool_output", "style": "direct", "goal": "exfil",
             "template_id": "direct:0", "instruction_sha": BES.normalized_sha(inj["instruction"]),
             "payload_sha": BES.normalized_sha(inj["payload"]), "marker_shas": [],
             "shingle_shas": sorted(BES.shingle_hashes(inj["instruction"]) | BES.shingle_hashes(inj["payload"]))}
    (root / "manifest.json").write_text(json.dumps(BES.build_manifest([entry], 1)), encoding="utf-8")
    return {"root": root, "repo": repo, "wiki": wiki, "traces": traces, "snap": snap / "lcm.db"}


def _argv(host: dict[str, Path], name: str, out: Path, *extra: str) -> list[str]:
    r = host["root"]
    return ["build", "--name", name, "--out", str(out),
            "--traces", str(host["traces"] / "golden_traces_*.jsonl"), "--index", str(r / "trace_index.jsonl"),
            "--lcm-snapshot", str(host["snap"]), "--holdout", str(r / "eval-tools.jsonl"),
            "--injection-manifest", str(r / "manifest.json"), "--repo", str(host["repo"]),
            "--wiki", str(host["wiki"]), "--wiki-sections", "topics", "--pack", str(r / "absent-pack.json"),
            "--tool-schemas", str(r / "tools.json"), "--cron-jobs", str(r / "cron_jobs.json"),
            "--dictionary", str(r / "words.txt"), "--base-date", "2026-11-01", *extra]


def _run(host: dict[str, Path], name: str, *extra: str) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    out = host["root"] / f"out-{name}"
    assert tg.main(_argv(host, name, out, *extra)) == 0
    rows = tg.load_taskset(out / f"{name}.jsonl")
    report = json.loads((out / f"{name}.report.json").read_text(encoding="utf-8"))
    return rows, report, out


@pytest.fixture(scope="module")
def built(host: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    return _run(host, "base", "--per-pattern", "6", "--max-uses", "8", "--coding-cap", "0.3")


def _generator(host: dict[str, Path], holdout_texts: tuple[str, ...] = (), **settings: Any) -> Any:
    """The same build as ``built`` through the API, for access to templates."""
    import collections
    r = host["root"]
    stats: collections.Counter = collections.Counter()
    persons = tg.PersonNames.from_wiki(host["wiki"])
    inv = tg.Inventory.build(repo=host["repo"], wiki=host["wiki"], wiki_sections=["topics"],
                             tool_names=HOST["tools"], cron_names=[j["name"] for j in HOST["cron_jobs"]],
                             persons=persons, stats=stats)
    hold = tg.Holdout()
    hold.load(r / "eval-tools.jsonl")
    for t in holdout_texts:
        hold.add_text(t)
    manifest = BES.InjectionManifest.load(r / "manifest.json")
    s = tg.Settings(base_date="2026-11-01", **settings)
    golden = tg.load_golden(sorted(host["traces"].glob("*.jsonl")), tg.load_index(r / "trace_index.jsonl"),
                            s.surfaces, stats)
    injected: set[str] = set()
    prompts = dict(golden)
    for k, v in tg.load_lcm(host["snap"], s.surfaces, stats, injected).items():
        if k in prompts:
            prompts[k].in_lcm = True
        else:
            prompts[k] = v
    gen = tg.Generator(s, inv, holdout=hold, manifest=manifest, persons=persons,
                       lexicon=tg.Lexicon(HOST["dictionary"]), injected=injected, require_user_row=True)
    gen.add_prompts(prompts.values())
    return gen


# ── ported functions ──────────────────────────────────────────────────────


def test_norm_and_hash_match_build_py() -> None:
    build = _load("distill_build", "build.py")
    for s in ["Fix the bug in abc1234def at 10:30", "  Mixed   CASE\n42 and 7 ", "", "ünïcode 0xFFAA12"]:
        assert tg.norm(s) == build.norm(s)
        assert tg.h(tg.norm(s)) == build.h(build.norm(s))
    assert tg.DAEMON_PROMPT.pattern == build.DAEMON_PROMPT.pattern
    assert tg.COMPACTION.pattern == build.COMPACTION.pattern
    assert tg.HUMAN_SURFACES == build.HUMAN_SURFACES


# ── slot extraction ─────────────────────────────────────────────────────────


def _extractor(**pools: list[str]) -> Any:
    inv = tg.Inventory({k: [(v, "repo") for v in vals] for k, vals in pools.items()})
    return tg.SlotExtractor(inv, tg.Lexicon(HOST["dictionary"]))


def _types(ex: Any, text: str) -> list[tuple[str, str]]:
    slots, _ = ex.extract(text)
    return [(s.type, text[s.start:s.end]) for s in slots]


def test_slot_extraction_types() -> None:
    ex = _extractor(**{"tool": ["web_fetch", "bash"], "path:any": ["src/app/util.py"], "dir": ["src/app"],
                       "wiki_page": ["Release Process", "LoRA Training Notes"], "component": ["symbiote", "gym"],
                       "topic": ["Retry Policy", "LoRA Training Notes"], "job": ["nightly digest"]})
    got = _types(ex, "Summarize docs/guide/setup.md and run web_fetch on Friday at 9am, then src/app/ too.")
    assert got == [("path", "docs/guide/setup.md"), ("tool", "web_fetch"), ("weekday", "Friday"),
                   ("time", "9am"), ("path", "src/app/")]
    got = _types(ex, 'See "Retry Policy" at https://docs.example.org/x. Bump 0.9.17 per #12, commit abc1234f.')
    assert got == [("topic", "Retry Policy"), ("url", "https://docs.example.org/x"), ("version", "0.9.17"),
                   ("issue", "#12"), ("sha", "abc1234f")]
    got = _types(ex, "Run 0 6 * * * on 2026-11-20 and Oct 12th at 14:30 for 16,384 rows from config.yaml")
    assert got == [("cron", "0 6 * * *"), ("iso_date", "2026-11-20"), ("month_date", "Oct 12th"),
                   ("time", "14:30"), ("number", "16"), ("number", "384"), ("file", "config.yaml")]
    # a distinctive package name is a slot, an everyday word is not; a title needs the wiki or a distinctive word
    assert ("component", "symbiote") in _types(ex, "Explain what the symbiote module does.")
    assert _types(ex, "describe how the gym scores a run") == []
    assert ("wiki_page", "Release Process") in _types(ex, "Add a wiki page about the Release Process.")
    assert _types(ex, "Our release process is slow.") == []
    assert ("wiki_page", "LoRA Training Notes") in _types(ex, "Read the LoRA Training Notes before the run.")
    assert ("job", "nightly digest") in _types(ex, "When does the nightly digest go out?")


def test_quoted_code_is_protected_and_home_paths_must_be_refilled() -> None:
    ex = _extractor()
    slots, protected = ex.extract("Why does `x = compute(3)` fail? It runs at 5.")
    assert [(s.type) for s in slots] == ["number"] and len(protected) == 1
    slots, _ = ex.extract("Open /home/someone/notes/todo.md and the ~/work/plan.md file.")
    assert [(s.type, s.must_fill) for s in slots] == [("path", True), ("path", False)]
    assert _types(ex, "Ship it and/or wait 24/7 for the TCP/IP fix") == [("number", "24"), ("number", "7")]
    assert _types(ex, "Set up Node.js first") == []


def test_home_path_outside_a_path_slot_drops_the_template() -> None:
    gen = tg.Generator(tg.Settings(), tg.Inventory({"path:any": [("src/app/util.py", "repo")]}))
    user = "some" + "one"
    literal = [f"Run `du -sh /home/{user}/models` and tell me which folder is biggest.",
               f'Why does "cat /Users/{user}/notes/todo.md" fail on the box?',
               f"Open file:///home/{user}/Downloads/report.pdf and summarize it.",
               f"Copy the report to box:/home/{user}/reports/today.txt when it is done."]
    for text in literal:
        rp = tg.RealPrompt(text, tg.h(tg.norm(text)), "lcm")
        assert gen.template_problem(rp) == "home_path_literal", text
    # a bare home path is a must-fill path slot: kept, and refilled in every variant
    bare = f"Open /home/{user}/notes/todo.md and fix the typo in it."
    rp = tg.RealPrompt(bare, tg.h(tg.norm(bare)), "lcm")
    assert gen.template_problem(rp) is None
    gen.add_prompts([rp])
    rows = gen.generate("home")
    assert rows and not any(user in r["prompt"] or "/home/" in r["prompt"] for r in rows)
    # the last guard on every variant: no home path, whatever made it
    tpl = gen.templates[0]
    c = tg.Candidate(f"Then read /home/{user}/notes/x.md.", [(tpl, 0, len(tpl.text))], {}, "slot_fill",
                     tpl.pattern_hash)
    assert gen.problem(c) == "home_path"


def test_modal_may_is_not_a_month_date() -> None:
    ex = _extractor()
    assert _types(ex, "These 2 may conflict with the cron entries, run the check.") == [("number", "2")]
    assert _types(ex, "Both 3 may be wrong") == [("number", "3")]
    assert ("month_date", "3 May") in _types(ex, "Ship it on 3 May please")
    assert ("month_date", "May 3rd") in _types(ex, "due May 3rd")
    assert ("month_date", "3rd of may") in _types(ex, "move it to the 3rd of may")
    assert ("month_date", "12 oct") in _types(ex, "the notes from 12 oct")


def test_common_tech_package_names_need_a_code_unit_context() -> None:
    inv = tg.Inventory({"component": [(c, "repo") for c in ("config", "mcp", "cli", "symbiote")]})
    ex = tg.SlotExtractor(inv, tg.Lexicon(["the", "and", "check", "restart", "daemon"]))  # word list lacks them
    assert _types(ex, "Check the config and restart the daemon.") == []
    assert _types(ex, "Why is the MCP server slow?") == []
    assert _types(ex, "Explain the config module.") == [("component", "config")]
    assert _types(ex, "Read prometheus.mcp first") == [("component", "mcp")]
    assert _types(ex, "Explain what the symbiote does") == [("component", "symbiote")]


def test_sentences_never_split_inside_a_slot() -> None:
    ex = _extractor()
    text = 'Check "Phase one. Phase two" now. Then rerun it.\nAfter that, stop.'
    slots, prot = ex.extract(text)
    sents = tg.sentences(text, [(s.start, s.end) for s in slots] + prot)
    assert [text[a:b] for a, b in sents] == ['Check "Phase one. Phase two" now. ', "Then rerun it.\n",
                                              "After that, stop."]


# ── intent and subcategory ──────────────────────────────────────────────────


def test_leading_verb_outranks_a_keyword_anywhere() -> None:
    gen = tg.Generator(tg.Settings(), tg.Inventory({}))
    actions = {"Delete the old summary emails from the archive folder please.": "edit",
               "Fix the summary generator so it skips empty days.": "fix",
               "Update the roadmap with the new dates and push it.": "edit",
               "Rename the strategy module to planner and update imports.": "edit",
               "Please restart the job that sends the summary every morning.": "run"}
    for text, intent in actions.items():
        assert tg.classify_intent(text) == intent, text
        assert gen.make_template(tg.RealPrompt(text, tg.h(tg.norm(text)), "lcm")).chat_ok is False, text
    asks = {"Give me a summary of the release notes.": "summarize",
            "I want a short summary of the morning briefing.": "summarize",
            "what is the best way to keep the reading list organized?": "plan",
            "How should we split the ingest pipeline into smaller stages?": "plan",
            "Could we set up a reminder every morning for the standup?": "schedule",
            "Is the telemetry exporter still running after the update?": "question",
            "Remember that the staging server restarts on Friday.": "remember"}
    for text, intent in asks.items():
        assert tg.classify_intent(text) == intent, text


def test_phase1_coding_label_keeps_coding_work_under_the_cap() -> None:
    ex = _extractor()
    probes = ["Compare commit abc1234f with the previous release and list what changed.",
              "Plan the migration of the config loader. Keep the old keys working.",
              "Fix the bug in the server config loader.",
              "Update the cron parser so it accepts step values."]
    for text in probes:
        slots, _ = ex.extract(text)
        assert tg.classify_subcategory(text, slots, "coding", tg.classify_intent(text)) == "coding", text
        assert tg.classify_subcategory(text, slots, None, tg.classify_intent(text)) != "coding", text
    # what the request asks for still wins, and so do clear non-coding keywords
    assert tg.classify_subcategory("Remind me every Monday at 9am to review the open tasks.", (), "coding",
                                   "schedule") == "scheduling"
    assert tg.classify_subcategory("Restart the daemon and check the logs on the server.", (), "coding",
                                   "run") == "daemon-ops"


def test_phase1_coding_prompts_count_against_the_cap(host: dict[str, Path]) -> None:
    gen = _generator(host, per_pattern=6, max_uses=8, coding_cap=0.3)
    tpl = next(t for t in gen.templates if t.text.startswith("Plan the migration of the config loader"))
    assert tpl.subcategory == "coding"  # Phase 1 coding + a daemon keyword ("config") used to be daemon-ops
    rows = gen.generate("cap")
    assert all(r["category"] in ("coding", "chat-mode") for r in rows if r["template_id"] == tpl.id)


# ── personal data ───────────────────────────────────────────────────────────


def test_pii_detection_finds_each_kind() -> None:
    cases = _pii_prompts()
    for kind, text in cases.items():
        assert tg.pii_kinds(text), kind
    assert "ip" in tg.pii_kinds("talk to 2001:db8::1 now")
    assert "card" in tg.pii_kinds("charge 4111 1111 1111 1111 now")
    assert "ssn" in tg.pii_kinds("ssn 123-45-6789")
    assert "token" in tg.pii_kinds("value Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0LmNb")
    clean = ["serve on 127.0.0.1:8080 and ::1", "bump 1.2.3 at 10:30:00", "16,384 tokens in commit abc1234",
             "read src/app/util.py", "the @channel ping", "timestamp 1786815693.47",
             "open https://docs.example.org/guide/setup?page=2#install", "the key insight is that tokens = words",
             "the primary key: id", "what does the access_token field mean?"]
    for text in clean:
        assert tg.pii_kinds(text) == [], text
    for kind in ("webhook_url", "access_token_url", "key_query_url", "sig_query_url"):
        assert "secret_url" in tg.pii_kinds(cases[kind]), kind
    assert "credential" in tg.pii_kinds("set client" + "_secret: " + "abcd1234 in the env")
    assert "secret_url" in tg.pii_kinds("send it to https://discord.com/api/webhooks/123/" + "aB3dE5fG7hJ9kL1mN3pQ")


def test_person_names_deny_list(tmp_path: Path) -> None:
    (tmp_path / "people").mkdir()
    (tmp_path / "people" / "jordan-blake.md").write_text("# Jordan Blake\n", encoding="utf-8")
    (tmp_path / "clients").mkdir()
    (tmp_path / "clients" / "northwind-traders.md").write_text("# Northwind Traders\n", encoding="utf-8")
    names = tg.PersonNames.from_wiki(tmp_path)
    assert names.mentions("Ask Jordan about it")
    assert names.mentions("email northwind traders today")
    assert not names.mentions("the jordan river")  # lowercase single token: an ordinary word
    assert not names.mentions("Release the build")
    # typed in lowercase next to a person cue, possessive, or in capitals
    assert names.mentions("ask jordan about the deploy")
    assert names.mentions("ASK JORDAN about it")
    assert names.mentions("is jordan's laptop still on the old build?")
    assert names.mentions("set up a call with blake tomorrow")
    assert not names.mentions("the blake poems are in the reading list")


def test_pii_templates_are_dropped_not_filled(built: tuple, host: dict[str, Path]) -> None:
    rows, report, _ = built
    excluded = report["sources"]["excluded"]
    pii_total = sum(v for k, v in excluded.items() if k.startswith("pii:"))
    assert pii_total == len(_pii_prompts())
    pii_hashes = {tg.h(tg.norm(t)) for t in _pii_prompts().values()}
    for r in rows:
        assert not tg.pii_kinds(r["prompt"])
        assert not {s["pattern_hash"] for s in r["sources"]} & pii_hashes
        assert "Jordan" not in r["prompt"] and "Northwind" not in r["prompt"]
    assert excluded["person_name"] == 1
    # wiki pages named after people, clients or IPs never become slot values
    inv_values = {v for r in rows for s in r["slots"] for v in [s["value"]]}
    assert not any("192.0.2" in v or "jordan" in v.lower() or "northwind" in v.lower() for v in inv_values)


# ── exclusions ──────────────────────────────────────────────────────────────


def test_holdout_unit() -> None:
    hold = tg.Holdout(k=5, containment=0.6)
    hold.add_text("Check the disk usage on the storage volume and report the totals.")
    hold.task_ids.add("feedfacecafebeef")
    assert hold.check("check the disk usage on the storage volume and report the totals.") == "holdout_task"
    assert hold.check("Please check the disk usage on the storage volume and report the totals today.") \
        == "holdout_near_dup"
    assert hold.check("Summarize the release notes for the parser.") is None


def test_holdout_catches_a_prompt_that_differs_only_in_a_slot() -> None:
    gen = tg.Generator(tg.Settings(), tg.Inventory({}), holdout=tg.Holdout(k=5, containment=0.6))
    hold = gen.holdout
    held = "Run the linter on scripts/tools/report.py and fix what it finds."
    hold.add_text(held)  # added after the masker is set: indexed both ways
    assert hold.check("Run the linter on src/app/cli.py and fix what it finds.") == "holdout_slot_variant"
    assert hold.check("Run the linter on scripts/tools/export.py and fix what it finds.") == "holdout_slot_variant"
    assert hold.check("Please run the linter on src/app/cli.py and fix what it finds.") == "holdout_slot_variant"
    # the raw check alone missed these (one token of 12 differs)
    raw = tg.Holdout(k=5, containment=0.6)
    raw.add_text(held)
    assert raw.check("Run the linter on src/app/cli.py and fix what it finds.") is None
    # a different request with a path is not excluded, nor is a mostly-placeholder one
    assert hold.check("Refactor src/app/parser.py so the tokenizer lives in its own module.") is None
    assert hold.check("Run the tests on src/app/cli.py and then deploy the build.") is None
    hold.add_text("open src/app/cli.py")  # 1 literal word: never a masked match
    assert hold.check("open docs/guide/setup.md") is None


def test_holdout_slot_variant_excludes_template_and_variants(host: dict[str, Path]) -> None:
    held = "Run the linter on src/app/cli.py and fix what it finds."
    gen = _generator(host, holdout_texts=(held,), per_pattern=6, max_uses=8, coding_cap=0.3)
    assert gen.excluded["holdout_slot_variant"] == 1
    assert not any(t.text.startswith("Run the linter on") for t in gen.templates)
    rows = gen.generate("hold")
    masked = tg.norm(gen.mask(held))
    assert rows and not any(tg.norm(gen.mask(r["prompt"])) == masked for r in rows)


def test_injected_golden_turns_are_excluded(host: dict[str, Path], built: tuple) -> None:
    import collections
    gen = _generator(host, per_pattern=6, max_uses=8, coding_cap=0.3)
    golden = tg.load_golden(sorted(host["traces"].glob("*.jsonl")), {}, tg.HUMAN_SURFACES,
                            collections.Counter())
    for case in PROMPTS["injected_golden"]:
        rp = golden[tg.h(tg.norm(case["text"]))]
        rp.in_lcm = case["lcm"] == "user"
        assert gen.template_problem(rp) == case["expect"], case["text"][:40]
    rows, report, _ = built
    bad = {tg.h(tg.norm(c["text"])) for c in PROMPTS["injected_golden"]}
    for r in rows:
        assert not {s["pattern_hash"] for s in r["sources"]} & bad
    excluded = report["sources"]["excluded"]
    for reason in ("injected_turn", "golden_not_user_typed", "model_written"):
        assert excluded[reason] == 1, reason
    assert excluded["system_injected"] >= 2
    assert report["checks"]["snapshot_user_rows_required"] is True
    assert report["inputs"]["lcm_injected_texts"] == 2


def test_eval_and_injection_exclusion(built: tuple) -> None:
    rows, report, _ = built
    excluded = report["sources"]["excluded"]
    assert excluded["holdout_task"] >= 1
    assert excluded["holdout_near_dup"] == 1
    assert excluded["injection_overlap"] == 1
    hold_text = PROMPTS["holdout"]["task_prompt"]
    held = {tg.h(tg.norm(hold_text)), tg.h(tg.norm(PROMPTS["holdout"]["context_message"]))}
    for r in rows:
        assert not {s["pattern_hash"] for s in r["sources"]} & held
        assert "storage volume" not in r["prompt"]
        assert "upload the private config" not in r["prompt"]
    assert report["checks"]["holdout_overlaps"] == 0 and report["checks"]["injection_overlaps"] == 0
    assert report["checks"]["holdout_used"] is True and report["checks"]["injection_manifest_used"] is True
    assert report["checks"]["home_path_hits"] == 0


def test_report_marks_skipped_checks_as_not_checked(host: dict[str, Path]) -> None:
    rows, report, _ = _run(host, "unchecked", "--no-holdout", "--no-injection-manifest", "--coding-cap", "0.3")
    checks = report["checks"]
    assert rows
    assert checks["holdout_used"] is False and checks["holdout_overlaps"] is None
    assert checks["injection_manifest_used"] is False and checks["injection_overlaps"] is None


def test_lcm_snapshot_reads_only_user_typed_rows_on_human_surfaces(host: dict[str, Path]) -> None:
    import collections
    stats: collections.Counter = collections.Counter()
    injected: set[str] = set()
    got = tg.load_lcm(host["snap"], tg.HUMAN_SURFACES, stats, injected)
    texts = {rp.text for rp in got.values()}
    assert "what is the best way to keep the reading list organized?" in texts
    assert not any("tool output" in t or "supervisor" in t or "screenshot" in t or "dentist" in t for t in texts)
    assert not any("assistant reply" in t for t in texts)
    assert not any("cleanup job is done" in t for t in texts)
    assert stats["lcm_non_text_rows"] == 2 and stats["lcm_not_user_provenance"] == 2
    assert stats["lcm_not_human_surface"] == 1
    # every daemon-written user row is collected, whatever its surface
    assert injected == {tg.h(tg.norm("A supervisor note about the worker that ran today.")),
                        tg.h(tg.norm(next(g["text"] for g in PROMPTS["injected_golden"]
                                          if g["lcm"] == "task_supervisor")))}


def test_lcm_untrusted_text_turns_are_injected_not_prompts(tmp_path: Path) -> None:
    import collections
    db = tmp_path / "lcm.db"
    _lcm_db(db, [("beacon:a", "user", "Please tidy the notes folder and sort it by date.", None, "user"),
                 ("beacon:a", "user", "Look at the job output and say if it passed.", None, "user"),
                 ("beacon:a", "user", "", json.dumps([{"type": "tool_result", "tool_use_id": "t", "content": "x"}]),
                  "user")])
    con = sqlite3.connect(db)
    con.execute("UPDATE lcm_messages SET is_trusted = 0 WHERE id IN ('m1', 'm2')")
    con.commit()
    con.close()
    stats: collections.Counter = collections.Counter()
    injected: set[str] = set()
    got = tg.load_lcm(db, tg.HUMAN_SURFACES, stats, injected)
    assert [rp.text for rp in got.values()] == ["Please tidy the notes folder and sort it by date."]
    assert injected == {tg.h(tg.norm("Look at the job output and say if it passed."))}
    # a tool result counts as non-text whatever its trust flag
    assert stats["lcm_untrusted_user_rows"] == 1 and stats["lcm_non_text_rows"] == 1


def test_refuses_live_store_and_output_in_a_repo(host: dict[str, Path], tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    live = tmp_path / ".prometheus" / "data" / "lcm.db"
    live.parent.mkdir(parents=True)
    shutil.copy(host["snap"], live)
    assert tg.main(_argv(host, "live", tmp_path / "out", "--lcm-snapshot", str(live))) == 2
    assert tg.main(_argv(host, "inrepo", host["repo"] / "out")) == 2
    assert not (host["repo"] / "out").exists()


# ── output contract ─────────────────────────────────────────────────────────


def test_schema_every_row(built: tuple) -> None:
    rows, report, out = built
    assert rows and report["output"]["tasks"] == len(rows)
    for r in rows:
        assert tg.validate_task(r) == []
        assert r["provenance"] == "real-pattern-v1"
        assert r["flags"]["suppress_tools"] == (r["mode"] == "chat")
        if r["mode"] == "chat":
            assert r["intent"] in tg.CHAT_INTENTS and r["category"] == "chat-mode"
            assert not any(s["type"] in ("path", "file", "url") for s in r["slots"])
        else:
            assert r["category"] == tg.SUBCAT_TO_CATEGORY[r["subcategory"]]
    assert (out / "base.jsonl").stat().st_mode & 0o077 == 0
    assert {r["mode"] for r in rows} == {"chat", "agentic"}
    assert {"slot_fill", "recombine"} <= {r["transform"] for r in rows}


def test_validate_task_rejects_bad_rows(built: tuple) -> None:
    rows, _, _ = built
    good = rows[0]
    bad = []
    for mutate in (
        lambda r: r.pop("provenance"),
        lambda r: r.update(provenance="free-text"),
        lambda r: r.update(prompt=r["prompt"] + " extra"),
        lambda r: r["flags"].update(suppress_tools=not r["flags"]["suppress_tools"]),
        lambda r: r.update(category="nonsense"),
        lambda r: r.update(extra_key=1),
        lambda r: r.update(task_group=r["id"]),
        lambda r: r.pop("task_group"),
        lambda r: r["slots"].append({"name": "x", "type": "path", "value": "v", "from": "llm", "source": 0,
                                     "span": [0, 1]}),
    ):
        r = copy.deepcopy(good)
        mutate(r)
        bad.append(tg.validate_task(r))
    assert all(bad)


def test_load_taskset_refuses_a_bad_row(tmp_path: Path, built: tuple) -> None:
    rows, _, _ = built
    r = copy.deepcopy(rows[0])
    r["mode"] = "chat" if r["mode"] == "agentic" else "agentic"
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(rows[1]) + "\n" + json.dumps(r) + "\n", encoding="utf-8")
    with pytest.raises(tg.TasksetError, match="line 2"):
        tg.load_taskset(p)
    assert tg.main(["validate", str(p)]) == 2


def test_report_holds_no_text_ids_or_hashes(built: tuple) -> None:
    rows, report, out = built
    text = (out / "base.report.json").read_text(encoding="utf-8")
    for p in PROMPTS["golden"] + PROMPTS["lcm"]:
        assert p["text"] not in text
    for r in rows[:50]:
        assert r["id"] not in text and r["template_id"] not in text and r["source_pattern_hash"] not in text
    assert not re.search(r"\b[0-9a-f]{16}\b", text)
    assert str(Path.home()) not in text


# ── determinism, dedupe, diversity, caps, flags ─────────────────────────────


def test_determinism_by_seed(host: dict[str, Path], built: tuple) -> None:
    _, _, out = built
    _, _, out2 = _run(host, "base2", "--per-pattern", "6", "--max-uses", "8", "--coding-cap", "0.3")
    assert (out / "base.jsonl").read_bytes() == (out2 / "base2.jsonl").read_bytes().replace(b'"base2"', b'"base"')
    rows3, _, _ = _run(host, "seed2", "--per-pattern", "6", "--max-uses", "8", "--coding-cap", "0.3",
                       "--seed", "7")
    rows1 = tg.load_taskset(out / "base.jsonl")
    assert [r["prompt"] for r in rows1] != [r["prompt"] for r in rows3]


def test_determinism_across_hash_seeds(host: dict[str, Path], tmp_path: Path) -> None:
    """Set iteration order changes with PYTHONHASHSEED; the task file must not."""
    outs = []
    for seed in ("1", "2"):
        out = tmp_path / f"o{seed}"
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, str(_DIR / "taskset_gen.py"),
                              *_argv(host, "x", out, "--per-pattern", "6", "--coding-cap", "0.3")],
                             capture_output=True, text=True, env=env)
        assert res.returncode == 0, res.stderr[-500:]
        outs.append((out / "x.jsonl").read_bytes())
    assert outs[0] == outs[1] and outs[0]


def test_holdout_short_messages_do_not_over_exclude() -> None:
    hold = tg.Holdout(k=5, containment=0.6)
    hold.add_text("continue")
    hold.add_text("ship the parser fix")
    assert hold.check("continue") == "holdout_task"
    assert hold.check("Continue with the release checklist for the parser tomorrow.") is None
    assert hold.check("please ship the parser fix") == "holdout_near_dup"
    assert hold.check("Once the tests pass, ship the parser fix and tag the release notes.") is None
    # a long held-out request quoted inside a longer prompt is caught both ways
    hold.add_text("Rotate the archive folder and keep only the last three weekly snapshots.")
    assert hold.check("Rotate the archive folder and keep only the last three weekly snapshots. "
                      "After that, write a short summary of what was removed and why it was safe.") \
        == "holdout_near_dup"


def test_missing_repo_is_a_clean_error(host: dict[str, Path], tmp_path: Path) -> None:
    assert tg.main(_argv(host, "norepo", tmp_path / "o", "--repo", str(tmp_path / "absent"))) == 2


def test_dedupe_and_novelty(built: tuple) -> None:
    rows, report, _ = built
    norms = [tg.norm(r["prompt"]) for r in rows]
    assert len(set(norms)) == len(norms)
    real = {tg.norm(p["text"]) for p in PROMPTS["golden"] + PROMPTS["lcm"]}
    assert not set(norms) & real
    grams = [{g for g in tg.ngrams(tg.tokens(r["prompt"]), 4)} for r in rows]
    for a, b in itertools.combinations(range(len(rows)), 2):
        inter = len(grams[a] & grams[b])
        assert inter / (len(grams[a]) + len(grams[b]) - inter) < 0.8
    d = report["diversity"]["output"]
    for k in ("distinct_1", "distinct_2", "distinct_3", "self_bleu4"):
        assert 0.0 < d[k] <= 1.0


def test_diversity_metrics() -> None:
    same = [["a", "b", "c", "d", "e"]] * 4
    assert tg.self_bleu(same, sample=4, seed=1) == 1.0
    assert tg.distinct_n(same, 2) == 0.25
    disjoint = [[f"w{i}{j}" for j in range(6)] for i in range(5)]
    assert tg.self_bleu(disjoint, sample=5, seed=1) < 0.05
    assert tg.distinct_n(disjoint, 1) == 1.0
    assert tg.self_bleu([["x"]], sample=3, seed=1) is None


def test_caps_per_pattern_and_per_real_prompt(built: tuple) -> None:
    rows, report, _ = built
    import collections
    per_pattern = collections.Counter(r["source_pattern_hash"] for r in rows)
    assert max(per_pattern.values()) <= 6
    uses = collections.Counter(s["template_id"] for r in rows for s in r["sources"])
    assert max(uses.values()) <= 8


def test_coding_cap(host: dict[str, Path]) -> None:
    rows, report, _ = _run(host, "nocode", "--coding-cap", "0")
    assert rows and not any(r["category"] == "coding" for r in rows)
    rows, report, _ = _run(host, "cap20", "--coding-cap", "0.2", "--max-total", "40")
    coding = sum(1 for r in rows if r["category"] == "coding")
    assert coding / len(rows) <= 0.2 and len(rows) <= 40
    assert report["output"]["coding_share"] <= 0.2


def test_variants_of_one_real_prompt_share_a_task_group(built: tuple) -> None:
    """teacher_gen writes task_group as _meta.task_id; build.py unions rows on
    it, so slot-fill siblings can never split across train and eval-tools."""
    import collections
    rows, _, _ = built
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        assert r["task_group"] == "tg-" + r["sources"][0]["pattern_hash"]
        groups[r["sources"][0]["template_id"]].add(r["task_group"])
    assert all(len(g) == 1 for g in groups.values())
    siblings = [tid for tid, n in collections.Counter(r["template_id"] for r in rows).items() if n >= 2]
    assert siblings  # the fixture has real prompts with several variants
    for tid in siblings:
        mine = [r for r in rows if r["template_id"] == tid]
        assert len({r["id"] for r in mine}) == len(mine) and len({r["task_group"] for r in mine}) == 1


def test_flag_shares_exact(built: tuple) -> None:
    rows, _, _ = built
    agentic = [r for r in rows if r["mode"] == "agentic"]
    assert sum(r["flags"]["inject_errors"] for r in rows) == round(0.2 * len(agentic))
    assert sum(r["flags"]["force_compaction"] for r in rows) == round(0.2 * len(rows))
    assert not any(r["flags"]["inject_errors"] for r in rows if r["mode"] == "chat")


def test_recombination_keeps_intent(host: dict[str, Path]) -> None:
    gen = _generator(host, per_pattern=6, max_uses=8, coding_cap=0.3)
    rows = gen.generate("api")
    rec = [r for r in rows if r["transform"] == "recombine"]
    assert rec
    for r in rec:
        a, b = (gen.by_id[s["template_id"]] for s in r["sources"])
        assert a.intent == b.intent and a.id != b.id


# ── provenance: every output text traceable to fixture text + real values ──


def _independent_values(host: dict[str, Path]) -> set[str]:
    """Every value the host fixture could supply, collected without the module."""
    vals: set[str] = set(HOST["tools"]) | {j["name"] for j in HOST["cron_jobs"]}
    for rel, content in HOST["repo_files"].items():
        p = Path(rel)
        vals |= {rel, p.name} | {q.as_posix() for q in list(p.parents)[:-1]}
        if rel.startswith("src/prometheus/") and rel.count("/") >= 3:
            vals.add(rel.split("/")[2])
            if rel.startswith("src/prometheus/jobs/"):
                vals.add(p.stem)
        vals |= {m.group(2).strip() for m in re.finditer(r"(?m)^(#{1,3})\s+(.+)$", content)}
        vals |= set(re.findall(r"https://[^\s)]+", content))
    for rel, content in HOST["wiki"].items():
        if rel.startswith("topics/"):
            vals.add("wiki/" + rel)
            vals |= {m.group(1).strip() for m in re.finditer(r"(?m)^#{1,3}\s+(.+)$", content)}
    out = subprocess.run(["git", "-C", str(host["repo"]), "log", "--format=%H"], capture_output=True, text=True)
    vals |= set(out.stdout.split())
    return vals


_GENERATED = {
    "number": r"\d+(?:\.\d+)?", "version": r"[vV]?\d+(?:\.\d+)+", "iso_date": r"\d{4}-\d{2}-\d{2}",
    "weekday": r"(?i)(?:mon|tues|wednes|thurs|fri|satur|sun)day", "issue": r"#\d+",
    "cron": r"\d+ (?:\d+|\*/\d) \* \* (?:\*|\d(?:-\d)?)",
    "time": r"(?i)\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}:\d{2}(?::\d{2})?",
    "month_date": r"(?i)(?:[a-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[a-z]{3,9})",
    "sha": r"[0-9a-f]{7,40}",
}


def test_every_output_rebuilds_from_fixture_text_and_real_values(host: dict[str, Path], built: tuple) -> None:
    rows, _, _ = built
    real = {tg.h(tg.norm(p["text"])): p["text"].strip() for p in PROMPTS["golden"] + PROMPTS["lcm"]}
    values = _independent_values(host)
    shas = [v for v in values if re.fullmatch(r"[0-9a-f]{40}", v)]
    for r in rows:
        pieces = []
        for k, src in enumerate(r["sources"]):
            text = real[src["pattern_hash"]]  # the source is a fixture prompt
            s, e = src["span"]
            out, pos = [], s
            for sl in sorted((x for x in r["slots"] if x["source"] == k), key=lambda x: x["span"][0]):
                a, b = sl["span"]
                assert s <= a < b <= e
                out += [text[pos:a], sl["value"]]
                pos = b
                if sl["from"] == "original":
                    assert sl["value"] == text[a:b]
                elif sl["from"] == "generated":
                    assert re.fullmatch(_GENERATED[sl["type"]], sl["value"]), (sl["type"], sl["value"])
                elif sl["type"] == "sha":
                    assert any(x.startswith(sl["value"]) for x in shas)
                else:
                    assert sl["value"].rstrip("/") in values, (sl["type"], sl["value"])
            out.append(text[pos:e])
            piece = "".join(out)
            if pieces and not pieces[-1][-1:].isspace():
                pieces.append("\n")  # the only text not taken from a fixture prompt or a value
            pieces.append(piece)
        assert "".join(pieces).strip() == r["prompt"]


def test_no_word_outside_fixtures_values_or_generators(host: dict[str, Path], built: tuple) -> None:
    rows, _, _ = built
    vocab: set[str] = set()
    for t in [p["text"] for p in PROMPTS["golden"] + PROMPTS["lcm"]] + sorted(_independent_values(host)):
        vocab |= {w.lower() for w in re.findall(r"[A-Za-z]+", t)}
    vocab |= {m.lower() for m in tg._MONTH_FULL} | {m.lower()[:3] for m in tg._MONTH_FULL} | {"sept"}
    vocab |= {d.lower() for d in tg._WEEKDAYS} | {"am", "pm", "a", "m", "p", "st", "nd", "rd", "th", "of", "v"}
    for r in rows:
        text = r["prompt"]
        for sl in r["slots"]:
            if sl["type"] == "sha":  # hex, checked against git by the rebuild test
                text = text.replace(sl["value"], " ")
        stray = {w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in vocab}
        assert not stray, stray


def test_verify_provenance_catches_tampering(host: dict[str, Path]) -> None:
    gen = _generator(host, per_pattern=4, max_uses=6, coding_cap=0.3)
    rows = gen.generate("api")
    filled = next(r for r in rows if any(s["from"] not in ("original", "generated") for s in r["slots"]))
    assert tg.verify_provenance(filled, gen.by_id, gen.inv) == []
    r = copy.deepcopy(filled)
    r["prompt"] += " Also, a sentence nobody typed."
    assert tg.verify_provenance(r, gen.by_id, gen.inv)
    r = copy.deepcopy(filled)
    slot = next(s for s in r["slots"] if s["from"] not in ("original", "generated"))
    slot["value"] = "invented/value.py"
    r["prompt"] = r["prompt"]  # left as is: both the value and the rebuild must fail
    assert tg.verify_provenance(r, gen.by_id, gen.inv)
    r = copy.deepcopy(filled)
    r["sources"][0]["span"] = [1, r["sources"][0]["span"][1]]
    assert tg.verify_provenance(r, gen.by_id, gen.inv)


def test_generated_values_keep_their_shape() -> None:
    rng = random.Random(3)
    base = tg.date(2026, 11, 1)
    for typ, orig in [("number", "15"), ("number", "07"), ("number", "2.5"), ("version", "v0.9.17"),
                      ("iso_date", "2026-10-30"), ("month_date", "Oct 12th"), ("month_date", "3 of March"),
                      ("weekday", "friday"), ("time", "9am"), ("time", "14:30"), ("time", "7:05 P.M."),
                      ("cron", "0 6 * * *"), ("issue", "#42"), ("sha", "abc1234f")]:
        for _ in range(20):
            v = tg.gen_value(typ, orig, rng, base)
            assert tg.generated_shape_ok(typ, v), (typ, orig, v)
    assert tg.gen_value("weekday", "FRIDAY", rng, base).isupper()
    assert len(tg.gen_value("number", "07", rng, base)) == 2


def test_settings_are_checked(host: dict[str, Path]) -> None:
    with pytest.raises(tg.TasksetError):
        tg.Settings(coding_cap=1.5).check()
    with pytest.raises(tg.TasksetError):
        tg.Settings(per_pattern=0).check()
    assert tg.main(_argv(host, "bad name!", host["root"] / "o")) == 2
