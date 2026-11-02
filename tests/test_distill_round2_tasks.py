"""round2_tasks.py: round-2 tasksets from real-pattern rows, disjoint from eval-injection and
eval-longctx, with real failure patterns; steer.py and gen_stats.py. Synthetic fixtures only."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_distill_teacher_gen import _load, _taskset_row

r2t = _load("distill_round2_tasks", "round2_tasks.py")
ts = _load("distill_taskset_gen", "taskset_gen.py")
ev = _load("distill_build_eval_sets", "build_eval_sets.py")
steer = _load("distill_steer", "steer.py")
gs = _load("distill_gen_stats", "gen_stats.py")
tg = _load("distill_teacher_gen", "teacher_gen.py")


def _eval_manifest(seed: int = 4242) -> Any:
    """The full eval-injection design (every vector x style x goal slot) over synthetic docs,
    as build_eval_sets builds it: what the real manifest hashes, at another seed."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    docs = []
    for i in range(8):
        lines = [f"## Section {i}"] + [" ".join(random.Random(i * 100 + k).choices(words, k=12)) for k in range(40)]
        docs.append(ev.Doc(path=f"docs/n{i}.md", kind="docs", lines=lines))
        docs.append(ev.Doc(path=f"wiki/w{i}.md", kind="wiki", lines=lines))
        docs.append(ev.Doc(path=f"src/m{i}.py", kind="code", lines=[f"def f_{i}_{k}():" for k in range(40)]))
    b = ev.InjectionBuilder(docs, seed=seed, system_prompt="You are a test agent.", tools=[])
    cases, manifest = b.build()
    assert len(cases) == len(ev.VECTORS) * len(ev.STYLES) * len(ev.GOAL_KEYS)
    return ev.InjectionManifest(manifest)


# ── disjointness from eval-injection ─────────────────────────────────────────


def test_the_catalog_shares_no_goal_style_template_or_action_with_eval_injection() -> None:
    out = r2t.static_disjointness()
    assert out["pass"], out
    assert not set(r2t.GOALS) & set(ev.GOAL_KEYS) and not set(r2t.STYLES) & set(ev.STYLES)


def test_no_instruction_the_catalog_can_build_overlaps_the_eval_injection_manifest() -> None:
    manifest = _eval_manifest()
    n = 0
    for goal in r2t.GOALS:
        for style, templates in r2t.STYLES.items():
            for i in range(len(templates)):
                for k in range(6):
                    rng = random.Random(f"{goal}|{style}|{i}|{k}")
                    g = r2t.make_goal(goal, rng)
                    inst, _tid = r2t.render_instruction(style, i, g.action, rng)
                    rec = {"instruction": inst, "setup": g.setup_files, "fixtures": g.fixtures}
                    assert manifest.record_overlaps(rec) == {}, (goal, style, i)
                    n += 1
    assert n == len(r2t.GOALS) * sum(len(t) for t in r2t.STYLES.values()) * 6


def test_the_manifest_check_does_catch_an_eval_instruction() -> None:
    """The guard is not vacuous: an instruction that IS an eval case's is found."""
    manifest = _eval_manifest()
    rng = random.Random(1)
    goal = ev.make_goal("wipe_home", rng, "PRM-00000001")
    inst, _payload, _tid = ev.render_style("direct", goal.action, random.Random(0))
    assert manifest.overlaps(inst)


# ── the build ────────────────────────────────────────────────────────────────


def _git_repo(root: Path) -> Path:
    repo = root / "repo"
    files = {}
    for i in range(30):
        files[f"tests/test_mod_{i:02d}.py"] = "\n".join(
            [f"def test_case_{i}_{k}_unique_name():" if k % 5 == 0 else f"    value_{k} = {k}  # check {i}"
             for k in range(260)]) + "\n"
    for i in range(6):
        files[f"wiki/page_{i}.md"] = f"# Page {i} heading words\n" + "\n".join(
            f"Line {k} of wiki page {i} with plain words." for k in range(60)) + "\n"
    files["docs/guide.md"] = "# Guide\n" + "\n".join(f"guide line {k}" for k in range(100)) + "\n"
    files["docs/sprints/DISTILL-LOG.md"] = "# notes\n" + "\n".join("x" for _ in range(50)) + "\n"
    files["src/eval_used.py"] = "\n".join(f"x_{k} = {k}" for k in range(80)) + "\n"
    files["docs/private.md"] = "# Private\n" + "\n".join("contact me at someone@example.com" for _ in range(30)) + "\n"
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"], ["update-ref", "refs/remotes/origin/main", "HEAD"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env, capture_output=True)
    return repo


def _fail_inputs(root: Path) -> tuple[Path, Path]:
    fails = [
        {"_meta": {"tool_name": "bash", "failure": {"error_detail": "Command timed out after 120 seconds"}}},
        {"_meta": {"tool_name": "bash", "failure": {"error_detail": "Command timed out after 120 seconds"}}},
        {"_meta": {"tool_name": "write_file", "failure": {"error_detail": "1 validation error for FileWriteToolInput"}}},
        {"_meta": {"tool_name": "edit_file", "failure": {"error_detail": "old_str was not found in the file"}}},
    ]
    fp = root / "fail_traces.jsonl"
    fp.write_text("".join(json.dumps(r) + "\n" for r in fails))
    db = root / "lcm.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE lcm_messages (id INTEGER PRIMARY KEY, content_json TEXT)")
    blocks = [
        [{"type": "tool_use", "id": "u1", "name": "web_fetch", "input": {"url": "https://a.example/x"}}],
        [{"type": "tool_result", "tool_use_id": "u1", "is_error": True,
          "content": "web_fetch failed: Client error '404 Not Found' for url 'https://a.example/x'\n"
                     "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"}],
        [{"type": "tool_use", "id": "u2", "name": "read_file", "input": {"path": "/home/someone/x"}}],
        [{"type": "tool_result", "tool_use_id": "u2", "is_error": True, "content": "File not found: /home/someone/x"}],
        [{"type": "tool_use", "id": "u3", "name": "task_list", "input": {}}],
        [{"type": "tool_result", "tool_use_id": "u3", "is_error": True,
          "content": "Tool task_list timed out after 60s and was cancelled. If this tool legitimately needs longer, "
                     "it can set a higher limit."}],
    ]
    con.executemany("INSERT INTO lcm_messages (content_json) VALUES (?)", [(json.dumps(b),) for b in blocks])
    con.commit()
    con.close()
    return fp, db


def _base_rows() -> list[dict[str, Any]]:
    rows = []
    prompts = [("Find where test_case_3_0_unique_name is defined in tests/test_mod_03.py", "agentic", "coding"),
               ("Explain what tests/test_mod_05.py checks", "agentic", "other"),
               ("Check the wiki page about page 2 and summarize it", "agentic", "memory/wiki"),
               ("Delete the old cron job for the backups", "agentic", "sentinel/daemon"),
               ("Run the status check for the indexer service", "agentic", "sentinel/daemon"),
               ("Edit the README to mention the new flag", "agentic", "coding"),
               ("What is a good way to plan a release week?", "chat", "chat-mode")]
    for prompt, mode, cat in prompts:
        r = _taskset_row(prompt, suppress_tools=mode == "chat")
        r["category"] = cat
        r["subcategory"] = {"coding": "coding", "memory/wiki": "wiki", "sentinel/daemon": "daemon-ops"}.get(cat, "general")
        r["intent"] = {"Find": "find", "Explain": "explain", "Check": "check", "Delete": "edit", "Run": "run",
                       "Edit": "edit"}.get(prompt.split()[0], "question")
        if "tests/test_mod_03.py" in prompt:
            s = prompt.index("tests/test_mod_03.py")
            r["slots"] = [{"name": "path_1", "type": "path", "value": "tests/test_mod_03.py", "from": "repo",
                           "source": 0, "span": [s, s + len("tests/test_mod_03.py")]}]
        assert ts.validate_task(r) == [], ts.validate_task(r)
        rows.append(r)
    return rows


@pytest.fixture
def built(tmp_path: Path) -> tuple[dict[str, Any], Path, Any]:
    repo = _git_repo(tmp_path)
    fails, lcm = _fail_inputs(tmp_path)
    base = tmp_path / "base.jsonl"
    base.write_text("".join(json.dumps(r) + "\n" for r in _base_rows()))
    lc_eval = tmp_path / "eval-longctx.jsonl"
    lc_eval.write_text(json.dumps({"id": "c1", "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": "src/eval_used.py"})}}]},
        {"role": "user", "content": '<file path="tests/test_mod_29.py">\nx\n</file>'}]}) + "\n")
    manifest = _eval_manifest()
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest.data))
    # an eval-tools case whose user message shares 8 tokens with one real row: that row is dropped
    et = tmp_path / "eval-tools.jsonl"
    et.write_text(json.dumps({"id": "e1", "meta": {"task_id": "x"}, "messages": [
        {"role": "user", "content": "please Run the status check for the indexer service now"}]}) + "\n")
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"schema": "distill-judge-calibration/2", "decision": {"decision": "use-local-judge"},
                               "by_category": [{"category": "coding", "apply": True},
                                               {"category": "sentinel/daemon", "apply": True},
                                               {"category": "other", "apply": False}]}))
    out = tmp_path / "out"
    a = argparse.Namespace(name="real-t-r2", out=str(out), base=str(base), seed=7, injection_manifest=str(mpath),
                           eval_longctx=[str(lc_eval)], fail_traces=str(fails), lcm_snapshot=str(lcm),
                           repo=str(repo), repo_ref="origin/main", pack=None, calibration=str(cal), tokenizer_dir=None,
                           steer=str(tmp_path / "no-steer.json"), inj_variants=2, lc_variants=1, eval_tools=[str(et)],
                           ctx_buckets="32k,64k,128k", live_prompts="skip")
    old = dict(r2t.CTX_BUCKETS)
    r2t.CTX_BUCKETS.update({"32k": 14_000, "64k": 20_000, "128k": 26_000})  # a small synthetic repo
    try:
        rep = r2t.build(a)
    finally:
        r2t.CTX_BUCKETS.clear()
        r2t.CTX_BUCKETS.update(old)
    return rep, out, (a, lc_eval, mpath)


def test_build_writes_valid_round2_rows_for_every_area(built: Any) -> None:
    rep, out, _ = built
    rows = ts.load_taskset(out / "real-t-r2.jsonl")
    assert len(rows) == rep["combined"]["rows"] == sum(rep["rows_by_area"].values())
    assert rep["stats"]["base_rows_eval_tools_overlap_dropped"] == 1
    assert not any("indexer service" in r["prompt"] for r in rows)
    areas = {r["r2"]["area"] for r in rows}
    assert areas == set(ts.R2_AREAS)
    for r in rows:
        base = r["r2"]["base_id"]
        assert base == "ts-" + ts.h(ts.norm(r["prompt"])) and r["id"] == ts.r2_task_id(r["prompt"], r["r2"]["variant"])
        assert r["provenance"] == ts.PROVENANCE and r["r2"]["needs_ctx"] > 0
    # a request about the owner's live hosts (here: a cron job) is daily-agent material only
    cron = [r for r in rows if "cron job" in r["prompt"]]
    assert cron and {r["r2"]["area"] for r in cron} == {"daily-agent"}
    assert rep["stats"]["conditioned_areas_live_host_prompts_skipped"] == 1
    assert all(not r2t.live_host_prompt(r["prompt"]) for r in rows if r["r2"]["area"] in r2t.CONDITIONED_AREAS)
    # the injection rows never pair a goal with a request that would legitimately do it
    for r in rows:
        inj = r["r2"].get("injection")
        if inj and inj["goal"] == "cron_purge":
            assert "Delete the old cron job" not in r["prompt"]
        if inj and inj["goal"] == "planted_link":
            assert "README" not in r["prompt"]
    # every row runs as a teacher_gen task (validated by taskset_gen, conditions applied)
    tasks = tg.tasks_from_taskset_rows(rows, Path("."))
    assert all(t.r2 for t in tasks)
    assert {t.kind for t in tasks if t.area == "error-recovery"} == {"error-recovery"}
    assert all(t.failure and not t.inject for t in tasks if t.area == "error-recovery")
    assert all(t.context_reads for t in tasks if t.area == "long-context")
    assert all(t.adversarial for t in tasks if t.area == "injection-resistance")
    # the planned verifiers follow the calibration
    assert rep["planned_verifier"]["injection-resistance"] == {"deterministic": rep["rows_by_area"]["injection-resistance"]}
    daily = [r for r in rows if r["r2"]["area"] == "daily-agent"]
    assert {r["r2"]["verifier"] for r in daily if r["category"] == "sentinel/daemon"} == {"calibrated-judge"}
    assert {r["r2"]["verifier"] for r in daily if r["category"] == "chat-mode"} == {"tool-success"}


def test_long_context_reads_real_files_and_never_an_eval_source(built: Any) -> None:
    rep, out, (a, lc_eval, _m) = built
    rows = ts.load_taskset(out / "real-t-r2-longctx.jsonl")
    assert rows and rep["long_context"]["files_read_eval_overlap"] == 0
    eval_paths = r2t.eval_longctx_paths([lc_eval])
    assert eval_paths == {"src/eval_used.py", "tests/test_mod_29.py"}
    for r in rows:
        reads = [x["path"] for x in r["r2"]["longctx"]["reads"]]
        assert all(p.startswith("Prometheus/") for p in reads)
        rel = {p[len("Prometheus/"):] for p in reads}
        assert not rel & eval_paths
        assert not any(p.startswith("docs/sprints/") or p == "docs/private.md" for p in rel)
        assert r["r2"]["ctx_bucket"] in r2t.CTX_BUCKETS
    named = [r for r in rows if "tests/test_mod_03.py" in r["prompt"]]
    for r in named:
        reads = [x["path"] for x in r["r2"]["longctx"]["reads"]]
        assert "Prometheus/tests/test_mod_03.py" in reads
        g = r["r2"]["longctx"]["grounding"]
        assert g and g["path"] == "Prometheus/tests/test_mod_03.py" and r["r2"]["verifier"] == "deterministic"
        assert all(t not in r["prompt"] for t in g["terms"])


def test_lc2_rows_list_where_their_reads_are_before_reading_them(built: Any, tmp_path: Path) -> None:
    rep, out, (a, lc_eval, mpath) = built
    rows = ts.load_taskset(out / "real-t-r2-longctx.jsonl")
    assert rows and rep["long_context"]["style"] == "lc2"
    assert rep["long_context"]["with_listings"] == len(rows)
    for r in rows:
        lc = r["r2"]["longctx"]
        assert r["r2"]["variant"].startswith("lc2:")
        pats = [x["pattern"] for x in lc["listings"]]
        assert all(p.startswith("Prometheus/") and p.endswith("*") for p in pats)
        # every listing is a directory a read came from (whole, or the read's name prefix in tests/)
        dirs = {str(Path(x["path"]).parent) for x in lc["reads"]}
        assert all(str(Path(p).parent) in dirs for p in pats)
        assert ts.validate_task(r) == []
    tasks = tg.tasks_from_taskset_rows(rows, Path("."))
    assert all(t.context_listings for t in tasks)
    # the first round-2 style still builds its own rows (variant lc:, no listings)
    rep1 = _rebuild(a, tmp_path / "v1", lc_style="lc")
    old = ts.load_taskset(tmp_path / "v1" / "real-t-r2-longctx.jsonl")
    assert old and all(r["r2"]["variant"].startswith("lc:") and "listings" not in r["r2"]["longctx"] for r in old)
    assert rep1["long_context"]["with_listings"] == 0
    assert not {r["id"] for r in old} & {r["id"] for r in rows}


def test_relevance_puts_the_files_a_request_is_about_first_and_listings_fit_their_budget() -> None:
    def f(path: str, text: str) -> Any:
        lines = text.split("\n")
        return r2t.SrcFile(path, lines, 100, frozenset(),
                           r2t.words_of(path.replace("/", " ").replace(".", " ") + "\n" + text))
    pool = [f(f"tests/test_other_{i}.py", f"def test_{i}():\n    assert value == {i}") for i in range(20)]
    pool.append(f("src/pkg/mcp_gate.py", "MCP servers ship off by default\nclass McpGate: ..."))
    pool.append(f("tests/test_mcp_gate.py", "def test_mcp_off_by_default(): ..."))
    relv = r2t.Relevance(pool)
    assert "mcp" in relv.keywords("what else ships off by default? i think mcp is like that too")
    scores = relv.scores("what else ships off by default? i think mcp is like that too")
    top = sorted(scores, key=lambda p: -scores[p])[:2]
    assert set(top) == {"src/pkg/mcp_gate.py", "tests/test_mcp_gate.py"}
    row = {"id": "ts-x", "slots": []}
    chosen, _t, _d = r2t.plan_reads(row, pool, 250, 7, "lc2:32k|ts-x", relevance=scores)
    assert {c.path for c in chosen[:2]} == set(top)
    children = {"tests": [f"tests/test_x_{i}.py" for i in range(200)] + ["tests/test_mcp_gate.py"],
                "src/pkg": ["src/pkg/__init__.py", "src/pkg/mcp_gate.py"]}
    listings, used = r2t.plan_listings(chosen[:2], children, lambda t: len(t.split()), 10_000)
    pats = {x["pattern"]: x for x in listings}
    assert pats["Prometheus/src/pkg/*"]["limit"] == 2               # a small directory, whole
    assert pats["Prometheus/tests/test_mcp*"]["entries"] == 1       # tests/ (big): by the read's name prefix
    many = [f(f"tests/test_k{i}_x.py", "x") for i in range(9)]
    capped, _u = r2t.plan_listings(many, {"tests": [x.path for x in many] * 20}, lambda t: 1, 10_000)
    assert len(capped) == r2t.LC_LIST_MAX_CALLS
    none, used0 = r2t.plan_listings(chosen[:2], children, lambda t: 10_000, 50)
    assert none == [] and used0 == 0


def test_failure_patterns_come_from_the_harvest_and_never_carry_a_home_path(built: Any) -> None:
    rep, out, _ = built
    fams = rep["failures"]["families"]
    assert set(fams) == {"bash_timeout", "edit_old_str_missing", "web_fetch_client_error", "read_file_missing",
                         "tool_timeout"}
    assert fams["bash_timeout"]["count"] == 2 and fams["web_fetch_client_error"]["sticky"] is True
    rows = ts.load_taskset(out / "real-t-r2-errrec.jsonl")
    by = {f["family"]: f for f in rows[0]["r2"]["failure"]["families"]}
    assert by["web_fetch_client_error"]["output"].startswith("web_fetch failed: Client error '404 Not Found' for url '{url}'")
    assert by["read_file_missing"]["output"] == "File not found: {path}"
    assert by["tool_timeout"]["output"].startswith("Tool {tool} timed out after {n}s") and by["tool_timeout"]["tool"] == "*"
    assert "/home/" not in json.dumps(rows)
    # never a model mistake (validation errors) as an injected failure
    assert "validation" not in json.dumps(rep["failures"])


def test_check_disjoint_passes_on_the_build_and_fails_on_a_planted_eval_source(built: Any, tmp_path: Path) -> None:
    rep, out, (a, lc_eval, mpath) = built
    ns = argparse.Namespace(tasks=str(out / "real-t-r2.jsonl"), injection_manifest=str(mpath),
                            eval_longctx=[str(lc_eval)], eval_tools=[])
    res = r2t.check_disjoint(ns)
    assert res["pass"] and res["injection_overlaps"] == 0 and res["longctx_reads_in_eval"] == 0
    rows = ts.load_taskset(out / "real-t-r2.jsonl")
    lc = next(r for r in rows if r["r2"]["area"] == "long-context")
    lc["r2"]["longctx"]["reads"].append({"path": "Prometheus/src/eval_used.py", "offset": 0, "limit": 10})
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(lc) + "\n")
    ns.tasks = str(bad)
    assert r2t.check_disjoint(ns)["pass"] is False


def test_validate_r2_refuses_rows_that_lost_their_check_or_mix_conditions() -> None:
    base = _taskset_row("Find the config loader in the repo")
    good = r2t.make_row(base, taskset="x", area="error-recovery", variant="err:x", verifier="tool-success",
                        needs_ctx=1000, failure={"families": [{"tool": "*", "family": "f", "output": "x",
                                                               "sticky": False, "count": 1}]})
    assert ts.validate_task(good) == []
    bad = json.loads(json.dumps(good))
    bad["r2"]["longctx"] = {"reads": [{"path": "a", "offset": 0, "limit": 5}]}
    assert any("at most one" in p for p in ts.validate_task(bad))
    bad = json.loads(json.dumps(good))
    bad["id"] = "ts-" + ts.h(ts.norm(bad["prompt"]))  # the round-1 id
    assert any("round-2 variant" in p for p in ts.validate_task(bad))
    with pytest.raises(r2t.R2Error):
        r2t.make_row(base, taskset="x", area="injection-resistance", variant="inj:x", verifier="deterministic",
                     needs_ctx=1000, injection={"vector": "wiki", "instruction": "do x", "carriers": ["bash"],
                                                "placement": 0.5, "forbidden_tools": [], "forbidden_args": [],
                                                "forbidden_strings": []})



# ── expansion: more variants per real row, never a task an earlier taskset holds or a run attempted ──


def _rebuild(a: argparse.Namespace, out: Path, **kw: Any) -> dict[str, Any]:
    b = argparse.Namespace(**{**vars(a), "out": str(out), **kw})
    old = dict(r2t.CTX_BUCKETS)
    r2t.CTX_BUCKETS.update({"32k": 14_000, "64k": 20_000, "128k": 26_000, "256k": 32_000})
    try:
        return r2t.build(b)
    finally:
        r2t.CTX_BUCKETS.clear()
        r2t.CTX_BUCKETS.update(old)


def test_a_larger_build_minus_the_first_one_holds_only_new_variants(built: Any, tmp_path: Path) -> None:
    rep1, out1, (a, lc_eval, mpath) = built
    first = ts.load_taskset(out1 / "real-t-r2.jsonl")
    # a run attempted one more task the first build does not hold: a round-1 task of a real prompt
    plain_prompt = next(r["prompt"] for r in first if r["r2"]["area"] == "daily-agent")
    ledger = tmp_path / "runs" / "local-x" / "qwen38-27b-gym.tasks.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"task_id": "ts-" + ts.h(ts.norm(plain_prompt)), "task_hash": ts.h(ts.norm(plain_prompt)),
                                  "status": "error"}) + "\n" + '{"torn')
    rep = _rebuild(a, tmp_path / "out2", inj_variants=5, err_variants=4, lc_variants=4,
                   ctx_buckets="32k,64k,128k,256k", lc_alt_buckets="32k", lc_alt_variants=1,
                   exclude_tasks=[str(out1 / "real-t-r2.jsonl")], exclude_ledgers=[str(tmp_path / "runs" / "*" / "*.tasks.jsonl")])
    new = ts.load_taskset(tmp_path / "out2" / "real-t-r2.jsonl")
    old_ids = {r["id"] for r in first}
    assert new and not {r["id"] for r in new} & old_ids
    # the exclusion took exactly the first build's rows (same seed: its variants come first in every row)
    ex = rep["excluded"]["by_area"]
    assert ex["injection-resistance"] == rep1["rows_by_area"]["injection-resistance"]
    assert ex["error-recovery"] == rep1["rows_by_area"]["error-recovery"]
    assert ex["daily-agent"] == rep1["rows_by_area"]["daily-agent"] and rep["rows_by_area"]["daily-agent"] == 0
    assert rep["rows_by_area"]["coding"] == 0
    # injection: other goal x style x vector combinations of the same real rows, a goal once per row
    inj = [r for r in new if r["r2"]["area"] == "injection-resistance"]
    assert inj
    per_row: dict[str, list[str]] = {}
    for r in first + new:
        if r["r2"]["area"] == "injection-resistance":
            per_row.setdefault(r["r2"]["base_id"], []).append(r["r2"]["injection"]["goal"])
    assert all(len(g) == len(set(g)) and len(g) <= 5 for g in per_row.values())
    # error recovery: focus variants, one real family alone, never a web family, edit only where it edits
    err = [r for r in new if r["r2"]["area"] == "error-recovery"]
    assert err and all(r["r2"]["variant"].startswith("err:focus:") for r in err)
    for r in err:
        fams = r["r2"]["failure"]["families"]
        assert len(fams) == 1 and fams[0]["family"] == r["r2"]["variant"].split(":")[2]
        assert fams[0]["tool"] not in ("web_fetch", "web_search")
        if fams[0]["tool"] == "edit_file":
            assert r["category"] == "coding" or r["intent"] in ("edit", "fix", "create")
    tasks = tg.tasks_from_taskset_rows(new, Path("."))
    assert all(t.failure for t in tasks if t.area == "error-recovery")
    # long context: the other buckets (256k too) and another file mix in 32k, never an eval source
    lc = [r for r in new if r["r2"]["area"] == "long-context"]
    assert {r["r2"]["ctx_bucket"] for r in lc} >= {"256k"}
    alt = [r for r in lc if r["r2"]["variant"] == "lc2:32k:alt1"]
    assert alt
    for r in alt:
        main = next((x for x in first + new if x["r2"]["base_id"] == r["r2"]["base_id"]
                     and x["r2"]["variant"] == "lc2:32k"), None)
        if main is not None:
            assert {x["path"] for x in r["r2"]["longctx"]["reads"]} != {x["path"] for x in main["r2"]["longctx"]["reads"]}
    assert all(r["r2"]["needs_ctx"] <= r2t.MAX_NEEDS_CTX for r in lc)
    assert set(rep["fits"]) == set(r2t.TEACHER_CAPS)
    ns = argparse.Namespace(tasks=str(tmp_path / "out2" / "real-t-r2.jsonl"), injection_manifest=str(mpath),
                            eval_longctx=[str(lc_eval)], eval_tools=[])
    assert r2t.check_disjoint(ns)["pass"]


def test_a_plain_row_whose_prompt_a_run_attempted_is_not_made_again(built: Any, tmp_path: Path) -> None:
    _rep1, _out1, (a, _lc, _m) = built
    rows = ts.load_taskset(_out1 / "real-t-r2.jsonl")
    plain = next(r for r in rows if r["r2"]["area"] == "daily-agent")
    ledger = tmp_path / "l.tasks.jsonl"
    ledger.write_text(json.dumps({"task_id": "ts-other", "task_hash": ts.h(ts.norm(plain["prompt"])),
                                  "status": "kept"}) + "\n")
    rep = _rebuild(a, tmp_path / "out3", areas="daily-agent,coding", exclude_ledgers=[str(ledger)])
    got = ts.load_taskset(tmp_path / "out3" / "real-t-r2.jsonl")
    assert {r["r2"]["area"] for r in got} <= {"daily-agent", "coding"}
    assert plain["prompt"] not in {r["prompt"] for r in got} and rep["excluded"]["by_area"]["daily-agent"] == 1
    with pytest.raises(r2t.R2Error, match="no file matches"):
        _rebuild(a, tmp_path / "out4", exclude_ledgers=[str(tmp_path / "nothing-*.jsonl")])
    with pytest.raises(r2t.R2Error, match="--areas"):
        _rebuild(a, tmp_path / "out5", areas="daily")


def test_merge_keeps_each_task_once_and_writes_the_area_files(built: Any, tmp_path: Path) -> None:
    rep1, out1, (a, _lc, _m) = built
    _rebuild(a, tmp_path / "more", inj_variants=4, exclude_tasks=[str(out1 / "real-t-r2.jsonl")],
             areas="injection-resistance")
    more = ts.load_taskset(tmp_path / "more" / "real-t-r2.jsonl")
    ns = argparse.Namespace(name="real-t-m", out=str(tmp_path / "m"), steer=str(tmp_path / "no-steer.json"),
                            tasks=[str(out1 / "real-t-r2.jsonl"), str(tmp_path / "more" / "real-t-r2.jsonl"),
                                   str(out1 / "real-t-r2-injection.jsonl")],
                            exclude_tasks=[], exclude_ledgers=[])
    rep = r2t.merge(ns)
    got = ts.load_taskset(tmp_path / "m" / "real-t-m.jsonl")
    assert len(got) == len({r["id"] for r in got}) == rep1["combined"]["rows"] + len(more)
    assert rep["stats"]["duplicate_id"] == rep1["rows_by_area"]["injection-resistance"]
    inj = ts.load_taskset(tmp_path / "m" / "real-t-m-injection.jsonl")
    assert len(inj) == rep["rows_by_area"]["injection-resistance"] == rep1["rows_by_area"]["injection-resistance"] + len(more)
    # a filtered merge: only the long-context rows of some buckets
    lc_rows = ts.load_taskset(out1 / "real-t-r2-longctx.jsonl")
    one = lc_rows[0]["r2"]["ctx_bucket"]
    rep2 = r2t.merge(argparse.Namespace(**{**vars(ns), "name": "real-t-lc", "areas": "long-context",
                                           "ctx_buckets": one}))
    got2 = ts.load_taskset(tmp_path / "m" / "real-t-lc.jsonl")
    assert got2 and {r["r2"]["ctx_bucket"] for r in got2} == {one}
    assert len(got2) == sum(1 for r in lc_rows if r["r2"]["ctx_bucket"] == one) and rep2["stats"]["filtered_out"] > 0
    # a round-1 taskset is refused
    r1 = tmp_path / "r1.jsonl"
    r1.write_text(json.dumps(_base_rows()[0]) + "\n")
    ns.tasks, ns.name = [str(r1)], "x"
    with pytest.raises(r2t.R2Error, match="not a round-2 taskset"):
        r2t.merge(ns)

# ── steer.py ─────────────────────────────────────────────────────────────────


def _summary(inj: list[bool], lc: dict[str, list[bool]], tools: dict[str, bool], sanity: list[bool]) -> dict:
    lc_out = {f"lc-{b}-{i}": ok for b, oks in lc.items() for i, ok in enumerate(oks)}
    return {"sets": {"eval-injection": {"outcomes": {f"i{i}": ok for i, ok in enumerate(inj)}},
                     "eval-longctx": {"outcomes": lc_out,
                                      "case_bucket": {k: k.split("-")[1] for k in lc_out}},
                     "eval-tools": {"outcomes": tools},
                     "sanity": {"outcomes": {f"s{i}": ok for i, ok in enumerate(sanity)}}}}


def test_defaults_follow_the_operator_order_and_flatten_to_buckets() -> None:
    cfg = steer.validate(steer.default_config())
    w = cfg["weights"]
    assert w["injection-resistance"] > w["error-recovery"] > w["long-context"] and w["coding"] <= steer.CODING_MAX
    flat = steer.flatten(cfg)
    assert abs(sum(flat.values()) - 1.0) < 1e-3 and "daily-agent:chat-mode" in flat
    assert steer.bucket_of("daily-agent", "other") == "daily-agent:other"


def test_steer_moves_weight_to_the_weakest_area_and_keeps_coding_small(tmp_path: Path) -> None:
    facts = {"t1": {"category": "coding", "error_recovery": False},
             "t2": {"category": "other", "error_recovery": True},
             "t3": {"category": "memory/wiki", "error_recovery": False}}
    sm = _summary([False] * 9 + [True], {"32k": [True] * 9 + [False], "256k": [False] * 10},
                  {"t1": True, "t2": False, "t3": True}, [True] * 9 + [False])
    scores = steer.area_scores([sm], facts)
    assert scores["injection-resistance"]["rate"] == 0.1 and scores["long-context"]["rate"] == 0.9
    assert "long-context:256k" not in scores  # beyond what round 2 generates
    cfg = steer.steer(scores)
    d = steer.validate(steer.default_config())["weights"]
    assert cfg["weights"]["injection-resistance"] > d["injection-resistance"]
    assert cfg["weights"]["long-context"] < d["long-context"]
    assert cfg["weights"]["coding"] <= steer.CODING_MAX + 1e-6
    assert any("injection-resistance" in line for line in cfg["reasoning"])
    out = tmp_path / "steer.json"
    sp = tmp_path / "q4.summary.json"
    sp.write_text(json.dumps(sm))
    assert steer.main(["steer", "--scorecard", str(sp), "--out", str(out)]) == 0
    assert steer.load(out)["basis"] == "scorecard"


# ── gen_stats.py ─────────────────────────────────────────────────────────────


def test_gen_stats_counts_verified_rows_per_hour_by_area_and_category(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    run = tmp_path / "runs" / "local-x"
    run.mkdir(parents=True)
    evs = [
        {"ts": now - 30 * 3600, "status": "kept", "rows": 4, "domain": "coding", "task_id": "a"},
        {"ts": now - 3600, "status": "kept", "rows": 2, "domain": "other", "area": "injection-resistance",
         "task_id": "b"},
        {"ts": now - 1800, "status": "rejected", "rows": 0, "domain": "other", "area": "error-recovery",
         "task_id": "c"},
        {"ts": now - 600, "status": "kept", "rows": 3, "domain": "memory/wiki", "area": "error-recovery",
         "task_id": "d"},
        {"ts": now - 500, "status": "relaxed", "rows": 5, "domain": "other", "area": "daily-agent", "task_id": "e"},
    ]
    (run / "qwen38-27b-gym.tasks.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs) + '{"torn')
    out = gs.stats([str(tmp_path / "runs" / "*")], 24, now)
    last = out["runs"]["local-x"]["last_hours"]
    assert last["rows"] == 5 and last["by_area"] == {"error-recovery": 3, "injection-resistance": 2}
    assert last["by_category"] == {"memory/wiki": 3, "other": 2} and last["rows_per_hour"] == round(5 / 24, 2)
    assert last["active_hours"] == 1 and last["keep_rate"] == round(2 / 4, 3)
    tot = out["all"]["since_start"]
    assert tot["rows"] == 9 and tot["by_area"]["round-1"] == 4
    assert "local-x" in gs.render(out)


def test_live_host_requests_are_recognised() -> None:
    for live in ("merge PR #534 once the tests pass", "restart the indexer service on the mini",
                 "check the 4090 GPU memory", "send the summary to my email"):
        assert r2t.live_host_prompt(live), live
    for offline in ("Explain what tests/test_mod_05.py checks", "Summarize the wiki page about memory",
                    "Find where the config loader is defined"):
        assert not r2t.live_host_prompt(offline), offline
