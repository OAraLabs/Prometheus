"""WP-X.33 — how far does GEPA's judge move on a change that means nothing?

This is the measurement behind ``learning.gepa_min_margin: 0.15``. It runs
GEPA's own scoring path (``GEPAOptimizer._score`` → ``PrometheusJudge.evaluate``
→ ``parsed_score``) against a judge, on PUBLIC or SYNTHETIC content only: the
three package builtins, three short auto-style skills and every run, all
written below. It prints aggregates.

Per skill (3 runs each), the judge scores:
  original ×2   the same text twice — is the judge even deterministic?
  surface  ×5   formatting-only rewrites — the NOISE a margin must clear
  degraded ×3   steps removed / swapped in from another skill / scrambled
  improved ×1   adds exactly what the runs show was missing

Run (from a checkout, against a local judge — nothing here is user data):

    PYTHONPATH=src python docs/audits/gepa-margin/judge_noise.py \
        qwen2.5:7b-instruct src/prometheus/skills/builtin

Results, 2026-09-26, the mini, qwen2.5:7b-instruct on loopback Ollama (the
configured ``evals.judge_model``). Two runs of 198 judge calls each: run 1 from
the branch before WP-X.22 merged (GEPA's interim strict re-read of the reply),
run 2 from the final code (the judge's verdict status). 0 unparseable in both.

                                            run 1          run 2
  same text scored twice, pairs differing   1 of 18        0 of 18
  surface rewrites (30), per-run |delta|    mean .031      mean .031
                                            p90 .20        p90 .10
                                            max .30        max .50
  ... on the 3-run mean                     p95 .133       p95 .133
                                            max .167       max .167
  rewrites passing the WHOLE rule
  (mean gain >= margin, lower on no run)
     at a margin of 0.05                    3 of 30        2 of 30
     at 0.10                                0 of 30        1 of 30
     at 0.15                                0 of 30        0 of 30
  degradations (18): mean drop              .37            .356
     dropping by >= 0.10                    18 of 18       18 of 18
  evidence-matching improvements (6)        +.20 passes; the rest +.13 at
                                            most, or lower on a run — in
                                            both runs

0.15 is the smallest margin at which no formatting-only rewrite passed in
either run (0 of 60; 1 of 60 at 0.10, 5 of 60 at 0.05). With three variants
per candidate, 0.10 would stage a noise-only proposal about one time in
twenty. The one real improvement that cleared 0.10 (+0.20) clears 0.15 too.
This judge rewards real fixes weakly, so GEPA will propose rarely; the margin
errs that way on purpose — a false proposal costs a person's review, a missed
one costs nothing. Sixty comparisons of thirty rewrites bound the rate only
loosely.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
from pathlib import Path

from prometheus.evals.judge import PrometheusJudge
from prometheus.learning import gepa_evidence as ev
from prometheus.learning.gepa import GEPAOptimizer, GEPAReport

JUDGE_URL = "http://127.0.0.1:11434"
JUDGE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b-instruct"
BUILTIN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("src/prometheus/skills/builtin")


def C(tool, inp, ok=True, et=None, err=""):
    return ev.CallRecord(tool=tool, ok=ok, error_type=None if ok else (et or "tool_error"),
                         input_text=json.dumps(inp), error_text=err)


def R(request, calls, reply="done"):
    return ev.LoadRun(load_ts=0.0, session_id="synthetic", end_ts=0.0, request=request,
                      request_withheld=False,
                      calls=tuple(calls), calls_not_shown=0, replied=True, reply=reply, bounded=True)


def fm(name, desc, body):
    return f"---\nname: {name}\ndescription: {desc}\n---\n{body}"


SYN = {
    "restart-web-service": (
        fm("restart-web-service", "Restart the web service after a config change and confirm it is healthy",
           "# Restart web service\n\n## When to use\nAfter changing the web service's configuration.\n\n"
           "## Steps\n1. Run `systemctl --user restart web.service`.\n2. Check `systemctl --user status web.service`.\n"
           "3. Report the result.\n"),
        "## Steps\n1. Run `systemctl --user restart web.service`.\n2. If it fails, read "
        "`journalctl --user -u web.service -n 50` and fix the config error it names before retrying.\n"
        "3. Wait for health: retry `curl -fsS localhost:8080/health` for up to 30 s.\n4. Report the result.\n",
        [R("restart the web service, I changed the port", [
            C("bash", {"command": "systemctl --user restart web.service"}),
            C("bash", {"command": "curl -fsS localhost:8080/health"}, False, "nonzero_exit", "curl: (7) connection refused"),
            C("bash", {"command": "sleep 5 && curl -fsS localhost:8080/health"})], "The service is back and healthy."),
         R("apply the new web config", [
            C("bash", {"command": "systemctl --user restart web.service"}, False, "nonzero_exit", "Job for web.service failed"),
            C("bash", {"command": "journalctl --user -u web.service -n 50"}),
            C("edit_file", {"path": "web.toml"}),
            C("bash", {"command": "systemctl --user restart web.service"}),
            C("bash", {"command": "curl -fsS localhost:8080/health"})], "Fixed a typo in web.toml; healthy."),
         R("restart web after the TLS change", [
            C("bash", {"command": "systemctl --user restart web.service"}),
            C("bash", {"command": "systemctl --user status web.service"}),
            C("bash", {"command": "curl -fsS localhost:8080/health"})], "Restarted, healthy.")]),
    "rotate-logs": (
        fm("rotate-logs", "Rotate and compress large application logs",
           "# Rotate logs\n\n## Steps\n1. Find large logs with `du -sh /var/log/app/*`.\n"
           "2. Run `logrotate -f /etc/logrotate.d/app`.\n3. Confirm the new log file exists.\n"),
        "## Steps\n1. Find large logs with `du -sh /var/log/app/*`.\n2. Run `logrotate -f /etc/logrotate.d/app`.\n"
        "3. Send the app `SIGHUP` so it reopens its log file (or use `copytruncate`).\n"
        "4. Confirm the new log file is the one being written.\n\n## Notes\n"
        "- logrotate skips a directory with group-writable permissions; fix them first.\n",
        [R("the app log is 20GB, rotate it", [
            C("bash", {"command": "du -sh /var/log/app/*"}),
            C("bash", {"command": "logrotate -f /etc/logrotate.d/app"}, False, "nonzero_exit",
              "skipping because parent directory has insecure permissions"),
            C("bash", {"command": "chmod 755 /var/log/app"}),
            C("bash", {"command": "logrotate -f /etc/logrotate.d/app"})], "Rotated after fixing permissions."),
         R("rotate the logs", [
            C("bash", {"command": "du -sh /var/log/app/*"}),
            C("bash", {"command": "logrotate -f /etc/logrotate.d/app"}),
            C("bash", {"command": "ls -l /var/log/app"})], "Rotated."),
         R("logs rotated but disk still full", [
            C("bash", {"command": "lsof +L1 | grep app.log"}),
            C("bash", {"command": "kill -HUP $(pidof app)"}),
            C("bash", {"command": "df -h /var"})], "The app held the old file open; SIGHUP freed it.")]),
    "release-check": (
        fm("release-check", "Check a release before tagging it",
           "# Release check\n\n## Steps\n1. Run the tests.\n2. Tag the release.\n"),
        "## Steps\n1. Check the tree is clean with `git status`.\n2. Run `pytest -q`.\n"
        "3. Build with `python -m build` and run `twine check dist/*`.\n4. Tag the release.\n",
        [R("cut the 1.2 release", [
            C("bash", {"command": "pytest -q"}),
            C("bash", {"command": "python -m build"}),
            C("bash", {"command": "twine check dist/*"}),
            C("bash", {"command": "git tag v1.2"})], "Tagged v1.2."),
         R("release 1.3", [
            C("bash", {"command": "pytest -q"}, False, "nonzero_exit", "2 failed"),
            C("edit_file", {"path": "tests/test_api.py"}),
            C("bash", {"command": "pytest -q"}),
            C("bash", {"command": "git tag v1.3"})], "Fixed two tests, tagged v1.3."),
         R("tag 1.4", [
            C("bash", {"command": "python -m build"}, False, "nonzero_exit", "error: working tree is dirty"),
            C("bash", {"command": "git status"}),
            C("bash", {"command": "git stash"}),
            C("bash", {"command": "python -m build"}),
            C("bash", {"command": "git tag v1.4"})], "Stashed local edits, built, tagged v1.4.")]),
}

BUILTIN_RUNS = {
    "commit": (
        "\n9. If a push reports no upstream branch, use `git push -u origin <branch>`.\n"
        "10. If the hook reports lint errors, run the linter's fix, re-stage and commit again.\n",
        [R("commit the login fix", [
            C("bash", {"command": "git status"}), C("bash", {"command": "git diff"}),
            C("bash", {"command": "git add src/login.py"}),
            C("bash", {"command": "git commit -m 'fix(auth): handle expired tokens'"}, False, "nonzero_exit",
              "pre-commit: ruff found 2 errors"),
            C("bash", {"command": "ruff check --fix src/login.py"}), C("bash", {"command": "git add src/login.py"}),
            C("bash", {"command": "git commit -m 'fix(auth): handle expired tokens'"})], "Committed."),
         R("commit and push the docs update", [
            C("bash", {"command": "git status"}), C("bash", {"command": "git add docs/"}),
            C("bash", {"command": "git commit -m 'docs: update install guide'"}),
            C("bash", {"command": "git push"}, False, "nonzero_exit", "fatal: The current branch has no upstream branch"),
            C("bash", {"command": "git push -u origin docs-update"})], "Pushed."),
         R("commit the test changes", [
            C("bash", {"command": "git status"}), C("bash", {"command": "git add tests/test_api.py"}),
            C("bash", {"command": "git commit -m 'test(api): cover pagination'"})], "Committed.")]),
    "debug": (
        "\n\nReproduce under the failing environment's settings (TZ, env vars, permissions) before "
        "changing code; a failure that only happens elsewhere is usually environmental.\n",
        [R("the /health endpoint returns 500", [
            C("bash", {"command": "curl -s localhost:8080/health"}), C("read_file", {"path": "app/health.py"}),
            C("bash", {"command": "pytest tests/test_health.py"}, False, "nonzero_exit", "1 failed"),
            C("edit_file", {"path": "app/health.py"}), C("bash", {"command": "pytest tests/test_health.py"})],
           "The DB ping used a closed pool; fixed."),
         R("the nightly cron job did not run", [
            C("bash", {"command": "crontab -l"}), C("bash", {"command": "journalctl -u cron --since yesterday"}),
            C("bash", {"command": "ls -l scripts/nightly.sh"}), C("bash", {"command": "chmod +x scripts/nightly.sh"})],
           "The script lacked the exec bit."),
         R("tests fail on CI but pass locally", [
            C("read_file", {"path": "ci.log"}),
            C("bash", {"command": "TZ=UTC pytest tests/test_dates.py"}, False, "nonzero_exit", "1 failed"),
            C("edit_file", {"path": "tests/test_dates.py"}), C("bash", {"command": "TZ=UTC pytest tests/test_dates.py"})],
           "A timezone assumption; fixed.")]),
    "plan": (
        "\n\nBefore proposing steps, read the modules the change touches and list their callers.\n",
        [R("plan adding rate limiting to the API", [
            C("read_file", {"path": "api/routes.py"}), C("grep", {"pattern": "middleware"}),
            C("read_file", {"path": "api/config.py"})], "Plan: a token-bucket middleware, per key, config-driven."),
         R("plan the move from sqlite to postgres", [
            C("grep", {"pattern": "sqlite3"}), C("read_file", {"path": "db/store.py"})],
           "Plan: a storage interface, then a postgres backend behind it."),
         R("plan splitting the monolith module", [
            C("code_outline", {"path": "core/app.py"}), C("grep", {"pattern": "from core.app import"})],
           "Plan: extract three modules along the import seams.")]),
}


def surface(doc: str) -> list[str]:
    """Formatting-only rewrites: nothing a reader could act on differently."""
    head, _, body = doc.partition("\n---\n")
    head += "\n---\n"
    out = []
    out.append(head + body.replace("\n", "\n\n").replace("\n\n\n\n", "\n\n"))       # airier
    title = next((ln for ln in body.splitlines() if ln.startswith("# ")), None)
    setext = body.replace(title, f"{title[2:]}\n{'=' * len(title[2:])}", 1) if title else body
    out.append(head + setext.replace("\n- ", "\n* "))                               # setext title, * bullets
    out.append(head + "\n".join(
        (ln.replace(". ", ") ", 1) if ln[:1].isdigit() and ". " in ln[:4] else ln)
        for ln in body.splitlines()) + "\n")                                          # 1. → 1)
    out.append(head + body.replace("\n## ", "\n### "))                                # heading level
    out.append(head + body.replace(". ", ".  "))                                      # double spaces
    return out


def _is_step(ln: str) -> bool:
    import re
    return ln[:1].isdigit() or ln.lstrip().startswith("- ") or bool(re.match(r"#+ \d", ln))


def degraded(doc: str, other: str, rng: random.Random) -> list[str]:
    head, _, body = doc.partition("\n---\n")
    head += "\n---\n"
    lines = body.splitlines()
    steps = [i for i, ln in enumerate(lines) if _is_step(ln)]
    no_steps = head + "\n".join(ln for i, ln in enumerate(lines) if i not in steps) + "\n"
    _, _, other_body = other.partition("\n---\n")
    other_steps = [ln for ln in other_body.splitlines() if _is_step(ln)]
    swapped = head + "\n".join(
        (other_steps[k % len(other_steps)] if i in steps else ln)
        for k, (i, ln) in enumerate(enumerate(lines))) + "\n"
    kept = [lines[i] for i in steps]
    rng.shuffle(kept)
    kept = kept[: max(1, len(kept) // 2)]
    it = iter(kept)
    scrambled = head + "\n".join(
        (next(it, None) if i in steps else ln) or "" for i, ln in enumerate(lines)) + "\n"
    return [no_steps, swapped, scrambled]


def improve_builtin(doc: str, add: str) -> str:
    if "## Rules" in doc:
        return doc.replace("\n## Rules", add + "\n## Rules")
    return doc.rstrip("\n") + add


async def main() -> None:
    judge = PrometheusJudge(base_url=JUDGE_URL, model=JUDGE_MODEL)
    opt = GEPAOptimizer(None, provider_name="llama_cpp", judge=judge, config={"gepa_enabled": True})
    report = GEPAReport(timestamp=0.0)
    rng = random.Random(7)

    cases = {}
    for name, (doc, improved_steps, runs) in SYN.items():
        head, _, body = doc.partition("## Steps")
        cases[name] = (doc, head + improved_steps, runs)
    for name, (add, runs) in BUILTIN_RUNS.items():
        doc = (BUILTIN / f"{name}.md").read_text(encoding="utf-8")
        cases[name] = (doc, improve_builtin(doc, add), runs)
    names = list(cases)

    async def scores(doc, runs):
        return [await opt._score(judge, doc, run, report) for run in runs]

    results = {}
    for n_i, name in enumerate(names):
        doc, improved, runs = cases[name]
        other = cases[names[(n_i + 1) % len(names)]][0]
        r = {"orig": await scores(doc, runs), "orig2": await scores(doc, runs),
             "surface": [await scores(d, runs) for d in surface(doc)],
             "degraded": [await scores(d, runs) for d in degraded(doc, other, rng)],
             "improved": await scores(improved, runs)}
        results[name] = r
        print(f"  measured {name}", flush=True)

    def mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    def ok(xs):
        return all(x is not None for x in xs)

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else float("nan")

    det_pairs = sum(1 for r in results.values() for a, b in zip(r["orig"], r["orig2"]) if a != b)
    single, means, fp = [], [], {0.05: 0, 0.10: 0, 0.15: 0}
    surf_n = 0
    for r in results.values():
        if not ok(r["orig"]):
            continue
        for s in r["surface"]:
            if not ok(s):
                continue
            surf_n += 1
            single += [abs(a - b) for a, b in zip(s, r["orig"])]
            d = mean(s) - mean(r["orig"])
            means.append(abs(d))
            for m in fp:
                if d >= m - 1e-9 and all(a >= b - 1e-9 for a, b in zip(s, r["orig"])):
                    fp[m] += 1
    deg = [mean(r["orig"]) - mean(s) for r in results.values() if ok(r["orig"])
           for s in r["degraded"] if ok(s)]
    imp = {name: (round(mean(r["orig"]), 3), round(mean(r["improved"]), 3),
                  all(a >= b - 1e-9 for a, b in zip(r["improved"], r["orig"])))
           for name, r in results.items() if ok(r["orig"]) and ok(r["improved"])}

    print(json.dumps({
        "judge_model": JUDGE_MODEL,
        "documents": len(results), "runs_per_document": 3,
        "judge_calls": report.judged, "unparseable": report.unparseable,
        "determinism_pairs_that_differed": det_pairs,
        "surface_rewrites": surf_n,
        "surface_single_run_abs_delta": {"mean": round(mean(single), 3), "p50": q(single, .5),
                                         "p90": q(single, .9), "p95": q(single, .95), "max": max(single, default=None)},
        "surface_3run_mean_abs_delta": {"mean": round(mean(means), 3), "p90": round(q(means, .9), 3),
                                        "p95": round(q(means, .95), 3), "max": round(max(means, default=0), 3)},
        "surface_rewrites_passing_rule_at_margin": fp,
        "degraded_mean_drop": {"n": len(deg), "mean": round(mean(deg), 3), "min": round(min(deg, default=0), 3),
                               "share_dropping_0.10_or_more": round(sum(d >= 0.1 for d in deg) / len(deg), 3) if deg else None},
        "improved_live_to_variant_and_no_run_worse": imp,
        "orig_score_distribution": sorted({x for r in results.values() for x in r["orig"] if x is not None}),
    }, indent=1))


asyncio.run(main())
