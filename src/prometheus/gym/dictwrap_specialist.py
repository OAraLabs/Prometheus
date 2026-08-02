"""Dict-wrap specialist corpus: a TRAIN set and a disjoint HELD-OUT eval set.

The breadth finding (#41) settled it: the model's real failure is ~88-98%
dict-wrap, so the first LoRA is a dict-wrap specialist. The whole result hinges
on the eval being on shapes the train NEVER drilled (the Goodhart guard) — so
this module produces two corpora with DISJOINT (tool, param) coverage:

  TRAIN   — dict-wrap on task_create.command/.prompt, grep.root, task_list.status
            (the shapes #40/#41 proved the model wraps), expanded value pools.
  HELDOUT — dict-wrap on grep.PATTERN, task_get.task_id, task_update.status_note,
            cron_create.command, download_file.url — different tools AND params,
            disjoint nouns. If the LoRA reduces wrapping HERE, it generalized.

Both deterministic, schema-valid, side-effecting tools stubbed (the unwrap fires
at validation, pre-execution). Pools are intentionally large so the train set
reaches the target distinct-task count.
"""

from __future__ import annotations

import itertools
from typing import Any

from prometheus.gym.corpus import SYSTEM_PROMPT, WORKSPACE, _tag

# ── expanded train value pools (cross-products → distinct tasks) ─────────────

_TR_VERBS = ["sleep 1 &&", "echo", "ls -la", "cat", "wc -l", "head -n 5", "tail -n 5",
             "sort", "grep -c TODO", "find . -name", "du -sh", "stat", "md5sum",
             "touch", "rm -f", "cp", "mv", "chmod 644", "gzip", "tar -tf"]
_TR_ARGS = ["report.txt", "/var/log/app.log", "data.csv", "*.py", "results/", "build.log",
            "notes.md", "config.yaml", "metrics.json", "out.tmp", "backup.tar", "index.html",
            "schema.sql", "errors.log", "tmp/", "cache/", "dump.json", "audit.log",
            "staging.env", "release.txt"]
_TR_GREP_PAT = ["TODO", "FIXME", "import os", "def main", "API_KEY", "password",
                "raise ValueError", "async def", "NotImplemented", "deprecated", "XXX",
                "HACK", "BUG", "secret", "localhost", "127.0.0.1", "panic", "traceback",
                "select *", "DROP TABLE"]
_TR_GREP_PATH = ["/tmp/proj/src", "/var/log", "/etc/app", "/home/user/code", "/opt/service",
                 "/srv/www", "/data/in", "/workspace/lib", "/usr/local/share", "/mnt/repo",
                 "/srv/api", "/home/dev/app", "/var/lib/svc", "/opt/bin", "/data/out"]
_TR_GOALS = ["research recent transformer papers", "summarize the Q3 sales pipeline",
             "monitor the staging error logs", "audit open security advisories",
             "draft a brief on competitor pricing", "triage the customer churn cohort",
             "reconcile the weekly ad spend", "benchmark the API latency dashboard",
             "collect metrics on the nightly backups", "analyze the license inventory",
             "investigate the failed deploys", "compile the on-call handoff notes",
             "review the dependency upgrades", "chart the support ticket volume"]
_STATUSES = ["running", "completed", "failed", "killed", "pending"]


def generate_dictwrap_train(per_shape: int = 250) -> dict[str, Any]:
    """Dict-wrap TRAIN corpus — the shapes the model proven-wraps, expanded."""
    tasks: list[dict[str, Any]] = []
    cmds = [f"{v} {a}" for v, a in itertools.product(_TR_VERBS, _TR_ARGS)][:per_shape]
    for i, c in enumerate(cmds):
        tasks.append(_tag({"id": f"tr_cmd_{i:04d}",
                           "prompt": f"Create a background task that runs this exact shell command: {c}",
                           "stub_tools": ["task_create"], "score": {"expect_tool": "task_create"}},
                          "dict_wrap_unwrap"))
    greps = list(itertools.product(_TR_GREP_PAT, _TR_GREP_PATH))[:per_shape]
    for i, (p, path) in enumerate(greps):
        tasks.append(_tag({"id": f"tr_grep_{i:04d}",
                           "prompt": f"Search for the pattern '{p}' under the directory {path} and report matches.",
                           "score": {"expect_tool": "grep"}}, "dict_wrap_unwrap"))
    for i, g in enumerate(_TR_GOALS[:per_shape]):
        tasks.append(_tag({"id": f"tr_goal_{i:04d}",
                           "prompt": f"Create a background agent task whose goal is: {g}",
                           "stub_tools": ["task_create"], "score": {"expect_tool": "task_create"}},
                          "dict_wrap_unwrap"))
    for i, s in enumerate(_STATUSES[:per_shape]):
        tasks.append(_tag({"id": f"tr_status_{i:04d}",
                           "prompt": f"List the background tasks that currently have status '{s}'.",
                           "score": {"expect_tool": "task_list"}}, "dict_wrap_unwrap"))
    return {"version": 1, "workspace": WORKSPACE, "system_prompt": SYSTEM_PROMPT, "tasks": tasks}


# ── held-out: dict-wrap on tools/params the TRAIN set never touched ──────────

_HO_TASK_IDS = ["t-9f2a", "task-0042", "bg-7c31", "j-aa19", "t-5e80", "run-2b6d",
                "task-118c", "t-d4f0", "bg-6611", "j-3a7e"]
_HO_CRON = [("nightly-backup", "tar -czf /backup/db.tgz /data"), ("log-rotate", "rotate /var/log/app.log"),
            ("metrics-flush", "curl -X POST localhost:9000/flush"), ("cache-purge", "redis-cli FLUSHDB"),
            ("health-ping", "curl -sf localhost:8080/health"), ("cert-renew", "certbot renew --quiet"),
            ("digest-mail", "python send_digest.py"), ("index-rebuild", "python reindex.py")]
_HO_URLS = ["https://example.com/data/report.pdf", "https://cdn.site.net/assets/model.bin",
            "https://api.dev/v1/export.csv", "https://files.org/archive.tar.gz",
            "https://repo.io/releases/v2.tgz", "https://docs.site/spec.json",
            "https://mirror.net/iso/boot.img", "https://hub.dev/pkg/wheel.whl"]
_HO_NOTES = ["blocked on review", "waiting for QA", "merged to staging", "rolled back",
             "pending data backfill", "escalated to oncall", "deferred to next sprint", "done, verifying"]


def generate_dictwrap_heldout(per_shape: int = 8) -> dict[str, Any]:
    """HELD-OUT eval corpus — dict-wrap on DIFFERENT (tool, param) than train.
    adapter_unwrap must include these tools in the eval manifest so wraps capture."""
    tasks: list[dict[str, Any]] = []
    # grep.PATTERN (train drilled grep.root)
    for i, p in enumerate(_TR_GREP_PAT[:per_shape]):
        tasks.append(_tag({"id": f"ho_greppat_{i:04d}",
                           "prompt": f"Search the whole workspace for the pattern '{p}'.",
                           "score": {"expect_tool": "grep"}}, "dict_wrap_unwrap"))
    # task_get.task_id (new tool)
    for i, tid in enumerate(_HO_TASK_IDS[:per_shape]):
        tasks.append(_tag({"id": f"ho_taskget_{i:04d}",
                           "prompt": f"Get the full details of background task {tid}.",
                           "score": {"expect_tool": "task_get"}}, "dict_wrap_unwrap"))
    # cron_create.command (new tool)
    for i, (name, cmd) in enumerate(_HO_CRON[:per_shape]):
        tasks.append(_tag({"id": f"ho_cron_{i:04d}",
                           "prompt": f"Create a cron job named '{name}' on schedule '0 3 * * *' that runs: {cmd}",
                           "stub_tools": ["cron_create"], "score": {"expect_tool": "cron_create"}},
                          "dict_wrap_unwrap"))
    # download_file.url (new tool)
    for i, u in enumerate(_HO_URLS[:per_shape]):
        tasks.append(_tag({"id": f"ho_dl_{i:04d}",
                           "prompt": f"Download the file at {u}.",
                           "stub_tools": ["download_file"], "score": {"expect_tool": "download_file"}},
                          "dict_wrap_unwrap"))
    # task_update.status_note (new tool)
    for i, n in enumerate(_HO_NOTES[:per_shape]):
        tasks.append(_tag({"id": f"ho_taskupd_{i:04d}",
                           "prompt": f"Update background task t-77{i:02d}: set its status note to '{n}'.",
                           "stub_tools": ["task_update"], "score": {"expect_tool": "task_update"}},
                          "dict_wrap_unwrap"))
    return {"version": 1, "workspace": WORKSPACE, "system_prompt": SYSTEM_PROMPT, "tasks": tasks}


# Tools the held-out wraps target (the eval manifest's adapter_unwrap list).
HELDOUT_UNWRAP_TOOLS = ["grep", "task_get", "cron_create", "download_file", "task_update"]


def train_heldout_tool_param_overlap() -> set[tuple[str, str]]:
    """The (tool, param) combos the train set drills — held-out must avoid all of
    these. Used by the test to assert disjointness."""
    return {("task_create", "command"), ("task_create", "prompt"),
            ("grep", "root"), ("task_list", "status")}
