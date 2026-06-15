"""Harvest-corpus generator — failure-mode BREADTH for the repair-pair flywheel.

#40 proved value diversity (3.2× pairs/run) but the resulting corpus was 97.7%
ONE transition (dict-unwrap) — a LoRA trained on that learns one lesson many
times and Goodharts the gym. What a fine-tune actually learns is the TRANSITION
(prometheus.learning.transition_taxonomy), so this generator is organized BY
transition and deliberately constructs prompts that provoke the under-represented
ones: fuzzy-rename, missing-discriminator, JSON-stuffed-string, type-coercion —
alongside the proven dict-unwrap shapes, with a per-transition CAP so no single
lesson dominates the projected corpus.

Inducement is probabilistic (the model chooses what to emit); the verification
harvest measures which prompts actually widen the histogram. Generation stays
deterministic (curated cross-products, no RNG), schema-valid, side-effecting
tools stubbed so the repair fires at the adapter/validation boundary.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

WORKSPACE = "/tmp/prometheus-gym-harvest"
SYSTEM_PROMPT = (
    "You are an assistant with access to tools. Use the appropriate tool to "
    "carry out the request. Be concise."
)

# ── value pools (cross-products → many distinct scalars, deterministically) ──

_CMD_VERBS = ["sleep 1 &&", "echo", "ls -la", "cat", "wc -l", "head -n 5",
              "tail -n 5", "sort", "grep -c TODO", "find . -name", "du -sh",
              "stat", "md5sum", "touch", "rm -f"]
_CMD_ARGS = ["report.txt", "/var/log/app.log", "data.csv", "*.py", "results/",
             "build.log", "notes.md", "config.yaml", "metrics.json", "out.tmp",
             "backup.tar", "index.html", "schema.sql", "errors.log", "tmp/"]

_AGENT_GOALS = ["research recent transformer papers", "summarize the Q3 sales pipeline",
                "monitor the staging error logs", "audit open security advisories",
                "draft a brief on competitor pricing", "triage the customer churn cohort",
                "reconcile the weekly ad spend", "benchmark the API latency dashboard",
                "collect metrics on the nightly backups", "analyze the license inventory",
                "investigate the failed deploys", "compile the on-call handoff notes"]

_GREP_PATTERNS = ["TODO", "FIXME", "import os", "def main", "API_KEY", "password",
                  "raise ValueError", "async def", "NotImplemented", "deprecated",
                  "XXX", "HACK", "BUG", "secret", "localhost"]
_GREP_PATHS = ["/tmp/proj/src", "/var/log", "/etc/app", "/home/user/code",
               "/opt/service", "/srv/www", "/data/in", "/workspace/lib",
               "/usr/local/share", "/mnt/repo", "/srv/api", "/home/dev/app"]

_STATUSES = ["running", "completed", "failed", "killed", "pending"]

# Plausible-but-wrong tool names → the real tool (fuzzy/levenshtein targets).
_FUZZY_NAMES = [
    ("list_tasks", "task_list", "show me the running background tasks"),
    ("tasks_list", "task_list", "list the completed background tasks"),
    ("create_task", "task_create", "start a background job that runs `echo hi`"),
    ("make_task", "task_create", "spin up a background task to run `ls -la`"),
    ("get_task", "task_get", "fetch the details of background task abc123"),
    ("stop_task", "task_stop", "cancel the background task abc123"),
    ("task_output", "task_output", "show the output of background task abc123"),
    ("list_sessions", "sessions_list", "list the active chat sessions"),
    ("session_list", "sessions_list", "show all the chat sessions"),
    ("search_web", "web_search", "search the web for 'gemma fine-tuning'"),
    ("websearch", "web_search", "look up 'llama.cpp grammar' on the web"),
    ("fetch_url", "web_fetch", "fetch the page at https://example.com/docs"),
    ("read", "read_file", "read the file /tmp/notes.txt"),
    ("search_files", "grep", "find 'TODO' across the files in /tmp/src"),
]

# JSON-stuffed-string bait: the args presented as a JSON object to "pass".
_JSON_BLOBS = [
    '{"command": "sleep 1 && echo done"}',
    '{"command": "ls -la /var/log"}',
    '{"command": "cat report.txt", "type": "local_bash"}',
    '{"status": "failed"}',
    '{"status": "running"}',
    '{"pattern": "TODO", "root": "/tmp/src"}',
    '{"prompt": "summarize the logs", "type": "local_agent"}',
    '{"command": "wc -l data.csv"}',
    '{"command": "find . -name *.py"}',
    '{"status": "completed"}',
]

# type-coercion bait: an integer/boolean param phrased so the model may quote it.
_COERCE_LIMITS = ["five", "ten", "twenty", "3", "7", "15", "two", "fifty", "8", "12"]


def _tag(task: dict[str, Any], transition: str) -> dict[str, Any]:
    task["category"] = f"harvest_{transition}"
    task["notes"] = f"intended_transition={transition}"
    return task


# ── transition-targeted shape generators ────────────────────────────────────


def _dict_wrap(n: int) -> list[dict[str, Any]]:
    """Proven: wrap-prone params the adapter unwraps. Capped (it's the easy one)."""
    out: list[dict[str, Any]] = []
    cmds = [f"{v} {a}" for v, a in itertools.product(_CMD_VERBS, _CMD_ARGS)]
    greps = list(itertools.product(_GREP_PATTERNS, _GREP_PATHS))
    third = max(1, n // 3)
    for i, cmd in enumerate(cmds[:third]):
        out.append(_tag({"id": f"hv_dictwrap_cmd_{i:04d}",
                         "prompt": f"Create a background task that runs this exact shell command: {cmd}",
                         "stub_tools": ["task_create"],
                         "score": {"expect_tool": "task_create"}}, "dict_wrap_unwrap"))
    for i, (pat, path) in enumerate(greps[:third]):
        out.append(_tag({"id": f"hv_dictwrap_grep_{i:04d}",
                         "prompt": f"Search for the pattern '{pat}' under the directory {path} and report matches.",
                         "score": {"expect_tool": "grep"}}, "dict_wrap_unwrap"))
    for i, status in enumerate(_STATUSES[: max(1, n - 2 * third)]):
        out.append(_tag({"id": f"hv_dictwrap_status_{i:04d}",
                         "prompt": f"List the background tasks that currently have status '{status}'.",
                         "score": {"expect_tool": "task_list"}}, "dict_wrap_unwrap"))
    return out[:n]


def _missing_discriminator(n: int) -> list[dict[str, Any]]:
    """Agent-intent task_create that omits `type` → 'command required for
    local_bash' (the default) → repair to type=local_agent (D2 shape)."""
    goals = _AGENT_GOALS[:n]
    return [
        _tag({"id": f"hv_missingdisc_{i:04d}",
              "prompt": f"Create a background task to {goal}. (Use the task_create tool.)",
              "stub_tools": ["task_create"],
              "score": {"expect_tool": "task_create"}}, "missing_discriminator")
        for i, goal in enumerate(goals)
    ]


# Network / side-effecting tools the fuzzy + json shapes may reach — stubbed so
# the REPAIR fires at the adapter boundary without a real external call.
_SAFE_STUBS = ["task_create", "task_stop", "task_update", "web_search", "web_fetch",
               "download_file", "browser", "sessions_send", "sessions_spawn"]


def _fuzzy_rename(n: int) -> list[dict[str, Any]]:
    """Prime the model with a plausible-but-wrong tool name → levenshtein repair.
    The score expects the REAL tool (the repair target), which executes after the
    rename — read-only reals run for real, side-effecting ones are stubbed."""
    items = _FUZZY_NAMES[:n]
    out = []
    for i, (wrong, real, action) in enumerate(items):
        out.append(_tag({"id": f"hv_fuzzy_{i:04d}",
                         "prompt": f"Use the `{wrong}` tool to {action}.",
                         "stub_tools": _SAFE_STUBS,
                         "score": {"expect_tool": real}}, "fuzzy_rename"))
    return out


def _json_stuffed(n: int) -> list[dict[str, Any]]:
    """Present the args as a JSON object 'to pass' → model may stuff it as a string."""
    blobs = _JSON_BLOBS[:n]
    out = []
    for i, blob in enumerate(blobs):
        tool = "task_create" if "command" in blob or "prompt" in blob else (
            "task_list" if "status" in blob else "grep")
        out.append(_tag({"id": f"hv_jsonstuff_{i:04d}",
                         "prompt": f"Call the {tool} tool, passing exactly these arguments as JSON: {blob}",
                         "stub_tools": _SAFE_STUBS,
                         "score": {"expect_tool": tool}}, "json_stuffed_string"))
    return out


def _type_coercion(n: int) -> list[dict[str, Any]]:
    """A numeric param phrased as a word/string → model may quote it (best-effort)."""
    limits = _COERCE_LIMITS[:n]
    return [
        _tag({"id": f"hv_coerce_{i:04d}",
              "prompt": f"List the most recent background tasks — limit the result to {lim} entries.",
              "score": {"expect_tool": "task_list"}}, "type_coercion")
        for i, lim in enumerate(limits)
    ]


# transition → its shape generator. Each is capped to `per_transition`.
_BY_TRANSITION: dict[str, Callable[[int], list[dict[str, Any]]]] = {
    "dict_wrap_unwrap": _dict_wrap,
    "missing_discriminator": _missing_discriminator,
    "fuzzy_rename": _fuzzy_rename,
    "json_stuffed_string": _json_stuffed,
    "type_coercion": _type_coercion,
}


def generate_harvest_corpus(per_transition: int = 20) -> dict[str, Any]:
    """Return a schema-valid gym taskset balanced ACROSS transition types: up to
    ``per_transition`` tasks per targeted transition (the per-transition cap), so
    no single repair lesson dominates. Deterministic."""
    tasks: list[dict[str, Any]] = []
    for transition, shape in _BY_TRANSITION.items():
        tasks.extend(shape(per_transition))
    return {
        "version": 1,
        "workspace": WORKSPACE,
        "system_prompt": SYSTEM_PROMPT,
        "tasks": tasks,
    }


def intended_distribution(corpus: dict[str, Any]) -> dict[str, int]:
    """The corpus's INTENDED transition mix (from task categories) — distinct from
    the ACHIEVED mix the harvest measures."""
    out: dict[str, int] = {}
    for t in corpus["tasks"]:
        key = t["category"].removeprefix("harvest_")
        out[key] = out.get(key, 0) + 1
    return out
