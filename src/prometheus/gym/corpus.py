"""Harvest-corpus generator — task DIVERSITY for the repair-pair flywheel.

The harvest mechanism is proven (gym/harvest.py): drive wrap-prone tool calls,
capture the adapter repairs as training pairs. But pairs dedup on
``sha256(context + rejected_call)`` (pair_capture.py) — so re-running the SAME
prompts re-captures the SAME contexts. Volume comes from VARYING THE TARGET
VALUE across wrap-prone shapes: each distinct (tool, param, value) the model
nests-then-gets-unwrapped is a distinct pair.

This generator emits a deterministic, schema-valid gym taskset (no RNG — the
gym is deterministic and Math.random is banned) whose every task targets a
high-ceiling wrap-prone shape with a distinct scalar value, drawn from
cross-products of curated word lists. Side-effecting tools are stubbed: the
unwrap fires at the adapter/validation layer BEFORE execution, so stubbing is
safe (and is how v1 already captured the task_create pairs).
"""

from __future__ import annotations

import itertools
from typing import Any

WORKSPACE = "/tmp/prometheus-gym-harvest"
SYSTEM_PROMPT = (
    "You are an assistant with access to tools. Use the appropriate tool to "
    "carry out the request. Be concise."
)

# ── Value pools — cross-products give many distinct scalars deterministically ──

_CMD_VERBS = ["sleep 1 &&", "echo", "ls -la", "cat", "wc -l", "head -n 5",
              "tail -n 5", "sort", "grep -c TODO", "find . -name"]
_CMD_ARGS = ["report.txt", "/var/log/app.log", "data.csv", "*.py", "results/",
             "build.log", "notes.md", "config.yaml", "metrics.json", "out.tmp"]

_GOAL_ACTIONS = ["Research", "Summarize", "Monitor", "Analyze", "Draft a brief on",
                 "Collect metrics on", "Audit", "Benchmark", "Triage", "Reconcile"]
_GOAL_TOPICS = ["the Q3 sales pipeline", "recent transformer papers",
                "the staging error logs", "competitor pricing pages",
                "the nightly backup status", "open security advisories",
                "the customer churn cohort", "the API latency dashboard",
                "the open-source license inventory", "the weekly ad spend"]

_GREP_PATTERNS = ["TODO", "FIXME", "import os", "def main", "API_KEY", "password",
                  "raise ValueError", "async def", "NotImplemented", "deprecated"]
_GREP_PATHS = ["/tmp/proj/src", "/var/log", "/etc/app", "/home/user/code",
               "/opt/service", "/srv/www", "/data/in", "/workspace/lib",
               "/usr/local/share", "/mnt/repo"]

_STATUSES = ["running", "completed", "failed", "killed", "pending"]


def _cross(a: list[str], b: list[str]) -> list[str]:
    return [f"{x} {y}" for x, y in itertools.product(a, b)]


def _shape_task_create_command(n: int) -> list[dict[str, Any]]:
    cmds = _cross(_CMD_VERBS, _CMD_ARGS)[:n]
    return [
        {
            "id": f"hv_tc_cmd_{i:04d}",
            "category": "harvest",
            "prompt": f"Create a background task that runs this exact shell command: {cmd}",
            "stub_tools": ["task_create"],
            "score": {"expect_tool": "task_create"},
        }
        for i, cmd in enumerate(cmds)
    ]


def _shape_task_create_goal(n: int) -> list[dict[str, Any]]:
    goals = [f"{a} {t}" for a, t in itertools.product(_GOAL_ACTIONS, _GOAL_TOPICS)][:n]
    return [
        {
            "id": f"hv_tc_goal_{i:04d}",
            "category": "harvest",
            "prompt": f"Create a background agent task whose goal is: {goal}",
            "stub_tools": ["task_create"],
            "score": {"expect_tool": "task_create"},
        }
        for i, goal in enumerate(goals)
    ]


def _shape_grep(n: int) -> list[dict[str, Any]]:
    combos = list(itertools.product(_GREP_PATTERNS, _GREP_PATHS))[:n]
    return [
        {
            "id": f"hv_grep_{i:04d}",
            "category": "harvest",
            "prompt": f"Search for the pattern '{pat}' under the directory {path} "
                      f"and report which files match.",
            "score": {"expect_tool": "grep"},
        }
        for i, (pat, path) in enumerate(combos)
    ]


def _shape_task_list_status(n: int) -> list[dict[str, Any]]:
    statuses = _STATUSES[: min(n, len(_STATUSES))]
    return [
        {
            "id": f"hv_tl_status_{i:04d}",
            "category": "harvest",
            "prompt": f"List the background tasks that currently have status '{status}'.",
            "score": {"expect_tool": "task_list"},
        }
        for i, status in enumerate(statuses)
    ]


# Order: the high-ceiling shapes first (task_create ×2, grep), then the capped one.
_SHAPES = (
    _shape_task_create_command,
    _shape_task_create_goal,
    _shape_grep,
    _shape_task_list_status,
)


def generate_harvest_corpus(per_shape: int = 25) -> dict[str, Any]:
    """Return a schema-valid gym taskset dict: up to ``per_shape`` distinct
    tasks per wrap-prone shape. Deterministic — same input, same output."""
    tasks: list[dict[str, Any]] = []
    for shape in _SHAPES:
        tasks.extend(shape(per_shape))
    return {
        "version": 1,
        "workspace": WORKSPACE,
        "system_prompt": SYSTEM_PROMPT,
        "tasks": tasks,
    }
