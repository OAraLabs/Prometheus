"""Task-set loading for the gym.

A task set is a versioned YAML file — FROZEN per experiment series. Editing
a task set means starting a new series; the loader records the file's sha256
so every run row is pinned to the exact bytes it ran against.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Scoring predicate keys a task may declare (all present predicates must
# pass — AND semantics). Deterministic only; no LLM judging.
ALLOWED_SCORE_KEYS = frozenset({
    "expect_tool",            # str: ≥1 non-error execution of this tool
    "expect_tool_args_string",  # list[str]: these params, if present on the
                                # successful expect_tool call, must be strings
    "expect_tool_args_require",  # dict: exact param values on that call
    "prompt_not_json_blob",   # bool: expect_tool call's `prompt` arg must be a
                              # plain string, not a serialized JSON object
    "expect_file",            # str path: file must exist after the run
    "expect_file_contains",   # str: substring of expect_file's content
    "expect_text_contains",   # str: case-insensitive substring of final text
    "expect_text_min_len",    # int: final text length ≥ N
    "forbid_bash_containing",  # str: no bash call's command may contain this
    "forbid_breaker_trip",    # bool: circuit-breaker trip = fail
    "forbid_malformed",       # bool: any provider-dropped malformed call = fail
    "require_graceful",       # bool: non-empty final text AND no breaker trip
})

ALLOWED_SEED_KEYS = frozenset({"user", "assistant_text", "assistant_tool_call", "tool_result"})


@dataclass
class TaskSpec:
    id: str
    category: str
    prompt: str
    score: dict[str, Any]
    # Conversation seeded BEFORE the live continuation (collapse-arc replay).
    # List of single-key dicts: {user: str} | {assistant_text: str} |
    # {assistant_tool_call: {name, input}} | {tool_result: {content, is_error}}
    # tool_result pairs with the closest preceding assistant_tool_call.
    seed: list[dict[str, Any]] = field(default_factory=list)
    # Files to create under the workspace before the run: {relpath: content}
    setup_files: dict[str, str] = field(default_factory=dict)
    # Tools whose execute() is stubbed side-effect-free for this task
    # (schema/parse/validation stay real). Currently only task_create.
    stub_tools: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TaskSet:
    version: int
    system_prompt: str
    workspace: str
    tasks: list[TaskSpec]
    sha256: str
    path: str


def _validate_task(raw: dict[str, Any], idx: int) -> None:
    for key in ("id", "category", "prompt", "score"):
        if key not in raw:
            raise ValueError(f"task #{idx} missing required key {key!r}")
    unknown = set(raw["score"]) - ALLOWED_SCORE_KEYS
    if unknown:
        raise ValueError(
            f"task {raw['id']!r}: unknown score predicate(s) {sorted(unknown)}; "
            f"allowed: {sorted(ALLOWED_SCORE_KEYS)}"
        )
    for step in raw.get("seed", []) or []:
        if len(step) != 1 or next(iter(step)) not in ALLOWED_SEED_KEYS:
            raise ValueError(
                f"task {raw['id']!r}: seed steps must be single-key dicts "
                f"from {sorted(ALLOWED_SEED_KEYS)}, got {step!r}"
            )


def load_taskset(path: str | Path) -> TaskSet:
    p = Path(path)
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(f"{p}: not a task set (missing 'tasks')")

    tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(data["tasks"]):
        _validate_task(raw, i)
        if raw["id"] in seen_ids:
            raise ValueError(f"duplicate task id {raw['id']!r}")
        seen_ids.add(raw["id"])
        tasks.append(
            TaskSpec(
                id=raw["id"],
                category=raw["category"],
                prompt=raw["prompt"],
                score=raw["score"],
                seed=raw.get("seed") or [],
                setup_files=raw.get("setup_files") or {},
                stub_tools=raw.get("stub_tools") or [],
                notes=raw.get("notes", ""),
            )
        )

    return TaskSet(
        version=int(data.get("version", 1)),
        system_prompt=data.get("system_prompt", "You are a helpful assistant with tools."),
        workspace=data.get("workspace", "/tmp/prometheus-gym"),
        tasks=tasks,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=str(p),
    )
