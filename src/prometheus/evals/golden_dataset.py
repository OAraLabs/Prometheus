"""Golden evaluation dataset — 26 canonical tasks for Prometheus quality measurement.

Source: Derived from Sprint 8 benchmark categories.

Tier 1: 21 atomic single-tool tasks.
Tier 2: 5 multi-step workflow tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class TaskTier(IntEnum):
    """Evaluation task tier levels."""

    TIER_1 = 1  # Atomic single-tool tasks
    TIER_2 = 2  # Multi-step workflows


@dataclass
class GoldenTask:
    """A single golden evaluation task."""

    id: str
    name: str
    tier: int
    input: str
    expected_behavior: str
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    requires_network: bool = False
    max_turns: int = 10


def _tier1_tasks() -> list[GoldenTask]:
    """21 Tier 1 atomic tasks."""
    return [
        # --- Bash ---
        GoldenTask(
            id="t1-bash-echo",
            name="Bash echo",
            tier=TaskTier.TIER_1,
            input="Echo 'hello world' using bash",
            expected_behavior="Uses bash tool to run echo command, output contains 'hello world'",
            expected_tools=["bash"],
            tags=["bash", "simple"],
        ),
        GoldenTask(
            id="t1-bash-pwd",
            name="Bash pwd",
            tier=TaskTier.TIER_1,
            input="Use bash to print the current working directory",
            expected_behavior="Uses bash pwd command, returns a valid directory path",
            expected_tools=["bash"],
            tags=["bash", "filesystem"],
        ),
        GoldenTask(
            id="t1-bash-arithmetic",
            name="Bash arithmetic",
            tier=TaskTier.TIER_1,
            input="Use bash to calculate 42 * 17",
            expected_behavior="Uses bash to compute 714, output contains the correct result",
            expected_tools=["bash"],
            tags=["bash", "math"],
        ),
        GoldenTask(
            id="t1-bash-sysinfo",
            name="Bash system info",
            tier=TaskTier.TIER_1,
            input="How much free memory is on this system?",
            expected_behavior="Uses bash free -h or similar, returns memory statistics",
            expected_tools=["bash"],
            tags=["bash", "system"],
        ),
        # --- File Read ---
        GoldenTask(
            id="t1-file-read",
            name="File read",
            tier=TaskTier.TIER_1,
            input="Read the contents of README.md",
            expected_behavior="Uses read_file tool to read README.md, returns file contents",
            expected_tools=["read_file"],
            tags=["filesystem", "read"],
        ),
        GoldenTask(
            id="t1-file-read-lines",
            name="File read specific lines",
            tier=TaskTier.TIER_1,
            input="Show me the first 10 lines of pyproject.toml",
            expected_behavior="Uses read_file or bash to read first 10 lines of pyproject.toml",
            expected_tools=["read_file", "bash"],
            tags=["filesystem", "read"],
        ),
        # --- File Write ---
        GoldenTask(
            id="t1-file-write",
            name="File write",
            tier=TaskTier.TIER_1,
            input="Create a file called /tmp/prom_eval_test.txt with the text 'evaluation test'",
            expected_behavior="Uses write_file tool to create the file with specified content",
            expected_tools=["write_file"],
            tags=["filesystem", "write"],
        ),
        GoldenTask(
            id="t1-file-write-script",
            name="File write Python script",
            tier=TaskTier.TIER_1,
            input="Write a Python script at /tmp/prom_eval_hello.py that prints 'Hello from Prometheus'",
            expected_behavior="Uses write_file to create a valid Python script with print statement",
            expected_tools=["write_file"],
            tags=["filesystem", "write", "python"],
        ),
        # --- File Edit ---
        GoldenTask(
            id="t1-file-edit",
            name="File edit",
            tier=TaskTier.TIER_1,
            input="First create a file /tmp/prom_eval_edit.txt with 'line one\\nline two\\nline three', then replace 'line two' with 'line TWO EDITED'",
            expected_behavior="Uses write_file then edit_file to modify the file in place",
            expected_tools=["write_file", "edit_file"],
            tags=["filesystem", "edit"],
        ),
        # --- Glob ---
        GoldenTask(
            id="t1-glob-py",
            name="Glob Python files",
            tier=TaskTier.TIER_1,
            input="Find all Python files in the src/prometheus/evals/ directory",
            expected_behavior="Uses glob tool with pattern like '**/*.py', returns list of .py files",
            expected_tools=["glob"],
            tags=["filesystem", "search"],
        ),
        GoldenTask(
            id="t1-glob-yaml",
            name="Glob YAML files",
            tier=TaskTier.TIER_1,
            input="Find all .yaml files in the config/ directory",
            expected_behavior="Uses glob tool, returns prometheus.yaml and any other YAML files",
            expected_tools=["glob"],
            tags=["filesystem", "search"],
        ),
        # --- Grep ---
        GoldenTask(
            id="t1-grep-literal",
            name="Grep literal",
            tier=TaskTier.TIER_1,
            input="Search for the string 'ToolRegistry' in src/prometheus/tools/base.py",
            expected_behavior="Uses grep tool to find occurrences of ToolRegistry in the file",
            expected_tools=["grep"],
            tags=["search", "code"],
        ),
        GoldenTask(
            id="t1-grep-regex",
            name="Grep regex",
            tier=TaskTier.TIER_1,
            input="Search for all class definitions (lines starting with 'class ') in src/prometheus/engine/agent_loop.py",
            expected_behavior="Uses grep with regex pattern, finds LoopContext, RunResult, AgentLoop classes",
            expected_tools=["grep"],
            tags=["search", "code", "regex"],
        ),
        # --- Cron ---
        GoldenTask(
            id="t1-cron-list",
            name="Cron list",
            tier=TaskTier.TIER_1,
            input="List all registered cron jobs",
            expected_behavior="Uses cron_list tool to show registered cron entries",
            expected_tools=["cron_list"],
            tags=["cron", "read"],
        ),
        GoldenTask(
            id="t1-cron-create",
            name="Cron create",
            tier=TaskTier.TIER_1,
            input="Create a cron job named 'eval_test' that runs 'echo test' every hour",
            expected_behavior="Uses cron_create tool to register a new cron entry",
            expected_tools=["cron_create"],
            tags=["cron", "write"],
        ),
        # --- Todo ---
        GoldenTask(
            id="t1-todo",
            name="Todo creation",
            tier=TaskTier.TIER_1,
            input="Add a todo item: 'Review evaluation results'",
            expected_behavior="Uses todo_write tool to create a todo entry",
            expected_tools=["todo_write"],
            tags=["todo"],
        ),
        # --- Web ---
        GoldenTask(
            id="t1-web-search",
            name="Web search",
            tier=TaskTier.TIER_1,
            input="Search the web for 'Python asyncio tutorial'",
            expected_behavior="Uses web_search tool and returns relevant results",
            expected_tools=["web_search"],
            tags=["web", "search"],
            requires_network=True,
        ),
        GoldenTask(
            id="t1-web-fetch",
            name="Web fetch",
            tier=TaskTier.TIER_1,
            input="Fetch the robots.txt from example.com",
            expected_behavior="Uses web_fetch or bash curl to retrieve the page content",
            expected_tools=["web_fetch", "bash"],
            tags=["web", "fetch"],
            requires_network=True,
        ),
        # --- Git ---
        GoldenTask(
            id="t1-git-branch",
            name="Git branch",
            tier=TaskTier.TIER_1,
            input="What git branch am I on?",
            expected_behavior="Uses bash git branch or git status, returns current branch name",
            expected_tools=["bash"],
            tags=["git", "read"],
        ),
        GoldenTask(
            id="t1-git-log",
            name="Git log",
            tier=TaskTier.TIER_1,
            input="Show the last 5 git commits",
            expected_behavior="Uses bash git log -5 or similar, returns commit history",
            expected_tools=["bash"],
            tags=["git", "read"],
        ),
        # --- Simple answer ---
        GoldenTask(
            id="t1-simple-answer",
            name="Simple answer (no tool)",
            tier=TaskTier.TIER_1,
            input="What is 2 + 2?",
            expected_behavior="Returns 4, may answer directly without tools or use bash",
            expected_tools=[],
            tags=["math", "simple"],
        ),
    ]


def _tier2_tasks() -> list[GoldenTask]:
    """5 Tier 2 multi-step tasks."""
    return [
        GoldenTask(
            id="t2-write-then-read",
            name="Write then read roundtrip",
            tier=TaskTier.TIER_2,
            input="Create a file /tmp/prom_eval_roundtrip.txt with 'roundtrip test data', then read it back and confirm the contents",
            expected_behavior="Uses write_file to create the file, then read_file to verify contents match",
            expected_tools=["write_file", "read_file"],
            tags=["filesystem", "multi-step"],
            max_turns=15,
        ),
        GoldenTask(
            id="t2-write-and-run",
            name="Write Python and execute",
            tier=TaskTier.TIER_2,
            input="Write a Python script at /tmp/prom_eval_fib.py that prints the first 10 Fibonacci numbers, then run it",
            expected_behavior="Creates a Python file with Fibonacci logic, executes it with bash, output shows the sequence",
            expected_tools=["write_file", "bash"],
            tags=["python", "multi-step"],
            max_turns=15,
        ),
        GoldenTask(
            id="t2-search-and-read",
            name="Glob then read",
            tier=TaskTier.TIER_2,
            input="Find all Python files in src/prometheus/tracing/, then read the contents of the __init__.py file",
            expected_behavior="Uses glob to find .py files in tracing/, then read_file to show __init__.py contents",
            expected_tools=["glob", "read_file"],
            tags=["filesystem", "search", "multi-step"],
            max_turns=15,
        ),
        GoldenTask(
            id="t2-edit-and-verify",
            name="Edit then grep verify",
            tier=TaskTier.TIER_2,
            input="Create /tmp/prom_eval_config.txt with 'debug=false', then edit it to say 'debug=true', then grep for 'debug=true' to verify",
            expected_behavior="Creates file, edits it, then uses grep or read_file to confirm the change",
            expected_tools=["write_file", "edit_file", "grep"],
            tags=["filesystem", "edit", "multi-step"],
            max_turns=15,
        ),
        GoldenTask(
            id="t2-multi-file",
            name="Multi-file workflow",
            tier=TaskTier.TIER_2,
            input="Create three files: prom_eval_a.txt with 'alpha', prom_eval_b.txt with 'beta', prom_eval_c.txt with 'gamma'. Then use glob to list all prom_eval_*.txt files.",
            expected_behavior="Creates all three files using write_file, then uses glob with pattern 'prom_eval_*.txt' to list them",
            expected_tools=["write_file", "glob"],
            tags=["filesystem", "multi-step"],
            max_turns=20,
        ),
    ]


def load_golden_dataset(
    tier: int | None = None,
    skip_network: bool = True,
) -> list[GoldenTask]:
    """Load the built-in golden dataset, optionally filtered.

    Args:
        tier: Filter to specific tier (1 or 2). None = all tiers.
        skip_network: If True, exclude tasks requiring web access.
    """
    tasks = _tier1_tasks() + _tier2_tasks()
    if tier is not None:
        tasks = [t for t in tasks if t.tier == tier]
    if skip_network:
        tasks = [t for t in tasks if not t.requires_network]
    return tasks
