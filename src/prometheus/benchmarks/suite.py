"""Benchmark test case definitions — YAML-backed test suite.

Source: Novel code for Prometheus Sprint 8.

Tier 1: 20+ atomic tool call tests (single tool invocation).
Tier 2: 5 multi-step tests (require multiple tool calls in sequence).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml


class TestTier(IntEnum):
    """Benchmark tier levels."""

    TIER_1 = 1  # Atomic single-tool tests
    TIER_2 = 2  # Multi-step workflows


@dataclass
class TestCase:
    """A single benchmark test case."""

    id: str
    name: str
    tier: int
    prompt: str
    expected_tools: list[str] = field(default_factory=list)
    expected_output_contains: list[str] = field(default_factory=list)
    expected_output_not_contains: list[str] = field(default_factory=list)
    expected_file_exists: list[str] = field(default_factory=list)
    expected_file_contains: dict[str, str] = field(default_factory=dict)
    max_turns: int = 10
    setup_commands: list[str] = field(default_factory=list)
    teardown_commands: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite:
    """Collection of test cases, loadable from YAML."""

    def __init__(self, cases: list[TestCase] | None = None) -> None:
        self._cases = cases or []

    @property
    def cases(self) -> list[TestCase]:
        return list(self._cases)

    def filter_tier(self, tier: int) -> list[TestCase]:
        """Return test cases for a specific tier."""
        return [c for c in self._cases if c.tier == tier]

    def filter_tags(self, tags: list[str]) -> list[TestCase]:
        """Return test cases matching any of the given tags."""
        tag_set = set(tags)
        return [c for c in self._cases if tag_set & set(c.tags)]

    def get(self, case_id: str) -> TestCase | None:
        """Look up a test case by ID."""
        for c in self._cases:
            if c.id == case_id:
                return c
        return None

    def add(self, case: TestCase) -> None:
        """Add a test case to the suite."""
        self._cases.append(case)

    def __len__(self) -> int:
        return len(self._cases)

    def to_yaml(self) -> str:
        """Serialize suite to YAML."""
        data = []
        for c in self._cases:
            entry: dict[str, Any] = {
                "id": c.id,
                "name": c.name,
                "tier": c.tier,
                "prompt": c.prompt,
            }
            if c.expected_tools:
                entry["expected_tools"] = c.expected_tools
            if c.expected_output_contains:
                entry["expected_output_contains"] = c.expected_output_contains
            if c.expected_output_not_contains:
                entry["expected_output_not_contains"] = c.expected_output_not_contains
            if c.expected_file_exists:
                entry["expected_file_exists"] = c.expected_file_exists
            if c.expected_file_contains:
                entry["expected_file_contains"] = c.expected_file_contains
            if c.max_turns != 10:
                entry["max_turns"] = c.max_turns
            if c.setup_commands:
                entry["setup_commands"] = c.setup_commands
            if c.teardown_commands:
                entry["teardown_commands"] = c.teardown_commands
            if c.tags:
                entry["tags"] = c.tags
            data.append(entry)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "BenchmarkSuite":
        """Load suite from YAML string."""
        data = yaml.safe_load(yaml_str) or []
        cases = [TestCase(**entry) for entry in data]
        return cls(cases)

    @classmethod
    def from_file(cls, path: Path | str) -> "BenchmarkSuite":
        """Load suite from a YAML file."""
        return cls.from_yaml(Path(path).read_text())


def _builtin_tier1() -> list[TestCase]:
    """20+ atomic tool call tests."""
    return [
        # --- Bash tool tests ---
        TestCase(
            id="t1_bash_echo",
            name="Bash echo",
            tier=1,
            prompt="Run: echo 'hello prometheus'",
            expected_tools=["bash"],
            expected_output_contains=["hello prometheus"],
            tags=["bash"],
        ),
        TestCase(
            id="t1_bash_pwd",
            name="Bash pwd",
            tier=1,
            prompt="Run pwd and tell me the current directory.",
            expected_tools=["bash"],
            tags=["bash"],
        ),
        TestCase(
            id="t1_bash_math",
            name="Bash arithmetic",
            tier=1,
            prompt="Use bash to compute: echo $((42 * 17))",
            expected_tools=["bash"],
            expected_output_contains=["714"],
            tags=["bash"],
        ),
        TestCase(
            id="t1_bash_env",
            name="Bash environment variable",
            tier=1,
            prompt="Run: echo $HOME",
            expected_tools=["bash"],
            tags=["bash"],
        ),
        # --- FileWrite tests ---
        TestCase(
            id="t1_write_file",
            name="Write a file",
            tier=1,
            prompt='Create a file called /tmp/prom_test_hello.txt with the content "Hello, Prometheus!"',
            expected_tools=["write_file"],
            expected_file_exists=["/tmp/prom_test_hello.txt"],
            expected_file_contains={"/tmp/prom_test_hello.txt": "Hello, Prometheus!"},
            teardown_commands=["rm -f /tmp/prom_test_hello.txt"],
            tags=["file_write"],
        ),
        TestCase(
            id="t1_write_python",
            name="Write Python file",
            tier=1,
            prompt='Create /tmp/prom_test_add.py with a function add(a, b) that returns a + b.',
            expected_tools=["write_file"],
            expected_file_exists=["/tmp/prom_test_add.py"],
            expected_file_contains={"/tmp/prom_test_add.py": "def add("},
            teardown_commands=["rm -f /tmp/prom_test_add.py"],
            tags=["file_write"],
        ),
        # --- FileRead tests ---
        TestCase(
            id="t1_read_file",
            name="Read a file",
            tier=1,
            prompt="Read the file /tmp/prom_test_read.txt and tell me its contents.",
            expected_tools=["read_file"],
            expected_output_contains=["benchmark read test"],
            setup_commands=["echo 'benchmark read test' > /tmp/prom_test_read.txt"],
            teardown_commands=["rm -f /tmp/prom_test_read.txt"],
            tags=["file_read"],
        ),
        TestCase(
            id="t1_read_partial",
            name="Read file with offset",
            tier=1,
            prompt="Read lines 2-3 of /tmp/prom_test_lines.txt.",
            expected_tools=["read_file"],
            setup_commands=[
                "printf 'line1\\nline2\\nline3\\nline4\\n' > /tmp/prom_test_lines.txt",
            ],
            teardown_commands=["rm -f /tmp/prom_test_lines.txt"],
            tags=["file_read"],
        ),
        # --- FileEdit tests ---
        TestCase(
            id="t1_edit_replace",
            name="Edit file — replace string",
            tier=1,
            prompt='In /tmp/prom_test_edit.txt, replace "foo" with "bar".',
            expected_tools=["edit_file"],
            expected_file_contains={"/tmp/prom_test_edit.txt": "bar"},
            setup_commands=["echo 'hello foo world' > /tmp/prom_test_edit.txt"],
            teardown_commands=["rm -f /tmp/prom_test_edit.txt"],
            tags=["file_edit"],
        ),
        # --- Glob tests ---
        TestCase(
            id="t1_glob_py",
            name="Glob for Python files",
            tier=1,
            prompt="Find all .py files under /tmp/prom_test_glob/ using glob.",
            expected_tools=["glob"],
            setup_commands=[
                "mkdir -p /tmp/prom_test_glob/sub",
                "touch /tmp/prom_test_glob/a.py /tmp/prom_test_glob/sub/b.py",
            ],
            teardown_commands=["rm -rf /tmp/prom_test_glob"],
            tags=["glob"],
        ),
        TestCase(
            id="t1_glob_md",
            name="Glob for Markdown files",
            tier=1,
            prompt="Find all .md files in /tmp/prom_test_glob_md/.",
            expected_tools=["glob"],
            setup_commands=[
                "mkdir -p /tmp/prom_test_glob_md",
                "touch /tmp/prom_test_glob_md/README.md /tmp/prom_test_glob_md/notes.md",
            ],
            teardown_commands=["rm -rf /tmp/prom_test_glob_md"],
            tags=["glob"],
        ),
        # --- Grep tests ---
        TestCase(
            id="t1_grep_pattern",
            name="Grep for pattern",
            tier=1,
            prompt='Search for the word "error" in /tmp/prom_test_grep/log.txt.',
            expected_tools=["grep"],
            expected_output_contains=["error"],
            setup_commands=[
                "mkdir -p /tmp/prom_test_grep",
                "printf 'info: started\\nerror: failed\\ninfo: done\\n' > /tmp/prom_test_grep/log.txt",
            ],
            teardown_commands=["rm -rf /tmp/prom_test_grep"],
            tags=["grep"],
        ),
        TestCase(
            id="t1_grep_regex",
            name="Grep with regex",
            tier=1,
            prompt='Search for lines matching "def \\w+" in /tmp/prom_test_grep_re/code.py.',
            expected_tools=["grep"],
            setup_commands=[
                "mkdir -p /tmp/prom_test_grep_re",
                "printf 'def hello():\\n    pass\\ndef world():\\n    pass\\n' > /tmp/prom_test_grep_re/code.py",
            ],
            teardown_commands=["rm -rf /tmp/prom_test_grep_re"],
            tags=["grep"],
        ),
        # --- CronCreate / CronList / CronDelete ---
        TestCase(
            id="t1_cron_create",
            name="Create a cron job",
            tier=1,
            prompt='Create a cron job that runs every 5 minutes with prompt "health check".',
            expected_tools=["cron_create"],
            tags=["cron"],
        ),
        TestCase(
            id="t1_cron_list",
            name="List cron jobs",
            tier=1,
            prompt="List all currently scheduled cron jobs.",
            expected_tools=["cron_list"],
            tags=["cron"],
        ),
        # --- TodoWrite ---
        TestCase(
            id="t1_todo_write",
            name="Create a todo list",
            tier=1,
            prompt='Create a todo list with these items: "Write tests" (pending), "Fix bug" (in_progress).',
            expected_tools=["todo_write"],
            tags=["todo"],
        ),
        # --- Skill tool ---
        TestCase(
            id="t1_skill_not_found",
            name="Skill tool — unknown skill",
            tier=1,
            prompt="Run the skill called /nonexistent-skill-xyz.",
            expected_tools=["skill"],
            tags=["skill"],
        ),
        # --- Agent tool ---
        TestCase(
            id="t1_agent_spawn",
            name="Agent tool — spawn explorer",
            tier=1,
            prompt='Use the Agent tool to spawn an explorer subagent with prompt "List files in /tmp".',
            expected_tools=["Agent"],
            tags=["agent"],
        ),
        # --- Error handling ---
        TestCase(
            id="t1_bash_fail",
            name="Bash — handle command failure",
            tier=1,
            prompt="Run: ls /nonexistent_dir_xyz_12345",
            expected_tools=["Bash"],
            tags=["bash", "error"],
        ),
        TestCase(
            id="t1_read_missing",
            name="Read — handle missing file",
            tier=1,
            prompt="Read the file /tmp/absolutely_nonexistent_file_xyz.txt.",
            expected_tools=["read_file"],
            tags=["file_read", "error"],
        ),
        # --- JSON / structured output ---
        TestCase(
            id="t1_bash_json",
            name="Bash — produce JSON",
            tier=1,
            prompt="Run: echo '{\"status\": \"ok\", \"count\": 42}'",
            expected_tools=["Bash"],
            expected_output_contains=["ok"],
            tags=["bash"],
        ),
    ]


def _builtin_tier2() -> list[TestCase]:
    """5 multi-step workflow tests."""
    return [
        TestCase(
            id="t2_create_and_read",
            name="Write file then read it back",
            tier=2,
            prompt=(
                "Create a file /tmp/prom_t2_roundtrip.txt with content 'round trip test', "
                "then read it back and confirm the content."
            ),
            expected_tools=["write_file", "read_file"],
            expected_output_contains=["round trip test"],
            expected_file_exists=["/tmp/prom_t2_roundtrip.txt"],
            teardown_commands=["rm -f /tmp/prom_t2_roundtrip.txt"],
            max_turns=15,
            tags=["multi_step", "file"],
        ),
        TestCase(
            id="t2_write_and_run",
            name="Write Python script then execute it",
            tier=2,
            prompt=(
                "Create /tmp/prom_t2_script.py that prints 'hello from script', "
                "then run it with python3 and show the output."
            ),
            expected_tools=["write_file", "bash"],
            expected_output_contains=["hello from script"],
            expected_file_exists=["/tmp/prom_t2_script.py"],
            teardown_commands=["rm -f /tmp/prom_t2_script.py"],
            max_turns=15,
            tags=["multi_step", "bash", "file_write"],
        ),
        TestCase(
            id="t2_search_and_read",
            name="Glob for files then read one",
            tier=2,
            prompt=(
                "Find all .txt files in /tmp/prom_t2_search/, then read the one "
                "called 'target.txt' and tell me its contents."
            ),
            expected_tools=["glob", "read_file"],
            expected_output_contains=["found me"],
            setup_commands=[
                "mkdir -p /tmp/prom_t2_search",
                "echo 'decoy' > /tmp/prom_t2_search/decoy.txt",
                "echo 'found me' > /tmp/prom_t2_search/target.txt",
            ],
            teardown_commands=["rm -rf /tmp/prom_t2_search"],
            max_turns=15,
            tags=["multi_step", "glob", "file_read"],
        ),
        TestCase(
            id="t2_edit_and_verify",
            name="Edit a file then grep to verify",
            tier=2,
            prompt=(
                'Edit /tmp/prom_t2_edit.txt: replace "old_value" with "new_value", '
                'then grep the file to confirm "new_value" is present.'
            ),
            expected_tools=["edit_file", "grep"],
            expected_output_contains=["new_value"],
            setup_commands=[
                "echo 'config=old_value' > /tmp/prom_t2_edit.txt",
            ],
            teardown_commands=["rm -f /tmp/prom_t2_edit.txt"],
            max_turns=15,
            tags=["multi_step", "file_edit", "grep"],
        ),
        TestCase(
            id="t2_multi_file_workflow",
            name="Create directory, write files, list them",
            tier=2,
            prompt=(
                "Use bash to create directory /tmp/prom_t2_multi/, "
                "write two files inside it (a.py and b.py) each with a simple function, "
                "then glob to list all .py files in that directory."
            ),
            expected_tools=["bash", "write_file", "glob"],
            expected_file_exists=["/tmp/prom_t2_multi/a.py", "/tmp/prom_t2_multi/b.py"],
            teardown_commands=["rm -rf /tmp/prom_t2_multi"],
            max_turns=20,
            tags=["multi_step", "bash", "file_write", "glob"],
        ),
    ]


def load_suite(tier: int | None = None) -> BenchmarkSuite:
    """Load the built-in benchmark suite, optionally filtered by tier."""
    cases = _builtin_tier1() + _builtin_tier2()
    suite = BenchmarkSuite(cases)
    if tier is not None:
        return BenchmarkSuite(suite.filter_tier(tier))
    return suite
