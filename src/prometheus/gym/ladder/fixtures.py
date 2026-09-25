"""Sandbox, fixture tools and seeding for ladder runs.

Every ladder run gets a FRESH sandbox: a workspace for the task's files and a
private stand-in for ``~/.prometheus`` (config dir, data dir, wiki), selected
through the same ``PROMETHEUS_*`` environment variables the daemon honours.
So a scheduling task's ``cron_create`` writes a throwaway ``cron_jobs.json``,
a memory task's ``MEMORY.md`` is the sandbox's, and nothing a model does in a
run can reach the real stores or the next run.

The web is OFFLINE by default. ``web_fetch`` and ``web_search`` keep their
real name, description and schema — the model sees and calls exactly the
tools the daemon registers — but answer from the task's fixture pages, on a
reserved ``.example`` domain that cannot resolve publicly. That makes a web
verdict reproducible across machines and dates. Live-web tasks use the real
tools and are opt-in.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus.tools.base import ToolRegistry, ToolResult

# The env vars that relocate everything a ladder task can write outside its
# workspace. PROMETHEUS_NODE_DIR is handled separately (see activate): the
# node identity is only READ (telemetry stamps node_id on rows), and a row
# that names the machine that produced it is the point.
SANDBOX_ENV_KEYS = (
    "PROMETHEUS_CONFIG_DIR",
    "PROMETHEUS_DATA_DIR",
    "PROMETHEUS_LOGS_DIR",
    "PROMETHEUS_WIKI",
    "PROMETHEUS_VAULT",
    "PROMETHEUS_DOCUMENTS_DIR",
    "PROMETHEUS_ARTIFACTS_DIR",
    "PROMETHEUS_WORKSPACE_DIR",
)


@dataclass
class Sandbox:
    root: Path
    _lock_fh: Any = None

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def workspace(self) -> Path:
        return self.root / "ws"

    @property
    def data(self) -> Path:
        return self.home / "data"

    @property
    def wiki(self) -> Path:
        return self.home / "wiki"

    def env(self) -> dict[str, str]:
        return {
            "PROMETHEUS_CONFIG_DIR": str(self.home),
            "PROMETHEUS_DATA_DIR": str(self.data),
            "PROMETHEUS_LOGS_DIR": str(self.home / "logs"),
            "PROMETHEUS_WIKI": str(self.wiki),
            "PROMETHEUS_VAULT": str(self.home / "vault"),
            "PROMETHEUS_DOCUMENTS_DIR": str(self.home / "documents"),
            "PROMETHEUS_ARTIFACTS_DIR": str(self.home / "artifacts"),
            "PROMETHEUS_WORKSPACE_DIR": str(self.home / "workspaces"),
        }

    def activate(self) -> dict[str, str | None]:
        """Point the process's PROMETHEUS_* paths into the sandbox.

        The node identity stays the machine's: the node dir defaults to
        ``<config dir>/node``, and relocating the config dir would otherwise
        make every row's node_id NULL. It is pinned only when an identity
        already exists — reading one must never create one.

        Returns the previous values so a caller can restore them.
        """
        from prometheus.config.paths import config_dir_path

        keys = (*SANDBOX_ENV_KEYS, "PROMETHEUS_NODE_DIR")
        previous = {k: os.environ.get(k) for k in keys}
        env = self.env()
        if "PROMETHEUS_NODE_DIR" not in os.environ:
            real_node = config_dir_path() / "node"
            if real_node.is_dir():
                env["PROMETHEUS_NODE_DIR"] = str(real_node)
        os.environ.update(env)
        return previous

    @staticmethod
    def restore(previous: dict[str, str | None]) -> None:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def reset(self) -> None:
        """Wipe and recreate the home and workspace — one clean slate per run.

        ``data/security`` is KEPT: the SecurityGate's audit logger opened its
        database there when the pipeline was built (once per ladder run), and
        wiping it mid-run leaves the logger writing to a file with no table.
        Nothing a task can do writes there; it is the gate's own record.
        """
        keep = self.data / "security"
        _remove(self.workspace)
        if self.home.exists():
            for child in self.home.iterdir():
                if child == self.data and not child.is_symlink():
                    for sub in child.iterdir():
                        if sub != keep:
                            _remove(sub)
                else:
                    _remove(child)
        for d in (self.home, self.data, keep, self.workspace):
            d.mkdir(parents=True, exist_ok=True)
        leftovers = [p.name for p in self.workspace.iterdir()]
        if leftovers:
            raise SandboxError(f"workspace not empty after reset: {leftovers[:5]}")

    def lock(self) -> None:
        """Hold the sandbox for this process. Two ladder runs on one root
        would wipe each other's workspace, cron registry and wiki mid-task."""
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_fh = open(self.root / ".ladder.lock", "w")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_fh.close()
            self._lock_fh = None
            raise SandboxError(
                f"another ladder run is using {self.root} — pass a different --workdir"
            ) from None

    def unlock(self) -> None:
        fh = getattr(self, "_lock_fh", None)
        if fh is not None:
            fh.close()
            self._lock_fh = None


class SandboxError(RuntimeError):
    """The sandbox cannot give a run a clean, private slate."""


def _remove(path: Path) -> None:
    """Delete *path* whatever the run left there — read-only directories
    (``chmod -R a-w``) and symlinks included. A leftover that survives is a
    SandboxError from reset(), not a silently dirty next run."""
    import stat

    def _chmod_and_retry(func, target, _exc):  # noqa: ANN001
        parent = os.path.dirname(target)
        for p in (parent, target):
            try:
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
            except OSError:
                pass
        func(target)

    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path.parent, os.stat(path.parent).st_mode | stat.S_IWUSR | stat.S_IXUSR)
            path.unlink()
    elif path.is_dir():
        import sys as _sys

        if _sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_chmod_and_retry)
        else:  # pragma: no cover — 3.11
            shutil.rmtree(path, onerror=_chmod_and_retry)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def write_files(base: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        target = (base / rel).resolve()
        if base.resolve() not in target.parents and target != base.resolve():
            raise ValueError(f"fixture path {rel!r} escapes {base}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def seed_cron_jobs(jobs: list[dict[str, Any]], workspace: Path) -> None:
    from prometheus.gateway.cron_service import upsert_cron_job

    for job in jobs:
        upsert_cron_job({
            "name": job["name"],
            "schedule": job["schedule"],
            "command": job["command"],
            "cwd": str(workspace),
            "enabled": bool(job.get("enabled", True)),
        })


def seed_memory_files(sandbox: Sandbox, memory_md: str | None, user_md: str | None) -> None:
    if memory_md:
        (sandbox.home / "MEMORY.md").write_text(memory_md.strip() + "\n")
    if user_md:
        (sandbox.home / "USER.md").write_text(user_md.strip() + "\n")


def seed_lcm(provider: Any, sandbox: Sandbox, history: list[dict[str, Any]]) -> Any:
    """A real LCMEngine on the sandbox DB, carrying the task's prior sessions.

    Always wired (an empty history is a real, empty store), so ``lcm_grep``
    behaves the same in every task — the model is never told "not initialised"
    in one class and given a store in another.
    """
    from prometheus.config.paths import get_lcm_db_path
    from prometheus.memory.lcm_engine import LCMEngine
    from prometheus.tools.builtin.lcm_grep import set_lcm_engine

    # The one resolver every site uses; with the sandbox active it lands in
    # the sandbox's data dir.
    db_path = get_lcm_db_path()
    if sandbox.data.resolve() not in db_path.resolve().parents:
        raise RuntimeError(f"LCM path {db_path} is outside the sandbox — is it active?")
    engine = LCMEngine(provider, db_path=db_path)
    turn: dict[str, int] = {}
    for msg in history:
        sid = str(msg["session"])
        engine.ingest_sync(sid, msg["role"], msg["content"], turn_index=turn.get(sid, 0))
        turn[sid] = turn.get(sid, 0) + 1
    set_lcm_engine(engine)
    return engine


def seed_task(task: Any, sandbox: Sandbox, provider: Any) -> Any:
    """Everything a task needs on disk before the model sees it.

    Returns the LCM engine so the caller can close it after the run.
    """
    fx = task.fixtures
    write_files(sandbox.workspace, task.setup_files)
    if fx.get("wiki_pages"):
        write_files(sandbox.wiki, fx["wiki_pages"])
    if fx.get("cron_jobs"):
        seed_cron_jobs(fx["cron_jobs"], sandbox.workspace)
    seed_memory_files(sandbox, fx.get("memory_md"), fx.get("user_md"))
    return seed_lcm(provider, sandbox, fx.get("lcm_history") or [])


def memory_prompt_section() -> str:
    """MEMORY.md + USER.md rendered exactly as the daemon renders them."""
    from prometheus.memory.hermes_memory_tool import format_memory_for_prompt

    return format_memory_for_prompt()


# ---------------------------------------------------------------------------
# Offline web
# ---------------------------------------------------------------------------


def _norm_url(url: str) -> str:
    u = url.strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.rstrip("/").lower()


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


class FixtureWeb:
    """The task's fixture pages, addressable by URL and searchable by words."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self._by_url = {_norm_url(p["url"]): p for p in pages}

    def fetch(self, url: str) -> dict[str, Any] | None:
        return self._by_url.get(_norm_url(url))

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        q = _tokens(query)
        scored = []
        for i, page in enumerate(self.pages):
            title_hits = len(q & _tokens(page.get("title", "")))
            body_hits = len(q & _tokens(page.get("body", "")))
            score = 3 * title_hits + body_hits
            if score:
                scored.append((-score, i, page))
        scored.sort()
        return [p for _s, _i, p in scored[:limit]]


def fixture_web_tools(web: FixtureWeb) -> list[Any]:
    """web_fetch / web_search with the real schema, answering from fixtures."""
    from prometheus.tools.builtin.web_fetch import WebFetchTool
    from prometheus.tools.builtin.web_search import WebSearchTool

    class FixtureWebFetch(WebFetchTool):
        async def execute(self, arguments, context):  # noqa: ANN001
            page = web.fetch(arguments.url)
            if page is None:
                # The same shape the real tool returns for a 404.
                return ToolResult(
                    output=(
                        f"web_fetch failed: Client error '404 Not Found' for url "
                        f"'{arguments.url}'"
                    ),
                    is_error=True,
                )
            body = f"{page.get('title', '')}\n\n{page.get('body', '')}".strip()
            if len(body) > arguments.max_chars:
                body = body[: arguments.max_chars].rstrip() + "\n...[truncated]"
            return ToolResult(
                output=(
                    f"URL: {page['url']}\nStatus: 200\n"
                    f"Content-Type: text/html; charset=utf-8\n\n{body}"
                )
            )

    class FixtureWebSearch(WebSearchTool):
        async def execute(self, arguments, context):  # noqa: ANN001
            hits = web.search(arguments.query, arguments.max_results)
            if not hits:
                return ToolResult(output="No search results found.", is_error=True)
            lines = [f"Search results for: {arguments.query}"]
            for i, page in enumerate(hits, start=1):
                snippet = page.get("snippet") or " ".join(page.get("body", "").split())[:160]
                lines.append(f"{i}. {page.get('title', page['url'])}")
                lines.append(f"   URL: {page['url']}")
                if snippet:
                    lines.append(f"   {snippet}")
            return ToolResult(output="\n".join(lines))

    return [FixtureWebFetch(), FixtureWebSearch()]


# ---------------------------------------------------------------------------
# The ladder's tool surface — ONE registry for every class
# ---------------------------------------------------------------------------

LADDER_TOOLS: tuple[str, ...] = (
    "bash", "read_file", "write_file", "edit_file", "grep", "glob",
    "web_fetch", "web_search", "lcm_grep", "wiki_query", "memory",
    "cron_create", "cron_list", "cron_delete",
)


def build_ladder_registry(
    workspace: Path,
    *,
    web_pages: list[dict[str, Any]] | None = None,
    live_web: bool = False,
) -> ToolRegistry:
    """The same fourteen tools for every task, so a class's pass rate reflects
    the task, not a different tool menu. Web is fixture-backed unless the task
    is a live-web task."""
    from prometheus.memory.hermes_memory_tool import MemoryTool
    from prometheus.tools.builtin import (
        BashTool,
        FileEditTool,
        FileReadTool,
        FileWriteTool,
        GlobTool,
        GrepTool,
    )
    from prometheus.tools.builtin.cron_create import CronCreateTool
    from prometheus.tools.builtin.cron_delete import CronDeleteTool
    from prometheus.tools.builtin.cron_list import CronListTool
    from prometheus.tools.builtin.lcm_grep import LCMGrepTool
    from prometheus.tools.builtin.wiki_query import WikiQueryTool

    registry = ToolRegistry()
    registry.register(BashTool(workspace=str(workspace)))
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GrepTool())
    registry.register(GlobTool())
    if live_web:
        from prometheus.tools.builtin.web_fetch import WebFetchTool
        from prometheus.tools.builtin.web_search import WebSearchTool

        registry.register(WebFetchTool())
        registry.register(WebSearchTool())
    else:
        for tool in fixture_web_tools(FixtureWeb(web_pages or [])):
            registry.register(tool)
    registry.register(LCMGrepTool())
    registry.register(WikiQueryTool())
    registry.register(MemoryTool())
    registry.register(CronCreateTool())
    registry.register(CronListTool())
    registry.register(CronDeleteTool())
    return registry
