# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/config/paths.py
# License: MIT
# Modified: renamed .openharness → .prometheus, OPENHARNESS_* env vars → PROMETHEUS_*,
#           get_project_config_dir returns .prometheus/, added get_workspace_dir()

"""Path resolution for Prometheus configuration and data directories.

Follows XDG-like conventions with ~/.prometheus/ as the default base directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_BASE_DIR = ".prometheus"
_CONFIG_FILE_NAME = "settings.json"


def get_config_dir() -> Path:
    """Return the configuration directory, creating it if needed.

    Resolution order:
    1. PROMETHEUS_CONFIG_DIR environment variable
    2. ~/.prometheus/
    """
    env_dir = os.environ.get("PROMETHEUS_CONFIG_DIR")
    if env_dir:
        config_dir = Path(env_dir)
    else:
        config_dir = Path.home() / _DEFAULT_BASE_DIR

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file_path() -> Path:
    """Return the path to the main settings file (~/.prometheus/settings.json)."""
    return get_config_dir() / _CONFIG_FILE_NAME


def get_data_dir() -> Path:
    """Return the data directory for caches, history, etc.

    Resolution order:
    1. PROMETHEUS_DATA_DIR environment variable
    2. ~/.prometheus/data/
    """
    env_dir = os.environ.get("PROMETHEUS_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        data_dir = get_config_dir() / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_logs_dir() -> Path:
    """Return the logs directory.

    Resolution order:
    1. PROMETHEUS_LOGS_DIR environment variable
    2. ~/.prometheus/logs/
    """
    env_dir = os.environ.get("PROMETHEUS_LOGS_DIR")
    if env_dir:
        logs_dir = Path(env_dir)
    else:
        logs_dir = get_config_dir() / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_sessions_dir() -> Path:
    """Return the session storage directory."""
    sessions_dir = get_data_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def get_tasks_dir() -> Path:
    """Return the background task output directory."""
    tasks_dir = get_data_dir() / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir


def get_feedback_dir() -> Path:
    """Return the feedback storage directory."""
    feedback_dir = get_data_dir() / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    return feedback_dir


def get_feedback_log_path() -> Path:
    """Return the feedback log file path."""
    return get_feedback_dir() / "feedback.log"


def get_cron_registry_path() -> Path:
    """Return the cron registry file path."""
    return get_data_dir() / "cron_jobs.json"


def get_kanban_db_path() -> Path:
    """Return the Kanban (projects/stories board) SQLite database path."""
    return get_data_dir() / "kanban.db"


def get_tasks_db_path() -> Path:
    """Return the managed-task durable store SQLite database path.

    Separate from :func:`get_tasks_dir` (which holds per-task ``.log`` output
    files): this DB persists ``TaskRecord`` rows so the supervisor can resume
    or reap ``running`` tasks across a daemon restart.
    """
    return get_data_dir() / "tasks.db"


def get_project_config_dir(cwd: str | Path) -> Path:
    """Return the per-project .prometheus directory."""
    project_dir = Path(cwd).resolve() / ".prometheus"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_project_issue_file(cwd: str | Path) -> Path:
    """Return the per-project issue context file."""
    return get_project_config_dir(cwd) / "issue.md"


def get_project_pr_comments_file(cwd: str | Path) -> Path:
    """Return the per-project PR comments context file."""
    return get_project_config_dir(cwd) / "pr_comments.md"


def get_workspace_dir() -> Path:
    """Return the agent workspace directory (~/.prometheus/workspace)."""
    env_dir = os.environ.get("PROMETHEUS_WORKSPACE_DIR")
    if env_dir:
        workspace_dir = Path(env_dir)
    else:
        workspace_dir = get_config_dir() / "workspace"

    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def get_documents_dir() -> Path:
    """Return the confined documents-editor root (~/.prometheus/documents).

    The Documents Editor (daemon docs service + Beacon panel) may only read,
    save, and edit files resolving UNDER this root. It is a sibling of the
    coding-mode jail (~/.prometheus/coding) — a dedicated, confined tree, NOT
    the user's real ~/Documents (a v1 safety decision: a surprise-free root
    inside the daemon's own data area). Repointable via PROMETHEUS_DOCUMENTS_DIR
    or the config ``documents.root`` key.

    Resolution order:
    1. PROMETHEUS_DOCUMENTS_DIR environment variable
    2. ~/.prometheus/documents/
    """
    env_dir = os.environ.get("PROMETHEUS_DOCUMENTS_DIR")
    if env_dir:
        documents_dir = Path(env_dir)
    else:
        documents_dir = get_config_dir() / "documents"

    documents_dir.mkdir(parents=True, exist_ok=True)
    return documents_dir


# ---------------------------------------------------------------------------
# Wiki root — THE single resolution point.
#
# Before 2026-08-02 nine call sites derived this path independently, in three
# different ways: six as ``get_config_dir() / "wiki"``, three as
# ``Path.home() / ".prometheus" / "wiki"`` (which ignores PROMETHEUS_CONFIG_DIR
# entirely, so a non-default config dir split the wiki in two — the compiler
# wrote one tree while Beacon's /api/wiki and both Telegram surfaces read
# another), and one shell script as ``${PROMETHEUS_WIKI:-$HOME/.prometheus/wiki}``.
#
# Everything now reads through here. No consumer keeps a fallback.
# ---------------------------------------------------------------------------

_wiki_root_override: Path | None = None


def set_wiki_root(root: str | Path | None) -> None:
    """Pin the wiki root for this process (called once at daemon start).

    Exists because :class:`~prometheus.tools.builtin.wiki_query.WikiQueryTool`
    resolves the root inside ``execute()`` rather than ``__init__`` — it is
    registered with no arguments, so it cannot be handed the root at
    construction the way the other three consumers are. Mirrors the existing
    ``set_wiki_compiler`` / ``set_wiki_linter`` idiom in ``daemon.py``.

    Passing ``None`` clears the override (used by tests).
    """
    global _wiki_root_override
    _wiki_root_override = Path(root).expanduser() if root is not None else None


def resolve_wiki_root(config: dict | None = None) -> Path:
    """Resolve the wiki root from config, env, then the default.

    Resolution order:
    1. ``wiki.root`` in the supplied config mapping
    2. ``PROMETHEUS_WIKI`` environment variable
    3. ``get_config_dir() / "wiki"``

    Pure: does not read or mutate the process-wide override, and — unlike the
    other directory helpers here — **does not create the directory**. Several
    consumers branch on ``wiki_root.exists()`` to detect "no wiki compiled
    yet"; creating it would silently change that behaviour.
    """
    if config:
        configured = (config.get("wiki") or {}).get("root")
        if configured:
            return Path(str(configured)).expanduser()
    env_dir = os.environ.get("PROMETHEUS_WIKI")
    if env_dir:
        return Path(env_dir).expanduser()
    return get_config_dir() / "wiki"


def get_wiki_root() -> Path:
    """Return the wiki root: the pinned override, else env, else the default.

    This is the function every consumer calls. Out of the box — no override
    pinned, no ``PROMETHEUS_WIKI`` set, no ``wiki.root`` key — it returns
    ``get_config_dir() / "wiki"``, exactly what the nine sites it replaced
    resolved to.
    """
    if _wiki_root_override is not None:
        return _wiki_root_override
    return resolve_wiki_root()


def get_artifacts_dir() -> Path:
    """Return the agent's artifact OUTBOX (~/.prometheus/files).

    The delivery boundary for files the agent produces FOR the human: anything
    saved here is published — indexed by /api/artifacts and downloadable by
    remote clients (Beacon's chat download chips). Nothing outside it is ever
    served, so the agent controls exactly what crosses the wire by choosing
    where it writes. Sibling of the workspace (agent working files) and the
    documents root (editor surface); ~/.prometheus/files is where the agent
    already saves deliverables by convention.

    Resolution order:
    1. PROMETHEUS_ARTIFACTS_DIR environment variable
    2. ~/.prometheus/files/
    """
    env_dir = os.environ.get("PROMETHEUS_ARTIFACTS_DIR")
    if env_dir:
        artifacts_dir = Path(env_dir)
    else:
        artifacts_dir = get_config_dir() / "files"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir
