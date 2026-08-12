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


# ---------------------------------------------------------------------------
# LCM DATABASE — THE single resolution point.
#
# Before 2026-08-12 four sites named this file and they did not agree. Three
# private ``_default_db_path()`` copies (``memory/lcm_conversation_store.py``,
# ``memory/lcm_summary_store.py``, ``coordinator/divergence.py``) returned
# ``get_config_dir() / "lcm.db"``, while ``LCMEngine`` passed
# ``get_data_dir() / "lcm.db"`` explicitly to the two stores it builds.
#
# So "the shared lcm.db" — a phrase in all three class docstrings — was TWO
# files: conversations and summaries in ``data/``, checkpoints in the config
# root. ``CheckpointStore()`` is bare-constructed at daemon start regardless of
# ``divergence.enabled``, so every install grows the config-root file whether or
# not the feature is on.
#
# Which location is correct was never ambiguous: LCMConversationStore's own
# schema creates the ``checkpoints`` table (lcm_conversation_store.py, "Checkpoint
# table for divergence detection"), in the data-dir file, and CheckpointStore's
# docstring says it "uses the same database as LCMConversationStore and
# LCMSummaryStore to keep all conversation state in one place." The intent was
# always one file in ``data/``; only the checkpoint writer's default disagreed.
#
# Everything now reads through here. No consumer keeps a fallback, and
# ``tests/test_lcm_db_path_resolution.py`` fails the build on a re-derived path.
# ---------------------------------------------------------------------------

_LCM_DB_NAME = "lcm.db"


def get_lcm_db_path() -> Path:
    """Return the LCM database path (``<data dir>/lcm.db``).

    One file, three table families: conversation messages
    (:class:`~prometheus.memory.lcm_conversation_store.LCMConversationStore`),
    the summary DAG (:class:`~prometheus.memory.lcm_summary_store.LCMSummaryStore`)
    and divergence checkpoints
    (:class:`~prometheus.coordinator.divergence.CheckpointStore`).

    Callers that need a different file — tests, the gym, per-run sandboxes —
    pass ``db_path`` explicitly. This is only the default.
    """
    return get_data_dir() / _LCM_DB_NAME


def get_legacy_lcm_db_path() -> Path:
    """Return the pre-2026-08-12 config-root ``lcm.db``.

    NOT a location anything writes to any more. It exists so the one reader
    that must still know about it — ``prometheus reset-data``, which promises
    to delete all user data — can clean up a file left behind on installs that
    predate the single-resolution-point fix. ``checkpoints.messages_json``
    holds full conversation messages, so a forgotten file here is a privacy
    miss, not just clutter.

    Deliberately NOT migrated into :func:`get_lcm_db_path`'s file: see the
    FL-3 PR body for the trade, and ``tests/test_lcm_db_path_resolution.py``
    for the test that pins the decision as deliberate.
    """
    return get_config_dir() / _LCM_DB_NAME


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


# ---------------------------------------------------------------------------
# BRAIN VAULT root — the second brain (~/brain-vault)
#
# A SECOND, SEPARATE root. It is NOT the Prometheus wiki: that one lives under
# ``get_wiki_root()`` (default ``~/.prometheus/wiki``) and is a machine-owned
# projection of ``memory.db``. The brain vault is a git repo of hand- and
# ingest-compiled knowledge with its own zone rules (its BRAIN.md §1), and
# conflating the two would let a writer aimed at one land in the other.
# Whether the Prometheus wiki eventually becomes a zone inside the vault is
# explicitly DEFERRED — until then they are two roots and two idioms.
#
# NAMING: everything user-facing says "brain vault", never bare "vault".
# ``symbiote.backup.vault_root`` already exists and means something entirely
# different (snapshot storage), and one word meaning two things in one config
# file is the collision class that has already cost a session.
# ---------------------------------------------------------------------------

_vault_root_override: Path | None = None

_DEFAULT_VAULT_DIR = "brain-vault"


def set_vault_root(root: str | Path | None) -> None:
    """Pin the brain-vault root for this process (called once at daemon start).

    Same reason as :func:`set_wiki_root`: the vault tools are registered with
    no arguments and resolve the root inside ``execute()``, so they cannot be
    handed it at construction.

    Passing ``None`` clears the override (used by tests).
    """
    global _vault_root_override
    _vault_root_override = Path(root).expanduser() if root is not None else None


def resolve_vault_root(config: dict | None = None) -> Path:
    """Resolve the brain-vault root from config, env, then the default.

    Resolution order:
    1. ``vault.root`` in the supplied config mapping
    2. ``PROMETHEUS_VAULT`` environment variable
    3. ``Path.home() / "brain-vault"``

    Pure: does not read or mutate the process-wide override, and — like
    :func:`resolve_wiki_root` and unlike the other helpers here — **does not
    create the directory**. An absent vault is a real, reportable state
    ("the brain vault is not present at X"), and creating an empty one would
    convert that loud failure into a silent no-results.
    """
    if config:
        configured = (config.get("vault") or {}).get("root")
        if configured:
            return Path(str(configured)).expanduser()
    env_dir = os.environ.get("PROMETHEUS_VAULT")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / _DEFAULT_VAULT_DIR


def get_vault_root() -> Path:
    """Return the brain-vault root: pinned override, else env, else default.

    This is the function every consumer calls. There is exactly one other way
    to name this location and it is a bug — see
    ``tests/test_vault_root_resolution.py``, which fails the build on a
    re-derived root.
    """
    if _vault_root_override is not None:
        return _vault_root_override
    return resolve_vault_root()


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
