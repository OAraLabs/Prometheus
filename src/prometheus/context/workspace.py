"""Per-session workspace — validation shared by every surface that can set one.

Item W (2026-09-01). A conversation can be pointed at a directory; from the
next turn that directory is where relative paths resolve, where project
instruction files are discovered, and — the gate follows the session — the
conversation's write boundary for `write_file`/`edit_file`, bash's lock and
the bash write floor. Setting it is therefore a security-relevant write, and
the rules below apply identically over REST, Telegram, Slack, Discord and
the web chat: one function, every surface.
"""

from __future__ import annotations

from pathlib import Path


def validate_workspace_path(raw: str, security_cfg: dict | None) -> tuple[Path | None, str | None]:
    """Resolve *raw* to a workspace directory, or explain why it is refused.

    Refused: a relative path (a chat surface has no cwd to resolve it
    against, so the answer would depend on which surface asked), a path that
    is not an existing directory, the filesystem root, and anything under
    ``security.denied_paths``. Returns ``(resolved, None)`` or ``(None, why)``.
    """
    from prometheus.config.shipped_defaults import resolve_denied_paths

    text = (raw or "").strip()
    if not text:
        return None, "a path is required"
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return None, "workspace must be an absolute path (or start with ~)"
    resolved = candidate.resolve()
    if not resolved.is_dir():
        return None, f"{resolved} is not an existing directory"
    if resolved == Path(resolved.anchor):
        return None, "the filesystem root cannot be a workspace"
    for denied in resolve_denied_paths(security_cfg or {}):
        droot = Path(denied).expanduser().resolve()
        try:
            resolved.relative_to(droot)
        except ValueError:
            continue
        return None, f"{resolved} is under denied path {droot}"
    return resolved, None


def describe_workspace(session_id: str, bound: str | None, daemon_cwd: str, roots: list[str]) -> str:
    """One-paragraph, surface-neutral description for /workspace with no args."""
    if bound:
        return (
            f"Workspace for this conversation: {bound}\n"
            "Relative paths resolve there, its instruction files (PROMETHEUS.md, "
            "CLAUDE.md, AGENTS.md, …) are loaded, and it is this conversation's "
            "write boundary.\n"
            "Change: /workspace <absolute path>   Clear: /workspace clear"
        )
    return (
        "No workspace bound to this conversation — it follows the daemon: "
        f"cwd {daemon_cwd}, write boundary {', '.join(roots) or '(none)'}.\n"
        "Set one: /workspace <absolute path>"
    )
