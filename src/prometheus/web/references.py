"""@-references — server-side resolution of composer references (item 6).

A reference is what the composer's ``@src/app.py`` / ``@diff HEAD~1`` /
``@https://…`` chip stands for. The CLIENT names it; the DAEMON reads it.
Beacon runs remote from this host, so "the file" is the daemon's file, "the
diff" is the session workspace's diff, and a URL is fetched under the same
SSRF guard ``web_fetch`` uses. Nothing is resolved client-side and nothing is
pasted into the message text: each reference becomes its own ``TextBlock``
persisted WITH the user turn (the #339 blocks path), so history shows what the
model was actually given.

Resolution happens BEFORE the turn is queued. A reference that cannot be
resolved is a ``4xx`` on ``POST /api/chat/send`` or an ``error`` frame on the
WS ``send_message`` command — never a silently dropped chip and never a turn
that runs with less context than the user believed they attached.

Scope rules
-----------
* File paths resolve against the session's workspace (item W, #380) when the
  session has one, and must stay inside it. Without a workspace they resolve
  against the browse root ``/api/files`` uses (``PROMETHEUS_FILES_ROOT`` or the
  agent workspace) and may also sit under a configured
  ``security.workspace_root``.
* Denied paths are the gate's own list — the same ``SecurityGate`` matcher
  (config ``security.denied_paths`` plus the always-denied floor). A path the
  agent could not ``read_file`` cannot be @-referenced either.
* ``diff`` runs ``git diff`` in the session workspace (or the browse root) with
  a validated ref — option-shaped refs are refused before git sees them.
* ``url`` is http(s) only and goes through ``web_fetch``'s SSRF check.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from prometheus.config.shipped_defaults import resolve_denied_paths, resolve_workspace_root
from prometheus.engine.messages import TextBlock

log = logging.getLogger(__name__)

REFERENCE_TYPES: tuple[str, ...] = ("file", "diff", "url")
MAX_REFERENCES = 16
FILE_CAP_BYTES = 256 * 1024   # matches /api/files/read's text-preview cap
DIFF_CAP_CHARS = 128 * 1024
URL_CAP_CHARS = 64 * 1024
_GIT_TIMEOUT_S = 15.0
_BINARY_SNIFF_BYTES = 8 * 1024

# One revision or a two-dot / three-dot range. First char alphanumeric so a
# ref can never be option-shaped (``--output=…``), and no whitespace so it
# can never smuggle a second argument.
_GIT_REF_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_./~^@{}-]{0,118}"
    r"(\.\.\.?[A-Za-z0-9][A-Za-z0-9_./~^@{}-]{0,118})?$"
)

_STATUS_FOR_KIND = {
    "bad_request": 400,
    "forbidden": 403,
    "not_found": 404,
    "fetch_failed": 502,
    "unavailable": 503,
}


class ReferenceRefused(ValueError):
    """A reference the daemon will not resolve — with the reason, said loudly.

    ``kind`` is the wire vocabulary (``bad_request`` / ``forbidden`` /
    ``not_found`` / ``fetch_failed`` / ``unavailable``); ``status`` is the
    HTTP twin the REST route answers with.
    """

    def __init__(self, kind: str, message: str) -> None:
        if kind not in _STATUS_FOR_KIND:
            raise ValueError(f"unknown refusal kind {kind!r}")
        super().__init__(message)
        self.kind = kind

    @property
    def status(self) -> int:
        return _STATUS_FOR_KIND[self.kind]


@dataclass(frozen=True)
class Reference:
    type: str
    target: str

    def describe(self) -> str:
        return f"@{self.type} {self.target}".rstrip()


def parse_references(raw: object) -> list[Reference]:
    """Validate the wire shape ``[{"type": "file"|"diff"|"url", "target": str}]``.

    Raises :class:`ReferenceRefused` (``bad_request``) on anything else. A
    ``diff`` may omit ``target`` (working tree vs index); ``file`` and ``url``
    must name something.
    """
    if not isinstance(raw, list):
        raise ReferenceRefused("bad_request", "references must be a list")
    if len(raw) > MAX_REFERENCES:
        raise ReferenceRefused(
            "bad_request", f"too many references ({len(raw)} > {MAX_REFERENCES})"
        )
    out: list[Reference] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ReferenceRefused("bad_request", f"references[{i}] must be an object")
        rtype = item.get("type")
        if rtype not in REFERENCE_TYPES:
            raise ReferenceRefused(
                "bad_request",
                f"references[{i}].type {rtype!r} — expected one of {', '.join(REFERENCE_TYPES)}",
            )
        target = item.get("target", "")
        if target is None:
            target = ""
        if not isinstance(target, str):
            raise ReferenceRefused("bad_request", f"references[{i}].target must be a string")
        target = target.strip()
        if rtype != "diff" and not target:
            raise ReferenceRefused("bad_request", f"references[{i}] ({rtype}) needs a target")
        out.append(Reference(type=rtype, target=target))
    return out


def default_files_root() -> Path:
    """The browse root ``/api/files`` serves — the no-workspace fallback scope.

    ``PROMETHEUS_FILES_ROOT`` widens/repoints the BROWSE root only (read-only,
    token-gated). Deliberately separate from ``PROMETHEUS_WORKSPACE_DIR``,
    which doubles as an image_generate WRITE root — repointing that to browse
    more would widen the agent's write surface too.
    """
    from prometheus.config.paths import get_workspace_dir

    env_root = os.environ.get("PROMETHEUS_FILES_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return get_workspace_dir().resolve()


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


class ReferenceResolver:
    """Turn parsed references into ``TextBlock``s, scoped to one session."""

    def __init__(
        self,
        *,
        security_cfg: dict | None,
        files_root: Callable[[], Path] = default_files_root,
        workspace_resolver: Callable[[str], Any] | None = None,
        fetch_url: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        # The gate's matcher, not a second one: config denied_paths + the
        # always-denied floor, normalised once, exactly as read_file sees them.
        from prometheus.permissions.checker import SecurityGate

        self._gate = SecurityGate(denied_paths=resolve_denied_paths(security_cfg))
        roots = resolve_workspace_root(security_cfg)
        self._configured_roots: tuple[Path, ...] = tuple(
            Path(r).expanduser().resolve()
            for r in ([roots] if isinstance(roots, str) else list(roots))
            if str(r).strip()
        )
        self._files_root = files_root
        self._workspace_resolver = workspace_resolver
        self._fetch_url = fetch_url

    # ------------------------------------------------------------------ scope
    def scope_for(self, session_id: str) -> tuple[Path, tuple[Path, ...]]:
        """``(cwd, allowed_roots)`` for a session.

        A bound workspace is the ONLY root; a resolver failure is surfaced,
        not collapsed into the fallback scope (a reference resolved against
        the wrong root is a plausible degraded state, which is the trap).
        """
        if self._workspace_resolver is not None:
            try:
                ws = self._workspace_resolver(session_id)
            except Exception as exc:  # noqa: BLE001 — reported, never swallowed
                log.warning("workspace lookup failed for session=%s: %s", session_id, exc)
                raise ReferenceRefused(
                    "unavailable", f"session workspace lookup failed: {exc}"
                ) from exc
            if ws:
                cwd = Path(str(ws)).expanduser().resolve()
                return cwd, (cwd,)
        cwd = self._files_root().resolve()
        return cwd, (cwd, *self._configured_roots)

    # ---------------------------------------------------------------- resolve
    async def resolve(self, session_id: str, refs: list[Reference]) -> list[TextBlock]:
        if not refs:
            return []
        cwd, roots = self.scope_for(session_id)
        blocks: list[TextBlock] = []
        for ref in refs:
            if ref.type == "file":
                blocks.append(await asyncio.to_thread(self.resolve_file, cwd, roots, ref.target))
            elif ref.type == "diff":
                blocks.append(await self.resolve_diff(cwd, ref.target))
            elif ref.type == "url":
                blocks.append(await self.resolve_url(ref.target))
            else:  # parse_references already refused this; belt and braces
                raise ReferenceRefused("bad_request", f"unknown reference type {ref.type!r}")
        return blocks

    # ------------------------------------------------------------------- file
    def resolve_file(self, cwd: Path, roots: tuple[Path, ...], target: str) -> TextBlock:
        raw = Path(target).expanduser()
        candidate = (raw if raw.is_absolute() else cwd / raw).resolve()
        if not any(_under(candidate, root) for root in roots):
            raise ReferenceRefused(
                "forbidden",
                f"@file {target}: outside the session scope ({cwd})",
            )
        decision = self._gate.evaluate("read_file", is_read_only=True, file_path=str(candidate))
        if not decision.allowed:
            raise ReferenceRefused("forbidden", f"@file {target}: {decision.reason}")
        if not candidate.exists():
            raise ReferenceRefused("not_found", f"@file {target}: no such file")
        if not candidate.is_file():
            raise ReferenceRefused("bad_request", f"@file {target}: not a regular file")
        try:
            size = candidate.stat().st_size
            with candidate.open("rb") as fh:
                data = fh.read(FILE_CAP_BYTES + 1)
        except OSError as exc:
            raise ReferenceRefused("not_found", f"@file {target}: {exc.strerror or exc}") from exc
        if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
            raise ReferenceRefused("bad_request", f"@file {target}: binary file")
        truncated = len(data) > FILE_CAP_BYTES
        text = data[:FILE_CAP_BYTES].decode("utf-8", errors="replace")
        try:
            shown = str(candidate.relative_to(cwd))
        except ValueError:
            shown = str(candidate)
        attrs = f'type="file" path="{_attr(shown)}" bytes="{size}"'
        if truncated:
            attrs += ' truncated="true"'
            text = text.rstrip() + f"\n...[truncated at {FILE_CAP_BYTES} bytes]"
        return TextBlock(text=f"\n\n<reference {attrs}>\n{text}\n</reference>")

    # ------------------------------------------------------------------- diff
    async def resolve_diff(self, cwd: Path, target: str) -> TextBlock:
        if target and not _GIT_REF_RE.match(target):
            raise ReferenceRefused("bad_request", f"@diff {target}: not a valid git revision or range")
        top = await self._git(cwd, ["rev-parse", "--show-toplevel"])
        if top.returncode != 0:
            raise ReferenceRefused("not_found", f"@diff: {cwd} is not inside a git repository")
        argv = ["diff", "--no-color", "--no-ext-diff"]
        if target:
            argv.append(target)
        argv.append("--")
        result = await self._git(cwd, argv)
        if result.returncode != 0:
            first = (result.stderr.decode("utf-8", errors="replace").strip().splitlines() or ["git diff failed"])[0]
            raise ReferenceRefused("bad_request", f"@diff {target}: {first}".rstrip())
        text = result.stdout.decode("utf-8", errors="replace")
        truncated = len(text) > DIFF_CAP_CHARS
        if truncated:
            text = text[:DIFF_CAP_CHARS].rstrip() + f"\n...[truncated at {DIFF_CAP_CHARS} chars]"
        if not text.strip():
            text = "(no changes)"
        attrs = f'type="diff" ref="{_attr(target or "worktree")}" cwd="{_attr(cwd)}"'
        if truncated:
            attrs += ' truncated="true"'
        return TextBlock(text=f"\n\n<reference {attrs}>\n{text}\n</reference>")

    @staticmethod
    async def _git(cwd: Path, argv: list[str]) -> subprocess.CompletedProcess:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat")

        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", str(cwd), *argv],
                capture_output=True,
                timeout=_GIT_TIMEOUT_S,
                env=env,
                check=False,
            )

        try:
            return await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as exc:
            raise ReferenceRefused("fetch_failed", f"@diff: git timed out after {_GIT_TIMEOUT_S:.0f}s") from exc
        except OSError as exc:
            raise ReferenceRefused("unavailable", f"@diff: cannot run git: {exc}") from exc

    # -------------------------------------------------------------------- url
    async def resolve_url(self, target: str) -> TextBlock:
        from urllib.parse import urlparse

        from prometheus.tools.builtin.web_fetch import _is_safe_url, fetch_url_text

        parsed = urlparse(target)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ReferenceRefused("bad_request", f"@url {target}: only http(s) URLs can be referenced")
        if not _is_safe_url(target):
            raise ReferenceRefused(
                "forbidden", f"@url {target}: resolves to a private or reserved address"
            )
        fetch = self._fetch_url or fetch_url_text
        try:
            page = await fetch(target, max_chars=URL_CAP_CHARS)
        except Exception as exc:  # noqa: BLE001 — httpx's hierarchy, surfaced as-is
            raise ReferenceRefused("fetch_failed", f"@url {target}: {exc}") from exc
        attrs = (
            f'type="url" url="{_attr(page.url)}" status="{page.status}" '
            f'content_type="{_attr(page.content_type or "(unknown)")}"'
        )
        if page.truncated:
            attrs += ' truncated="true"'
        return TextBlock(text=f"\n\n<reference {attrs}>\n{page.body}\n</reference>")
