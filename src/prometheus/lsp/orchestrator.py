"""LSP orchestrator — manages multiple language server clients.

Routes requests to the right client based on file extension and spawns
servers lazily. When no server can answer for a file (none configured for
it, binary not installed, failed to start, or the request itself failed),
the routed methods raise :class:`LSPUnavailable` with the reason. They never
return an empty result for it: an empty result is the server saying "nothing
there", and "couldn't check" must not read the same.

A failed start is not permanent. A missing binary is retried as soon as it
appears on the daemon's PATH; any other start failure after a backoff.

Modeled after OpenCode's ``index.ts`` orchestration layer.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prometheus.lsp.client import (
    Diagnostic,
    DocumentSymbol,
    HoverInfo,
    Location,
    LSPClient,
    LSPError,
)
from prometheus.lsp.languages import (
    LSPServerDef,
    find_project_root,
    get_server_for_file,
    install_hint,
    merged_servers,
)

log = logging.getLogger(__name__)

# Backoff for a start that failed with the binary present: 30 s after the
# first failure, doubling per failed attempt, capped at 10 minutes.
_RETRY_FIRST_S = 30.0
_RETRY_MAX_S = 600.0


class LSPUnavailable(Exception):
    """No language server could answer for this file.

    The message is written for the model: what could not be checked, why,
    and what would fix it.
    """


@dataclass
class _StartFailure:
    at: float                           # clock() when the attempt failed
    attempts: int                       # failed attempts in a row
    error: str | None = None            # why start() failed
    missing_binary: str | None = None   # command[0] when it was not on PATH ("" = no command)

    def retry_after_s(self) -> float:
        return min(_RETRY_FIRST_S * 2 ** (self.attempts - 1), _RETRY_MAX_S)


class LSPOrchestrator:
    """Manages multiple LSP clients with lazy spawning and failure tracking."""

    def __init__(
        self,
        custom_servers: dict[str, dict] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._custom_servers = custom_servers or {}
        self._clients: dict[str, LSPClient] = {}   # key → active client
        self._failures: dict[str, _StartFailure] = {}  # key → last failed start
        self._spawning: dict[str, asyncio.Task] = {}  # keys currently spawning
        self._clock = clock

    def _key(self, server_def: LSPServerDef, project_root: Path) -> str:
        return f"{server_def.language_id}:{project_root}"

    # -- server lifecycle -----------------------------------------

    async def ensure_server(self, filepath: str | Path) -> LSPClient | None:
        """Return a running client for *filepath*, or ``None`` if there is none.

        For callers that treat "no server" as "nothing to do" (the post-edit
        diagnostics hook). Anything that answers the model uses
        :meth:`require_server`, which says why.
        """
        try:
            return await self.require_server(filepath)
        except LSPUnavailable:
            return None

    async def require_server(self, filepath: str | Path) -> LSPClient:
        """Return a running client for *filepath*, spawning lazily if needed.

        Raises :class:`LSPUnavailable` when there is none, with the reason.
        """
        filepath = Path(filepath).resolve()
        server_def = get_server_for_file(filepath, self._custom_servers)
        if server_def is None:
            raise LSPUnavailable(self._describe_unsupported(filepath))

        project_root = find_project_root(filepath, server_def.root_markers)
        key = self._key(server_def, project_root)

        # Already running
        if key in self._clients and self._clients[key].is_alive:
            return self._clients[key]

        failure = self._failures.get(key)
        if failure is not None and not self._retry_due(failure):
            raise LSPUnavailable(self._describe_failure(server_def, failure))

        # Spawn in-flight — await existing task (promise coalescing)
        if key in self._spawning:
            task, owned = self._spawning[key], False
        else:
            task = asyncio.create_task(self._spawn(server_def, project_root, key))
            self._spawning[key], owned = task, True
        try:
            client = await task
        except Exception:
            log.debug("LSP spawn raised (%s)", server_def.language_id, exc_info=True)
            client = None
        finally:
            if owned and self._spawning.get(key) is task:
                self._spawning.pop(key, None)

        if client is None:
            failure = self._failures.get(key)
            if failure is not None:
                raise LSPUnavailable(self._describe_failure(server_def, failure))
            raise LSPUnavailable(
                f"Couldn't check: the {server_def.language_id} language server did not start."
            )
        return client

    def _retry_due(self, failure: _StartFailure) -> bool:
        if failure.missing_binary is not None:
            # Installing the binary is the fix, so its appearance is the retry.
            return bool(failure.missing_binary) and shutil.which(failure.missing_binary) is not None
        return self._clock() - failure.at >= failure.retry_after_s()

    async def _spawn(
        self, server_def: LSPServerDef, project_root: Path, key: str,
    ) -> LSPClient | None:
        """Spawn and initialize a language server. Records why on failure."""
        binary = server_def.command[0] if server_def.command else ""
        if not binary or shutil.which(binary) is None:
            self._record_failure(key, server_def, missing_binary=binary)
            return None

        client = LSPClient(server_def, project_root)
        try:
            await client.start()
        except Exception as exc:
            self._record_failure(key, server_def, error=f"{type(exc).__name__}: {exc}")
            try:
                await client.stop()
            except Exception:
                pass
            return None

        previous = self._failures.pop(key, None)
        if previous is not None:
            log.info(
                "LSP server %s started after %d failed attempt(s) (root=%s)",
                server_def.language_id, previous.attempts, project_root,
            )

        # Race check: another spawn may have completed first
        if key in self._clients and self._clients[key].is_alive:
            await client.stop()
            return self._clients[key]

        self._clients[key] = client
        return client

    def _record_failure(
        self,
        key: str,
        server_def: LSPServerDef,
        *,
        error: str | None = None,
        missing_binary: str | None = None,
    ) -> None:
        previous = self._failures.get(key)
        failure = _StartFailure(
            at=self._clock(),
            attempts=(previous.attempts + 1) if previous else 1,
            error=error,
            missing_binary=missing_binary,
        )
        self._failures[key] = failure
        if missing_binary is not None:
            log.warning(
                "LSP server not installed (%s): %r is not on PATH; retried when it appears",
                server_def.language_id, missing_binary,
            )
        else:
            log.warning(
                "LSP server failed to start (%s, attempt %d): %s; retrying in %.0fs",
                server_def.language_id, failure.attempts, error, failure.retry_after_s(),
            )

    def _describe_failure(self, server_def: LSPServerDef, failure: _StartFailure) -> str:
        lang = server_def.language_id
        if failure.missing_binary == "":
            return (
                f"Couldn't check: the {lang} language server has no command configured "
                f"(lsp.servers.{lang}.command)."
            )
        if failure.missing_binary is not None:
            return (
                f"Couldn't check: the {lang} language server is not installed. "
                f"`{failure.missing_binary}` is not on the daemon's PATH. "
                f"{install_hint(server_def)}, into a directory on the daemon's PATH. "
                "The next lsp call picks it up; no restart needed."
            )
        wait = max(0.0, failure.retry_after_s() - (self._clock() - failure.at))
        return (
            f"Couldn't check: the {lang} language server (`{' '.join(server_def.command)}`) "
            f"failed to start: {failure.error}. It is retried in {wait:.0f}s "
            f"(failed attempts so far: {failure.attempts})."
        )

    def _describe_unsupported(self, filepath: Path) -> str:
        ext = filepath.suffix.lower()
        what = f"`{ext}` files" if ext else "files without an extension"
        covered = sorted({e for s in merged_servers(self._custom_servers).values() for e in s.extensions})
        return (
            f"Couldn't check: no language server is configured for {what}. "
            f"lsp covers {', '.join(covered)}; add others under lsp.servers in the config."
        )

    @staticmethod
    def _request_failed(client: LSPClient, what: str, exc: Exception) -> LSPUnavailable:
        return LSPUnavailable(
            f"Couldn't check: the {client.server_def.language_id} language server "
            f"failed the {what} request: {exc}"
        )

    # -- routed LSP methods ---------------------------------------
    # Each returns what the server answered, including an empty answer, and
    # raises LSPUnavailable when there was no answer to return.

    async def get_definition(
        self, filepath: str, line: int, col: int,
    ) -> list[Location]:
        client = await self.require_server(filepath)
        try:
            return await client.get_definition(filepath, line, col)
        except LSPError as exc:
            raise self._request_failed(client, "definition", exc) from exc

    async def get_references(
        self, filepath: str, line: int, col: int,
    ) -> list[Location]:
        client = await self.require_server(filepath)
        try:
            return await client.get_references(filepath, line, col)
        except LSPError as exc:
            raise self._request_failed(client, "references", exc) from exc

    async def get_hover(
        self, filepath: str, line: int, col: int,
    ) -> HoverInfo | None:
        client = await self.require_server(filepath)
        try:
            return await client.get_hover(filepath, line, col)
        except LSPError as exc:
            raise self._request_failed(client, "hover", exc) from exc

    async def get_diagnostics(
        self, filepath: str, *, wait_s: float | None = None,
    ) -> list[Diagnostic]:
        """Diagnostics the server published for *filepath*.

        With *wait_s*, the file is re-sent and the answer is the server's
        next publish for it. Without, it is the last publish. Either way, no
        publish at all raises: an empty list here always means "clean".
        """
        client = await self.require_server(filepath)
        try:
            diags = await client.get_diagnostics(filepath, wait_s=wait_s)
        except LSPError as exc:
            raise self._request_failed(client, "diagnostics", exc) from exc
        if diags is None:
            lang, name = client.server_def.language_id, Path(filepath).name
            if wait_s:
                raise LSPUnavailable(
                    f"Couldn't check: the {lang} language server published no diagnostics "
                    f"for {name} within {wait_s:.0f}s, so the file was not checked."
                )
            raise LSPUnavailable(
                f"Couldn't check: the {lang} language server has not published "
                f"diagnostics for {name}."
            )
        return diags

    async def get_symbols(self, filepath: str) -> list[DocumentSymbol]:
        client = await self.require_server(filepath)
        try:
            return await client.get_document_symbols(filepath)
        except LSPError as exc:
            raise self._request_failed(client, "symbols", exc) from exc

    async def rename(
        self, filepath: str, line: int, col: int, new_name: str,
    ) -> dict[str, list[dict]]:
        client = await self.require_server(filepath)
        try:
            return await client.rename_symbol(filepath, line, col, new_name)
        except LSPError as exc:
            raise self._request_failed(client, "rename", exc) from exc

    async def get_symbol_context(
        self, filepath: str, line: int, col: int,
    ) -> str:
        """The power move — one call that packages definition + references + type info.

        Instead of the model making 3 separate tool calls, this returns everything
        in one formatted text block. Claude Code's symbolContext concept.

        A part whose request failed says so ("couldn't check") instead of
        being left out or reported as none found.
        """
        client = await self.require_server(filepath)

        # Fan out all three requests concurrently
        results = await asyncio.gather(
            client.get_definition(filepath, line, col),
            client.get_references(filepath, line, col),
            client.get_hover(filepath, line, col),
            return_exceptions=True,
        )
        failed = {
            label: result
            for label, result in zip(("definition", "references", "hover"), results)
            if isinstance(result, BaseException)
        }
        if len(failed) == len(results):
            raise self._request_failed(client, "context", failed["definition"])

        definitions: list[Location] = [] if "definition" in failed else results[0]
        references: list[Location] = [] if "references" in failed else results[1]
        hover: HoverInfo | None = None if "hover" in failed else results[2]

        # Build formatted output
        parts: list[str] = []

        if "hover" in failed:
            parts.append(f"Type: couldn't check ({failed['hover']})")
        elif hover:
            parts.append(f"Type: {hover.contents}")

        if "definition" in failed:
            parts.append(f"Defined: couldn't check ({failed['definition']})")
        elif definitions:
            parts.append(f"Defined: {definitions[0]}")
            for d in definitions[1:]:
                parts.append(f"  also: {d}")

        if "references" in failed:
            parts.append(f"References: couldn't check ({failed['references']})")
        elif references:
            parts.append(f"References ({len(references)}):")
            for ref in references[:20]:  # cap display at 20
                parts.append(f"  - {ref}")
            if len(references) > 20:
                parts.append(f"  ... and {len(references) - 20} more")
        else:
            parts.append("References: none found")

        return "\n".join(parts) if parts else "No information available."

    # -- file change notification ---------------------------------

    async def notify_file_changed(self, filepath: str | Path) -> None:
        """Notify the relevant LSP server that a file changed on disk."""
        client = await self.ensure_server(filepath)
        if client is not None:
            await client.did_change(str(filepath))

    # -- shutdown -------------------------------------------------

    async def shutdown_all(self) -> None:
        """Stop all running language servers. Call on daemon shutdown."""
        tasks = [client.stop() for client in self._clients.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
        self._spawning.clear()
        log.info("All LSP servers shut down")
