"""LSP orchestrator — manages multiple language server clients.

Routes requests to the right client based on file extension, handles lazy
spawning, and tracks broken servers so they don't retry within a session.

Modeled after OpenCode's ``index.ts`` orchestration layer.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

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
)

log = logging.getLogger(__name__)


class LSPOrchestrator:
    """Manages multiple LSP clients with lazy spawning and failure tracking."""

    def __init__(self, custom_servers: dict[str, dict] | None = None) -> None:
        self._custom_servers = custom_servers or {}
        self._clients: dict[str, LSPClient] = {}   # key → active client
        self._broken: set[str] = set()              # keys that failed
        self._spawning: dict[str, asyncio.Task] = {}  # keys currently spawning

    def _key(self, server_def: LSPServerDef, project_root: Path) -> str:
        return f"{server_def.language_id}:{project_root}"

    # -- server lifecycle -----------------------------------------

    async def ensure_server(self, filepath: str | Path) -> LSPClient | None:
        """Return a running client for *filepath*, spawning lazily if needed.

        Returns ``None`` if the language is unsupported or the server is broken.
        """
        filepath = Path(filepath).resolve()
        server_def = get_server_for_file(filepath, self._custom_servers)
        if server_def is None:
            return None

        project_root = find_project_root(filepath, server_def.root_markers)
        key = self._key(server_def, project_root)

        # Already broken — don't retry
        if key in self._broken:
            return None

        # Already running
        if key in self._clients and self._clients[key].is_alive:
            return self._clients[key]

        # Spawn in-flight — await existing task (promise coalescing)
        if key in self._spawning:
            try:
                return await self._spawning[key]
            except Exception:
                return None

        # Spawn new server
        task = asyncio.create_task(self._spawn(server_def, project_root, key))
        self._spawning[key] = task
        try:
            return await task
        except Exception:
            return None
        finally:
            if self._spawning.get(key) is task:
                self._spawning.pop(key, None)

    async def _spawn(
        self, server_def: LSPServerDef, project_root: Path, key: str,
    ) -> LSPClient | None:
        """Spawn and initialize a language server. Marks broken on failure."""
        client = LSPClient(server_def, project_root)
        try:
            await client.start()
        except Exception as exc:
            log.warning(
                "LSP server failed to start (%s): %s", server_def.language_id, exc,
            )
            self._broken.add(key)
            try:
                await client.stop()
            except Exception:
                pass
            return None

        # Race check: another spawn may have completed first
        if key in self._clients and self._clients[key].is_alive:
            await client.stop()
            return self._clients[key]

        self._clients[key] = client
        return client

    # -- routed LSP methods ---------------------------------------

    async def get_definition(
        self, filepath: str, line: int, col: int,
    ) -> list[Location]:
        client = await self.ensure_server(filepath)
        if client is None:
            return []
        try:
            return await client.get_definition(filepath, line, col)
        except LSPError as exc:
            log.debug("LSP definition failed: %s", exc)
            return []

    async def get_references(
        self, filepath: str, line: int, col: int,
    ) -> list[Location]:
        client = await self.ensure_server(filepath)
        if client is None:
            return []
        try:
            return await client.get_references(filepath, line, col)
        except LSPError as exc:
            log.debug("LSP references failed: %s", exc)
            return []

    async def get_hover(
        self, filepath: str, line: int, col: int,
    ) -> HoverInfo | None:
        client = await self.ensure_server(filepath)
        if client is None:
            return None
        try:
            return await client.get_hover(filepath, line, col)
        except LSPError as exc:
            log.debug("LSP hover failed: %s", exc)
            return None

    async def get_diagnostics(self, filepath: str) -> list[Diagnostic]:
        client = await self.ensure_server(filepath)
        if client is None:
            return []
        try:
            return await client.get_diagnostics(filepath)
        except LSPError as exc:
            log.debug("LSP diagnostics failed: %s", exc)
            return []

    async def get_symbols(self, filepath: str) -> list[DocumentSymbol]:
        client = await self.ensure_server(filepath)
        if client is None:
            return []
        try:
            return await client.get_document_symbols(filepath)
        except LSPError as exc:
            log.debug("LSP symbols failed: %s", exc)
            return []

    async def rename(
        self, filepath: str, line: int, col: int, new_name: str,
    ) -> dict[str, list[dict]]:
        client = await self.ensure_server(filepath)
        if client is None:
            return {}
        try:
            return await client.rename_symbol(filepath, line, col, new_name)
        except LSPError as exc:
            log.debug("LSP rename failed: %s", exc)
            return {}

    async def get_symbol_context(
        self, filepath: str, line: int, col: int,
    ) -> str:
        """The power move — one call that packages definition + references + type info.

        Instead of the model making 3 separate tool calls, this returns everything
        in one formatted text block. Claude Code's symbolContext concept.
        """
        client = await self.ensure_server(filepath)
        if client is None:
            return "No language server available for this file."

        # Fan out all three requests concurrently
        definition_task = client.get_definition(filepath, line, col)
        references_task = client.get_references(filepath, line, col)
        hover_task = client.get_hover(filepath, line, col)

        results = await asyncio.gather(
            definition_task, references_task, hover_task,
            return_exceptions=True,
        )

        definitions: list[Location] = results[0] if not isinstance(results[0], Exception) else []
        references: list[Location] = results[1] if not isinstance(results[1], Exception) else []
        hover: HoverInfo | None = results[2] if not isinstance(results[2], Exception) else None

        # Build formatted output
        parts: list[str] = []

        if hover:
            parts.append(f"Type: {hover.contents}")

        if definitions:
            parts.append(f"Defined: {definitions[0]}")
            for d in definitions[1:]:
                parts.append(f"  also: {d}")

        if references:
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
