"""LSP client — JSON-RPC 2.0 over stdin/stdout.

Manages a single connection to one running language server process.
Modeled after OpenCode's ``client.ts`` pattern.

Key design choices:
- Full document sync (no incremental) — simpler, good enough for agent edits
- Diagnostics cached from ``textDocument/publishDiagnostics`` notifications
- Async reader loop dispatches responses and notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.lsp.languages import EXTENSION_TO_LANGUAGE, LSPServerDef

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes for LSP responses
# ------------------------------------------------------------------

@dataclass
class Location:
    """A source location returned by definition/references."""
    path: str
    line: int  # 1-indexed
    col: int   # 1-indexed

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}"


@dataclass
class Diagnostic:
    """A compiler diagnostic (error/warning/info)."""
    path: str
    line: int   # 1-indexed
    col: int    # 1-indexed
    severity: int  # 1=Error 2=Warning 3=Info 4=Hint
    message: str
    source: str = ""

    @property
    def severity_str(self) -> str:
        return {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}.get(
            self.severity, "UNKNOWN"
        )

    def __str__(self) -> str:
        return f"{self.severity_str} L{self.line}: {self.message}"


@dataclass
class HoverInfo:
    """Hover information (type, docs)."""
    contents: str

    def __str__(self) -> str:
        return self.contents


@dataclass
class DocumentSymbol:
    """A symbol in a document (function, class, variable, etc.)."""
    name: str
    kind: int
    range_start_line: int  # 1-indexed
    range_end_line: int    # 1-indexed
    detail: str = ""
    children: list[DocumentSymbol] = field(default_factory=list)

    @property
    def kind_str(self) -> str:
        return _SYMBOL_KINDS.get(self.kind, "Unknown")

    def __str__(self) -> str:
        detail = f" ({self.detail})" if self.detail else ""
        return f"{self.kind_str} {self.name}{detail} L{self.range_start_line}-{self.range_end_line}"


_SYMBOL_KINDS: dict[int, str] = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package",
    5: "Class", 6: "Method", 7: "Property", 8: "Field",
    9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
    13: "Variable", 14: "Constant", 15: "String", 16: "Number",
    17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
    21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter",
}


class LSPError(Exception):
    """Error returned by the language server."""

    def __init__(self, error: dict) -> None:
        self.code = error.get("code", -1)
        self.data = error.get("data")
        super().__init__(error.get("message", "LSP error"))


# ------------------------------------------------------------------
# URI helpers
# ------------------------------------------------------------------

def _path_to_uri(path: Path | str) -> str:
    p = Path(path).resolve()
    return f"file://{p}"


def _uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[7:]
    return uri


# ------------------------------------------------------------------
# LSP Client
# ------------------------------------------------------------------

class LSPClient:
    """Manages a connection to one running language server."""

    def __init__(self, server_def: LSPServerDef, project_root: Path) -> None:
        self.server_def = server_def
        self.project_root = project_root
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._diagnostics: dict[str, list[Diagnostic]] = {}
        self._open_files: dict[str, int] = {}  # path → version
        self._reader_task: asyncio.Task | None = None
        self._initialized: bool = False
        self._server_capabilities: dict[str, Any] = {}

    @property
    def is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and self._initialized
        )

    # -- lifecycle ------------------------------------------------

    async def start(self) -> None:
        """Spawn the server process and perform the LSP initialize handshake."""
        self._process = await asyncio.create_subprocess_exec(
            *self.server_def.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.project_root),
        )
        self._reader_task = asyncio.create_task(self._reader_loop())

        root_uri = _path_to_uri(self.project_root)
        result = await self._send_request("initialize", {
            "processId": self._process.pid,
            "rootUri": root_uri,
            "workspaceFolders": [{"name": "workspace", "uri": root_uri}],
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didOpen": True, "didChange": True},
                    "publishDiagnostics": {"versionSupport": True},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "definition": {},
                    "references": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "rename": {"prepareSupport": False},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "configuration": True,
                },
            },
            "initializationOptions": self.server_def.initialization_options or {},
        })
        self._server_capabilities = (result or {}).get("capabilities", {})

        self._send_notification("initialized", {})

        # Send config if server definition provides initialization options
        if self.server_def.initialization_options:
            self._send_notification(
                "workspace/didChangeConfiguration",
                {"settings": self.server_def.initialization_options},
            )

        self._initialized = True
        log.info(
            "LSP server started: %s (root=%s)",
            self.server_def.language_id,
            self.project_root,
        )

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._process and self._initialized:
            try:
                await asyncio.wait_for(
                    self._send_request("shutdown", None), timeout=5,
                )
                self._send_notification("exit", None)
            except Exception:
                log.debug("LSP shutdown handshake failed", exc_info=True)

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.kill()
                await self._process.wait()
            except Exception:
                pass

        self._initialized = False
        log.info("LSP server stopped: %s", self.server_def.language_id)

    # -- document notifications -----------------------------------

    async def did_open(self, filepath: str | Path) -> None:
        """Notify the server a document was opened (full content sync)."""
        path = str(Path(filepath).resolve())
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        uri = _path_to_uri(path)
        ext = Path(path).suffix.lower()
        lang_id = EXTENSION_TO_LANGUAGE.get(ext, "plaintext")

        if path in self._open_files:
            # Already open — send didChange with new content
            self._open_files[path] += 1
            self._send_notification("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": self._open_files[path]},
                "contentChanges": [{"text": content}],
            })
        else:
            self._open_files[path] = 0
            self._send_notification("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": lang_id,
                    "version": 0,
                    "text": content,
                },
            })

    async def did_change(self, filepath: str | Path) -> None:
        """Notify the server a document changed on disk."""
        path = str(Path(filepath).resolve())
        if path not in self._open_files:
            await self.did_open(filepath)
            return
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        self._open_files[path] += 1
        self._send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": _path_to_uri(path),
                "version": self._open_files[path],
            },
            "contentChanges": [{"text": content}],
        })

    async def did_close(self, filepath: str | Path) -> None:
        """Notify the server a document was closed."""
        path = str(Path(filepath).resolve())
        self._open_files.pop(path, None)
        self._send_notification("textDocument/didClose", {
            "textDocument": {"uri": _path_to_uri(path)},
        })

    # -- protocol methods -----------------------------------------

    async def get_definition(
        self, filepath: str, line: int, col: int,
    ) -> list[Location]:
        """Get definition location(s). Line/col are 1-indexed."""
        await self.did_open(filepath)
        result = await self._send_request("textDocument/definition", {
            "textDocument": {"uri": _path_to_uri(filepath)},
            "position": {"line": line - 1, "character": col - 1},
        })
        return _parse_locations(result)

    async def get_references(
        self, filepath: str, line: int, col: int,
    ) -> list[Location]:
        """Get all references. Line/col are 1-indexed."""
        await self.did_open(filepath)
        result = await self._send_request("textDocument/references", {
            "textDocument": {"uri": _path_to_uri(filepath)},
            "position": {"line": line - 1, "character": col - 1},
            "context": {"includeDeclaration": True},
        })
        return _parse_locations(result)

    async def get_hover(
        self, filepath: str, line: int, col: int,
    ) -> HoverInfo | None:
        """Get hover information (type, docs). Line/col are 1-indexed."""
        await self.did_open(filepath)
        result = await self._send_request("textDocument/hover", {
            "textDocument": {"uri": _path_to_uri(filepath)},
            "position": {"line": line - 1, "character": col - 1},
        })
        if not result or "contents" not in result:
            return None
        contents = result["contents"]
        if isinstance(contents, str):
            return HoverInfo(contents=contents)
        if isinstance(contents, dict):
            return HoverInfo(contents=contents.get("value", str(contents)))
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", ""))
            return HoverInfo(contents="\n".join(parts))
        return HoverInfo(contents=str(contents))

    async def get_diagnostics(self, filepath: str) -> list[Diagnostic]:
        """Return cached diagnostics for a file (from publishDiagnostics)."""
        path = str(Path(filepath).resolve())
        return list(self._diagnostics.get(path, []))

    async def get_document_symbols(self, filepath: str) -> list[DocumentSymbol]:
        """Get document symbols (outline)."""
        await self.did_open(filepath)
        result = await self._send_request("textDocument/documentSymbol", {
            "textDocument": {"uri": _path_to_uri(filepath)},
        })
        if not result:
            return []
        return _parse_symbols(result)

    async def rename_symbol(
        self, filepath: str, line: int, col: int, new_name: str,
    ) -> dict[str, list[dict]]:
        """Rename a symbol across the workspace. Returns a workspace edit.

        Line/col are 1-indexed. Returns ``{path: [{range, newText}]}``.
        """
        await self.did_open(filepath)
        result = await self._send_request("textDocument/rename", {
            "textDocument": {"uri": _path_to_uri(filepath)},
            "position": {"line": line - 1, "character": col - 1},
            "newName": new_name,
        })
        if not result:
            return {}
        return _parse_workspace_edit(result)

    # -- JSON-RPC transport ---------------------------------------

    async def _send_request(self, method: str, params: Any) -> Any:
        """Send a request and wait for the response."""
        if not self._process or self._process.returncode is not None:
            raise LSPError({"message": "Server process not running"})
        self._request_id += 1
        rid = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[rid] = future

        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._write_message(msg)

        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise LSPError({"message": f"Request timed out: {method}"})

    def _send_notification(self, method: str, params: Any) -> None:
        """Send a notification (no response expected)."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write_message(msg)

    def _write_message(self, msg: dict) -> None:
        """Write a JSON-RPC message with Content-Length framing."""
        if not self._process or not self._process.stdin:
            return
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)

    async def _reader_loop(self) -> None:
        """Read and dispatch messages from the server's stdout."""
        assert self._process and self._process.stdout
        reader = self._process.stdout
        try:
            while True:
                # Read headers until blank line
                content_length = 0
                while True:
                    line = await reader.readline()
                    if not line:
                        return  # EOF — server died
                    decoded = line.decode("ascii", errors="replace").strip()
                    if not decoded:
                        break
                    if decoded.lower().startswith("content-length:"):
                        content_length = int(decoded.split(":", 1)[1].strip())

                if content_length == 0:
                    continue

                body = await reader.readexactly(content_length)
                try:
                    msg = json.loads(body)
                except json.JSONDecodeError:
                    log.warning("LSP: invalid JSON from server")
                    continue

                if "id" in msg and "method" not in msg:
                    # Response to our request
                    rid = msg["id"]
                    future = self._pending.pop(rid, None)
                    if future and not future.done():
                        if "error" in msg:
                            future.set_exception(LSPError(msg["error"]))
                        else:
                            future.set_result(msg.get("result"))
                elif "method" in msg and "id" not in msg:
                    # Server notification
                    self._handle_notification(msg["method"], msg.get("params", {}))
                elif "method" in msg and "id" in msg:
                    # Server-to-client request — acknowledge with null
                    self._write_message({
                        "jsonrpc": "2.0", "id": msg["id"], "result": None,
                    })
        except asyncio.CancelledError:
            return
        except asyncio.IncompleteReadError:
            log.debug("LSP reader: server closed stdout")
        except Exception:
            log.debug("LSP reader loop error", exc_info=True)
        finally:
            # Cancel all pending futures
            for future in self._pending.values():
                if not future.done():
                    future.cancel()
            self._pending.clear()

    def _handle_notification(self, method: str, params: dict) -> None:
        """Handle server-to-client notifications."""
        if method == "textDocument/publishDiagnostics":
            uri = params.get("uri", "")
            path = _uri_to_path(uri)
            self._diagnostics[path] = [
                Diagnostic(
                    path=path,
                    line=d["range"]["start"]["line"] + 1,
                    col=d["range"]["start"]["character"] + 1,
                    severity=d.get("severity", 1),
                    message=d.get("message", ""),
                    source=d.get("source", ""),
                )
                for d in params.get("diagnostics", [])
            ]


# ------------------------------------------------------------------
# Response parsers
# ------------------------------------------------------------------

def _parse_locations(result: Any) -> list[Location]:
    """Parse definition/references response into Location list."""
    if result is None:
        return []
    if isinstance(result, dict):
        result = [result]
    locations: list[Location] = []
    for item in result:
        if "targetUri" in item:
            # LocationLink
            uri = item["targetUri"]
            pos = item.get("targetSelectionRange", item.get("targetRange", {})).get("start", {})
        elif "uri" in item:
            # Location
            uri = item["uri"]
            pos = item.get("range", {}).get("start", {})
        else:
            continue
        locations.append(Location(
            path=_uri_to_path(uri),
            line=pos.get("line", 0) + 1,
            col=pos.get("character", 0) + 1,
        ))
    return locations


def _parse_symbols(result: list) -> list[DocumentSymbol]:
    """Parse documentSymbol response."""
    symbols: list[DocumentSymbol] = []
    for item in result:
        if "range" in item:
            # Hierarchical DocumentSymbol
            children = _parse_symbols(item.get("children", []))
            symbols.append(DocumentSymbol(
                name=item["name"],
                kind=item.get("kind", 0),
                range_start_line=item["range"]["start"]["line"] + 1,
                range_end_line=item["range"]["end"]["line"] + 1,
                detail=item.get("detail", ""),
                children=children,
            ))
        elif "location" in item:
            # Flat SymbolInformation
            loc = item["location"]
            rng = loc.get("range", {})
            symbols.append(DocumentSymbol(
                name=item["name"],
                kind=item.get("kind", 0),
                range_start_line=rng.get("start", {}).get("line", 0) + 1,
                range_end_line=rng.get("end", {}).get("line", 0) + 1,
                detail=item.get("containerName", ""),
            ))
    return symbols


def _parse_workspace_edit(result: dict) -> dict[str, list[dict]]:
    """Parse workspace/edit into ``{path: [{range, newText}]}``."""
    edits_by_file: dict[str, list[dict]] = {}
    changes = result.get("changes", {})
    for uri, edits in changes.items():
        path = _uri_to_path(uri)
        edits_by_file[path] = [
            {
                "start_line": e["range"]["start"]["line"] + 1,
                "end_line": e["range"]["end"]["line"] + 1,
                "newText": e.get("newText", ""),
            }
            for e in edits
        ]

    # documentChanges format
    for doc_change in result.get("documentChanges", []):
        if "textDocument" not in doc_change:
            continue
        path = _uri_to_path(doc_change["textDocument"]["uri"])
        edits_by_file[path] = [
            {
                "start_line": e["range"]["start"]["line"] + 1,
                "end_line": e["range"]["end"]["line"] + 1,
                "newText": e.get("newText", ""),
            }
            for e in doc_change.get("edits", [])
        ]
    return edits_by_file
