"""JSON-RPC 2.0 framing for ACP v2 — newline-delimited JSON, no batching.

One frame per line. v2 removed JSON-RPC batching (the pinned schema's top-level
branches are all single objects), so a JSON array on the wire is not an ACP
message and is refused as Invalid Request — the TCK judges the same shape the
same way.

This module is transport-agnostic: it parses and renders frames, and nothing
else. It never touches sessions, permissions or the daemon loop.

Source: novel code for Prometheus, 2026-11-25.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: The pinned schema's RequestId: string or integer, never null, never fractional.
RequestId = str | int

#: JSON-RPC 2.0 / ACP error codes used by this layer (schema $defs/ErrorCode).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
REQUEST_CANCELLED = -32800
UNSUPPORTED_PROTOCOL_VERSION = -32022

#: The most one frame may weigh. A prompt carries text and content blocks, not
#: files; anything larger is a malformed or hostile peer, refused before parse.
MAX_FRAME_BYTES = 4 * 1024 * 1024


class FrameError(Exception):
    """A line that is not a legal ACP v2 frame. ``code`` is the JSON-RPC error
    to answer with; ``request_id`` is the id to answer under when one could be
    read (None otherwise — the error response then carries ``id: null``, the
    one place v2 allows it)."""

    def __init__(self, code: int, message: str, request_id: RequestId | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


def _valid_id(value: Any) -> bool:
    # bool is an int subclass; JSON-RPC ids are strings or integers, not booleans.
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


@dataclass(frozen=True)
class Request:
    id: RequestId
    method: str
    params: dict[str, Any] | None


@dataclass(frozen=True)
class Notification:
    method: str
    params: dict[str, Any] | None


@dataclass(frozen=True)
class Response:
    id: RequestId | None
    result: Any = None
    error: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


Frame = Request | Notification | Response


def parse_frame(line: str | bytes) -> Frame:
    """Parse ONE newline-delimited frame. Raises FrameError with the right
    JSON-RPC code for every malformed shape; never returns a partial frame."""
    raw = line.encode("utf-8") if isinstance(line, str) else line
    if len(raw) > MAX_FRAME_BYTES:
        raise FrameError(INVALID_REQUEST, "frame exceeds the size cap")
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FrameError(PARSE_ERROR, f"invalid JSON: {exc}") from exc
    # v2 has no batching: an array is one Invalid Request, not N messages.
    if not isinstance(msg, dict):
        raise FrameError(INVALID_REQUEST, "a frame must be a single JSON object (v2 has no batching)")
    if msg.get("jsonrpc") != "2.0":
        raise FrameError(INVALID_REQUEST, 'missing or wrong "jsonrpc" (must be "2.0")',
                         msg.get("id") if _valid_id(msg.get("id")) else None)
    if "method" in msg:
        method = msg["method"]
        if not isinstance(method, str) or not method:
            raise FrameError(INVALID_REQUEST, '"method" must be a non-empty string')
        params = msg.get("params")
        if params is not None and not isinstance(params, dict):
            raise FrameError(INVALID_REQUEST, '"params" must be an object when present',
                             msg.get("id") if _valid_id(msg.get("id")) else None)
        if "id" in msg:
            if not _valid_id(msg["id"]):
                raise FrameError(INVALID_REQUEST, '"id" must be a string or integer, never null')
            return Request(id=msg["id"], method=method, params=params)
        return Notification(method=method, params=params)
    # A response: id required (null only on error), exactly one of result/error.
    if "id" not in msg:
        raise FrameError(INVALID_REQUEST, 'frame has neither "method" nor "id"')
    rid = msg["id"]
    has_result, has_error = "result" in msg, "error" in msg
    if has_result == has_error:
        raise FrameError(INVALID_REQUEST, 'a response carries exactly one of "result" and "error"',
                         rid if _valid_id(rid) else None)
    if has_error:
        err = msg["error"]
        if not isinstance(err, dict) or not isinstance(err.get("code"), int) \
                or not isinstance(err.get("message"), str):
            raise FrameError(INVALID_REQUEST, '"error" must be {code: int, message: str}',
                             rid if _valid_id(rid) else None)
        if rid is not None and not _valid_id(rid):
            raise FrameError(INVALID_REQUEST, 'error response "id" must be a string, integer or null')
        return Response(id=rid, error=err)
    if not _valid_id(rid):
        raise FrameError(INVALID_REQUEST, 'result response "id" must be a string or integer, never null')
    return Response(id=rid, result=msg["result"])


def render(frame: Frame) -> str:
    """One frame as one line (no trailing newline; the transport adds it)."""
    if isinstance(frame, Request):
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": frame.id, "method": frame.method}
        if frame.params is not None:
            msg["params"] = frame.params
    elif isinstance(frame, Notification):
        msg = {"jsonrpc": "2.0", "method": frame.method}
        if frame.params is not None:
            msg["params"] = frame.params
    elif frame.is_error:
        msg = {"jsonrpc": "2.0", "id": frame.id, "error": frame.error}
    else:
        msg = {"jsonrpc": "2.0", "id": frame.id, "result": frame.result}
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)


def error_response(request_id: RequestId | None, code: int, message: str,
                   data: Any = None) -> Response:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return Response(id=request_id, error=err)