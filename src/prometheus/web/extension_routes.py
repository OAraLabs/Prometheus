"""Extension discovery, status proxy, two-step control proxy, and consent-gated trace export.

All routes registered here are classified ``OPERATOR_ONLY`` in
``prometheus.web.route_access``: ordinary scoped devices (phones paired for
chat only) receive ``403 operator_only`` automatically from the bearer
middleware.

Never in an answer: an extension's backend URL, bearer token, or env var name.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from prometheus.extensions.registry import (
    ExtensionConfigError,
    ExtensionRegistry,
    ExtensionUnavailable,
    load_extension_registry,
)
from prometheus.telemetry.tracker import telemetry_enabled

_NO_STORE = {"Cache-Control": "no-store"}
TRACE_EXPORT_MAX_LIMIT = 500


def _err(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": code, "detail": message},
        headers=_NO_STORE,
    )


def _export_local_traces(
    db_path: Path | None,
    *,
    limit: int,
    since_id: int,
    include_text: bool,
) -> dict[str, Any]:
    """Read recent request/tool-call telemetry metadata from Prometheus's local SQLite DB.

    Defaults to counts/metadata only (``include_text=False``) so no raw prompt
    or trace text leaves the host unless explicitly requested by an operator on
    loopback.
    """
    if db_path is None or not db_path.is_file():
        return {"rows": [], "count": 0, "next_since_id": since_id, "available": False}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "responses" not in tables:
            return {"rows": [], "count": 0, "next_since_id": since_id, "available": True}

        cols = {r[1] for r in conn.execute("PRAGMA table_info(responses)").fetchall()}
        select_cols = [
            c
            for c in (
                "id",
                "ts",
                "session_id",
                "model",
                "provider",
                "tool_set_hash",
                "tool_ok",
                "turn_outcome",
                "input_tokens",
                "output_tokens",
                "latency_ms",
            )
            if c in cols
        ]
        if include_text:
            for c in ("prompt_json", "response_json"):
                if c in cols:
                    select_cols.append(c)

        if not select_cols or "id" not in select_cols:
            return {"rows": [], "count": 0, "next_since_id": since_id, "available": True}

        sql = (
            f"SELECT {', '.join(select_cols)} FROM responses "
            "WHERE id > ? ORDER BY id ASC LIMIT ?"
        )
        raw_rows = conn.execute(sql, (since_id, limit)).fetchall()
        out_rows = [dict(r) for r in raw_rows]
        next_id = int(out_rows[-1]["id"]) if out_rows else since_id
        return {
            "rows": out_rows,
            "count": len(out_rows),
            "next_since_id": next_id,
            "available": True,
        }
    finally:
        conn.close()


def register_extension_routes(
    app: FastAPI,
    *,
    config: dict[str, Any] | None = None,
    registry: ExtensionRegistry | None = None,
    telemetry_db_path: Callable[[], Path | None] | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> ExtensionRegistry:
    """Register the five ``OPERATOR_ONLY`` extension hook routes on *app*."""
    cfg = config or {}
    if registry is None:
        try:
            reg = load_extension_registry(cfg)
        except ExtensionConfigError:
            reg = ExtensionRegistry()
    else:
        reg = registry

    def _get_client() -> Any:
        return client_factory() if client_factory is not None else None

    @app.get("/api/extensions")
    async def list_extensions() -> JSONResponse:
        items: list[dict[str, Any]] = []
        cli = _get_client()
        for name in reg.names():
            entry = reg.get(name)
            if entry is None:
                continue
            if not entry.enabled:
                items.append(entry.public_summary(healthy=False))
                continue
            items.append(await reg.probe(entry, client=cli))
        return JSONResponse(
            content={"extensions": items, "count": len(items)},
            headers=_NO_STORE,
        )

    @app.get("/api/extensions/traces/export")
    async def export_extension_traces(
        request: Request,
        limit: int = Query(default=100, ge=1, le=TRACE_EXPORT_MAX_LIMIT),
        since_id: int = Query(default=0, ge=0),
        include_text: bool = Query(default=False),
    ) -> JSONResponse:
        if not telemetry_enabled(cfg):
            return _err(403, "telemetry_disabled", "trace export refused: infrastructure.telemetry_enabled is false")
        if include_text:
            host = request.client.host if request.client else ""
            if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
                return _err(403, "loopback_only", "include_text=true is restricted to loopback callers")
        db_p = telemetry_db_path() if telemetry_db_path is not None else None
        payload = _export_local_traces(
            db_p,
            limit=limit,
            since_id=since_id,
            include_text=include_text,
        )
        return JSONResponse(content=payload, headers=_NO_STORE)

    @app.get("/api/extensions/{name}/status")
    async def extension_status(name: str) -> JSONResponse:
        entry = reg.get(name)
        if entry is None or not entry.enabled:
            return _err(404, "extension_not_found", f"unknown or disabled extension {name!r}")
        try:
            data = await reg.fetch_status(entry, client=_get_client())
        except ExtensionUnavailable as exc:
            return _err(exc.status_code, exc.code, str(exc))
        return JSONResponse(content=data, headers=_NO_STORE)

    @app.post("/api/extensions/{name}/control/intent")
    async def extension_control_intent(name: str, request: Request) -> JSONResponse:
        entry = reg.get(name)
        if entry is None or not entry.enabled:
            return _err(404, "extension_not_found", f"unknown or disabled extension {name!r}")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return _err(400, "invalid_body", "JSON object required")
        except Exception:
            return _err(400, "invalid_body", "valid JSON body required")
        try:
            status_code, resp = await reg.post_control(entry, "intent", body, client=_get_client())
        except ExtensionUnavailable as exc:
            return _err(exc.status_code, exc.code, str(exc))
        return JSONResponse(status_code=status_code, content=resp, headers=_NO_STORE)

    @app.post("/api/extensions/{name}/control/confirm")
    async def extension_control_confirm(name: str, request: Request) -> JSONResponse:
        entry = reg.get(name)
        if entry is None or not entry.enabled:
            return _err(404, "extension_not_found", f"unknown or disabled extension {name!r}")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return _err(400, "invalid_body", "JSON object required")
        except Exception:
            return _err(400, "invalid_body", "valid JSON body required")
        if not body.get("intent_id") or not body.get("plan_sha256"):
            return _err(400, "missing_confirm_fields", "intent_id and plan_sha256 are required")
        try:
            status_code, resp = await reg.post_control(entry, "confirm", body, client=_get_client())
        except ExtensionUnavailable as exc:
            return _err(exc.status_code, exc.code, str(exc))
        return JSONResponse(status_code=status_code, content=resp, headers=_NO_STORE)

    return reg
