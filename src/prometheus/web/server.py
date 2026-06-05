"""FastAPI server for Mission Control.

Serves REST endpoints on port 8005 and mounts static files for the UI.
Run alongside the main Prometheus process, not as a replacement.

Usage:
    from prometheus.web.server import create_app, start_web
    app = create_app(config, signal_bus, session_mgr, telemetry, ...)
    await start_web(app, host="0.0.0.0", port=8005)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


def create_app(
    config: dict[str, Any],
    signal_bus: Any | None = None,
    session_mgr: Any | None = None,
    telemetry: Any | None = None,
    skill_registry: Any | None = None,
    profile_store: Any | None = None,
    lcm_engine: Any | None = None,
    agent_loop: Any | None = None,
    approval_queue: Any | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI application with all routes."""

    _api_token = config.get("web", {}).get("api_token") or os.environ.get("PROMETHEUS_API_TOKEN", "")
    app = FastAPI(
        title="Prometheus Mission Control",
        version="0.1.0",
        docs_url=None if _api_token else "/docs",
        redoc_url=None if _api_token else "/redoc",
        openapi_url=None if _api_token else "/openapi.json",
    )
    _start_time = time.time()

    # CORS for dev (next dev on different port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Bearer token auth on /api/* routes ──────────────────────────

    @app.middleware("http")
    async def _check_bearer_token(request: Request, call_next):
        if _api_token and request.url.path.startswith("/api/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {_api_token}":
                return JSONResponse(status_code=401, content={"error": "unauthorized — set Authorization: Bearer <token>"})
        return await call_next(request)

    # Store references for route handlers
    app.state.config = config
    app.state.signal_bus = signal_bus
    app.state.session_mgr = session_mgr
    app.state.telemetry = telemetry
    app.state.skill_registry = skill_registry
    app.state.profile_store = profile_store
    app.state.lcm_engine = lcm_engine
    app.state.agent_loop = agent_loop
    app.state.approval_queue = approval_queue
    app.state.start_time = _start_time
    app.state.agent_state = "idle"
    app.state.current_model = config.get("model", {}).get("model", "unknown")
    app.state.current_provider = config.get("model", {}).get("provider", "unknown")
    app.state.active_profile = config.get("profiles", {}).get("default", "full")

    # ── Root ────────────────────────────────────────────────────────

    @app.get("/")
    async def root():
        return {
            "name": "Prometheus Mission Control",
            "version": "0.1.0",
            "endpoints": [
                "/api/status", "/api/sessions", "/api/telemetry",
                "/api/cron", "/api/approvals", "/api/chat",
                "/api/config", "/api/skills", "/api/profiles",
                "/api/wiki/stats", "/api/sentinel", "/api/events/recent",
            ],
        }

    # ── Health (unauthenticated; for external monitors) ─────────────
    # Lives outside /api/ so the bearer-token middleware never blocks it.
    # Fixes the 404 flood from the host polling GET /health (audit #6).
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "prometheus",
            "uptime_seconds": time.time() - app.state.start_time,
        }

    # ── Status ──────────────────────────────────────────────────────

    @app.get("/api/status")
    async def get_status():
        return {
            "state": app.state.agent_state,
            "model": app.state.current_model,
            "provider": app.state.current_provider,
            "profile": app.state.active_profile,
            "uptime_seconds": time.time() - app.state.start_time,
        }

    # ── Sessions ────────────────────────────────────────────────────

    @app.get("/api/sessions")
    async def list_sessions():
        if not session_mgr:
            return []
        lcm = getattr(app.state, "lcm_engine", None)
        store = lcm.conversation_store if lcm is not None else None
        sessions = []
        for sid, session in session_mgr._sessions.items():
            # gateway = the id prefix. The daemon assigns session ids as `<gateway>:<chat_id>`
            # (telegram:123, desktop:smoke), so this is authoritative — the client no longer
            # has to infer it from the prefix itself.
            colon = sid.find(":")
            gateway = sid[:colon] if colon > 0 else "unknown"
            # watermark = the SAME durable max-rowid cursor GET /api/sessions/{id}/messages
            # returns (reuses max_rowid). last_active = max message timestamp (sort/display).
            # Both fall back to 0 only if LCM isn't wired; any real store error propagates (500).
            sessions.append({
                "session_id": sid,
                "gateway": gateway,
                "created_at": session.created_at,
                "last_active": store.max_timestamp(sid) if store is not None else 0.0,
                "message_count": len(session.messages),
                "watermark": store.max_rowid(sid) if store is not None else 0,
            })
        return sessions

    @app.post("/api/sessions")
    async def create_session():
        import uuid
        sid = str(uuid.uuid4())
        if session_mgr:
            session_mgr.get_or_create(sid)
        return {"session_id": sid}

    @app.post("/api/chat/send")
    async def send_chat(request: Request):
        """Dispatch a user message into a Prometheus session.

        Body: ``{ "session_id": str, "message": str, "idempotency_key"?: str }``
        Returns: ``{ "run_id": str, "status": "sent" }``

        This is the HTTP entry point Beacon's chat surface needs (its tRPC
        ``sessions.send`` router POSTs here first, then falls back to the
        WebSocket bridge). The actual agent work is kicked off as a
        background task on the shared WebSocketBridge — the response streams
        back over the bridge's broadcast so any connected WS client sees
        deltas. The 200 returns as soon as the dispatch is queued.

        Failure modes (fail loud, no silent fallbacks):
          * 400 when ``session_id`` or ``message`` is missing/empty
          * 503 when the daemon hasn't wired a ``ws_bridge`` (e.g. ``web``
            disabled but route still mounted) — explicit so the caller
            doesn't think a queued message succeeded
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid JSON body"},
            )

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "body must be a JSON object"},
            )

        session_id = (body.get("session_id") or "").strip()
        message = (body.get("message") or "").strip()
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={"error": "session_id is required"},
            )
        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "message is required and must be non-empty"},
            )

        bridge = getattr(app.state, "ws_bridge", None)
        if bridge is None or not hasattr(bridge, "dispatch_user_message"):
            return JSONResponse(
                status_code=503,
                content={
                    "error": "chat dispatch unavailable — ws_bridge not wired",
                },
            )

        import uuid as _uuid

        idempotency_key = (body.get("idempotency_key") or "").strip() or _uuid.uuid4().hex
        client_msg_id = (body.get("client_msg_id") or "").strip() or None
        await bridge.dispatch_user_message(session_id, message, client_msg_id=client_msg_id)
        return {"run_id": idempotency_key, "status": "sent"}

    @app.get("/api/sessions/{session_id}/messages")
    async def get_messages(session_id: str, since: str | None = None):
        """Durable conversation history from the LCM store.

        Response: ``{ "messages": [...], "watermark": <int> }``. Each message is
        ``{message_id: <int>, ordinal: <int>, session_id, role, content, timestamp}``.

        * ``message_id`` is the durable LCM rowid — **monotonic, unique, restart-stable**
          (the store is append-only). It is BOTH the canonical identity AND the cursor.
        * ``ordinal`` is ``turn_index`` (the in-memory list position) — an explicitly
          NON-UNIQUE display position that repeats across restart/trim. Do not key on it.
        * ``timestamp`` is display-only — a whole turn can share one timestamp; order by
          ``message_id`` instead.
        * top-level ``watermark`` is the session's current max ``message_id`` (so a client
          knows it is caught up even when an incremental read is empty).

        ``?since=<message_id>`` returns only messages with ``rowid > since`` (incremental
        sync). FAIL LOUD: an unparseable ``since`` is a 400 — never silently ignored.
        """
        lcm = getattr(app.state, "lcm_engine", None)
        if lcm is None:
            return JSONResponse(
                status_code=503,
                content={"error": "conversation store unavailable — lcm_engine not wired"},
            )
        store = lcm.conversation_store

        since_id = 0
        if since is not None:
            try:
                since_id = int(since)
            except (TypeError, ValueError):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"invalid 'since' cursor {since!r}: expected an integer message_id"},
                )

        parts = store.messages_after_id(since_id, session_id=session_id, include_compacted=True)
        messages = [
            {
                "message_id": p.row_id,
                "ordinal": p.turn_index,
                "session_id": session_id,
                "role": p.role,
                "content": p.content,
                "content_json": p.content_json,
                "timestamp": p.timestamp,
            }
            for p in parts
        ]
        return {"messages": messages, "watermark": store.max_rowid(session_id)}

    @app.delete("/api/sessions/{session_id}")
    async def clear_session(session_id: str):
        if session_mgr:
            session_mgr.clear(session_id)
        return {"ok": True}

    # ── Telemetry ───────────────────────────────────────────────────

    @app.get("/api/telemetry")
    async def get_telemetry():
        if not telemetry:
            return {"total_calls": 0, "overall_success_rate": 0, "tools": {}}
        return telemetry.report()

    # ── Config (sanitized) ──────────────────────────────────────────

    @app.get("/api/config")
    async def get_config():
        import yaml
        sanitized = _sanitize_config(config)
        return {
            "raw_yaml": yaml.dump(sanitized, default_flow_style=False, sort_keys=False),
            "parsed": sanitized,
        }

    # ── Skills ──────────────────────────────────────────────────────

    @app.get("/api/skills")
    async def get_skills():
        if not skill_registry:
            return []
        return [
            {"name": s.name, "description": s.description, "source": s.source}
            for s in skill_registry.list_skills()
        ]

    # ── Profiles ────────────────────────────────────────────────────

    @app.get("/api/profiles")
    async def get_profiles():
        if not profile_store:
            return []
        return [
            {
                "name": p.name,
                "description": p.description,
                "is_active": p.name == app.state.active_profile,
            }
            for p in profile_store.list_profiles()
        ]

    @app.put("/api/profiles/active")
    async def set_active_profile(body: dict):
        name = body.get("name", "")
        if profile_store and profile_store.get(name):
            app.state.active_profile = name
            return {"ok": True, "profile": name}
        return JSONResponse(status_code=404, content={"error": "profile not found"})

    # ── Wiki Stats ──────────────────────────────────────────────────

    @app.get("/api/wiki/stats")
    async def get_wiki_stats():
        wiki_dir = Path.home() / ".prometheus" / "wiki"
        if not wiki_dir.exists():
            return {"page_count": 0, "entity_counts": {}, "last_compiled": None}

        entity_counts: dict[str, int] = {}
        page_count = 0
        for subdir in wiki_dir.iterdir():
            if subdir.is_dir() and subdir.name not in {"queries", "__pycache__"}:
                count = len(list(subdir.glob("*.md")))
                entity_counts[subdir.name] = count
                page_count += count

        log = wiki_dir / "wiki_log.md"
        last_compiled = log.stat().st_mtime if log.exists() else None

        return {
            "page_count": page_count,
            "entity_counts": entity_counts,
            "last_compiled": last_compiled,
        }

    # ── LCM / Context ──────────────────────────────────────────────

    @app.get("/api/lcm/{session_id}")
    async def get_lcm_state(session_id: str):
        if not lcm_engine:
            return {
                "session_id": session_id,
                "total_tokens": 0,
                "limit": 24000,
                "compression_ratio": 0,
                "fresh_count": 0,
                "summary_count": 0,
            }
        # Attempt to read from LCM engine stores
        try:
            result = lcm_engine.assemble(session_id, token_budget=24000)
            return {
                "session_id": session_id,
                "total_tokens": result.total_tokens,
                "limit": 24000,
                "compression_ratio": result.compression_ratio,
                "fresh_count": len(result.fresh_messages),
                "summary_count": len(result.summaries),
            }
        except Exception:
            return {
                "session_id": session_id,
                "total_tokens": 0,
                "limit": 24000,
                "compression_ratio": 0,
                "fresh_count": 0,
                "summary_count": 0,
            }

    # ── SENTINEL ────────────────────────────────────────────────────

    @app.get("/api/sentinel")
    async def get_sentinel():
        if not signal_bus:
            return {
                "state": "idle",
                "last_dream": None,
                "dream_count": 0,
                "idle_since": None,
                "dream_log_tail": [],
            }

        recent = signal_bus.recent(limit=100)
        dream_signals = [s for s in recent if s.kind.startswith("dream_")]
        idle_signals = [s for s in recent if s.kind == "idle_start"]

        # Determine current state
        state = "active"
        if dream_signals and dream_signals[0].kind == "dream_start":
            state = "dreaming"
        elif idle_signals:
            state = "idle"

        # Build dream log from recent signals
        dream_count = len([s for s in recent if s.kind == "dream_complete"])
        last_dream = None
        if dream_signals:
            completes = [s for s in dream_signals if s.kind == "dream_complete"]
            if completes:
                last_dream = completes[0].timestamp

        return {
            "state": state,
            "last_dream": last_dream,
            "dream_count": dream_count,
            "idle_since": idle_signals[0].timestamp if idle_signals else None,
            "dream_log_tail": [],  # Populated from dream_log.md in production
        }

    # ── Events (SignalBus Persistence sprint) ──────────────────────

    @app.get("/api/events/recent")
    async def get_events_recent(limit: int = 50, type: str | None = None):
        """Hydrate Beacon's activity feed from the durable signal_events tail.

        Closes the "blank activity feed until something happens" UX gap:
        the UI calls this on mount, then subscribes to live events via the
        existing WebSocket bridge (which broadcasts emissions in real time).

        Query params:
          - limit (int, default 50, capped to 500): max rows.
          - type (str, optional): single signal_type filter
            (``skill_created``, ``memory_updated``, …).
        """
        from prometheus.telemetry.tracker import get_telemetry_handle

        tel = get_telemetry_handle()
        if tel is None:
            return []
        capped_limit = max(1, min(int(limit), 500))
        rows = tel.signal_events_since(
            signal_type=type,
            limit=capped_limit,
        )
        # Already shaped as the spec expects (dicts with id, timestamp,
        # signal_type, payload, source_subsystem). Return as-is.
        return rows

    # ── Activity feed (Polish sprint WS2) ──────────────────────────
    # Alias for /api/events/recent under the "activity" namespace. The
    # Beacon frontend calls this on mount to hydrate the live feed before
    # subscribing to the WebSocket.

    @app.get("/api/activity/recent")
    async def get_activity_recent(limit: int = 100, type: str | None = None):
        """Recent durable signal-bus events, capped at *limit* (max 500)."""
        from prometheus.telemetry.tracker import get_telemetry_handle

        tel = get_telemetry_handle()
        if tel is None:
            return []
        capped = max(1, min(int(limit), 500))
        return tel.signal_events_since(signal_type=type, limit=capped)

    # ── Memory (Polish sprint WS2) ─────────────────────────────────

    @app.get("/api/memory/current")
    async def get_memory_current():
        """Return current MEMORY.md and USER.md content with usage stats."""
        try:
            from prometheus.memory.hermes_memory_tool import (
                _MEMORY_MAX_CHARS,
                _USER_MAX_CHARS,
                get_memory_store,
                get_user_store,
            )
        except Exception:
            return {"memory": None, "user": None, "error": "memory module unavailable"}

        def _snapshot(store, limit: int) -> dict[str, Any]:
            entries = store.list_entries()
            chars = sum(len(e) + 1 for e in entries)
            return {
                "entries": list(entries),
                "char_count": chars,
                "char_limit": limit,
                "entry_count": len(entries),
                "content": "\n".join(entries),
            }

        return {
            "memory": _snapshot(get_memory_store(), _MEMORY_MAX_CHARS),
            "user": _snapshot(get_user_store(), _USER_MAX_CHARS),
        }

    # ── Skills (richer list + content + pin/unpin) ─────────────────

    @app.get("/api/skills/list")
    async def get_skills_list():
        """Auto-skills with state, pinned flag, last-used mtime.

        Mirrors the cmd_skills_auto_list shape so the Beacon panel and
        the Telegram /skills command can be reasoned about together.
        """
        try:
            from prometheus.config.paths import get_config_dir
            from prometheus.learning.skill_state import SkillStateStore
        except Exception:
            return []

        auto_dir = get_config_dir() / "skills" / "auto"
        if not auto_dir.is_dir():
            return []

        store = SkillStateStore()
        rows: list[dict[str, Any]] = []
        for path in sorted(auto_dir.glob("*.md")):
            if path.name.startswith("_"):
                continue
            name = path.stem
            rec = store.get_skill(name)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows.append({
                "name": name,
                "pinned": rec.pinned,
                "state": rec.state,
                "first_seen_at": rec.first_seen_at,
                "last_modified": mtime,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "notes": rec.notes,
            })
        return rows

    @app.get("/api/skills/{name}")
    async def get_skill_content(name: str):
        """Return the SKILL.md content for *name* (auto/ first, then root)."""
        from prometheus.config.paths import get_config_dir

        if not name or "/" in name or ".." in name:
            return JSONResponse(status_code=400, content={"error": "invalid name"})

        skills_root = get_config_dir() / "skills"
        for candidate in (
            skills_root / "auto" / f"{name}.md",
            skills_root / f"{name}.md",
        ):
            if candidate.exists():
                try:
                    return {
                        "name": name,
                        "path": str(candidate),
                        "content": candidate.read_text(encoding="utf-8"),
                        "size_bytes": candidate.stat().st_size,
                        "last_modified": candidate.stat().st_mtime,
                    }
                except OSError as exc:
                    return JSONResponse(status_code=500, content={"error": str(exc)})
        return JSONResponse(status_code=404, content={"error": "skill not found"})

    @app.post("/api/skills/{name}/pin")
    async def pin_skill(name: str):
        try:
            from prometheus.learning.skill_state import SkillStateStore
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        if not name or "/" in name or ".." in name:
            return JSONResponse(status_code=400, content={"error": "invalid name"})
        rec = SkillStateStore().set_pinned(name, True)
        return {"name": name, "pinned": rec.pinned, "state": rec.state}

    @app.delete("/api/skills/{name}/pin")
    async def unpin_skill(name: str):
        try:
            from prometheus.learning.skill_state import SkillStateStore
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        if not name or "/" in name or ".." in name:
            return JSONResponse(status_code=400, content={"error": "invalid name"})
        rec = SkillStateStore().set_pinned(name, False)
        return {"name": name, "pinned": rec.pinned, "state": rec.state}

    # ── Cron ────────────────────────────────────────────────────────

    @app.get("/api/cron")
    async def get_cron_jobs():
        try:
            from prometheus.gateway.cron_service import load_cron_jobs
            return load_cron_jobs()
        except Exception:
            return []

    # ── Approvals ──────────────────────────────────────────────────

    @app.get("/api/approvals")
    async def get_approvals():
        queue = app.state.approval_queue
        if not queue:
            return []
        return [
            {
                "request_id": a.request_id,
                "tool_name": a.tool_name,
                "description": a.description,
                "created_at": a.created_at,
            }
            for a in queue.list_pending()
        ]

    @app.post("/api/approvals/{request_id}/approve")
    async def approve_action(request_id: str):
        queue = app.state.approval_queue
        if not queue:
            return JSONResponse(status_code=404, content={"error": "approval queue not enabled"})
        ok = await queue.approve(request_id)
        return {"ok": ok}

    @app.post("/api/approvals/{request_id}/deny")
    async def deny_action(request_id: str):
        queue = app.state.approval_queue
        if not queue:
            return JSONResponse(status_code=404, content={"error": "approval queue not enabled"})
        ok = await queue.deny(request_id)
        return {"ok": ok}

    # ── Chat ───────────────────────────────────────────────────────

    @app.post("/api/chat")
    async def send_chat(body: dict):
        """Send a message to the agent — mirrors Telegram dispatch."""
        session_id = body.get("session_id", "")
        content = body.get("content", "")
        if not session_id or not content:
            return JSONResponse(status_code=400, content={"error": "session_id and content required"})
        if not agent_loop:
            return JSONResponse(status_code=503, content={"error": "agent loop not available"})
        if not session_mgr:
            return JSONResponse(status_code=503, content={"error": "session manager not available"})

        session = session_mgr.get_or_create(f"web:{session_id}")
        session.add_user_message(content)

        try:
            system_prompt = config.get("gateway", {}).get(
                "system_prompt",
                "You are Prometheus, a sovereign AI agent. Be concise and helpful.",
            )
            result = await agent_loop.run_async(
                system_prompt=system_prompt,
                messages=session.get_messages(),
                tools=app.state.skill_registry.list_schemas() if app.state.skill_registry else None,
            )
            session.add_result_messages(result.messages, len(session.get_messages()) - 1)
            return {
                "text": result.text,
                "turns": result.turns,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                },
            }
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    # ── Benchmarks (placeholder) ────────────────────────────────────

    @app.post("/api/benchmarks/run")
    async def run_benchmarks(body: dict | None = None):
        return {"status": "not_implemented", "message": "Benchmark runner not yet wired"}

    # ── Static files (must be last — catch-all) ─────────────────────

    if static_dir:
        static_path = Path(static_dir)
        if static_path.exists():
            app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

    return app


def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove secrets from config before exposing via API."""
    import copy
    safe = copy.deepcopy(config)

    secret_keys = {"token", "api_key", "secret", "password", "credential"}
    def _redact(d: dict) -> None:
        for k, v in list(d.items()):
            if isinstance(v, dict):
                _redact(v)
            elif isinstance(v, str) and any(s in k.lower() for s in secret_keys):
                d[k] = "***REDACTED***"

    _redact(safe)
    return safe


async def start_web(app: FastAPI, host: str = "0.0.0.0", port: int = 8005) -> None:
    """Start the FastAPI server using uvicorn."""
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
