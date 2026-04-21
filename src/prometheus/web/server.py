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
                "/api/wiki/stats", "/api/sentinel",
            ],
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
        sessions = []
        for sid, session in session_mgr._sessions.items():
            sessions.append({
                "session_id": sid,
                "created_at": session.created_at,
                "message_count": len(session.messages),
            })
        return sessions

    @app.post("/api/sessions")
    async def create_session():
        import uuid
        sid = str(uuid.uuid4())
        if session_mgr:
            session_mgr.get_or_create(sid)
        return {"session_id": sid}

    @app.get("/api/sessions/{session_id}/messages")
    async def get_messages(session_id: str):
        if not session_mgr:
            return []
        session = session_mgr._sessions.get(session_id)
        if not session:
            return []
        messages = []
        for msg in session.get_messages():
            messages.append({
                "message_id": getattr(msg, "id", "") or f"msg-{len(messages)}",
                "session_id": session_id,
                "role": msg.role,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                "timestamp": getattr(msg, "timestamp", 0),
            })
        return messages

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
