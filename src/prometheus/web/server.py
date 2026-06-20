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

from prometheus.context.environment import git_head_sha

# Cap on the coding diff payload (matches the files-preview 256 KB cap).
_CODING_DIFF_CAP = 256 * 1024


def _load_coding_report(output_file: "Path") -> dict | None:
    """Parse a coding run's final JSON report from its managed-task output file.

    The ``prometheus code`` subprocess prints exactly one JSON object (the run
    report) to stdout at the end; while running, the file holds no such object.
    Returns the LAST balanced top-level ``{...}`` that parses to a dict with a
    ``status`` field, or ``None`` if no report has been written yet.
    """
    import json as _json

    try:
        text = output_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Scan from the end for a line-anchored object, then the whole-text fallback.
    candidates: list[str] = []
    idx = text.rfind("\n{")
    if idx >= 0:
        candidates.append(text[idx + 1 :].strip())
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    for cand in candidates:
        try:
            obj = _json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("status"), str):
            return obj
    return None


def _coding_sandbox_root() -> "Path":
    """The directory API-launched coding runs clone into (~/.prometheus/coding)."""
    from prometheus.config.paths import get_data_dir

    return (get_data_dir().parent / "coding").resolve()


# ── Loop Manager project files (feat/loop-file-endpoint) — NARROW write expansion ──
# The daemon may read/write ONLY these basenames, ONLY at the top level of an existing git
# repo, validated by the SAME check POST /api/code uses (_resolve_coding_repo). This is the
# deliberate, scoped widening that lets a REMOTE Beacon (e.g. on a Mac) edit a project's
# TASKS.md/LOOP.md/PROGRESS.md that live on the daemon host. Every rejection is specific —
# no generic "could not write" (that swallowed error is what made the original bug hard to
# diagnose). This does NOT touch /api/files or its ~/.prometheus/workspace sandbox.
_PROJECT_FILES = ("TASKS.md", "LOOP.md", "PROGRESS.md")


def _resolve_coding_repo(repo: str) -> Path:
    """The SINGLE coding-repo validator, shared by POST /api/code and /api/project-file (one
    validator, reused — not a second one). Resolve the path; assert it exists and is a git
    repo. Raises ``ValueError`` with a SPECIFIC reason."""
    repo = (repo or "").strip()
    if not repo:
        raise ValueError("repo path is required")
    repo_path = Path(repo).expanduser()
    if not repo_path.exists():
        raise ValueError(f"path does not exist: {repo}")
    if not (repo_path / ".git").exists():
        raise ValueError(f"not a git repository (no .git): {repo}")
    return repo_path


def _resolve_project_file(repo: str, name: str) -> Path:
    """Resolve a Loop Manager project file to its absolute path under the repo root, enforcing
    the NARROW scope. Raises ``ValueError`` with a SPECIFIC reason for each rejection: blank or
    sub-path ``name``, disallowed basename, and — via :func:`_resolve_coding_repo` — missing
    path / non-git-repo. Never widens past the three allowlisted files at the repo top level."""
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    if name != Path(name).name or name in ("..", "."):
        raise ValueError(f"name must be a top-level filename, not a path: {name!r}")
    if name not in _PROJECT_FILES:
        raise ValueError(f"filename not permitted (only {', '.join(_PROJECT_FILES)}): {name!r}")
    repo_path = _resolve_coding_repo(repo)
    target = repo_path / name
    # Defense in depth: the resolved target must sit DIRECTLY in the repo root (so a symlink at
    # <repo>/TASKS.md can't redirect a write outside the repo).
    if target.resolve().parent != repo_path.resolve():
        raise ValueError(f"path does not resolve to the repo root: {name!r}")
    return target


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
    boot_sha: str = "unknown",
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

    # ── Bearer token auth + body-size guard on /api/* routes ────────

    _MAX_BODY_BYTES = 2 * 1024 * 1024  # REST bodies are JSON/text; file uploads use the WS, not /api

    @app.middleware("http")
    async def _check_bearer_token(request: Request, call_next):
        # Reject oversized bodies up front (by Content-Length, before any read).
        # NOTE: a chunked request without Content-Length sidesteps this — a hard
        # streaming cap belongs at the uvicorn/reverse-proxy layer.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > _MAX_BODY_BYTES:
                    return JSONResponse(status_code=413, content={"error": "request body too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"error": "invalid Content-Length"})
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
    app.state.boot_sha = boot_sha
    app.state.agent_state = "idle"
    app.state.current_model = config.get("model", {}).get("model", "unknown")
    app.state.current_provider = config.get("model", {}).get("provider", "unknown")
    app.state.active_profile = config.get("profiles", {}).get("default", "full")

    # Coding live-stream (feat/coding-livestream): tails telemetry.db per coding
    # run and fans coding_round/coding_complete/coding_stream_error over the WS
    # bridge. Read-only; started by POST /api/code, stopped by the run's
    # task_completed/task_failed event (subscribed here).
    from prometheus.coding.livestream import CodingLiveStream, DEFAULT_DB_PATH
    coding_stream = CodingLiveStream(
        signal_bus,
        db_path=getattr(telemetry, "db_path", DEFAULT_DB_PATH),
    )
    coding_stream.subscribe_lifecycle()
    app.state.coding_stream = coding_stream

    # ── Boot-SHA staleness (merged-but-dark detector) ───────────────
    # running_sha = repo HEAD captured at process boot (frozen on app.state);
    # tree_head = live repo HEAD now. They differ when new code is on disk that
    # this process hasn't loaded. "unknown" on either side → never stale.
    def _staleness() -> tuple[str, str, bool]:
        running = getattr(app.state, "boot_sha", "unknown")
        tree = git_head_sha()
        stale = running != "unknown" and tree != "unknown" and running != tree
        return running, tree, stale

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
                "/api/files", "/api/documents",
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
            # Bare staleness bool for external monitors — no SHA leaked on this
            # unauthenticated endpoint (full SHAs are on bearer-gated /api/status).
            "stale": _staleness()[2],
        }

    # ── Status ──────────────────────────────────────────────────────

    @app.get("/api/status")
    async def get_status():
        running_sha, tree_head, stale = _staleness()
        return {
            "state": app.state.agent_state,
            "model": app.state.current_model,
            "provider": app.state.current_provider,
            "profile": app.state.active_profile,
            "uptime_seconds": time.time() - app.state.start_time,
            "running_sha": running_sha,
            "tree_head": tree_head,
            "stale": stale,
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
        # "Forget session": REMOVE the in-memory entry so the session stops
        # being listed by GET /api/sessions (which enumerates
        # session_mgr._sessions). NOT session_mgr.clear() — that only empties
        # the message list and leaves the entry listed until a daemon restart,
        # so junk/test sessions could never be made to disappear. Durable LCM
        # rows are append-only and deliberately left intact; we forget only the
        # live session handle. See SessionManager.remove().
        if session_mgr:
            session_mgr.remove(session_id)
        return {"ok": True}

    # ── Telemetry ───────────────────────────────────────────────────

    @app.get("/api/telemetry")
    async def get_telemetry():
        if not telemetry:
            return {"total_calls": 0, "overall_success_rate": 0, "tools": {}}
        return telemetry.report()

    @app.get("/api/pairs")
    async def get_pairs():
        """Repair-pair flywheel stats (training_pairs in training.db)."""
        try:
            from prometheus.learning.pair_capture import PairStore, get_store

            store = get_store() or PairStore()
            return store.stats()
        except Exception as exc:
            return {"total": 0, "error": str(exc)}

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
            # Don't silently disguise a real failure as "no data" — that hid the
            # missing get_leaf_summaries for a long time. Log it; still return a
            # safe zeros shape so the panel degrades instead of 500-ing.
            import logging

            logging.getLogger(__name__).exception(
                "LCM assemble failed for session %s — returning empty context state", session_id
            )
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

    @app.post("/api/cron")
    async def create_cron_job(body: dict):
        """Create a cron job. Body: {name, schedule, command, cwd?, enabled?}.

        `schedule` must be a valid 5-field cron expression (the REST surface is
        strict — natural-language parsing stays in the cron_create tool). Names
        are unique: a duplicate is 409, not a silent overwrite (use PUT to edit).
        """
        from prometheus.gateway.cron_service import (
            get_cron_job,
            upsert_cron_job,
            validate_cron_expression,
        )

        name = str(body.get("name", "")).strip()
        schedule = str(body.get("schedule", "")).strip()
        command = str(body.get("command", "")).strip()
        if not name or not schedule or not command:
            return JSONResponse(
                status_code=400,
                content={"error": "name, schedule and command are required"},
            )
        if not validate_cron_expression(schedule):
            return JSONResponse(
                status_code=400,
                content={"error": f"invalid cron expression: {schedule!r}"},
            )
        if get_cron_job(name) is not None:
            return JSONResponse(
                status_code=409,
                content={"error": f"cron job {name!r} already exists"},
            )
        # SECURITY: refuse to even store a job whose command would be blocked at
        # system trust (the execute path enforces this too — this is fail-fast).
        from prometheus.gateway.cron_scheduler import vet_cron_command

        allowed, reason = vet_cron_command(command)
        if not allowed:
            return JSONResponse(
                status_code=400,
                content={"error": f"command rejected by SecurityGate: {reason}"},
            )
        job: dict[str, Any] = {
            "name": name,
            "schedule": schedule,
            "command": command,
            "enabled": bool(body.get("enabled", True)),
        }
        if body.get("cwd"):
            job["cwd"] = str(body["cwd"])
        upsert_cron_job(job)
        return JSONResponse(status_code=201, content={"ok": True, "job": get_cron_job(name)})

    @app.put("/api/cron/{name}")
    async def update_cron_job(name: str, body: dict):
        """Edit an existing cron job in place (merge of any of schedule/command/
        cwd/enabled). 404 if the name is unknown; 400 on an invalid schedule."""
        from prometheus.gateway.cron_service import (
            get_cron_job,
            upsert_cron_job,
            validate_cron_expression,
        )

        existing = get_cron_job(name)
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "cron job not found"})
        if "schedule" in body:
            schedule = str(body.get("schedule", "")).strip()
            if not validate_cron_expression(schedule):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"invalid cron expression: {schedule!r}"},
                )
            existing["schedule"] = schedule
        if "command" in body:
            command = str(body.get("command", "")).strip()
            if not command:
                return JSONResponse(status_code=400, content={"error": "command cannot be empty"})
            from prometheus.gateway.cron_scheduler import vet_cron_command

            allowed, reason = vet_cron_command(command)
            if not allowed:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"command rejected by SecurityGate: {reason}"},
                )
            existing["command"] = command
        if "cwd" in body:
            existing["cwd"] = str(body["cwd"]) if body["cwd"] else None
        if "enabled" in body:
            existing["enabled"] = bool(body["enabled"])
        upsert_cron_job(existing)  # replaces by name + recomputes next_run
        return {"ok": True, "job": get_cron_job(name)}

    @app.delete("/api/cron/{name}")
    async def remove_cron_job(name: str):
        from prometheus.gateway.cron_service import delete_cron_job

        if not delete_cron_job(name):
            return JSONResponse(status_code=404, content={"error": "cron job not found"})
        return {"ok": True, "name": name}

    @app.post("/api/cron/{name}/run")
    async def run_cron_job_now(name: str):
        """Trigger a job immediately. Fire-and-forget: a job may run up to 300s, so
        the executor is dispatched as a background task and the call returns at once;
        the outcome surfaces via the job's last_run/last_status (re-GET /api/cron)."""
        import asyncio

        from prometheus.gateway.cron_scheduler import execute_job
        from prometheus.gateway.cron_service import get_cron_job

        job = get_cron_job(name)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "cron job not found"})

        async def _run() -> None:
            try:
                await execute_job(job)
            except Exception:  # never leave a fire-and-forget task's exception unretrieved
                pass

        asyncio.create_task(_run())
        return {"ok": True, "name": name, "started": True}

    # ── Files (Beacon file browser) ─────────────────────────────────
    #
    # Sandboxed read-only browse of the agent WORKSPACE (get_workspace_dir() —
    # ~/.prometheus/workspace, or $PROMETHEUS_WORKSPACE_DIR). Beacon Desktop runs
    # REMOTE from this host, so it can't read the workspace via local fs — this is
    # its only file access. Every path is resolved and confined to the root:
    # .. / absolute / symlink escapes are rejected (resolve() follows symlinks).

    from prometheus.config.paths import get_workspace_dir

    _FILES_READ_CAP = 256 * 1024  # 256 KB text-preview cap

    def _files_root() -> Path:
        # PROMETHEUS_FILES_ROOT widens/repoints the BROWSE root only (read-only,
        # token-gated). Deliberately separate from PROMETHEUS_WORKSPACE_DIR, which
        # doubles as an image_generate WRITE root — repointing that to browse more
        # would widen the agent's write surface too.
        env_root = os.environ.get("PROMETHEUS_FILES_ROOT", "").strip()
        if env_root:
            return Path(env_root).expanduser().resolve()
        return get_workspace_dir().resolve()

    def _safe_file_path(rel: str) -> Path | None:
        """Resolve `rel` under the workspace root; None if it escapes the root."""
        root = _files_root()
        candidate = (root / rel.lstrip("/")).resolve()
        if candidate == root or root in candidate.parents:
            return candidate
        return None

    @app.get("/api/files")
    async def list_files(path: str = ""):
        root = _files_root()
        target = _safe_file_path(path)
        if target is None:
            return JSONResponse(status_code=403, content={"error": "path escapes workspace"})
        if not target.exists() or not target.is_dir():
            return JSONResponse(status_code=404, content={"error": "directory not found"})
        try:
            children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as exc:
            return JSONResponse(status_code=500, content={"error": f"listdir failed: {exc}"})
        entries = []
        for child in children:
            try:
                st = child.stat()
                is_dir = child.is_dir()
            except OSError:
                continue  # skip dangling symlinks / unreadable entries
            entries.append({
                "name": child.name,
                "type": "dir" if is_dir else "file",
                "size": 0 if is_dir else st.st_size,
                "mtime": st.st_mtime,
            })
        return {"path": "" if target == root else str(target.relative_to(root)), "entries": entries}

    @app.get("/api/files/read")
    async def read_file(path: str = ""):
        root = _files_root()
        target = _safe_file_path(path)
        if target is None:
            return JSONResponse(status_code=403, content={"error": "path escapes workspace"})
        if not target.exists() or not target.is_file():
            return JSONResponse(status_code=404, content={"error": "file not found"})
        size = target.stat().st_size
        with target.open("rb") as fh:
            raw = fh.read(_FILES_READ_CAP)
        binary = b"\x00" in raw
        content = ""
        if not binary:
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                binary = True
        return {
            "path": str(target.relative_to(root)),
            "size": size,
            "truncated": size > _FILES_READ_CAP,
            "binary": binary,
            "content": content,
        }

    # ── Documents Editor (SPRINT Documents Editor, Phase A) ─────────
    #
    # A confined, disk-backed writing surface under a GENERAL documents root
    # (config documents.root, default ~/.prometheus/documents) — distinct from
    # coding mode's per-run clone jail. Reuses the coding sandbox's path
    # confinement (resolve: traversal / absolute-outside / symlink-escape all
    # rejected) and the code_str_replace edit primitive; composes SecurityGate
    # denied-paths on top. read/save/edit are model-free; suggest is one-shot.

    def _documents_service():
        from prometheus.config.paths import get_documents_dir
        from prometheus.documents import DocumentsService
        from prometheus.permissions.checker import SecurityGate

        cfg = app.state.config if isinstance(app.state.config, dict) else {}
        sec = cfg.get("security", {}) or {}
        docs_cfg = cfg.get("documents", {}) or {}
        root = (
            Path(docs_cfg["root"]).expanduser()
            if docs_cfg.get("root")
            else get_documents_dir()
        )
        denied = sec.get("denied_paths") or []
        # Defense in depth: denied paths feed BOTH the sandbox confinement and
        # the SecurityGate composed over it.
        gate = SecurityGate(denied_paths=denied, workspace_root=sec.get("workspace_root"))
        return DocumentsService(root, denied_paths=denied, gate=gate)

    def _documents_error(exc) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.message})

    @app.get("/api/documents")
    async def list_documents(path: str = ""):
        from prometheus.documents import DocumentsError

        try:
            rel, entries = _documents_service().list_dir(path)
        except DocumentsError as exc:
            return _documents_error(exc)
        return {
            "path": rel,
            "entries": [
                {"name": e.name, "type": e.type, "size": e.size, "mtime": e.mtime}
                for e in entries
            ],
        }

    @app.get("/api/documents/content")
    async def read_document(path: str = ""):
        from prometheus.documents import DocumentsError

        if not path:
            return JSONResponse(status_code=400, content={"error": "path is required"})
        try:
            return _documents_service().read(path)
        except DocumentsError as exc:
            return _documents_error(exc)

    @app.put("/api/documents/content")
    async def save_document(body: dict):
        from prometheus.documents import DocumentsError

        path = str(body.get("path", "")).strip()
        if not path:
            return JSONResponse(status_code=400, content={"error": "path is required"})
        content = body.get("content", "")
        if not isinstance(content, str):
            return JSONResponse(status_code=400, content={"error": "content must be a string"})
        try:
            return _documents_service().save(path, content)
        except DocumentsError as exc:
            return _documents_error(exc)

    @app.post("/api/documents/edit")
    async def edit_document(body: dict):
        from prometheus.documents import DocumentsError

        path = str(body.get("path", "")).strip()
        # Accept the str_replace primitive's names, or the suggest mode's
        # {find, replace} aliases (so an approved redline applies unchanged).
        old_str = body.get("old_str", body.get("find", ""))
        new_str = body.get("new_str", body.get("replace", ""))
        if not path:
            return JSONResponse(status_code=400, content={"error": "path is required"})
        if not isinstance(old_str, str) or not isinstance(new_str, str):
            return JSONResponse(status_code=400, content={"error": "old_str/new_str must be strings"})
        try:
            result = await _documents_service().apply_edit(path, old_str, new_str)
        except DocumentsError as exc:
            return _documents_error(exc)
        if not result.ok:
            # The primitive's loud no-match / multi-match error — 422, file untouched.
            return JSONResponse(status_code=422, content={"ok": False, "error": result.error})
        return {"ok": True, "diff": result.diff}

    @app.post("/api/documents/suggest")
    async def suggest_document_edits(body: dict):
        from prometheus.documents import DocumentsError
        from prometheus.documents.ai import generate_suggestions

        path = str(body.get("path", "")).strip()
        instruction = str(body.get("instruction", "")).strip()
        if not path or not instruction:
            return JSONResponse(
                status_code=400, content={"error": "path and instruction are required"}
            )

        cfg = app.state.config if isinstance(app.state.config, dict) else {}
        model_cfg = cfg.get("model", {}) or {}
        model_name = model_cfg.get("model") or app.state.current_model
        try:
            from prometheus.providers.registry import ProviderRegistry

            provider = ProviderRegistry.create(model_cfg)
        except Exception as exc:  # noqa: BLE001 — surface, never 500-with-traceback
            return JSONResponse(
                status_code=503, content={"error": f"model provider unavailable: {exc}"}
            )

        async def _gen(content: str, instr: str) -> list[dict]:
            return await generate_suggestions(provider, model_name, content, instr)

        try:
            edits = await _documents_service().suggest(path, instruction, _gen)
        except DocumentsError as exc:
            return _documents_error(exc)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=502, content={"error": f"suggest failed: {exc}"})
        return {
            "edits": [
                {
                    "find": e.find,
                    "replace": e.replace,
                    "reason": e.reason,
                    "applicable": e.applicable,
                    "note": e.note,
                }
                for e in edits
            ]
        }

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

    # ── Coding mode (SPRINT-coding-mode v2) — Beacon/API reachability ──
    #
    # POST launches one sandboxed iterate-to-green run as a managed task
    # (durable in tasks.db, SecurityGate-vetted at launch, completion
    # notification via the existing task_completed/task_failed signal
    # path). GET inspects it. The JSON run report is the task's output.

    @app.post("/api/code")
    async def create_coding_run(body: dict):
        from uuid import uuid4 as _uuid4

        from prometheus.coding.managed import create_coding_managed_task
        from prometheus.tasks.manager import get_task_manager

        repo = str(body.get("repo", "")).strip()
        description = str(body.get("description", "")).strip()
        acceptance = str(body.get("acceptance_command", "")).strip()
        if not (repo and description and acceptance):
            return JSONResponse(status_code=400, content={
                "error": "repo, description and acceptance_command are required"
            })
        try:
            repo_path = _resolve_coding_repo(repo)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        coding_task_id = str(body.get("task_id") or f"c{_uuid4().hex[:8]}")
        try:
            record = await create_coding_managed_task(
                get_task_manager(),
                repo=str(repo_path),
                description=description,
                acceptance_command=acceptance,
                task_id=coding_task_id,
                cwd=str(Path.home()),
                max_rounds=int(body.get("max_rounds", 30)),
                max_wall_seconds=int(body.get("max_wall_seconds", 1200)),
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        if record.status == "failed":
            # SecurityGate rejected the command at registration — surface it.
            return JSONResponse(status_code=400, content={
                "error": record.error or "task rejected at registration"
            })
        # Begin live-streaming this run's rounds over the WS (read-only tail).
        # The session_id matches both the coding telemetry key and the managed
        # task's session_id, so the run's terminal event stops this tail.
        stream = getattr(app.state, "coding_stream", None)
        if stream is not None:
            stream.start_tail(f"coding:{coding_task_id}")
        return {
            "task_id": record.id,            # managed-task handle (poll this)
            "coding_task_id": coding_task_id,  # names the branch coding/<id>
            "status": record.status,
            "output_file": str(record.output_file),
        }

    # ── Loop Manager project files — read/write TASKS.md/LOOP.md/PROGRESS.md at a git repo's
    # root (Option B: remote Beacon edits files on the daemon host). NARROW + specific errors.
    @app.get("/api/project-file")
    async def read_project_file(repo: str = "", name: str = ""):
        """Read a project file. 400 (specific) on a bad request; 404 when the file is absent."""
        try:
            target = _resolve_project_file(repo, name)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        if not target.exists():
            return JSONResponse(status_code=404, content={"error": f"file not found: {name}"})
        try:
            content = target.read_text(encoding="utf-8")
            mtime_ms = target.stat().st_mtime * 1000.0
        except OSError as exc:
            return JSONResponse(status_code=500, content={"error": f"read failed: {exc}"})
        return {"content": content, "mtimeMs": mtime_ms}

    @app.put("/api/project-file")
    async def write_project_file(body: dict):
        """Write a project file — ONLY {TASKS.md, LOOP.md, PROGRESS.md}, ONLY at the top level of
        an existing git repo (the SAME repo/.git validation /api/code uses). Every rejection
        returns its specific reason; this never touches /api/files' sandbox."""
        content = body.get("content", "")
        if not isinstance(content, str):
            return JSONResponse(status_code=400, content={"error": "content must be a string"})
        try:
            target = _resolve_project_file(str(body.get("repo", "")), str(body.get("name", "")))
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            target.write_text(content, encoding="utf-8")
            mtime_ms = target.stat().st_mtime * 1000.0
        except OSError as exc:
            return JSONResponse(status_code=500, content={"error": f"write failed: {exc}"})
        return {"mtimeMs": mtime_ms}

    @app.get("/api/code/{task_id}")
    async def get_coding_run(task_id: str):
        from prometheus.tasks.manager import get_task_manager

        record = get_task_manager().get_task(task_id)
        if record is None:
            return JSONResponse(status_code=404, content={"error": "no such task"})
        output_tail = ""
        try:
            output_tail = record.output_file.read_text(
                encoding="utf-8", errors="replace"
            )[-4_000:]
        except OSError:
            pass
        return {
            "task_id": record.id,
            "status": record.status,
            "description": record.description,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
            "return_code": record.return_code,
            "error": record.error,
            "output_tail": output_tail,
        }

    @app.post("/api/code/{task_id}/stop")
    async def stop_coding_run(task_id: str):
        """Cancel a running coding task — SIGTERM→SIGKILL the subprocess. Thin
        wrapper over BackgroundTaskManager.stop_task (which already marks the
        task ``killed``, persists, and emits completion)."""
        from prometheus.tasks.manager import get_task_manager

        mgr = get_task_manager()
        if mgr.get_task(task_id) is None:
            return JSONResponse(status_code=404, content={"error": "no such task"})
        try:
            record = await mgr.stop_task(task_id)
        except ValueError:
            # Already terminal / not a running task.
            return JSONResponse(status_code=409, content={"error": "task is not running"})
        except Exception as exc:  # noqa: BLE001 — surface, never 500-with-traceback
            return JSONResponse(status_code=500, content={"error": str(exc)})
        return {"task_id": record.id, "status": record.status}

    # ── Mid-run supervision (Loop Manager Sprint 2) — pause / inject / resume ──
    # Thin writers of the run's control file (coding/control.py); the coding
    # subprocess polls it at its episode seam. Run-id resolution mirrors /stop
    # (managed task_id → 404 if unknown); the control file is keyed by the coding
    # id (the task's session_id, ``coding:<id>``). No shell command is executed
    # here, so there is nothing to SecurityGate — the bearer auth on /api/ is the
    # access boundary, and the injected steer is TRUSTED human guidance
    # (provenance=supervisor), not an executable. Idempotent + non-destructive on
    # already-in-state; 409 once the run is terminal (a finished run can't steer).

    def _control_path_for(record) -> "Path":
        from prometheus.coding.control import control_path
        from prometheus.coding.managed import coding_control_dir
        coding_id = (record.session_id or "").removeprefix("coding:") or record.id
        return control_path(coding_control_dir(coding_id))

    @app.post("/api/code/{task_id}/pause")
    async def pause_coding_run(task_id: str):
        from prometheus.coding.control import read_state_for_write, with_paused, write_state
        from prometheus.tasks.manager import get_task_manager
        from prometheus.tasks.types import TERMINAL_STATUSES

        record = get_task_manager().get_task(task_id)
        if record is None:
            return JSONResponse(status_code=404, content={"error": "no such task"})
        if record.status in TERMINAL_STATUSES:
            return JSONResponse(status_code=409, content={"error": "run is not active"})
        path = _control_path_for(record)
        state = read_state_for_write(path)
        if state.paused:
            return {"task_id": task_id, "status": "already-paused"}
        write_state(path, with_paused(state, True))
        return {"task_id": task_id, "status": "paused"}

    @app.post("/api/code/{task_id}/resume")
    async def resume_coding_run(task_id: str):
        from prometheus.coding.control import read_state_for_write, with_paused, write_state
        from prometheus.tasks.manager import get_task_manager
        from prometheus.tasks.types import TERMINAL_STATUSES

        record = get_task_manager().get_task(task_id)
        if record is None:
            return JSONResponse(status_code=404, content={"error": "no such task"})
        if record.status in TERMINAL_STATUSES:
            return JSONResponse(status_code=409, content={"error": "run is not active"})
        path = _control_path_for(record)
        state = read_state_for_write(path)
        if not state.paused:
            return {"task_id": task_id, "status": "not-paused"}
        write_state(path, with_paused(state, False))
        return {"task_id": task_id, "status": "resumed"}

    @app.post("/api/code/{task_id}/inject")
    async def inject_coding_run(task_id: str, body: dict):
        """Body: ``{ "text": <the correction> }``. Queues a human steer the run applies at its
        next episode seam, trust-tagged ``provenance="supervisor"``."""
        from uuid import uuid4 as _uuid4

        from prometheus.coding.control import Injection, read_state_for_write, with_injection, write_state
        from prometheus.tasks.manager import get_task_manager
        from prometheus.tasks.types import TERMINAL_STATUSES

        text = str(body.get("text", "")).strip()
        if not text:
            return JSONResponse(status_code=400, content={"error": "text (the correction) is required"})
        if len(text) > 8000:
            return JSONResponse(status_code=400, content={"error": "correction too long (max 8000 chars)"})
        record = get_task_manager().get_task(task_id)
        if record is None:
            return JSONResponse(status_code=404, content={"error": "no such task"})
        if record.status in TERMINAL_STATUSES:
            return JSONResponse(status_code=409, content={"error": "run is not active — cannot steer a finished run"})
        path = _control_path_for(record)
        injection = Injection(id=_uuid4().hex[:8], text=text)
        write_state(path, with_injection(read_state_for_write(path), injection))
        return {"task_id": task_id, "status": "injected", "injection_id": injection.id}

    @app.get("/api/code/{task_id}/diff")
    async def get_coding_diff(task_id: str):
        """Full unified diff of a finished coding run's artifact commit.

        Reads ``sandbox_root`` from the run's report (the daemon's own trusted
        output), validates it resolves UNDER ~/.prometheus/coding and is a git
        repo (defense in depth), and runs ``git -C <root> diff HEAD~1..HEAD`` —
        the same range that produced the report's ``diff_stat``. ``ready:false``
        while the run hasn't reported yet."""
        import subprocess as _sp

        from prometheus.tasks.manager import get_task_manager

        record = get_task_manager().get_task(task_id)
        if record is None:
            return JSONResponse(status_code=404, content={"error": "no such task"})
        report = _load_coding_report(record.output_file)
        if report is None:
            return {"ready": False, "diff": "", "branch": None}
        sandbox_root = report.get("sandbox_root")
        if not isinstance(sandbox_root, str) or not sandbox_root:
            return JSONResponse(status_code=422, content={"error": "report has no sandbox_root"})
        root = Path(sandbox_root).resolve()
        coding_root = _coding_sandbox_root()
        if not (root == coding_root or root.is_relative_to(coding_root)) or not (root / ".git").is_dir():
            return JSONResponse(status_code=422, content={"error": "sandbox path is not a coding repo"})
        try:
            proc = _sp.run(
                ["git", "-C", str(root), "--no-pager", "diff", "HEAD~1..HEAD"],
                capture_output=True, text=True, timeout=20,
            )
        except (OSError, _sp.SubprocessError) as exc:
            return JSONResponse(status_code=500, content={"error": f"git diff failed: {exc}"})
        diff = proc.stdout or ""
        truncated = len(diff) > _CODING_DIFF_CAP
        if truncated:
            diff = diff[:_CODING_DIFF_CAP]
        return {
            "ready": True,
            "branch": report.get("branch"),
            "diff": diff,
            "truncated": truncated,
        }

    # ── Kanban (Projects + Stories board — Beacon Desktop) ──────────
    #
    # Daemon-backed projects/stories store (prometheus.kanban). Mirrors the
    # Beacon web app's model so a client pointed at this daemon shares state.
    # snake_case out; `labels` is a list; timestamps are epoch seconds.

    @app.get("/api/projects")
    async def list_projects():
        from prometheus.kanban import get_kanban_store

        return get_kanban_store().list_projects()

    @app.post("/api/projects")
    async def create_project(body: dict):
        from prometheus.kanban import get_kanban_store

        name = str(body.get("name", "")).strip()
        if not name:
            return JSONResponse(status_code=400, content={"error": "name is required"})
        project = get_kanban_store().create_project(
            name=name,
            description=(body.get("description") or None),
            color=str(body.get("color") or "#58a6ff"),
        )
        return JSONResponse(status_code=201, content={"ok": True, "project": project})

    @app.put("/api/projects/{project_id}")
    async def update_project(project_id: str, body: dict):
        from prometheus.kanban import get_kanban_store

        if "name" in body and not str(body["name"]).strip():
            return JSONResponse(status_code=400, content={"error": "name cannot be empty"})
        fields = {k: body[k] for k in ("name", "description", "color") if k in body}
        updated = get_kanban_store().update_project(project_id, **fields)
        if updated is None:
            return JSONResponse(status_code=404, content={"error": "project not found"})
        return {"ok": True, "project": updated}

    @app.delete("/api/projects/{project_id}")
    async def delete_project(project_id: str):
        from prometheus.kanban import get_kanban_store

        if not get_kanban_store().delete_project(project_id):
            return JSONResponse(status_code=404, content={"error": "project not found"})
        return {"ok": True, "id": project_id}

    @app.get("/api/stories")
    async def list_stories(project_id: str | None = None):
        from prometheus.kanban import get_kanban_store

        return get_kanban_store().list_stories(project_id=project_id)

    @app.post("/api/stories")
    async def create_story(body: dict):
        from prometheus.kanban import STORY_PRIORITIES, STORY_STATUSES, get_kanban_store

        story_id = str(body.get("story_id", "")).strip()
        title = str(body.get("title", "")).strip()
        if not story_id or not title:
            return JSONResponse(status_code=400, content={"error": "story_id and title are required"})
        status = str(body.get("status") or "todo")
        priority = str(body.get("priority") or "medium")
        if status not in STORY_STATUSES:
            return JSONResponse(status_code=400, content={"error": f"invalid status: {status!r}"})
        if priority not in STORY_PRIORITIES:
            return JSONResponse(status_code=400, content={"error": f"invalid priority: {priority!r}"})
        labels = body.get("labels")
        story = get_kanban_store().create_story(
            story_id=story_id,
            title=title,
            project_id=(body.get("project_id") or None),
            description=(body.get("description") or None),
            status=status,
            priority=priority,
            assigned_agent=(body.get("assigned_agent") or None),
            labels=labels if isinstance(labels, list) else [],
            position=float(body.get("position") or 0),
        )
        return JSONResponse(status_code=201, content={"ok": True, "story": story})

    @app.put("/api/stories/{story_pk}")
    async def update_story(story_pk: str, body: dict):
        from prometheus.kanban import STORY_PRIORITIES, STORY_STATUSES, get_kanban_store

        if "status" in body and str(body["status"]) not in STORY_STATUSES:
            return JSONResponse(status_code=400, content={"error": f"invalid status: {body['status']!r}"})
        if "priority" in body and str(body["priority"]) not in STORY_PRIORITIES:
            return JSONResponse(status_code=400, content={"error": f"invalid priority: {body['priority']!r}"})
        allowed = (
            "project_id", "story_id", "title", "description", "status",
            "priority", "assigned_agent", "blocked_reason", "labels", "position",
        )
        fields = {k: body[k] for k in allowed if k in body}
        updated = get_kanban_store().update_story(story_pk, **fields)
        if updated is None:
            return JSONResponse(status_code=404, content={"error": "story not found"})
        return {"ok": True, "story": updated}

    @app.delete("/api/stories/{story_pk}")
    async def delete_story(story_pk: str):
        from prometheus.kanban import get_kanban_store

        if not get_kanban_store().delete_story(story_pk):
            return JSONResponse(status_code=404, content={"error": "story not found"})
        return {"ok": True, "id": story_pk}

    @app.post("/api/stories/reorder")
    async def reorder_stories(body: dict):
        from prometheus.kanban import STORY_STATUSES, get_kanban_store

        items = body.get("items")
        if not isinstance(items, list):
            return JSONResponse(
                status_code=400,
                content={"error": "items (list of {id, position, status}) is required"},
            )
        for item in items:
            st = str((item or {}).get("status", ""))
            if st and st not in STORY_STATUSES:
                return JSONResponse(status_code=400, content={"error": f"invalid status: {st!r}"})
        get_kanban_store().reorder_stories(items)
        return {"ok": True, "count": len(items)}

    @app.post("/api/stories/{story_pk}/dispatch")
    async def dispatch_story(story_pk: str, body: dict):
        """Send a story's task to a gateway session, then stamp it in-progress.

        Reuses the same ws_bridge path as POST /api/chat/send. Matches the web
        semantics: if no bridge is wired, 503 WITHOUT stamping (the task did not
        actually go out)."""
        from prometheus.kanban import get_kanban_store

        store = get_kanban_store()
        story = store.get_story(story_pk)
        if story is None:
            return JSONResponse(status_code=404, content={"error": "story not found"})
        session_key = str(body.get("session_key", "")).strip()
        if not session_key:
            return JSONResponse(status_code=400, content={"error": "session_key is required"})

        bridge = getattr(app.state, "ws_bridge", None)
        if bridge is None or not hasattr(bridge, "dispatch_user_message"):
            return JSONResponse(
                status_code=503,
                content={"error": "chat dispatch unavailable — ws_bridge not wired; story NOT dispatched"},
            )

        parts = [f"**Task: {story.get('title', '')}**"]
        if story.get("description"):
            parts.append(str(story["description"]))
        if story.get("story_id"):
            parts.append(f"_Task ID: {story['story_id']}_")
        await bridge.dispatch_user_message(session_key, "\n".join(parts), client_msg_id=None)

        updated = store.mark_dispatched(story_pk, session_key)
        return {"ok": True, "story": updated, "session_key": session_key}

    @app.post("/api/stories/{story_pk}/undispatch")
    async def undispatch_story(story_pk: str):
        from prometheus.kanban import get_kanban_store

        updated = get_kanban_store().undispatch(story_pk)
        if updated is None:
            return JSONResponse(status_code=404, content={"error": "story not found"})
        return {"ok": True, "story": updated}

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
