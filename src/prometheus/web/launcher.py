"""Convenience launcher for the Mission Control web stack.

Starts both the FastAPI REST server (8005) and WebSocket bridge (8010)
as async tasks. Can be called from __main__.py or run standalone.

Usage in __main__.py:
    from prometheus.web.launcher import launch_web
    await launch_web(config, signal_bus=bus, session_mgr=mgr, ...)

Standalone:
    python -m prometheus.web --config ~/.prometheus/config/prometheus.yaml
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def launch_web(
    config: dict[str, Any],
    boot_sha: str = "unknown",
    signal_bus: Any | None = None,
    session_mgr: Any | None = None,
    telemetry: Any | None = None,
    skill_registry: Any | None = None,
    profile_store: Any | None = None,
    profile_state: Any | None = None,
    lcm_engine: Any | None = None,
    agent_loop: Any | None = None,
    approval_queue: Any | None = None,
    loop_context: Any | None = None,
    skill_creator: Any | None = None,
    gateway_adapter: Any | None = None,
    mcp_runtime: Any | None = None,
    static_dir: str | None = None,
    detected_context_size: int | None = None,
    local_model: str | None = None,
    detected_kv_cache: dict[str, Any] | None = None,
    api_host: str = "0.0.0.0",
    api_port: int = 8005,
    ws_host: str = "0.0.0.0",
    ws_port: int = 8010,
) -> None:
    """Start both REST API and WebSocket servers."""

    from pathlib import Path
    from prometheus.web.server import create_app, start_web
    from prometheus.web.ws_server import WebSocketBridge

    # Polish sprint WS2: when no explicit static_dir is passed, fall back to
    # the shipped frontend at <package>/web/static/. Users can override via
    # config.web.static_dir, but the package default means a fresh install
    # gets a working dashboard without extra wiring.
    if static_dir is None:
        cfg_static = config.get("web", {}).get("static_dir")
        if cfg_static:
            static_dir = str(cfg_static)
        else:
            shipped = Path(__file__).resolve().parent / "static"
            if shipped.exists():
                static_dir = str(shipped)

    # Shared mutable state ref for agent state
    agent_state_ref = {"state": "idle"}

    # GRAFT-MOBILE-BRIDGE 1: ONE DeviceStore shared by the REST middleware and
    # the WS bridge, so a revocation or last_seen stamp is immediately visible
    # to both. Created unconditionally — on an open (token-less) daemon device
    # tokens are simply never consulted.
    from prometheus.config.device_store import DeviceStore

    device_store = DeviceStore()

    # Create FastAPI app
    app = create_app(
        config=config,
        boot_sha=boot_sha,
        signal_bus=signal_bus,
        session_mgr=session_mgr,
        telemetry=telemetry,
        skill_registry=skill_registry,
        profile_store=profile_store,
        profile_state=profile_state,
        lcm_engine=lcm_engine,
        agent_loop=agent_loop,
        approval_queue=approval_queue,
        model_router=getattr(loop_context, "model_router", None),
        static_dir=static_dir,
        skill_creator=skill_creator,
        device_store=device_store,
        # The resolved-budget inputs. Without these the web app has no context
        # window in scope, which is how /api/lcm came to publish a literal.
        detected_context_size=detected_context_size,
        local_model=local_model,
        detected_kv_cache=detected_kv_cache,
    )

    # Wire agent state ref into the app
    app.state.agent_state_ref = agent_state_ref
    # Sprint 2 (OAra): surface the compactor so /api/status can report the
    # compaction block — the middleware config audit probes it (config-dark law).
    app.state.compactor = getattr(loop_context, "compactor", None)
    # Gateway liveness so /api/status can report a dark gateway. None when no
    # gateway is wired — /api/status then reports "wired: false" rather than
    # inventing a health verdict for something that isn't there.
    app.state.gateway_adapter = gateway_adapter
    # #332: the live MCP runtime (None = MCP unconfigured this boot). The
    # /api/mcp routes drive connect/disconnect/probe through this handle and
    # reach the tool registry via loop_context, same as _deferred_status.
    app.state.mcp_runtime = mcp_runtime

    # Create WebSocket bridge. The WS uses the SAME token as the REST
    # middleware (config.web.api_token or PROMETHEUS_API_TOKEN); empty => auth
    # disabled on both, in lockstep. Resolved here, never logged.
    import os as _os
    _api_token = config.get("web", {}).get("api_token") or _os.environ.get(
        "PROMETHEUS_API_TOKEN", ""
    )
    bridge = WebSocketBridge(
        signal_bus=signal_bus,
        session_mgr=session_mgr,
        loop_context=loop_context,
        agent_state_ref=agent_state_ref,
        api_token=_api_token,
        config=config,
        approval_queue=approval_queue,
        device_store=device_store,
        detected_context_size=detected_context_size,
        local_model=local_model,
    )

    # Expose the bridge on the FastAPI app so REST routes (e.g.
    # POST /api/chat/send) can dispatch user messages through the same flow
    # the WebSocket uses, without duplicating the session+agent plumbing.
    app.state.ws_bridge = bridge

    # GRAFT Piece 2: APNs push. Enabled-but-broken fails the BOOT, loudly —
    # a missing key or missing deps must not degrade to silent no-pushes
    # (config-dark law, same stance as the Paperclip gateway below).
    push_cfg = (config.get("push") or {})
    if isinstance(push_cfg, dict) and push_cfg.get("enabled", False):
        try:
            import cryptography  # noqa: F401
            import h2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "push.enabled is true but the push deps are missing — "
                "pip install 'oara-prometheus[push]' (cryptography + h2)"
            ) from exc
        from prometheus.push.apns import APNsSender, ApnsConfig
        from prometheus.push.dispatcher import PushDispatcher

        apns_config = ApnsConfig.from_config(push_cfg)  # raises on bad config
        sender = APNsSender(apns_config)
        dispatcher = PushDispatcher(device_store, sender, bridge)
        bridge.push_dispatcher = dispatcher
        if signal_bus:
            signal_bus.subscribe("*", dispatcher.on_signal)
        logger.info("APNs push enabled — topic %s, key %s",
                    apns_config.topic, apns_config.key_id)

    # Paperclip fleet orchestration: when gateway.paperclip.enabled, mount the
    # heartbeat client behind POST /api/paperclip/wake. Runs work turns through
    # the SAME bridge flow as Beacon chat (sessions paperclip:issue:{id}).
    # A bad config raises here — an enabled-but-broken gateway must fail the
    # boot loudly, not silently drop wakes (config-dark law).
    paperclip_cfg = (config.get("gateway", {}) or {}).get("paperclip") or {}
    if isinstance(paperclip_cfg, dict) and paperclip_cfg.get("enabled"):
        from prometheus.gateway.paperclip import PaperclipGateway

        app.state.paperclip_gateway = PaperclipGateway(
            paperclip_cfg, bridge, daemon_config=config
        )
        logger.info(
            "Paperclip gateway enabled — wake endpoint POST /api/paperclip/wake -> %s",
            paperclip_cfg.get("api_url"),
        )

    logger.info("Starting Mission Control — REST on :%d, WebSocket on :%d", api_port, ws_port)

    # Run both servers concurrently
    await asyncio.gather(
        start_web(app, host=api_host, port=api_port),
        bridge.start(host=ws_host, port=ws_port),
    )
