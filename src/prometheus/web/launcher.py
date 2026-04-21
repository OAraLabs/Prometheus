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
    signal_bus: Any | None = None,
    session_mgr: Any | None = None,
    telemetry: Any | None = None,
    skill_registry: Any | None = None,
    profile_store: Any | None = None,
    lcm_engine: Any | None = None,
    agent_loop: Any | None = None,
    approval_queue: Any | None = None,
    loop_context: Any | None = None,
    static_dir: str | None = None,
    api_host: str = "0.0.0.0",
    api_port: int = 8005,
    ws_host: str = "0.0.0.0",
    ws_port: int = 8010,
) -> None:
    """Start both REST API and WebSocket servers."""

    from prometheus.web.server import create_app, start_web
    from prometheus.web.ws_server import WebSocketBridge

    # Shared mutable state ref for agent state
    agent_state_ref = {"state": "idle"}

    # Create FastAPI app
    app = create_app(
        config=config,
        signal_bus=signal_bus,
        session_mgr=session_mgr,
        telemetry=telemetry,
        skill_registry=skill_registry,
        profile_store=profile_store,
        lcm_engine=lcm_engine,
        agent_loop=agent_loop,
        approval_queue=approval_queue,
        static_dir=static_dir,
    )

    # Wire agent state ref into the app
    app.state.agent_state_ref = agent_state_ref

    # Create WebSocket bridge
    bridge = WebSocketBridge(
        signal_bus=signal_bus,
        session_mgr=session_mgr,
        loop_context=loop_context,
        agent_state_ref=agent_state_ref,
    )

    logger.info("Starting Mission Control — REST on :%d, WebSocket on :%d", api_port, ws_port)

    # Run both servers concurrently
    await asyncio.gather(
        start_web(app, host=api_host, port=api_port),
        bridge.start(host=ws_host, port=ws_port),
    )
