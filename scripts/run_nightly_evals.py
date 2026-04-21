#!/usr/bin/env python3
"""Nightly evaluation runner — designed for cron execution.

Usage:
    python scripts/run_nightly_evals.py
    python scripts/run_nightly_evals.py --tier 1
    python scripts/run_nightly_evals.py --no-skip-network
    PROMETHEUS_TRACING=1 python scripts/run_nightly_evals.py

Cron example (run at 3 AM daily):
    0 3 * * * cd ~/Prometheus && .venv/bin/python scripts/run_nightly_evals.py >> ~/.prometheus/eval_results/nightly.log 2>&1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

log = logging.getLogger("prometheus.nightly_evals")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_nightly_evals",
        description="Run Prometheus nightly evaluation suite",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to prometheus.yaml"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=None,
        help="Filter to specific tier",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        default=True,
        help="Skip tasks requiring web access (default)",
    )
    parser.add_argument(
        "--no-skip-network",
        dest="skip_network",
        action="store_false",
        help="Include tasks requiring web access",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override results directory"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import factory functions from __main__ (same pattern as scripts/daemon.py)
    from prometheus.__main__ import (
        load_config,
        create_provider,
        create_tool_registry,
        create_adapter,
        create_security_gate,
        build_system_prompt,
    )
    from prometheus.engine.agent_loop import AgentLoop
    from prometheus.evals.judge import PrometheusJudge
    from prometheus.evals.runner import EvalRunner
    from prometheus.tracing import init_tracing, shutdown_tracing

    # Load config
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    security_cfg = config.get("security", {})
    evals_cfg = config.get("evals", {})

    # Optionally init tracing
    init_tracing(config)

    # Build components
    provider, model_name = create_provider(model_cfg)
    security_gate = create_security_gate(security_cfg)
    registry = create_tool_registry(security_cfg, security_gate=security_gate)
    adapter = create_adapter(model_cfg)
    system_prompt = build_system_prompt(config)

    agent_loop = AgentLoop(
        provider=provider,
        model=model_name,
        tool_registry=registry,
        adapter=adapter,
        cwd=Path.cwd(),
    )

    # Judge — use evals config or fall back to model endpoint
    judge_url = evals_cfg.get(
        "judge_base_url", model_cfg.get("base_url", "http://localhost:8080")
    )

    # Health check judge endpoint
    import httpx

    try:
        resp = httpx.get(f"{judge_url.rstrip('/')}/v1/models", timeout=10)
        resp.raise_for_status()
        log.info("Judge endpoint healthy: %s", judge_url)
    except Exception as exc:
        log.error("Cannot reach judge at %s: %s", judge_url, exc)
        sys.exit(1)

    judge = PrometheusJudge(base_url=judge_url)
    runner = EvalRunner(
        agent_loop=agent_loop,
        judge=judge,
        system_prompt=system_prompt,
        config=evals_cfg,
    )

    async def _run() -> list:
        results = await runner.run_all(
            tier=args.tier,
            skip_network=args.skip_network,
        )
        output_dir = Path(args.output_dir) if args.output_dir else None
        runner.save_results(results, output_dir)
        runner.print_summary(results)
        return results

    try:
        results = asyncio.run(_run())
    finally:
        shutdown_tracing()

    # Exit code: 1 if any task errored
    if any(r.error for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
