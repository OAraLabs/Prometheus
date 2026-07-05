"""``prometheus setup`` — the ONE canonical setup wizard.

Onboarding Phase 0, item 1: Prometheus previously shipped two competing
init paths — the rich wizard (``prometheus --setup``, identity + gateway
+ smoke test) and the fast probe (``prometheus-init``) — with different
outputs and no guidance. They are now a single subcommand:

    prometheus setup                   # rich wizard (identity, gateway,
                                       # smoke test) — the default
    prometheus setup --fast            # probe → yaml → env, 3 questions
    prometheus setup --noninteractive  # fast path, zero questions
    prometheus setup --gateway-only    # add/change a gateway only

Back-compat aliases (thin forwards, no behavior of their own):

- ``prometheus --setup`` / ``--setup-gateway-only``  → this command
- ``prometheus-init`` console script                 → ``setup --fast``
"""

from __future__ import annotations

import argparse
from pathlib import Path


def run_setup(args: argparse.Namespace) -> int:
    """Route to the rich wizard or the fast path. Returns an exit code."""
    fast = bool(getattr(args, "fast", False)) or bool(
        getattr(args, "noninteractive", False)
    )
    if fast:
        from prometheus.cli.init import run_init

        target_dir = getattr(args, "target_dir", None)
        config = run_init(
            noninteractive=bool(getattr(args, "noninteractive", False)),
            target_dir=Path(target_dir) if target_dir else None,
            timeout=float(getattr(args, "timeout", 1.0)),
        )
        # run_init returns None when it exited cleanly WITHOUT writing a
        # config (no server detected and the user chose the install-
        # instructions path) — nonzero so scripts notice, but not a crash.
        return 0 if config is not None else 2

    from prometheus.setup_wizard import SetupWizard

    wizard = SetupWizard(gateway_only=bool(getattr(args, "gateway_only", False)))
    return 0 if wizard.run() else 1


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (used by the forwarding aliases)."""
    parser = argparse.ArgumentParser(
        prog="prometheus setup",
        description="Set up Prometheus: detect your inference server, write "
                    "the config, generate identity, and smoke-test the loop.",
    )
    add_setup_arguments(parser)
    return run_setup(parser.parse_args(argv))


def add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``setup`` flags to *parser* (shared with the subparser)."""
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast path: probe local servers, write yaml + env template, done "
             "(skips identity generation and the smoke test)",
    )
    parser.add_argument(
        "--noninteractive", action="store_true",
        help="No prompts (implies --fast): first detected server, CLI gateway",
    )
    parser.add_argument(
        "--gateway-only", action="store_true",
        help="Only add or change a messaging gateway (rich wizard)",
    )
    parser.add_argument(
        "--target-dir", type=str, default=None,
        help="[fast path] config directory override (default: standard location)",
    )
    parser.add_argument(
        "--timeout", type=float, default=1.0,
        help="[fast path] per-server probe timeout in seconds (default 1.0)",
    )


def add_setup_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``setup`` subcommand on the main CLI parser."""
    p = subparsers.add_parser(
        "setup", help="First-run setup wizard (the one canonical path)",
    )
    add_setup_arguments(p)
