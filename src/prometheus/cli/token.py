"""``prometheus token`` — show or rotate the web API token.

Part of the onboarding overhaul (Phase 0, item 2): the daemon mints a
token on first start with the web bridge enabled; this command is how
the user retrieves it later ("prometheus token show") or invalidates it
("prometheus token rotate").
"""

from __future__ import annotations

import argparse

from prometheus.config.api_token import (
    TOKEN_ENV_VAR,
    resolve_api_token,
    rotate_api_token,
)
from prometheus.config.env_file import get_env_file_path


def run_token_command(args: argparse.Namespace, config: dict | None = None) -> int:
    """Execute ``prometheus token <show|rotate>``. Returns an exit code."""
    action = getattr(args, "token_action", None)
    if action == "show":
        token, source = resolve_api_token(config)
        if not token:
            print("No web API token is set — the web API is OPEN (no auth).")
            print(f"Set one with: prometheus token rotate")
            print(f"(env file: {get_env_file_path()}, var: {TOKEN_ENV_VAR})")
            return 1
        print(token)
        print(f"(source: {source}; env file: {get_env_file_path()})")
        return 0
    if action == "rotate":
        token = rotate_api_token()
        print("New web API token generated and saved.")
        print(f"\n  {token}\n")
        print(f"Saved to: {get_env_file_path()}")
        print("Restart the daemon for it to take effect, then update your clients.")
        return 0
    print("Usage: prometheus token <show|rotate>")
    return 2


def add_token_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``token`` subcommand on the main CLI parser."""
    token_parser = subparsers.add_parser(
        "token", help="Show or rotate the web API token (PROMETHEUS_API_TOKEN)",
    )
    token_sub = token_parser.add_subparsers(dest="token_action")
    token_sub.add_parser("show", help="Print the current web API token")
    token_sub.add_parser(
        "rotate", help="Generate a new token and persist it to the env file",
    )
