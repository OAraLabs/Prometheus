"""``oara token`` — show or rotate the web API token.

Part of the onboarding overhaul (Phase 0, item 2): the daemon mints a
token on first start with the web bridge enabled; this command is how
the user retrieves it later ("oara token show") or invalidates it
("oara token rotate").
"""

from __future__ import annotations

import argparse

from prometheus.config.api_token import (
    TOKEN_ENV_VAR,
    TokenRotationBlocked,
    resolve_api_token,
    rotate_api_token,
)
from prometheus.config.env_file import get_env_file_path


def _location_of(source: str) -> str:
    """Human-readable home of the token `resolve_api_token` picked."""
    if source == "config":
        from prometheus.config.defaults import resolve_config_path

        return f"web.api_token in {resolve_config_path()}"
    if source == "env":
        return f"{TOKEN_ENV_VAR} in the process environment"
    return f"env file: {get_env_file_path()}"


def run_token_command(args: argparse.Namespace, config: dict | None = None) -> int:
    """Execute ``oara token <show|rotate>``. Returns an exit code."""
    action = getattr(args, "token_action", None)
    if action == "show":
        token, source = resolve_api_token(config)
        if not token:
            print("No web API token is set — the web API is OPEN (no auth).")
            print("Set one with: oara token rotate")
            print(f"(env file: {get_env_file_path()}, var: {TOKEN_ENV_VAR})")
            return 1
        print(token)
        # Name where the token ACTUALLY is. This used to print the env-file
        # path unconditionally, so a YAML-pinned token was reported as living
        # in a file that did not contain it (#320, the `show` half).
        print(f"(source: {source}; {_location_of(source)})")
        return 0
    if action == "rotate":
        try:
            token = rotate_api_token(config)
        except TokenRotationBlocked as exc:
            # Loud refusal beats a silent no-op: rotating the env file here
            # would report success and leave the live token valid (#320).
            print("REFUSING to rotate — the new token would not take effect.\n")
            print(f"  {exc}\n")
            print("The current token is UNCHANGED and still valid.")
            return 1
        print("New web API token generated and saved.")
        print(f"\n  {token}\n")
        print(f"Saved to: {get_env_file_path()}")
        print("Restart the daemon for it to take effect, then update your clients.")
        return 0
    print("Usage: oara token <show|rotate>")
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
