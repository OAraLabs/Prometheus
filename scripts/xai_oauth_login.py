#!/usr/bin/env python3
"""Log in to xAI Grok with a SuperGrok / X Premium+ subscription (device-code).

Usage:
    uv run python scripts/xai_oauth_login.py           # log in
    uv run python scripts/xai_oauth_login.py --status  # show current state
    uv run python scripts/xai_oauth_login.py --logout  # delete stored tokens

After a successful login the xAI provider (/xai, model.provider: xai) uses the
subscription automatically — OAuth takes precedence over XAI_API_KEY. Tokens
live at ~/.prometheus/xai_oauth.json (owner-only) and refresh on their own.

On a headless box, run this in a terminal and open the printed URL in any
browser — the device-code flow needs no loopback callback.
"""
from __future__ import annotations

import sys

from prometheus.providers import xai_oauth


def main(argv: list[str]) -> int:
    if "--status" in argv:
        tok = xai_oauth.get_access_token()
        if tok:
            print(f"Logged in — access token valid (len={len(tok)}).")
        elif xai_oauth.is_logged_in():
            print("Token store present but no valid token (refresh failed?). Re-run to log in.")
        else:
            print("Not logged in. Run without flags to log in.")
        return 0

    if "--logout" in argv:
        print("Logged out." if xai_oauth.logout() else "Nothing to log out — no token store.")
        return 0

    # Default: interactive login. Don't auto-open a browser on a headless box.
    open_browser = "--open" in argv
    try:
        xai_oauth.device_code_login(open_browser=open_browser)
    except Exception as exc:  # surface a clean message, not a traceback
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
