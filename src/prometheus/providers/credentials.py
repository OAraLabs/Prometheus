"""Provider credential status — subscription (OAuth) vs API key.

Single source of truth for "how would this provider authenticate right now",
shared by every surface that gates or reports provider availability (the
gateway /claude, /gpt, … slash commands and the /api/models catalog).

Precedence contract: a subscription source ALWAYS wins over an API key when
both are present — mirroring the runtime registry (_resolve_xai_credential
prefers a live OAuth store over XAI_API_KEY), so what these surfaces report
is what the provider will actually use.

Probes must be cheap: store-presence only, no network I/O, no token refresh —
they run on catalog GETs and on every provider-switch command.
"""

from __future__ import annotations

import os
from typing import Any, Callable


def _xai_subscription_active() -> bool:
    from prometheus.providers import xai_oauth

    return xai_oauth.is_logged_in()


# Provider name → subscription source descriptor. A provider absent here has
# no subscription path (API key only). ``is_active`` must be a no-network
# store-presence probe (see module docstring).
SUBSCRIPTION_SOURCES: dict[str, dict[str, Any]] = {
    "xai": {
        "label": "SuperGrok subscription",
        "hint": (
            "sign in with `uv run python scripts/xai_oauth_login.py` "
            "or via Beacon (Config → Models)"
        ),
        "is_active": _xai_subscription_active,
    },
}


def credential_status(provider: str, api_key_env: str = "") -> dict[str, Any]:
    """How ``provider`` would authenticate right now.

    Returns a dict:
      - ``mode``: ``"subscription"`` | ``"api_key"`` | ``None`` (no credential)
      - ``detail``: human string for the active mode (e.g. ``"SuperGrok
        subscription"`` / ``"API key (XAI_API_KEY)"``), ``None`` if no mode
      - ``has_subscription`` / ``has_api_key``: the individual probes
      - ``subscription_label`` / ``subscription_hint``: set when the provider
        HAS a subscription source (regardless of login state), for
        actionable error messages; ``None`` otherwise

    Subscription wins over an API key when both are present.
    """
    source = SUBSCRIPTION_SOURCES.get(provider)
    has_subscription = bool(source and source["is_active"]())
    has_api_key = bool(api_key_env and os.environ.get(api_key_env))

    if has_subscription:
        mode: str | None = "subscription"
        detail: str | None = source["label"]  # type: ignore[index]
    elif has_api_key:
        mode = "api_key"
        detail = f"API key ({api_key_env})"
    else:
        mode = None
        detail = None

    return {
        "mode": mode,
        "detail": detail,
        "has_subscription": has_subscription,
        "has_api_key": has_api_key,
        "subscription_label": source["label"] if source else None,
        "subscription_hint": source["hint"] if source else None,
    }
