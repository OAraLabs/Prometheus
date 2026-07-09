# Provenance: ported from NousResearch/hermes-agent (hermes_cli/auth.py,
# tools/xai_http.py) — the xAI SuperGrok device-code OAuth flow. MIT-licensed.
# The public desktop client_id and scope are the shared Grok-CLI values Hermes
# and the opencode-grok-auth plugin both use; verified live 2026-07-09.

"""SuperGrok OAuth credential source for the xAI provider.

Lets Prometheus authenticate to ``api.x.ai`` with a SuperGrok / X Premium+
subscription instead of a metered ``XAI_API_KEY``. The inference wire path is
unchanged — an OAuth access token is just a ``Bearer`` credential — so this
module only owns the *token lifecycle*:

    device-code login  ->  ~/.prometheus/xai_oauth.json  ->  auto-refresh

``get_access_token()`` is the one function the provider path calls: it returns a
currently-valid access token (refreshing transparently when within
:data:`REFRESH_SKEW_SECONDS` of expiry), or ``None`` when the user hasn't logged
in — in which case the registry falls back to ``XAI_API_KEY``.

Known limitation: xAI allowlist-gates ``GET /v1/models`` to HTTP 403 for
standard SuperGrok subscribers even though ``/v1/chat/completions`` works. We
never call ``/v1/models`` on this path; the model list is configured, not
enumerated.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

import httpx

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

# --- constants (verbatim from Hermes hermes_cli/auth.py) --------------------
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
ISSUER = "https://auth.x.ai"
DEVICE_CODE_URL = f"{ISSUER}/oauth2/device/code"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

# Refresh a little before the token actually expires so an in-flight request
# never races the boundary.
REFRESH_SKEW_SECONDS = 120

# Serialize refreshes/writes across the daemon's concurrent request handlers so
# two coroutines don't refresh the same token twice or interleave file writes.
_LOCK = threading.Lock()


def _store_path() -> Path:
    return get_config_dir() / "xai_oauth.json"


def _validate_xai_endpoint(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    if not url.startswith("https://") or not (host == "x.ai" or host.endswith(".x.ai")):
        raise ValueError(f"refusing non-xAI OAuth endpoint: {url!r}")
    return url


def _discover_token_endpoint(client: httpx.Client) -> str:
    r = client.get(DISCOVERY_URL, headers={"Accept": "application/json"})
    r.raise_for_status()
    ep = str(r.json().get("token_endpoint", "")).strip()
    if not ep:
        raise ValueError("xAI OIDC discovery did not return a token_endpoint")
    return _validate_xai_endpoint(ep)


def _read_store() -> dict[str, Any] | None:
    path = _store_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("xai_oauth: token store is unreadable/corrupt: %s", path)
        return None
    return data if isinstance(data, dict) and data.get("access_token") else None


def _write_store(data: dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.chmod(0o600)  # tokens are secrets — owner-only
    tmp.replace(path)


def _persist_tokens(payload: dict[str, Any], token_endpoint: str) -> dict[str, Any]:
    """Normalize a token response into the on-disk store shape and write it."""
    access = str(payload.get("access_token", "")).strip()
    refresh = str(payload.get("refresh_token", "")).strip()
    if not access or not refresh:
        raise ValueError("xAI token response missing access_token/refresh_token")
    try:
        expires_in = int(payload.get("expires_in") or 0)
    except (TypeError, ValueError):
        expires_in = 0
    store = {
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": time.time() + expires_in if expires_in else 0.0,
        "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        "token_endpoint": token_endpoint,
        "obtained_at": time.time(),
    }
    _write_store(store)
    return store


def is_logged_in() -> bool:
    """Cheap probe: True when a token store with an access_token exists.

    Does no network I/O and does not refresh — safe for hot paths (provider
    construction, catalog rendering). Freshness is handled by
    :func:`get_access_token`.
    """
    return _read_store() is not None


def _refresh(store: dict[str, Any], client: httpx.Client) -> dict[str, Any]:
    token_endpoint = _validate_xai_endpoint(
        str(store.get("token_endpoint") or _discover_token_endpoint(client))
    )
    r = client.post(
        token_endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": store["refresh_token"],
        },
    )
    if r.status_code != 200:
        raise RuntimeError(f"xAI token refresh failed (HTTP {r.status_code}): {r.text}")
    payload = r.json()
    # Some providers omit refresh_token on refresh — keep the prior one.
    payload.setdefault("refresh_token", store["refresh_token"])
    return _persist_tokens(payload, token_endpoint)


def get_access_token() -> str | None:
    """Return a currently-valid xAI OAuth access token, or ``None``.

    ``None`` means "not logged in" — the caller should fall back to
    ``XAI_API_KEY``. Refreshes transparently when the stored token is within
    :data:`REFRESH_SKEW_SECONDS` of expiry. On refresh failure, logs and
    returns ``None`` so a stale-refresh outage degrades to the API key rather
    than hard-failing every request.
    """
    with _LOCK:
        store = _read_store()
        if store is None:
            return None
        expires_at = float(store.get("expires_at") or 0.0)
        # expires_at == 0 means "unknown" — treat as valid (xAI omitted it).
        if expires_at and time.time() >= expires_at - REFRESH_SKEW_SECONDS:
            try:
                with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                    store = _refresh(store, client)
            except Exception:
                log.warning("xai_oauth: token refresh failed; falling back", exc_info=True)
                return None
        return str(store.get("access_token") or "") or None


def logout() -> bool:
    """Delete the token store. Returns True if a store was removed."""
    with _LOCK:
        path = _store_path()
        if path.exists():
            path.unlink()
            return True
        return False


def token_status() -> dict[str, Any]:
    """Return login status WITHOUT any network I/O or refresh.

    ``{"logged_in": bool, "expires_at": float | None}`` — safe for a status
    endpoint. ``expires_at`` is epoch seconds (or ``None`` when unknown/absent).
    """
    store = _read_store()
    if store is None:
        return {"logged_in": False, "expires_at": None}
    return {"logged_in": True, "expires_at": float(store.get("expires_at") or 0.0) or None}


def begin_device_login() -> dict[str, Any]:
    """Request a device code (fast, non-blocking).

    Does OIDC discovery + the device-code request and returns the bits a caller
    needs to (a) show the user where to approve and (b) later poll to
    completion via :func:`complete_device_login`. Does NOT wait for approval —
    so a REST endpoint can return the code immediately.
    """
    with httpx.Client(timeout=httpx.Timeout(30.0), headers={"Accept": "application/json"}) as client:
        token_endpoint = _discover_token_endpoint(client)
        dc = client.post(
            DEVICE_CODE_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={"client_id": CLIENT_ID, "scope": SCOPE},
        )
        if dc.status_code != 200:
            raise RuntimeError(f"xAI device-code request failed (HTTP {dc.status_code}): {dc.text}")
        d = dc.json()
        return {
            "verification_uri": str(d.get("verification_uri_complete") or d["verification_uri"]),
            "user_code": str(d["user_code"]),
            "expires_in": int(d["expires_in"]),
            "interval": max(1, int(d["interval"])),
            "device_code": str(d["device_code"]),
            "token_endpoint": token_endpoint,
        }


def complete_device_login(begin: dict[str, Any]) -> dict[str, Any]:
    """Poll the token endpoint until the user approves, then persist tokens.

    Blocking (up to the code's ``expires_in``) — call from a thread/background
    task, never the event loop. Takes the dict returned by
    :func:`begin_device_login`. Returns the persisted store; raises on
    denial/timeout.
    """
    token_endpoint = begin["token_endpoint"]
    with httpx.Client(timeout=httpx.Timeout(30.0), headers={"Accept": "application/json"}) as client:
        deadline = time.monotonic() + int(begin["expires_in"])
        cur = max(1, int(begin["interval"]))
        while time.monotonic() < deadline:
            resp = client.post(
                token_endpoint,
                headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                data={"grant_type": DEVICE_GRANT, "client_id": CLIENT_ID, "device_code": begin["device_code"]},
            )
            if resp.status_code == 200:
                return _persist_tokens(resp.json(), token_endpoint)
            try:
                err = str(resp.json().get("error") or "")
            except Exception:
                raise RuntimeError(f"non-JSON error polling device token (HTTP {resp.status_code}): {resp.text}")
            if err == "authorization_pending":
                time.sleep(cur)
            elif err == "slow_down":
                cur = min(cur + 1, 30)
                time.sleep(cur)
            else:
                raise RuntimeError(f"xAI device authorization failed: {resp.json()}")
        raise TimeoutError("timed out waiting for device approval")


def device_code_login(*, open_browser: bool = True, print_fn=print) -> dict[str, Any]:
    """Interactive CLI login: begin, show the code, then block until approved."""
    begin = begin_device_login()
    print_fn(f"Open this URL and sign in with SuperGrok:\n  {begin['verification_uri']}")
    print_fn(f"If prompted, enter code: {begin['user_code']}")
    print_fn(f"(code expires in {begin['expires_in']}s)")
    if open_browser:
        try:
            webbrowser.open(begin["verification_uri"])
        except Exception:
            pass
    store = complete_device_login(begin)
    print_fn("Logged in — tokens stored at " + str(_store_path()))
    return store
