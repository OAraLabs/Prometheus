"""Does this provider key work? — the check behind ``POST /api/setup/check-key``.

Beacon asks it BEFORE it configures anything: after its own OpenRouter sign-in, and after a person pastes an
Anthropic or OpenAI key. The answer is one of four words (Beacon's O9 contract):

    ok            the provider accepted the key
    bad_key       the provider refused it (or it cannot be a key at all)
    no_credit     the key is good and the ACCOUNT has no credit (OpenRouter, where it is the common failure)
    unreachable   we could not tell: no network, a timeout, a redirect, a 5xx, a rate limit, anything unexpected

Rules this module exists to keep:

* **Classification is by HTTP status, never by a provider's wording.** OpenAI's 401 body echoes a masked copy of the
  key, and any service may reword a message tomorrow; a verdict must not depend on either.
  One exception, and it is narrow: **Gemini and xAI answer an invalid key with 400, not 401**, which by status alone
  is "could not tell" and would report a mistyped key as a network problem. For a 400 from those two a
  machine-readable error CODE names the key (Google's ``ErrorInfo.reason == "API_KEY_INVALID"``; xAI's
  ``code == "invalid-argument"`` on a request that has no other argument). Any other 400 stays ``unreachable``, and
  the English message is still never read. Both answers are recorded, not guessed
  (``tests/fixtures/provider_key_checks``).
* **Nothing the provider says is relayed.** ``detail`` is OUR sentence. A provider body can contain the key.
* **The key goes in the auth header only** — never a URL, a body, a log line or an exception message — and every
  URL is fixed here. This is a fixed-destination check, not a proxy: only ``CHECKABLE_PROVIDERS`` are known.
* **Nothing follows a redirect** and every call has a short timeout.
* **Anything unexpected is ``unreachable``, never ``ok``.**
* **OpenRouter is checked with a request that SPENDS (one token), after the free call.** No credit cannot be seen
  any other way: ``GET /key`` answers 200 for a key whose account is empty. The free call goes first so a key that
  is already refused never spends anything. Anthropic, OpenAI, Gemini and xAI use their free model-list call, so for
  them a valid key on an empty account still answers ``ok`` — the first chat says otherwise.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OK = "ok"
BAD_KEY = "bad_key"
NO_CREDIT = "no_credit"
UNREACHABLE = "unreachable"

#: The providers this will call, at the fixed URLs below. Anything else is refused by the route.
CHECKABLE_PROVIDERS = ("anthropic", "openai", "openrouter", "gemini", "xai")

_OPENROUTER = "https://openrouter.ai/api/v1"
_ANTHROPIC = "https://api.anthropic.com/v1"
_OPENAI = "https://api.openai.com/v1"
#: Gemini's NATIVE model list, not the OpenAI-compatible endpoint the daemon calls: that one answers a bad key with a
#: bare 400 and no machine-readable reason. Same key, same service.
_GEMINI = "https://generativelanguage.googleapis.com/v1beta"
_XAI = "https://api.x.ai/v1"
_ANTHROPIC_VERSION = "2023-06-01"

#: Printable ASCII with no space: the only thing that is a safe HTTP header value. A pasted key with a newline or
#: a stray space is not a key, and sending it would raise inside the HTTP library with the key in the message.
_KEY_SHAPE = re.compile(r"[\x21-\x7e]{8,512}")

_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class KeyCheck:
    """The verdict. ``status`` is the provider's HTTP status when it answered; ``detail`` is our own words."""

    result: str
    status: int | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"result": self.result}
        if self.status is not None:
            out["status"] = self.status
        if self.detail:
            out["detail"] = self.detail
        return out


_DETAIL = {
    OK: "the provider accepted the key",
    BAD_KEY: "the provider refused the key",
    NO_CREDIT: "the key is valid but the account has no credit",
}


def _make_client() -> httpx.Client:
    """The HTTP client for a real check. A function so a test can put a stand-in provider behind it."""
    return httpx.Client(timeout=httpx.Timeout(_TIMEOUT_SECONDS), follow_redirects=False)


def _verdict(status: int) -> str:
    """HTTP status → one of the four words. The ONLY place a response is interpreted."""
    if 200 <= status < 300:
        return OK
    if status in (401, 403):
        return BAD_KEY
    if status == 402:
        return NO_CREDIT
    return UNREACHABLE  # 3xx (never followed), 400, 404, 408, 429, 5xx: we could not tell


def _result(verdict: str, status: int) -> KeyCheck:
    detail = _DETAIL.get(verdict) or f"the provider answered {status}, which says nothing about the key"
    return KeyCheck(verdict, status, detail)


def _check_openrouter(client: httpx.Client, key: str) -> KeyCheck:
    from prometheus.providers.registry import CLOUD_DEFAULTS

    auth = {"Authorization": f"Bearer {key}"}
    # 1. Free. A refused key stops here and nothing is spent.
    first = client.get(f"{_OPENROUTER}/key", headers=auth)
    verdict = _verdict(first.status_code)
    if verdict != OK:
        return _result(verdict, first.status_code)
    # 2. One token on the model setup would configure (paid, never a free one — a free model would answer for an
    #    account with no credit and hide exactly the failure this check is for).
    second = client.post(
        f"{_OPENROUTER}/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        json={
            "model": CLOUD_DEFAULTS["openrouter"]["model"],
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    return _result(_verdict(second.status_code), second.status_code)


def _check_anthropic(client: httpx.Client, key: str) -> KeyCheck:
    resp = client.get(f"{_ANTHROPIC}/models", headers={"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION})
    return _result(_verdict(resp.status_code), resp.status_code)


def _check_openai(client: httpx.Client, key: str) -> KeyCheck:
    resp = client.get(f"{_OPENAI}/models", headers={"Authorization": f"Bearer {key}"})
    return _result(_verdict(resp.status_code), resp.status_code)


def _json_body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:                                  # not JSON (an HTML error page, an empty body)
        return None


def _gemini_error_reasons(resp: httpx.Response) -> set[str]:
    """The ``google.rpc.ErrorInfo`` reasons in a Google error body. Only the enum, never the message."""
    body = _json_body(resp)
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, list):
        return set()
    return {d["reason"] for d in details if isinstance(d, dict) and isinstance(d.get("reason"), str)}


def _check_gemini(client: httpx.Client, key: str) -> KeyCheck:
    # One model, not the catalogue. The key goes in Google's own header; it also accepts ``?key=``, which would put
    # it in a URL.
    resp = client.get(f"{_GEMINI}/models", params={"pageSize": 1}, headers={"x-goog-api-key": key})
    if resp.status_code == 400 and "API_KEY_INVALID" in _gemini_error_reasons(resp):
        return _result(BAD_KEY, resp.status_code)
    return _result(_verdict(resp.status_code), resp.status_code)


def _check_xai(client: httpx.Client, key: str) -> KeyCheck:
    resp = client.get(f"{_XAI}/models", headers={"Authorization": f"Bearer {key}"})
    if resp.status_code == 400:
        body = _json_body(resp)
        # ``invalid-argument`` on a body-less GET that takes no parameters: the only argument is the credential. (A
        # missing or empty one is a 401 ``unauthenticated:*``, which the status rule already reads as bad_key.)
        if isinstance(body, dict) and body.get("code") == "invalid-argument":
            return _result(BAD_KEY, resp.status_code)
    return _result(_verdict(resp.status_code), resp.status_code)


_CHECKS = {
    "openrouter": _check_openrouter, "anthropic": _check_anthropic, "openai": _check_openai,
    "gemini": _check_gemini, "xai": _check_xai,
}


def check_provider_key(provider: str, api_key: str, *, client: httpx.Client | None = None) -> KeyCheck:
    """Ask *provider* whether *api_key* works. Blocking (a couple of short HTTP calls); never raises on a
    network or provider failure — that is ``unreachable`` — and never puts the key anywhere but the auth header.

    Raises ``ValueError`` for a provider that is not in ``CHECKABLE_PROVIDERS``: the route refuses those first.
    """
    check = _CHECKS.get(provider)
    if check is None:
        raise ValueError(f"unsupported provider {provider!r}")
    key = api_key if isinstance(api_key, str) else ""
    if not _KEY_SHAPE.fullmatch(key):
        return KeyCheck(BAD_KEY, None, "that is not shaped like an API key")
    own = client is None
    http = _make_client() if client is None else client
    try:
        verdict = check(http, key)
    except (httpx.HTTPError, OSError) as exc:
        # The exception CLASS only: a message can carry a URL or a header, and a header carries the key.
        logger.info("key check %s: unreachable (%s)", provider, type(exc).__name__)
        return KeyCheck(UNREACHABLE, None, "could not reach the provider")
    finally:
        if own:
            http.close()
    logger.info("key check %s: %s (HTTP %s)", provider, verdict.result, verdict.status)
    return verdict
