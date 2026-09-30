"""Classify a failed agent turn into something a human can act on.

The WS ``error`` frame used to carry ``str(exc)`` and nothing else. For an
HTTP failure that renders as::

    Client error '400 Bad Request' for url 'https://api.anthropic.com/v1/messages'

which tells the user nothing about the actual cause — the useful sentence
("Your credit balance is too low…") is in the *response body*, and the
actionable next step ("top up, or switch this session's model") is nowhere at
all. A live incident on 2026-07-24 burned real debugging time on exactly this:
Beacon showed a generic "no reply from the gateway", while the daemon knew it
was a billing rejection.

This module maps an exception to ``{kind, provider, status, message, hint}``:

* ``kind``     — a stable machine token clients can branch on (never freeform).
* ``provider`` — derived from the request HOST only.
* ``message``  — the provider's own words when available, else ``str(exc)``.
* ``hint``     — one actionable sentence for a human.

Deliberately duck-typed (no ``httpx`` import): it inspects ``exc.response`` and
exception class names, so it works for any HTTP client, stays importable in
minimal environments, and is trivially unit-testable with fakes.

SECURITY: only the URL *host* is ever read — never the full URL, query string,
or request headers. Some providers (Gemini) pass the API key as a ``?key=``
query parameter, so echoing a URL into a client-visible frame would leak the
credential. Response bodies are truncated and are the only provider text used.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Stable tokens for the wire. Clients branch on these; add, never repurpose.
KIND_BILLING = "billing"
KIND_AUTH = "auth"
KIND_RATE_LIMIT = "rate_limit"
KIND_TIMEOUT = "timeout"
KIND_UNREACHABLE = "unreachable"
KIND_PROVIDER_ERROR = "provider_error"
KIND_BAD_REQUEST = "bad_request"
KIND_UNKNOWN = "unknown"

# Longest provider message echoed onto the wire.
_MAX_MESSAGE_CHARS = 400

# host → canonical provider key (matches the keys GET /api/models reports).
_HOST_PROVIDERS = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "api.x.ai": "xai",
    "generativelanguage.googleapis.com": "gemini",
    "api.deepseek.com": "deepseek",
    "api.moonshot.cn": "kimi",
    "open.bigmodel.cn": "glm",
}

# Substrings that mean "the account cannot pay", regardless of status code.
# Providers disagree on the code (Anthropic 400, others 402/429), so the body
# is the reliable signal.
_BILLING_MARKERS = (
    "credit balance is too low",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "billing",
    "payment required",
)


# A URL's query string is where credentials hide (Gemini's ``?key=``), and
# httpx puts the FULL request URL into ``str(exc)`` — so the fallback message
# path leaks the key unless the query is stripped. Verified by
# test_never_echoes_a_url_or_query_string.
#
# IGNORECASE, because RFC 3986 makes the scheme case-insensitive and so does
# httpx: without the flag ``HTTP://h/x?key=…`` matched nothing and printed the
# key. This pattern is shared with _redact, so the wire-text path gains the same
# widening — deliberately, and it can only ever redact more.
_URL_QUERY_RE = re.compile(r"(https?://[^\s'\"?]+)\?[^\s'\"]*", re.IGNORECASE)

# The OTHER place a credential rides in a URL: the userinfo of the authority,
# ``scheme://user:password@host``. That is a real configuration — it is how a
# ``model.base_url`` reaches a llama.cpp or ollama server behind an
# authenticating proxy, and httpx turns it into a Basic auth header — and both
# doctors printed the URL exactly as written.
#
# Anchored so it can only match inside the AUTHORITY: the userinfo run is
# forbidden to contain ``/``, ``?`` or ``#``, so an ``@`` in a path
# (``/models@v1``) is left alone rather than being mistaken for a credential
# boundary. Everything after the ``@`` — host and port — is kept, because that
# is the actionable half of a "server not responding" row. IPv6 literals survive
# for the same reason: the ``@`` is consumed, not the bracket.
#
# The run is GREEDY and deliberately does not exclude ``@``, so it backtracks to
# the LAST ``@`` in the authority. That is what httpx itself does, and the
# difference is not cosmetic: a password may contain ``@``, and matching the
# FIRST one left the tail of the password in the output while also reporting a
# host that does not exist (``ss@host`` for ``user:pa@ss@host``). An operator
# sent to look for the wrong server is worse off than one shown the credential.
#
# IGNORECASE for the same reason as the query pattern above.
_URL_USERINFO_RE = re.compile(r"(https?://)[^/?#\s'\"]*@", re.IGNORECASE)

# The schemeless form: ``user:pass@host``. httpx accepts a bare ``host:port`` as
# a ``base_url``, so this reaches the "not responding" row too, and neither
# pattern above can match it — both are anchored to ``https?://``.
#
# There is no scheme to key off, so the authority is recognised by what it is
# NOT: the negative lookahead rejects any string that already opens with a
# scheme (``[a-z][a-z0-9+.-]*://`` — the RFC's scheme grammar, so a URL that the
# pattern above already handled is not touched twice), and the run stops at the
# first ``/``, ``?`` or ``#``, which is what keeps it out of a path. ``@`` is
# still allowed inside it, for the same last-``@`` reason as above.
#
# A bare ``127.0.0.1:8080`` has no ``@`` at all and is unchanged, which is the
# overwhelmingly common form of a schemeless ``base_url``.
_BARE_USERINFO_RE = re.compile(r"^(?![a-z][a-z0-9+.-]*://)[^/?#\s'\"]*@", re.IGNORECASE)

# Belt-and-braces: redact credential-shaped tokens anywhere in the text, since
# provider bodies can echo request material back at us and this string is
# broadcast to every connected client.
_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}"
    r"|ghp_[A-Za-z0-9]{8,}|gho_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}"
    r"|AIza[A-Za-z0-9_-]{8,}"
    r"|xai-[A-Za-z0-9_-]{8,})"
)


def _redact(text: str) -> str:
    """Strip URL query strings and credential-shaped tokens from wire text."""
    text = _URL_QUERY_RE.sub(r"\1", text)
    return _SECRET_RE.sub("<redacted>", text)


def _exception_names(exc: BaseException) -> set[str]:
    """Class names of ``exc`` and its bases — lets us detect httpx/asyncio
    error families without importing either."""
    return {klass.__name__ for klass in type(exc).__mro__}


def _host_of(exc: BaseException) -> str:
    """Request host, or "" — HOST ONLY, never the full URL (see module docstring)."""
    request = getattr(exc, "request", None)
    url = getattr(request, "url", None)
    host = getattr(url, "host", None)
    return str(host) if host else ""


def _provider_from_host(host: str) -> str:
    if not host:
        return "the model provider"
    if host in _HOST_PROVIDERS:
        return _HOST_PROVIDERS[host]
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return "the local model backend"
    return host


def _provider_message(body: str, exc: BaseException) -> str:
    """The provider's own error sentence, when it sent one.

    Both Anthropic and OpenAI shape errors as ``{"error": {"message": ...}}``.
    Falls back to a trimmed raw body, then to ``str(exc)``.
    """
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    return _redact(err["message"])[:_MAX_MESSAGE_CHARS]
                if isinstance(parsed.get("message"), str):
                    return _redact(parsed["message"])[:_MAX_MESSAGE_CHARS]
        except (ValueError, TypeError):
            pass
        stripped = body.strip()
        if stripped:
            return _redact(stripped)[:_MAX_MESSAGE_CHARS]
    return _redact(str(exc))[:_MAX_MESSAGE_CHARS]


def _hint(kind: str, provider: str, status: int | None) -> str:
    """One actionable sentence. Every path ends with a way forward."""
    switch = "or switch this session's model (composer dropdown)"
    if kind == KIND_BILLING:
        where = (
            " Add credits at console.anthropic.com."
            if provider == "anthropic"
            else " Check the account balance."
        )
        return f"{provider} rejected the request for billing reasons.{where} Otherwise, {switch}."
    if kind == KIND_AUTH:
        return (
            f"{provider} rejected the credentials — check the API key or re-authenticate, {switch}."
        )
    if kind == KIND_RATE_LIMIT:
        return f"{provider} is rate-limiting this account — wait a moment and retry, {switch}."
    if kind == KIND_TIMEOUT:
        return f"{provider} did not respond in time — retry, {switch}."
    if kind == KIND_UNREACHABLE:
        return (
            f"Could not reach {provider} — check that the backend is running and reachable, "
            f"{switch}."
        )
    if kind == KIND_PROVIDER_ERROR:
        return f"{provider} returned a server error ({status}) — usually transient, so retry."
    if kind == KIND_BAD_REQUEST:
        return f"{provider} rejected the request ({status})."
    return "Check the daemon log for the full traceback."


# ── Quota capture (#321) ─────────────────────────────────────────────────────
#
# WHY HEADERS AND NOT JUST THE BODY. A 429 alone cannot say whether the right
# response is "wait twenty seconds" or "wait three days". Three real failures
# against a flat-plan host on 2026-08-26/29/31 were logged as:
#
#     httpx.HTTPStatusError: Client error '429 Too Many Requests' for url '...'
#
# and nothing else. The retry ladder then ran (attempt 1/4, 2/4, 3/4 with 1.1s /
# 2.1s / 4.1s backoff), the turn died, and the fallback never fired. The status
# was captured; the thing that distinguishes the two cases was not, because
# nobody logged the headers.
#
# This list is deliberately an ALLOWLIST. A response's headers can carry
# `authorization`, `set-cookie` and provider request ids; dumping them all into
# a log that is read by humans and shipped in bug reports trades one blind spot
# for a credential leak.
QUOTA_HEADERS: tuple[str, ...] = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset",
    "ratelimit-reset",
    "ratelimit-remaining",
    "ratelimit-limit",
    "x-request-id",
)


def quota_headers(response: Any) -> dict[str, str]:
    """The rate/quota headers present on *response*, lowercased, allowlisted.

    Returns ``{}`` rather than raising for anything that is not a response —
    this runs on the error path and must never become the reason a failure is
    reported as a different failure.
    """
    try:
        headers = getattr(response, "headers", None)
        if headers is None:
            return {}
        out: dict[str, str] = {}
        for name in QUOTA_HEADERS:
            value = headers.get(name)
            if value is not None:
                out[name] = str(value)[:200]
        return out
    except Exception:  # noqa: BLE001 — diagnostics must not raise
        return {}


def redact_url(url: str) -> str:
    """Strip a URL's query string AND its userinfo, keeping host and port.

    Gemini puts its API key in ``?key=``; a reverse-proxy setup puts a password
    in ``user:pass@``. Either way the host and port are what an operator needs
    from a "server not responding" row, so those survive and the credentials do
    not.

    The same ``_URL_QUERY_RE`` the message path uses, applied where a URL is
    about to be written to a log rather than returned to a client. This is the
    ONE display helper for inference URLs — ``cli.doctor`` and ``infra.doctor``
    both call it, deliberately: two copies of a redaction function is the shape
    where one gets fixed and the other keeps leaking.

    Display only. The REQUEST must still use the real URL — a caller that
    redacted before ``httpx.get`` would connect somewhere else entirely.
    """
    try:
        # ORDER MATTERS, and it is the order the patterns are declared in.
        #
        # Query first: it consumes everything after the ``?``, so the authority
        # that the next two steps match against can no longer contain a ``?`` to
        # trip over. (``?`` is excluded from the userinfo run anyway; doing the
        # query first keeps the two concerns from interacting at all.)
        #
        # Then the schemed userinfo, then the schemeless one. The bare pattern is
        # anchored at ``^`` with a lookahead that rejects anything already opening
        # with a scheme, so a URL the previous step handled is left alone rather
        # than being matched a second time at its new start.
        text = _URL_QUERY_RE.sub(r"\1?<redacted>", str(url))
        text = _URL_USERINFO_RE.sub(r"\1", text)
        return _BARE_USERINFO_RE.sub("", text)
    except Exception:  # noqa: BLE001
        return "<unprintable url>"


def classify_turn_error(exc: BaseException) -> dict[str, Any]:
    """Map a turn-killing exception to a structured, client-renderable payload.

    Returns ``{kind, provider, status, message, hint}``. Never raises — a
    classifier that throws inside an error handler would mask the original
    failure, so every lookup is defensive.
    """
    try:
        names = _exception_names(exc)
        host = _host_of(exc)
        provider = _provider_from_host(host)

        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        status = int(status) if isinstance(status, int) else None
        body = ""
        if response is not None:
            # `.text` on a STREAMED response that was never read raises httpx.ResponseNotRead,
            # and getattr's default does not help: it only covers AttributeError. That exception
            # escaped to the defensive handler at the bottom of this function, which returns
            # KIND_UNKNOWN — so every provider failure in this daemon classified as "unknown",
            # because every provider call is a streaming call (`stream_message` ->
            # `raise_for_status()` on an unread response).
            #
            # The body is optional: it only sharpens BILLING detection. The status alone already
            # decides auth / rate-limit / provider-error, so losing the body must degrade the
            # answer, never discard it.
            try:
                raw = response.text
            except Exception:  # noqa: BLE001 — an unreadable body is not a classification failure
                raw = ""
            if isinstance(raw, str):
                body = raw

        if status is not None:
            lowered = body.lower()
            billing = any(marker in lowered for marker in _BILLING_MARKERS)
            if status == 402 or (billing and status in (400, 402, 403, 429)):
                kind = KIND_BILLING
            elif status in (401, 403):
                kind = KIND_AUTH
            elif status == 429:
                kind = KIND_RATE_LIMIT
            elif status >= 500:
                kind = KIND_PROVIDER_ERROR
            else:
                kind = KIND_BAD_REQUEST
            return {
                "kind": kind,
                "provider": provider,
                "status": status,
                "message": _provider_message(body, exc),
                "hint": _hint(kind, provider, status),
            }

        if names & {"TimeoutException", "ReadTimeout", "ConnectTimeout",
                    "WriteTimeout", "PoolTimeout", "TimeoutError"}:
            kind = KIND_TIMEOUT
        elif names & {"ConnectError", "ConnectionRefusedError", "ConnectionError",
                      "NetworkError", "RemoteProtocolError"}:
            kind = KIND_UNREACHABLE
        else:
            kind = KIND_UNKNOWN

        return {
            "kind": kind,
            "provider": provider,
            "status": None,
            "message": _redact(str(exc))[:_MAX_MESSAGE_CHARS] or type(exc).__name__,
            "hint": _hint(kind, provider, None),
        }
    except Exception:  # defensive; never mask the real error
        try:
            message = _redact(str(exc))[:_MAX_MESSAGE_CHARS]
        except Exception:
            message = type(exc).__name__
        return {
            "kind": KIND_UNKNOWN,
            "provider": "the model provider",
            "status": None,
            "message": message,
            "hint": _hint(KIND_UNKNOWN, "the model provider", None),
        }
