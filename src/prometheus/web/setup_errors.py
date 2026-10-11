"""Every way setup mode can say no, as a stable code (Beacon v0.4, P8 item 7).

Beacon shows a person a plain-English message when setup fails ("that key was not accepted", "nothing answered at
that address"). It writes the words itself, in its own language, **from the code**; the English ``detail`` in the
response is for logs, ``curl`` and clients that have no copy for a code yet. So the code is the contract:

* **one shape**: ``{"error": <code>, "detail": <English>, "fields"?: [<request fields it is about>], ...}``.
  ``fields`` is there when the failure is about particular inputs (``["api_key"]``, ``["slack_bot_token",
  "slack_app_token"]``), so a client can mark the right boxes; it is absent when it is about the request as a whole;
* **one status per code**, held here, so a code never means a 400 on one route and a 409 on another;
* **a code is a promise**: it keeps its name and status when the wording changes, and a new situation gets a new
  code instead of reusing one. Adding to the catalog needs a row in ``docs/contracts/setup-errors.md``
  (``tests/test_setup_error_codes.py`` holds the two together);
* **nothing in ``setup_server.py`` builds an error response by hand**: it calls :func:`refuse`, which only knows
  the catalog, so a failure that has no code cannot be written.

``shared`` marks codes the setup app can also emit from middleware it shares with the full daemon (the host check,
the public ``/api/hello``, the same-Mac pairing route); they are catalogued so a client sees the whole vocabulary,
and are produced and tested where they live.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class SetupError:
    """One catalogued failure: the HTTP *status* it always has and what it *means*, in a sentence."""

    status: int
    meaning: str
    shared: bool = False


CATALOG: dict[str, SetupError] = {
    # ── the request itself ──
    "bad_request": SetupError(400, "The request was not the JSON this route expects, or a field is missing or of "
                                   "the wrong type."),
    "bad_base_url": SetupError(400, "An address has to start with http:// or https://."),
    "unexpected_base_url": SetupError(400, "A cloud provider is called at its own address and takes none of "
                                           "yours."),
    "unsupported_provider": SetupError(400, "That provider is not one setup offers."),
    "bad_model": SetupError(400, "The model name has a control character or is far too long."),
    # ── secrets ──
    "cloud_key_missing": SetupError(400, "A cloud provider needs an API key, and none was sent or is already "
                                         "saved."),
    "invalid_key": SetupError(400, "That cannot be an API key: it is empty, far too long, or has a line break."),
    "invalid_token": SetupError(400, "A messaging token has a line break in it or is far too long."),
    "slack_token_pair_incomplete": SetupError(400, "Slack needs both its bot token and its app token, and only "
                                                   "one was sent."),
    "bad_discord_guild_ids": SetupError(400, "Discord server ids are whole numbers."),
    # ── the machine ──
    "backend_unreachable": SetupError(400, "Nothing at that address answered like an inference server of that "
                                           "kind."),
    "config_write_failed": SetupError(500, "The settings could not be written to disk (permissions, a full disk or "
                                           "a read-only folder)."),
    "internal_error": SetupError(500, "Something unexpected went wrong inside the daemon; the cause is in its "
                                      "log, not in this answer."),
    # ── where setup is ──
    "not_configured": SetupError(409, "Setup cannot finish yet: there is no valid configuration."),
    "setup_mode": SetupError(403, "The daemon is still in setup mode, so only the setup routes exist."),
    "pairing_unavailable": SetupError(403, "A request to join cannot be made until setup has finished."),
    # ── pairing and the setup token ──
    "not_paired": SetupError(401, "Nobody has paired with this daemon yet, so setup calls are refused."),
    "unauthorized": SetupError(401, "The setup token was missing or wrong."),
    "invalid_code": SetupError(401, "The pairing code was wrong."),
    "pairing_used": SetupError(403, "The pairing code was already used."),
    "pairing_expired": SetupError(403, "The pairing code has expired; restart the daemon for a new one."),
    "pairing_locked": SetupError(403, "Too many wrong pairing codes; restart the daemon to try again."),
    # ── produced by middleware and routes setup mode shares ──
    "not_loopback": SetupError(403, "This pairing secret is only accepted from this Mac, over localhost.", True),
    "browser_not_allowed": SetupError(400, "This route is for apps, not for a web page.", True),
    "host_not_allowed": SetupError(403, "The daemon only answers requests addressed to localhost while it is "
                                        "bound to this machine.", True),
    "rate_limited": SetupError(429, "Too many requests from this address; wait the number of seconds given.", True),
}


def refuse(code: str, detail: str, *, fields: list[str] | None = None, **extra: Any) -> JSONResponse:
    """The one way setup mode refuses. Unknown *code* is a ``KeyError``: add it to :data:`CATALOG` (and the
    contract document) first.

    *detail* is the specific English for this occurrence; *fields* names the request fields it is about;
    *extra* is machine data a client may use (``attempts_remaining``).
    """
    entry = CATALOG[code]
    content: dict[str, Any] = {"error": code, "detail": detail}
    if fields:
        content["fields"] = list(fields)
    content.update(extra)
    return JSONResponse(status_code=entry.status, content=content)
