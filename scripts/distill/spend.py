#!/usr/bin/env python3
"""The one shared budget for every paid model call in the distillation project.

Qwen Max is reached only through a PAY-PER-TOKEN Model Studio key,
``QWEN_PAYG_KEY`` (with ``QWEN_PAYG_BASE_URL`` for a workspace key). It is never
the Token Plan key (``QWEN_API_KEY``), which is for interactive use only. Every
paid call goes through this module, so one ledger and one cap cover all of
them: Phase 2 judging (``judge.py --provider max-payg``) and Phase 2b teacher
generation (``teacher_gen.py``). Nothing else may use the key: ``PaygClient``
refuses any tool not in ``ALLOWED_TOOLS``, and no daemon or background job is
one of them.

    python3 spend.py preflight [--model qwen3.8-max]
    python3 spend.py status
    python3 spend.py prices

Standard library only. It needs no Prometheus import and never makes a call
of its own: ``preflight`` checks the key, the host, the price table, the
ledger and the STOP file locally.

API (judge.py, teacher_gen.py)
------------------------------
::

    creds = resolve_payg()                   # KeyRefused unless the key can be used
    ledger = Ledger()                        # ~/distill/spend-ledger.csv, $150 cap, $140 stop
    client = PaygClient(creds, ledger, tool="teacher_gen", phase="2b-teacher",
                        model="qwen3.8-max", phase_budget_usd=60)
    result = client.chat(payload)            # reserve -> POST -> settle

``chat`` takes an OpenAI-compatible chat-completions body. It must set
``max_tokens`` and ``enable_thinking`` (and ``thinking_budget`` when thinking
is on), must not stream, and holds text only. It may use only the fields in
``PAYLOAD_KEYS`` and only ``function`` tools: anything else (``enable_search``,
``search_options``, ``plugins``, a built-in tool) can add billed input or a fee
of its own that the worst case does not cover, so it is refused. ``chat`` returns a
``PaidResult``, or raises ``SpendRefused`` (``StopActive``, ``CapRefused``)
before anything is sent, or ``PaidCallError`` for an attempt that failed (its
cost is already settled in the ledger). It never retries: backoff is the
caller's, and each attempt is its own reservation, under a ``call_id`` the
ledger has never seen (a reused id is refused; leave it out and a fresh one
is drawn). A caller with its own HTTP stack can use ``Ledger.reserve`` /
``Ledger.settle`` directly, with ``worst_case_tokens`` for the reservation
and ``parse_usage`` for the settlement.

The ledger
----------
One CSV, ``~/distill/spend-ledger.csv`` (mode 0600), with the columns
``LEDGER_FIELDS``. It is the only ledger a paid call may be counted in:
``Ledger.reserve`` refuses any other path, so no command-line option, script
variable or second file can start a fresh $150. The cap and the alert can be
set lower (tests do) but never above $150 / $140. Every read-and-append
happens under an exclusive ``flock`` on ``<ledger>.lock``, held only for that
update, so any number of tools and processes share it. The row kind is the
first word of ``note``:

* ``reserve``: written before a call. ``usd`` is the call's worst case: the
  prompt's UTF-8 bytes as a token bound (no token is shorter than a byte),
  plus a fixed allowance for chat framing, plus ``max_tokens`` and the
  thinking budget as output, all at the full input price.
* ``settle``: written after the call, keyed by ``call_id``. ``usd`` is what
  the call cost, from the API's ``usage`` block: uncached input, cached input
  and output, where output counts the reasoning tokens (see ``parse_usage``).
  A call that failed after it may have reached the model (a read timeout, a
  dropped connection, a broken body, a 5xx or 408 status such as a gateway
  timeout) is charged its whole reservation, and so is a reply without a
  usage block. A call the API refused with a 4xx status (other than 408), or
  that never connected, costs nothing.
* ``free``: a local-model call, counted at $0 (judge.py's local judge).
* ``alert``: a stop and its notify (below). The second word says which:
  ``threshold`` (the $140 alert, with its level as the third word),
  ``over-reservation`` or ``renotify``; the fourth is ``notify:sent`` or
  ``notify:<what went wrong>``.

``cumulative_usd`` on each row is the committed total after it: every settled
cost plus every open reservation at its worst case. A process that dies
mid-call leaves its reservation open, so the ledger keeps charging that call
at its worst case. That errs high, never low. A process that is stopped on
purpose (teacher_gen.py on SIGTERM) instead closes its client, waits briefly
for its calls in flight to settle from their usage, and settles the rest as
``error:interrupted charged-reservation`` (``PaygClient.settle_open``).

The cap and the stop
--------------------
* Hard cap (``DEFAULT_CAP_USD``, $150.00): a reservation that would take the
  committed total above it is refused (``CapRefused``), and nothing is sent.
  A caller's per-phase sub-budget (``phase_budget_usd``) is checked the same
  way against that tool and phase's own committed total.
* Alert (``DEFAULT_ALERT_USD``, $140.00): the first time the committed total
  reaches it, the ledger writes the STOP file (``~/distill/spend-ledger.STOP``),
  sends at most five lines through the notify command (default
  ``~/distill/notify.sh``, message on stdin, the line limit as its argument;
  secrets are removed from its environment) and writes an ``alert threshold``
  row. The reservation that crossed stays valid, because it is under the
  cap. Every later reservation, by any tool, is refused while the STOP file
  exists. The alert can fire at a reservation, since reservations count at
  their worst case, so the stop comes a little early rather than late.
* A settled call that cost more than its reservation means the worst-case
  bound did not hold, so the cap cannot be guaranteed. That also writes the
  STOP file and notifies (an ``alert over-reservation`` row). It does not
  use up the $140 alert: that still fires when the total reaches $140.
* A notify that failed (missing command, non-zero exit, timeout) is not a
  sent alert. While the STOP file exists, every refused reservation, the
  judge's start-up check and ``preflight`` send it again (``alert renotify``),
  up to ``NOTIFY_ATTEMPTS`` tries in all, and each refusal says the operator
  was NOT notified until one gets through.
* The $140 alert fires once. To resume on purpose, the operator removes the
  STOP file. The hard cap still holds and the $140 alert is not sent again.

Prices
------
``PRICES`` is keyed by (model, region), with the source page and the date it
was read. A model or region that is not in the table is refused
(``PriceUnknown``). The figures are the Model Studio pay-as-you-go list
prices: one tier (0 < input tokens <= 1M), and one output price for thinking
and non-thinking mode. Where the page lists two deployment scopes for one
region, the higher price is used. The cache-hit price for qwen3.8-max is not
published (the context-cache page says it is not the usual 20% and points to
the console), so cached input is priced as full input. That over-estimates
and never under-bills.

The key
-------
``resolve_payg`` reads ``QWEN_PAYG_KEY`` and ``QWEN_PAYG_BASE_URL`` from the
environment, or else from the Prometheus env file (``PROMETHEUS_ENV_FILE``,
else ``$XDG_CONFIG_HOME/prometheus/env``, else ``~/.config/prometheus/env``).
That file must be owned by the user and not readable by group or others. It
refuses:

* a missing key;
* a ``sk-sp-`` key, or a Token Plan or Coding Plan host (``token-plan.``,
  ``coding-intl.``): those are the subscription plans;
* a key equal to ``QWEN_API_KEY`` (environment or env file);
* a key whose prefix does not match its host. ``sk-ws-`` goes only with a
  workspace host ``ws-<id>.<region>.maas.aliyuncs.com``, which must be set in
  ``QWEN_PAYG_BASE_URL``. ``sk-`` goes only with a shared DashScope host
  (``dashscope-intl``, ``dashscope``, ``dashscope-us``,
  ``cn-hongkong.dashscope``), defaulting to ``dashscope-intl``. Any other
  prefix, or a host that is not one of those, is refused;
* anything but ``https://<host>/compatible-mode/v1``.

The key is never printed, logged, put in an exception message or passed on
a command line. Nor is the base URL, which names the workspace. Requests go
out with no proxy and no redirects, so the Authorization header reaches only
the checked host.

Exit codes (CLI): 0 ok; 2 refused (key, host, price, ledger, STOP).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import http.client
import json
import logging
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

log = logging.getLogger("distill.spend")

DEFAULT_LEDGER = Path("~/distill/spend-ledger.csv")
DEFAULT_NOTIFY = "~/distill/notify.sh"
DEFAULT_CAP_USD = 150.00
DEFAULT_ALERT_USD = 140.00
LOCK_TIMEOUT_S = 120.0
NOTIFY_TIMEOUT_S = 60.0
NOTIFY_MAX_LINES = 5
#: Tries in all (the first plus the retries) to get one stop's notify through.
NOTIFY_ATTEMPTS = 5
EPS = 1e-9
#: The second word of an ``alert`` row.
ALERT_KINDS = ("threshold", "over-reservation", "renotify")

LEDGER_FIELDS = (
    "ts", "tool", "phase", "model", "call_id", "input_tokens", "cached_input_tokens",
    "output_tokens", "usd", "cumulative_usd", "note",
)
ROW_KINDS = ("reserve", "settle", "free", "alert")

KEY_ENV = "QWEN_PAYG_KEY"
BASE_URL_ENV = "QWEN_PAYG_BASE_URL"
TOKEN_PLAN_KEY_ENV = "QWEN_API_KEY"
#: Never handed to a child process (the notify command gets an environment without them).
SECRET_ENV = (KEY_ENV, BASE_URL_ENV, TOKEN_PLAN_KEY_ENV)

#: The only tools that may spend on the key: Phase 2 judging and Phase 2b teacher generation.
#: No daemon or background job is (or may become) one of them.
ALLOWED_TOOLS = frozenset({"judge", "teacher_gen"})

# The worst-case input bound adds these to the request's UTF-8 bytes: chat-template framing per
# message and per tool, a fixed request overhead, and the tool-calling preamble a template adds
# when tools are offered.
FRAMING_TOKENS = 64
REQUEST_OVERHEAD_TOKENS = 256
TOOLS_PREAMBLE_TOKENS = 1024

#: The request fields whose cost the worst case covers. Anything else is refused: Model Studio
#: options such as ``enable_search``, ``search_options`` or ``plugins`` add retrieved text to the
#: billed input or carry a per-call fee that is not in ``usage``.
PAYLOAD_KEYS = frozenset({
    "model", "messages", "max_tokens", "enable_thinking", "thinking_budget", "stream", "n",
    "temperature", "top_p", "top_k", "seed", "stop", "presence_penalty", "frequency_penalty",
    "repetition_penalty", "tools", "tool_choice", "parallel_tool_calls", "response_format",
})

# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

PRICE_SOURCE = "https://www.alibabacloud.com/help/en/model-studio/model-pricing"
CACHE_SOURCE = "https://www.alibabacloud.com/help/en/model-studio/context-cache"
#: "Last Updated" on the pricing page when it was read, and the day it was read.
PRICE_PAGE_UPDATED = "2026-10-29"
PRICE_CHECKED = "2026-10-30"

#: Model Studio region codes (https://www.alibabacloud.com/help/en/model-studio/base-url).
REGION_NAMES = {
    "ap-southeast-1": "Singapore",
    "us-east-1": "US (Virginia)",
    "cn-beijing": "China (Beijing)",
    "cn-hongkong": "China (Hong Kong)",
    "eu-central-1": "Germany (Frankfurt)",
    "ap-northeast-1": "Japan (Tokyo)",
}
#: Shared (any-workspace) pay-as-you-go hosts, by region.
SHARED_HOSTS = {
    "dashscope-intl.aliyuncs.com": "ap-southeast-1",
    "dashscope.aliyuncs.com": "cn-beijing",
    "dashscope-us.aliyuncs.com": "us-east-1",
    "cn-hongkong.dashscope.aliyuncs.com": "cn-hongkong",
}
DEFAULT_SHARED_HOST = "dashscope-intl.aliyuncs.com"
COMPAT_PATH = "/compatible-mode/v1"
_WORKSPACE_HOST = re.compile(r"^ws-[a-z0-9][a-z0-9-]*\.(?P<region>[a-z0-9-]+)\.maas\.aliyuncs\.com$")
#: Subscription-plan hosts (Token Plan, Coding Plan): interactive use only, never here.
PLAN_HOST_MARKERS = ("token-plan", "coding-intl.", "coding.dashscope")
PLAN_KEY_PREFIX = "sk-sp-"
WORKSPACE_KEY_PREFIX = "sk-ws-"
_KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9._-]{8,}")


@dataclass(frozen=True)
class Price:
    """USD per million tokens for one (model, region), one tier."""

    model: str
    region: str
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float
    max_input_tokens: int
    scope: str
    note: str = ""
    source: str = PRICE_SOURCE
    checked: str = PRICE_CHECKED

    def usd(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
        """The cost of one call. *input_tokens* includes *cached_input_tokens*
        (the OpenAI-compatible ``prompt_tokens`` does)."""
        if input_tokens > self.max_input_tokens:
            raise PriceUnknown(
                f"{input_tokens} input tokens is beyond the priced tier for {self.model} "
                f"({self.max_input_tokens})")
        cached = max(0, min(int(cached_input_tokens), int(input_tokens)))
        uncached = max(0, int(input_tokens) - cached)
        return (uncached * self.input_per_m + cached * self.cached_input_per_m
                + max(0, int(output_tokens)) * self.output_per_m) / 1e6


def _prices() -> dict[tuple[str, str], Price]:
    one_tier = 1_000_000  # "0<Token<=1M"
    cache_note = ("cache-hit price not published for this model (context-cache page: not the usual "
                  "20%; see the console): cached input priced as full input")
    # (region, input, output, deployment scope, note)
    rows = [
        ("ap-southeast-1", 2.00, 6.00, "International", ""),
        ("us-east-1", 2.00, 6.00, "US / Global",
         "the page lists $2/$6 (US scope) and $1.65/$4.951 (Global scope); the higher is used"),
        ("cn-beijing", 1.65, 4.951, "Chinese mainland",
         "the page shows '$' for this region without naming the currency"),
        ("cn-hongkong", 1.65, 4.951, "Global", ""),
        ("eu-central-1", 1.65, 4.951, "Global", ""),
        ("ap-northeast-1", 1.65, 4.951, "Global", ""),
    ]
    table: dict[tuple[str, str], Price] = {}
    for model in ("qwen3.8-max", "qwen3.8-max-0902"):
        for region, p_in, p_out, scope, note in rows:
            table[(model, region)] = Price(
                model=model, region=region, input_per_m=p_in, cached_input_per_m=p_in,
                output_per_m=p_out, max_input_tokens=one_tier, scope=scope,
                note="; ".join(n for n in (note, cache_note) if n))
    return table


PRICES: dict[tuple[str, str], Price] = _prices()


def price_for(model: str, region: str) -> Price:
    """The price for (*model*, *region*), or ``PriceUnknown``: exact match only."""
    price = PRICES.get((model, region))
    if price is None:
        raise PriceUnknown(f"no price for model {model!r} in region {region!r}; "
                           f"add it to spend.PRICES from {PRICE_SOURCE} first")
    return price


def highest_price(model: str) -> Price:
    """The most expensive region's price for *model* (estimates made before the region is known)."""
    found = [p for (m, _r), p in PRICES.items() if m == model]
    if not found:
        raise PriceUnknown(f"no price for model {model!r} in any region")
    return max(found, key=lambda p: (p.output_per_m, p.input_per_m))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SpendError(RuntimeError):
    """Base class. Messages never hold the key or the base URL."""


class KeyRefused(SpendError):
    """The pay-per-token key is missing, is the Token Plan's, or does not match its host."""


class PriceUnknown(SpendError):
    """No price for this (model, region), or the request is beyond the priced tier."""


class LedgerError(SpendError):
    """The ledger cannot be read or written as expected."""


class LedgerBusy(LedgerError):
    """The ledger lock was not free within the timeout."""


class SpendRefused(SpendError):
    """A paid call refused before anything was sent."""


class StopActive(SpendRefused):
    """The STOP file is present (or the alert was just reached)."""


class CapRefused(SpendRefused):
    """The reservation would cross the hard cap or the caller's sub-budget."""


class PaidCallError(SpendError):
    """One attempt that failed. Its cost (``usd``) is already settled in the ledger.

    ``kind`` follows Prometheus's turn-error kinds: rate_limit, provider_error,
    timeout, unreachable, auth, billing, bad_request, context_overflow.
    """

    def __init__(self, kind: str, message: str = "", *, status: int | None = None,
                 retry_after: float | None = None, maybe_billed: bool = False,
                 call_id: str = "", usd: float = 0.0) -> None:
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after = retry_after
        self.maybe_billed = maybe_billed
        self.call_id = call_id
        self.usd = usd


# ---------------------------------------------------------------------------
# The key and its host
# ---------------------------------------------------------------------------


def env_file_path(env: Mapping[str, str] | None = None) -> Path:
    """Prometheus's env file: ``PROMETHEUS_ENV_FILE``, else ``$XDG_CONFIG_HOME/prometheus/env``,
    else ``~/.config/prometheus/env`` (the resolution ``prometheus.config.env_file`` uses)."""
    env = os.environ if env is None else env
    if env.get("PROMETHEUS_ENV_FILE"):
        return Path(env["PROMETHEUS_ENV_FILE"]).expanduser()
    xdg = env.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "prometheus" / "env"


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` lines (``export`` and quotes tolerated). Missing file -> {}.

    Refuses a file another user owns or that group or others can read: it holds secrets.
    """
    path = Path(path).expanduser()
    try:
        st = path.stat()
    except FileNotFoundError:
        return {}
    if not stat.S_ISREG(st.st_mode):
        raise KeyRefused(f"the env file {display_path(path)} is not a regular file")
    if st.st_uid != os.getuid():
        raise KeyRefused(f"the env file {display_path(path)} belongs to another user")
    if st.st_mode & 0o077:
        raise KeyRefused(f"the env file {display_path(path)} is readable by others "
                         f"(mode {stat.S_IMODE(st.st_mode):04o}); chmod 600 it")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(raw)
        if m:
            value = m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[m.group(1)] = value
    return values


@dataclass(frozen=True)
class PaygCredentials:
    """A checked key and endpoint. ``repr`` shows neither the key nor the URL."""

    region: str
    host_kind: str  # "workspace" | "shared"
    key_source: str  # "environment" | "env file"
    base_url: str = field(repr=False)
    _key: str = field(repr=False, compare=False)

    def describe(self) -> str:
        return (f"{self.host_kind} pay-as-you-go endpoint in {self.region} "
                f"({REGION_NAMES.get(self.region, '?')}), key from the {self.key_source}")

    def scrub(self, text: str) -> str:
        """*text* with the key and the base URL's host replaced."""
        text = str(text)
        if self._key:
            text = text.replace(self._key, "<key>")
        host = urlsplit(self.base_url).hostname or ""
        if host:
            text = text.replace(host, "<host>")
        return text


def check_endpoint(key: str, base_url: str | None) -> tuple[str, str, str]:
    """(normalized base URL, region, host kind) for *key*, or ``KeyRefused``."""
    if key.startswith(PLAN_KEY_PREFIX):
        raise KeyRefused(f"{KEY_ENV} is a subscription-plan key (sk-sp-): the Token Plan is for "
                         "interactive use only and is never used here")
    if not _KEY_SHAPE.fullmatch(key):
        raise KeyRefused(f"{KEY_ENV} is not a Model Studio key (expected sk-..., no spaces)")
    workspace_key = key.startswith(WORKSPACE_KEY_PREFIX)
    if not base_url:
        if workspace_key:
            raise KeyRefused(f"a workspace key (sk-ws-) needs {BASE_URL_ENV}: its "
                             "https://ws-<id>.<region>.maas.aliyuncs.com/compatible-mode/v1 host")
        base_url = f"https://{DEFAULT_SHARED_HOST}{COMPAT_PATH}"
    parts = urlsplit(base_url.strip())
    host = (parts.hostname or "").lower()
    if any(marker in host for marker in PLAN_HOST_MARKERS):
        raise KeyRefused(f"{BASE_URL_ENV} names a subscription-plan host (Token Plan / Coding Plan): "
                         "never used here")
    if parts.scheme != "https" or not host:
        raise KeyRefused(f"{BASE_URL_ENV} must be an https:// URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise KeyRefused(f"{BASE_URL_ENV} must not carry credentials, a query or a fragment")
    if parts.port not in (None, 443):
        raise KeyRefused(f"{BASE_URL_ENV} must use the default https port")
    if parts.path.rstrip("/") != COMPAT_PATH:
        raise KeyRefused(f"{BASE_URL_ENV} must end in {COMPAT_PATH}")
    ws = _WORKSPACE_HOST.match(host)
    if ws and ws.group("region") in REGION_NAMES:
        region, kind = ws.group("region"), "workspace"
    elif host in SHARED_HOSTS:
        region, kind = SHARED_HOSTS[host], "shared"
    else:
        raise KeyRefused(f"{BASE_URL_ENV}'s host is not a Model Studio pay-as-you-go endpoint "
                         "(ws-<id>.<region>.maas.aliyuncs.com or a shared DashScope host)")
    if workspace_key and kind != "workspace":
        raise KeyRefused("a workspace key (sk-ws-) only works on its workspace host "
                         "(ws-<id>.<region>.maas.aliyuncs.com); the host is a shared one")
    if not workspace_key and kind != "shared":
        raise KeyRefused("a workspace host (ws-<id>...) needs its workspace's sk-ws- key; "
                         f"{KEY_ENV} is a plain sk- key")
    return f"https://{host}{COMPAT_PATH}", region, kind


def resolve_payg(env: Mapping[str, str] | None = None, *, env_file: Path | None = None) -> PaygCredentials:
    """The checked pay-per-token credentials, or ``KeyRefused`` (module docstring)."""
    env = os.environ if env is None else env
    path = Path(env_file).expanduser() if env_file is not None else env_file_path(env)
    from_file = read_env_file(path)

    def pick(name: str) -> tuple[str, str]:
        value = (env.get(name) or "").strip()
        if value:
            return value, "environment"
        value = (from_file.get(name) or "").strip()
        return (value, "env file") if value else ("", "")

    key, source = pick(KEY_ENV)
    if not key:
        raise KeyRefused(f"{KEY_ENV} is not set (environment or {display_path(path)})")
    plan_keys = {v.strip() for v in (env.get(TOKEN_PLAN_KEY_ENV), from_file.get(TOKEN_PLAN_KEY_ENV)) if v}
    if key in plan_keys:
        raise KeyRefused(f"{KEY_ENV} holds the same key as {TOKEN_PLAN_KEY_ENV} (the Token Plan key)")
    base_url, _src = pick(BASE_URL_ENV)
    url, region, kind = check_endpoint(key, base_url or None)
    return PaygCredentials(region=region, host_kind=kind, key_source=source, base_url=url, _key=key)


# ---------------------------------------------------------------------------
# Usage and worst cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int  # billed output, reasoning included
    reasoning_tokens: int = 0
    note: str = ""


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def parse_usage(usage: Any) -> Usage | None:
    """The billed tokens in an OpenAI-compatible ``usage`` block, or None when it is unusable.

    Cached input is ``prompt_tokens_details.cached_tokens`` (DashScope also
    reports a top-level ``cached_tokens``), and it is part of ``prompt_tokens``.
    Output counts the reasoning tokens. When ``total_tokens`` is given, output
    is at least ``total_tokens - prompt_tokens``, which catches a reasoning
    count kept outside ``completion_tokens``. Without it, a reported
    ``reasoning_tokens`` is added on top of ``completion_tokens``. That can
    double-count, but never under-counts.
    """
    if not isinstance(usage, dict):
        return None
    prompt = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _int(usage.get("completion_tokens", usage.get("output_tokens")))
    if prompt is None or completion is None:
        return None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else None
    if cached is None:
        cached = _int(usage.get("cached_tokens")) or 0
    out_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    reasoning = (_int(out_details.get("reasoning_tokens")) if isinstance(out_details, dict) else None) or 0
    total = _int(usage.get("total_tokens"))
    billed_out, note = completion, ""
    if total is not None and total - prompt > billed_out:
        billed_out, note = total - prompt, "output-from-total"
    elif total is None and reasoning:
        billed_out, note = completion + reasoning, "reasoning-added"
    return Usage(input_tokens=prompt, cached_input_tokens=min(cached, prompt), output_tokens=billed_out,
                 reasoning_tokens=reasoning, note=note)


def check_payload(payload: dict[str, Any]) -> None:
    """Refuse a request whose worst case this module cannot bound (``SpendRefused``)."""
    if not isinstance(payload, dict):
        raise SpendRefused("the request body must be a JSON object")
    unknown = sorted(str(k) for k in payload if k not in PAYLOAD_KEYS)
    if unknown:
        raise SpendRefused(f"request field(s) {', '.join(unknown)} are not supported: their cost is not "
                           "bounded here (search, plugins and the like add billed input or a fee of their own)")
    tools = payload.get("tools")
    if tools is not None and (not isinstance(tools, list) or any(
            not isinstance(t, dict) or t.get("type") != "function" for t in tools)):
        raise SpendRefused("tools must be a list of function tools (a built-in tool such as web search "
                           "has a fee of its own)")
    if payload.get("stream"):
        raise SpendRefused("streaming requests are not supported (the usage block is needed)")
    if payload.get("n") not in (None, 1):
        raise SpendRefused("n must be 1")
    max_tokens = payload.get("max_tokens")
    if _int(max_tokens) is None or not max_tokens:
        raise SpendRefused("max_tokens must be set (a positive integer): it bounds the output")
    thinking = payload.get("enable_thinking")
    if not isinstance(thinking, bool):
        raise SpendRefused("enable_thinking must be set (true or false)")
    if thinking and not _int(payload.get("thinking_budget")):
        raise SpendRefused("thinking needs a thinking_budget: it bounds the reasoning tokens")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SpendRefused("messages must be a non-empty list")
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise SpendRefused("only text content is supported (an image's tokens are not "
                                       "bounded by its bytes)")


def worst_case_tokens(payload: dict[str, Any]) -> tuple[int, int]:
    """(input bound, output bound) for a checked request body."""
    check_payload(payload)
    core = {k: payload[k] for k in ("messages", "tools", "tool_choice", "response_format") if k in payload}
    tools = payload.get("tools") or []
    est_in = (len(json.dumps(core, ensure_ascii=False).encode("utf-8"))
              + FRAMING_TOKENS * (len(payload["messages"]) + len(tools) + 1)
              + REQUEST_OVERHEAD_TOKENS + (TOOLS_PREAMBLE_TOKENS if tools else 0))
    est_out = int(payload["max_tokens"]) + (int(payload["thinking_budget"]) if payload["enable_thinking"] else 0)
    return est_in, est_out


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def display_path(path: Path) -> str:
    """*path* with the home directory written as ``~`` (keeps user names out of messages)."""
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if home and (text == home or text.startswith(home + os.sep)) else text


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Totals:
    spent_usd: float = 0.0  # settled
    held_usd: float = 0.0  # open reservations at their worst case
    calls: int = 0  # settled paid calls
    free_calls: int = 0
    open: dict[str, float] = field(default_factory=dict)
    #: The levels of the ``alert threshold`` rows (an over-reservation stop is not one).
    threshold_alerts: list[float] = field(default_factory=list)
    #: The latest stop's notify has not got through yet, after this many tries.
    notify_pending: bool = False
    notify_attempts: int = 0

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self.held_usd

    @property
    def alerted(self) -> bool:
        """True once a threshold alert has fired (at any level)."""
        return bool(self.threshold_alerts)

    def threshold_alerted(self, level: float) -> bool:
        """True when a threshold alert at *level* or above is already recorded."""
        return any(x + EPS >= level for x in self.threshold_alerts)


@dataclass(frozen=True)
class Reservation:
    call_id: str
    tool: str
    phase: str
    model: str
    region: str
    input_tokens: int
    output_tokens: int
    usd: float
    price: Price = field(repr=False)


@dataclass(frozen=True)
class Settlement:
    call_id: str
    usd: float
    cumulative_usd: float
    note: str
    stopped: bool


def _num(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except ValueError:
        raise LedgerError(f"ledger row {row.get('call_id')!r}: {key} {value!r} is not a number") from None
    if not math.isfinite(number):
        raise LedgerError(f"ledger row {row.get('call_id')!r}: {key} is not finite")
    return number


def totals_of(rows: list[dict[str, str]], *, tool: str | None = None, phase: str | None = None) -> Totals:
    """Totals over the ledger rows, optionally for one tool and phase. The alerts are global."""
    t = Totals()
    for row in rows:
        note = row.get("note") or ""
        kind = note.split(" ", 1)[0]
        if kind not in ROW_KINDS:
            raise LedgerError(f"ledger row {row.get('call_id')!r} has an unknown kind {kind!r}")
        if kind == "alert":
            # alert <threshold|over-reservation|renotify> <level|-> notify:<sent|...> <reason>
            parts = note.split(" ", 4) + ["", "", ""]
            if parts[1] == "threshold":
                try:
                    level = float(parts[2])
                except ValueError:
                    level = 0.0   # unreadable: never suppresses a later alert
                t.threshold_alerts.append(level if math.isfinite(level) else 0.0)
            # The tries count from each new stop; a renotify row is another try for the same stop.
            tries = (t.notify_attempts if parts[1] == "renotify" else 0) + 1
            if parts[3] == "notify:sent":
                t.notify_pending, t.notify_attempts = False, 0
            else:
                t.notify_pending, t.notify_attempts = True, tries
            continue
        if (tool is not None and row.get("tool") != tool) or (phase is not None and row.get("phase") != phase):
            continue
        usd = _num(row, "usd")
        if kind == "reserve":
            t.open[row.get("call_id") or ""] = usd
        elif kind == "settle":
            t.open.pop(row.get("call_id") or "", None)
            t.spent_usd += usd
            t.calls += 1
        else:
            t.free_calls += 1
    t.held_usd = sum(t.open.values())
    return t


def _scrubbed_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SECRET_ENV}


def run_notify(command: str, message: str) -> str:
    """Send *message* (at most five lines) to the notify command on stdin, the line limit as its
    argument. Returns ``sent``, ``missing`` or ``failed (...)``; never raises."""
    text = "\n".join(message.strip().splitlines()[:NOTIFY_MAX_LINES]) + "\n"
    path = Path(command).expanduser()
    if not (path.is_file() and os.access(path, os.X_OK)):
        log.warning("notify command %s is missing or not executable; the operator was NOT told",
                    display_path(path))
        return "missing"
    try:
        cp = subprocess.run([str(path), str(NOTIFY_MAX_LINES)], input=text, capture_output=True, text=True,
                            timeout=NOTIFY_TIMEOUT_S, env=_scrubbed_env(), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("notify failed: %s", type(exc).__name__)
        return f"failed ({type(exc).__name__})"
    if cp.returncode != 0:
        log.warning("notify exited %d", cp.returncode)
        return f"failed (exit {cp.returncode})"
    return "sent"


def canonical_ledger() -> Path:
    """The one ledger every paid call is counted in (``~/distill/spend-ledger.csv``, expanded now)."""
    return DEFAULT_LEDGER.expanduser()


def _limit(name: str, value: float, ceiling: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"the {name} must be a positive amount")
    if number > ceiling + EPS:
        raise ValueError(f"the {name} cannot be raised above ${ceiling:.2f} (it can only be lowered)")
    return number


class Ledger:
    """The shared spend ledger, its cap, its alert and its STOP file (module docstring).

    Any path can be read (``totals``, ``status``) and can count free local
    calls, but only the canonical ledger can reserve a paid call. The cap and
    the alert can be lowered, never raised above ``DEFAULT_CAP_USD`` /
    ``DEFAULT_ALERT_USD``. The STOP file sits beside the ledger and has no
    path of its own to change.
    """

    def __init__(self, path: Path | str = DEFAULT_LEDGER, *, cap_usd: float = DEFAULT_CAP_USD,
                 alert_usd: float = DEFAULT_ALERT_USD,
                 notify_cmd: str | None = DEFAULT_NOTIFY, lock_timeout: float = LOCK_TIMEOUT_S,
                 clock: Callable[[], float] = time.time,
                 notifier: Callable[[str, str], str] | None = None) -> None:
        self.cap_usd = cap_usd
        self.alert_usd = alert_usd
        if self.alert_usd > self.cap_usd:
            raise ValueError("the alert threshold must not be above the cap")
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.stop_path = self.path.with_suffix(".STOP")
        self.notify_cmd = notify_cmd
        self.lock_timeout = lock_timeout
        self.clock = clock
        self.notifier = notifier or run_notify

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    @cap_usd.setter
    def cap_usd(self, value: float) -> None:
        self._cap_usd = _limit("cap", value, DEFAULT_CAP_USD)

    @property
    def alert_usd(self) -> float:
        return self._alert_usd

    @alert_usd.setter
    def alert_usd(self, value: float) -> None:
        self._alert_usd = _limit("alert threshold", value, DEFAULT_ALERT_USD)

    def is_canonical(self) -> bool:
        return os.path.realpath(self.path) == os.path.realpath(canonical_ledger())

    def check_paid_path(self) -> None:
        """``LedgerError`` unless this is the canonical ledger: one file holds the one cap and STOP."""
        if not self.is_canonical():
            raise LedgerError(
                f"paid calls are counted only in {display_path(canonical_ledger())}, not in "
                f"{display_path(self.path)}: one ledger holds the one ${DEFAULT_CAP_USD:.2f} cap and its STOP file")

    # -- file plumbing -------------------------------------------------------

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """The exclusive ledger lock, held for one read-and-append. Waits up to ``lock_timeout``."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LedgerBusy(f"the ledger lock {display_path(self.lock_path)} stayed busy "
                                         f"for {self.lock_timeout:g}s") from None
                    time.sleep(0.02)
            yield
        finally:
            os.close(fd)  # releases the lock

    def read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists() or not self.path.stat().st_size:
            return []
        with open(self.path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
                raise LedgerError(f"{display_path(self.path)} has columns {reader.fieldnames}, expected "
                                  f"{list(LEDGER_FIELDS)}: not this ledger (move it aside)")
            return list(reader)

    def _append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or not self.path.stat().st_size
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})
            fh.flush()
            os.fsync(fh.fileno())

    def _row(self, *, tool: str, phase: str, model: str, call_id: str, input_tokens: int = 0,
             cached_input_tokens: int = 0, output_tokens: int = 0, usd: float = 0.0,
             cumulative: float, note: str) -> dict[str, Any]:
        return {"ts": utc_iso(self.clock()), "tool": tool, "phase": phase, "model": model,
                "call_id": call_id, "input_tokens": int(input_tokens),
                "cached_input_tokens": int(cached_input_tokens), "output_tokens": int(output_tokens),
                "usd": f"{usd:.6f}", "cumulative_usd": f"{cumulative:.6f}", "note": note}

    # -- the stop --------------------------------------------------------------

    def stop_reason(self) -> str | None:
        """The STOP file's first line, or None when there is no STOP file."""
        try:
            text = self.stop_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return "the STOP file exists (unreadable)"
        return (text.splitlines() or ["the STOP file exists"])[0]

    def _notify_locked(self, rows_totals: Totals, committed: float, reason: str, *, kind: str,
                       tool: str, phase: str, model: str) -> str:
        """Send the five-line notify and record it as an ``alert <kind>`` row (lock held)."""
        settled, held = rows_totals.spent_usd, committed - rows_totals.spent_usd
        message = "\n".join([
            f"distill: paid Max calls STOPPED at ${committed:.2f} "
            f"(alert ${self.alert_usd:.2f}, cap ${self.cap_usd:.2f})",
            f"reason: {reason}",
            f"last: {tool} / {phase} / {model}",
            f"settled ${settled:.2f}, held for calls in flight ${max(0.0, held):.2f}",
            f"remove {display_path(self.stop_path)} to resume (the cap still holds)",
        ])
        sent = self.notifier(self.notify_cmd, message) if self.notify_cmd else "missing"
        level = f"{self.alert_usd:.2f}" if kind == "threshold" else "-"
        token = "notify:" + (sent.split(" ", 1)[0] or "failed")
        self._append(self._row(tool=tool, phase=phase, model=model, call_id="", cumulative=committed,
                               note=f"alert {kind} {level} {token} {reason} (notify {sent})"))
        if sent != "sent":
            log.warning("the stop notify did not get through (%s): the operator was NOT told", sent)
        return sent

    def _stop_locked(self, rows_totals: Totals, committed: float, reason: str, *, kind: str, tool: str,
                     phase: str, model: str) -> str:
        """Write the STOP file, send the notify and record the alert row (lock held, so exactly
        one caller does it). *kind* is ``threshold`` (the one-time alert) or ``over-reservation``."""
        settled, held = rows_totals.spent_usd, committed - rows_totals.spent_usd
        stop_text = (
            f"{utc_iso(self.clock())} distill spend STOP: {reason}\n"
            f"committed ${committed:.2f} of the ${self.cap_usd:.2f} cap (alert ${self.alert_usd:.2f}); "
            f"settled ${settled:.2f}, held for calls in flight ${max(0.0, held):.2f}\n"
            "Every paid call (judge.py, teacher_gen.py) refuses while this file exists.\n"
            f"Remove it only to resume on purpose: the ${self.cap_usd:.2f} cap still holds and the "
            f"${self.alert_usd:.2f} alert is not sent again.\n")
        self.stop_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.stop_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(stop_text)
        log.warning("spend STOP written: %s", reason)
        return self._notify_locked(rows_totals, committed, reason, kind=kind, tool=tool, phase=phase, model=model)

    def _refuse_if_stopped_locked(self, t: Totals, *, tool: str, phase: str, model: str) -> None:
        """``StopActive`` while the STOP file exists (lock held). A stop whose notify has not got
        through is notified again first, up to ``NOTIFY_ATTEMPTS`` tries in all."""
        reason = self.stop_reason()
        if reason is None:
            return
        told = ""
        if t.notify_pending:
            if t.notify_attempts < NOTIFY_ATTEMPTS:
                sent = self._notify_locked(t, t.committed_usd, f"still stopped: {reason}", kind="renotify",
                                           tool=tool, phase=phase, model=model)
                told = "" if sent == "sent" else f"; the operator was NOT notified (notify {sent})"
            else:
                told = (f"; the operator was NOT notified ({t.notify_attempts} notify tries failed, no more "
                        "are made): tell them yourself")
        raise StopActive(f"paid calls are stopped ({display_path(self.stop_path)}): {reason}{told}")

    def check_open(self, *, tool: str = "spend", phase: str = "", model: str = "") -> None:
        """The checks a paid run makes before it starts: ``LedgerError`` for another ledger than the
        canonical one or an unreadable one, ``StopActive`` while the STOP file exists (notifying
        again when the stop's notify has not got through)."""
        self.check_paid_path()
        with self.locked():
            self._refuse_if_stopped_locked(totals_of(self.read_rows()), tool=tool, phase=phase, model=model)

    # -- reserve / settle --------------------------------------------------------

    def reserve(self, *, tool: str, phase: str, model: str, region: str, input_tokens: int,
                output_tokens: int, phase_budget_usd: float | None = None, call_id: str | None = None,
                note: str = "") -> Reservation:
        """Hold the call's worst case under the cap, or refuse it (``SpendRefused``). Only the
        canonical ledger reserves (``LedgerError`` otherwise), and only under a new ``call_id``."""
        if tool not in ALLOWED_TOOLS:
            raise SpendRefused(f"tool {tool!r} may not spend on the pay-per-token key "
                               f"(allowed: {', '.join(sorted(ALLOWED_TOOLS))})")
        self.check_paid_path()
        price = price_for(model, region)
        worst = price.usd(input_tokens, 0, output_tokens)
        call_id = call_id or uuid.uuid4().hex
        with self.locked():
            rows = self.read_rows()
            t = totals_of(rows)
            self._refuse_if_stopped_locked(t, tool=tool, phase=phase, model=model)
            if any(r.get("call_id") == call_id for r in rows):
                # A reused id would replace an open reservation (a crashed call's hold) in the totals.
                raise LedgerError(f"call_id {call_id!r} is already in the ledger: every attempt needs "
                                  "its own id (leave it out to get a fresh one)")
            committed = t.committed_usd
            if committed + EPS >= self.alert_usd and not t.threshold_alerted(self.alert_usd):
                sent = self._stop_locked(t, committed, f"committed spend reached the ${self.alert_usd:.2f} alert",
                                         kind="threshold", tool=tool, phase=phase, model=model)
                raise StopActive(f"committed ${committed:.2f} reached the ${self.alert_usd:.2f} alert"
                                 + ("" if sent == "sent" else f"; the operator was NOT notified (notify {sent})"))
            if committed + worst > self.cap_usd + EPS:
                raise CapRefused(f"this call could cost up to ${worst:.4f}; ${committed:.4f} of the "
                                 f"${self.cap_usd:.2f} cap is spent or held")
            if phase_budget_usd is not None:
                mine = totals_of(rows, tool=tool, phase=phase).committed_usd
                if mine + worst > float(phase_budget_usd) + EPS:
                    raise CapRefused(f"this call could cost up to ${worst:.4f}; ${mine:.4f} of the "
                                     f"{tool}/{phase} budget of ${float(phase_budget_usd):.2f} is spent or held")
            after = committed + worst
            self._append(self._row(tool=tool, phase=phase, model=model, call_id=call_id,
                                   input_tokens=input_tokens, output_tokens=output_tokens, usd=worst,
                                   cumulative=after, note=" ".join(x for x in ("reserve", region, note) if x)))
            if committed < self.alert_usd <= after + EPS and not t.threshold_alerted(self.alert_usd):
                # This call stays valid (it is under the cap); every later one is refused.
                self._stop_locked(t, after, f"committed spend crossed the ${self.alert_usd:.2f} alert",
                                  kind="threshold", tool=tool, phase=phase, model=model)
        return Reservation(call_id=call_id, tool=tool, phase=phase, model=model, region=region,
                           input_tokens=int(input_tokens), output_tokens=int(output_tokens), usd=worst,
                           price=price)

    def settle(self, res: Reservation, *, usage: Usage | None = None, ok: bool = True,
               maybe_billed: bool = False, error_kind: str | None = None, note: str = "") -> Settlement:
        """Replace the reservation with what the call cost (module docstring)."""
        tags: list[str] = ["ok" if ok else f"error:{error_kind or 'unknown'}"]
        if usage is not None:
            usd = res.price.usd(usage.input_tokens, usage.cached_input_tokens, usage.output_tokens)
            tokens = (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens)
            if usage.note:
                tags.append(usage.note)
        elif ok or maybe_billed:
            usd = res.usd
            tokens = (res.input_tokens, 0, res.output_tokens)
            tags.append("usage-missing charged-reservation" if ok else "charged-reservation")
        else:
            usd, tokens = 0.0, (0, 0, 0)
        over = usd > res.usd + EPS
        if over:
            tags.append("over-reservation")
        if note:
            tags.append(note)
        stopped = False
        with self.locked():
            rows = self.read_rows()
            t = totals_of(rows)
            if res.call_id not in t.open:
                raise LedgerError(f"no open reservation {res.call_id} to settle")
            before = t.committed_usd
            after = before - t.open[res.call_id] + usd
            self._append(self._row(tool=res.tool, phase=res.phase, model=res.model, call_id=res.call_id,
                                   input_tokens=tokens[0], cached_input_tokens=tokens[1],
                                   output_tokens=tokens[2], usd=usd, cumulative=after,
                                   note="settle " + " ".join(tags)))
            if over:
                # Not the $140 alert: that still fires when the total reaches it (after a resume).
                self._stop_locked(t, after, f"call {res.call_id} cost ${usd:.4f}, more than its "
                                  f"${res.usd:.4f} reservation: the cap can no longer be guaranteed",
                                  kind="over-reservation", tool=res.tool, phase=res.phase, model=res.model)
                stopped = True
            elif (after + EPS >= self.alert_usd and not t.threshold_alerted(self.alert_usd)
                  and self.stop_reason() is None):
                self._stop_locked(t, after, f"committed spend reached the ${self.alert_usd:.2f} alert",
                                  kind="threshold", tool=res.tool, phase=res.phase, model=res.model)
                stopped = True
        return Settlement(call_id=res.call_id, usd=usd, cumulative_usd=after,
                          note="settle " + " ".join(tags), stopped=stopped)

    def record_free(self, *, tool: str, phase: str, model: str, input_tokens: int = 0,
                    output_tokens: int = 0, call_id: str | None = None, note: str = "") -> None:
        """Count a local-model call at $0 (no cap, no STOP: it spends nothing)."""
        with self.locked():
            committed = totals_of(self.read_rows()).committed_usd
            self._append(self._row(tool=tool, phase=phase, model=model, call_id=call_id or uuid.uuid4().hex,
                                   input_tokens=input_tokens, output_tokens=output_tokens, usd=0.0,
                                   cumulative=committed, note=" ".join(x for x in ("free", note) if x)))

    def totals(self, *, tool: str | None = None, phase: str | None = None) -> Totals:
        with self.locked():
            return totals_of(self.read_rows(), tool=tool, phase=phase)

    def describe(self, *, tool: str | None = None, phase: str | None = None,
                 phase_budget_usd: float | None = None) -> str:
        t = self.totals()
        text = f"${t.committed_usd:.4f} of ${self.cap_usd:.2f} committed (alert ${self.alert_usd:.2f})"
        if tool and phase:
            mine = self.totals(tool=tool, phase=phase).committed_usd
            text += f"; {tool}/{phase} ${mine:.4f}" + (
                f" of ${phase_budget_usd:.2f}" if phase_budget_usd is not None else "")
        return text


# ---------------------------------------------------------------------------
# The paid client
# ---------------------------------------------------------------------------


@dataclass
class PaidResult:
    call_id: str
    text: str
    reasoning_text: str
    served_model: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    usd: float
    usage_estimated: bool
    message: dict[str, Any] = field(repr=False, default_factory=dict)
    finish_reason: str | None = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would re-send the Authorization header to another URL: refuse it."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def default_opener() -> urllib.request.OpenerDirector:
    """No proxies from the environment, no redirects."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


BILLING_ERROR_CODES = frozenset({"arrearage", "insufficientbalance", "insufficient_quota",
                                 "billing_hard_limit_reached", "accountarrearage"})


def classify_status(status: int, code: str) -> str:
    code_l = code.replace(" ", "").replace("-", "").lower()
    if status == 429:
        return "billing" if code_l in BILLING_ERROR_CODES else "rate_limit"
    if status in (401, 403):
        return "billing" if code_l in BILLING_ERROR_CODES else "auth"
    if status == 402 or code_l in BILLING_ERROR_CODES:
        return "billing"
    if status == 408:
        return "timeout"
    if status == 413:
        return "context_overflow"
    if status >= 500:
        return "provider_error"
    return "bad_request"


def status_maybe_billed(status: int) -> bool:
    """True for an error status that can come after the model generated (5xx, 408): such a call is
    charged its reservation. A 4xx refusal comes before any generation and costs nothing."""
    return status >= 500 or status == 408


def _error_detail(body: bytes) -> tuple[str, str]:
    """(code, message) from an error body, clipped; ("", "") when it is not JSON."""
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return "", ""
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or ""), str(err.get("message") or "")[:200]
    if isinstance(obj, dict):
        return str(obj.get("code") or ""), str(obj.get("message") or "")[:200]
    return "", ""


def _retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("Retry-After") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class PaygClient:
    """Paid chat completions on the pay-per-token key, every call through the ledger.

    One attempt per ``chat``: reserve the worst case, POST, settle from the
    usage block. ``opener`` (an ``OpenerDirector``-like object) is for tests.
    ``chat(..., timeout=)`` bounds one call more tightly than the client's
    ``timeout`` (a caller that must be done by a deadline). ``close`` refuses
    every later call (``SpendRefused``); ``open_calls`` / ``settle_open`` let a
    process that is being stopped wait for its calls in flight and then settle
    the rest as ``interrupted`` (charged their reservation: the request may have
    reached the model), so a stop never leaves a reservation open.
    """

    def __init__(self, creds: PaygCredentials, ledger: Ledger, *, tool: str, phase: str, model: str,
                 phase_budget_usd: float | None = None, timeout: float = 300.0, opener: Any = None) -> None:
        if tool not in ALLOWED_TOOLS:
            raise SpendRefused(f"tool {tool!r} may not spend on the pay-per-token key")
        ledger.check_paid_path()
        self.creds = creds
        self.ledger = ledger
        self.tool = tool
        self.phase = phase
        self.model = model
        self.price = price_for(model, creds.region)
        self.phase_budget_usd = phase_budget_usd
        self.timeout = timeout
        self._opener = opener or default_opener()
        #: held from the closed check to the reservation's registration, so ``close`` never
        #: misses a reservation that is being written
        self._gate = threading.Lock()
        self._closed = ""
        #: this client's reservations whose call has not settled yet
        self._open: dict[str, Reservation] = {}

    @property
    def region(self) -> str:
        return self.creds.region

    def worst_case(self, payload: dict[str, Any]) -> tuple[int, int, float]:
        est_in, est_out = worst_case_tokens(payload)
        return est_in, est_out, self.price.usd(est_in, 0, est_out)

    def chat(self, payload: dict[str, Any], *, call_id: str | None = None,
             timeout: float | None = None) -> PaidResult:
        payload = dict(payload)
        if payload.setdefault("model", self.model) != self.model:
            raise SpendRefused(f"the request names {payload['model']!r}; this client is priced for "
                               f"{self.model!r}")
        if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
            raise ValueError(f"timeout must be > 0 seconds, not {timeout!r}")
        est_in, est_out = worst_case_tokens(payload)
        with self._gate:
            if self._closed:
                raise SpendRefused(f"this paid client is closed ({self._closed}): no new paid call")
            res = self.ledger.reserve(tool=self.tool, phase=self.phase, model=self.model, region=self.region,
                                      input_tokens=est_in, output_tokens=est_out,
                                      phase_budget_usd=self.phase_budget_usd, call_id=call_id)
            self._open[res.call_id] = res
        try:
            return self._send(payload, res, self.timeout if timeout is None else min(self.timeout, timeout))
        except PaidCallError:
            raise
        except BaseException:
            # Anything unexpected after the reservation (a bug, KeyboardInterrupt): the request may
            # have gone out, so it is charged at its reservation before the error propagates.
            with contextlib.suppress(SpendError):
                self.ledger.settle(res, ok=False, maybe_billed=True, error_kind="interrupted")
            raise
        finally:
            self._open.pop(res.call_id, None)

    def close(self, reason: str = "closed", *, wait: float = 5.0) -> None:
        """Refuse every later ``chat`` (``SpendRefused``). Calls already reserved go on. Waits up to
        *wait* seconds for a reservation being written (never forever: a signal handler calls this)."""
        got = self._gate.acquire(timeout=max(0.0, wait))
        try:
            self._closed = reason or "closed"
        finally:
            if got:
                self._gate.release()

    def open_calls(self) -> int:
        """How many of this client's calls are reserved and not settled yet."""
        return len(self._open)

    def settle_open(self, *, error_kind: str = "interrupted", lock_timeout: float = 5.0) -> list[str]:
        """Settle every call of this client still in flight as failed and maybe billed (charged its
        reservation: the request may have reached the model), so a process that is being stopped leaves
        no open reservation behind. Never raises (a signal handler calls this); returns the call ids it
        settled. A call that settles itself meanwhile is left as it settled. Use after ``close``."""
        ledger = copy.copy(self.ledger)  # the same file, a shorter wait for its lock
        ledger.lock_timeout = min(self.ledger.lock_timeout, max(0.0, lock_timeout))
        done: list[str] = []
        for res in list(self._open.values()):
            try:
                ledger.settle(res, ok=False, maybe_billed=True, error_kind=error_kind)
            except SpendError:  # settled by its own call just now, or the lock stayed busy
                continue
            done.append(res.call_id)
            self._open.pop(res.call_id, None)
        return done

    def _fail(self, res: Reservation, kind: str, message: str, *, status: int | None = None,
              retry_after: float | None = None, maybe_billed: bool,
              usage: Usage | None = None) -> PaidCallError:
        settled = self.ledger.settle(res, usage=usage, ok=False, maybe_billed=maybe_billed, error_kind=kind)
        log.warning("paid call %s failed (%s%s); charged $%.4f", res.call_id, kind,
                    f", HTTP {status}" if status else "", settled.usd)
        return PaidCallError(kind, self.creds.scrub(message), status=status, retry_after=retry_after,
                             maybe_billed=maybe_billed, call_id=res.call_id, usd=settled.usd)

    def _send(self, payload: dict[str, Any], res: Reservation, timeout: float) -> PaidResult:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.creds.base_url.rstrip("/") + "/chat/completions", data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "distill-spend/1"})
        request.add_unredirected_header("Authorization", "Bearer " + self.creds._key)
        try:
            resp = self._opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # A 4xx (but 408) is a refusal before anything was generated: $0. A 5xx or a 408 can come
            # after the model ran (a gateway timeout on a long generation): charged its reservation.
            code, message = _error_detail(exc.read(65536) if exc.fp is not None else b"")
            kind = classify_status(exc.code, code)
            raise self._fail(res, kind, f"HTTP {exc.code} {code} {message}".strip(), status=exc.code,
                             retry_after=_retry_after(exc.headers),
                             maybe_billed=status_maybe_billed(exc.code)) from None
        except urllib.error.URLError as exc:
            # Raised while connecting or sending (urllib wraps those): never reached the model.
            reason = exc.reason
            kind = "timeout" if isinstance(reason, TimeoutError) else "unreachable"
            raise self._fail(res, kind, f"{type(reason).__name__} before the request was sent",
                             maybe_billed=False) from None
        except (TimeoutError, http.client.HTTPException, ConnectionError, OSError) as exc:
            # Sent, then no (complete) response: the model may have run and billed it.
            kind = "timeout" if isinstance(exc, TimeoutError) else "unreachable"
            raise self._fail(res, kind, f"{type(exc).__name__} after the request was sent",
                             maybe_billed=True) from None
        try:
            with resp:
                raw = resp.read()
        except (TimeoutError, http.client.HTTPException, OSError) as exc:
            kind = "timeout" if isinstance(exc, TimeoutError) else "unreachable"
            raise self._fail(res, kind, f"{type(exc).__name__} while reading the reply",
                             maybe_billed=True) from None
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise self._fail(res, "provider_error", "the reply is not JSON", maybe_billed=True) from None
        usage = parse_usage(body.get("usage")) if isinstance(body, dict) else None
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._fail(res, "provider_error", "the reply has no choices", maybe_billed=True, usage=usage)
        found = choices[0].get("message")
        reply: dict[str, Any] = found if isinstance(found, dict) else {}
        settled = self.ledger.settle(res, usage=usage, ok=True)
        text = reply.get("content")
        return PaidResult(
            call_id=res.call_id, text=text if isinstance(text, str) else "",
            reasoning_text=str(reply.get("reasoning_content") or ""),
            served_model=body.get("model") if isinstance(body.get("model"), str) else None,
            input_tokens=usage.input_tokens if usage else res.input_tokens,
            cached_input_tokens=usage.cached_input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else res.output_tokens,
            reasoning_tokens=usage.reasoning_tokens if usage else 0,
            usd=settled.usd, usage_estimated=usage is None, message=reply,
            finish_reason=choices[0].get("finish_reason"),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def preflight(*, ledger: Ledger, model: str, env: Mapping[str, str] | None = None,
              env_file: Path | None = None) -> dict[str, Any]:
    """Everything a paid run needs, checked locally. Raises ``SpendError`` on the first problem."""
    creds = resolve_payg(env, env_file=env_file)
    price = price_for(model, creds.region)
    ledger.check_open(tool="preflight", model=model)   # the canonical ledger; STOP (and its notify)
    t = ledger.totals()
    if t.committed_usd + EPS >= ledger.alert_usd:
        raise StopActive(f"committed ${t.committed_usd:.2f} is at or above the ${ledger.alert_usd:.2f} alert")
    notify = Path(ledger.notify_cmd or "").expanduser()
    if not ledger.notify_cmd or not (notify.is_file() and os.access(notify, os.X_OK)):
        raise SpendError(f"the notify command {display_path(notify)} is missing or not executable: the "
                         "alert could not reach the operator")
    return {
        "endpoint": creds.describe(),
        "model": model,
        "price_usd_per_m": {"input": price.input_per_m, "cached_input": price.cached_input_per_m,
                            "output": price.output_per_m, "scope": price.scope,
                            "source": price.source, "checked": price.checked},
        "ledger": display_path(ledger.path),
        "committed_usd": round(t.committed_usd, 4),
        "open_reservations": len(t.open),
        "cap_usd": ledger.cap_usd,
        "alert_usd": ledger.alert_usd,
        "left_before_alert_usd": round(ledger.alert_usd - t.committed_usd, 4),
        "notify": display_path(notify),
    }


def status(ledger: Ledger) -> dict[str, Any]:
    rows = []
    with ledger.locked():
        rows = ledger.read_rows()
    t = totals_of(rows)
    by: dict[str, dict[str, Any]] = {}
    for key in sorted({(r["tool"], r["phase"]) for r in rows if r.get("tool")}):
        s = totals_of(rows, tool=key[0], phase=key[1])
        by[f"{key[0]}/{key[1]}"] = {"spent_usd": round(s.spent_usd, 4), "held_usd": round(s.held_usd, 4),
                                    "paid_calls": s.calls, "free_calls": s.free_calls}
    return {"ledger": display_path(ledger.path), "spent_usd": round(t.spent_usd, 4),
            "held_usd": round(t.held_usd, 4), "committed_usd": round(t.committed_usd, 4),
            "open_reservations": len(t.open), "cap_usd": ledger.cap_usd, "alert_usd": ledger.alert_usd,
            "alerted": t.alerted, "notify_pending": t.notify_pending, "stop": ledger.stop_reason(),
            "by_tool_phase": by}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "status"):
        s = sub.add_parser(name, help=f"the one ledger, {DEFAULT_LEDGER} (not configurable)")
        s.add_argument("--notify-cmd", default=DEFAULT_NOTIFY)
        if name == "preflight":
            s.add_argument("--model", default="qwen3.8-max")
            s.add_argument("--env-file", type=Path, default=None,
                           help="the 0600 env file (default: Prometheus's own)")
    sub.add_parser("prices")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "prices":
            print(json.dumps({f"{m}@{r}": {"input": p.input_per_m, "cached_input": p.cached_input_per_m,
                                          "output": p.output_per_m, "scope": p.scope, "note": p.note}
                              for (m, r), p in sorted(PRICES.items())}, indent=1))
            print(f"source: {PRICE_SOURCE} (page updated {PRICE_PAGE_UPDATED}, read {PRICE_CHECKED}); "
                  f"cache: {CACHE_SOURCE}")
            return 0
        ledger = Ledger(notify_cmd=args.notify_cmd)
        if args.cmd == "status":
            print(json.dumps(status(ledger), indent=1))
            return 0
        print(json.dumps(preflight(ledger=ledger, model=args.model, env_file=args.env_file), indent=1))
        return 0
    except SpendError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
