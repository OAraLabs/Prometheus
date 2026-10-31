#!/usr/bin/env python3
"""The one shared budget and ledger for every batch Max call in the distillation project.

Two routes reach Qwen Max, and every call on either goes through this module,
so one ledger, one STOP file and one notify path cover all of them: Phase 2
judging (``judge.py``) and Phase 2b teacher generation (``teacher_gen.py``).
Nothing else may use either key: both clients refuse any tool not in
``ALLOWED_TOOLS``, and no daemon or background job is one of them.

* The PAY-PER-TOKEN key ``QWEN_PAYG_KEY`` (with ``QWEN_PAYG_BASE_URL`` for a
  workspace key): ``PaygClient``, under the $150 cap below. ``PaygClient``
  never takes the Token Plan key.
* The Token Plan subscription key ``QWEN_API_KEY`` (operator override,
  informed, 2026-10-31): ``TokenPlanClient``, $0 per call, no $ cap and no
  pacing beyond the plan's own limits, only while ``QWEN_PAYG_KEY`` is absent.
  Its guardrails are in "The Token Plan" below.

    python3 spend.py preflight [--model qwen3.8-max]          # the pay-per-token route
    python3 spend.py tokenplan-preflight [--model qwen3.8-max] # the Token Plan route
    python3 spend.py tokenplan-status                         # trial, usage, errors, STOP, pause
    python3 spend.py tokenplan-accept-key                     # after a deliberate key change only
    python3 spend.py which-key                                # payg | token-plan (presence only)
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

The Token Plan
--------------
``QWEN_API_KEY`` (prefix ``sk-sp-``) on
``https://token-plan.<region>.maas.aliyuncs.com/compatible-mode/v1``
(``QWEN_BASE_URL``, default the Singapore host), both read from the
environment or the 0600 env file like the pay-per-token key. The operator
accepted the risk that batch use may breach the plan's terms. Guardrails,
all enforced in ``TokenPlanClient`` before and after every call:

1. Concurrency adapts (``AdaptiveLimiter``): it starts at
   ``TP_START_CONCURRENCY`` (4) calls in flight per process and never goes
   above ``TP_MAX_CONCURRENCY`` (8). A burst of ``TP_BURST_429`` (3) rate-limit
   429s within ``TP_BURST_WINDOW_S`` (60 s) halves it (down to 1); it grows
   back by one after ``TP_GROW_AFTER_OK`` (50) clean calls in a row, at most
   once per ``TP_GROW_MIN_INTERVAL_S`` (5 min). A rate-limit 429 is retried
   with exponential backoff (``TP_BACKOFF_BASE_S`` 5 s doubling to
   ``TP_BACKOFF_MAX_S`` 300 s, 10% jitter, or the server's Retry-After when
   longer, capped at 15 min), up to ``retries_429`` times (default
   ``TP_429_RETRIES``, 6; judge.py sets 0 and backs off itself). Model
   Studio's per-minute token limit (429 ``Throttling.AllocationQuota``
   "Allocated quota exceeded", 429 ``insufficient_quota`` "...check your plan
   and billing details") is such a rate limit. The request ``timeout`` bounds
   each attempt from its send; the wait for a slot, for another thread's probe
   and the 429 backoff are bounded apart, by the client's ``wait_budget``
   (default none: until ``close`` or a STOP). There is no other pacing, no
   quota reserve and no ramp.
2. STOP (``tp_classify`` says ``stop``): any 401 or 403, and any error whose
   code or message names an account-level problem (AccessDenied,
   Unpurchased, InvalidApiKey, suspended, Arrearage, risk control, policy,
   banned, revoked, disabled, overdue, insufficient balance, an expired plan
   or key; "billing" except on a 429; a DataInspectionFailed that is not a
   401/403 only when its message names the account, the subscription, a
   suspension or a ban). An error body that cannot be read is classified on
   its status alone. The STOP line is written before anything else (its
   ledger row, the notify), outside the ledger lock, and the client stays
   stopped: no later call on it, nor any thread waiting for a slot or a probe,
   sends again. The client appends to
   the ledger's STOP file (``~/distill/spend-ledger.STOP``, the one every batch
   tool already honours, so pay-per-token calls stop too), notifies the
   operator with the exact error (HTTP status, error code, message and
   request id, the key scrubbed) and raises ``TokenPlanStop``. The operator is
   told once per STOP file: every call in flight gets the same 403, and each
   adds its STOP line and row but no second notify. Nothing retries around it,
   and nothing here ever removes the STOP file.
3. Quota exhausted (``quota``: a reply saying the plan's quota, package or
   allowance is used up, or a free tier exhausted, as opposed to a
   per-minute rate limit): batch work
   pauses until the reset time the reply gives (an ISO time, naive ones read
   as UTC+8, the plan's region; "in N hours"; or a Retry-After of a minute or
   more), else for ``TP_QUOTA_DEFAULT_PAUSE_S`` (1 h). The pause is the file
   ``~/distill/token-plan.PAUSE``; every Token Plan call refuses with
   ``QuotaPaused`` until it ends. Then exactly one call (across processes: a
   claim in that file) goes out as the probe: success removes the file,
   another quota error extends it. The operator is told once per exhausted
   episode. Not a STOP.
4. Every attempt (429 retries too) is a ledger row whose note starts
   ``token-plan``: ``usd`` 0, the tokens from the usage block, the outcome
   (``ok``, ``error:<kind>``, ``probe``). Those rows count toward no $ cap: the
   pay-per-token $150 cap and $140 alert see only ``reserve`` / ``settle`` rows.
5. Batch tooling only: ``ALLOWED_TOOLS``; nothing here reads or writes the
   daemon's config, and the daemon's own jobs never import this module.
6. ``QWEN_PAYG_KEY`` wins: ``resolve_token_plan`` refuses
   (``TokenPlanSuperseded``) while it is present (environment or env file),
   and the client re-reads the env file before every call (when it changed),
   so a key that appears mid-run stops Token Plan use at the next call and
   for good on that client (an env file others can read is refused at every
   call until it is fixed, then read).
   ``which-key`` tells the runners which route to start.
7. The key's fingerprint (HMAC-SHA256 under a random salt, in
   ``~/distill/token-plan-state.json``, mode 0600) is stored at first use. A
   different key later (at a client's start, a preflight, or an env file
   change mid-run) is a STOP with a notify: a new key may be another
   account. ``tokenplan-accept-key`` records the new fingerprint after a
   deliberate change; it does not remove the STOP file.

The 50-call trial is not a gate (the operator dropped the ramp): it is the
first ``TP_TRIAL_CALLS`` Token Plan attempts in the ledger, which with the
runners' order are Phase 2 judge calls (``judge_run.sh --trial``), and
``tokenplan-status`` reports whether they were clean.

Exit codes (CLI): 0 ok; 2 refused (key, host, price, ledger, STOP); 3
``tokenplan-preflight`` refused because ``QWEN_PAYG_KEY`` is present (use the
pay-per-token route); 4 ``tokenplan-preflight`` while the quota pause stands.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import fcntl
import hashlib
import hmac
import http.client
import json
import logging
import math
import os
import random
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
from datetime import datetime, timedelta, timezone
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
ALERT_KINDS = ("threshold", "over-reservation", "renotify", "token-plan-stop", "token-plan-pause")

LEDGER_FIELDS = (
    "ts", "tool", "phase", "model", "call_id", "input_tokens", "cached_input_tokens",
    "output_tokens", "usd", "cumulative_usd", "note",
)
#: ``token-plan``: one Token Plan attempt ($0, tokens counted; never under the $ cap).
ROW_KINDS = ("reserve", "settle", "free", "alert", "token-plan")

KEY_ENV = "QWEN_PAYG_KEY"
BASE_URL_ENV = "QWEN_PAYG_BASE_URL"
TOKEN_PLAN_KEY_ENV = "QWEN_API_KEY"
TOKEN_PLAN_BASE_URL_ENV = "QWEN_BASE_URL"
#: Never handed to a child process (the notify command gets an environment without them).
SECRET_ENV = (KEY_ENV, BASE_URL_ENV, TOKEN_PLAN_KEY_ENV, TOKEN_PLAN_BASE_URL_ENV)

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
    #: Token Plan attempts ($0): all of them, the ``ok`` ones, and their tokens.
    plan_calls: int = 0
    plan_ok: int = 0
    plan_input_tokens: int = 0
    plan_cached_input_tokens: int = 0
    plan_output_tokens: int = 0

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
            # alert <threshold|over-reservation|renotify|token-plan-*> <level|-> notify:<sent|...> <reason>
            parts = note.split(" ", 4) + ["", "", ""]
            if parts[1] == "token-plan-pause":
                continue  # a pause is not a stop: its notify is tracked in the PAUSE file
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
        if kind == "token-plan":
            t.plan_calls += 1
            t.plan_ok += note.split(" ", 2)[1:2] == ["ok"]
            t.plan_input_tokens += int(_num(row, "input_tokens"))
            t.plan_cached_input_tokens += int(_num(row, "cached_input_tokens"))
            t.plan_output_tokens += int(_num(row, "output_tokens"))
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
             cumulative: float | None, note: str) -> dict[str, Any]:
        """One ledger row. *cumulative* None (a Token Plan row, which changes no $ total) leaves
        ``cumulative_usd`` empty, so the row is appended without reading the whole ledger."""
        return {"ts": utc_iso(self.clock()), "tool": tool, "phase": phase, "model": model,
                "call_id": call_id, "input_tokens": int(input_tokens),
                "cached_input_tokens": int(cached_input_tokens), "output_tokens": int(output_tokens),
                "usd": f"{usd:.6f}", "cumulative_usd": "" if cumulative is None else f"{cumulative:.6f}",
                "note": note}

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
        if kind == "token-plan-stop" or TP_STOP_MARK in reason:
            message = "\n".join([
                "distill: Token Plan batch use STOPPED (every batch Max call refuses)",
                f"error: {reason}"[:900],
                f"last: {tool} / {phase} / {model}",
                "no retry; the daemon and its config are untouched",
                f"remove {display_path(self.stop_path)} to resume (never removed automatically)",
            ])
        else:
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


    # -- the Token Plan's STOP, pause, fingerprint and rows (module docstring) -------

    @property
    def tp_state_path(self) -> Path:
        return self.path.with_name(TP_STATE_NAME)

    @property
    def tp_pause_path(self) -> Path:
        return self.path.with_name(TP_PAUSE_NAME)

    def _write_private(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _tp_state_locked(self) -> dict[str, Any]:
        try:
            data = json.loads(self.tp_state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise LedgerError(f"{display_path(self.tp_state_path)} is unreadable ({type(exc).__name__}): "
                              "the key fingerprint cannot be checked (move it aside only on purpose)") from None
        return data if isinstance(data, dict) else {}

    def tp_state(self) -> dict[str, Any]:
        with self.locked():
            return self._tp_state_locked()

    def _tp_update_state_locked(self, fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        state = self._tp_state_locked()
        fn(state)
        self._write_private(self.tp_state_path, json.dumps(state, indent=1, sort_keys=True) + "\n")
        return state

    def tp_stop(self, *, reason: str, exact: str, tool: str, phase: str, model: str) -> str:
        """Guardrail 2 / 7: append to the STOP file (never removed here), notify the operator with
        the exact error and record an ``alert token-plan-stop`` row. Returns the notify outcome.

        The STOP line is appended first, without the ledger lock (an O_APPEND write), so a busy lock
        or an unreadable ledger never keeps the STOP from being written. The operator is told once
        per STOP file: later hits while the same file stands (every call in flight gets the same
        403) add their line but no second notify."""
        wrote = self._tp_append_stop_line(reason, exact)
        with self.locked():
            return self._tp_stop_notify_locked(reason=reason, exact=exact, wrote=wrote, tool=tool, phase=phase,
                                               model=model)

    def _tp_stop_locked(self, *, reason: str, exact: str, tool: str, phase: str, model: str) -> str:
        wrote = self._tp_append_stop_line(reason, exact)
        return self._tp_stop_notify_locked(reason=reason, exact=exact, wrote=wrote, tool=tool, phase=phase,
                                           model=model)

    def _tp_append_stop_line(self, reason: str, exact: str) -> str:
        """Append one STOP line (lock not needed). Returns "" or why the line could not be written."""
        line = f"{utc_iso(self.clock())} distill spend STOP: {TP_STOP_MARK} {reason}: {exact}"[:1200]
        try:
            existed = self.stop_path.exists()
            self.stop_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.stop_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                if not existed:
                    fh.write("Every batch Max call (judge.py, teacher_gen.py, gen_runner.py; the Token Plan and "
                             "the pay-per-token key) refuses while this file exists.\n"
                             "Nothing removes it automatically. Check the account first; remove it only to "
                             "resume on purpose.\n")
        except OSError as exc:
            log.error("the Token Plan STOP file could not be written (%s): %s", type(exc).__name__, reason)
            return f"the STOP file could not be written ({type(exc).__name__})"
        log.warning("Token Plan STOP written: %s", reason)
        return ""

    def _stop_file_id(self) -> str:
        """The STOP file standing now (inode and first line), or "" when there is none."""
        try:
            st = self.stop_path.stat()
            first = (self.stop_path.read_text(encoding="utf-8").splitlines() or [""])[0]
        except OSError:
            return ""
        return f"{st.st_ino}:{first}"

    def _tp_stop_notify_locked(self, *, reason: str, exact: str, wrote: str, tool: str, phase: str,
                               model: str) -> str:
        state = self._tp_state_locked()
        file_id = self._stop_file_id()
        if file_id and state.get("stop_file_id") == file_id and state.get("stop_notify") == "sent":
            log.warning("Token Plan STOP already notified for this STOP file: no second notify (%s)", reason)
            return "sent"
        t = totals_of(self.read_rows())
        detail = f"{reason}: {exact}" + (f" [{wrote}]" if wrote else "")
        sent = self._notify_locked(t, t.committed_usd, detail, kind="token-plan-stop",
                                   tool=tool, phase=phase, model=model)
        self._tp_update_state_locked(lambda st: st.update(
            stopped_at=utc_iso(self.clock()), stop_reason=detail[:600], stop_notify=sent, stop_file_id=file_id))
        return sent

    def tp_pause_info(self) -> dict[str, Any] | None:
        with self.locked():
            return self._tp_pause_locked()

    def _tp_pause_locked(self) -> dict[str, Any] | None:
        """The PAUSE file's record, or None. An unreadable one counts as a pause of the default
        length from its modification time (fail closed)."""
        try:
            raw = self.tp_pause_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            raw = ""
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or not isinstance(data.get("until"), (int, float)):
                raise ValueError
            return data
        except ValueError:
            try:
                since = self.tp_pause_path.stat().st_mtime
            except OSError:
                since = self.clock()
            return {"until": since + TP_QUOTA_DEFAULT_PAUSE_S, "reason": "unreadable PAUSE file",
                    "notify": "sent", "notify_attempts": 0, "probe": None}

    def _tp_write_pause_locked(self, data: dict[str, Any]) -> None:
        data = dict(data, until_utc=utc_iso(float(data["until"])))
        self._write_private(self.tp_pause_path, json.dumps(data, indent=1, sort_keys=True) + "\n")

    def _tp_pause_notify_locked(self, data: dict[str, Any], *, tool: str, phase: str, model: str) -> None:
        """The pause's one notify (retried while it has not got through, up to ``NOTIFY_ATTEMPTS``)."""
        if data.get("notify") == "sent" or int(data.get("notify_attempts") or 0) >= NOTIFY_ATTEMPTS:
            return
        until = float(data["until"])
        message = "\n".join([
            "distill: Token Plan quota exhausted: batch work PAUSED (not a STOP)",
            f"error: {data.get('reason', '')}"[:900],
            f"paused until {utc_iso(until)} ({data.get('basis', '')})",
            f"last: {tool} / {phase} / {model}",
            "resumes after one clean probe call; told once per exhaustion",
        ])
        sent = self.notifier(self.notify_cmd, message) if self.notify_cmd else "missing"
        data["notify"] = sent
        data["notify_attempts"] = int(data.get("notify_attempts") or 0) + 1
        token = "notify:" + (sent.split(" ", 1)[0] or "failed")
        self._append(self._row(tool=tool, phase=phase, model=model, call_id="", cumulative=None,
                               note=f"alert token-plan-pause - {token} quota paused until {utc_iso(until)} "
                                    f"(notify {sent})"))

    def tp_quota(self, *, exact: str, reset_at: float | None, basis: str, tool: str, phase: str,
                 model: str) -> float:
        """Guardrail 3: start (or extend, after a failed probe) the pause; the operator is told once
        per episode. Returns the time it ends."""
        now = self.clock()
        until = reset_at if reset_at is not None else now + TP_QUOTA_DEFAULT_PAUSE_S
        until = min(max(until, now + TP_QUOTA_MIN_PAUSE_S), now + TP_QUOTA_MAX_PAUSE_S)
        with self.locked():
            data = self._tp_pause_locked() or {"since": now, "since_utc": utc_iso(now), "notify": "",
                                               "notify_attempts": 0, "episode": uuid.uuid4().hex[:12]}
            data.update(until=until, reason=exact[:900], basis=basis, probe=None,
                        probes=int(data.get("probes") or 0))
            self._tp_pause_notify_locked(data, tool=tool, phase=phase, model=model)
            self._tp_write_pause_locked(data)
        log.warning("Token Plan quota exhausted: batch work paused until %s", utc_iso(until))
        return until

    def tp_record(self, *, tool: str, phase: str, model: str, call_id: str, outcome: str,
                  usage: Usage | None = None, error: dict[str, Any] | None = None) -> None:
        """Guardrail 4: one ``token-plan`` row per attempt ($0, tokens counted); an error's
        status, code and message also go to the state file's ``last_errors``."""
        if tool not in ALLOWED_TOOLS:
            raise SpendRefused(f"tool {tool!r} may not use the Token Plan key")
        tokens = (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) if usage else (0, 0, 0)
        with self.locked():
            self._append(self._row(tool=tool, phase=phase, model=model, call_id=call_id,
                                   input_tokens=tokens[0], cached_input_tokens=tokens[1],
                                   output_tokens=tokens[2], usd=0.0, cumulative=None,
                                   note=f"{TOKEN_PLAN_TAG} {outcome}"))
            if error is not None:
                entry = {"ts": utc_iso(self.clock()), "tool": tool, "phase": phase, "outcome": outcome, **error}

                def push(st: dict[str, Any]) -> None:
                    st["last_errors"] = (list(st.get("last_errors") or []) + [entry])[-TP_LAST_ERRORS:]
                self._tp_update_state_locked(push)

    def tp_record_limits(self, *, tool: str, phase: str, snapshot: dict[str, Any]) -> None:
        with self.locked():
            self._tp_update_state_locked(lambda st: st.setdefault("limits", {}).update(
                {f"{tool}/{phase}": dict(snapshot, at=utc_iso(self.clock()))}))

    def tp_check_key(self, creds: "TokenPlanCredentials", *, tool: str, phase: str = "",
                     model: str = "") -> str:
        """Guardrail 7: the key's salted fingerprint is stored at first use; a different key later
        writes the STOP file, notifies, and raises ``TokenPlanStop``. Returns ``recorded`` or ``matches``."""
        with self.locked():
            state = self._tp_state_locked()
            salt = state.get("salt")
            if not (isinstance(salt, str) and state.get("fingerprint")):
                salt = os.urandom(16).hex()
                fp = creds.fingerprint(salt)
                self._tp_update_state_locked(lambda st: st.update(
                    salt=salt, fingerprint=fp, fingerprint_recorded=utc_iso(self.clock())))
                return "recorded"
            if hmac.compare_digest(creds.fingerprint(salt), str(state["fingerprint"])):
                return "matches"
            sent = self._tp_stop_locked(
                reason="key changed",
                exact=(f"{TOKEN_PLAN_KEY_ENV} is not the key first used on "
                       f"{state.get('fingerprint_recorded', '?')} (salted fingerprint mismatch): a new key may "
                       "be another account"),
                tool=tool, phase=phase, model=model)
        raise TokenPlanStop(f"{TOKEN_PLAN_KEY_ENV} changed since its first use: STOP written"
                            + ("" if sent == "sent" else f"; the operator was NOT notified (notify {sent})"))

    def tp_accept_key(self, creds: "TokenPlanCredentials") -> None:
        """Record the current key as the Token Plan key (an operator action after a deliberate
        change). The STOP file is left as it is."""
        with self.locked():
            salt = os.urandom(16).hex()
            fp = creds.fingerprint(salt)
            self._tp_update_state_locked(lambda st: st.update(
                salt=salt, fingerprint=fp, fingerprint_recorded=utc_iso(self.clock()), key_accepted=True))


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
# The Token Plan (operator override, informed, 2026-10-31; module docstring)
# ---------------------------------------------------------------------------

TOKEN_PLAN_TAG = "token-plan"
#: The STOP file's line for a Token Plan stop starts with this (after the time).
TP_STOP_MARK = "token-plan"
TOKEN_PLAN_DEFAULT_HOST = "token-plan.ap-southeast-1.maas.aliyuncs.com"
_TOKEN_PLAN_HOST = re.compile(r"^token-plan\.(?P<region>[a-z0-9-]+)\.maas\.aliyuncs\.com$")
#: The models batch work may ask the Token Plan for.
TOKEN_PLAN_MODELS = frozenset({"qwen3.8-max"})
TP_STATE_NAME = "token-plan-state.json"
TP_PAUSE_NAME = "token-plan.PAUSE"
#: Guardrail 1 (adaptive concurrency, per process). The operator set 4 to start and 8 at most.
TP_START_CONCURRENCY = 4
TP_MAX_CONCURRENCY = 8
#: A burst: this many rate-limit 429s within the window halves the concurrency. One stray 429
#: is ordinary backoff; three in a minute at 4+ calls in flight means the plan's rate limit binds.
TP_BURST_429 = 3
TP_BURST_WINDOW_S = 60.0
#: Grow back slowly: +1 after this many clean calls in a row, at most once per interval.
TP_GROW_AFTER_OK = 50
TP_GROW_MIN_INTERVAL_S = 300.0
#: Exponential backoff on a rate-limit 429 (unbilled: refused before generation).
TP_429_RETRIES = 6
TP_BACKOFF_BASE_S = 5.0
TP_BACKOFF_MAX_S = 300.0
TP_RETRY_AFTER_MAX_S = 900.0
#: Guardrail 3: the pause when the reply gives no reset time, and its bounds.
TP_QUOTA_DEFAULT_PAUSE_S = 3600.0
TP_QUOTA_MIN_PAUSE_S = 60.0
TP_QUOTA_MAX_PAUSE_S = 31 * 86400.0
#: A probe claim older than this is stale (its process died): another call may probe.
TP_PROBE_STALE_S = 900.0
#: A naive reset time in a reply is read in the plan's region's zone (Singapore, UTC+8).
TP_NAIVE_RESET_TZ = timezone(timedelta(hours=8))
TP_TRIAL_CALLS = 50
TP_USAGE_WINDOW_S = 7 * 86400.0
TP_LAST_ERRORS = 10
TP_MESSAGE_MAX = 600


class TokenPlanStop(StopActive):
    """An account-level error or a key change: the STOP file is written and the operator told."""


class TokenPlanSuperseded(SpendRefused):
    """``QWEN_PAYG_KEY`` is present: batch work uses it, and Token Plan batch use stops."""


class QuotaPaused(SpendRefused):
    """The plan's quota is exhausted: batch work waits for ``until`` (a timestamp)."""

    def __init__(self, message: str, *, until: float) -> None:
        super().__init__(message)
        self.until = until


@dataclass(frozen=True)
class TokenPlanCredentials:
    """A checked Token Plan key and endpoint. ``repr`` shows neither the key nor the URL."""

    region: str
    key_source: str  # "environment" | "env file"
    env_file: Path
    base_url: str = field(repr=False)
    _key: str = field(repr=False, compare=False)
    host_kind = "token-plan"

    def describe(self) -> str:
        return (f"Token Plan endpoint in {self.region} ({REGION_NAMES.get(self.region, '?')}), "
                f"key from the {self.key_source}")

    def scrub(self, text: str) -> str:
        text = str(text)
        return text.replace(self._key, "<key>") if self._key else text

    def fingerprint(self, salt: str) -> str:
        return hmac.new(bytes.fromhex(salt), self._key.encode("utf-8"), hashlib.sha256).hexdigest()


def check_token_plan_endpoint(key: str, base_url: str | None) -> tuple[str, str]:
    """(normalized base URL, region) for a Token Plan *key*, or ``KeyRefused``."""
    if not key.startswith(PLAN_KEY_PREFIX):
        raise KeyRefused(f"{TOKEN_PLAN_KEY_ENV} is not a Token Plan key (expected the {PLAN_KEY_PREFIX} prefix)")
    if not _KEY_SHAPE.fullmatch(key):
        raise KeyRefused(f"{TOKEN_PLAN_KEY_ENV} is not a Model Studio key (expected sk-sp-..., no spaces)")
    parts = urlsplit((base_url or f"https://{TOKEN_PLAN_DEFAULT_HOST}{COMPAT_PATH}").strip())
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise KeyRefused(f"{TOKEN_PLAN_BASE_URL_ENV} must be an https:// URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise KeyRefused(f"{TOKEN_PLAN_BASE_URL_ENV} must not carry credentials, a query or a fragment")
    if parts.port not in (None, 443):
        raise KeyRefused(f"{TOKEN_PLAN_BASE_URL_ENV} must use the default https port")
    if parts.path.rstrip("/") != COMPAT_PATH:
        raise KeyRefused(f"{TOKEN_PLAN_BASE_URL_ENV} must end in {COMPAT_PATH}")
    m = _TOKEN_PLAN_HOST.match(host)
    if not m or m.group("region") not in REGION_NAMES:
        raise KeyRefused(f"{TOKEN_PLAN_BASE_URL_ENV}'s host is not a Token Plan host "
                         "(token-plan.<region>.maas.aliyuncs.com)")
    return f"https://{host}{COMPAT_PATH}", m.group("region")


def _env_and_file(env: Mapping[str, str] | None, env_file: Path | None) -> tuple[Mapping[str, str], Path,
                                                                                   dict[str, str]]:
    env = os.environ if env is None else env
    path = Path(env_file).expanduser() if env_file is not None else env_file_path(env)
    return env, path, read_env_file(path)


def payg_key_present(env: Mapping[str, str] | None = None, *, env_file: Path | None = None) -> bool:
    """Guardrail 6: ``QWEN_PAYG_KEY`` is set (non-empty) in the environment or the env file.
    Presence only: the value is not checked or kept."""
    env, _path, from_file = _env_and_file(env, env_file)
    return bool((env.get(KEY_ENV) or "").strip() or (from_file.get(KEY_ENV) or "").strip())


def which_key(env: Mapping[str, str] | None = None, *, env_file: Path | None = None) -> str:
    """``payg`` when ``QWEN_PAYG_KEY`` is present (it always wins), else ``token-plan`` when
    ``QWEN_API_KEY`` is, else ``none``. Presence only."""
    env, _path, from_file = _env_and_file(env, env_file)
    if (env.get(KEY_ENV) or "").strip() or (from_file.get(KEY_ENV) or "").strip():
        return "payg"
    if (env.get(TOKEN_PLAN_KEY_ENV) or "").strip() or (from_file.get(TOKEN_PLAN_KEY_ENV) or "").strip():
        return "token-plan"
    return "none"


def resolve_token_plan(env: Mapping[str, str] | None = None, *,
                       env_file: Path | None = None) -> TokenPlanCredentials:
    """The checked Token Plan credentials. ``TokenPlanSuperseded`` while ``QWEN_PAYG_KEY`` is
    present; ``KeyRefused`` for a missing key, a key that is not ``sk-sp-``, or a host that is not
    a Token Plan host."""
    env, path, from_file = _env_and_file(env, env_file)
    if (env.get(KEY_ENV) or "").strip() or (from_file.get(KEY_ENV) or "").strip():
        raise TokenPlanSuperseded(f"{KEY_ENV} is present: batch work uses the pay-per-token key, and "
                                  "Token Plan batch use stops")

    def pick(name: str) -> tuple[str, str]:
        value = (env.get(name) or "").strip()
        if value:
            return value, "environment"
        value = (from_file.get(name) or "").strip()
        return (value, "env file") if value else ("", "")

    key, source = pick(TOKEN_PLAN_KEY_ENV)
    if not key:
        raise KeyRefused(f"{TOKEN_PLAN_KEY_ENV} is not set (environment or {display_path(path)})")
    base_url, _src = pick(TOKEN_PLAN_BASE_URL_ENV)
    url, region = check_token_plan_endpoint(key, base_url or None)
    return TokenPlanCredentials(region=region, key_source=source, env_file=path, base_url=url, _key=key)


# -- classifying an error ------------------------------------------------------------

_W = r"[\s._-]*"
#: Account-level words (guardrail 2). Any of them in an error's code or message is a STOP.
TP_STOP_RE = re.compile(
    r"access" + _W + r"denied|unpurchased|not" + _W + r"(?:purchased|activated|authori[sz]ed)"
    r"|invalid" + _W + r"api" + _W + r"key|api" + _W + r"key[^.]{0,40}(?:invalid|revoked|disabled|expired)"
    r"|suspend|arrearage|overdue|(?<![a-z])risk|policy|(?<![a-z])bann?ed(?![a-z])|revoked|disabled"
    r"|forbidden|unauthori[sz]ed|insufficient" + _W + r"balance"
    r"|(?:subscription|plan|account|key)[^.]{0,40}expired|expired[^.]{0,40}(?:subscription|plan|account|key)",
    re.I)
#: DataInspectionFailed is per request (content moderation) unless its message names the account.
TP_ACCOUNT_RE = re.compile(r"account|subscription|suspend|(?<![a-z])bann?ed(?![a-z])|revoked|disabled|terms",
                           re.I)
#: "billing" alone is an account word except on a 429: Model Studio's TPM/TPS limit (429
#: insufficient_quota) says "please check your plan and billing details".
TP_BILLING_RE = re.compile(r"billing", re.I)
#: Wording that says the plan itself is used up (not a per-minute limit).
TP_EXHAUSTED_RE = re.compile(
    r"exhaust|used" + _W + r"up|deplet|run" + _W + r"out|ran" + _W + r"out|no" + _W + r"remaining"
    r"|free" + _W + r"(?:tier|allocated)|resets?" + _W + r"(?:at|on|in|after)", re.I)
#: A per-minute rate limit, never an exhausted plan.
TP_RATE_RE = re.compile(r"rate" + _W + r"limit|ratequota|per" + _W + r"(?:minute|second)|too many requests"
                        r"|(?<![a-z])(?:rpm|tpm|qps)(?![a-z])", re.I)
#: The plan's quota (package, allowance) is used up: pause until it resets (guardrail 3).
TP_QUOTA_RE = re.compile(
    r"(?:quota|token" + _W + r"plan|package|allowance|credit|usage" + _W + r"limit)[^.]{0,60}"
    r"(?:exhaust|used" + _W + r"up|deplet|run" + _W + r"out|ran" + _W + r"out|exceed|reached|insufficient"
    r"|no" + _W + r"remaining)"
    r"|(?:exhaust|used" + _W + r"up|deplet|run" + _W + r"out|ran" + _W + r"out|no" + _W + r"remaining)"
    r"[^.]{0,60}(?:quota|plan|package|allowance|credit)"
    r"|free" + _W + r"tier" + _W + r"only",
    re.I)


def tp_classify(status: int, code: str, message: str) -> str:
    """A Token Plan error: ``stop`` (account-level, guardrail 2), ``quota`` (exhausted, guardrail 3),
    ``rate_limit`` (backoff, guardrail 1), else spend.py's kinds (provider_error, timeout,
    context_overflow, bad_request)."""
    code, message = str(code or ""), str(message or "")
    text = f"{code} {message}"
    if status in (401, 402, 403):
        return "stop"  # any 401/403, DataInspectionFailed included (guardrail 2)
    norm = re.sub(r"[\s_-]", "", code).lower()
    if norm.replace(".", "").startswith("datainspectionfailed"):
        return "stop" if TP_ACCOUNT_RE.search(message) else "bad_request"
    if TP_STOP_RE.search(text) or (status != 429 and TP_BILLING_RE.search(text)):
        return "stop"
    if status == 429 and (norm == "insufficientquota" or norm.startswith("throttling.allocationquota")):
        # Model Studio's TPM/TPS limit ("Allocated quota exceeded", "You exceeded your current quota,
        # please check your plan and billing details"): a rate limit, unless it says the plan is used up
        return "quota" if TP_EXHAUSTED_RE.search(message) else "rate_limit"
    if not TP_RATE_RE.search(text) and TP_QUOTA_RE.search(text):
        return "quota"
    if status == 429:
        return "rate_limit"
    if 400 <= status < 500 and re.search(r"(?<![a-z])account(?![a-z])", text, re.I):
        return "stop"
    if status == 408:
        return "timeout"
    if status == 413:
        return "context_overflow"
    if status >= 500 or status == 200:
        return "provider_error"
    return "bad_request"


_ISO_RESET = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)(?:\.\d+)?\s*(Z|UTC|GMT|[+-]\d{2}:?\d{2})?",
                        re.I)
_IN_RESET = re.compile(r"(?:in|after)\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr|day)s?", re.I)


def tp_reset_time(message: str, retry_after: float | None, now: float) -> tuple[float | None, str]:
    """(the reset time a quota reply gives, how it was read), or (None, why not)."""
    m = _ISO_RESET.search(message or "")
    if m:
        zone = (m.group(3) or "").upper()
        text = f"{m.group(1)}T{m.group(2)}"
        try:
            when = datetime.fromisoformat(text)
            if zone in ("Z", "UTC", "GMT"):
                when = when.replace(tzinfo=timezone.utc)
            elif zone:
                when = datetime.fromisoformat(text + (zone if ":" in zone else zone[:3] + ":" + zone[3:]))
            else:
                when = when.replace(tzinfo=TP_NAIVE_RESET_TZ)
            ts = when.timestamp()
            if ts > now:
                return ts, "reset time from the reply" + ("" if zone else " (read as UTC+8)")
        except ValueError:
            pass
    m = _IN_RESET.search(message or "")
    if m:
        unit = m.group(2).lower()
        mult = 1 if unit.startswith("sec") else 60 if unit.startswith("min") else 86400 if unit == "day" else 3600
        return now + float(m.group(1)) * mult, "reset delay from the reply"
    if retry_after is not None and retry_after >= 60:
        return now + min(float(retry_after), TP_QUOTA_MAX_PAUSE_S), "the reply's Retry-After"
    return None, f"no reset time in the reply: {int(TP_QUOTA_DEFAULT_PAUSE_S // 60)} min, then one probe call"


def _tp_error_detail(body: bytes) -> tuple[str, str, str]:
    """(code, message, request id) from an error body; the message up to ``TP_MESSAGE_MAX``."""
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        text = body.decode("utf-8", "replace").strip()
        return "", text[:TP_MESSAGE_MAX], ""
    if not isinstance(obj, dict):
        return "", str(obj)[:TP_MESSAGE_MAX], ""
    inner = obj.get("error")
    err: dict[str, Any] = inner if isinstance(inner, dict) else obj
    code = str(err.get("code") or err.get("type") or obj.get("code") or "")
    message = str(err.get("message") or obj.get("message") or "")
    rid = str(obj.get("request_id") or obj.get("id") or err.get("request_id") or "")
    return code, message[:TP_MESSAGE_MAX], rid


# -- adaptive concurrency (guardrail 1) -----------------------------------------------


class AdaptiveLimiter:
    """At most ``limit`` calls in flight. A burst of rate-limit 429s halves the limit; clean calls
    grow it back by one, slowly. Thread-safe; one per client (per process)."""

    def __init__(self, start: int = TP_START_CONCURRENCY, maximum: int = TP_MAX_CONCURRENCY, *,
                 minimum: int = 1, burst: int = TP_BURST_429, burst_window: float = TP_BURST_WINDOW_S,
                 grow_after: int = TP_GROW_AFTER_OK, grow_interval: float = TP_GROW_MIN_INTERVAL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not 1 <= minimum <= start <= maximum <= TP_MAX_CONCURRENCY:
            raise ValueError(f"concurrency must satisfy 1 <= start ({start}) <= max ({maximum}) <= "
                             f"{TP_MAX_CONCURRENCY}")
        self.limit, self.minimum, self.maximum = int(start), int(minimum), int(maximum)
        self.burst, self.burst_window = int(burst), float(burst_window)
        self.grow_after, self.grow_interval = int(grow_after), float(grow_interval)
        self.clock = clock
        self.active = 0
        self._cond = threading.Condition()
        self._r429: collections.deque[float] = collections.deque()
        self._ok_streak = 0
        self._changed_at = clock()
        self.changes: list[tuple[str, int]] = []

    def acquire(self, timeout: float | None = None) -> bool:
        with self._cond:
            ok = self._cond.wait_for(lambda: self.active < self.limit, timeout=timeout)
            if ok:
                self.active += 1
            return bool(ok)

    def release(self) -> None:
        with self._cond:
            self.active = max(0, self.active - 1)
            self._cond.notify_all()

    def on_rate_limit(self) -> bool:
        """One rate-limit 429. True when it completed a burst and the limit was halved."""
        with self._cond:
            now = self.clock()
            self._ok_streak = 0
            self._r429.append(now)
            while self._r429 and self._r429[0] < now - self.burst_window:
                self._r429.popleft()
            if len(self._r429) < self.burst:
                return False
            self._r429.clear()
            new = max(self.minimum, self.limit // 2)
            changed = new != self.limit
            self.limit, self._changed_at = new, now
            self.changes.append(("halve", new))
            return changed

    def on_success(self) -> bool:
        """One clean call. True when the limit grew."""
        with self._cond:
            now = self.clock()
            self._ok_streak += 1
            if (self._ok_streak >= self.grow_after and self.limit < self.maximum
                    and now - self._changed_at >= self.grow_interval):
                self.limit += 1
                self._ok_streak, self._changed_at = 0, now
                self.changes.append(("grow", self.limit))
                self._cond.notify_all()
                return True
            return False

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {"limit": self.limit, "max": self.maximum, "min": self.minimum, "active": self.active}


class _RateLimited(Exception):
    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TokenPlanClient:
    """Batch chat completions on the Token Plan key, every guardrail enforced per call (module
    docstring). The same interface as ``PaygClient`` (``chat``, ``close``, ``open_calls``,
    ``settle_open``, ``ledger``, ``model``, ``region``), so judge.py and teacher_gen.py use either.

    ``chat`` returns a ``PaidResult`` (``usd`` 0) or raises: ``TokenPlanStop`` (account-level
    error or key change: STOP written, operator told), ``QuotaPaused`` (the plan's quota),
    ``TokenPlanSuperseded`` (``QWEN_PAYG_KEY`` appeared), ``StopActive`` (the STOP file),
    ``SpendRefused`` (closed, a bad request body), or ``PaidCallError`` for a failed attempt
    (``rate_limit`` once its backoff retries are used up)."""

    price = None

    def __init__(self, creds: TokenPlanCredentials, ledger: Ledger, *, tool: str, phase: str, model: str,
                 timeout: float = 300.0, opener: Any = None, start_concurrency: int = TP_START_CONCURRENCY,
                 max_concurrency: int = TP_MAX_CONCURRENCY, retries_429: int = TP_429_RETRIES,
                 backoff_base: float = TP_BACKOFF_BASE_S, backoff_max: float = TP_BACKOFF_MAX_S,
                 sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None,
                 limiter: AdaptiveLimiter | None = None, wait_budget: float | None = None) -> None:
        if tool not in ALLOWED_TOOLS:
            raise SpendRefused(f"tool {tool!r} may not use the Token Plan key "
                               f"(allowed: {', '.join(sorted(ALLOWED_TOOLS))})")
        if model not in TOKEN_PLAN_MODELS:
            raise SpendRefused(f"model {model!r} is not one batch work asks the Token Plan for "
                               f"({', '.join(sorted(TOKEN_PLAN_MODELS))})")
        ledger.check_paid_path()
        self.creds = creds
        self.ledger = ledger
        self.tool = tool
        self.phase = phase
        self.model = model
        self.region = creds.region
        self.timeout = timeout
        #: how long one ``chat`` may wait for a slot, the probe and 429 backoff, apart from the
        #: request's own ``timeout`` (None: no bound; ``close`` and a STOP still end the wait)
        self.wait_budget = wait_budget
        self.retries_429 = max(0, int(retries_429))
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.limiter = limiter or AdaptiveLimiter(int(start_concurrency), int(max_concurrency))
        self._opener = opener or default_opener()
        self._owner = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._gate = threading.Lock()
        self._closed = ""
        self._open: dict[str, str] = {}
        self._probe_cond = threading.Condition()
        self._probing = False
        #: a STOP or the pay-per-token key's arrival: terminal for this client (every later call raises it)
        self._terminal: SpendRefused | None = None
        self._env_sig = self._stat_env()
        # guardrail 7 at the client's start (stores the fingerprint at first use)
        ledger.tp_check_key(creds, tool=tool, phase=phase, model=model)
        self._record_limits()

    # -- guards -------------------------------------------------------------------------

    def _stat_env(self) -> tuple[int, int, int] | None:
        try:
            st = self.creds.env_file.stat()
        except OSError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    def _recheck_env(self) -> None:
        """Guardrails 6 and 7 mid-run: when the env file changed, a ``QWEN_PAYG_KEY`` there stops
        Token Plan use; a different ``QWEN_API_KEY`` there (for a key read from it) is a STOP."""
        sig = self._stat_env()
        if sig == self._env_sig:
            return
        # KeyRefused for a file others can read; the signature is kept only once the file was read,
        # so a refused read is tried again at every call until the file is fixed
        values = read_env_file(self.creds.env_file)
        self._env_sig = sig
        if (values.get(KEY_ENV) or "").strip():
            self._terminal = TokenPlanSuperseded(f"{KEY_ENV} appeared in the env file: batch work moves to the "
                                                 "pay-per-token key; Token Plan batch use stops")
            raise self._terminal
        if self.creds.key_source == "env file":
            current = (values.get(TOKEN_PLAN_KEY_ENV) or "").strip()
            if current != self.creds._key:
                self._terminal = TokenPlanStop(f"{TOKEN_PLAN_KEY_ENV} changed mid-run: STOP written")
                sent = self.ledger.tp_stop(
                    reason="key changed",
                    exact=(f"{TOKEN_PLAN_KEY_ENV} in the env file changed or was removed mid-run: a new key "
                           "may be another account"),
                    tool=self.tool, phase=self.phase, model=self.model)
                if sent != "sent":
                    self._terminal = TokenPlanStop(f"{TOKEN_PLAN_KEY_ENV} changed mid-run: STOP written; the "
                                                   f"operator was NOT notified ({sent})")
                raise self._terminal

    def _check_stopped(self) -> None:
        """This client's terminal state, then the STOP file (``StopActive``)."""
        if self._terminal is not None:
            raise type(self._terminal)(str(self._terminal))
        if self.ledger.stop_reason() is not None:
            self.ledger.check_open(tool=self.tool, phase=self.phase, model=self.model)  # raises StopActive

    def _guard(self, deadline: float | None) -> bool:
        """Before every attempt: the env file, the STOP file, the quota pause. Returns True when this
        attempt is the one probe call after a pause. The STOP is checked on every pass, so a thread
        that waited on another thread's probe never sends after that probe wrote a STOP."""
        self._check_closed()
        self._check_stopped()
        self._recheck_env()
        while True:
            self._check_stopped()
            with self.ledger.locked():
                pause = self.ledger._tp_pause_locked()
                if pause is None:
                    return False
                now = self.ledger.clock()
                until = float(pause["until"])
                if now < until:
                    if pause.get("notify") != "sent" and int(pause.get("notify_attempts") or 0) < NOTIFY_ATTEMPTS:
                        self.ledger._tp_pause_notify_locked(pause, tool=self.tool, phase=self.phase,
                                                            model=self.model)
                        self.ledger._tp_write_pause_locked(pause)
                    raise QuotaPaused(f"the Token Plan quota is exhausted: batch work is paused until "
                                      f"{utc_iso(until)} ({pause.get('basis', '')})", until=until)
                probe = pause.get("probe") if isinstance(pause.get("probe"), dict) else None
                fresh = probe is not None and float(probe.get("claimed_at") or 0) > now - TP_PROBE_STALE_S
                if probe is not None and fresh and probe.get("owner") != self._owner:
                    raise QuotaPaused("the quota pause has ended and another process's probe call is in "
                                      "flight", until=float(probe.get("claimed_at") or now) + TP_PROBE_STALE_S)
                with self._probe_cond:
                    if not self._probing:
                        self._probing = True
                        pause["probe"] = {"owner": self._owner, "claimed_at": now, "claimed_utc": utc_iso(now)}
                        pause["probes"] = int(pause.get("probes") or 0) + 1
                        self.ledger._tp_write_pause_locked(pause)
                        return True
            # another thread of this process is probing: wait for its outcome, then look again
            with self._probe_cond:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise PaidCallError("timeout", "waited for the probe call past the wait budget",
                                        maybe_billed=False)
                self._probe_cond.wait_for(lambda: not self._probing or bool(self._closed),
                                          timeout=1.0 if left is None else min(1.0, left))
            self._check_closed()

    def _end_probe(self, is_probe: bool, *, ok: bool) -> None:
        if not is_probe:
            return
        try:
            with self.ledger.locked():
                pause = self.ledger._tp_pause_locked()
                if pause is not None:
                    probe = pause.get("probe") if isinstance(pause.get("probe"), dict) else {}
                    if ok:
                        with contextlib.suppress(FileNotFoundError):
                            self.ledger.tp_pause_path.unlink()
                        log.warning("Token Plan quota pause over: the probe call succeeded")
                    elif probe and probe.get("owner") == self._owner:
                        pause["probe"] = None
                        self.ledger._tp_write_pause_locked(pause)
        finally:
            with self._probe_cond:
                self._probing = False
                self._probe_cond.notify_all()

    def _record_limits(self) -> None:
        with contextlib.suppress(SpendError, OSError):
            self.ledger.tp_record_limits(tool=self.tool, phase=self.phase, snapshot=self.limiter.snapshot())

    # -- the call -----------------------------------------------------------------------

    def chat(self, payload: dict[str, Any], *, call_id: str | None = None,
             timeout: float | None = None) -> PaidResult:
        payload = dict(payload)
        if payload.setdefault("model", self.model) != self.model:
            raise SpendRefused(f"the request names {payload['model']!r}; this client asks for {self.model!r}")
        if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
            raise ValueError(f"timeout must be > 0 seconds, not {timeout!r}")
        check_payload(payload)  # bounded output, text only, no search or plugins
        self._check_closed()
        self._check_stopped()
        call_id = call_id or uuid.uuid4().hex
        # The request timeout bounds each attempt from its send; the wait for a slot, the probe and
        # the 429 backoff are bounded apart, by ``wait_budget`` (None: until close or a STOP).
        per_attempt = self.timeout if timeout is None else min(self.timeout, timeout)
        deadline = None if self.wait_budget is None else time.monotonic() + self.wait_budget
        while not self.limiter.acquire(timeout=0.5 if deadline is None
                                       else max(0.0, min(0.5, deadline - time.monotonic()))):
            self._check_closed()
            self._check_stopped()
            if deadline is not None and time.monotonic() >= deadline:
                raise PaidCallError("timeout", "no concurrency slot within the wait budget", call_id=call_id,
                                    maybe_billed=False)
        self._open[call_id] = call_id
        try:
            attempt = 0
            while True:
                is_probe = self._guard(deadline)
                attempt_id = call_id if attempt == 0 else f"{call_id}-r{attempt}"
                try:
                    result = self._send(payload, attempt_id, per_attempt, is_probe)
                except _RateLimited as rl:
                    self._end_probe(is_probe, ok=False)
                    if self.limiter.on_rate_limit():
                        log.warning("Token Plan 429 burst: concurrency halved to %d", self.limiter.limit)
                        self._record_limits()
                    delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
                    delay += self._rng.uniform(0, delay * 0.1)
                    if rl.retry_after:
                        delay = max(delay, min(float(rl.retry_after), TP_RETRY_AFTER_MAX_S))
                    if attempt >= self.retries_429 or (deadline is not None
                                                        and time.monotonic() + delay >= deadline):
                        raise PaidCallError("rate_limit", str(rl), status=429, retry_after=rl.retry_after,
                                            call_id=call_id) from None
                    log.warning("Token Plan rate limit (attempt %d); backing off %.0fs", attempt + 1, delay)
                    self._sleep(delay)
                    self._check_closed()
                    attempt += 1
                    continue
                except BaseException:
                    self._end_probe(is_probe, ok=False)
                    raise
                self._end_probe(is_probe, ok=True)
                if self.limiter.on_success():
                    self._record_limits()
                return result
        finally:
            self._open.pop(call_id, None)
            self.limiter.release()

    def _exact(self, status: int | None, code: str, message: str, rid: str) -> str:
        text = (f"HTTP {status} " if status is not None else "") + (f"{code}: " if code else "") + message
        if rid:
            text += f" (request_id {rid})"
        return self.creds.scrub(text.strip())

    def _fail(self, call_id: str, kind: str, message: str, *, status: int | None = None, code: str = "",
              rid: str = "", retry_after: float | None = None, maybe_billed: bool = False,
              is_probe: bool = False) -> BaseException:
        """Record the attempt and return the exception to raise (after the STOP or the pause when the
        error calls for one)."""
        exact = self._exact(status, code, message, rid)
        outcome = f"error:{kind}" + (f" http:{status}" if status is not None else "") + (" probe" if is_probe else "")
        error = {"kind": kind, "status": status, "code": code,
                 "message": self.creds.scrub(message)[:TP_MESSAGE_MAX], "request_id": rid}
        if kind == "stop":
            # the STOP comes first: neither a busy ledger lock nor an unreadable ledger gates it
            self._terminal = TokenPlanStop(f"the Token Plan refused the key or the account: {exact}; STOP written")
            try:
                sent = self.ledger.tp_stop(reason=f"account-level error (HTTP {status})", exact=exact,
                                           tool=self.tool, phase=self.phase, model=self.model)
            except (SpendError, OSError) as exc:
                sent = f"failed ({type(exc).__name__}: {exc})"
            try:
                self.ledger.tp_record(tool=self.tool, phase=self.phase, model=self.model, call_id=call_id,
                                      outcome=outcome, error=error)
            except (SpendError, OSError) as exc:
                log.error("the Token Plan STOP's ledger row was not written (%s)", type(exc).__name__)
            if sent != "sent":
                self._terminal = TokenPlanStop(f"{self._terminal}; the operator was NOT notified (notify {sent})")
            return self._terminal
        self.ledger.tp_record(tool=self.tool, phase=self.phase, model=self.model, call_id=call_id, outcome=outcome,
                              error=error)
        if kind == "quota":
            reset, basis = tp_reset_time(message, retry_after, self.ledger.clock())
            until = self.ledger.tp_quota(exact=exact, reset_at=reset, basis=basis, tool=self.tool,
                                         phase=self.phase, model=self.model)
            return QuotaPaused(f"the Token Plan quota is exhausted ({exact}): batch work paused until "
                               f"{utc_iso(until)}", until=until)
        if kind == "rate_limit":
            return _RateLimited(exact, retry_after)
        return PaidCallError(kind, exact, status=status, retry_after=retry_after, maybe_billed=maybe_billed,
                             call_id=call_id)

    def _send(self, payload: dict[str, Any], call_id: str, timeout: float, is_probe: bool) -> PaidResult:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.creds.base_url.rstrip("/") + "/chat/completions", data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "distill-batch/1"})
        request.add_unredirected_header("Authorization", "Bearer " + self.creds._key)
        try:
            resp = self._opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                raw_err = exc.read(65536) if exc.fp is not None else b""
            except (OSError, http.client.HTTPException, ValueError):
                raw_err = b""  # a cut connection: classified on the status alone (a 401/403 still stops)
            code, message, rid = _tp_error_detail(raw_err)
            if not (code or message):
                message = f"(no readable error body; HTTP {exc.code})"
            kind = tp_classify(exc.code, code, message)
            raise self._fail(call_id, kind, message, status=exc.code, code=code, rid=rid,
                             retry_after=_retry_after(exc.headers), maybe_billed=status_maybe_billed(exc.code),
                             is_probe=is_probe) from None
        except urllib.error.URLError as exc:
            reason = exc.reason
            kind = "timeout" if isinstance(reason, TimeoutError) else "unreachable"
            raise self._fail(call_id, kind, f"{type(reason).__name__} before the request was sent",
                             is_probe=is_probe) from None
        except (TimeoutError, http.client.HTTPException, ConnectionError, OSError) as exc:
            kind = "timeout" if isinstance(exc, TimeoutError) else "unreachable"
            raise self._fail(call_id, kind, f"{type(exc).__name__} after the request was sent", maybe_billed=True,
                             is_probe=is_probe) from None
        try:
            with resp:
                raw = resp.read()
        except (TimeoutError, http.client.HTTPException, OSError) as exc:
            kind = "timeout" if isinstance(exc, TimeoutError) else "unreachable"
            raise self._fail(call_id, kind, f"{type(exc).__name__} while reading the reply", maybe_billed=True,
                             is_probe=is_probe) from None
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise self._fail(call_id, "provider_error", "the reply is not JSON", maybe_billed=True,
                             is_probe=is_probe) from None
        choices = body.get("choices") if isinstance(body, dict) else None
        if isinstance(body, dict) and body.get("error") and not choices:
            # an error inside a 200 reply: classified like any other (a STOP word still stops)
            code, message, rid = _tp_error_detail(raw)
            kind = tp_classify(200, code, message)
            raise self._fail(call_id, kind, message, status=200, code=code, rid=rid, maybe_billed=True,
                             is_probe=is_probe)
        usage = parse_usage(body.get("usage")) if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._fail(call_id, "provider_error", "the reply has no choices", maybe_billed=True,
                             is_probe=is_probe)
        found = choices[0].get("message")
        reply: dict[str, Any] = found if isinstance(found, dict) else {}
        self.ledger.tp_record(tool=self.tool, phase=self.phase, model=self.model, call_id=call_id,
                              outcome="ok probe" if is_probe else "ok", usage=usage)
        text = reply.get("content")
        return PaidResult(
            call_id=call_id, text=text if isinstance(text, str) else "",
            reasoning_text=str(reply.get("reasoning_content") or ""),
            served_model=body.get("model") if isinstance(body.get("model"), str) else None,
            input_tokens=usage.input_tokens if usage else 0,
            cached_input_tokens=usage.cached_input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            reasoning_tokens=usage.reasoning_tokens if usage else 0,
            usd=0.0, usage_estimated=usage is None, message=reply,
            finish_reason=choices[0].get("finish_reason"),
        )

    # -- stopping -----------------------------------------------------------------------

    def close(self, reason: str = "closed", *, wait: float = 5.0) -> None:
        got = self._gate.acquire(timeout=max(0.0, wait))
        try:
            self._closed = reason or "closed"
        finally:
            if got:
                self._gate.release()
        with self._probe_cond:
            self._probe_cond.notify_all()

    def _check_closed(self) -> None:
        if self._closed:
            raise SpendRefused(f"this Token Plan client is closed ({self._closed}): no new call")

    def open_calls(self) -> int:
        return len(self._open)

    def settle_open(self, *, error_kind: str = "interrupted", lock_timeout: float = 5.0) -> list[str]:
        """Record every call still in flight as interrupted (one ``token-plan`` row each), so a stopped
        process leaves the ledger's count of attempts complete. Never raises."""
        ledger = copy.copy(self.ledger)
        ledger.lock_timeout = min(self.ledger.lock_timeout, max(0.0, lock_timeout))
        done: list[str] = []
        for call_id in list(self._open):
            try:
                ledger.tp_record(tool=self.tool, phase=self.phase, model=self.model, call_id=call_id,
                                 outcome=f"error:{error_kind}")
            except (SpendError, OSError):
                continue
            done.append(call_id)
            self._open.pop(call_id, None)
        return done


def tokenplan_preflight(*, ledger: Ledger, model: str, env: Mapping[str, str] | None = None,
                        env_file: Path | None = None) -> dict[str, Any]:
    """Everything a Token Plan run needs, checked locally (no call is made). Raises
    ``TokenPlanSuperseded`` (``QWEN_PAYG_KEY`` present), ``QuotaPaused``, ``StopActive``
    (``TokenPlanStop`` for a key change) or another ``SpendError``."""
    creds = resolve_token_plan(env, env_file=env_file)
    if model not in TOKEN_PLAN_MODELS:
        raise SpendRefused(f"model {model!r} is not one batch work asks the Token Plan for")
    ledger.check_open(tool="preflight", model=model)  # the canonical ledger; STOP (and its renotify)
    pause = ledger.tp_pause_info()
    if pause is not None and ledger.clock() < float(pause["until"]):
        raise QuotaPaused(f"the Token Plan quota pause stands until {utc_iso(float(pause['until']))}",
                          until=float(pause["until"]))
    fingerprint = ledger.tp_check_key(creds, tool="preflight", model=model)
    notify = Path(ledger.notify_cmd or "").expanduser()
    if not ledger.notify_cmd or not (notify.is_file() and os.access(notify, os.X_OK)):
        raise SpendError(f"the notify command {display_path(notify)} is missing or not executable: a STOP "
                         "could not reach the operator")
    return {"endpoint": creds.describe(), "model": model, "key_fingerprint": fingerprint,
            "ledger": display_path(ledger.path), "stop": None,
            "pause": None if pause is None else {"until": utc_iso(float(pause["until"])), "over": True},
            "concurrency": {"start": TP_START_CONCURRENCY, "max": TP_MAX_CONCURRENCY},
            "notify": display_path(notify)}


def tokenplan_status(ledger: Ledger, *, env: Mapping[str, str] | None = None,
                     env_file: Path | None = None) -> dict[str, Any]:
    """Read-only: the route in use, the key's fingerprint state, STOP and pause, the 50-call trial,
    7-day usage (no budget: the operator allows up to 100% of the plan), the last errors and the
    adaptive concurrency each tool last recorded. Never prints the key."""
    with ledger.locked():
        rows = ledger.read_rows()
        state = ledger._tp_state_locked()
        pause = ledger._tp_pause_locked()
    now = ledger.clock()
    out: dict[str, Any] = {"ledger": display_path(ledger.path)}
    try:
        route = which_key(env, env_file=env_file)
    except KeyRefused as exc:
        route = f"unknown ({exc})"
    out["route"] = {"payg": "pay-per-token (QWEN_PAYG_KEY present: Token Plan batch use stops)",
                    "token-plan": "token-plan", "none": "none (no key)"}.get(route, route)
    try:
        creds = resolve_token_plan(env, env_file=env_file)
        fp = state.get("fingerprint")
        if not fp or not isinstance(state.get("salt"), str):
            out["key"] = "present; fingerprint not recorded yet (the first batch use records it)"
        elif hmac.compare_digest(creds.fingerprint(state["salt"]), str(fp)):
            out["key"] = f"present; matches the fingerprint recorded {state.get('fingerprint_recorded', '?')}"
        else:
            out["key"] = "present; DIFFERENT from the recorded fingerprint: the next batch use writes STOP"
        out["endpoint"] = creds.describe()
    except SpendError as exc:
        out["key"] = f"not usable for Token Plan batch work: {exc}"
    out["stop"] = ledger.stop_reason()
    if pause is not None:
        until = float(pause["until"])
        out["pause"] = {"until": utc_iso(until), "active": now < until, "basis": pause.get("basis"),
                        "error": pause.get("reason"), "notify": pause.get("notify"),
                        "probe_in_flight": bool(pause.get("probe")), "probes": pause.get("probes", 0)}
    else:
        out["pause"] = None
    plan = [r for r in rows if (r.get("note") or "").split(" ", 1)[0] == TOKEN_PLAN_TAG]
    first = plan[:TP_TRIAL_CALLS]
    outcomes = [(r.get("note") or "").split(" ")[1] if len((r.get("note") or "").split(" ")) > 1 else ""
                for r in first]
    issues = collections.Counter(o for o in outcomes if o.startswith("error:"))
    serious = {k: v for k, v in issues.items() if k in ("error:stop", "error:quota")}
    rate = issues.get("error:rate_limit", 0)
    out["trial"] = {
        "calls": len(first), "of": TP_TRIAL_CALLS, "done": len(first) >= TP_TRIAL_CALLS,
        "clean": not serious and rate < TP_BURST_429 and not (out["stop"] and TP_STOP_MARK in out["stop"]),
        "errors": dict(issues), "gate": "none (the operator dropped the ramp; this is a report)",
        "how": "judge_run.sh --execute --trial (the first 50 Phase 2 judge calls), then tokenplan-status",
    }
    out["ramp"] = (f"none (operator 2026-10-31: no ramp, no pacing). Adaptive concurrency per process: start "
                   f"{TP_START_CONCURRENCY}, max {TP_MAX_CONCURRENCY}, halve on {TP_BURST_429} rate-limit 429s "
                   f"in {int(TP_BURST_WINDOW_S)} s, +1 after {TP_GROW_AFTER_OK} clean calls")
    out["concurrency_last_recorded"] = state.get("limits") or {}
    cutoff = utc_iso(now - TP_USAGE_WINDOW_S)
    week = [r for r in plan if (r.get("ts") or "") >= cutoff]
    t = totals_of(week)
    by: dict[str, int] = collections.Counter(f"{r.get('tool')}/{r.get('phase')}" for r in week)
    out["usage_7d"] = {"attempts": t.plan_calls, "ok": t.plan_ok, "input_tokens": t.plan_input_tokens,
                       "cached_input_tokens": t.plan_cached_input_tokens, "output_tokens": t.plan_output_tokens,
                       "errors": dict(collections.Counter(
                           (r.get("note") or "").split(" ")[1] for r in week
                           if (r.get("note") or "").split(" ")[1:2] != ["ok"])),
                       "by_tool_phase": dict(by)}
    out["budget"] = ("none: up to 100% of the subscription (operator override 2026-10-31); $0 per call, "
                     "separate from the pay-per-token $150 cap")
    out["last_errors"] = state.get("last_errors") or []
    return out


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
    for name in ("preflight", "status", "tokenplan-preflight", "tokenplan-status", "tokenplan-accept-key",
                 "which-key"):
        s = sub.add_parser(name, help=f"the one ledger, {DEFAULT_LEDGER} (not configurable)")
        s.add_argument("--notify-cmd", default=DEFAULT_NOTIFY)
        if name in ("preflight", "tokenplan-preflight"):
            s.add_argument("--model", default="qwen3.8-max")
        if name != "status":
            s.add_argument("--env-file", type=Path, default=None,
                           help="the 0600 env file (default: Prometheus's own)")
    sub.add_parser("prices")
    args = ap.parse_args(argv)
    if args.cmd == "which-key":
        try:
            route = which_key(env_file=args.env_file)
        except SpendError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print(route)
        return 0 if route != "none" else 2
    if args.cmd.startswith("tokenplan-"):
        return _tokenplan_cli(args)
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


def _tokenplan_cli(args: argparse.Namespace) -> int:
    ledger = Ledger(notify_cmd=args.notify_cmd)
    try:
        if args.cmd == "tokenplan-status":
            print(json.dumps(tokenplan_status(ledger, env_file=args.env_file), indent=1))
            return 0
        if args.cmd == "tokenplan-accept-key":
            creds = resolve_token_plan(env_file=args.env_file)
            ledger.tp_accept_key(creds)
            print("recorded the current QWEN_API_KEY as the Token Plan key; the STOP file (if any) is unchanged")
            return 0
        print(json.dumps(tokenplan_preflight(ledger=ledger, model=args.model, env_file=args.env_file), indent=1))
        return 0
    except TokenPlanSuperseded as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    except QuotaPaused as exc:
        print(f"paused: {exc}", file=sys.stderr)
        return 4
    except SpendError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
