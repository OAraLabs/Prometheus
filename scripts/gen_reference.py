#!/usr/bin/env python3
"""Generate the mechanical reference docs FROM SOURCE (audit P11.3).

WHY THIS EXISTS
---------------
Three inventories were maintained by hand and drifted, because nothing
compared them to the code. Measured on 2026-09-10, at the commit this
script was written:

* ``docs/guide/api.md`` documented 89 of 104 real paths. **22 product
  endpoints were undocumented** — the whole ``/api/devices`` and
  ``/api/tasks`` families, ``/api/wiki/*``, ``/api/search``,
  ``/api/media``, ``/api/usage``, five ``/api/sessions/{id}/*`` verbs,
  ``/api/approvals/grants`` and ``/api/tools/recent``.
* ``docs/guide/features.md``'s Telegram table was missing six registered
  commands: ``/backends /ephemeral /gate /grants /remember /revoke``.
* There was no config-key reference at all, for a 396-key template.

WHAT THIS DOES **NOT** DO, DELIBERATELY
---------------------------------------
It does not replace the hand-written guides. ``api.md``'s Purpose column
carries explanation no generator can produce — *"a dead subprocess reads
unhealthy, never as empty success"* — and that prose IS the product. A
generator that overwrote it would trade a drift problem for a much worse
information problem.

So the split is: **the generated files are the COMPLETE inventory, the
guides are the CURATED explanation.** Completeness stops being a thing a
human has to remember, which is the only part that was actually failing.

THE RATCHET
-----------
``tests/test_generated_reference.py`` regenerates into memory and compares.
Move a route, rename a command, add a config key, and the test fails until
the doc is regenerated. That — not this script — is what stops the drift
class recurring; the script alone would rot exactly like the hand-written
tables did.

Usage:
    uv run python scripts/gen_reference.py          # write the files
    uv run python scripts/gen_reference.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "config" / "prometheus.yaml.default"
OUT_DIR = REPO / "docs" / "reference"

BANNER = (
    "<!-- GENERATED FILE — DO NOT EDIT BY HAND.\n"
    "     Regenerate with:  uv run python scripts/gen_reference.py\n"
    "     Pinned current by tests/test_generated_reference.py. -->\n"
)

# FastAPI mounts these itself; they are real and reachable, and listing
# them is the point — /openapi.json publishes the whole surface.
_FASTAPI_BUILTINS = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}


def _template() -> dict[str, Any]:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


# ── routes ─────────────────────────────────────────────────────────────

def _routes_of(app: Any) -> list[tuple[str, str]]:
    """(path, "GET, POST") per PATH, HEAD/OPTIONS dropped.

    Verbs are merged per path rather than per route object. FastAPI creates
    one route per decorator, so ``@get("/api/cron")`` and
    ``@post("/api/cron")`` are two objects for one endpoint — listing them
    as two rows made the table 124 lines for 100 paths and read as though
    the surface were bigger than it is.
    """
    by_path: dict[str, set[str]] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods:
            continue
        verbs = {m for m in methods if m not in ("HEAD", "OPTIONS")}
        if verbs:
            by_path.setdefault(path, set()).update(verbs)
    return sorted((p, ", ".join(sorted(v))) for p, v in by_path.items())


def _collect_routes() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Both apps. The setup app is a SEPARATE surface, not a flag on the main
    one — its routes are unreachable once setup completes, and five of them
    were documented in api.md as though they lived on the daemon."""
    from prometheus.web.server import create_app

    daemon = _routes_of(create_app(_template()))

    from prometheus.web.setup_server import PairingState, create_setup_app

    setup = _routes_of(create_setup_app(PairingState()))
    return daemon, setup


def _render_routes() -> str:
    daemon, setup = _collect_routes()
    lines = [BANNER, "# Route reference (generated)", ""]
    lines += [
        "Every HTTP route the daemon mounts, read out of the live FastAPI",
        "app. For what each one is *for*, see the curated",
        "[API guide](../guide/api.md) — this file is the complete list, that",
        "one is the explanation.",
        "",
    ]

    builtins = [(p, m) for p, m in daemon if p in _FASTAPI_BUILTINS]
    product = [(p, m) for p, m in daemon if p not in _FASTAPI_BUILTINS]

    lines += [f"## Daemon app — {len(product)} paths", ""]
    lines += ["| Path | Methods |", "|---|---|"]
    lines += [f"| `{p}` | {m} |" for p, m in product]
    lines += [""]

    lines += [
        f"## Setup app — {len(setup)} paths",
        "",
        "A separate FastAPI app (`web/setup_server.py`), not the daemon app",
        "behind a flag: the real route surface is deliberately never mounted",
        "in setup mode.",
        "",
        "| Path | Methods |",
        "|---|---|",
    ]
    lines += [f"| `{p}` | {m} |" for p, m in setup]
    lines += [""]

    lines += [
        f"## FastAPI built-ins — {len(builtins)} paths",
        "",
        "Mounted by FastAPI itself. Listed because they are reachable:",
        "`/openapi.json` publishes the entire surface.",
        "",
        "| Path | Methods |",
        "|---|---|",
    ]
    lines += [f"| `{p}` | {m} |" for p, m in builtins]
    return "\n".join(lines) + "\n"


# ── commands ───────────────────────────────────────────────────────────

def _telegram_commands() -> list[str]:
    src = (REPO / "src" / "prometheus" / "gateway" / "telegram.py").read_text(
        encoding="utf-8")
    return sorted(set(re.findall(r'CommandHandler\(\s*"([a-z0-9_]+)"', src)))


def _slack_commands() -> list[str]:
    src = (REPO / "src" / "prometheus" / "gateway" / "slack.py").read_text(
        encoding="utf-8")
    return sorted(set(re.findall(r'command\(\s*"/prometheus-([a-z0-9-]+)"', src)))


def _discord_commands() -> dict[str, str]:
    """{bare name: "/prometheus <group> <name>"} from the real _register calls.

    Discord does NOT expose flat slash commands. It builds one ``/prometheus``
    root with four sub-groups (core, session, ops, provider) and registers
    every command into one of them, so the invocation is
    ``/prometheus core help`` — not ``/help``.

    An earlier version of this function matched ``name="..."`` anywhere in
    the module. That returned fifteen strings, of which the real answer was
    ZERO: it had caught the four GROUP names, the root, the provider-override
    commands, and the logger name ``discord_gateway``. It is the same trap
    that once produced a false audit finding about Slack — the registration
    does not spell the command the way the reader expects, so a grep for the
    expected shape confirms whatever it already believed. Match the
    REGISTRATION CALL.
    """
    src = (REPO / "src" / "prometheus" / "gateway" / "discord.py").read_text(
        encoding="utf-8")

    # group variable -> the group's Discord name
    groups = dict(re.findall(
        r'(\w+)\s*=\s*app_commands\.Group\(\s*\n?\s*name="([a-z0-9_-]+)"', src))

    out: dict[str, str] = {}
    for var, name in re.findall(
            r'self\._register\(\s*(\w+)\s*,\s*"([a-z0-9_-]+)"', src):
        group = groups.get(var, var)
        out[name] = f"/prometheus {group} {name}" if group != "prometheus" \
            else f"/prometheus {name}"
    return dict(sorted(out.items()))


def _render_commands() -> str:
    from prometheus.web.slash_router import WEB_NATIVE_ONLY

    tg = _telegram_commands()
    slack = _slack_commands()
    discord = _discord_commands()
    web_only = sorted(WEB_NATIVE_ONLY)

    lines = [BANNER, "# Command reference (generated)", ""]
    lines += [
        "Slash commands as REGISTERED, read out of each gateway's own",
        "registration calls — not from a hand-mirrored list. Slack's are",
        "exposed as `/prometheus-<name>`; the bare name is shown here.",
        "",
        "Enumerating the registry is deliberate: grepping for an expected",
        "name once produced a false finding, because Slack's registrations",
        "carry the `/prometheus-` prefix and the bare name matched nothing.",
        "",
    ]

    every = sorted(set(tg) | set(slack) | set(discord))
    lines += [
        f"## By gateway — {len(every)} distinct commands",
        "",
        "The three gateways spell the same command differently. Telegram",
        "uses the bare name, Slack prefixes every one with `/prometheus-`,",
        "and Discord registers a single `/prometheus` root with four",
        "sub-groups — so its invocation is `/prometheus core help`, never",
        "`/help`. The Discord column shows the real invocation for that",
        "reason.",
        "",
        "| Command | Telegram | Slack | Discord |",
        "|---|---|---|---|",
    ]
    for name in every:
        lines.append(
            f"| `/{name}` "
            f"| {'`/' + name + '`' if name in tg else '—'} "
            f"| {'`/prometheus-' + name + '`' if name in slack else '—'} "
            f"| {'`' + discord[name] + '`' if name in discord else '—'} |"
        )
    lines += [""]

    lines += [
        f"## Not available in web chat — {len(web_only)}",
        "",
        "`WEB_NATIVE_ONLY` in `web/slash_router.py`. These are handled by the",
        "chat gateways but deferred in Beacon's web chat.",
        "",
        "| Command |",
        "|---|",
    ]
    lines += [f"| `/{n}` |" for n in web_only]
    return "\n".join(lines) + "\n"


# ── config keys ────────────────────────────────────────────────────────

def _flatten(node: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = prefix + (str(key),)
            out.append((".".join(path), value))
            out += _flatten(value, path)
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return "*(section)*" if value else "`{}` *(open map)*"
    if isinstance(value, list):
        return "`[]`" if not value else f"`{value}`"
    if value is None:
        return "*(empty — no code default)*"
    return f"`{value}`"


def _render_config_keys() -> str:
    rows = _flatten(_template())
    lines = [BANNER, "# Config key reference (generated)", ""]
    lines += [
        f"Every key in `config/prometheus.yaml.default` — **{len(rows)}** of",
        "them — with the value the template ships. This is what a fresh",
        "install gets, not what the code falls back to when a key is absent:",
        "those are pinned equal to each other by",
        "`tests/test_config_defaults_equality.py`, and where they disagree",
        "that file carries the debt register.",
        "",
        "An empty value means the template ships no default and the code has",
        "none either — the key is documented so its absence is visible.",
        "",
        "| Key | Template value |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {_fmt(v)} |" for k, v in rows]
    return "\n".join(lines) + "\n"


# ── driver ─────────────────────────────────────────────────────────────

FILES: dict[str, Any] = {
    "routes.md": _render_routes,
    "commands.md": _render_commands,
    "config-keys.md": _render_config_keys,
}


def render_all() -> dict[str, str]:
    return {name: fn() for name, fn in FILES.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if any file on disk differs from a fresh generation",
    )
    args = parser.parse_args()

    rendered = render_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    for name, text in rendered.items():
        path = OUT_DIR / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO)}")

    if stale:
        print(
            "STALE generated reference: "
            + ", ".join(stale)
            + "\nRegenerate with: uv run python scripts/gen_reference.py",
            file=sys.stderr,
        )
        return 1
    if args.check:
        print(f"generated reference is current ({len(rendered)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
