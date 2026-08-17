"""SPRINT G1 (+G2) — the gateway parity chart, as CI.

Single source of truth for the slash-command surface across gateways. Every
command family must have:

  (a) a shared implementation in ``prometheus.gateway.commands`` (or an
      explicit ``shared_gap`` reason for the handful of one-liners where a
      shared function would be pure ceremony),
  (b) a registered Telegram handler,
  (c) a registered Slack handler,
  (d) a registered Discord handler (SPRINT G2),

with an explicit, commented allowlist for deliberate platform gaps.

Drift-proofing runs BOTH directions:
  * a manifest entry whose command isn't registered on a platform FAILS, and
  * a command registered on any platform but missing from the manifest FAILS.

So adding a command to one gateway and forgetting the others (or forgetting
the chart) breaks CI.

Extending for a new gateway is one line per layer: add the platform to
``PLATFORMS`` (name, source file, registration regex) and a
``"<platform>": "<command>"`` (or ``None`` + gap reason) key to each
manifest entry — exactly what G2 did for Discord.

Discord note: the manifest stores the family LEAF name; the user-facing
command is ``/prometheus <section> <leaf>`` (Discord caps a command at 25
options, so the 43 families sit one section-group deep — see discord.py's
module docstring). Every family is registered on Discord: ZERO discord
allowlist entries.

No adapters are instantiated and nothing touches tokens, env files, or the
network — registration is asserted by scanning the adapter sources, and
handler existence by ``hasattr`` on the classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

import prometheus.gateway.commands as commands_mod


def _web_registered() -> dict[str, str]:
    """Commands the web (Beacon) chat surface actually HANDLES.

    ``web.slash_router.route_slash`` dispatches a leading-slash message through
    exactly two shared tables; anything else is either an explicit boundary
    reply (WEB_NATIVE_ONLY) or falls through to the agent as chat text. So the
    tables ARE the registration, and reading them live is the web equivalent of
    scraping an adapter's CommandHandler calls.
    """
    return {
        **{name: "formatter_table" for name in commands_mod._FORMATTER_COMMANDS},
        **{name: "session_table" for name in commands_mod._SESSION_COMMANDS},
    }

GATEWAY_DIR = Path(commands_mod.__file__).resolve().parent


@dataclass(frozen=True)
class PlatformSpec:
    """How to find a platform's registered commands + handler names.

    Two discovery strategies, because the surfaces genuinely differ:

    * the three chat GATEWAYS register via an adapter method call, so they are
      scraped from source with ``registration_re`` and checked against
      ``adapter_import``;
    * the WEB (Beacon) surface has no adapter and no registration call — a
      command is reachable there iff it is in one of the shared dispatch tables
      ``gateway.commands`` exposes. That is a ``resolver`` instead.

    Modelling web as a regex-over-source platform would have been modelling it
    wrong; the variation is *how you discover what a surface handles*, so that
    is the part that varies.
    """

    name: str
    source: Path | None = None
    # Regex with two groups: (command_name, handler_attr)
    registration_re: str = ""
    adapter_import: str = ""  # "module:Class" for handler hasattr checks
    # Alternative to source+regex: returns {command_name: handler_label}.
    resolver: Callable[[], dict[str, str]] | None = None


PLATFORMS: tuple[PlatformSpec, ...] = (
    PlatformSpec(
        name="telegram",
        source=GATEWAY_DIR / "telegram.py",
        registration_re=r'CommandHandler\(\s*"([\w-]+)",\s*self\.(\w+)\s*\)',
        adapter_import="prometheus.gateway.telegram:TelegramAdapter",
    ),
    PlatformSpec(
        name="slack",
        source=GATEWAY_DIR / "slack.py",
        registration_re=r'self\._app\.command\("/([\w-]+)"\)\(self\.(\w+)\)',
        adapter_import="prometheus.gateway.slack:SlackAdapter",
    ),
    PlatformSpec(
        name="discord",
        source=GATEWAY_DIR / "discord.py",
        registration_re=r'self\._register\(\s*\w+,\s*"([\w-]+)",\s*self\.(\w+)',
        adapter_import="prometheus.gateway.discord:DiscordAdapter",
    ),
    # SPRINT-WEB-PARITY: the fourth surface. The chart was telegram/slack/
    # discord only, so it went green while Beacon had no /reset at all —
    # /reset is registered on all three chat gateways, which fully satisfied a
    # chart that never asked about web. That blind spot is why /revoke and
    # /remember were caught (they drift on the gateway axis) and the six below
    # were not.
    PlatformSpec(name="web", resolver=_web_registered),
)


# Deliberate, DOCUMENTED web gaps: registered on Telegram, and the web surface
# answers with an explicit "not on web" boundary message instead of silently
# eating the command. THE CHART'S OWN COPY, deliberately not derived from
# ``slash_router.WEB_NATIVE_ONLY`` — deriving it would make the set unable to
# fail, which is the exact defect this column exists to close: nothing used to
# break when WEB_NATIVE_ONLY grew. Pinned equal to the router by
# TestWebSurface.test_deferred_set_is_pinned_to_the_router, so the set cannot
# grow (or shrink) without this chart changing in the same commit.
WEB_DEFERRED: frozenset[str] = frozenset({
    # Session/daemon state bound to the TelegramGateway instance.
    "start", "clear", "reset", "route",
    "benchmark", "voice", "tools", "pairs",
    "approve", "deny", "pending",
    "gepa", "symbiote", "audit", "press",
    "escalations",
    # Per-session provider overrides.
    "claude", "gpt", "gemini", "xai", "grok", "local",
    "deepseek", "kimi", "glm", "mimo",
})

_SAME = object()  # sentinel: web uses the same command name as Telegram


@dataclass(frozen=True)
class Family:
    """One command family in the parity chart."""

    name: str
    # Function names that must exist in prometheus.gateway.commands.
    shared: tuple[str, ...]
    # platform name -> registered command name, or None = deliberate gap.
    commands: dict[str, str | None] = field(default_factory=dict)
    # Required when any platform value is None.
    gap_reason: str = ""
    # Required when shared == () — why no shared function exists.
    shared_gap: str = ""


def _cmds(
    telegram: str | None, slack: str | None, discord: str | None,
    web: str | None | object = _SAME,
) -> dict[str, str | None]:
    """Per-platform command names. ``web`` defaults to the Telegram name —
    the web router matches the same bare names — unless the family is a
    documented boundary (``WEB_DEFERRED``), where it is an explicit gap."""
    if web is _SAME:
        web = None if telegram in WEB_DEFERRED else telegram
    return {"telegram": telegram, "slack": slack, "discord": discord, "web": web}


# ---------------------------------------------------------------------------
# THE PARITY MANIFEST (the chart)
# ---------------------------------------------------------------------------

MANIFEST: tuple[Family, ...] = (
    # -- core -------------------------------------------------------------
    Family(
        "start", (), _cmds("start", None, "start"),
        gap_reason=(
            "Telegram-native onboarding ping (/start is a Telegram platform "
            "convention); Slack onboarding happens via app-home/@mention."
        ),
        shared_gap="one fixed greeting string; a shared fn would be ceremony",
    ),
    Family(
        "clear", (), _cmds("clear", None, "clear"),
        gap_reason="alias of /reset kept for Telegram muscle memory; Slack has -reset",
        shared_gap="one-line session_manager.clear alias of reset",
    ),
    Family(
        "reset", (), _cmds("reset", "prometheus-reset", "reset"),
        shared_gap="one-line session_manager.clear + fixed reply string",
    ),
    Family(
        "ephemeral", ("cmd_ephemeral",), _cmds("ephemeral", None, None),
        gap_reason=(
            "The COMMAND is Telegram-only for now; the BEHAVIOUR is not. "
            "Suppression lives in SessionManager.get_or_create and run_loop, "
            "both surface-agnostic and keyed on the session id, so a Slack or "
            "Discord session whose id is flagged is already honoured — there "
            "is simply no command there yet to set the flag. cmd_ephemeral is "
            "in the shared commands layer precisely so wiring the other two is "
            "a handler apiece, not a reimplementation."
        ),
    ),
    Family("help", ("cmd_help",), _cmds("help", "prometheus-help", "help")),
    Family("status", ("cmd_status",), _cmds("status", "prometheus-status", "status")),
    Family("model", ("cmd_model",), _cmds("model", "prometheus-model", "model")),
    Family("wiki", ("cmd_wiki",), _cmds("wiki", "prometheus-wiki", "wiki")),
    Family("note", ("cmd_note",), _cmds("note", "prometheus-note", "note")),
    Family("sentinel", ("cmd_sentinel",), _cmds("sentinel", "prometheus-sentinel", "sentinel")),
    Family(
        "benchmark", (), _cmds("benchmark", "prometheus-benchmark", "benchmark"),
        shared_gap=(
            "the handler IS the benchmark (one agent_loop.run_async smoke "
            "call); no formatter logic to share"
        ),
    ),
    Family("context", ("cmd_context",), _cmds("context", "prometheus-context", "context")),
    Family(
        "skills",
        ("cmd_skills", "cmd_skills_auto_list", "cmd_skills_show",
         "cmd_skills_pin", "cmd_skills_unpin", "cmd_skills_history"),
        _cmds("skills", "prometheus-skills", "skills"),
    ),
    Family(
        "memory", ("cmd_memory_show", "cmd_memory_limits"),
        _cmds("memory", "prometheus-memory", "memory"),
    ),
    Family(
        "curator", ("cmd_curator_show", "cmd_curator_status", "cmd_curator_run"),
        _cmds("curator", "prometheus-curator", "curator"),
    ),
    Family(
        "notifications", ("cmd_notifications",),
        _cmds("notifications", "prometheus-notifications", "notifications"),
    ),
    Family("health", ("cmd_health",), _cmds("health", "prometheus-health", "health")),
    Family("events", ("cmd_events",), _cmds("events", "prometheus-events", "events")),
    # -- steering / durability --------------------------------------------
    Family("steer", ("cmd_steer",), _cmds("steer", "prometheus-steer", "steer")),
    Family("queue", ("cmd_queue",), _cmds("queue", "prometheus-queue", "queue")),
    Family("unqueue", ("cmd_unqueue",), _cmds("unqueue", "prometheus-unqueue", "unqueue")),
    Family(
        "clearsteers", ("cmd_clearsteers",),
        _cmds("clearsteers", "prometheus-clearsteers", "clearsteers"),
    ),
    # -- infra / observability ---------------------------------------------
    Family("anatomy", ("cmd_anatomy",), _cmds("anatomy", "prometheus-anatomy", "anatomy")),
    Family("doctor", ("cmd_doctor",), _cmds("doctor", "prometheus-doctor", "doctor")),
    Family("profile", ("cmd_profile",), _cmds("profile", "prometheus-profile", "profile")),
    Family("gate", ("cmd_gate",), _cmds("gate", None, None),
           gap_reason="Telegram-only runtime toggle; slack/discord adapters get it when requested"),
    Family("beacon", ("cmd_beacon",), _cmds("beacon", "prometheus-beacon", "beacon")),
    Family("tools", ("cmd_tools",), _cmds("tools", "prometheus-tools", "tools")),
    Family("pairs", ("cmd_pairs",), _cmds("pairs", "prometheus-pairs", "pairs")),
    # -- approvals ----------------------------------------------------------
    Family("approve", ("cmd_approve",), _cmds("approve", "prometheus-approve", "approve")),
    # The approval prompt was trimmed to two option lines; the four
    # verb+extent lines moved behind /remember. Telegram-only would rebuild
    # the write-only asymmetry the revoke entry below exists to prevent — an
    # operator on Slack or Discord would see a prompt offering a command
    # their surface does not have.
    Family("remember", ("cmd_remember",), _cmds("remember", "prometheus-remember", "remember")),
    Family("deny", ("cmd_deny",), _cmds("deny", "prometheus-deny", "deny")),
    Family("grants", ("cmd_grants",), _cmds("grants", "prometheus-grants", "grants")),
    # SPRINT-CONSENT Phase 2: revocation reaches every surface. Wiring it to
    # Telegram alone would reproduce the write-only asymmetry the sprint
    # exists to remove, one level down.
    Family("revoke", ("cmd_revoke",), _cmds("revoke", "prometheus-revoke", "revoke")),
    Family("pending", ("cmd_pending",), _cmds("pending", "prometheus-pending", "pending")),
    # -- autonomy subsystems -------------------------------------------------
    Family("gepa", ("cmd_gepa",), _cmds("gepa", "prometheus-gepa", "gepa")),
    Family("symbiote", ("cmd_symbiote",), _cmds("symbiote", "prometheus-symbiote", "symbiote")),
    Family("audit", ("cmd_audit",), _cmds("audit", "prometheus-audit", "audit")),
    Family("press", ("cmd_press",), _cmds("press", "prometheus-press", "press")),
    Family(
        "escalations", ("cmd_escalations",),
        _cmds("escalations", "prometheus-escalations", "escalations"),
    ),
    # -- voice ---------------------------------------------------------------
    # The Slack handler is registered but platform-honest: it explains that
    # the TTS voice-note reply pipeline is Telegram-only. The registration
    # itself is asserted; the functional gap is documented in
    # NON_COMMAND_GAPS below.
    Family("voice", ("cmd_voice",), _cmds("voice", "prometheus-voice", "voice")),
    # -- provider overrides ---------------------------------------------------
    Family(
        "claude", ("cmd_provider_override",),
        _cmds("claude", "prometheus-claude", "claude"),
    ),
    Family("gpt", ("cmd_provider_override",), _cmds("gpt", "prometheus-gpt", "gpt")),
    Family(
        "gemini", ("cmd_provider_override",),
        _cmds("gemini", "prometheus-gemini", "gemini"),
    ),
    Family("xai", ("cmd_provider_override",), _cmds("xai", "prometheus-xai", "xai")),
    Family("grok", ("cmd_provider_override",), _cmds("grok", "prometheus-grok", "grok")),
    # CLOUD EXPANSION (2026-07): DeepSeek / Kimi (Moonshot) / GLM (Z.ai) /
    # MiMo (Xiaomi) — same shared handler, all three platforms, no gaps.
    Family(
        "deepseek", ("cmd_provider_override",),
        _cmds("deepseek", "prometheus-deepseek", "deepseek"),
    ),
    Family("kimi", ("cmd_provider_override",), _cmds("kimi", "prometheus-kimi", "kimi")),
    Family("glm", ("cmd_provider_override",), _cmds("glm", "prometheus-glm", "glm")),
    Family("mimo", ("cmd_provider_override",), _cmds("mimo", "prometheus-mimo", "mimo")),
    Family("qwen", ("cmd_provider_override",), _cmds("qwen", "prometheus-qwen", "qwen")),
    Family("local", ("cmd_local_override",), _cmds("local", "prometheus-local", "local")),
    Family("route", ("cmd_route",), _cmds("route", "prometheus-route", "route")),
)


# Deliberate NON-slash-command platform gaps — documented here so the
# allowlist is versioned next to the chart. These are capabilities, not
# commands, so they aren't mechanically asserted.
NON_COMMAND_GAPS: tuple[tuple[str, str], ...] = (
    ("slack: media ingestion (photo/voice/document/sticker)",
     "the shared media pipeline now exists (G2: gateway/media_services.py, "
     "used by telegram + discord); Slack file_shared wiring is still open"),
    ("slack + discord: TTS voice-note replies",
     "piper→opus/ogg pipeline is bound to Telegram's voice-message API; "
     "the voice command on both surfaces replies with an explicit "
     "not-supported boundary (Discord voice-message INPUT works — Whisper)"),
    ("slack + discord: inline message dispatch on override commands "
     "(e.g. '/claude what is 2+2?')",
     "Slack slash payloads / Discord interactions have no message-dispatch "
     "context wired; handlers append an explicit note instead of silently "
     "dropping the text"),
    ("telegram: emoji reaction ack (eyes → white_check_mark)",
     "Slack/Discord-native affordance; Telegram uses typing indicator instead"),
    ("discord: sticker vision analysis",
     "Telegram stickers ride a dedicated sticker_cache; Discord stickers "
     "arrive as message.stickers (not attachments) and are not wired — "
     "image ATTACHMENTS get full vision analysis"),
    ("approval prompt delivery",
     "ApprovalQueue's outbound prompt transport is the Telegram adapter; "
     "approve/deny/pending work from every gateway"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registered(spec: PlatformSpec) -> dict[str, str]:
    """Return {command_name: handler_attr} for a platform.

    Resolver-based platforms (web) read their live dispatch tables; the chat
    gateways are scraped from their adapter source.
    """
    if spec.resolver is not None:
        return spec.resolver()
    src = spec.source.read_text(encoding="utf-8")
    return dict(re.findall(spec.registration_re, src))


def _adapter_class(spec: PlatformSpec):
    """The adapter class, or None for a platform that has no adapter (web)."""
    if not spec.adapter_import:
        return None
    import importlib
    mod_name, cls_name = spec.adapter_import.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def _registration_problems(names: set[str]) -> list[str]:
    """(b)+(c) for the named platforms — factored out so the known-red web
    column can be asserted separately and NOT mask a gateway regression."""
    problems: list[str] = []
    for spec in PLATFORMS:
        if spec.name not in names:
            continue
        registered = _registered(spec)
        adapter_cls = _adapter_class(spec)
        for fam in MANIFEST:
            cmd = fam.commands.get(spec.name)
            if cmd is None:
                continue
            handler = registered.get(cmd)
            if handler is None:
                where = spec.source.name if spec.source else "the shared dispatch tables"
                problems.append(
                    f"{spec.name}: family {fam.name!r} expects command "
                    f"{cmd!r} but it is not registered in {where}"
                )
            elif adapter_cls is not None and not hasattr(adapter_cls, handler):
                problems.append(
                    f"{spec.name}: {cmd!r} registers {handler!r} which "
                    f"does not exist on {adapter_cls.__name__}"
                )
    return problems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManifestInternalConsistency:
    def test_family_names_unique(self):
        names = [f.name for f in MANIFEST]
        assert len(names) == len(set(names))

    def test_every_family_covers_every_platform(self):
        """Each family must take an explicit stance on each platform —
        a command name or a deliberate (reasoned) gap. When a new platform
        is added to PLATFORMS this test fails until every family says what
        that platform does (this is how G2 forced the discord column)."""
        platform_names = {p.name for p in PLATFORMS}
        for fam in MANIFEST:
            assert set(fam.commands) == platform_names, (
                f"family {fam.name!r} must declare an entry for every "
                f"platform {sorted(platform_names)} (got {sorted(fam.commands)})"
            )

    def test_gaps_have_reasons(self):
        """A deliberate gap must be documented.

        A WEB gap is documented at ONE site — the ``WEB_DEFERRED`` block above,
        which carries the reason for the whole class (state bound to the
        TelegramGateway instance, or a per-session provider override) and is
        pinned to the router. Requiring 26 copies of that same sentence on
        individual families would be ceremony, not documentation, so a web gap
        is reasoned by membership; every OTHER platform's gap still needs its
        own ``gap_reason``.
        """
        for fam in MANIFEST:
            gapped = {p for p, v in fam.commands.items() if v is None}
            if gapped == {"web"} and fam.commands["telegram"] in WEB_DEFERRED:
                continue
            if gapped:
                assert fam.gap_reason, (
                    f"family {fam.name!r} has a platform gap without a "
                    "gap_reason — deliberate gaps must be documented"
                )

    def test_shared_gaps_have_reasons(self):
        for fam in MANIFEST:
            if not fam.shared:
                assert fam.shared_gap, (
                    f"family {fam.name!r} has no shared function and no "
                    "shared_gap reason"
                )


class TestSharedLayer:
    def test_every_family_has_its_shared_functions(self):
        missing: list[str] = []
        for fam in MANIFEST:
            for fn_name in fam.shared:
                fn = getattr(commands_mod, fn_name, None)
                if not callable(fn):
                    missing.append(f"{fam.name}: commands.{fn_name}")
        assert not missing, (
            "shared commands.py functions missing:\n  " + "\n  ".join(missing)
        )


class TestRegistrations:
    def test_manifest_commands_are_registered(self):
        """(b)+(c): every non-gap manifest command is registered on its
        platform and its handler method exists on the adapter class.

        Scoped to the three chat GATEWAYS. The web column is asserted by
        TestWebSurface below, which is a known red — quarantining it there
        keeps THIS test strict, so a telegram/slack/discord regression still
        fails loudly instead of hiding inside an xfail.
        """
        problems = _registration_problems({"telegram", "slack", "discord"})
        assert not problems, "\n".join(problems)

    def test_no_unlisted_registrations(self):
        """Reverse tripwire: a command registered on any platform but absent
        from the manifest fails CI — the chart cannot silently drift."""
        problems: list[str] = []
        for spec in PLATFORMS:
            listed = {
                fam.commands.get(spec.name)
                for fam in MANIFEST
                if fam.commands.get(spec.name) is not None
            }
            for cmd in _registered(spec):
                if cmd not in listed:
                    problems.append(
                        f"{spec.name}: command {cmd!r} is registered in "
                        f"{spec.source.name} but missing from the parity "
                        "manifest (tests/test_gateway_parity.py) — add a "
                        "Family entry (with per-platform names or documented "
                        "gaps)"
                    )
        assert not problems, "\n".join(problems)


class TestWebSurface:
    """The fourth surface. Two guarantees the chart could not make before.

    KNOWN RED, deliberately landed that way: piece 2 of the web-parity arc adds
    the measurement, piece 3 wires the six and removes the xfails. They are
    ``strict=True`` so the day a command IS wired, the XPASS fails the build and
    forces the marker out — a skip would have rotted silently, and folding the
    wiring into this PR would have hidden the size of the gap behind its fix.
    """

    def test_deferred_set_is_pinned_to_the_router(self):
        """WEB_NATIVE_ONLY may not grow (or shrink) without the chart moving.

        This is the coverage half. Before it, WEB_NATIVE_ONLY was only ever
        asserted DISJOINT from the shared tables (test_web_slash_router.py) —
        so a command could be added to the boundary set, permanently deferring
        it from the web surface, and nothing anywhere failed.
        """
        from prometheus.web.slash_router import WEB_NATIVE_ONLY

        assert set(WEB_DEFERRED) == set(WEB_NATIVE_ONLY), (
            "the chart's deliberate-web-gap list and "
            "slash_router.WEB_NATIVE_ONLY have diverged.\n"
            f"  only in the chart : {sorted(WEB_DEFERRED - WEB_NATIVE_ONLY)}\n"
            f"  only in the router: {sorted(WEB_NATIVE_ONLY - WEB_DEFERRED)}\n"
            "Deferring a command from the web surface is a charted decision — "
            "add it to WEB_DEFERRED with the reason, or wire it."
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN GAP, piece 3 fixes it: /ephemeral /gate /grants /qwen "
            "/remember /revoke are registered on Telegram but are in NEITHER "
            "the web dispatch tables NOR WEB_NATIVE_ONLY, so route_slash falls "
            "through and the agent eats them as chat text — no reply, no "
            "effect, no boundary message. Five already have shared "
            "implementations (cmd_ephemeral/cmd_gate/cmd_grants/cmd_remember/"
            "cmd_revoke); the web router simply never dispatches to them."
        ),
    )
    def test_web_surface_reaches_every_manifest_command(self):
        problems = _registration_problems({"web"})
        assert not problems, "\n".join(problems)

    @pytest.mark.xfail(
        strict=True,
        reason="same six — every command must be handled OR explicitly refused",
    )
    def test_no_command_falls_through_to_the_agent(self):
        """The defect stated directly, independent of the manifest.

        Every command registered on Telegram must be either handled by the web
        surface or explicitly refused by it. A command in neither is not
        "missing on web" — it is WORSE than the deferred set, because the user
        gets no boundary message at all. /revoke typing a sentence at the model
        is the write-only asymmetry the manifest comment beside it warns about,
        on the surface nobody added.
        """
        from prometheus.web.slash_router import WEB_NATIVE_ONLY

        telegram = set(_registered(next(p for p in PLATFORMS if p.name == "telegram")))
        reachable = set(_web_registered()) | set(WEB_NATIVE_ONLY)
        fell_through = sorted(telegram - reachable)
        assert not fell_through, (
            "registered on Telegram, silently swallowed by the agent on web: "
            + ", ".join("/" + c for c in fell_through)
        )


class TestParityReport:
    def test_print_parity_chart(self, capsys):
        """Always-green reporter: prints the chart + deliberate-gap allowlist
        so the parity state is visible in CI logs (`pytest -s` locally)."""
        cols = [p.name for p in PLATFORMS]
        header = f"{'family':<14}" + "".join(f"{c:<26}" for c in cols) + "shared"
        lines = [header, "-" * len(header)]
        for fam in MANIFEST:
            row = f"{fam.name:<14}"
            for c in cols:
                if fam.commands.get(c):
                    cell = fam.commands[c]
                elif c == "web" and fam.commands["telegram"] in WEB_DEFERRED:
                    cell = "— (boundary reply)"
                else:
                    cell = f"— ({fam.gap_reason[:18]}…)"
                row += f"{cell:<26}"
            row += ",".join(fam.shared) if fam.shared else f"— ({fam.shared_gap[:40]})"
            lines.append(row)
        lines.append("")
        lines.append("Deliberate non-command gaps (allowlist):")
        for what, why in NON_COMMAND_GAPS:
            lines.append(f"  * {what}\n      {why}")
        print("\n".join(lines))
        # Sanity: the chart currently covers every registered command.
        assert len(MANIFEST) >= 40
