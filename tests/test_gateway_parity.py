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

import prometheus.gateway.commands as commands_mod

GATEWAY_DIR = Path(commands_mod.__file__).resolve().parent


@dataclass(frozen=True)
class PlatformSpec:
    """How to find a platform's registered commands + handler names."""

    name: str
    source: Path
    # Regex with two groups: (command_name, handler_attr)
    registration_re: str
    adapter_import: str  # "module:Class" for handler hasattr checks


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
)


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
) -> dict[str, str | None]:
    return {"telegram": telegram, "slack": slack, "discord": discord}


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
    Family("beacon", ("cmd_beacon",), _cmds("beacon", "prometheus-beacon", "beacon")),
    Family("tools", ("cmd_tools",), _cmds("tools", "prometheus-tools", "tools")),
    Family("pairs", ("cmd_pairs",), _cmds("pairs", "prometheus-pairs", "pairs")),
    # -- approvals ----------------------------------------------------------
    Family("approve", ("cmd_approve",), _cmds("approve", "prometheus-approve", "approve")),
    Family("deny", ("cmd_deny",), _cmds("deny", "prometheus-deny", "deny")),
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
    """Return {command_name: handler_attr} scraped from the adapter source."""
    src = spec.source.read_text(encoding="utf-8")
    return dict(re.findall(spec.registration_re, src))


def _adapter_class(spec: PlatformSpec):
    import importlib
    mod_name, cls_name = spec.adapter_import.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


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
        for fam in MANIFEST:
            if any(v is None for v in fam.commands.values()):
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
        platform and its handler method exists on the adapter class."""
        problems: list[str] = []
        for spec in PLATFORMS:
            registered = _registered(spec)
            adapter_cls = _adapter_class(spec)
            for fam in MANIFEST:
                cmd = fam.commands.get(spec.name)
                if cmd is None:
                    continue
                handler = registered.get(cmd)
                if handler is None:
                    problems.append(
                        f"{spec.name}: family {fam.name!r} expects command "
                        f"{cmd!r} but it is not registered in {spec.source.name}"
                    )
                elif not hasattr(adapter_cls, handler):
                    problems.append(
                        f"{spec.name}: {cmd!r} registers {handler!r} which "
                        f"does not exist on {adapter_cls.__name__}"
                    )
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
                cell = fam.commands.get(c) or f"— ({fam.gap_reason[:18]}…)"
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
