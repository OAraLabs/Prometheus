"""P2 — every gateway command passes its allowlist BY CONSTRUCTION.

THE DEFECT (audit, verified-high; the #202 shape left unfixed on two surfaces):

* Slack — `channel_allowed()` ran only in `_handle_message` and `_handle_mention`.
  All ~52 `/prometheus-*` slash handlers were structurally exempt, and Slack
  slash commands are workspace-global: any member (incl. Slack Connect external
  users) could run `/prometheus-approve always <id>` — which persists a
  SecurityGate grant to prometheus.yaml — from any channel or DM.
* Discord — `discord_inbound_allowed()` ran only on the message path. All 43
  app-command families bypassed it, and with `allowed_guild_ids` empty (the
  shipped default) the tree syncs GLOBALLY, so every guild the bot was invited
  to plus DMs could reach `ops approve`, `ops gate`, `session workspace` and the
  provider overrides. Discord DMs additionally had no user allowlist at all.

THE FIX is structural, not per-handler: one Slack Bolt global middleware
(`app.use`) ahead of every listener, and one check in Discord's `_register`
callback that all 43 families funnel through. Both cover handlers added in
FUTURE, which a per-handler decorator would not.

WHY THESE TESTS RUN WITHOUT slack-bolt / discord.py INSTALLED: CI installs only
the web+anthropic+mcp extras, so a test that imports the gateway libs at module
scope would ERROR there — which is precisely how the audit's P3.5 finding
("security-floor tests never execute in CI") happens. So: the predicates are
tested against plain attribute stubs, the Slack middleware against an injected
fake `slack_bolt.response`, and the WIRING is guarded structurally via AST. The
Bolt middleware contract itself (a global middleware intercepts slash commands,
`body["channel_id"]` is present, returning a BoltResponse stops the listener)
was verified empirically against real slack-bolt 1.28 before this was written.

BOTH DIRECTIONS ARE ASSERTED (Standing-Principles §2c): a guard that only
refuses is half a control — the allowed origin must still get through, or the
fix locks the operator out of their own bot.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.gateway.config import Platform, PlatformConfig

REPO = Path(__file__).resolve().parents[1]
SLACK_SRC = REPO / "src" / "prometheus" / "gateway" / "slack.py"
DISCORD_SRC = REPO / "src" / "prometheus" / "gateway" / "discord.py"

def _slack_bolt_missing() -> bool:
    """True when slack-bolt is unavailable (CI installs web+anthropic+mcp only).

    Catches ImportError as well as a None spec: a half-installed or shadowed
    slack_bolt (a finder that raises, a broken .dist-info) must SKIP these
    tests, not error during collection. The unconditional guards above are what
    CI actually proves; this class only adds the real-framework layer where the
    lib is genuinely present.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("slack_bolt") is None
    except (ImportError, ValueError, AttributeError):
        return True


ALLOWED_CHANNEL = "C_OPERATOR"
STRANGER_CHANNEL = "C_STRANGER"
ALLOWED_GUILD = 111
ALLOWED_CHANNEL_ID = 222
STRANGER_GUILD = 999
STRANGER_CHANNEL_ID = 888
OPERATOR_USER = 4242
STRANGER_USER = 1337


# --------------------------------------------------------------------------- #
# Slack — the global middleware, tested directly (fake BoltResponse injected).
# --------------------------------------------------------------------------- #


@pytest.fixture()
def fake_bolt_response(monkeypatch):
    """Inject a minimal `slack_bolt.response` so the middleware is testable
    with the lib uninstalled (CI parity). Records every refusal constructed."""
    created: list = []

    class BoltResponse:
        def __init__(self, status=200, body=""):
            self.status = status
            self.body = body
            created.append(self)

    mod = types.ModuleType("slack_bolt.response")
    mod.BoltResponse = BoltResponse
    monkeypatch.setitem(sys.modules, "slack_bolt.response", mod)
    monkeypatch.setitem(
        sys.modules, "slack_bolt", types.ModuleType("slack_bolt")
    )
    return created


def _slack_adapter(allowed_channels):
    from prometheus.gateway.slack import SlackAdapter

    return SlackAdapter(
        config=PlatformConfig(
            platform=Platform.SLACK,
            token="xoxb-test",
            app_token="xapp-test",
            allowed_channels=allowed_channels,
        ),
        agent_loop=MagicMock(),
        tool_registry=MagicMock(),
        system_prompt="",
        model_name="m",
        model_provider="llama_cpp",
    )


class TestSlackMiddleware:
    @pytest.mark.asyncio
    async def test_allowed_channel_reaches_the_listener(self, fake_bolt_response):
        """THE LOCK-OUT DIRECTION: an allowed channel must pass through."""
        a = _slack_adapter([ALLOWED_CHANNEL])
        nxt = AsyncMock()
        await a._authorize_request({"channel_id": ALLOWED_CHANNEL}, nxt)
        nxt.assert_awaited_once()
        assert not fake_bolt_response, "an allowed channel got a refusal body"

    @pytest.mark.asyncio
    async def test_stranger_channel_never_reaches_the_listener(
        self, fake_bolt_response
    ):
        a = _slack_adapter([ALLOWED_CHANNEL])
        nxt = AsyncMock()
        out = await a._authorize_request({"channel_id": STRANGER_CHANNEL}, nxt)
        nxt.assert_not_awaited()
        assert isinstance(out, type(fake_bolt_response[0])), (
            "a refusal must return a BoltResponse, not await next()"
        )
        assert out.status == 200, (
            "Slack needs a 200 or it tells the user the command failed"
        )

    @pytest.mark.parametrize(
        "body",
        [
            {"channel_id": STRANGER_CHANNEL, "command": "/prometheus-approve"},
            {"channel_id": STRANGER_CHANNEL, "type": "event_callback"},
            {"channel_id": STRANGER_CHANNEL, "type": "shortcut"},
        ],
    )
    @pytest.mark.asyncio
    async def test_every_request_shape_is_gated(self, body, fake_bolt_response):
        """Slash commands, events and shortcuts alike — the point of GLOBAL
        middleware over a per-handler check."""
        a = _slack_adapter([ALLOWED_CHANNEL])
        nxt = AsyncMock()
        await a._authorize_request(body, nxt)
        nxt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_body_with_no_channel_is_let_through(self, fake_bolt_response):
        """url-verification / ssl_check / OAuth callbacks carry no channel and
        act on nothing. Fail-open here is deliberate and bounded."""
        a = _slack_adapter([ALLOWED_CHANNEL])
        nxt = AsyncMock()
        await a._authorize_request({"type": "url_verification"}, nxt)
        nxt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_allowlist_still_allows(self, fake_bolt_response):
        """NOT changed by this PR: empty allowed_channels keeps its
        empty-means-unrestricted reading. Flipping it to deny is the separate
        #219-shaped follow-up (needs boot refusal + doctor + wizard), so this
        PR must not silently alter it."""
        a = _slack_adapter([])
        nxt = AsyncMock()
        await a._authorize_request({"channel_id": STRANGER_CHANNEL}, nxt)
        nxt.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Discord — the interaction predicate, tested against attribute stubs.
# --------------------------------------------------------------------------- #


class _Interaction:
    """Minimal stand-in for discord.Interaction: only the id attributes the
    predicate reads. No lib import, so this runs in CI."""

    def __init__(self, *, guild_id, channel_id, user_id=None):
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.user = types.SimpleNamespace(id=user_id) if user_id is not None else None


def _discord_adapter(*, guild_ids=None, channel_ids=None, user_ids=None):
    from prometheus.gateway.discord import DiscordAdapter

    return DiscordAdapter(
        config=PlatformConfig(
            platform=Platform.DISCORD,
            token="t",
            allowed_guild_ids=guild_ids or [],
            allowed_channel_ids=channel_ids or [],
            allowed_user_ids=user_ids or [],
        ),
        agent_loop=MagicMock(),
        tool_registry=MagicMock(),
    )


class TestDiscordInteractionAllowance:
    def test_allowed_guild_passes(self):
        a = _discord_adapter(guild_ids=[ALLOWED_GUILD])
        assert a._interaction_allowed(
            _Interaction(guild_id=ALLOWED_GUILD, channel_id=1)
        ) is True

    def test_allowed_channel_passes(self):
        a = _discord_adapter(channel_ids=[ALLOWED_CHANNEL_ID])
        assert a._interaction_allowed(
            _Interaction(guild_id=STRANGER_GUILD, channel_id=ALLOWED_CHANNEL_ID)
        ) is True

    def test_stranger_guild_is_refused(self):
        """THE CRITICAL: a non-whitelisted guild must not reach ops approve."""
        a = _discord_adapter(guild_ids=[ALLOWED_GUILD])
        assert a._interaction_allowed(
            _Interaction(guild_id=STRANGER_GUILD, channel_id=STRANGER_CHANNEL_ID)
        ) is False

    def test_dm_with_empty_user_allowlist_is_allowed(self):
        """Historical posture preserved: empty allowed_user_ids = any DM.
        The deny-by-default flip is the deliberate follow-up."""
        a = _discord_adapter()
        assert a._interaction_allowed(
            _Interaction(guild_id=None, channel_id=5, user_id=STRANGER_USER)
        ) is True

    def test_dm_user_allowlist_enforced_when_set(self):
        a = _discord_adapter(user_ids=[OPERATOR_USER])
        assert a._interaction_allowed(
            _Interaction(guild_id=None, channel_id=5, user_id=OPERATOR_USER)
        ) is True
        assert a._interaction_allowed(
            _Interaction(guild_id=None, channel_id=5, user_id=STRANGER_USER)
        ) is False

    def test_dm_with_no_attributable_user_is_refused_when_gated(self):
        """Fail-CLOSED on an interaction we cannot attribute, once a user
        allowlist exists."""
        a = _discord_adapter(user_ids=[OPERATOR_USER])
        assert a._interaction_allowed(
            _Interaction(guild_id=None, channel_id=5, user_id=None)
        ) is False

    def test_empty_everything_refuses_guild_commands(self):
        """The shipped default (guild_ids: [], channel_ids: []) means DMs-only,
        so a GUILD command must be refused — this is what closes the
        global-sync exposure."""
        a = _discord_adapter()
        assert a._interaction_allowed(
            _Interaction(guild_id=STRANGER_GUILD, channel_id=STRANGER_CHANNEL_ID)
        ) is False


# --------------------------------------------------------------------------- #
# Structural wiring — the property must be enforced, not remembered.
# --------------------------------------------------------------------------- #


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


class TestWiringIsStructural:
    def test_slack_registers_global_middleware_before_any_listener(self):
        """`app.use(self._authorize_request)` must precede the first
        `app.command(...)` / `app.event(...)` in start(), or a listener can run
        before the gate."""
        tree = _tree(SLACK_SRC)
        first_use = first_listener = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            recv = node.func.value
            if not (isinstance(recv, ast.Attribute) and recv.attr == "_app"):
                continue
            attr = node.func.attr
            if attr == "use" and first_use is None:
                first_use = node.lineno
            elif attr in ("command", "event") and first_listener is None:
                first_listener = node.lineno
        assert first_use is not None, "no app.use(...) registration found"
        assert first_listener is not None, "no listeners found — guard is moot"
        assert first_use < first_listener, (
            f"app.use (line {first_use}) must come BEFORE the first listener "
            f"(line {first_listener}) or listeners can run ungated"
        )

    def test_slack_middleware_passes_the_gate_fn(self):
        src = SLACK_SRC.read_text(encoding="utf-8")
        assert "self._app.use(self._authorize_request)" in src

    def test_discord_register_callback_gates_the_handler(self):
        """Every family is attached through `_register`; its `_callback` must
        call `_interaction_allowed` BEFORE `handler(...)`. That single chokepoint
        is what makes all 43 families safe by construction."""
        tree = _tree(DISCORD_SRC)
        reg = None
        for node in ast.walk(tree):
            # `_register` is a SYNC method whose inner `_callback` is async —
            # match either FunctionDef form so the guard does not silently
            # pass by never finding the node.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_register":
                reg = node
                break
        assert reg is not None, "DiscordAdapter._register not found"

        gate_line = handler_line = None
        for node in ast.walk(reg):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name == "_interaction_allowed" and gate_line is None:
                    gate_line = node.lineno
                if isinstance(fn, ast.Name) and fn.id == "handler" and handler_line is None:
                    handler_line = node.lineno
        assert gate_line is not None, (
            "_register's callback never calls _interaction_allowed — the "
            "families are ungated (the P2.2/P2.3 regression)"
        )
        assert handler_line is not None, "_register no longer calls the handler"
        assert gate_line < handler_line, (
            f"the gate (line {gate_line}) must run before the handler "
            f"(line {handler_line})"
        )

    def test_every_discord_family_goes_through_register(self):
        """No family may attach itself to the tree bypassing `_register`."""
        src = DISCORD_SRC.read_text(encoding="utf-8")
        assert src.count("self._register(") >= 40, (
            "expected the 43 families to be attached via self._register(...); "
            "a drop suggests a family now bypasses the chokepoint"
        )
        tree = _tree(DISCORD_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("add_command", "command") and isinstance(
                    node.func.value, ast.Attribute
                ):
                    recv = node.func.value.attr
                    if recv == "_tree":
                        pytest.fail(
                            f"line {node.lineno}: a command is attached straight "
                            f"to self._tree, bypassing the _register gate"
                        )


# --------------------------------------------------------------------------- #
# The DM allowlist must apply to the MESSAGE path too, not only commands.
#
# A `user_ids` list that gated app commands while a stranger could still DM the
# bot a plain message — and get a full agent turn with shell — would be a
# half-fix that READS as a complete one. One predicate, both paths.
# --------------------------------------------------------------------------- #


class TestDiscordDMAllowlistIsCoherent:
    def test_dm_message_from_allowed_user_passes(self):
        cfg = PlatformConfig(
            platform=Platform.DISCORD, allowed_user_ids=[OPERATOR_USER]
        )
        assert cfg.discord_inbound_allowed(
            is_dm=True, guild_id=None, channel_id=5, user_id=OPERATOR_USER
        ) is True

    def test_dm_message_from_stranger_is_refused_when_gated(self):
        """The whole point: a populated user_ids closes the DM MESSAGE path,
        which is the path that hands a stranger an agent turn with tools."""
        cfg = PlatformConfig(
            platform=Platform.DISCORD, allowed_user_ids=[OPERATOR_USER]
        )
        assert cfg.discord_inbound_allowed(
            is_dm=True, guild_id=None, channel_id=5, user_id=STRANGER_USER
        ) is False

    def test_dm_with_no_user_id_fails_closed_when_gated(self):
        cfg = PlatformConfig(
            platform=Platform.DISCORD, allowed_user_ids=[OPERATOR_USER]
        )
        assert cfg.discord_inbound_allowed(
            is_dm=True, guild_id=None, channel_id=5, user_id=None
        ) is False

    def test_dm_empty_allowlist_still_allows_messages(self):
        """Unchanged historical posture — this PR must not silently flip it."""
        cfg = PlatformConfig(platform=Platform.DISCORD)
        assert cfg.discord_inbound_allowed(
            is_dm=True, guild_id=None, channel_id=5, user_id=STRANGER_USER
        ) is True

    def test_guild_path_ignores_user_allowlist(self):
        """A guild whitelist hit is sufficient on its own; user_ids is the DM
        control and must not start refusing whitelisted guild channels."""
        cfg = PlatformConfig(
            platform=Platform.DISCORD,
            allowed_guild_ids=[ALLOWED_GUILD],
            allowed_user_ids=[OPERATOR_USER],
        )
        assert cfg.discord_inbound_allowed(
            is_dm=False, guild_id=ALLOWED_GUILD, channel_id=1,
            user_id=STRANGER_USER,
        ) is True

    def test_command_and_message_paths_agree(self):
        """One predicate answers both paths — asserted, not assumed. If the
        adapter's interaction gate and the config predicate ever diverge, a
        stranger gets one path but not the other."""
        a = _discord_adapter(user_ids=[OPERATOR_USER])
        for uid, expected in ((OPERATOR_USER, True), (STRANGER_USER, False), (None, False)):
            via_command = a._interaction_allowed(
                _Interaction(guild_id=None, channel_id=5, user_id=uid)
            )
            via_message = a.config.discord_inbound_allowed(
                is_dm=True, guild_id=None, channel_id=5, user_id=uid
            )
            assert via_command is expected and via_message is expected, (
                f"user {uid}: command path {via_command}, message path "
                f"{via_message}, expected {expected}"
            )


class TestDaemonThreadsTheUserAllowlist:
    """A PlatformConfig field no construction site populates is config-dark —
    the defect class this repo's drift guard cannot see (the key is present and
    read; only the wiring is missing)."""

    def test_daemon_passes_allowed_user_ids_from_gateway_discord(self):
        src = (REPO / "src" / "prometheus" / "daemon.py").read_text(encoding="utf-8")
        assert "allowed_user_ids=" in src, (
            "daemon.py never threads allowed_user_ids into the Discord "
            "PlatformConfig — gateway.discord.user_ids would be inert"
        )
        assert '"user_ids"' in src or "'user_ids'" in src, (
            "daemon.py does not read the gateway.discord.user_ids config key"
        )

    def test_template_documents_user_ids(self):
        tmpl = (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
        assert "user_ids:" in tmpl, (
            "the new control is undocumented in the shipped template — an "
            "operator cannot discover it (the FL-2 class)"
        )


class TestBothDiscordCallSitesPassUserId:
    """The predicate only protects DMs if every caller names the sender.

    `discord_inbound_allowed(user_id=None)` fails CLOSED when a user allowlist
    is populated — correct for an unattributable interaction, but it would also
    mean a caller that simply forgot the argument locks the operator out of
    their own DMs. Both message-path call sites must pass it; this guard keeps
    them honest, since neither is reachable without discord.py installed.
    """

    def test_every_discord_inbound_allowed_call_passes_user_id(self):
        tree = _tree(DISCORD_SRC)
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "discord_inbound_allowed"
        ]
        assert calls, "no discord_inbound_allowed call sites found — guard is moot"
        missing = [
            n.lineno for n in calls
            if not any(kw.arg == "user_id" for kw in n.keywords)
        ]
        assert not missing, (
            f"discord_inbound_allowed called WITHOUT user_id at line(s) "
            f"{missing} — the DM user allowlist cannot apply there, so a "
            f"stranger keeps the message path (or the operator is locked out, "
            f"depending on whether the allowlist is populated)"
        )

    def test_user_id_is_threaded_through_raw_for_the_second_check(self):
        """on_message reads raw['user_id']; _handle_discord_message must put it
        there. Two call sites, one source of truth."""
        src = DISCORD_SRC.read_text(encoding="utf-8")
        assert '"user_id": getattr(author, "id", None)' in src, (
            "the author's id is not carried in `raw`, so on_message's check "
            "cannot see the sender"
        )
        assert 'user_id=raw.get("user_id")' in src, (
            "on_message does not pass the carried user_id to the predicate"
        )


# --------------------------------------------------------------------------- #
# The real Bolt contract — runs when slack-bolt is installed, skips in CI.
#
# CI installs only web+anthropic+mcp, so this SKIPs there; the unconditional
# guards above are what CI proves. This is the layer that proves the Bolt
# contract itself: that a GLOBAL middleware intercepts SLASH commands (not just
# events), that `body["channel_id"]` is present on them, and that returning a
# BoltResponse stops the listener — asserted against the REAL production
# bound method through a real AsyncApp.async_dispatch, not a hand-rolled stub.
# Verified locally against slack-bolt 1.28 before the fix was written.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    _slack_bolt_missing(),
    reason="slack-bolt not installed (CI installs web+anthropic+mcp only); "
           "the unconditional guards above cover CI",
)
class TestSlackMiddlewareAgainstRealBolt:
    """End-to-end through a real Bolt dispatch with the real adapter method."""

    @staticmethod
    def _dispatch(adapter, channel_id):
        import asyncio

        from slack_bolt.async_app import AsyncApp
        from slack_bolt.authorization import AuthorizeResult
        from slack_bolt.request.async_request import AsyncBoltRequest

        seen = {"handler": False}

        async def handler(ack, body, respond):
            seen["handler"] = True
            await ack()

        async def _auth(**kw):
            return AuthorizeResult(
                enterprise_id=None, team_id="T1", bot_id="B1", bot_user_id="U_BOT"
            )

        async def main():
            app = AsyncApp(
                token="xoxb-test", signing_secret="s",
                request_verification_enabled=False, ssl_check_enabled=False,
                url_verification_enabled=False, authorize=_auth,
            )
            app.use(adapter._authorize_request)   # the REAL bound method
            app.command("/prometheus-approve")(handler)
            body = {
                "token": "x", "team_id": "T1", "team_domain": "d",
                "channel_id": channel_id, "channel_name": "r",
                "user_id": "U1", "user_name": "u",
                "command": "/prometheus-approve", "text": "all",
                "response_url": "https://hooks.slack.com/x", "trigger_id": "t",
            }
            req = AsyncBoltRequest(
                body=body,
                headers={"content-type": ["application/x-www-form-urlencoded"]},
                mode="socket_mode",
            )
            resp = await app.async_dispatch(req)
            return seen["handler"], resp.status

        return asyncio.run(main())

    def test_a_stranger_channel_never_reaches_a_slash_handler(self):
        """THE FINDING, proven at the framework boundary: a workspace-global
        slash command from a non-allowed channel must not execute."""
        a = _slack_adapter([ALLOWED_CHANNEL])
        ran, status = self._dispatch(a, STRANGER_CHANNEL)
        assert ran is False, "the slash handler ran for a non-allowed channel"
        assert status == 200, (
            "Slack must get a 200 or it tells the user the command failed and "
            "invites retries"
        )

    def test_an_allowed_channel_still_reaches_the_handler(self):
        """The lock-out direction, at the framework boundary."""
        a = _slack_adapter([ALLOWED_CHANNEL])
        ran, _status = self._dispatch(a, ALLOWED_CHANNEL)
        assert ran is True, "the operator's own channel was locked out"
