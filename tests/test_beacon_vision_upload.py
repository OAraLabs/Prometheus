"""Beacon image upload → vision description, through the REAL bridge.

WHY THIS FILE EXISTS
--------------------
#230 changed ``WebSocketBridge._describe_image`` and merged with **zero test
files** and green CI on two Python versions. The change could not work: it
read ``self._config``, an attribute that does not exist (the real one is
``self.config``), and the ``if self._config else`` guard could not save it
because evaluating the name for truthiness IS the AttributeError. A bare
``except Exception`` turned that programming error into a log line and a
``None``, so **every** beacon image upload silently produced no description
and nothing anywhere went red.

A second defect sat underneath, masked by the first: the same line did
``Path(self._config.workspace_root)``, and ``security.workspace_root`` is a
LIST on any multi-root install — ``Path(list)`` is a TypeError. Fixing only
the first would have surfaced the second in production.

WHAT THIS SUITE ASSERTS, AND WHY IT IS SHAPED THIS WAY
-----------------------------------------------------
"It did not raise" and "the method exists" are exactly the assertions that
would have passed against the broken code. The defect's signature was a
quiet ``None``, so the load-bearing assertion here is **a non-empty
description came back** through a real ``WebSocketBridge`` instance.

The only double is the PROVIDER — an external model service, the one thing
the project's rules say to mock. Everything between the bridge and it is
real: real bridge, real ``media_services``, real ``VisionTool``, real
``ToolExecutionContext``, real PNG on disk.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiMessageCompleteEvent, UsageSnapshot
from prometheus.web.ws_server import WebSocketBridge

DESCRIPTION = "A small solid red square, 1 by 1 pixel."

# Smallest valid PNG — a single red pixel. Written to disk so VisionTool's
# real read/encode path runs rather than being stubbed.
_RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


class StubProvider:
    """Stands in for the multimodal model — the external service, nothing else.

    Records the request so a test can prove the image actually reached the
    provider rather than the call being short-circuited somewhere.
    """

    def __init__(self, text: str = DESCRIPTION) -> None:
        self.text = text
        self.requests: list[object] = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self.text)]),
            usage=UsageSnapshot(),
        )


@pytest.fixture
def image(tmp_path: Path) -> str:
    p = tmp_path / "upload.png"
    p.write_bytes(_RED_PIXEL_PNG)
    return str(p)


def _bridge(loop_context) -> WebSocketBridge:
    """A real WebSocketBridge, constructed the way the daemon constructs it."""
    return WebSocketBridge(
        signal_bus=None,
        session_mgr=None,
        loop_context=loop_context,
        config={},
    )


class _Ctx:
    """Minimal stand-in for LoopContext carrying only what vision needs."""

    def __init__(self, provider=None):
        self.provider = provider


# --------------------------------------------------------------------------- #
# THE LOAD-BEARING TEST — a description comes back, non-empty
# --------------------------------------------------------------------------- #


def test_beacon_upload_returns_a_non_empty_description(image):
    """The assertion #230 needed and did not have.

    Against the merged #230 this returns None (AttributeError, swallowed).
    Against a 'registration' or 'did not raise' assertion, #230 passes.
    Only a non-empty result distinguishes them."""
    provider = StubProvider()
    bridge = _bridge(_Ctx(provider))

    result = asyncio.run(bridge._describe_image(image))

    assert result is not None, "vision returned None — the #230 failure mode"
    assert result.strip(), "vision returned an empty description"
    assert result == DESCRIPTION
    assert provider.requests, "the image never reached the provider"


def test_provider_actually_received_the_image_bytes(image):
    """Guard identity: prove the PNG travelled, not just that a string came back.

    Without this, a fix that returned a canned string from anywhere would
    satisfy the test above."""
    provider = StubProvider()
    bridge = _bridge(_Ctx(provider))

    asyncio.run(bridge._describe_image(image))

    req = provider.requests[0]
    blocks = req.messages[0]["content"]
    kinds = {b.get("type") for b in blocks}
    assert "image_url" in kinds, f"no image block in the request: {kinds}"
    url = next(b["image_url"]["url"] for b in blocks if b.get("type") == "image_url")
    assert url.startswith("data:image/"), url[:40]
    assert base64.b64encode(_RED_PIXEL_PNG).decode() in url


def test_no_provider_degrades_to_none_not_a_crash(image):
    """A bridge with no provider wired is a legitimate 'vision unavailable'.

    The admission/breach pair for the test above: this MUST be None, so a
    fix that hardcodes a non-empty return fails here."""
    bridge = _bridge(_Ctx(None))
    assert asyncio.run(bridge._describe_image(image)) is None


# --------------------------------------------------------------------------- #
# Defect 2 — list-valued workspace_root, exercised explicitly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "workspace_root",
    [
        pytest.param(["~/projects", "~/.prometheus", "/tmp"], id="list-multi-root"),
        pytest.param(["/tmp"], id="list-single-root"),
        pytest.param("/tmp", id="str-root"),
        pytest.param(None, id="absent"),
    ],
)
def test_workspace_root_shape_cannot_break_vision(image, workspace_root):
    """``Path(list)`` is a TypeError, and the live config IS a list.

    The fix removes the workspace_root read entirely — vision resolves one
    already-cached absolute path and needs no confinement root — so every
    shape below must behave identically. Parametrised over the shapes rather
    than the one the author happened to have."""
    provider = StubProvider()
    ctx = _Ctx(provider)
    ctx.workspace_root = workspace_root  # noqa: SLF001 - deliberately shaped
    bridge = _bridge(ctx)
    bridge.config = {"security": {"workspace_root": workspace_root}}

    result = asyncio.run(bridge._describe_image(image))

    assert result == DESCRIPTION, (
        f"workspace_root={workspace_root!r} changed the outcome; vision must "
        f"not read it at all"
    )


def test_describe_image_does_not_read_workspace_root_at_all():
    """Total invariant, not a shape survey: the source must not touch it.

    Cheaper and stronger than enumerating shapes — if the reader is gone,
    no shape can ever break it again. ``resolve_workspace_root`` is the
    canonical resolver and this site was the only undocumented bypass."""
    import inspect

    src = inspect.getsource(WebSocketBridge._describe_image)
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("*")
    )
    body = code.split('"""')[-1]  # strip the docstring, which discusses it
    assert "workspace_root" not in body, (
        "vision reads workspace_root again — it must not; see "
        "config/shipped_defaults.resolve_workspace_root"
    )


# --------------------------------------------------------------------------- #
# Defect 3 — a programming error must not read as a failed analysis
# --------------------------------------------------------------------------- #


def test_programming_error_in_setup_propagates(image, monkeypatch):
    """#78's broad-except shape, and the reason #230 shipped silently.

    A bug in OUR setup code (import, construction, context building) is not
    'vision unavailable' — it must escape, not become a ``None``. Simulated
    by breaking the tool constructor, which lives in the un-guarded region.

    Against the pre-fix ``media_services`` (whole body inside ``try``) this
    returns None and the test goes red."""
    import prometheus.tools.builtin.vision as vision_mod

    class Exploding(vision_mod.VisionTool):
        def __init__(self, *a, **k):
            raise AttributeError("simulated programming error in setup")

    monkeypatch.setattr(vision_mod, "VisionTool", Exploding)

    bridge = _bridge(_Ctx(StubProvider()))
    with pytest.raises(AttributeError, match="simulated programming error"):
        asyncio.run(bridge._describe_image(image))


def test_tool_failure_is_swallowed_but_loud(image, monkeypatch, caplog):
    """The other direction: a genuine tool failure still degrades to None.

    Environmental failure (no mmproj, model down) is a legitimate reason to
    return None — but it must no longer be invisible. Pre-fix this logged at
    DEBUG; it now logs at WARNING with a traceback."""
    import prometheus.tools.builtin.vision as vision_mod

    async def boom(self, arguments, context):
        raise RuntimeError("model backend unreachable")

    monkeypatch.setattr(vision_mod.VisionTool, "execute", boom)

    bridge = _bridge(_Ctx(StubProvider()))
    with caplog.at_level(logging.WARNING, logger="prometheus.gateway.media_services"):
        result = asyncio.run(bridge._describe_image(image))

    assert result is None
    assert any(
        r.levelno >= logging.WARNING and "Vision analysis failed" in r.getMessage()
        for r in caplog.records
    ), "a swallowed tool failure must be logged at WARNING, not DEBUG"
    assert any(r.exc_info for r in caplog.records), "the traceback must be attached"


def test_the_bridge_owns_no_try_except():
    """The failure policy lives in one place, for all three surfaces.

    Telegram, Discord and Beacon all route through ``media_services``; a
    second ``except`` in the bridge would let Beacon drift from the other
    two again, which is precisely how #230 happened."""
    import inspect

    body = inspect.getsource(WebSocketBridge._describe_image).split('"""')[-1]
    assert "except" not in body, "the bridge must not re-implement failure policy"
