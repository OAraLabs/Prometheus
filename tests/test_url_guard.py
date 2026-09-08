"""P4.2 — one SSRF predicate, applied to every redirect hop.

THE DEFECT (audit, verified-high, three call sites):

1. The guard ran ONCE on the caller-supplied URL, then httpx followed the whole
   30x chain with ``follow_redirects=True`` and nothing looked at where it landed.
   A public URL that 302s to ``http://127.0.0.1:8005/api/…`` — the daemon's own
   REST surface, which includes bash — was fetched and its body returned to the
   model or written to disk. The tool advertised "Blocked: URL resolves to a
   private or reserved IP address" while the redirect walked straight past it.

2. The four flags it checked (is_private, is_loopback, is_reserved, is_link_local)
   miss addresses that are not publicly routable but are in none of those classes.
   Measured against this checkout's interpreter:

       100.64.0.1        all four False   -> NOT caught
       ::ffff:100.64.0.1 all four False   -> NOT caught
       ff02::1           all four False   -> NOT caught (multicast)

3. Any scheme was accepted: ftp:// and gopher:// had a hostname and passed.

There were TWO independent copies of that body (web_fetch and download_file) plus
a third consumer importing one of them (web/references.py @url). One predicate now
lives in security/url_guard.py and all three call it — fixing one copy and leaving
two is the shape this repo keeps paying for.

SCOPE, per the operator's CAREFUL ruling: "check the post-redirect host, not just
the pre-redirect one. Do NOT add a domain allowlist." There is no allowlist here
and no restriction on which public URLs may be fetched.

MECHANISM, VERIFIED RATHER THAN ASSUMED. httpx 0.28 runs request event hooks
inside ``_send_handling_redirects``'s ``while True`` loop, so a hook sees every hop
and raising there aborts the chain with the body unread. Both halves were checked
against a MockTransport before this was written: an async hook saw both hops and
the secret body never arrived; a SYNC hook raised TypeError (AsyncClient requires
async hooks) — which is why the hook below is async.
"""

from __future__ import annotations

import ipaddress

import httpx
import pytest

from prometheus.security.url_guard import (
    ALLOWED_SCHEMES,
    ENFORCE_TAILNET_BLOCK,
    TAILNET_RANGE,
    check_url,
    is_blocked_address,
    is_ip_literal_blocked,
    is_safe_url,
)

SECRET_BODY = "SECRET-BODY-MUST-NOT-BE-RETURNED"

#: The mock hosts below are not real DNS names, and ``check_url`` resolves
#: hostnames for real — so without this they would all be refused at hop 1 for
#: "the host does not resolve" and every redirect test would pass VACUOUSLY,
#: proving nothing about per-hop guarding. Making the public-looking example hosts
#: resolve to a public address is what lets the tests fail only for the reason
#: under test.
PUBLIC_IP = "93.184.216.34"
MOCK_HOSTS = {"public.example", "also-public.example"}


@pytest.fixture(autouse=True)
def _resolve_mock_hosts(monkeypatch):
    """Let the mock hosts resolve to a public IP; everything else hits real DNS."""
    import prometheus.security.url_guard as g

    real = g.socket.getaddrinfo

    def fake(host, *args, **kwargs):
        if host in MOCK_HOSTS:
            return [(2, 1, 6, "", (PUBLIC_IP, 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(g.socket, "getaddrinfo", fake)


# --------------------------------------------------------------------------- #
# 1 — the addresses the old four-flag check missed.
# --------------------------------------------------------------------------- #


class TestAddressesTheOldCheckMissed:
    @pytest.mark.parametrize("addr", [
        "127.0.0.1", "127.0.0.53", "::1", "::",
        "10.0.0.5", "172.16.4.4", "192.168.1.10",
        "169.254.169.254",                      # cloud metadata
        "fe80::1", "fc00::1",
        "ff02::1",                              # multicast — no is_multicast flag before
        "0.0.0.0",                              # unspecified
    ])
    def test_non_public_addresses_are_blocked(self, addr):
        assert is_blocked_address(ipaddress.ip_address(addr)) is True

    @pytest.mark.parametrize("addr", [
        "::ffff:127.0.0.1", "::ffff:10.0.0.5", "::ffff:169.254.169.254",
    ])
    def test_ipv4_mapped_ipv6_is_unwrapped_before_deciding(self, addr):
        """A mapped address inherits the gap of what it maps to. Unwrapped first,
        so ::ffff:<loopback> is blocked even though the IPv6 view of it is not."""
        assert is_blocked_address(ipaddress.ip_address(addr)) is True

    @pytest.mark.parametrize("addr", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
    def test_public_addresses_are_allowed(self, addr):
        """The other direction: a guard that blocks everything is an outage, not
        a control."""
        assert is_blocked_address(ipaddress.ip_address(addr)) is False


class TestTheTailnetDecisionIsExplicit:
    """The CGNAT range is two-sided and the operator has to choose.

    It is how a fetched page could reach the daemon's own :8005 or the GPU box's
    unauthenticated ollama/whisper/ComfyUI ports — and it is also how the operator
    legitimately points this tool at his own services. So the range is DEFINED and
    MEASURED here, the default is OFF, and this test pins the choice so flipping
    it is a deliberate edit that fails something, not a silent one.
    """

    def test_python_does_not_treat_cgnat_as_private(self):
        """The premise of the whole finding, measured: is_private does NOT cover
        100.64.0.0/10, so the old check passed every tailnet host."""
        for addr in ("100.64.0.1", "100.64.0.2"):
            ip = ipaddress.ip_address(addr)
            assert not (ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local)
            assert ip in TAILNET_RANGE

    def test_the_current_choice_is_recorded(self):
        if ENFORCE_TAILNET_BLOCK:
            assert is_blocked_address(ipaddress.ip_address("100.64.0.1")) is True
        else:
            assert is_blocked_address(ipaddress.ip_address("100.64.0.1")) is False
            assert is_safe_url("http://100.64.0.2:11434/") is True


# --------------------------------------------------------------------------- #
# 2 — check_url: scheme, host, resolution.
# --------------------------------------------------------------------------- #


class TestCheckUrl:
    @pytest.mark.parametrize("url,want", [
        ("https://example.com/", True),
        ("http://example.com/path?q=1", True),
        ("http://127.0.0.1:8005/api/status", False),
        ("http://localhost:8005/", False),
        ("http://[::1]:8005/", False),
        ("http://10.0.0.5/x", False),
        ("http://192.168.1.10/", False),
        ("http://169.254.169.254/latest/meta-data/", False),
        ("http://ff02::1/x", False),
    ])
    def test_addresses(self, url, want):
        ok, reason = check_url(url)
        assert ok is want
        if not want:
            assert reason, "a refusal must carry a reason"

    @pytest.mark.parametrize("scheme", sorted(ALLOWED_SCHEMES))
    def test_only_http_and_https_are_fetchable(self, scheme):
        ok, _ = check_url(f"{scheme}://example.com/")
        assert ok is True

    @pytest.mark.parametrize("url", [
        "file:///home/will/notes.txt", "ftp://internal.example/x",
        "gopher://127.0.0.1:6379/_INFO", "data:text/html,<script>1</script>",
        "javascript:alert(1)",
    ])
    def test_other_schemes_are_refused_not_passed_through(self, url):
        """These used to reach the transport: anything with a hostname passed, and
        file:// fell through only by accident of ordering."""
        ok, reason = check_url(url)
        assert ok is False
        assert "scheme" in reason or "host" in reason

    @pytest.mark.parametrize("url", ["", "   ", "not a url", "http://", "//example.com"])
    def test_degenerate_input_is_refused(self, url):
        assert check_url(url)[0] is False

    def test_unresolvable_host_is_refused_not_treated_as_safe(self):
        """Fail-closed on a resolution error. Treating it as safe would invert the
        whole control, and an unresolvable host is not fetchable anyway."""
        ok, reason = check_url("http://nonexistent.invalid/x")
        assert ok is False
        assert "resolve" in reason

    def test_a_name_with_one_public_and_one_private_answer_is_refused(self, monkeypatch):
        """EVERY resolved address is checked. A host answering with both a public
        and a loopback address must not pass on the strength of the public one."""
        import prometheus.security.url_guard as g

        def fake_getaddrinfo(host, *a, **k):
            return [
                (2, 1, 6, "", ("93.184.216.34", 0)),
                (2, 1, 6, "", ("127.0.0.1", 0)),
            ]

        monkeypatch.setattr(g.socket, "getaddrinfo", fake_getaddrinfo)
        assert check_url("http://split.example/")[0] is False

    def test_ip_literal_host_is_checked_without_dns(self, monkeypatch):
        """A literal needs no resolution, and must not be able to slip past by
        failing one."""
        import prometheus.security.url_guard as g

        def boom(*a, **k):
            raise AssertionError("an IP literal must not need getaddrinfo")

        monkeypatch.setattr(g.socket, "getaddrinfo", boom)
        assert check_url("http://127.0.0.1/x")[0] is False
        assert is_ip_literal_blocked("127.0.0.1") is True
        assert is_ip_literal_blocked("example.com") is False


# --------------------------------------------------------------------------- #
# 3 — THE FINDING: a redirect hop is checked, and the body never arrives.
# --------------------------------------------------------------------------- #


def _client(handler, hooks):
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        event_hooks=hooks,
    )


class TestRedirectHopsAreGuarded:
    @pytest.mark.asyncio
    async def test_a_public_url_redirecting_to_loopback_is_blocked(self):
        from prometheus.security.url_guard import SsrfBlocked, guard_request_hop

        hops: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:8005/api/status"})
            return httpx.Response(200, text=SECRET_BODY)

        async def record(request):
            hops.append(str(request.url))

        client = _client(handler, {"request": [record, guard_request_hop]})
        with pytest.raises(SsrfBlocked) as exc:
            await client.get("http://public.example/")
        await client.aclose()

        assert hops == ["http://public.example/", "http://127.0.0.1:8005/api/status"], (
            "the hook must see EVERY hop — that is the whole mechanism"
        )
        # The reason must be the ADDRESS, not a DNS failure. Without this the
        # test passes vacuously if hop 1 cannot resolve.
        assert "non-public" in str(exc.value), (
            f"blocked for the wrong reason — {exc.value!r} is a DNS failure, not "
            f"the loopback hop this test is about"
        )

    @pytest.mark.asyncio
    async def test_the_secret_body_is_never_returned(self):
        """The point is not that an exception is raised, it is that the body never
        reaches the caller. Asserted on what comes back, not on the raise."""
        from prometheus.security.url_guard import SsrfBlocked, guard_request_hop

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
            return httpx.Response(200, text=SECRET_BODY)

        client = _client(handler, {"request": [guard_request_hop]})
        body, reason = None, None
        try:
            r = await client.get("http://public.example/")
            body = r.text
        except SsrfBlocked as exc:
            reason = str(exc)
        finally:
            await client.aclose()
        assert body is None, f"the body arrived anyway: {body!r}"
        assert reason is not None and "non-public" in reason, (
            f"refused for the wrong reason: {reason!r}"
        )

    @pytest.mark.asyncio
    async def test_a_multi_hop_chain_is_blocked_at_the_bad_hop(self):
        from prometheus.security.url_guard import SsrfBlocked, guard_request_hop

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host == "public.example":
                return httpx.Response(302, headers={"Location": "http://also-public.example/x"})
            if host == "also-public.example":
                return httpx.Response(302, headers={"Location": "http://10.0.0.5/internal"})
            return httpx.Response(200, text=SECRET_BODY)

        client = _client(handler, {"request": [guard_request_hop]})
        with pytest.raises(SsrfBlocked) as exc:
            await client.get("http://public.example/")
        await client.aclose()
        assert "non-public" in str(exc.value), f"wrong reason: {exc.value!r}"

    @pytest.mark.asyncio
    async def test_a_chain_that_stays_public_is_not_blocked(self):
        """The lock-out direction. Redirects are ordinary; a guard that refuses
        them all breaks normal fetching."""
        from prometheus.security.url_guard import guard_request_hop

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "https://also-public.example/final"})
            return httpx.Response(200, text="public content")

        client = _client(handler, {"request": [guard_request_hop]})
        try:
            r = await client.get("http://public.example/")
            assert r.status_code == 200 and r.text == "public content"
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_the_hook_must_be_async(self):
        """VERIFIED, not assumed: AsyncClient raises TypeError on a sync request
        hook. If someone rewrites guard_request_hop as `def`, every guarded fetch
        dies with a confusing TypeError instead of guarding — so pin it."""
        import inspect

        from prometheus.security.url_guard import guard_request_hop

        assert inspect.iscoroutinefunction(guard_request_hop), (
            "guard_request_hop must be async — httpx.AsyncClient rejects sync hooks"
        )


# --------------------------------------------------------------------------- #
# 4 — through the REAL fetch path, not by calling the hook directly.
#
# The tests above exercise the hook against a client THEY build. That proves the
# hook works and proves nothing about whether production installs it — removing
# `event_hooks=` from fetch_url_text leaves every one of them green, which is the
# same failure shape test_gate_sees_the_path.py exists to stop: a test supplying
# the wiring the caller never supplied. These drive fetch_url_text itself, with
# the transport forced to a mock so no network is touched.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def mock_transport(monkeypatch):
    """Force every httpx.AsyncClient built by the code under test onto a
    MockTransport, while leaving its event_hooks alone — that is the whole
    point: the production hooks must be the ones that run."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(factory.handler)
        return real_client(*args, **kwargs)

    factory.handler = lambda request: httpx.Response(200, text="unused")
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return factory


class TestFetchUrlTextInstallsTheGuard:
    @pytest.mark.asyncio
    async def test_a_redirect_to_loopback_is_refused_by_fetch_url_text(self, mock_transport):
        from prometheus.tools.builtin.web_fetch import SsrfBlocked, fetch_url_text

        def handler(request):
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:8005/api/status"})
            return httpx.Response(200, text=SECRET_BODY)

        mock_transport.handler = handler
        with pytest.raises(SsrfBlocked) as exc:
            await fetch_url_text("http://public.example/", max_chars=1000)
        assert "non-public" in str(exc.value), (
            "refused for the wrong reason — a DNS failure here would mean the "
            "guard never saw the loopback hop"
        )

    @pytest.mark.asyncio
    async def test_the_body_never_reaches_the_caller(self, mock_transport):
        """Asserted on what fetch_url_text RETURNS, which is what enters model
        context — not on an internal raise."""
        from prometheus.tools.builtin.web_fetch import SsrfBlocked, fetch_url_text

        def handler(request):
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://10.0.0.5/internal"})
            return httpx.Response(200, text=SECRET_BODY)

        mock_transport.handler = handler
        page = None
        try:
            page = await fetch_url_text("http://public.example/", max_chars=1000)
        except SsrfBlocked:
            pass
        assert page is None, f"the body reached the caller: {page!r}"
        assert SECRET_BODY not in (page.body if page else "")

    @pytest.mark.asyncio
    async def test_a_public_chain_still_fetches(self, mock_transport):
        """The lock-out direction, through the real function: redirects are
        ordinary and must keep working."""
        from prometheus.tools.builtin.web_fetch import fetch_url_text

        def handler(request):
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "https://also-public.example/final"})
            return httpx.Response(200, text="public content", headers={"content-type": "text/plain"})

        mock_transport.handler = handler
        page = await fetch_url_text("http://public.example/", max_chars=1000)
        assert page.status == 200
        assert "public content" in page.body
