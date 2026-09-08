"""P4.3 — a URL cannot choose where download_file writes.

THE DEFECT (audit, verified-high). ``_filename_from_url`` took the last path
segment FIRST and unquoted it AFTERWARDS::

    name = unquote(Path(parsed.path).name)

With a percent-encoded separator that inverts the intent: ``Path(
'/x%2F..%2F..%2F..%2F.bashrc').name`` is the single segment
``'x%2F..%2F..%2F..%2F.bashrc'``, and unquoting afterwards turns the ``%2F``s into
real separators. The caller then did ``_DOWNLOAD_DIR / name``, so the URL chose
the write target — measured to resolve to ``$HOME/.bashrc``. Because
``destination`` is optional and the gate has nothing to rule on when it is unset,
there was no gate prompt; and the forbidden-prefix list (six locations) never saw
the escape, so it did not help either.

A page the model was asked to fetch could therefore aim this tool at
``~/.bashrc``, ``~/.profile`` or ``~/.config/systemd/user/*.service`` — one
injected instruction from code execution on the operator's next shell.

TWO LAYERS, BOTH PINNED. ``_filename_from_url`` unquotes before reducing to a
basename (a basename can never contain a separator), and ``_resolve_destination``
asserts containment for URL-derived names. The second is the property the first
implements: a denylist of six protected paths cannot express "stays inside the
download directory", which is what actually matters here.

NOT tested by restricting URLs — the same URL still downloads, it just lands under
the download directory with its own name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.tools.builtin.download_file import (
    _DOWNLOAD_DIR,
    _filename_from_url,
    _resolve_destination,
)

# The audit's own payloads, plus the shapes that follow from them.
TRAVERSAL_URLS = [
    "https://evil.example/x%2F..%2F..%2F..%2F.bashrc",
    "https://evil.example/a%2F.config%2Fsystemd%2Fuser%2Fpwn.service",
    "https://evil.example/%2F..%2F..%2F.bashrc",
    "https://evil.example/..%2F..%2F.ssh%2Fauthorized_keys",
    "https://evil.example/x%2F..%2F..%2F..%2Fetc%2Fpasswd",
    "https://evil.example/dir%2Fsub%2F..%2F..%2Fescape.txt",
]


class TestTheFilenameCannotCarryATraversal:
    @pytest.mark.parametrize("url", TRAVERSAL_URLS)
    def test_the_derived_name_is_a_single_flat_component(self, url):
        name = _filename_from_url(url)
        assert "/" not in name and "\\" not in name, (
            f"{url} produced {name!r}, which contains a separator"
        )
        assert name not in ("", ".", "..")

    @pytest.mark.parametrize("url", TRAVERSAL_URLS)
    def test_the_resolved_destination_stays_in_the_download_dir(self, url):
        """The impact assertion: where would this have WRITTEN?"""
        anchor = _DOWNLOAD_DIR.resolve()
        dest = _resolve_destination(url, None)
        assert dest.is_relative_to(anchor), (
            f"{url} would write to {dest}, outside {anchor}"
        )

    def test_the_original_attack_landed_in_home(self):
        """Names the specific outcome the audit measured, so a regression says so
        in terms a reader recognises rather than "not relative to anchor"."""
        anchor = _DOWNLOAD_DIR.resolve()
        dest = _resolve_destination(
            "https://evil.example/x%2F..%2F..%2F..%2F.bashrc", None
        )
        assert dest != Path.home() / ".bashrc"
        assert dest.is_relative_to(anchor)
        # And the name that survives is the harmless final component.
        assert dest.name == ".bashrc"
        assert dest.parent == anchor


class TestDegenerateAndControlShapes:
    @pytest.mark.parametrize("url", [
        "https://evil.example/..",
        "https://evil.example//..",
        "https://evil.example/./..",
        "https://evil.example/%2F..%2F..",
    ])
    def test_a_name_that_is_only_traversal_dots_becomes_a_placeholder(self, url):
        """``Path('/..').name`` is '..' — a basename with no separator, so it
        passes the flat-component check and still is not a filename."""
        assert _filename_from_url(url) == "download.bin"

    def test_a_nul_byte_is_replaced_not_passed_to_open(self):
        """A decoded %00 makes open() raise OSError. Replaced rather than left for
        the filesystem call to discover."""
        name = _filename_from_url("https://evil.example/a%00b.pdf")
        assert "\x00" not in name
        assert name == "download.bin"

    def test_a_control_character_is_replaced(self):
        name = _filename_from_url("https://evil.example/a%0Ab.pdf")
        assert "\n" not in name

    def test_no_filename_falls_back_to_the_host(self):
        assert _filename_from_url("https://example.com/") == "example.com.bin"


class TestLegitimateNamesAreNotRewritten:
    """The over-broad fix this test exists to prevent: sanitising the name to
    defend against a case ``.name`` already handles would rewrite names a real
    download asked for. The pinned contract (test_web_tools.py) says a URL-decoded
    space survives."""

    def test_a_decoded_space_survives(self):
        assert _filename_from_url("https://example.com/hello%20world.pdf") == "hello world.pdf"

    def test_a_query_string_is_not_part_of_the_name(self):
        assert _filename_from_url("https://example.com/foo.pdf?token=abc") == "foo.pdf"

    def test_a_plain_name_is_unchanged(self):
        assert _filename_from_url("https://example.com/report.pdf") == "report.pdf"

    def test_a_deep_path_keeps_its_final_component(self):
        assert _filename_from_url("https://example.com/a/b/c/data.tar.gz") == "data.tar.gz"

    def test_punctuation_and_unicode_survive(self):
        assert _filename_from_url("https://example.com/my%20report%20(v2).pdf") == "my report (v2).pdf"


class TestTheContainmentCheckIsTheProperty:
    """The forbidden-prefix list cannot express containment. Assert the invariant
    directly, independent of what the filename helper happens to return — so if
    that helper regresses in some new way, this still holds."""

    def test_a_derived_destination_is_always_inside_the_download_dir(self):
        anchor = _DOWNLOAD_DIR.resolve()
        for url in TRAVERSAL_URLS + [
            "https://ok.example/report.pdf",
            "https://ok.example/",
            "https://ok.example/..",
        ]:
            dest = _resolve_destination(url, None)
            assert dest.is_relative_to(anchor), f"{url} -> {dest}"

    def test_an_explicit_destination_may_leave_the_download_dir(self):
        """The containment check applies ONLY to URL-derived names. An explicit
        destination is the operator's (or the gate's) business — confining it here
        would break `download_file(url, destination=...)`, which the tool
        advertises."""
        dest = _resolve_destination("https://ok.example/x.bin", "~/Downloads/chosen.bin")
        assert dest == Path.home() / "Downloads" / "chosen.bin"

    def test_an_explicit_destination_still_hits_the_protected_paths(self):
        with pytest.raises(ValueError):
            _resolve_destination("https://x.com/x", str(Path.home() / ".ssh" / "authorized_keys"))


class TestDownloadFileInstallsTheHopGuard:
    """download_file has its OWN httpx client. Removing its event_hooks would
    leave the streaming path following redirects unchecked — the same wiring gap
    section 4 of test_url_guard.py pins for fetch_url_text, on the second caller."""

    @pytest.mark.asyncio
    async def test_a_redirect_to_a_private_address_is_refused(self, monkeypatch):
        import httpx

        from prometheus.tools.base import ToolExecutionContext
        from prometheus.tools.builtin.download_file import DownloadFileTool
        from prometheus.tools.builtin.web_fetch import SsrfBlocked  # noqa: F401

        def handler(request):
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:8005/api/x"})
            return httpx.Response(200, content=b"SECRET-BODY-MUST-NOT-BE-WRITTEN")

        real = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        # Make the mock hosts resolve to a public address so the guard refuses
        # for the ADDRESS, not for a DNS failure.
        import prometheus.security.url_guard as g
        real_gai = g.socket.getaddrinfo

        def fake_gai(host, *a, **k):
            if host in ("public.example",):
                return [(2, 1, 6, "", ("93.184.216.34", 0))]
            return real_gai(host, *a, **k)

        monkeypatch.setattr(g.socket, "getaddrinfo", fake_gai)

        tool = DownloadFileTool()
        result = await tool.execute(
            DownloadFileTool.input_model(url="http://public.example/file.bin"),
            ToolExecutionContext(cwd=Path("/tmp")),
        )
        assert result.is_error, f"the redirect to loopback was followed: {result.output}"
        assert "non-public" in result.output or "Blocked" in result.output
        assert "SECRET-BODY" not in result.output
        # Nothing was written.
        assert not (_DOWNLOAD_DIR / "file.bin").exists()


class TestTheContainmentBackstopOnItsOwn:
    """The containment check is DEFENCE IN DEPTH — with the filename fix in place
    it never triggers, so no end-to-end test can prove it exists. Removing it alone
    changes no behaviour, which is exactly how a backstop gets deleted silently by
    someone tidying up.

    So it is tested as a unit: monkeypatch the filename helper to return a
    traversal name (the pre-fix behaviour) and assert _resolve_destination still
    refuses. This is the layer that turns "the filename helper must be correct"
    into "even if it is not, nothing escapes".
    """

    def test_a_traversal_name_from_the_helper_is_still_refused(self, monkeypatch):
        import prometheus.tools.builtin.download_file as mod

        monkeypatch.setattr(mod, "_filename_from_url", lambda url: "../../.bashrc")
        with pytest.raises(ValueError, match="must land under"):
            mod._resolve_destination("https://evil.example/x", None)

    def test_an_absolute_name_from_the_helper_is_still_refused(self, monkeypatch):
        import prometheus.tools.builtin.download_file as mod

        monkeypatch.setattr(mod, "_filename_from_url", lambda url: "/etc/cron.d/x")
        with pytest.raises(ValueError):
            mod._resolve_destination("https://evil.example/x", None)

    def test_the_backstop_does_not_fire_for_a_normal_name(self, monkeypatch):
        """And it must not be so tight that it breaks ordinary downloads — a
        backstop that refuses everything gets disabled."""
        import prometheus.tools.builtin.download_file as mod

        monkeypatch.setattr(mod, "_filename_from_url", lambda url: "report.pdf")
        dest = mod._resolve_destination("https://ok.example/x", None)
        assert dest.name == "report.pdf"
        assert dest.is_relative_to(_DOWNLOAD_DIR.resolve())
