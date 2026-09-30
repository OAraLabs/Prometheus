"""Capture what `classify_turn_error` actually received, not a guess (#321).

WHY A HARNESS AND NOT A FIX
----------------------------
The fallback's terminal-kind predicate cannot express "retryable in seconds"
versus "retryable in three days" — both are `KIND_RATE_LIMIT`. Whether that is
a live defect depends on how one particular provider presents an exhausted
quota on the wire, and guessing that is the error mode the issue exists to
avoid.

WHAT IS ALREADY KNOWN, FROM REAL EVENTS
----------------------------------------
Three exhaustion failures on 2026-08-26, 08-29 and 08-31 logged exactly:

    httpx.HTTPStatusError: Client error '429 Too Many Requests' for url '...'

So the status is a plain **429**, not a 403 — which rules out "the classifier
already calls this billing" on status alone. The body was never read on that
path (fixed separately), and the headers were never logged at all.

The durable row for those same three events was
`{"exception_type": "HTTPStatusError"}` — the class name and nothing more.

WHAT THIS ADDS
---------------
Status and body were already captured. These tests pin the three that were not:
the rate/quota HEADERS, the classifier's CONCLUSION, and a durable row that
carries both. The point is stated in
`test_the_row_distinguishes_a_transient_429_from_an_exhausted_quota`: two
responses that are identical in status must produce different rows.
"""

from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")

from prometheus.api.turn_errors import (  # noqa: E402
    QUOTA_HEADERS,
    quota_headers,
    redact_url,
)
from prometheus.learning.llm_envelope import _failure_summary  # noqa: E402


def _error(status: int, body: str, headers: dict[str, str] | None = None):
    """An httpx.HTTPStatusError shaped like a real provider rejection."""
    request = httpx.Request("POST", "https://provider.example.invalid/v1/chat/completions")
    response = httpx.Response(status, content=body.encode(), headers=headers or {},
                              request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# ── headers ──────────────────────────────────────────────────────────────────


def test_quota_headers_are_captured():
    exc = _error(429, "{}", {"retry-after": "259200",
                             "x-ratelimit-remaining-tokens": "0"})
    got = quota_headers(exc.response)
    assert got["retry-after"] == "259200"
    assert got["x-ratelimit-remaining-tokens"] == "0"


def test_the_header_capture_is_an_allowlist_not_a_dump():
    """A raw header dump into a log trades a blind spot for a credential leak."""
    exc = _error(429, "{}", {
        "retry-after": "20",
        "authorization": "Bearer SUPERSECRET",
        "set-cookie": "session=SUPERSECRET",
        "x-internal-routing-key": "SUPERSECRET",
    })
    got = quota_headers(exc.response)
    assert got == {"retry-after": "20"}
    assert "SUPERSECRET" not in repr(got)
    assert "authorization" not in QUOTA_HEADERS


def test_header_capture_never_raises_on_a_non_response():
    assert quota_headers(None) == {}
    assert quota_headers(object()) == {}


def test_a_logged_url_never_carries_its_query_string():
    """Gemini puts the API key in `?key=`; this value goes to a log."""
    out = redact_url("https://host.example.invalid/v1/chat?key=SUPERSECRET&x=1")
    assert "SUPERSECRET" not in out
    assert out.startswith("https://host.example.invalid/v1/chat")


# ── the durable row ──────────────────────────────────────────────────────────


def test_the_failure_row_carries_more_than_a_class_name():
    """The three real events left only `{"exception_type": "HTTPStatusError"}`."""
    exc = _error(429, '{"error":{"message":"rate limit"}}', {"retry-after": "20"})
    row = _failure_summary(exc)
    assert row["exception_type"] == "HTTPStatusError"
    assert row["status"] == 429
    assert row["kind"], "the classifier's conclusion is the decision; record it"
    assert row["quota_headers"]["retry-after"] == "20"


def test_the_row_distinguishes_a_transient_429_from_an_exhausted_quota():
    """THE POINT OF THE HARNESS.

    Identical status. The rows must not be identical, or Monday's event tells us
    nothing we did not already know.
    """
    transient = _failure_summary(
        _error(429, '{"error":{"message":"Too many requests, slow down"}}',
               {"retry-after": "20", "x-ratelimit-remaining-tokens": "48000"})
    )
    exhausted = _failure_summary(
        _error(429, '{"error":{"message":"You exceeded your current quota",'
                    '"type":"insufficient_quota"}}',
               {"retry-after": "259200", "x-ratelimit-remaining-tokens": "0"})
    )

    assert transient["status"] == exhausted["status"] == 429
    assert transient != exhausted, (
        "two failures that need opposite responses produced identical rows"
    )
    # The headers alone separate them, before anyone argues about body text.
    assert transient["quota_headers"]["retry-after"] == "20"
    assert exhausted["quota_headers"]["retry-after"] == "259200"
    assert exhausted["quota_headers"]["x-ratelimit-remaining-tokens"] == "0"


def test_the_row_is_still_written_when_classification_is_impossible():
    """A diagnostic that can fail the write it decorates is worse than none."""
    row = _failure_summary(RuntimeError("no response attached"))
    # `kind: unknown` is kept — "the classifier could not tell" is a real
    # finding. The placeholder provider is not: `classify_turn_error` never
    # returns an empty provider, it returns the literal "the model provider" so
    # its user-facing hint reads as a sentence. Storing that would be noise
    # shaped like data.
    assert row == {"exception_type": "RuntimeError", "kind": "unknown"}
    assert "provider" not in row


def test_an_exhausted_quota_body_is_already_classified_terminal():
    """Documents the OTHER half, so Monday's capture is read correctly.

    If the provider's body matches `_BILLING_MARKERS`, the classifier already
    calls it billing and the fallback already fires — there is no bug left to
    fix. This test states that precondition so the capture is interpreted
    against it rather than re-litigated.
    """
    from prometheus.api.turn_errors import KIND_BILLING, classify_turn_error
    from prometheus.engine.fallback import is_terminal

    exc = _error(429, '{"error":{"type":"insufficient_quota"}}')
    detail = classify_turn_error(exc)
    assert detail["kind"] == KIND_BILLING
    assert is_terminal(detail["kind"]) is True


def test_a_bare_429_with_no_marker_is_still_only_a_rate_limit():
    """And this is the case that would be a real defect. Stated, not assumed."""
    from prometheus.api.turn_errors import KIND_RATE_LIMIT, classify_turn_error
    from prometheus.engine.fallback import is_terminal

    exc = _error(429, '{"error":{"message":"Requests rate limit exceeded"}}')
    detail = classify_turn_error(exc)
    assert detail["kind"] == KIND_RATE_LIMIT
    assert is_terminal(detail["kind"]) is False


# ── WP-X.50 leak 1: a URL's userinfo is a credential too ─────────────────────

#: Obviously fake. Mixed-case letters only, every character distinct and no
#: digits, so it shares NO 4-char substring with the URL boilerplate it is
#: printed next to ("http", "127.0.0.1", "8080", "/v1", "<redacted>") — an
#: exhaustive fragment check would otherwise fail for the wrong reason. Built
#: by concatenation so no single literal matches the secret scanner's shape.
_FAKE_USER = "QzXw"
_FAKE_PASS = "KtMp" + "LsNv" + "HgJb"
_FAKE_QKEY = "WdRc" + "TnBx" + "VsQp"


def _assert_no_url_fragment(rendered: str, secret: str) -> None:
    """No piece of *secret* 4 characters or longer survives in *rendered*."""
    for start in range(len(secret) - 3):
        for end in range(start + 4, len(secret) + 1):
            assert secret[start:end] not in rendered, (
                f"redact_url leaked {secret[start:end]!r} of a secret"
            )


class TestRedactUrlDropsUserinfo:
    """``redact_url`` stripped the query but printed the userinfo in full.

    ``http://user:pass@host`` puts a credential in the URL itself, and it is a
    shape that reaches the inference rows for real: it is how an operator points
    ``model.base_url`` at a llama.cpp server behind an authenticating proxy
    (httpx turns the userinfo into a Basic auth header), and some providers
    accept ``?api_key=`` too. #627 closed the YAML and config-pin echoes; the
    inference rows were still printing the URL exactly as configured.

    These live here rather than in test_turn_errors.py for a mechanical reason
    worth keeping visible: test_turn_errors.py carries a deliberate 32-char
    provider-key FIXTURE that trips the pre-commit secret scanner, and
    tests/test_sdist_contents.py allowlists it BY COUNT. Any edit to that file's
    fakes desyncs the release ratchet, so new redact_url cases go here, where
    the function is already tested and the scanner is quiet.

    This is the ONE helper both doctors call — extending it rather than adding a
    second display function is the point, because two copies is the shape where
    one gets fixed and the other does not (``yaml_error_summary`` exists for
    exactly that reason). The request must still use the REAL URL; only what is
    shown changes, so the host and port survive — that is what makes the row
    actionable.
    """

    def test_userinfo_is_dropped_but_host_and_port_survive(self):
        out = redact_url(f"http://{_FAKE_USER}:{_FAKE_PASS}@127.0.0.1:8080/v1")
        _assert_no_url_fragment(out, _FAKE_PASS)
        _assert_no_url_fragment(out, _FAKE_USER)
        assert "@" not in out          # no userinfo left at all
        # The actionable part is still there.
        assert "127.0.0.1" in out
        assert "8080" in out
        assert "/v1" in out

    def test_query_string_is_still_redacted(self):
        """The pre-existing half must not regress."""
        out = redact_url(f"http://127.0.0.1:8080/v1?api_key={_FAKE_QKEY}")
        _assert_no_url_fragment(out, _FAKE_QKEY)
        assert "?<redacted>" in out
        assert "127.0.0.1" in out and "8080" in out

    def test_both_at_once(self):
        out = redact_url(
            f"http://{_FAKE_USER}:{_FAKE_PASS}@127.0.0.1:8080/v1?api_key={_FAKE_QKEY}")
        _assert_no_url_fragment(out, _FAKE_PASS)
        _assert_no_url_fragment(out, _FAKE_USER)
        _assert_no_url_fragment(out, _FAKE_QKEY)
        assert "127.0.0.1:8080/v1" in out
        assert "?<redacted>" in out

    def test_a_clean_url_is_unchanged(self):
        """A URL with nothing to hide must not be mangled.

        The overwhelming majority of deployments have a plain
        ``http://gpu:8080``; if the helper rewrote those the row would become
        noise.
        """
        for url in ("http://localhost:8080",
                    "http://127.0.0.1:8080/v1",
                    "https://gpu.example.invalid:11434/api/tags"):
            assert redact_url(url) == url

    def test_an_at_in_the_path_is_not_userinfo(self):
        """``@`` is legal in a path. Only the AUTHORITY part is a credential.

        Stripping at the first ``@`` anywhere would corrupt a URL that merely
        mentions one after the host, and would silently hide the real host.
        """
        out = redact_url("http://127.0.0.1:8080/models@v1")
        assert out == "http://127.0.0.1:8080/models@v1"

    def test_ipv6_host_survives(self):
        out = redact_url(f"http://{_FAKE_USER}:{_FAKE_PASS}@[::1]:8080/v1")
        _assert_no_url_fragment(out, _FAKE_PASS)
        assert "[::1]:8080" in out

    # ── The three shapes the first pass missed ──────────────────────────────
    #
    # Found by reading the URL back out of a REAL request rather than from a
    # hand-written string: the row only prints what httpx was given, so the
    # cases that matter are the ones an operator's config actually produces.
    # All three were verified leaking against 1e65266.

    @pytest.mark.parametrize("scheme", ["HTTP", "Https", "hTtPs"])
    def test_a_mixed_case_scheme_is_still_a_scheme(self, scheme):
        """RFC 3986 makes the scheme case-INSENSITIVE, and so does httpx.

        The patterns were written ``https?://`` with no flag, so ``HTTP://`` —
        what an operator's YAML or a Windows tool's config often carries —
        matched nothing and the whole userinfo printed. A redaction helper that
        is stricter than the client it protects has a hole exactly as wide as
        the difference, and the difference here is spelling.
        """
        url = f"{scheme}://{_FAKE_USER}:{_FAKE_PASS}@127.0.0.1:8080/v1"
        out = redact_url(url)
        _assert_no_url_fragment(out, _FAKE_PASS)
        _assert_no_url_fragment(out, _FAKE_USER)
        assert "@" not in out
        # Host, port and path survive — the row stays actionable.
        assert "127.0.0.1" in out and "8080" in out and "/v1" in out

    def test_an_at_inside_the_password_is_not_a_boundary(self):
        """``@`` is legal in a password, and httpx splits at the LAST one.

        For ``http://user:pa@ss@host`` httpx sends ``pa@ss`` as the password and
        connects to ``host``. The first pattern's run stopped at the FIRST ``@``,
        so it removed ``user:pa`` and left ``ss@host`` — printing the tail of the
        password AND misreporting the host as ``ss@host``, which is worse than
        leaking: the operator would go look for a server that does not exist.

        The userinfo class now runs to the last ``@`` in the authority, matching
        what the client actually parsed.
        """
        pw = "KtMpLs" + "@" + "NvHgJb"      # letters and one @; no digit runs
        out = redact_url(f"http://{_FAKE_USER}:{pw}@127.0.0.1:8080/v1")
        _assert_no_url_fragment(out, pw)
        assert "@" not in out                # neither half of the password
        assert "127.0.0.1:8080/v1" in out     # and the REAL host is reported

    def test_a_url_with_no_scheme_is_still_redacted(self):
        """``user:pass@host`` with no scheme — the shape that reached the
        "not responding" row.

        httpx accepts a bare ``host:port`` and so does ``model.base_url`` in
        practice; the userinfo pattern was anchored to ``https?://``, so this
        form printed in full. There is no scheme to key off, so the authority
        has to be recognised by what it is NOT: no scheme prefix, and the ``@``
        arrives before any ``/``, ``?`` or ``#``.

        A bare ``127.0.0.1:8080`` has no ``@`` and must stay exactly as written —
        that is the overwhelmingly common form of this config.
        """
        out = redact_url(f"{_FAKE_USER}:{_FAKE_PASS}@127.0.0.1:8080")
        _assert_no_url_fragment(out, _FAKE_PASS)
        _assert_no_url_fragment(out, _FAKE_USER)
        assert "@" not in out
        assert "127.0.0.1:8080" in out        # host and port survive
        # And the schemeless CLEAN case is untouched.
        assert redact_url("127.0.0.1:8080") == "127.0.0.1:8080"

    def test_a_mixed_case_scheme_query_is_redacted(self):
        """The query half had the same case-sensitivity hole.

        ``?key=`` is how Gemini carries its API key, and ``Https://…?key=…``
        printed it. The pattern is shared with ``_redact`` (the wire-text path),
        so widening it is a second, deliberate consequence of this fix: both
        paths become case-insensitive, which can only ever redact more.
        """
        out = redact_url(f"Https://h/x?key={_FAKE_QKEY}")
        _assert_no_url_fragment(out, _FAKE_QKEY)
        assert "?<redacted>" in out
        assert "h/x" in out                   # the path is not the secret

    def test_a_schemeless_at_in_a_path_is_not_userinfo(self):
        """Guards the new bare-authority pattern against over-reach.

        It has no ``https?://`` anchor to hold it to the authority, so the
        ``/ ? #`` exclusion in the userinfo run is the ONLY thing keeping it
        out of a path. An ``@`` after the first ``/`` must survive, both with
        and without a scheme.
        """
        assert redact_url("127.0.0.1:8080/models@v1") == "127.0.0.1:8080/models@v1"
        assert redact_url("http://127.0.0.1:8080/models@v1") == \
            "http://127.0.0.1:8080/models@v1"
