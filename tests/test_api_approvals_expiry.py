"""GET /api/approvals must state when a request expires.

Without ``expires_at`` a client can only show a countdown by computing
``created_at + 1800`` itself — a second surface deriving a truth the daemon
already holds. That is the defect SPRINT-CONSENT removed for grant extents
(one computed extent, rendered by both surfaces); this suite keeps it from
coming back in the expiry field.

The load-bearing case is ``test_expires_at_follows_a_NON_default_timeout``.
A test that only ever uses the default window passes just as happily against
a hardcoded 1800, which is the bug.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.permissions.approval_queue import (  # noqa: E402
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ApprovalQueue,
    PendingAction,
    approve_verbs,
)
from prometheus.web.server import create_app  # noqa: E402


def _queue_with_pending(timeout_seconds=None, **action_kw):
    kw = {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
    q = ApprovalQueue(**kw)
    action = PendingAction(
        request_id="abc12345",
        tool_name="write_file",
        description="write /etc/nope",
        grant_file_path=action_kw.get("grant_file_path", "/tmp/target.txt"),
    )
    q.pending[action.request_id] = action
    return q, action


def _client(queue):
    app = create_app({})
    app.state.approval_queue = queue
    return TestClient(app)


# --------------------------------------------------------------------------- #
# The field exists and is correct
# --------------------------------------------------------------------------- #


class TestExpiresAtIsServed:
    def test_payload_carries_expires_at(self):
        q, action = _queue_with_pending()
        body = _client(q).get("/api/approvals").json()
        assert len(body) == 1
        assert "expires_at" in body[0], (
            "Beacon cannot render a countdown without this; the alternative "
            "is a second surface computing created_at + a constant"
        )

    def test_expires_at_is_created_at_plus_the_window(self):
        q, action = _queue_with_pending()
        row = _client(q).get("/api/approvals").json()[0]
        # Compare the WINDOW, never the absolute epochs. pytest.approx is
        # relative by default, so approx(created_at + 1800) on a ~1.79e9
        # timestamp carries a ±1786-second tolerance — wide enough to admit
        # a completely wrong window. Deltas are small, so the tolerance is.
        window = row["expires_at"] - row["created_at"]
        assert window == pytest.approx(DEFAULT_APPROVAL_TIMEOUT_SECONDS)

    def test_expires_at_follows_a_NON_default_timeout(self):
        """The whole point of reading it off the queue.

        A queue built with a different window must report that window. If
        this passes while the handler reads the module constant, the field
        is confidently wrong for every non-default deployment — and the
        default-window test above would never notice.
        """
        q, action = _queue_with_pending(timeout_seconds=300)
        row = _client(q).get("/api/approvals").json()[0]
        window = row["expires_at"] - row["created_at"]
        assert window == pytest.approx(300)
        assert window != pytest.approx(DEFAULT_APPROVAL_TIMEOUT_SECONDS)

    def test_expires_at_is_in_the_future_for_a_fresh_request(self):
        q, _ = _queue_with_pending()
        row = _client(q).get("/api/approvals").json()[0]
        assert row["expires_at"] > time.time()

    def test_an_older_request_has_a_shorter_remaining_window(self):
        """Direction check: the field moves with created_at, not with now."""
        q, action = _queue_with_pending()
        action.created_at -= 600
        row = _client(q).get("/api/approvals").json()[0]
        remaining = row["expires_at"] - time.time()
        assert remaining < DEFAULT_APPROVAL_TIMEOUT_SECONDS - 500

    def test_empty_queue_serves_no_rows(self):
        assert _client(ApprovalQueue()).get("/api/approvals").json() == []

    def test_existing_fields_are_untouched(self):
        q, _ = _queue_with_pending()
        row = _client(q).get("/api/approvals").json()[0]
        for key in ("request_id", "tool_name", "description", "created_at",
                    "extents"):
            assert key in row


class TestQueueOwnsTheComputation:
    """One source: the queue holds the window, so the queue does the maths."""

    def test_timeout_seconds_exposes_the_instance_value(self):
        assert ApprovalQueue(timeout_seconds=42).timeout_seconds == 42

    def test_timeout_seconds_defaults_to_the_constant(self):
        assert ApprovalQueue().timeout_seconds == DEFAULT_APPROVAL_TIMEOUT_SECONDS

    def test_expires_at_matches_what_the_wait_would_use(self):
        q, action = _queue_with_pending(timeout_seconds=77)
        # `asyncio.wait_for(..., timeout=self._timeout)` is the real clock;
        # the served field must be derived from the same number.
        assert q.expires_at(action) == pytest.approx(action.created_at + 77)
        assert q.timeout_seconds == q._timeout


# --------------------------------------------------------------------------- #
# The stale docstring
# --------------------------------------------------------------------------- #


class TestApproveDocstringMatchesTheVocabulary:
    """Same binding as #235: prose that no executable check can catch."""

    def _doc(self):
        app = create_app({})
        for route in app.routes:
            if getattr(route, "path", "") == "/api/approvals/{request_id}/approve":
                return route.endpoint.__doc__ or ""
        raise AssertionError("approve route not found")

    def test_does_not_name_the_pre_232_verb_session(self):
        assert '"session"' not in self._doc(), (
            'the approve docstring again names "session", a verb #232 '
            "renamed to until-restart because there is one gate per process"
        )

    def test_points_at_the_single_source(self):
        assert "approve_verbs()" in self._doc(), (
            "the docstring must defer to approve_verbs() rather than "
            "restating a literal list that drifts"
        )

    def test_the_verbs_it_describes_are_the_ones_accepted(self):
        doc = self._doc()
        for verb in approve_verbs():
            assert verb in doc, f"accepted verb {verb!r} is undocumented"
