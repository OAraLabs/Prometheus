"""An approval push says WHICH Prometheus is asking.

A phone paired with several Prometheus instances gets an approval push from each, and until now the push
body gave it nothing to tell them apart: ``request_id``, ``tool_name`` and ``expires_at`` are the same shape
whichever daemon sent them. The instance has a public identity already, the fingerprint ``GET /api/hello``
advertises as ``fp`` (the first 16 hex characters of SHA-256 over the public half of the instance key,
``config/instance_key.py``). It is not a secret: it is served to anyone who can reach the port. So the approval
push carries it, under the key ``instance_fp``, and a paired phone matches it against the fingerprint it pinned
when it paired.

What this file holds:

  * both approval bodies carry it: the ordinary one and the narrower desktop-task one (the one that says only
    "Prometheus needs a decision", which is the one that most needs a name on it);
  * the value is the daemon's own ``instance_fingerprint()``, the same string hello advertises;
  * with no instance key there is NO ``instance_fp`` (a missing key is not an empty string a phone could match);
  * a value that is not a fingerprint (wrong shape, wrong type, a provider that raises) is dropped, not pushed;
    whatever else goes wrong, the approval is still delivered;
  * the quiet pushes (turn finished, task finished) are untouched: this is about approvals;
  * nothing else of the key goes out: not the key, not its DER, not a longer digest.

``tests/test_approval_push_fanout.py`` is the review gate for the body's key set; ``instance_fp`` was added to its
reviewed set on purpose, and says why.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("cryptography")

from prometheus.config import instance_key  # noqa: E402
from prometheus.push.dispatcher import PushDispatcher  # noqa: E402

FP_SHAPE = re.compile(r"^[0-9a-f]{16}$")


class _Store:
    def push_targets(self):
        return ["device-1", "device-2"]


def _dispatcher(**kwargs):
    delivered: list[dict] = []
    d = PushDispatcher(_Store(), object(), bridge=None, **kwargs)

    async def _deliver(target, body):
        delivered.append(body)

    d._deliver = _deliver  # type: ignore[method-assign]
    return d, delivered


def _approval(*, desktop_task: bool = False) -> dict:
    payload = {"request_id": "r1", "tool_name": "bash", "description": "run a command",
               "expires_at": 1_800_000_000.0, "arguments": {"command": "ls"}}
    if desktop_task:
        payload["task_id"] = "t1"
    return payload


def _push(d, payload) -> None:
    asyncio.run(d.on_signal(SimpleNamespace(kind="approval_pending", payload=payload)))


FP = "0123456789abcdef"


# ── both approval bodies name the instance ───────────────────────────────────

@pytest.mark.parametrize("desktop_task", [False, True], ids=["ordinary", "desktop-task"])
def test_an_approval_push_carries_the_instance_fingerprint(desktop_task):
    d, delivered = _dispatcher(instance_fp=lambda: FP)
    _push(d, _approval(desktop_task=desktop_task))
    assert len(delivered) == 2, "the approval was not delivered to both devices"
    for body in delivered:
        assert body["instance_fp"] == FP


def test_the_value_is_the_daemons_own_fingerprint_the_one_hello_advertises():
    instance_key.ensure_instance_key()
    fp = instance_key.instance_fingerprint()
    assert FP_SHAPE.match(fp), "the fixture has no real fingerprint; the test proves nothing"
    d, delivered = _dispatcher()                     # the default source: no injection
    _push(d, _approval())
    _push(d, _approval(desktop_task=True))
    assert [b["instance_fp"] for b in delivered] == [fp] * 4


# ── what is left out ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("desktop_task", [False, True], ids=["ordinary", "desktop-task"])
def test_with_no_instance_key_the_field_is_absent_and_the_push_still_goes(desktop_task):
    assert instance_key.instance_fingerprint() == "", "the fixture already has an instance key"
    d, delivered = _dispatcher()
    _push(d, _approval(desktop_task=desktop_task))
    assert len(delivered) == 2
    assert all("instance_fp" not in body for body in delivered), "an empty fingerprint was pushed as if it were one"


@pytest.mark.parametrize("bad", ["", "ABCDEF0123456789", "0123456789abcde", "0123456789abcdef0", "zzzzzzzzzzzzzzzz",
                                 None, 12345678901234567, b"0123456789abcdef", "0123456789abcdef\n"])
def test_a_value_that_is_not_a_fingerprint_is_not_pushed(bad):
    d, delivered = _dispatcher(instance_fp=lambda: bad)
    _push(d, _approval())
    assert len(delivered) == 2, "a bad fingerprint stopped the approval from being delivered"
    assert all("instance_fp" not in body for body in delivered)


def test_a_provider_that_raises_does_not_stop_the_approval():
    def boom():
        raise OSError("the node directory is unreadable")

    d, delivered = _dispatcher(instance_fp=boom)
    _push(d, _approval())
    _push(d, _approval(desktop_task=True))
    assert len(delivered) == 4
    assert all("instance_fp" not in body for body in delivered)


def test_only_the_fingerprint_goes_out_never_the_key():
    instance_key.ensure_instance_key()
    der = instance_key.instance_public_key_der()
    d, delivered = _dispatcher()
    _push(d, _approval())
    _push(d, _approval(desktop_task=True))
    blob = repr(delivered)
    assert der.hex() not in blob and der.hex()[:40] not in blob
    for body in delivered:
        assert FP_SHAPE.match(body["instance_fp"])


# ── what does not change ─────────────────────────────────────────────────────

def test_the_quiet_pushes_do_not_carry_it():
    d, delivered = _dispatcher(instance_fp=lambda: FP)
    d._store = SimpleNamespace(push_targets=lambda: [SimpleNamespace(id="device-1")])   # the quiet path reads .id
    asyncio.run(d.on_signal(SimpleNamespace(kind="turn_completed", payload={"session_id": "s1"})))
    assert delivered, "the quiet push was not delivered; the test proves nothing"
    assert all("instance_fp" not in body for body in delivered)


def test_the_desktop_task_body_still_carries_no_content():
    d, delivered = _dispatcher(instance_fp=lambda: FP)
    _push(d, _approval(desktop_task=True))
    for body in delivered:
        assert set(body) == {"aps", "request_id", "expires_at", "instance_fp"}
        assert body["aps"]["alert"]["body"] == "Prometheus needs a decision"
        assert "category" not in body["aps"]
