"""TRIPWIRE self-test (Piece 1, merge gate #1).

Proves the detector detects: a deliberately-bad acceptance test (marked, but
terminating in a registered double) is FAILED by the enforcement hook with the
prescribed message — and a clean acceptance test passes (no false positive).

The end-to-end proof runs a REAL inner pytest (subprocess) whose conftest
re-exports the actual hook/fixture from ``tests/conftest.py`` — so what is
being tested is the enforcement code itself, not a re-implementation. One
inner run carries the whole case matrix (bad / clean / escape-hatch /
wildcard-rejected / unknown-name / unmarked) to keep it to a single subprocess.

Direct unit tests below cover the registry mechanics (class/function/instance
instrumentation, touch counting, reset, signature preservation for the
wire-contract drift pin).
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.support.doubles import register_double, registry

REPO_ROOT = Path(__file__).resolve().parents[1]

INNER_CONFTEST = """\
import sys

sys.path.insert(0, {repo_root!r})

# Register the REAL enforcement machinery in this inner run — the hook under
# test is the actual one from tests/conftest.py, not a copy.
from tests.conftest import (  # noqa: E402,F401
    _tripwire_reset,
    pytest_configure,
    pytest_runtest_call,
)
"""

INNER_TESTS = """\
import sys

sys.path.insert(0, {repo_root!r})

import pytest

from tests.support.doubles import register_double


@register_double("inner_fake_provider", replaces="prometheus.providers.base.ModelProvider")
class FakeProvider:
    def stream_message(self, request):
        return "canned"


@pytest.mark.acceptance
def test_bad_acceptance_doubled():
    # Deliberately bad: acceptance-marked, but the "work" terminates in the
    # registered double. The test's own assertion passes — the HOOK must fail it.
    assert FakeProvider().stream_message(None) == "canned"


@pytest.mark.acceptance
def test_clean_acceptance():
    # Touches no registered double — must pass untouched (no false positive).
    assert 1 + 1 == 2


@pytest.mark.acceptance(allow_doubles=["inner_fake_provider"])
def test_escape_hatch_allows_named():
    assert FakeProvider().stream_message(None) == "canned"


@pytest.mark.acceptance(allow_doubles=["*"])
def test_wildcard_rejected():
    assert True


@pytest.mark.acceptance(allow_doubles=["no_such_double"])
def test_unknown_name_rejected():
    assert True


def test_unmarked_touches_double_ok():
    # No acceptance mark -> doubles are fine; enforcement must not leak here.
    assert FakeProvider().stream_message(None) == "canned"


def _unregistered_substitute():
    return "substituted"


@pytest.mark.acceptance
def test_sentinel_warns_on_unregistered_substitute(monkeypatch):
    # Coverage sentinel: monkeypatching an UNREGISTERED callable in an
    # acceptance test must emit the loud TRIPWIRE-2 warning (not a failure).
    import json as _mod

    monkeypatch.setattr(_mod, "dumps", _unregistered_substitute)
    assert _mod.dumps() == "substituted"
"""


def _run_inner(tmp_path: Path) -> tuple[int, str]:
    (tmp_path / "conftest.py").write_text(INNER_CONFTEST.format(repo_root=str(REPO_ROOT)))
    (tmp_path / "test_inner.py").write_text(INNER_TESTS.format(repo_root=str(REPO_ROOT)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-v", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_tripwire_end_to_end(tmp_path):
    code, out = _run_inner(tmp_path)

    # The deliberately-bad acceptance test is CAUGHT, with the prescribed message.
    assert "test_bad_acceptance_doubled FAILED" in out
    assert (
        "ACCEPTANCE TEST TERMINATED IN DOUBLE: inner_fake_provider "
        "(replaces prometheus.providers.base.ModelProvider)." in out
    )
    assert "An acceptance test must exercise the real path." in out
    assert "do not suppress this check" in out

    # The clean acceptance test passes — no false positive.
    assert "test_clean_acceptance PASSED" in out

    # Escape hatch: individually-named allowance works…
    assert "test_escape_hatch_allows_named PASSED" in out
    # …but wildcards and unknown names are rejected loudly.
    assert "test_wildcard_rejected FAILED" in out
    assert "wildcards are not permitted" in out
    assert "test_unknown_name_rejected FAILED" in out
    assert "unknown double(s) ['no_such_double']" in out

    # Enforcement applies ONLY to acceptance-marked tests.
    assert "test_unmarked_touches_double_ok PASSED" in out

    # Coverage sentinel: unregistered substitute in an acceptance test → loud
    # WARNING (test still passes) announcing the TRIPWIRE-2 gap.
    assert "test_sentinel_warns_on_unregistered_substitute PASSED" in out
    assert "ACCEPTANCE TEST USED UNREGISTERED SUBSTITUTE" in out
    assert "TRIPWIRE-2" in out

    # Overall: a run containing violations exits non-zero (the gate holds in CI).
    assert code != 0


# --------------------------------------------------------------------------- #
# Registry mechanics (direct)
# --------------------------------------------------------------------------- #


def test_class_double_touch_counting_and_reset():
    @register_double("tw_unit_class", replaces="unit.Class")
    class D:
        def hit(self):
            return "x"

        @staticmethod
        def s():
            return "s"

        @classmethod
        def c(cls):
            return "c"

    d = D()
    assert d.hit() == "x" and D.s() == "s" and D.c() == "c"
    touched = {r.name: r.touched for r in registry.touched_records()}
    assert touched.get("tw_unit_class") == 3
    registry.reset()
    assert "tw_unit_class" not in registry.touched_names()


def test_subclass_of_instrumented_double_still_touches():
    @register_double("tw_unit_base", replaces="unit.Base")
    class Base:
        def act(self):
            return "base"

    class Child(Base):
        pass

    registry.reset()
    assert Child().act() == "base"
    assert "tw_unit_base" in registry.touched_names()


def test_function_double_preserves_signature_for_drift_pins():
    async def fake_run_loop(ctx, messages, *, mode="agent", session_id=None):
        if False:
            yield

    wrapped = register_double("tw_unit_fn", replaces="engine.run_loop")(fake_run_loop)
    # inspect.signature must see through the wrapper (functools.wraps/__wrapped__),
    # so the wire-contract signature drift pin keeps working on registered doubles.
    assert str(inspect.signature(wrapped)) == str(inspect.signature(fake_run_loop))
    registry.reset()
    agen = wrapped(None, [])
    assert "tw_unit_fn" in registry.touched_names()
    assert hasattr(agen, "__anext__")  # still an async generator when called


def test_instance_registration_instruments_its_class():
    class Bridge:
        async def dispatch_user_message(self, *a, **k):
            return None

    b = Bridge()
    same = register_double("tw_unit_instance", replaces="web.WebSocketBridge")(b)
    assert same is b  # identity preserved (isinstance stays valid)
    registry.reset()
    import asyncio

    asyncio.run(b.dispatch_user_message("s", "m"))
    assert "tw_unit_instance" in registry.touched_names()


def test_registration_is_idempotent():
    @register_double("tw_unit_idem", replaces="unit.Idem")
    class D:
        def act(self):
            return 1

    register_double("tw_unit_idem", replaces="unit.Idem")(D)  # second registration: no double-wrap
    registry.reset()
    D().act()
    rec = {r.name: r.touched for r in registry.touched_records()}
    assert rec.get("tw_unit_idem") == 1  # exactly one touch, not two
