"""The media cache must not hand back a path to a file it did not write.

THE DEFECT
----------
All four `cache_*_from_bytes` helpers ended with::

    return written or str(path)

`_cache_write` returns `None` when it declines or fails — below the free-disk
floor, or on any write error. The `or` then fabricated the path the file WOULD
have had. Measured with the floor forced:

    photo     returned: /.../img_8ec90e18bb12.jpg          exists: False
    voice     returned: /.../audio_47eeb5854566.ogg        exists: False
    document  returned: /.../doc_1f0287453a55_report.pdf   exists: False

`_cache_write`'s fail-open is CORRECT and unchanged: caching is a convenience,
and its docstring says the caller must still process the media. The defect was
these wrappers turning "not cached" into "here is where it is" — which is not
fail-open, it is a false statement, and the content was lost anyway.

WHY None AND NOT AN EXCEPTION
------------------------------
Raising would make every surface fail CLOSED. The Telegram handlers already
catch broadly and would start answering "Failed to download the photo" and
returning — blocking the message on low disk, which is the opposite of the
stated contract. `None` keeps the message flowing and makes the absence
something the caller has to look at.

The caller-side half — that the message still arrives, degraded — is asserted
on the real surface in
`tests/test_telegram_surface_hardening.py::test_cache_below_the_floor_does_not_block_the_message`.
This file covers the helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prometheus.gateway.media_cache as mc  # noqa: E402


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    for name in ("image_cache_dir", "audio_cache_dir", "video_cache_dir",
                 "document_cache_dir"):
        monkeypatch.setattr(mc, name, lambda _t=tmp_path: _t, raising=False)
    return tmp_path


@pytest.fixture
def below_floor(monkeypatch):
    monkeypatch.setattr(mc, "_below_free_disk_floor", lambda: True)


CALLS = {
    "image": lambda: mc.cache_image_from_bytes(b"\xff\xd8\xff-jpeg"),
    "video": lambda: mc.cache_video_from_bytes(b"\x00\x00\x00 ftypmp4"),
    "audio": lambda: mc.cache_audio_from_bytes(b"OggS-voice"),
    "document": lambda: mc.cache_document_from_bytes(b"%PDF-doc", "report.pdf"),
}


@pytest.mark.parametrize("kind", sorted(CALLS))
def test_below_the_floor_returns_none_not_a_path(kind, cache_dir, below_floor):
    assert CALLS[kind]() is None


@pytest.mark.parametrize("kind", sorted(CALLS))
def test_nothing_was_written(kind, cache_dir, below_floor):
    """Pins the premise: the floor really did prevent the write.

    Without this, the None above could come from somewhere else while a file
    was quietly created — the test would pass and measure nothing.
    """
    CALLS[kind]()
    assert list(cache_dir.iterdir()) == []


@pytest.mark.parametrize("kind", sorted(CALLS))
def test_a_write_failure_also_returns_none(kind, cache_dir, monkeypatch):
    """The other `None` path: the floor is fine, the write itself fails."""
    monkeypatch.setattr(mc, "_below_free_disk_floor", lambda: False)
    monkeypatch.setattr(
        Path, "write_bytes",
        lambda self, data: (_ for _ in ()).throw(OSError("read-only fs")),
    )
    assert CALLS[kind]() is None


@pytest.mark.parametrize("kind", sorted(CALLS))
def test_the_normal_path_still_returns_a_real_file(kind, cache_dir, monkeypatch):
    """The fix must not break caching.

    Without this, `return None` unconditionally would satisfy every test above
    and disable inbound media entirely.
    """
    monkeypatch.setattr(mc, "_below_free_disk_floor", lambda: False)
    path = CALLS[kind]()
    assert path is not None
    assert Path(path).exists() and Path(path).stat().st_size > 0


def test_no_helper_ever_returns_a_path_that_does_not_exist(cache_dir, below_floor):
    """The property, stated once over every helper.

    This is the assertion the old code fails: it returned a string, and the
    string named nothing.
    """
    for kind, call in sorted(CALLS.items()):
        returned = call()
        if returned is None:
            continue
        assert Path(returned).exists(), (
            f"{kind} returned {returned!r} for a file that was never written"
        )
