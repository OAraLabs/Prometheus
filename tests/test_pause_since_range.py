"""A pause file whose `since` is Infinity, NaN, negative, too large for a float or past the year 5138 is unreadable
(counts as PAUSED), never a crash at boot or in a child (#920 review L1, carried into 0.9.21)."""

import pytest

from prometheus.engine import pause


@pytest.mark.parametrize("since", ["Infinity", "NaN", "-Infinity", "1e300", "-5", "1" + "0" * 400])
def test_an_out_of_range_since_is_unreadable_not_a_crash(tmp_path, since):
    path = tmp_path / "pause.json"
    path.write_text('{"paused": true, "since": ' + since + ', "by": "owner"}', encoding="utf-8")
    info = pause.read_file(path)
    assert info is not None and info.by == "unreadable"
    state = pause.PauseState(path)                    # a daemon booting on it: paused, no raise
    assert state.current is not None
