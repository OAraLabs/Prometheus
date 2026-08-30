"""The floors around bash are visible, not inferred.

WHAT WAS MISSING
----------------
Both floors existed and neither appeared on any surface. ``/api/status``
carried nothing about either; ``doctor`` reported the CODING sandbox backend
(``check_coding_sandbox``) while saying nothing about the floor around the
bash TOOL — the surface the model reaches every turn. So "is bash confined
right now" was answered by reading config and inferring, and the one state
that most needs saying — asked for, not happening, bash running anyway —
was invisible by construction.

WHY THESE TESTS PROBE RATHER THAN ECHO
--------------------------------------
A configured mode is not a state. Both floors depend on things config cannot
see: a root-loaded AppArmor profile, and a kernel that will grant an
unprivileged namespace. So every test here forces a DISAGREEMENT between what
config says and what the environment can do, and asserts the surface reports
the environment. A surface that echoed config would pass none of them.
"""

from __future__ import annotations

import pytest

from prometheus.permissions import confinement as C


@pytest.fixture(autouse=True)
def _clear_caches():
    C.reset_cache()
    C.reset_write_cache()
    yield
    C.reset_cache()
    C.reset_write_cache()


@pytest.fixture()
def no_bwrap(monkeypatch):
    """An environment that cannot provide the write floor.

    A VALUE swap, not a callable one: pointing BWRAP_BIN at a name that is
    not on PATH exercises the real ``shutil.which`` lookup rather than
    replacing it, so the failure path under test is the production one.
    """
    monkeypatch.setattr(C, "BWRAP_BIN", "bwrap-not-installed-probe")
    C.reset_write_cache()
    return None


# --------------------------------------------------------------------------- #
# The report primitive — one source of truth for both surfaces
# --------------------------------------------------------------------------- #


class TestFloorReport:
    def test_off_is_reported_as_off_not_as_absent(self):
        """"No row" and "no floor" look identical on a report.

        Only one of them is true, so `off` gets a state of its own rather
        than being omitted.
        """
        rep = C.floor_report(read_mode="off", write_mode="off")
        assert rep["bash_read_floor"]["state"] == C.STATE_OFF
        assert rep["bash_write_floor"]["state"] == C.STATE_OFF
        assert rep["dark"] is False

    def test_auto_without_the_mechanism_is_DARK(self, no_bwrap):
        """The state that is invisible everywhere else.

        Config says `auto`. bash runs. Nothing is enforced. This is the one
        combination a config-reader cannot distinguish from a working floor,
        and the whole reason this report probes.
        """
        rep = C.floor_report(write_mode="auto")
        block = rep["bash_write_floor"]
        assert block["mode"] == "auto", "config still says auto"
        assert block["state"] == C.STATE_DARK
        assert block["available"] is False
        assert rep["dark"] is True, "a monitor must be able to alert on this"

    def test_required_without_the_mechanism_is_REFUSING_not_dark(self, no_bwrap):
        """Loud, not silent — and told apart from dark on purpose.

        `required` + unavailable means every bash call fails, which announces
        itself. Alerting on it as though it were a silent hole would train an
        operator to ignore the field that matters.
        """
        rep = C.floor_report(write_mode="required")
        assert rep["bash_write_floor"]["state"] == C.STATE_REFUSING
        assert rep["dark"] is False, "refusing is not dark"
        assert rep["refusing"] is True

    def test_no_workspace_is_its_own_state(self):
        rep = C.floor_report(write_mode="required", has_workspace=False)
        assert rep["bash_write_floor"]["state"] == C.STATE_NO_WORKSPACE
        assert rep["dark"] is False

    def test_probe_false_never_forks(self, monkeypatch):
        """A caller that must not spawn a subprocess can say so."""
        def _boom(*a, **k):  # pragma: no cover — must not be reached
            raise AssertionError("floor_report(probe=False) forked a process")

        monkeypatch.setattr(C.subprocess, "run", _boom)
        rep = C.floor_report(read_mode="required", write_mode="required",
                             probe=False)
        assert rep["bash_write_floor"]["available"] is None
        assert rep["bash_write_floor"]["state"] == "unknown"

    def test_the_write_probe_carries_the_read_floor_when_composed(self, monkeypatch):
        """The reported state must be the state the NEXT bash call gets.

        bash wraps with aa-exec whenever the read floor is required, so a
        report that probed the bare wrapper would describe an argv nobody
        runs — optimism about a stack that might not compose.
        """
        seen: list[tuple] = []
        real = C.write_preflight

        def spy(*, inner_prefix=(), force=False):
            seen.append(tuple(inner_prefix))
            return real(inner_prefix=inner_prefix, force=force)

        monkeypatch.setattr(C, "write_preflight", spy)
        C.floor_report(read_mode="required", write_mode="auto")
        assert seen, "the write floor was never probed"
        assert any("aa-exec" in " ".join(p) for p in seen), (
            f"the write floor was probed WITHOUT the read floor: {seen}")

    def test_scope_is_stated_so_it_is_not_over_read(self):
        """Two green rows must not read as "bash is confined everywhere"."""
        rep = C.floor_report()
        assert "bash tool only" in str(rep["scope"])


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def _rows(config: dict):
    from prometheus.cli.doctor import check_bash_floors
    return {c.name: c for c in check_bash_floors(config)}


class TestDoctorReportsTheFloors:
    def test_a_dark_write_floor_is_an_ERROR_row_with_a_fix(self, no_bwrap):
        rows = _rows({"security": {"bash_write_confinement": "auto"}})
        row = rows["Bash write floor"]
        assert row.status == "error"
        assert "UNAVAILABLE" in row.message
        assert "WITHOUT this floor" in row.message
        assert row.fix and "bubblewrap" in row.fix

    def test_refusing_says_that_bash_calls_are_failing(self, no_bwrap):
        row = _rows({"security": {"bash_write_confinement": "required"}})["Bash write floor"]
        assert row.status == "error"
        assert "refused" in row.message.lower()

    def test_off_is_an_info_row_that_names_what_is_unprotected(self):
        rows = _rows({"security": {"bash_confinement": "off",
                                   "bash_write_confinement": "off"}})
        assert rows["Bash read floor"].status == "info"
        assert ".ssh" in rows["Bash read floor"].message
        assert rows["Bash write floor"].status == "info"
        assert "anywhere" in rows["Bash write floor"].message

    def test_the_scope_row_appears_only_when_something_is_actually_on(self):
        """An INFO caveat under two OFF rows would be noise, not a limit."""
        off = _rows({"security": {"bash_confinement": "off",
                                  "bash_write_confinement": "off"}})
        assert "Bash floor scope" not in off

    def test_both_floors_get_their_own_row(self):
        """They fail independently and their fixes differ."""
        rows = _rows({"security": {}})
        assert "Bash read floor" in rows and "Bash write floor" in rows

    def test_the_rows_are_in_the_default_check_list(self):
        """A check nothing runs is not a check."""
        import inspect

        from prometheus.cli import doctor

        src = inspect.getsource(doctor.run_extended_checks)
        assert "check_bash_floors" in src


# --------------------------------------------------------------------------- #
# /api/status
# --------------------------------------------------------------------------- #

fastapi = pytest.importorskip("fastapi")

from tests.support.real_app import BOUNDARY_DOUBLE, build_real_app  # noqa: E402


def _status_security(h) -> dict:
    r = h.client.get("/api/status", headers=h.auth())
    assert r.status_code == 200, r.text
    return r.json()["security"]


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_status_carries_both_floors():
    h = build_real_app()
    with h.client:
        sec = _status_security(h)
    assert "bash_read_floor" in sec and "bash_write_floor" in sec
    assert "dark" in sec, "the field a monitor alerts on must be present"


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_status_reports_the_PROBED_state_not_the_configured_mode(monkeypatch):
    """The whole claim of the field, in one assertion.

    Config says the write floor is on. The environment cannot provide it.
    An endpoint echoing config would report a floor that is not there — the
    exact false assurance this exists to remove — so it must report `dark`
    while still showing the mode that was asked for.
    """
    h = build_real_app()
    h.app.state.config["security"] = {"bash_write_confinement": "auto"}
    monkeypatch.setattr(C, "BWRAP_BIN", "bwrap-not-installed-probe")
    C.reset_write_cache()

    with h.client:
        sec = _status_security(h)

    block = sec["bash_write_floor"]
    assert block["mode"] == "auto", "the configured mode is still reported"
    assert block["available"] is False, "but the endpoint probed, and it failed"
    assert block["state"] == C.STATE_DARK
    assert sec["dark"] is True


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_status_reports_off_as_off():
    h = build_real_app()
    h.app.state.config["security"] = {
        "bash_confinement": "off", "bash_write_confinement": "off"}
    with h.client:
        sec = _status_security(h)
    assert sec["bash_write_floor"]["state"] == C.STATE_OFF
    assert sec["dark"] is False, "off was not asked for, so it is not dark"


# NOT marked acceptance, deliberately: this injects a fault by substituting
# floor_report itself. An acceptance test asserts real wiring end to end, and
# this one is the opposite — it proves the endpoint survives that wiring
# breaking. Mislabelling it would weaken what the marker means elsewhere.
def test_a_broken_probe_does_not_take_status_down(monkeypatch):
    """Status is what an operator reaches for when things are wrong.

    A floor probe that raises must degrade to the cache-only view, not 500
    the endpoint that would have explained the problem.
    """
    def _boom(*a, **k):
        raise OSError("probe exploded")

    monkeypatch.setattr(C, "floor_report", _boom)
    h = build_real_app()
    with h.client:
        r = h.client.get("/api/status", headers=h.auth())
    assert r.status_code == 200, r.text
