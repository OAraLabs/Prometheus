"""web.bind -- the one setting that decides which interface the daemon listens on.

Precedence, highest first::

    --bind ADDRESS         (oara daemon / python -m prometheus.daemon)
    PROMETHEUS_WEB_BIND    (the process environment, or the env file)
    web.bind               (prometheus.yaml)
    0.0.0.0                (the default -- unchanged, see TestCompatibility)

Everything here is the RESOLVER and what hangs off it (validation, the CLI flag,
the setup-mode ``configure`` pin, the doctor row, the warning text). Real sockets
and real HTTP/WebSocket exchanges are in test_web_bind_sockets.py.

THE RULE EVERY VALIDATION TEST PINS: a value that cannot be honoured makes the
daemon refuse to start. It never falls back to a wider bind than the one asked
for -- not to the next source in the precedence chain, and not to the default.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

from tests.support.listeners import (
    free_port,
    package_src_root,
    repo_config_of_loaded_package,
)

BIND_ENV = "PROMETHEUS_WEB_BIND"


def _bind():
    """Imported inside each test: a missing module fails THAT test, with the reason."""
    from prometheus.web import bind

    return bind


# ---------------------------------------------------------------------------
# The resolver: defaults, precedence
# ---------------------------------------------------------------------------


class TestDefaultsAndPrecedence:
    @pytest.mark.parametrize("config", [
        None, {}, {"web": None}, {"web": {}}, {"web": {"enabled": True}},
        {"web": {"bind": None}},       # `bind:` left empty in yaml is "unset"
    ])
    def test_nothing_set_is_every_interface(self, config):
        got = _bind().resolve_bind(config, env={})
        assert got.address == "0.0.0.0"
        assert got.source == "default"

    def test_the_default_constant_is_the_old_behaviour(self):
        assert _bind().DEFAULT_BIND == "0.0.0.0"

    def test_config_alone(self):
        got = _bind().resolve_bind({"web": {"bind": "127.0.0.1"}}, env={})
        assert (got.address, got.source) == ("127.0.0.1", "config")

    def test_env_beats_config(self):
        got = _bind().resolve_bind(
            {"web": {"bind": "0.0.0.0"}}, env={BIND_ENV: "127.0.0.1"})
        assert (got.address, got.source) == ("127.0.0.1", "env")

    def test_flag_beats_env_and_config(self):
        got = _bind().resolve_bind(
            {"web": {"bind": "0.0.0.0"}}, flag="::1", env={BIND_ENV: "0.0.0.0"})
        assert (got.address, got.source) == ("::1", "flag")

    def test_flag_beats_config_without_env(self):
        got = _bind().resolve_bind({"web": {"bind": "0.0.0.0"}}, flag="127.0.0.1", env={})
        assert (got.address, got.source) == ("127.0.0.1", "flag")

    def test_a_wider_higher_source_still_wins(self):
        """Precedence is not "narrowest wins": the operator who says 0.0.0.0 on
        the command line means it, whatever the file says."""
        got = _bind().resolve_bind(
            {"web": {"bind": "127.0.0.1"}}, flag="0.0.0.0", env={BIND_ENV: "127.0.0.1"})
        assert (got.address, got.source) == ("0.0.0.0", "flag")

    def test_env_defaults_to_the_process_environment(self, monkeypatch):
        monkeypatch.setenv(BIND_ENV, "127.0.0.1")
        assert _bind().resolve_bind({}).address == "127.0.0.1"
        monkeypatch.delenv(BIND_ENV)
        assert _bind().resolve_bind({}).address == "0.0.0.0"

    def test_an_explicit_env_mapping_is_the_only_environment_read(self, monkeypatch):
        monkeypatch.setenv(BIND_ENV, "127.0.0.1")
        assert _bind().resolve_bind({}, env={}).address == "0.0.0.0"


# ---------------------------------------------------------------------------
# The resolver: what an address may look like
# ---------------------------------------------------------------------------


class TestAcceptedValues:
    @pytest.mark.parametrize(("given", "canonical"), [
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "0.0.0.0"),
        ("192.0.2.10", "192.0.2.10"),
        ("localhost", "127.0.0.1"),        # never left to the resolver / /etc/hosts
        ("LOCALHOST", "127.0.0.1"),
        ("  127.0.0.1  ", "127.0.0.1"),
        ("::1", "::1"),
        ("[::1]", "::1"),
        ("0:0:0:0:0:0:0:1", "::1"),
        ("::", "::"),
        ("2001:db8::10", "2001:db8::10"),
    ])
    def test_ip_literals_and_localhost(self, given, canonical):
        for kwargs in ({"flag": given}, {"env": {BIND_ENV: given}}):
            assert _bind().resolve_bind({}, **kwargs).address == canonical
        assert _bind().resolve_bind({"web": {"bind": given}}, env={}).address == canonical

    @pytest.mark.parametrize(("address", "wide"), [
        ("0.0.0.0", True), ("::", True), ("127.0.0.1", False), ("::1", False),
        ("192.0.2.10", False), ("2001:db8::10", False),
    ])
    def test_is_all_interfaces(self, address, wide):
        assert _bind().is_all_interfaces(address) is wide


INVALID = [
    "not-an-address", "example.com", "*", "all", "any",
    "127.0.0.1:8005", "[::1]:8005", "0.0.0.0/0", "127.0.0.1/8",
    "256.1.1.1", "1.2.3", "127.1", "0x7f.0.0.1", "2130706433", "127.000.000.001",
    "fe80::1%en0", "localhost.", "localhost:8005", "[::1", "::1]",
    "http://127.0.0.1", "127.0.0.1,0.0.0.0", "127.0.0.1 0.0.0.0", "-1", "0",
    "", "   ",
]


class TestFailClosed:
    @pytest.mark.parametrize("value", INVALID)
    def test_invalid_flag_refuses_and_names_the_flag(self, value):
        with pytest.raises(_bind().BindError) as err:
            _bind().resolve_bind({}, flag=value, env={})
        assert "--bind" in str(err.value)

    @pytest.mark.parametrize("value", INVALID)
    def test_invalid_env_refuses_and_names_the_variable(self, value):
        with pytest.raises(_bind().BindError) as err:
            _bind().resolve_bind({}, env={BIND_ENV: value})
        assert BIND_ENV in str(err.value)

    @pytest.mark.parametrize("value", [v for v in INVALID if v.strip()])
    def test_invalid_config_refuses_and_names_the_key(self, value):
        with pytest.raises(_bind().BindError) as err:
            _bind().resolve_bind({"web": {"bind": value}}, env={})
        assert "web.bind" in str(err.value)

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_string_in_config_is_refused_not_ignored(self, value):
        """`bind:` with nothing after it is None (unset). `bind: ""` is someone
        typing a value, and "no value" is not one we can honour."""
        with pytest.raises(_bind().BindError):
            _bind().resolve_bind({"web": {"bind": value}}, env={})

    @pytest.mark.parametrize("value", [5, 1.5, True, False, ["127.0.0.1"], {"a": 1}])
    def test_non_string_config_values_are_refused(self, value):
        with pytest.raises(_bind().BindError) as err:
            _bind().resolve_bind({"web": {"bind": value}}, env={})
        assert "web.bind" in str(err.value)

    def test_the_message_says_what_would_have_been_valid(self):
        with pytest.raises(_bind().BindError) as err:
            _bind().resolve_bind({}, flag="nope", env={})
        text = str(err.value)
        assert "'nope'" in text
        assert "IPv4" in text and "IPv6" in text and "localhost" in text

    @pytest.mark.parametrize(("flag", "env", "config"), [
        ("garbage", {BIND_ENV: "127.0.0.1"}, {"web": {"bind": "127.0.0.1"}}),
        ("garbage", {}, {"web": {"bind": "0.0.0.0"}}),
        (None, {BIND_ENV: "garbage"}, {"web": {"bind": "127.0.0.1"}}),
        (None, {BIND_ENV: "garbage"}, {}),
        (None, {}, {"web": {"bind": "garbage"}}),
    ])
    def test_an_invalid_value_never_falls_through_to_another_source(
        self, flag, env, config,
    ):
        """The tempting implementation is `try the flag, else the env, else the
        config, else the default`. That turns a typo into 0.0.0.0."""
        with pytest.raises(_bind().BindError):
            _bind().resolve_bind(config, flag=flag, env=env)

    def test_a_bad_lower_source_is_not_reached_when_a_higher_one_is_valid(self):
        """Only the winning source is read: a stale bad value in the file does
        not block an operator who is overriding it on the command line."""
        got = _bind().resolve_bind(
            {"web": {"bind": "garbage"}}, flag="127.0.0.1", env={})
        assert got.address == "127.0.0.1"

    def test_an_empty_env_var_is_refused(self):
        """`PROMETHEUS_WEB_BIND=` is a set variable with no value, which is a
        launcher bug rather than "unset". Treating it as unset would widen."""
        with pytest.raises(_bind().BindError):
            _bind().resolve_bind({}, env={BIND_ENV: ""})


# ---------------------------------------------------------------------------
# The wording shared by the startup log and `oara doctor`
# ---------------------------------------------------------------------------


class TestAllInterfacesWarning:
    @pytest.mark.parametrize("address", ["0.0.0.0", "::"])
    def test_wide_binds_get_a_plain_warning(self, address):
        text = _bind().all_interfaces_warning(address)
        assert text
        assert "all interfaces" in text
        assert address in text
        assert "plain HTTP" in text
        assert "no TLS" in text
        assert "web.bind" in text and "--bind" in text

    def test_the_warning_does_not_claim_encryption(self):
        text = _bind().all_interfaces_warning("0.0.0.0").lower()
        for phrase in ("encrypted", "https", "tls enabled", "secure connection"):
            assert phrase not in text

    @pytest.mark.parametrize("address", ["127.0.0.1", "::1", "192.0.2.10", "2001:db8::10"])
    def test_narrow_binds_get_none(self, address):
        assert _bind().all_interfaces_warning(address) is None


# ---------------------------------------------------------------------------
# COMPATIBILITY: no key, no flag, no env keeps 0.0.0.0
# ---------------------------------------------------------------------------


class TestCompatibility:
    def test_a_real_world_config_without_web_bind_resolves_to_all_interfaces(self):
        """A Mac mini reached over Tailscale has a config like this one."""
        config = {"web": {"enabled": True, "api_port": 8005, "ws_port": 8010,
                          "api_token": "x"}}
        got = _bind().resolve_bind(config, env={})
        assert (got.address, got.source) == ("0.0.0.0", "default")

    def test_the_shipped_template_resolves_to_all_interfaces(self):
        from prometheus.config.template import load_template

        got = _bind().resolve_bind(load_template(), env={})
        assert got.address == "0.0.0.0", (
            "the shipped template must not change the default: copying it "
            "verbatim must leave an existing deployment reachable"
        )

    def test_the_template_ships_the_key_empty_not_set(self):
        """Empty (None) means unset. The key is present so the generated
        reference lists it and a live config that sets it is not "drift"."""
        from prometheus.config.template import load_template

        web = load_template()["web"]
        assert "bind" in web
        assert web["bind"] is None

    def test_the_template_documents_the_setting(self):
        from prometheus.config.template import get_template_path

        text = get_template_path().read_text(encoding="utf-8")
        assert "web.bind" in text or "bind:" in text
        assert BIND_ENV in text
        assert "--bind" in text
        assert "TLS" in text, "the template must say plainly that there is no TLS"


# ---------------------------------------------------------------------------
# The CLI flag, and the daemon refusing to start on a bad value (real children)
# ---------------------------------------------------------------------------


def _child_env(tmp_path, **extra: str) -> dict[str, str]:
    """A child daemon's environment. The ports are ALWAYS free ephemeral ones:
    on a parent without this feature these children do not refuse -- they start
    -- and a start must never land on the real daemon's 8005/8010."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROMETHEUS_")}
    env.update({
        "PYTHONPATH": str(package_src_root()),
        "PROMETHEUS_CONFIG_DIR": str(tmp_path / "confdir"),
        "PROMETHEUS_DATA_DIR": str(tmp_path / "datadir"),
        "PROMETHEUS_ENV_FILE": str(tmp_path / "envfile"),
        "PROMETHEUS_WEB_API_PORT": str(free_port()),
        "PROMETHEUS_WEB_WS_PORT": str(free_port()),
        "PYTHONUNBUFFERED": "1",
    })
    env.update(extra)
    return env


def _write_config(path, **web: object) -> None:
    """A loadable config whose web ports are free ones (see _child_env)."""
    path.write_text(yaml.safe_dump({
        "model": {"provider": "llama_cpp", "model": "m", "base_url": "http://127.0.0.1:1"},
        "web": {"api_port": free_port(), "ws_port": free_port(), **web},
    }), encoding="utf-8")


def _run_child(argv: list[str], tmp_path, **extra: str) -> subprocess.CompletedProcess:
    if repo_config_of_loaded_package().exists():
        pytest.skip(
            f"{repo_config_of_loaded_package()} exists: a child daemon would boot "
            "from that live config instead of the test's"
        )
    try:
        return subprocess.run(
            [sys.executable, *argv], env=_child_env(tmp_path, **extra), cwd=tmp_path,
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"") + (exc.stderr or b""))
        text = out.decode(errors="replace") if isinstance(out, bytes) else str(out)
        pytest.fail(
            "the daemon was still running after 30s instead of refusing to start "
            f"(subprocess.run killed it):\n{text[-1500:]}"
        )


class TestDaemonRefusesToStartOnABadBind:
    def test_a_bad_flag_in_setup_mode(self, tmp_path):
        proc = _run_child(["-m", "prometheus.daemon", "--bind", "nonsense"], tmp_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 2, out
        assert "--bind" in out and "'nonsense'" in out and "IPv4" in out, out
        assert "PROMETHEUS IS IN SETUP MODE" not in out, "it must refuse before it can listen"
        assert "Traceback" not in out, out

    def test_a_bad_env_var_in_setup_mode(self, tmp_path):
        proc = _run_child(["-m", "prometheus.daemon"], tmp_path, **{BIND_ENV: "nonsense"})
        out = proc.stdout + proc.stderr
        assert proc.returncode == 2, out
        assert BIND_ENV in out and "IPv4" in out, out
        assert "PROMETHEUS IS IN SETUP MODE" not in out
        assert "Traceback" not in out, out

    def test_a_bad_config_value_with_a_config_present(self, tmp_path):
        cfg = tmp_path / "prometheus.yaml"
        _write_config(cfg, enabled=True, bind="nonsense")
        proc = _run_child(["-m", "prometheus.daemon", "--config", str(cfg)], tmp_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 2, out
        assert "web.bind" in out and "'nonsense'" in out and "IPv4" in out, out
        assert "Traceback" not in out, out

    def test_an_invalid_flag_is_not_rescued_by_a_valid_env_var(self, tmp_path):
        """The flag outranks the env var; being invalid, it refuses the start.
        It does not fall through to the valid variable (or to the default)."""
        cfg = tmp_path / "prometheus.yaml"
        _write_config(cfg, enabled=False, bind="127.0.0.1")
        proc = _run_child(
            ["-m", "prometheus.daemon", "--config", str(cfg), "--bind", "nonsense"],
            tmp_path, **{BIND_ENV: "127.0.0.1"})
        out = proc.stdout + proc.stderr
        assert proc.returncode == 2, out
        assert "--bind" in out and "'nonsense'" in out, out

    def test_oara_daemon_accepts_and_forwards_the_flag(self, tmp_path):
        """`oara daemon` has its own argparse layer in __main__.py; the flag
        has to exist there too or `oara daemon --bind ...` dies as "unrecognized"."""
        proc = _run_child(["-m", "prometheus", "daemon", "--bind", "nonsense"], tmp_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 2, out
        assert "unrecognized arguments" not in out, out
        assert "--bind" in out and "'nonsense'" in out and "IPv4" in out, out

    @pytest.mark.parametrize("argv", [
        ["-m", "prometheus.daemon", "--help"],
        ["-m", "prometheus", "daemon", "--help"],
    ])
    def test_help_documents_the_flag(self, tmp_path, argv):
        proc = _run_child(argv, tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "--bind" in proc.stdout
        assert "127.0.0.1" in proc.stdout


# ---------------------------------------------------------------------------
# POST /api/setup/configure pins the address setup mode was serving on
# ---------------------------------------------------------------------------

CODE = "042999"


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    from prometheus.config.api_token import TOKEN_ENV_VAR

    confdir = tmp_path / "confdir"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(confdir))
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    monkeypatch.delenv(BIND_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    yield confdir
    os.environ.pop(TOKEN_ENV_VAR, None)


def _configure(setup_env, *, bind=None, env_bind=None, monkeypatch=None):
    """Pair + configure through the real setup app; return (response, written yaml)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.setup_server import PairingState, create_setup_app
    from tests.test_setup_api_phase2 import _FakeLlamaCppHandler, _serve

    if env_bind is not None:
        monkeypatch.setenv(BIND_ENV, env_bind)
    kwargs = {} if bind is None else {"bind": bind}
    app = create_setup_app(PairingState(code=CODE), api_port=8123, ws_port=8124, **kwargs)
    client = TestClient(app)
    token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    with _serve(_FakeLlamaCppHandler) as url:
        resp = client.post("/api/setup/configure", headers=headers, json={
            "provider": "llama_cpp", "base_url": url, "model": "gemma4-26b"})
    assert resp.status_code == 200, resp.text
    written = yaml.safe_load((setup_env / "prometheus.yaml").read_text(encoding="utf-8"))
    return resp.json(), written


class TestConfigurePinsTheBind:
    def test_a_loopback_setup_mode_writes_a_loopback_config(self, setup_env):
        body, cfg = _configure(setup_env, bind="127.0.0.1")
        assert cfg["web"]["bind"] == "127.0.0.1"
        assert body["web"]["bind"] == "127.0.0.1"
        # ...next to the ports it already pinned, which stay exactly as they were.
        assert (cfg["web"]["api_port"], cfg["web"]["ws_port"]) == (8123, 8124)

    def test_the_address_comes_from_the_environment_when_no_argument_is_given(
        self, setup_env, monkeypatch,
    ):
        body, cfg = _configure(setup_env, env_bind="127.0.0.1", monkeypatch=monkeypatch)
        assert cfg["web"]["bind"] == "127.0.0.1"
        assert body["web"]["bind"] == "127.0.0.1"

    def test_a_specific_non_loopback_address_is_pinned_as_given(self, setup_env):
        _body, cfg = _configure(setup_env, bind="192.0.2.10")
        assert cfg["web"]["bind"] == "192.0.2.10"

    def test_ipv6_loopback_is_pinned(self, setup_env):
        _body, cfg = _configure(setup_env, bind="::1")
        assert cfg["web"]["bind"] == "::1"

    def test_localhost_is_pinned_in_its_canonical_form(self, setup_env):
        _body, cfg = _configure(setup_env, bind="localhost")
        assert cfg["web"]["bind"] == "127.0.0.1"

    def test_the_pin_is_what_the_real_daemon_resolves_afterwards(self, setup_env):
        """The whole point: setup mode -> configure -> (restart) -> the daemon
        reads this file with no flag and no env, and lands on the same address."""
        _body, cfg = _configure(setup_env, bind="127.0.0.1")
        got = _bind().resolve_bind(cfg, env={})
        assert (got.address, got.source) == ("127.0.0.1", "config")

    def test_a_default_setup_mode_writes_loopback_explicitly_and_the_response_says_so(
        self, setup_env,
    ):
        """D10: a fresh install listens on this machine only. Setup mode that was reached on the unspecified
        DEFAULT (nobody chose 0.0.0.0) writes `web.bind: 127.0.0.1`, so a box set up from another machine
        is loopback-only afterwards until its owner sets the bind on purpose. This also keeps the file
        byte-identical to what `oara setup --fast` writes."""
        body, cfg = _configure(setup_env)
        assert cfg["web"]["bind"] == "127.0.0.1"
        assert body["web"]["bind"] == "127.0.0.1"
        got = _bind().resolve_bind(cfg, env={})
        assert (got.address, got.source) == ("127.0.0.1", "config")

    def test_a_wide_bind_somebody_chose_is_kept(self, setup_env, monkeypatch):
        """Only the unspecified DEFAULT becomes loopback. `--bind 0.0.0.0` or the environment variable is the
        owner asking for every interface, and configure must not undo it."""
        body, cfg = _configure(setup_env, bind="0.0.0.0")
        assert cfg["web"]["bind"] == "0.0.0.0" and body["web"]["bind"] == "0.0.0.0"

    def test_a_wide_bind_chosen_in_the_environment_is_kept(self, setup_env, monkeypatch):
        body, cfg = _configure(setup_env, env_bind="0.0.0.0", monkeypatch=monkeypatch)
        assert cfg["web"]["bind"] == "0.0.0.0" and body["web"]["bind"] == "0.0.0.0"

    def test_an_invalid_address_never_builds_a_setup_app(self, setup_env):
        pytest.importorskip("fastapi")
        from prometheus.web.setup_server import PairingState, create_setup_app

        with pytest.raises(_bind().BindError):
            create_setup_app(PairingState(code=CODE), api_port=8123, ws_port=8124,
                             bind="nonsense")


# ---------------------------------------------------------------------------
# Setup mode resolves from flag > process environment > env file > default
# ---------------------------------------------------------------------------


class TestSetupResolution:
    """Setup mode has no config, but the real daemon it hands over to loads the
    env file before it resolves -- so setup mode has to read it too, or the same
    PROMETHEUS_WEB_BIND would bind loopback one minute later and wide before."""

    @pytest.fixture
    def envfile(self, tmp_path, monkeypatch):
        path = tmp_path / "envfile"
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
        monkeypatch.delenv(BIND_ENV, raising=False)
        return path

    def _resolve(self, flag=None):
        from prometheus.web.setup_server import resolve_setup_bind

        return resolve_setup_bind(flag)

    def test_nothing_set_is_the_default(self, envfile):
        assert self._resolve() == "0.0.0.0"

    def test_the_env_file_is_read(self, envfile):
        envfile.write_text(f"OTHER=1\n{BIND_ENV}=127.0.0.1\n", encoding="utf-8")
        assert self._resolve() == "127.0.0.1"

    def test_the_process_environment_beats_the_env_file(self, envfile, monkeypatch):
        envfile.write_text(f"{BIND_ENV}=0.0.0.0\n", encoding="utf-8")
        monkeypatch.setenv(BIND_ENV, "::1")
        assert self._resolve() == "::1"

    def test_the_flag_beats_both(self, envfile, monkeypatch):
        envfile.write_text(f"{BIND_ENV}=0.0.0.0\n", encoding="utf-8")
        monkeypatch.setenv(BIND_ENV, "0.0.0.0")
        assert self._resolve("127.0.0.1") == "127.0.0.1"

    def test_an_invalid_env_file_value_refuses(self, envfile):
        envfile.write_text(f"{BIND_ENV}=nonsense\n", encoding="utf-8")
        with pytest.raises(_bind().BindError) as err:
            self._resolve()
        assert BIND_ENV in str(err.value)

    def test_an_unreadable_env_file_refuses_instead_of_guessing_wide(self, envfile):
        envfile.write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
        with pytest.raises(_bind().BindError) as err:
            self._resolve()
        assert "cannot read the env file" in str(err.value)

    def test_the_env_file_is_not_read_when_the_flag_decides(self, envfile):
        """A flag outranks it; an unreadable file must not block that."""
        envfile.write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
        assert self._resolve("127.0.0.1") == "127.0.0.1"


# ---------------------------------------------------------------------------
# The setup-mode banner must not send a loopback-only reader to a LAN address
# ---------------------------------------------------------------------------


class TestPairingBanner:
    def _banner(self, *args):
        from prometheus.web.setup_server import format_pairing_banner

        return format_pairing_banner("123456", 8005, *args)

    def test_unchanged_unless_the_bind_is_loopback(self):
        default = self._banner()
        assert "Tailscale / LAN address" in default
        assert self._banner("0.0.0.0") == default
        assert self._banner("192.0.2.10") == default

    @pytest.mark.parametrize("bind", ["127.0.0.1", "::1"])
    def test_loopback_names_this_machine_and_no_lan_address(self, bind):
        text = self._banner(bind)
        assert "THIS machine" in text
        assert "Tailscale" not in text and "LAN" not in text
        assert "123456" in text
        assert (f"[{bind}]:8005" if ":" in bind else f"{bind}:8005") in text


# ---------------------------------------------------------------------------
# `oara doctor`
# ---------------------------------------------------------------------------


class TestDoctorBindRow:
    def _check(self, config, env=None):
        from prometheus.cli.doctor import check_web_bind

        return check_web_bind(config, env={} if env is None else env)

    def test_default_warns_about_every_interface_without_tls(self):
        row = self._check({"web": {"enabled": True}})
        assert row is not None
        assert row.status == "warning"
        assert row.category == "connectivity"
        assert "all interfaces" in row.message
        assert "0.0.0.0" in row.message
        assert "no TLS" in row.message
        assert row.fix and "web.bind" in row.fix and "127.0.0.1" in row.fix

    def test_it_does_not_claim_tls_exists(self):
        row = self._check({"web": {"enabled": True}})
        text = (row.message + " " + (row.fix or "")).lower()
        for phrase in ("encrypted", "https", "tls enabled", "tls is enabled"):
            assert phrase not in text

    def test_an_explicit_wide_bind_does_not_warn_but_says_what_it_is(self):
        """D10: the warning is for an UNSET bind, the default nobody chose. A deliberate 0.0.0.0 (the Mac mini
        over Tailscale) is an owner's decision; it is still described honestly as plain HTTP on every interface."""
        for cfg in ({"web": {"enabled": True, "bind": "0.0.0.0"}},
                    {"web": {"enabled": True, "bind": "::"}}):
            row = self._check(cfg)
            assert row is not None and row.status == "ok"
            assert "all interfaces" in row.message and "no TLS" in row.message
            assert "web.bind" in row.message          # says where the choice was made

    def test_a_wide_bind_chosen_in_the_environment_does_not_warn(self):
        row = self._check({"web": {"enabled": True}}, env={BIND_ENV: "0.0.0.0"})
        assert row is not None and row.status == "ok" and "all interfaces" in row.message

    def test_the_unset_warning_tells_an_owner_how_to_choose(self):
        row = self._check({"web": {"enabled": True}})
        assert row.status == "warning"
        assert "not set" in row.message
        assert "127.0.0.1" in row.fix and "0.0.0.0" in row.fix, "narrow it, or keep it wide on purpose"

    def test_loopback_is_ok_and_says_so(self):
        row = self._check({"web": {"enabled": True, "bind": "127.0.0.1"}})
        assert row is not None and row.status == "ok"
        assert "127.0.0.1" in row.message
        assert "this machine only" in row.message

    def test_a_specific_address_is_ok_but_still_plain_http(self):
        row = self._check({"web": {"enabled": True, "bind": "192.0.2.10"}})
        assert row is not None and row.status == "ok"
        assert "192.0.2.10" in row.message
        assert "no TLS" in row.message

    def test_the_environment_variable_is_honoured(self):
        row = self._check({"web": {"enabled": True}}, env={BIND_ENV: "127.0.0.1"})
        assert row is not None and row.status == "ok"

    def test_an_invalid_value_is_an_error_row_not_a_traceback(self):
        row = self._check({"web": {"enabled": True, "bind": "nonsense"}})
        assert row is not None and row.status == "error"
        assert "web.bind" in row.message
        assert "refuse" in row.message.lower()

    def test_web_disabled_has_no_bind_row(self):
        assert self._check({"web": {"enabled": False}}) is None
        assert self._check({}) is None

    def test_the_row_is_part_of_the_extended_checks(self, monkeypatch):
        from prometheus.cli import doctor

        ok = doctor.DiagnosticCheck
        monkeypatch.setattr(
            doctor, "check_inference",
            lambda *a, **k: (ok("Inference", "connectivity", "ok", "x"),
                             ok("Model", "model", "ok", "x")))
        # The real port check connects to :8005; a developer's daemon may be there.
        monkeypatch.setattr(
            doctor, "check_web_port", lambda *a, **k: ok("Web", "connectivity", "ok", "x"))
        monkeypatch.delenv(BIND_ENV, raising=False)
        checks = doctor.run_extended_checks(
            {"web": {"enabled": True}},
            config_check=doctor.DiagnosticCheck("Config", "platform", "ok", "x"),
        )
        rows = [c for c in checks if c.name == "Web bind"]
        assert len(rows) == 1, [c.name for c in checks]
        assert rows[0].status == "warning"
