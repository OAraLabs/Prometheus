"""``oara tailscale check|setup|off`` and the "Tailscale HTTPS" doctor rows (HTTPS over Tailscale, PR 2).

Driven against a FAKE ``tailscale`` on PATH (a script run by this interpreter, never a crashing child) that answers
``status --json`` and ``serve status --json`` from files, applies ``serve --bg --https=…`` and ``serve --https=… off``
to its own state, and logs every call. Held here:

  * check is read-only; setup prints the plan and changes nothing without ``--apply``; with it, it runs exactly the
    Serve commands it needs and writes ``web.tailscale`` keeping the file's comments;
  * it never replaces a Serve entry it did not create (a port another app serves is reported and left alone), never
    turns Funnel on, and is blocked (changing nothing) by: Tailscale not running, HTTPS certificates off, a wide
    ``web.bind``, no API token, ``trusted_proxies`` covering loopback, Funnel on a port it would use;
  * when Serve refuses for permission it stops, changes no config, and prints the one command to run as the right
    user; it never runs ``sudo``, ``cert``, ``funnel``, ``up``, ``login`` or ``set`` itself;
  * off removes only its own two entries and sets ``web.tailscale.enabled: false``;
  * the doctor rows: none without the block; each failure named; a certificate that does not verify, or expires
    soon, says so.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

NAME = "studio.tail1234.ts.net"

_FAKE = '''#!{python}
import json, os, sys
d = os.environ["FAKE_TS_DIR"]
args = sys.argv[1:]
with open(os.path.join(d, "calls.log"), "a") as fh:
    fh.write(json.dumps(args) + "\\n")
def load(n, default):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else default
status = load("status.json", {{}})
serve = load("serve.json", {{}})
name = (status.get("Self") or {{}}).get("DNSName", "").rstrip(".")
if args == ["status", "--json"]:
    print(json.dumps(status)); sys.exit(0)
if args == ["serve", "status", "--json"]:
    print(json.dumps(serve)); sys.exit(0)
if args[:1] == ["serve"]:
    fail = load("fail.json", None)
    if fail:
        sys.stderr.write(fail["stderr"]); sys.exit(fail["rc"])
    port = next(a.split("=", 1)[1] for a in args if a.startswith("--https="))
    if args[-1] == "off":
        serve.get("TCP", {{}}).pop(port, None); serve.get("Web", {{}}).pop(name + ":" + port, None)
    else:
        serve.setdefault("TCP", {{}})[port] = {{"HTTPS": True}}
        serve.setdefault("Web", {{}})[name + ":" + port] = {{"Handlers": {{"/": {{"Proxy": args[-1]}}}}}}
    json.dump(serve, open(os.path.join(d, "serve.json"), "w")); sys.exit(0)
sys.stderr.write("unexpected: " + " ".join(args)); sys.exit(2)
'''


@pytest.fixture
def ts(tmp_path, monkeypatch):
    """A fake tailscale on PATH, a healthy node, no Serve entries, and a config with a token and a loopback bind."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "tailscale"
    fake.write_text(_FAKE.format(python=sys.executable), encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    state = tmp_path / "ts"
    state.mkdir()
    monkeypatch.setenv("FAKE_TS_DIR", str(state))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("PROMETHEUS_API_TOKEN", "tailscale-cli-test-token-not-real")
    (state / "status.json").write_text(json.dumps({
        "BackendState": "Running", "Self": {"DNSName": NAME + "."}, "CertDomains": [NAME]}))
    config = tmp_path / "prometheus.yaml"
    config.write_text("# my daemon\nweb:\n  bind: 127.0.0.1  # loopback\n  api_port: 8005\n  ws_port: 8010\n",
                      encoding="utf-8")

    class World:
        dir = state
        config_path = config

        def calls(self):
            log = state / "calls.log"
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

        def serve(self, body):
            (state / "serve.json").write_text(json.dumps(body))

        def status(self, **changes):
            body = json.loads((state / "status.json").read_text())
            body.update(changes)
            (state / "status.json").write_text(json.dumps(body))

        def run(self, action, **kw):
            from prometheus.cli.tailscale import run_tailscale_command

            args = argparse.Namespace(tailscale_action=action, config=str(config), apply=kw.pop("apply", False),
                                      rest_port=kw.pop("rest_port", None), ws_port=kw.pop("ws_port", None))
            loaded = yaml.safe_load(config.read_text()) or {}
            return run_tailscale_command(args, loaded)

    return World()


def _writes(calls):
    return [c for c in calls if c[:1] == ["serve"] and c[1:2] != ["status"]]


# ── check and setup ──────────────────────────────────────────────────────────

def test_check_is_read_only_and_shows_the_plan(ts, capsys):
    assert ts.run("check") == 0
    out = capsys.readouterr().out
    assert f"https://{NAME}" in out and f"wss://{NAME}:8443" in out and "will run: tailscale serve" in out
    assert _writes(ts.calls()) == []


def test_setup_without_apply_changes_nothing(ts, capsys):
    before = ts.config_path.read_text()
    assert ts.run("setup") == 0
    assert "Dry run" in capsys.readouterr().out
    assert _writes(ts.calls()) == [] and ts.config_path.read_text() == before


def test_setup_with_apply_runs_the_two_serve_commands_and_writes_the_block(ts):
    assert ts.run("setup", apply=True) == 0
    assert _writes(ts.calls()) == [["serve", "--bg", "--https=443", "http://127.0.0.1:8005"],
                                   ["serve", "--bg", "--https=8443", "http://127.0.0.1:8010"]]
    text = ts.config_path.read_text()
    assert "# my daemon" in text and "# loopback" in text                    # comments kept
    block = yaml.safe_load(text)["web"]["tailscale"]
    assert block == {"enabled": True, "name": NAME, "rest_port": 443, "ws_port": 8443}


def test_a_port_another_app_serves_is_left_alone(ts, capsys):
    other = {"TCP": {"443": {"HTTPS": True}}, "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8090"}}}}}
    ts.serve(other)
    assert ts.run("setup", apply=True) == 1
    assert "already serves something else" in capsys.readouterr().out
    assert _writes(ts.calls()) == [] and "tailscale" not in (yaml.safe_load(ts.config_path.read_text())["web"])
    assert ts.run("setup", apply=True, rest_port=8443, ws_port=8444) == 0
    serve = json.loads((ts.dir / "serve.json").read_text())
    assert serve["Web"][f"{NAME}:443"] == other["Web"][f"{NAME}:443"]           # untouched
    assert serve["Web"][f"{NAME}:8443"]["Handlers"]["/"]["Proxy"] == "http://127.0.0.1:8005"


def test_an_entry_that_already_relays_to_the_daemon_is_kept_not_redone(ts):
    ts.serve({"TCP": {"443": {"HTTPS": True}},
              "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8005"}}}}})
    assert ts.run("setup", apply=True) == 0
    assert _writes(ts.calls()) == [["serve", "--bg", "--https=8443", "http://127.0.0.1:8010"]]


@pytest.mark.parametrize(("why", "arrange"), [
    ("not running", lambda w: w.status(BackendState="Stopped")),
    ("HTTPS certificates off", lambda w: w.status(CertDomains=[])),
    ("Funnel on", lambda w: w.serve({"AllowFunnel": {f"{NAME}:443": True}})),
    ("no MagicDNS name", lambda w: w.status(Self={"DNSName": ""})),
])
def test_setup_is_blocked_and_changes_nothing(ts, why, arrange, capsys):
    arrange(ts)
    before = ts.config_path.read_text()
    assert ts.run("setup", apply=True) == 1, why
    assert "BLOCKED" in capsys.readouterr().out
    assert _writes(ts.calls()) == [] and ts.config_path.read_text() == before


@pytest.mark.parametrize(("why", "edit"), [
    ("a wide bind", lambda t: t.replace("bind: 127.0.0.1", "bind: 0.0.0.0")),
    ("trusted_proxies covering loopback", lambda t: t + "  trusted_proxies: [127.0.0.1]\n"),
])
def test_setup_is_blocked_by_the_config(ts, why, edit):
    ts.config_path.write_text(edit(ts.config_path.read_text()))
    assert ts.run("setup", apply=True) == 1, why
    assert _writes(ts.calls()) == []


def test_setup_is_blocked_without_an_api_token(ts, monkeypatch):
    monkeypatch.setattr("prometheus.cli.tailscale._has_token", lambda config: False)
    assert ts.run("setup", apply=True) == 1
    assert _writes(ts.calls()) == []


def test_a_permission_refusal_stops_changes_no_config_and_says_what_to_run(ts, capsys):
    (ts.dir / "fail.json").write_text(json.dumps({"rc": 1, "stderr": "Access denied: serve config denied"}))
    before = ts.config_path.read_text()
    assert ts.run("setup", apply=True) == 1
    out = capsys.readouterr().out
    assert "sudo tailscale set --operator=$USER" in out and "config was not changed" in out
    assert ts.config_path.read_text() == before
    assert not any("sudo" in c or "set" in c for c in ts.calls())


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_removes_only_its_own_entries_and_turns_the_block_off(ts):
    assert ts.run("setup", apply=True, rest_port=8443, ws_port=8444) == 0
    serve = json.loads((ts.dir / "serve.json").read_text())
    serve["TCP"]["443"] = {"HTTPS": True}
    serve["Web"][f"{NAME}:443"] = {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8090"}}}
    ts.serve(serve)
    assert ts.run("off") == 0                                             # dry run
    assert ts.run("off", apply=True) == 0
    assert [c for c in _writes(ts.calls()) if c[-1] == "off"] == [["serve", "--https=8443", "off"],
                                                                  ["serve", "--https=8444", "off"]]
    after = json.loads((ts.dir / "serve.json").read_text())
    assert f"{NAME}:443" in after["Web"] and f"{NAME}:8443" not in after["Web"]
    assert yaml.safe_load(ts.config_path.read_text())["web"]["tailscale"]["enabled"] is False


# ── what it will never run ───────────────────────────────────────────────────

@pytest.mark.parametrize("args", [("cert", NAME), ("funnel", "--https=443", "on"), ("up",), ("set", "--operator=x"),
                                  ("login",), ("serve", "--funnel"), ("sudo",), ("down",)])
def test_it_refuses_to_run_anything_but_status_and_serve(args):
    from prometheus.cli.tailscale import TailscaleCommandRefused, run_tailscale

    with pytest.raises(TailscaleCommandRefused):
        run_tailscale("/nonexistent/tailscale", *args)


def test_the_cli_is_registered():
    from prometheus.cli.tailscale import add_tailscale_subparser

    parser = argparse.ArgumentParser()
    add_tailscale_subparser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["tailscale", "setup", "--apply", "--rest-port", "8443", "--ws-port", "8444"])
    assert (args.command, args.tailscale_action, args.apply, args.rest_port, args.ws_port) == (
        "tailscale", "setup", True, 8443, 8444)
    import inspect

    import prometheus.__main__ as main_module

    source = inspect.getsource(main_module)
    assert "add_tailscale_subparser(subparsers)" in source and 'args.command == "tailscale"' in source


# ── doctor rows ──────────────────────────────────────────────────────────────

def _cfg(**web):
    return {"web": {"bind": "127.0.0.1", "api_port": 8005, "ws_port": 8010,
                    "tailscale": {"enabled": True, "name": NAME, "rest_port": 8443, "ws_port": 8444}, **web}}


def _state(entries=None, **kw):
    from prometheus.cli.tailscale import ServeEntry, TailscaleState

    default = {8443: ServeEntry(8443, {"/": "http://127.0.0.1:8005"}), 8444: ServeEntry(8444, {"/": "http://127.0.0.1:8010"})}
    return TailscaleState(binary="tailscale", running=kw.get("running", True), backend_state="Running",
                          name=kw.get("name", NAME), cert_domains=[NAME],
                          entries=default if entries is None else entries)


def _rows(config, state=None, days=60.0):
    from prometheus.cli.tailscale import doctor_rows

    def handshake(name, port):
        if isinstance(days, Exception):
            raise days
        return days

    return {r.name: r for r in doctor_rows(config, state=state or _state(), handshake=handshake)}


def test_no_block_no_rows():
    from prometheus.cli.tailscale import doctor_rows

    assert doctor_rows({"web": {}}) == []


def test_a_healthy_setup_is_all_ok(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_API_TOKEN", "x-not-real")
    rows = _rows(_cfg())
    assert {n: r.status for n, r in rows.items()} == {
        "Tailscale": "ok", "Tailscale Serve": "ok", "Tailscale Funnel": "ok", "Tailscale bind": "ok",
        "Tailscale certificate": "ok"}


def test_each_failure_is_named(monkeypatch):
    from prometheus.cli.tailscale import ServeEntry

    monkeypatch.setenv("PROMETHEUS_API_TOKEN", "x-not-real")
    assert _rows(_cfg(), _state(name="other.tail1234.ts.net"))["Tailscale"].status == "error"
    assert _rows(_cfg(), _state(entries={}))["Tailscale Serve"].status == "warning"
    taken = {8443: ServeEntry(8443, {"/": "http://127.0.0.1:9999"}), 8444: ServeEntry(8444, {"/": "http://127.0.0.1:8010"})}
    assert _rows(_cfg(), _state(entries=taken))["Tailscale Serve"].status == "error"
    funnel = {8443: ServeEntry(8443, {"/": "http://127.0.0.1:8005"}, funnel=True),
              8444: ServeEntry(8444, {"/": "http://127.0.0.1:8010"})}
    assert _rows(_cfg(), _state(entries=funnel))["Tailscale Funnel"].status == "error"
    assert _rows(_cfg(bind="0.0.0.0"))["Tailscale bind"].status == "error"
    assert _rows(_cfg(), days=5.0)["Tailscale certificate"].status == "warning"
    assert _rows(_cfg(), days=-1.0)["Tailscale certificate"].status == "error"
    bad = _rows(_cfg(), days=ConnectionRefusedError("no"))["Tailscale certificate"]
    assert bad.status == "error" and "did not verify" in bad.message


def test_a_block_the_daemon_would_ignore_is_one_error_row(monkeypatch):
    from prometheus.cli.tailscale import doctor_rows

    monkeypatch.setattr("prometheus.cli.tailscale._has_token", lambda config: False)
    rows = doctor_rows(_cfg(), state=_state(), handshake=lambda n, p: 60.0)
    assert len(rows) == 1 and rows[0].status == "error" and "ignores it" in rows[0].message


def test_a_missing_cli_is_not_checked_never_ok(monkeypatch):
    from prometheus.cli.tailscale import TailscaleState, doctor_rows

    monkeypatch.setenv("PROMETHEUS_API_TOKEN", "x-not-real")
    rows = doctor_rows(_cfg(), state=TailscaleState(problems=["the tailscale command was not found"]),
                       handshake=lambda n, p: 60.0)
    assert len(rows) == 1 and rows[0].status == "error" and "not checked" in (rows[0].fix or "")


def test_the_doctor_runs_the_rows():
    import inspect

    from prometheus.cli import doctor

    assert "tailscale_doctor_rows(config, timeout=timeout)" in inspect.getsource(doctor.run_extended_checks)


def test_the_daemon_settings_take_the_token_answer_from_the_cli(monkeypatch):
    from prometheus.web.tailscale_serve import serve_settings

    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    assert serve_settings(_cfg()) is None
    assert serve_settings(_cfg(), has_token=True) is not None
    assert serve_settings(_cfg(), has_token=False) is None


# ── #928 review ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("args", [
    ("serve", "reset"),
    ("serve", "--bg", "--https=443", "http://evil.example:80"),
    ("serve", "--bg", "--set-path=/x", "--https=443", "http://127.0.0.1:8005"),
    ("serve", "--bg", "--https=0", "http://127.0.0.1:8005"),
    ("serve", "--bg", "--https=70000", "http://127.0.0.1:8005"),
    ("serve", "--https=443", "on"),
    ("serve", "--tcp=443", "off"),
    ("status", "--peers"),
    ("status",),
    ("serve", "status"),
])
def test_only_the_four_exact_command_shapes_run(args):
    """Both reviews of #928: the guard was 'first word status or serve, no forbidden word', which let `serve reset`
    or a proxy to another host through. Now only the four documented shapes, ports checked as integers."""
    from prometheus.cli.tailscale import TailscaleCommandRefused, run_tailscale

    with pytest.raises(TailscaleCommandRefused):
        run_tailscale("/nonexistent/tailscale", *args)


def test_the_four_shapes_are_accepted(monkeypatch):
    import subprocess

    from prometheus.cli import tailscale as t

    seen: list = []

    def fake_run(argv, **kw):
        seen.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr(t.subprocess, "run", fake_run)
    for args in (("status", "--json"), ("serve", "status", "--json"),
                 ("serve", "--bg", "--https=8443", "http://127.0.0.1:8005"),
                 ("serve", "--bg", "--https=8444", "http://[::1]:8010"),
                 ("serve", "--https=8443", "off")):
        assert t.run_tailscale("/x/tailscale", *args)[0] == 0
    for _argv, kw in seen:                                          # no prompt can wait on the terminal; UTF-8 always
        assert kw.get("stdin") is subprocess.DEVNULL and kw.get("encoding") == "utf-8" and kw.get("errors") == "replace"


@pytest.mark.parametrize(("why", "edit", "env"), [
    ("an unset bind is every interface", lambda t: t.replace("  bind: 127.0.0.1  # loopback\n", ""), {}),
    ("the environment overrides the file", lambda t: t, {"PROMETHEUS_WEB_BIND": "0.0.0.0"}),
])
def test_the_bind_is_the_one_the_daemon_will_use(ts, monkeypatch, why, edit, env, capsys):
    ts.config_path.write_text(edit(ts.config_path.read_text()))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)        # a terminal install: no app LAN guard
    assert ts.run("setup", apply=True) == 1, why
    assert "not loopback" in capsys.readouterr().out
    assert _writes(ts.calls()) == []


def test_a_config_it_cannot_write_runs_no_serve_command(ts, capsys):
    """Review M2: Serve ran first and the write failed after, leaving live relays. The edit is checked first."""
    ts.config_path.write_text("web: {bind: 127.0.0.1, api_port: 8005, ws_port: 8010}\n")   # flow style: not editable
    assert ts.run("setup", apply=True) == 1
    assert _writes(ts.calls()) == []
    assert "Could not write" in capsys.readouterr().out


def test_a_plain_http_serve_entry_is_not_ours(ts):
    ts.serve({"TCP": {"443": {"HTTP": True}},
              "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8005"}}}}})
    assert ts.run("setup", apply=True) == 1                          # taken, not kept
    assert _writes(ts.calls()) == []


def test_a_port_in_two_layers_is_not_ours():
    from prometheus.cli.tailscale import _parse_serve, classify

    entries = _parse_serve({
        "TCP": {"443": {"HTTPS": True}},
        "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}}},
        "Foreground": {"s1": {"TCP": {"443": {"HTTPS": True}},
                              "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8005"}}}}}},
    }, NAME)
    assert classify(entries[443], 8005) == "taken"


def test_the_relay_follows_an_ipv6_loopback_bind(ts):
    ts.config_path.write_text(ts.config_path.read_text().replace("bind: 127.0.0.1", "bind: '::1'"))
    assert ts.run("setup", apply=True) == 0
    assert _writes(ts.calls())[0] == ["serve", "--bg", "--https=443", "http://[::1]:8005"]


@pytest.mark.parametrize("bad", ["abc", True, 0, 70000, [443]])
def test_a_bad_port_in_the_config_is_said_not_a_traceback(ts, bad, capsys):
    text = ts.config_path.read_text() + f"  tailscale:\n    rest_port: {json.dumps(bad)}\n"
    ts.config_path.write_text(text)
    assert ts.run("setup", apply=True) == 1
    assert "port" in capsys.readouterr().out.lower()
    assert _writes(ts.calls()) == []


def test_off_warns_when_it_cannot_read_serve(ts, capsys):
    (ts.dir / "status.json").write_text("not json")
    assert ts.run("off") == 0
    assert "WARNING" in capsys.readouterr().out


def test_when_tailscale_is_not_running_the_other_rows_are_not_checked(monkeypatch):
    from prometheus.cli.tailscale import doctor_rows

    monkeypatch.setenv("PROMETHEUS_API_TOKEN", "x-not-real")
    shaken: list = []
    rows = doctor_rows(_cfg(), state=_state(running=False), handshake=lambda n, p: shaken.append(n) or 60.0)
    assert [r.name for r in rows] == ["Tailscale"] and rows[0].status == "error"
    assert "not checked" in (rows[0].fix or "") and shaken == []


def test_the_literal_writer_never_writes_a_raw_line_separator():
    from prometheus.web.network import _literal

    assert _literal("a b") == '"a\\u2028b"'


def test_an_ipv4_mapped_loopback_bind_relays_to_127_0_0_1(ts):
    """#928 confirm round: '::ffff:127.0.0.1' counted as loopback but made a relay target the allowlist refuses."""
    ts.config_path.write_text(ts.config_path.read_text().replace("bind: 127.0.0.1", "bind: '::ffff:127.0.0.1'"))
    assert ts.run("setup", apply=True) == 0
    assert _writes(ts.calls())[0] == ["serve", "--bg", "--https=443", "http://127.0.0.1:8005"]


def test_a_blocked_plan_lists_no_commands(ts, capsys):
    ts.status(BackendState="Stopped")
    ts.run("check")
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "will run" not in out
