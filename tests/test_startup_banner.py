"""The startup banner: what it says, when it prints, and what it must never contain.

``render_banner`` is pure, so the layout, the colours and the omit-what-is-unknown rule are
pinned on strings. ``show_banner`` is pinned on fake streams: a terminal prints, a pipe does
not, and ``--quiet``, ``OARA_NO_BANNER`` and ``NO_COLOR`` each do their one job. The entry
points are pinned through the real parsers; the real processes on a real terminal are in
test_startup_banner_terminal.py.
"""

from __future__ import annotations

import argparse
import inspect
import io
import re
import sys

import pytest

from prometheus import banner
from prometheus.banner import (
    color_enabled,
    render_banner,
    should_show,
    show_banner,
    version_line,
    write_version,
)
from prometheus.version import package_version

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SANGUINE = "\x1b[38;2;184;71;58m"
BRASS = "\x1b[38;2;179;160;117m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"

# Will's spec, verbatim (2-space indent).
FULL = """\
  ┏━┓┏━┓┏━┓┏┳┓┏━╸╺┳╸╻ ╻┏━╸╻ ╻┏━┓
  ┣━┛┣┳┛┃ ┃┃┃┃┣╸  ┃ ┣━┫┣╸ ┃ ┃┗━┓
  ╹  ╹┗╸┗━┛╹ ╹┗━╸ ╹ ╹ ╹┗━╸┗━┛┗━┛
  ──────────────────────────────
  a fire you own doesn't go out

  ◉ daemon   listening on :8005
  ◉ model    qwen3.6-27b · llama.cpp
  ◉ beacon   oara.ai/beacon

  OAra Labs · prometheus 0.9.8
"""


class Tty(io.StringIO):
    """A stream that says it is a terminal."""

    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("OARA_NO_BANNER", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)


def _full(**over):
    args = dict(version="0.9.8", port=8005, model="qwen3.6-27b", backend="llama_cpp", color=False)
    args.update(over)
    return render_banner(**args)


# --- the layout -----------------------------------------------------------------------------


def test_the_plain_banner_is_the_spec_exactly():
    assert _full() == FULL


def test_the_colour_is_only_around_the_text_it_names():
    plain = [ANSI.sub("", line) for line in _full(color=True).splitlines()]
    assert "\n".join(plain) + "\n" == FULL


def test_colour_roles_are_the_specs():
    lines = _full(color=True).splitlines()
    for row in lines[0:3]:  # the wordmark: sanguine
        assert row.startswith(f"  {SANGUINE}") and row.endswith(RESET)
    assert lines[3].startswith(f"  {BRASS}─") and lines[3].endswith(RESET)  # the rule: brass
    assert lines[4] == f"  {BRASS}a fire you own doesn't go out{RESET}"  # the tagline: brass
    assert lines[6] == f"  {SANGUINE}◉{RESET} {DIM}daemon{RESET}   listening on :8005"  # dot sanguine, label dim, value plain
    assert lines[7] == f"  {SANGUINE}◉{RESET} {DIM}model{RESET}    qwen3.6-27b · llama.cpp"
    assert lines[8] == f"  {SANGUINE}◉{RESET} {DIM}beacon{RESET}   oara.ai/beacon"
    assert lines[10] == f"  {DIM}OAra Labs · prometheus 0.9.8{RESET}"  # the last line: dim


def test_colour_is_24_bit():
    assert (banner.SANGUINE, banner.BRASS) == ((0xB8, 0x47, 0x3A), (0xB3, 0xA0, 0x75))
    assert "\x1b[38;2;184;71;58m" in _full(color=True)


def test_plain_text_has_no_escape_and_no_trailing_space():
    text = _full(color=False)
    assert "\x1b" not in text
    assert all(line == line.rstrip() for line in text.splitlines())


def test_the_wordmark_and_rule_are_thirty_columns():
    wordmark_and_rule = FULL.splitlines()[:4]
    assert {len(line) for line in wordmark_and_rule} == {32}  # 2 indent + 30


def test_values_start_in_one_column():
    rows = [line for line in _full().splitlines() if line.startswith("  ◉")]
    assert len(rows) == 3
    assert {line.index(value) for line, value in zip(rows, ("listening", "qwen", "oara.ai"))} == {13}


# --- a value that is not known is left out, never guessed -----------------------------------


def test_no_port_means_no_daemon_line():
    text = _full(port=None)
    assert "daemon" not in text and "listening" not in text
    assert "model" in text and "beacon" in text


def test_no_model_means_no_model_line_even_with_a_backend():
    text = _full(model=None)
    assert "model" not in text and "llama" not in text


def test_no_backend_means_the_model_alone():
    assert "◉ model    qwen3.6-27b\n" in _full(backend=None)


def test_a_backend_without_a_display_name_is_shown_as_its_key():
    assert "qwen3.6-27b · anthropic\n" in _full(backend="anthropic")


def test_nothing_known_is_the_wordmark_and_the_beacon_line():
    text = render_banner(version="0.9.8", color=False)
    assert text.splitlines()[:5] == FULL.splitlines()[:5]
    assert [line for line in text.splitlines() if "◉" in line] == ["  ◉ beacon   oara.ai/beacon"]


@pytest.mark.parametrize("port", [0, -1, 65536, True, "8005", 80.5, None])
def test_a_port_that_is_not_a_port_is_left_out(port):
    assert "daemon" not in _full(port=port)


def test_the_highest_port_is_shown():
    assert "listening on :65535\n" in _full(port=65535)


# --- nothing that must not be shown ---------------------------------------------------------


def test_only_the_values_the_banner_shows_can_be_passed_in():
    """No parameter can carry a host, a token or a path: there is nowhere to put one."""
    assert set(inspect.signature(render_banner).parameters) == {"version", "port", "model", "backend", "color"}
    assert set(inspect.signature(show_banner).parameters) == {
        "port", "model", "backend", "version", "quiet", "stream", "env",
    }


@pytest.mark.parametrize(
    "model, shown",
    [
        ("/home/someone/models/qwen-27b-q4.gguf", "qwen-27b-q4.gguf"),
        ("C:\\Users\\someone\\models\\qwen-27b.gguf", "qwen-27b.gguf"),
        ("https://models.example.test/v1/qwen-27b", "qwen-27b"),
        ("hf.co/some-org/qwen-27b:Q4_K_M", "qwen-27b:Q4_K_M"),
        ("qwen3.6-27b", "qwen3.6-27b"),
    ],
)
def test_a_model_named_by_its_location_is_shown_by_its_last_part(model, shown):
    text = _full(model=model)
    assert f"◉ model    {shown} · llama.cpp\n" in text
    for leaked in ("someone", "example.test", "some-org", "/", "\\"):
        assert leaked not in text.replace("oara.ai/beacon", "")


def test_a_model_name_cannot_carry_an_escape_sequence():
    text = _full(model="evil\x1b[2J\x1b]0;owned\x07-27b\r\nfake  line", color=False)
    assert "\x1b" not in text and "\x07" not in text and "\r" not in text
    assert len(text.splitlines()) == len(FULL.splitlines())


def test_a_long_model_name_is_cut():
    line = next(row for row in _full(model="m" * 500).splitlines() if "model" in row)
    assert "…" in line and "m" * 61 not in line and line.endswith("llama.cpp")


def test_the_environment_does_not_reach_the_text(monkeypatch):
    token = "tok" + "en" * 8  # letters only
    monkeypatch.setenv("PROMETHEUS_API_TOKEN", token)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 20)
    monkeypatch.setenv("HOME", "/home/someone")
    out = Tty()
    assert show_banner(port=8005, model="qwen3.6-27b", backend="llama_cpp", stream=out) is True
    for secret in (token, "sk-ant-", "/home/someone", "someone"):
        assert secret not in out.getvalue()


def test_the_text_names_no_host_or_address():
    text = _full()
    assert not re.search(r"\d+\.\d+\.\d+\.\d+|localhost|0\.0\.0\.0|https?://|://|@", text)


# --- when it prints -------------------------------------------------------------------------


def test_a_terminal_gets_the_banner():
    out = Tty()
    assert show_banner(port=8005, model="qwen3.6-27b", backend="llama_cpp", version="0.9.8", stream=out, env={})
    assert ANSI.sub("", out.getvalue()) == FULL
    assert SANGUINE in out.getvalue()


def test_a_pipe_gets_nothing():
    out = io.StringIO()  # isatty() is False: a pipe, a file, journald, CI
    assert show_banner(port=8005, model="m", version="0.9.8", stream=out, env={}) is False
    assert out.getvalue() == ""


def test_quiet_gets_nothing_even_on_a_terminal():
    out = Tty()
    assert show_banner(port=8005, model="m", version="0.9.8", quiet=True, stream=out, env={}) is False
    assert out.getvalue() == ""


@pytest.mark.parametrize("value, printed", [("1", False), ("true", False), ("yes", False), ("0", True), ("", True), ("false", True)])
def test_oara_no_banner(value, printed):
    out = Tty()
    assert show_banner(version="0.9.8", stream=out, env={"OARA_NO_BANNER": value}) is printed
    assert bool(out.getvalue()) is printed


def test_no_color_prints_the_same_banner_without_colour():
    out = Tty()
    assert show_banner(port=8005, model="qwen3.6-27b", backend="llama_cpp", version="0.9.8", stream=out, env={"NO_COLOR": "1"})
    assert out.getvalue() == FULL
    assert "\x1b" not in out.getvalue()


def test_an_empty_no_color_is_not_set():
    assert color_enabled({"NO_COLOR": ""}) is True
    assert color_enabled({"NO_COLOR": "1"}) is False
    assert color_enabled({}) is True


def test_the_process_environment_is_what_counts_by_default(monkeypatch):
    out = Tty()
    monkeypatch.setenv("OARA_NO_BANNER", "1")
    assert show_banner(version="0.9.8", stream=out) is False
    monkeypatch.delenv("OARA_NO_BANNER")
    monkeypatch.setenv("NO_COLOR", "1")
    assert show_banner(version="0.9.8", stream=out) is True
    assert "\x1b" not in out.getvalue()


def test_stdout_is_looked_up_when_called(monkeypatch):
    out = Tty()
    monkeypatch.setattr(sys, "stdout", out)
    assert show_banner(version="0.9.8", env={}) is True
    assert "oara.ai/beacon" in out.getvalue()


def test_the_version_defaults_to_the_package_version():
    out = Tty()
    show_banner(stream=out, env={"NO_COLOR": "1"})
    assert out.getvalue().rstrip().endswith(f"OAra Labs · prometheus {package_version()}")


def test_a_stream_without_isatty_or_a_closed_one_gets_nothing():
    class NoIsatty:
        def write(self, text):  # pragma: no cover - must not be reached
            raise AssertionError("wrote to a stream that is not a terminal")

    closed = Tty()
    closed.close()
    assert should_show(NoIsatty()) is False
    assert show_banner(version="0.9.8", stream=closed, env={}) is False


def test_a_terminal_that_cannot_draw_the_wordmark_gets_nothing():
    raw = io.BytesIO()

    class Ascii(io.TextIOWrapper):
        def isatty(self) -> bool:
            return True

    out = Ascii(raw, encoding="ascii")
    assert show_banner(version="0.9.8", stream=out, env={}) is False
    out.flush()
    assert raw.getvalue() == b""


def test_a_stream_that_fails_does_not_fail_the_caller():
    class Broken(Tty):
        def write(self, text):
            raise OSError("closed pipe")

    assert show_banner(version="0.9.8", stream=Broken(), env={}) is False


# --- --version ------------------------------------------------------------------------------


def test_the_version_line_is_the_spec():
    assert version_line("0.9.8") == "◉ prometheus 0.9.8 · OAra Labs"


def test_write_version_is_one_plain_line():
    out = io.StringIO()
    write_version(out, "0.9.8")
    assert out.getvalue() == "◉ prometheus 0.9.8 · OAra Labs\n"


def test_write_version_degrades_to_ascii_instead_of_raising():
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="ascii")
    write_version(out, "0.9.8")
    out.flush()
    assert raw.getvalue() == b"prometheus 0.9.8 - OAra Labs\n"


# --- the entry points -----------------------------------------------------------------------


def _main(monkeypatch, *argv):
    from prometheus import __main__ as cli

    monkeypatch.setattr(sys, "argv", ["oara", *argv])
    return cli.main()


@pytest.mark.parametrize("name", ["oara", "prometheus"])
def test_dash_dash_version_prints_one_line_for_either_command_name(monkeypatch, capsys, name):
    from prometheus import __main__ as cli

    monkeypatch.setattr(sys, "argv", [name, "--version"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"◉ prometheus {package_version()} · OAra Labs\n"
    assert captured.err == ""


def test_dash_dash_version_is_listed_in_the_help(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--help")
    assert "--version" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, quiet",
    [(["setup"], False), (["setup", "--quiet"], True), (["setup", "-q"], True),
     (["--quiet", "setup"], True), (["-q", "setup"], True)],
)
def test_setup_takes_quiet_before_or_after_the_command(monkeypatch, argv, quiet):
    seen = {}

    def fake_run_setup(args):
        seen["quiet"] = args.quiet
        return 0

    monkeypatch.setattr("prometheus.cli.setup.run_setup", fake_run_setup)
    with pytest.raises(SystemExit) as exit_:
        _main(monkeypatch, *argv)
    assert exit_.value.code == 0
    assert seen["quiet"] is quiet


@pytest.mark.parametrize(
    "argv, quiet",
    [(["daemon"], False), (["daemon", "--quiet"], True), (["daemon", "-q"], True), (["--quiet", "daemon"], True)],
)
def test_daemon_is_handed_quiet_whichever_side_it_was_given_on(monkeypatch, argv, quiet):
    seen = {}

    def fake_daemon_main():
        seen["argv"] = list(sys.argv)

    monkeypatch.setattr("prometheus.daemon.main", fake_daemon_main)
    _main(monkeypatch, *argv)
    assert ("--quiet" in seen["argv"]) is quiet
    assert seen["argv"][0] == "prometheus-daemon"


def test_run_setup_prints_the_banner_on_a_terminal(monkeypatch):
    from prometheus.cli.setup import run_setup

    class Wizard:
        def __init__(self, **kwargs):
            pass

        def run(self):
            return True

    out = Tty()
    monkeypatch.setattr("prometheus.setup_wizard.SetupWizard", Wizard)
    monkeypatch.setattr(sys, "stdout", out)
    assert run_setup(argparse.Namespace(gateway_only=False)) == 0
    text = ANSI.sub("", out.getvalue())
    assert text.startswith(FULL.splitlines()[0]) and "a fire you own doesn't go out" in text
    assert "daemon" not in text and "model" not in text  # setup knows neither


@pytest.mark.parametrize("namespace", [argparse.Namespace(gateway_only=False, quiet=True), argparse.Namespace(gateway_only=False)])
def test_run_setup_is_silent_with_quiet_or_off_a_terminal(monkeypatch, namespace):
    from prometheus.cli.setup import run_setup

    class Wizard:
        def __init__(self, **kwargs):
            pass

        def run(self):
            return True

    monkeypatch.setattr("prometheus.setup_wizard.SetupWizard", Wizard)
    out = Tty() if namespace.__dict__.get("quiet") else io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert run_setup(namespace) == 0
    assert out.getvalue() == ""
