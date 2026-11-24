"""The approval prompt shows the real path it asks about.

(Key-shaped strings below are split into adjacent literals, so the source never holds one: the pre-commit scan is right
to refuse them.)

The prompt lists a call's arguments so the owner can see what is being approved. Two things in how a file path was
rendered meant the owner could be shown a target that was not the real one:

* The path went through the secret redactor the audit log uses, and one of its rules had no left boundary, so any name
  containing ``sk-``, ``pk-``, ``ghp_`` or ``gho_`` was masked in the middle: ``ask-test.txt`` was shown as ``a***.txt``.
  The audit log stored the same mangled text.
* A value longer than 160 characters was cut from the END, so a long path lost its filename. In ``strict`` and
  ``cautious`` mode the arguments block is the only place the path is shown.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue
from prometheus.permissions.argument_view import MAX_VALUE_CHARS, format_arguments, redact_arguments
from prometheus.permissions.audit import AuditDecision, AuditLogger
from prometheus.permissions.checker import SecurityGate

#: Ordinary names that contain a secret-looking prefix in the middle of a word.
NAMES = ["ask-test.txt", "task-list.md", "disk-usage.log", "risk-notes.txt", "desk-plan.md", "flask-app.py",
         "mask-it.txt", "sepk-data.csv", "oghp_notes.md"]


# ── the redactor ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", NAMES)
def test_a_name_that_merely_contains_a_key_prefix_is_not_masked(name):
    path = f"/home/capstone/workspace/{name}"
    assert AuditLogger.redact(path) == path
    assert AuditLogger.redact(f"write {name} now") == f"write {name} now"


@pytest.mark.parametrize("secret", [
    "sk" "-live_abc123", "sk-proj-abcdef0123456789", "ghp_abcdefghijklmnop1234", "xoxb-1234-abcdef", "pk-test_0123456789",
])
def test_a_real_key_is_still_masked_wherever_a_boundary_precedes_it(secret):
    for text in (f"token {secret}", f"export K={secret}", f"curl -u {secret}", f'"{secret}"', secret):
        out = AuditLogger.redact(text)
        assert secret not in out, f"{text!r} leaked {secret!r}: {out!r}"


# ── the arguments the prompt shows ───────────────────────────────────────────

@pytest.mark.parametrize("name", NAMES)
def test_a_path_argument_is_shown_exactly(name):
    path = f"/home/capstone/workspace/{name}"
    assert redact_arguments({"path": path}) == {"path": path}


@pytest.mark.parametrize("key", ["path", "file_path", "cwd", "directory", "dest", "destination", "output_path"])
def test_every_path_valued_argument_is_shown_exactly(key):
    path = "/home/capstone/workspace/ask-test.txt"
    assert redact_arguments({key: path}) == {key: path}


def test_credentials_in_a_url_shaped_path_are_still_masked():
    out = redact_arguments({"path": "https://user:hunter2pass@host.example/dir/file.txt"})
    assert out is not None and "hunter2pass" not in out["path"] and "***" in out["path"]


def test_other_arguments_are_still_redacted():
    out = redact_arguments({
        "content": "api_key=sk" "-live_abc123 and more",
        "command": "curl -H 'Authorization: Bearer abc123xyz456' https://x.example",
        "text": "sk" "-live_abcdef0123456789",
    })
    assert out is not None
    assert "sk" "-live_abc123" not in out["content"]
    assert "abc123xyz456" not in out["command"]
    assert "sk" "-live_abcdef0123456789" not in out["text"]


def test_a_word_like_ask_in_ordinary_text_is_left_alone():
    assert redact_arguments({"text": "ask-test and task-list"}) == {"text": "ask-test and task-list"}


# ── the boundary must not unmask a key that follows an ESCAPED separator ─────
# json.dumps and repr write a newline as the two characters `\n`, and that `n` is a letter. A key at the start of a line
# (a heredoc, a key list) then had no boundary in front of it and was stored in the clear.

KEY = "sk" "-ant-api03-ABCDEFGH12345678"


@pytest.mark.parametrize("before", ["\n", "\t", "\r", "\x0b", "\u2028", "%20", "%0A", "\x1b[32m", "\x1b[1;32m"],
                         ids=["LF", "TAB", "CR", "VT", "LS", "pct20", "pct0A", "ansi", "ansi-bold"])
def test_a_key_after_an_escaped_separator_is_still_masked_in_every_place_it_is_stored(before, tmp_path):
    command = f"cat <<EOF > k{before}{KEY}{before}EOF"
    log = AuditLogger(tmp_path)
    log.log("bash", AuditDecision.CONFIRM_PENDING, 1, f"Command requires approval: {command!r}",
            tool_input={"command": command})
    row = log.query_recent(limit=1)[0]
    assert KEY not in row.tool_input_summary, "json.dumps wrote the separator as backslash-n and the key went unmasked"
    assert KEY not in row.reason, "repr did the same in the reason"
    assert KEY not in AuditLogger.redact(repr(command))
    shown = redact_arguments({"edits": [{"new": f"a{before}{KEY}"}]})
    assert shown is not None and KEY not in shown["edits"]


def test_a_name_that_ends_in_a_letter_before_the_prefix_is_still_left_alone():
    # The escapes above are recognised by shape, so an ordinary word is not mistaken for one.
    for text in ("ask-test", "task-list", "knsk-x", "xtab-notes", "ptn-notes", "ox%zsk-x"):
        assert AuditLogger.redact(text) == text, text


# ── a path-NAMED argument that is not a filesystem path ──────────────────────

@pytest.mark.parametrize(("key", "value"), [
    ("destination", "https://" + "ghp" "_AbCdEfGhIjKlMnOpQrStUvWx" + "@github.com/o/r.git"),   # a token as the userinfo
    ("path", "https://x.example/cb#access_" + "token=abcdefghijklmnop123456"),                   # a fragment
    ("file", "https://hooks.example/services/" + KEY),                                           # a path segment
    ("path", KEY),                                                                               # a bare key
    ("dest", "pass" "word: hunter2hunter2"),                                                     # a labelled secret
], ids=["pat-userinfo", "fragment", "segment", "bare-key", "labelled"])
def test_a_path_named_argument_that_carries_a_credential_is_still_masked(key, value):
    secret = {"destination": "ghp" "_AbCdEfGhIjKlMnOpQrStUvWx", "path": None, "file": KEY, "dest": "hunter2hunter2"}[key]
    shown = redact_arguments({key: value})[key]
    for needle in (secret, "abcdefghijklmnop123456" if "access_" in value else secret, KEY if value == KEY else secret):
        if needle:
            assert needle not in shown, f"{key}={value!r} was shown as {shown!r}"


def test_a_name_that_starts_like_a_key_is_shown_and_a_real_key_is_masked():
    # Decided 2026-11-13: a key prefix masks only 16+ characters after it (a real key is 20+), so `sk-notes.txt`, a host called
    # `pk-evil` and `./sk-payload` are shown, and a key is masked. (Earlier a short name was masked: the safe side for a secret,
    # the unsafe one for a prompt: `scp x sk-evil:/tmp` hid the destination.)
    for name in ("sk" "-notes.txt", "pk" "-data.csv", "ghp" "_log.md"):
        assert redact_arguments({"path": f"/home/u/workspace/{name}"})["path"] == f"/home/u/workspace/{name}"
    key = "sk" "-live_abcdefghijklmnop12"
    assert key not in redact_arguments({"path": f"/home/u/workspace/{key}"})["path"]


# ── redact the WHOLE value, then cut: a middle cut must not remove the label or prefix that made a secret recognisable ──
# The tail kept by the middle cut carries no label, so redacting after the cut left the end of a secret in the clear.

def test_a_token_in_a_long_url_is_masked_before_the_middle_cut_removes_its_label():
    token = "A1b2C3d4E5" * 17
    url = "https://files.example.com/" + "p" * 80 + "/dl?access_" + "token=" + token
    shown = redact_arguments({"destination": url})["destination"]
    assert token[-40:] not in shown and token[:40] not in shown, shown


def test_a_key_split_by_the_middle_cut_is_not_shown_in_part():
    key = "ghp" "_" + "AbCdEf1234" * 10
    for padding in (0, 7, 31, 50, 52, 53, 54, 120):
        url = "https://h.example/" + "a" * (300 + padding) + "/" + key + "/f.txt"
        shown = redact_arguments({"path": url})["path"]
        assert "AbCdEf1234AbCdEf1234" not in shown, (padding, shown)


def test_a_long_value_that_is_a_labelled_secret_is_masked_before_the_cut():
    # The value BEGINS with the label, so it is not a path: it is masked whole, then cut. (A label in the middle of a
    # plain path is part of the path and is shown, see test_a_plain_path_is_shown_exactly_whatever_it_contains.)
    value = "pass" "word: " + "S3cretValue" * 20
    shown = redact_arguments({"path": value})["path"]
    assert "S3cretValueS3cretValue" not in shown, shown


def test_a_plain_long_path_is_still_cut_in_the_middle_and_keeps_its_filename():
    shown = redact_arguments({"path": "/home/capstone/workspace/" + "d" * 300 + "/final/report-2026.txt"})["path"]
    assert shown.endswith("/final/report-2026.txt") and "more chars" in shown


# ── more ways to write a separator before a key ──────────────────────────────

# Entries with a single backslash in the source are the REAL control character; entries with a doubled backslash are the
# escaped TEXT that repr and json.dumps write for it (backslash, then the letters), which is what the audit log stores.
@pytest.mark.parametrize("before", [
    "\x1b[01;31m", "\x1b[38;5;82m", "\x1b[m", "\x1b[2K", "\x1b[1;32;40m", "\x1b[32m",
    "\\x1b[01;31m", "\\x1b[38;5;82m", "\\x1b[m", "\\x1b[2K", "\\x1b[1;32;40m", "\\033[32m", "\\u001b[32m",
    "\\U0001F600", "\\U0001f4a9", "\\u00e9",
], ids=lambda c: c.encode("unicode_escape").decode().replace("\\", "bs-"))
def test_a_key_after_a_longer_ansi_code_or_a_wide_escape_is_still_masked(before):
    text = f"x{before}{KEY}y"
    assert KEY not in AuditLogger.redact(text)
    assert KEY not in AuditLogger.redact(repr(text))
    assert KEY not in AuditLogger.redact(__import__("json").dumps(text))


# ── ordinary paths are shown exactly; a value that carries a secret is masked whatever the argument is called ────────────

@pytest.mark.parametrize("path", [
    "/data/year=2024/month=11/day=07/file.csv",
    "/home/capstone/workspace/ask-test.txt",
    "relative/dir/task-list.md",
    "[ask-test.txt]",
    "/tmp/a b/c d.txt",
    "~/projects/risk-notes/desk-plan.md",
], ids=["hive", "absolute", "relative", "bracketed", "spaces", "tilde"])
def test_an_ordinary_path_is_shown_exactly(path):
    assert redact_arguments({"path": path}) == {"path": path}


@pytest.mark.parametrize("key", ["path", "file", "dest", "destination", "directory"])
@pytest.mark.parametrize("carrier", [
    "Authorization: Bearer ABCDEF0123456789XYZWxyz",
    "x-api-key: ABCDEF0123456789XYZW",
    "X-API-KEY=ABCDEF0123456789XYZW",
    '{"pass' 'word":"hunter2hunter2"}',
    '{"access_' 'token":"ABCDEF0123456789XYZW"}',
    "/tmp/x pass" "word=hunter2hunter2",
    "cfg to" "ken: ABCDEF0123456789XYZW",
    "host.example/dl?access_" "token=ABCDEF0123456789XYZW",
    "https://h/dl?sig=ABCDEF0123456789XYZW&x=1",
    '"sk' '-ABCDEF0123456789XYZW"',
    "file sk" "-ABCDEF0123456789XYZW",
    "sk" "-ABCDEF0123456789XYZW /tmp/x",
], ids=["bearer", "x-api-key", "X-API-KEY", "json-password", "json-token", "mid-password", "mid-token", "query", "sig", "quoted-key",
        "word-key", "key-then-path"])
def test_a_path_named_argument_holding_a_secret_carrier_is_masked_like_any_other_argument(key, carrier):
    secret = next(x for x in ("hunter2hunter2", "ABCDEF0123456789XYZW") if x in carrier)
    assert secret not in redact_arguments({key: carrier})[key], carrier


# ── a long value must not stall the daemon (the model chooses this argument) ─────────────────────────────────────────

def test_a_huge_path_named_value_is_handled_in_bounded_time():
    import time

    for value in ("a" * 300_000, "https://h.example/" + "a" * 300_000, "A1" * 150_000, ("x/" * 150_000)):
        start = time.monotonic()
        shown = redact_arguments({"path": value})["path"]
        assert time.monotonic() - start < 2.0, f"{len(value)} chars took {time.monotonic() - start:.1f}s"
        assert len(shown) < 400


def test_the_audit_redactor_is_linear_on_a_long_alphanumeric_run():
    import time

    start = time.monotonic()
    AuditLogger.redact("a" * 300_000)
    assert time.monotonic() - start < 2.0


# ── the remaining ways to write an ANSI sequence in front of a key ───────────────────────────────────────────────────

@pytest.mark.parametrize("before", ["\\e[32m", "\\x1B[32m", "\\u001B[32m", "\\x1b[?25l", "[32m", "[m", "\\033[1;31;40m"],
                         ids=lambda c: c.replace("\\", "bs-"))
def test_a_key_after_the_other_ansi_spellings_is_still_masked(before):
    text = f"x{before}{KEY}y"
    assert KEY not in AuditLogger.redact(text)
    assert KEY not in AuditLogger.redact(repr(text))


# ── URL credentials are masked wherever the scheme starts, and the cut count is true for a huge value ───────────────────
# A left boundary on the scheme pattern made it linear but stopped it matching after a digit, `.`, `-`, `+`, `%XX` or a
# colour code, so those lost the masking they had on main. The scheme is bounded in length instead.

@pytest.mark.parametrize("before", ["1", "9", ".", "-", "+", "%0A", "%20", "x1", ".git+", "\x1b[32m", "\x1b[0m", " ", "\n", '"', "="],
                         ids=lambda c: c.encode("unicode_escape").decode().replace("\\", "bs-"))
def test_url_credentials_are_masked_after_any_left_context(before):
    text = f"{before}https://deploy:hunter2hunter2@git.example/o/r.git"
    assert "hunter2hunter2" not in AuditLogger.redact(text), text
    assert "hunter2hunter2" not in redact_arguments({"path": text})["path"]


def test_the_count_of_hidden_characters_is_true_for_a_value_past_the_redaction_bound():
    path = "/a/" + "d" * 100_000 + "/f.txt"
    shown = redact_arguments({"path": path})["path"]
    assert f"(+{len(path) - MAX_VALUE_CHARS} more chars)" in shown, shown
    assert shown.endswith("/f.txt")


# ── a limit too small for the middle cut ─────────────────────────────────────

@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_a_tiny_limit_never_repeats_the_whole_path(limit):
    path = "/home/u/" + "d" * 80 + "/file.txt"
    shown = redact_arguments({"path": path}, max_value=limit)["path"]
    # A text that announces it was cut must be shorter than the original. With limit 0 the tail slice `text[-0:]` was the
    # WHOLE path, appended after the "+N more chars" marker.
    assert len(shown) < len(path), shown


# ── a long path ──────────────────────────────────────────────────────────────

def _long_path():
    return "/home/capstone/workspace/" + "d" * 300 + "/final/dir/report-2026.txt"


def test_a_long_path_keeps_its_filename_and_its_start():
    shown = redact_arguments({"path": _long_path()})["path"]
    assert shown.startswith("/home/capstone/workspace/")
    assert shown.endswith("/final/dir/report-2026.txt"), shown
    assert "more chars" in shown, "the cut must still be announced"
    assert len(shown) <= MAX_VALUE_CHARS + 60


def test_a_path_whose_filename_alone_is_too_long_still_shows_its_end():
    path = "/tmp/" + "x" * 400 + "-the-end.txt"
    assert redact_arguments({"path": path})["path"].endswith("-the-end.txt")


def test_a_long_value_that_is_not_a_path_is_still_cut_from_the_end():
    shown = redact_arguments({"content": "y" * 1000})["content"]
    assert shown.startswith("y" * MAX_VALUE_CHARS) and "more chars" in shown


# ── the prompt, end to end ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_pending_approval_carries_the_exact_path():
    path = "/home/capstone/workspace/ask-test.txt"
    queue = ApprovalQueue(security_gate=SecurityGate(), timeout_seconds=5)
    task = asyncio.create_task(queue.request_approval("write_file", "writes outside the workspace",
                                                      arguments={"path": path, "content": "hello"}))
    for _ in range(100):
        if queue.pending:
            break
        await asyncio.sleep(0.01)
    assert queue.pending, "request_approval did not queue"
    queue._test_task = task
    action = next(iter(queue.pending.values()))
    assert action.arguments["path"] == path
    assert f"  path: {path}" in format_arguments(action.arguments)
    await queue.deny(action.request_id, by=_BY)


# ── the audit log ────────────────────────────────────────────────────────────

def test_the_audit_log_stores_the_real_name(tmp_path):
    from prometheus.permissions.audit import AuditLogger as Logger

    log = Logger(tmp_path)
    path = "/home/capstone/workspace/ask-test.txt"
    log.log("write_file", AuditDecision.CONFIRM_PENDING, 1, f"write_file targets path outside workspace: {path}",
            tool_input={"path": path})
    row = log.query_recent(limit=1)[0]
    assert "ask-test.txt" in row.tool_input_summary
    assert "ask-test.txt" in row.reason


from prometheus.permissions.approver import in_process  # noqa: E402

_BY = in_process("test")
