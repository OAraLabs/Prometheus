"""packaging/macos/verify_app.py — the checks a built Prometheus.app must pass before anyone is asked to trust it.

Beacon downloads the app with Node, so the file carries no quarantine flag and Gatekeeper never assesses
it: Beacon's own checks are the only gate, and these are the same checks run at build time. The command
outputs below are REAL, captured from a Developer-ID-signed bundle and from codesign/stapler/syspolicy_check
on this Mac (paths shortened), not invented, because a parser tested against an imagined format passes.

Nothing here runs codesign or touches a Mac: commands go through an injected runner.
"""

from __future__ import annotations

import importlib.util
import plistlib
import subprocess
from pathlib import Path

import pytest

VERIFY_APP = Path(__file__).resolve().parents[1] / "packaging" / "macos" / "verify_app.py"

# `codesign -dvv` of a hardened, timestamped, Developer ID signed bundle.
GOOD = """\
Executable=<path>/Prometheus.app/Contents/MacOS/Prometheus
Identifier=com.oaralabs.prometheus
Format=app bundle with Mach-O thin (arm64)
CodeDirectory v=20500 size=1259 flags=0x10000(runtime) hashes=32+3 location=embedded
Signature size=9048
Authority=Developer ID Application: William Hieber (53JM8W47RL)
Authority=Developer ID Certification Authority
Authority=Apple Root CA
Timestamp=Oct 7, 2026 at 11:27:36 PM
Info.plist entries=10
TeamIdentifier=53JM8W47RL
Runtime Version=26.2.0
Sealed Resources version=2 rules=13 files=2
Internal requirements count=1 size=192
"""

ADHOC = """\
Executable=<path>/Prometheus
Identifier=Prometheus
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=500 flags=0x20002(adhoc,linker-signed) hashes=9+0 location=embedded
Signature=adhoc
Info.plist=not bound
TeamIdentifier=not set
Sealed Resources=none
Internal requirements=none
"""


@pytest.fixture(scope="module")
def verify():
    assert VERIFY_APP.is_file(), f"{VERIFY_APP} does not exist"
    spec = importlib.util.spec_from_file_location("macos_verify_app", VERIFY_APP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _facts(verify, text):
    return verify.parse_codesign_display(text)


# ── reading codesign ─────────────────────────────────────────────────────────

def test_parses_a_real_developer_id_signature(verify):
    f = _facts(verify, GOOD)
    assert f["identifier"] == "com.oaralabs.prometheus"
    assert f["team_id"] == "53JM8W47RL"
    assert f["authorities"][0] == "Developer ID Application: William Hieber (53JM8W47RL)"
    assert "runtime" in f["flags"]
    assert f["timestamp"] is True


def test_an_adhoc_signature_has_no_team_no_authority_and_no_timestamp(verify):
    f = _facts(verify, ADHOC)
    assert f["team_id"] is None
    assert f["authorities"] == []
    assert f["timestamp"] is False
    assert "runtime" not in f["flags"]


# ── what counts as a problem ─────────────────────────────────────────────────

def test_a_good_signature_has_no_problems(verify):
    assert verify.signature_problems(_facts(verify, GOOD), team_id="53JM8W47RL",
                                     identifier="com.oaralabs.prometheus") == []


def test_adhoc_is_refused_for_every_reason_notarization_would(verify):
    problems = " | ".join(verify.signature_problems(_facts(verify, ADHOC), team_id="53JM8W47RL"))
    assert "Developer ID Application" in problems
    assert "hardened" in problems
    assert "timestamp" in problems
    assert "team" in problems


def test_each_missing_property_is_named_on_its_own(verify):
    no_ts = GOOD.replace("Timestamp=Oct 7, 2026 at 11:27:36 PM\n", "")
    assert any("timestamp" in p for p in verify.signature_problems(_facts(verify, no_ts), team_id="53JM8W47RL"))
    no_rt = GOOD.replace("flags=0x10000(runtime)", "flags=0x0(none)")
    assert any("hardened" in p for p in verify.signature_problems(_facts(verify, no_rt), team_id="53JM8W47RL"))
    other_team = GOOD.replace("TeamIdentifier=53JM8W47RL", "TeamIdentifier=ABCDE12345")
    assert any("ABCDE12345" in p for p in verify.signature_problems(_facts(verify, other_team), team_id="53JM8W47RL"))
    wrong_id = GOOD.replace("Identifier=com.oaralabs.prometheus", "Identifier=com.example.other")
    assert any("com.example.other" in p for p in verify.signature_problems(
        _facts(verify, wrong_id), team_id="53JM8W47RL", identifier="com.oaralabs.prometheus"))


def test_an_apple_development_certificate_is_not_a_developer_id(verify):
    dev = GOOD.replace("Developer ID Application: William Hieber (53JM8W47RL)",
                       "Apple Development: someone@example.com (NVBFBBJ793)")
    assert any("Developer ID Application" in p for p in verify.signature_problems(
        _facts(verify, dev), team_id="53JM8W47RL"))


# ── entitlements start at none ───────────────────────────────────────────────

def test_no_entitlements_means_the_dump_is_only_the_executable_line(verify):
    assert verify.entitlement_problems("Executable=<path>/Prometheus\n") == []


def test_an_entitlement_is_named_and_an_allowed_one_is_not(verify):
    dump = (
        "Executable=<path>/python3.12\n"
        '<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>'
        "<key>com.apple.security.cs.allow-jit</key><true/>"
        "<key>com.apple.security.cs.disable-library-validation</key><true/></dict></plist>\n"
    )
    problems = verify.entitlement_problems(dump)
    assert any("allow-jit" in p for p in problems) and any("disable-library-validation" in p for p in problems)
    assert verify.entitlement_problems(
        dump, allowed={"com.apple.security.cs.allow-jit", "com.apple.security.cs.disable-library-validation"}) == []


# ── the bundle's own metadata ────────────────────────────────────────────────

INFO = {
    "CFBundleIdentifier": "com.oaralabs.prometheus", "CFBundleExecutable": "Prometheus",
    "CFBundleShortVersionString": "0.9.7", "LSMinimumSystemVersion": "13.0",
    "PrometheusAgentLabel": "com.oaralabs.prometheus.daemon",
    "PrometheusAgentPlist": "com.oaralabs.prometheus.daemon.plist",
}
AGENT = {
    "Label": "com.oaralabs.prometheus.daemon", "BundleProgram": "Contents/MacOS/Prometheus",
    "ProgramArguments": ["Prometheus", "--run"], "AssociatedBundleIdentifiers": ["com.oaralabs.prometheus"],
    "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 10,
}


def test_consistent_info_and_agent_plists_pass(verify):
    assert verify.metadata_problems(INFO, AGENT, bundle_id="com.oaralabs.prometheus") == []


def test_an_agent_that_does_not_run_the_launcher_is_a_problem(verify):
    wrong = {**AGENT, "BundleProgram": "Contents/Resources/python/bin/python3.12"}
    assert any("launcher" in p for p in verify.metadata_problems(INFO, wrong, bundle_id="com.oaralabs.prometheus"))


def test_the_agent_label_must_match_what_the_launcher_will_register(verify):
    wrong = {**AGENT, "Label": "com.oaralabs.prometheus.other"}
    assert any("Label" in p for p in verify.metadata_problems(INFO, wrong, bundle_id="com.oaralabs.prometheus"))


def test_a_user_path_in_a_bundled_plist_is_a_problem(verify):
    wrong = {**AGENT, "StandardErrorPath": "/Users/someone/Library/Logs/x.log"}
    assert any("StandardErrorPath" in p for p in verify.metadata_problems(INFO, wrong, bundle_id="com.oaralabs.prometheus"))


def test_the_wrong_bundle_id_or_too_low_a_minimum_macos_is_a_problem(verify):
    assert any("bundle id" in p for p in verify.metadata_problems(
        {**INFO, "CFBundleIdentifier": "com.example.x"}, AGENT, bundle_id="com.oaralabs.prometheus"))
    assert any("13.0" in p for p in verify.metadata_problems(
        {**INFO, "LSMinimumSystemVersion": "11.0"}, AGENT, bundle_id="com.oaralabs.prometheus"))


# ── the zip Beacon downloads ─────────────────────────────────────────────────

def test_a_zip_holds_exactly_one_app_and_nothing_else(verify):
    ok = ["Prometheus.app/", "Prometheus.app/Contents/", "Prometheus.app/Contents/Info.plist"]
    assert verify.zip_problems(ok) == []
    assert any("one" in p for p in verify.zip_problems(ok + ["Other.app/Contents/Info.plist"]))
    assert any("__MACOSX" in p for p in verify.zip_problems(ok + ["__MACOSX/._Prometheus.app"]))
    assert any("outside" in p for p in verify.zip_problems(ok + ["README.txt"]))


# ── notarization, from real exit codes ───────────────────────────────────────

class _Runner:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        for prefix, (code, out, err) in self.table.items():
            if argv[: len(prefix)] == list(prefix):
                return subprocess.CompletedProcess(argv, code, out, err)
        return subprocess.CompletedProcess(argv, 0, "", "")


APP = Path("/x/Prometheus.app")
ACCEPTED = {
    ("xcrun", "stapler", "validate"): (0, "The validate action worked!\n", ""),
    ("syspolicy_check", "distribution"): (0, "", ""),
    ("spctl", "--assess"): (0, "", f"{APP}: accepted\nsource=Notarized Developer ID\n"),
}


def test_a_notarized_app_passes_all_three_gatekeeper_questions(verify):
    runner = _Runner(ACCEPTED)
    assert verify.notarization_problems(APP, runner=runner) == []
    asked = [c[:3] for c in runner.calls]
    assert ["xcrun", "stapler", "validate"] in asked
    assert ["syspolicy_check", "distribution", str(APP)] in asked
    assert any(c[0] == "spctl" and c[1] == "--assess" for c in runner.calls)


def test_a_missing_ticket_is_reported_from_the_real_exit_codes(verify):
    runner = _Runner({
        ("xcrun", "stapler", "validate"): (65, "does not have a ticket stapled to it.\n", ""),
        ("syspolicy_check", "distribution"): (70, "Notary Ticket Missing\n", ""),
        ("spctl", "--assess"): (3, "", f"{APP}: rejected\nsource=Unnotarized Developer ID\n"),
    })
    problems = " | ".join(verify.notarization_problems(APP, runner=runner))
    assert "stapler" in problems and "syspolicy_check" in problems and "Notarized Developer ID" in problems


def test_per_executable_spctl_is_never_the_check(verify):
    """Beacon found `spctl -t exec` on nested executables gives wrong answers on a notarized release."""
    runner = _Runner(ACCEPTED)
    verify.notarization_problems(APP, runner=runner)
    for call in runner.calls:
        if call[0] == "spctl":
            assert call[-1] == str(APP), "spctl is asked about the bundle only"


# ── plist helper used by the CLI ─────────────────────────────────────────────

def test_plists_round_trip_through_the_reader(verify, tmp_path):
    p = tmp_path / "x.plist"
    p.write_bytes(plistlib.dumps(AGENT))
    assert verify.read_plist(p) == AGENT


# ── the icon ─────────────────────────────────────────────────────────────────

def test_a_bundle_whose_info_plist_names_an_icon_must_contain_it(verify, tmp_path):
    (tmp_path / "Contents" / "Resources").mkdir(parents=True)
    info = {"CFBundleIconFile": "Prometheus"}
    assert any("Prometheus.icns" in p for p in verify.icon_problems(tmp_path, info))
    (tmp_path / "Contents" / "Resources" / "Prometheus.icns").write_bytes(b"icns" + b"\x00" * 32)
    assert verify.icon_problems(tmp_path, info) == []


def test_an_info_plist_with_no_icon_is_a_problem(verify, tmp_path):
    assert any("CFBundleIconFile" in p for p in verify.icon_problems(tmp_path, {}))


def test_a_file_that_is_not_an_icns_is_a_problem(verify, tmp_path):
    (tmp_path / "Contents" / "Resources").mkdir(parents=True)
    (tmp_path / "Contents" / "Resources" / "Prometheus.icns").write_bytes(b"not an icon at all")
    assert any("icns" in p for p in verify.icon_problems(tmp_path, {"CFBundleIconFile": "Prometheus"}))


# ── zip-slip: nothing lands outside the directory the zip is unpacked into ───

def test_an_entry_that_is_absolute_or_climbs_out_is_refused(verify):
    ok = [("Prometheus.app/Contents/Info.plist", None),
          ("Prometheus.app/Contents/Resources/python/bin/python3", "python3.12")]
    assert verify.entry_problems(ok) == []
    for bad in ("../evil", "Prometheus.app/../../evil", "/etc/evil", "Prometheus.app/Contents/..\\..\\evil",
                "C:/evil", "\\evil"):
        assert verify.entry_problems([(bad, None)]), bad


def test_a_symlink_that_points_outside_the_archive_is_refused(verify):
    for target in ("/etc", "../../..", "../../../outside", "../Contents/../../.."):
        assert verify.entry_problems([("Prometheus.app/Contents/link", target)]), target
    assert verify.entry_problems([("Prometheus.app/Contents/Resources/link", "../MacOS/Prometheus")]) == []


def _zip(path, entries):
    import stat
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        for name, link in entries:
            if link is None:
                archive.writestr(name, "x")
            else:
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, link)
    return path


def test_a_slipping_zip_is_refused_before_anything_is_extracted(verify, tmp_path, monkeypatch):
    extracted = []
    monkeypatch.setattr(verify.subprocess, "run", lambda argv, **kw: extracted.append(argv))
    app = "Prometheus.app/Contents/Info.plist"
    for name, entries in {
        "climb": [(app, None), ("../evil", None)],
        "absolute": [(app, None), ("/tmp/evil", None)],
        "link": [(app, None), ("Prometheus.app/Contents/escape", "/etc")],
    }.items():
        problems = verify.verify_zip(_zip(tmp_path / f"{name}.zip", entries))
        assert any("outside" in p or "absolute" in p for p in problems), (name, problems)
    assert extracted == [], "ditto ran on a zip that was refused"


# ── the signature is OURS, not merely valid ──────────────────────────────────

TEAM_REQUIREMENT = '-R=anchor apple generic and certificate leaf[subject.OU] = "53JM8W47RL"'


def test_the_bundle_must_satisfy_a_requirement_that_names_our_team(verify):
    runner = _Runner({})
    assert verify.codesign_problems(APP, team_id="53JM8W47RL", runner=runner) == []
    assert ["codesign", "--verify", "--deep", "--strict", TEAM_REQUIREMENT, str(APP)] in runner.calls


def test_a_valid_signature_by_another_team_fails_the_requirement(verify):
    # Real: `codesign --verify --strict -R=<this requirement> /bin/ls` on this Mac (Apple-signed, not our team).
    runner = _Runner({("codesign", "--verify", "--deep", "--strict", TEAM_REQUIREMENT):
                      (3, "", "test-requirement: code failed to satisfy specified code requirement(s)\n")})
    problems = verify.codesign_problems(APP, team_id="53JM8W47RL", runner=runner)
    assert len(problems) == 1 and "53JM8W47RL" in problems[0] and "requirement" in problems[0], problems
