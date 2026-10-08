"""packaging/macos/build_app.py — the parts of the Prometheus.app build that can be tested without a Mac.

The build itself (download a Python, install the pinned wheels, compile the launcher, sign, notarize) runs
on a Mac with a signing identity and is verified by running it. What is pinned here is everything that
decides WHAT gets built, and that has already gone wrong somewhere else: which files get signed, by what
(a fingerprint, never an ambiguous name), what the bundle's static plists may contain, what a requirements
filter removes, whether a dirty tree can ship, and whether a downloaded runtime is checked.

Nothing here runs swiftc, codesign, uv or the network: every external command goes through an injected
runner, and the tests assert on the argv it received.
"""

from __future__ import annotations

import importlib.util
import plistlib
import subprocess
from pathlib import Path

import pytest

BUILD_APP = Path(__file__).resolve().parents[1] / "packaging" / "macos" / "build_app.py"
FINGERPRINT = "55059EA496841EEAF7B8CA6824622D2BAE913C24"

MACHO_ARM64 = bytes.fromhex("cffaedfe") + b"\x0c\x00\x00\x01" + b"\x00" * 64
FAT = bytes.fromhex("cafebabe") + b"\x00\x00\x00\x02" + b"\x00" * 64


@pytest.fixture(scope="module")
def app():
    assert BUILD_APP.is_file(), f"{BUILD_APP} does not exist"
    spec = importlib.util.spec_from_file_location("macos_build_app", BUILD_APP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Runner:
    """Records argv; answers from a script of (prefix, returncode, stdout)."""

    def __init__(self, answers=()):
        self.calls: list[list[str]] = []
        self.answers = list(answers)

    def __call__(self, argv, **kwargs):
        self.calls.append([str(a) for a in argv])
        for prefix, code, out in self.answers:
            if self.calls[-1][: len(prefix)] == list(prefix):
                return subprocess.CompletedProcess(argv, code, out, "")
        return subprocess.CompletedProcess(argv, 0, "", "")


# ── identity and version ─────────────────────────────────────────────────────

def test_version_is_read_from_the_package_not_typed_here(app):
    init = (Path(__file__).resolve().parents[1] / "src" / "prometheus" / "__init__.py").read_text()
    assert f'"{app.read_version()}"' in init


def test_bundle_identity_is_the_contract_with_beacon(app):
    assert app.BUNDLE_ID == "com.oaralabs.prometheus"
    assert app.TEAM_ID == "53JM8W47RL"
    assert app.MIN_MACOS == "13.0"


# ── Info.plist ───────────────────────────────────────────────────────────────

def test_info_plist_names_the_agent_the_launcher_registers(app):
    from prometheus.cli import launchd

    info = app.info_plist("1.2.3")
    assert info["CFBundleIdentifier"] == app.BUNDLE_ID
    assert info["CFBundleExecutable"] == "Prometheus"
    assert info["CFBundleShortVersionString"] == "1.2.3"
    assert info["LSMinimumSystemVersion"] == "13.0"
    # No Dock icon, no menu bar: Prometheus is a background service with a small window.
    assert info["LSUIElement"] is True
    assert info["PrometheusAgentLabel"] == launchd.LABEL
    assert info["PrometheusAgentPlist"] == f"{launchd.LABEL}.plist"
    # The launcher probes http://127.0.0.1 itself; say so rather than rely on ATS exempting IP literals.
    assert info["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}


# ── the bundled LaunchAgent plist ────────────────────────────────────────────

def test_agent_plist_starts_the_launcher_not_python(app):
    from prometheus.cli import launchd

    plist = plistlib.loads(app.agent_plist())
    assert plist["Label"] == launchd.LABEL
    # The agent's name in Login Items comes from the PROGRAM file name, so the program must be the
    # launcher ("Prometheus"), never python3.12.
    assert plist["BundleProgram"] == "Contents/MacOS/Prometheus"
    assert plist["ProgramArguments"] == ["Prometheus", "--run"]
    assert plist["AssociatedBundleIdentifiers"] == [app.BUNDLE_ID]


def test_agent_plist_is_static_so_it_carries_no_user_paths(app):
    text = app.agent_plist().decode()
    for needle in ("/Users/", "~", "$HOME", "/home/"):
        assert needle not in text, f"a bundled plist cannot know the user's home: found {needle!r}"
    plist = plistlib.loads(app.agent_plist())
    for key in ("StandardOutPath", "StandardErrorPath", "WorkingDirectory"):
        assert key not in plist, f"{key} is set by the launcher at start, not by the static plist"


def test_agent_plist_shares_its_restart_keys_with_install_service(app):
    from prometheus.cli import launchd

    bundled = plistlib.loads(app.agent_plist())
    cli = plistlib.loads(launchd.render_plist(program_arguments=["/x/oara", "daemon"]))
    for key in ("Label", "RunAtLoad", "KeepAlive", "ThrottleInterval"):
        assert bundled[key] == cli[key], f"{key} drifted between the app and install-service"


# ── what ships ───────────────────────────────────────────────────────────────

def test_only_the_manifest_fields_beacon_reads(app):
    m = app.manifest(
        version="1.2.3", asset="Prometheus-1.2.3-arm64.zip", sha256="ab" * 32, size=81 * 2**20,
        extras=("anthropic",), lock_sha256="cd" * 32, without=(),
    )
    for key in ("version", "asset", "alias", "sha256", "size", "team_id", "bundle_id", "min_macos"):
        assert key in m
    assert m["alias"] == "Prometheus-mac-arm64.zip"
    assert m["team_id"] == "53JM8W47RL"
    assert m["bundle_id"] == "com.oaralabs.prometheus"
    assert m["python"]["version"] == app.PBS_PYTHON
    assert m["python"]["release"] == app.PBS_RELEASE
    assert m["lock_sha256"] == "cd" * 32          # a build records what it was built from
    assert m["without"] == []


def test_a_build_records_what_it_left_out(app):
    m = app.manifest(
        version="1.2.3", asset="a.zip", sha256="00" * 32, size=1, extras=(), lock_sha256="11" * 32,
        without=("pymupdf",),
    )
    assert m["without"] == ["pymupdf"]


# ── the standalone Python is checked, not trusted ────────────────────────────

def test_a_downloaded_runtime_with_the_wrong_checksum_is_refused(app, tmp_path):
    def fake_download(url, dest):
        Path(dest).write_bytes(b"not the python you pinned")

    with pytest.raises(app.BuildError, match="sha256"):
        app.fetch_python(tmp_path, download=fake_download)
    assert not (tmp_path / app.PBS_ASSET).exists(), "a file that failed its checksum must not stay in the cache"


def test_a_cached_runtime_is_verified_again_before_use(app, tmp_path):
    (tmp_path / app.PBS_ASSET).write_bytes(b"tampered after download")
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        raise AssertionError("cache hit must not download")

    with pytest.raises(app.BuildError, match="sha256"):
        app.fetch_python(tmp_path, download=fake_download)


def test_the_pinned_runtime_names_one_release_tag(app):
    assert app.PBS_RELEASE.isdigit() and len(app.PBS_RELEASE) == 8
    assert f"+{app.PBS_RELEASE}-" in app.PBS_ASSET
    assert app.PBS_ASSET.endswith("aarch64-apple-darwin-install_only_stripped.tar.gz")
    assert len(app.PBS_SHA256) == 64


# ── requirements ─────────────────────────────────────────────────────────────

UV_EXPORT = """\
# This file was autogenerated by uv via the following command:
#    uv export --frozen --no-dev --no-emit-project
annotated-types==0.7.0 \\
    --hash=sha256:aaa \\
    --hash=sha256:bbb
    # via pydantic
pymupdf==1.27.2.3 \\
    --hash=sha256:ccc \\
    --hash=sha256:ddd
    # via oara-prometheus
pydantic==2.13.5 ; sys_platform == 'darwin' \\
    --hash=sha256:eee
    # via oara-prometheus
"""


def test_a_requirement_filter_removes_the_whole_hash_block(app):
    out = app.filter_requirements(UV_EXPORT, exclude=("pymupdf",))
    assert "pymupdf" not in out
    assert "ccc" not in out and "ddd" not in out, "orphaned --hash lines would corrupt the next requirement"
    assert "annotated-types==0.7.0" in out and "pydantic==2.13.5" in out
    assert "--hash=sha256:eee" in out


def test_the_filter_matches_names_the_way_pip_does(app):
    text = "PyMuPDF==1.0 \\\n    --hash=sha256:x\n"
    assert app.filter_requirements(text, exclude=("pymupdf",)).strip() == ""
    assert app.filter_requirements(text, exclude=("pymupdf-fonts",)) == text


def test_requirements_must_carry_hashes(app):
    with pytest.raises(app.BuildError, match="hash"):
        app.require_hashes("pydantic==2.13.5\n")
    app.require_hashes(UV_EXPORT)


# ── a dirty tree does not ship ───────────────────────────────────────────────

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


def test_an_untracked_file_in_the_source_tree_refuses_the_build(app, tmp_path):
    (tmp_path / "src" / "prometheus").mkdir(parents=True)
    (tmp_path / "src" / "prometheus" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    app.require_clean_tree(tmp_path)
    (tmp_path / "src" / "prometheus" / "stray.py").write_text("y = 2\n")
    with pytest.raises(app.BuildError, match="stray.py"):
        app.require_clean_tree(tmp_path)
    app.require_clean_tree(tmp_path, allow_dirty=True)


# ── Mach-O discovery and the arm64-only rule ─────────────────────────────────

def test_macho_files_are_found_by_content_not_by_extension(app, tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python3.12").write_bytes(MACHO_ARM64)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "thing.so").write_bytes(MACHO_ARM64)
    (tmp_path / "lib" / "renamed.data").write_bytes(MACHO_ARM64)       # no telltale extension
    (tmp_path / "lib" / "fat.dylib").write_bytes(FAT)
    (tmp_path / "lib" / "notes.txt").write_text("text\n")
    (tmp_path / "lib" / "m.cpython-312.pyc").write_bytes(b"\xa7\x0d\x0d\x0a" + b"\x00" * 60)
    (tmp_path / "bin" / "python3").symlink_to("python3.12")
    found = {p.relative_to(tmp_path).as_posix() for p in app.find_macho(tmp_path)}
    assert found == {"bin/python3.12", "lib/thing.so", "lib/renamed.data", "lib/fat.dylib"}, \
        "symlinks, text and bytecode are not code to sign"


def test_a_leftover_x86_64_slice_fails_the_build(app, tmp_path):
    f = tmp_path / "etree.so"
    f.write_bytes(FAT)
    runner = _Runner([(("lipo", "-archs"), 0, "x86_64 arm64\n")])
    with pytest.raises(app.BuildError, match="x86_64"):
        app.require_arm64_only([f], runner=runner)
    runner_ok = _Runner([(("lipo", "-archs"), 0, "arm64\n")])
    app.require_arm64_only([f], runner=runner_ok)


# ── pruning never cuts into what the daemon needs ───────────────────────────

def test_prune_removes_dev_weight_and_keeps_the_daemon(app, tmp_path):
    root = tmp_path / "python"
    lib = root / "lib" / "python3.12"
    for rel in (
        "ensurepip/x.py", "idlelib/x.py", "tkinter/x.py", "turtledemo/x.py", "lib2to3/x.py",
        "site-packages/pip/x.py", "site-packages/pip-26.0.1.dist-info/METADATA",
        "site-packages/prometheus/__init__.py", "site-packages/prometheus/web/server.py",
        "lib-dynload/_sqlite3.cpython-312-darwin.so", "lib-dynload/_tkinter.cpython-312-darwin.so",
        "site-packages/pymupdf/__init__.py", "os.py",
    ):
        (lib / rel).parent.mkdir(parents=True, exist_ok=True)
        (lib / rel).write_text("x")
    for rel in ("include/Python.h", "share/man/x", "lib/tcl9.0/x", "lib/tk9.0/x", "bin/idle3", "bin/pip3",
                "bin/python3.12", "lib/libpython3.12.dylib"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    (lib / "site-packages" / "prometheus" / "__pycache__").mkdir()
    (lib / "site-packages" / "prometheus" / "__pycache__" / "a.pyc").write_text("x")

    app.prune_python(root)

    gone = ["ensurepip", "idlelib", "tkinter", "turtledemo", "lib2to3", "site-packages/pip",
            "site-packages/pip-26.0.1.dist-info", "lib-dynload/_tkinter.cpython-312-darwin.so",
            "site-packages/prometheus/__pycache__"]
    for rel in gone:
        assert not (lib / rel).exists(), rel
    for rel in ("include", "share", "lib/tcl9.0", "lib/tk9.0", "bin/idle3", "bin/pip3"):
        assert not (root / rel).exists(), rel
    for rel in ("site-packages/prometheus/web/server.py", "lib-dynload/_sqlite3.cpython-312-darwin.so",
                "os.py", "site-packages/pymupdf/__init__.py"):
        assert (lib / rel).exists(), f"pruning must keep {rel}"
    for rel in ("bin/python3.12", "lib/libpython3.12.dylib"):
        assert (root / rel).exists(), f"pruning must keep {rel}"


# ── signing ──────────────────────────────────────────────────────────────────

def test_signing_is_by_fingerprint_never_by_name(app):
    argv = app.codesign_argv("/x/y.so", FINGERPRINT)
    assert argv[argv.index("--sign") + 1] == FINGERPRINT
    for good in (FINGERPRINT.lower(),):
        app.codesign_argv("/x/y.so", good)
    # Two Developer ID Application certificates in the keychain share one name; a name is "ambiguous".
    for bad in ("Developer ID Application: William Hieber (53JM8W47RL)", "53JM8W47RL", "-", ""):
        with pytest.raises(app.BuildError, match="fingerprint"):
            app.codesign_argv("/x/y.so", bad)


def test_every_signature_is_hardened_and_timestamped_with_no_entitlements(app):
    argv = app.codesign_argv("/x/y.so", FINGERPRINT)
    assert "--force" in argv
    assert argv[argv.index("--options") + 1] == "runtime"
    assert "--timestamp" in argv
    assert "--entitlements" not in argv, "entitlements start at none; add one only when a failing run proves it"
    with_ent = app.codesign_argv("/x/y.so", FINGERPRINT, entitlements=Path("/e.plist"))
    assert with_ent[with_ent.index("--entitlements") + 1] == "/e.plist"


def test_sign_plan_signs_all_code_then_the_app_last_once_each(app, tmp_path):
    bundle = tmp_path / "Prometheus.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    (bundle / "Contents" / "MacOS" / "Prometheus").write_bytes(MACHO_ARM64)
    py = bundle / "Contents" / "Resources" / "python"
    (py / "bin").mkdir(parents=True)
    (py / "bin" / "python3.12").write_bytes(MACHO_ARM64)
    (py / "lib").mkdir()
    (py / "lib" / "libpython3.12.dylib").write_bytes(MACHO_ARM64)
    sp = py / "lib" / "python3.12" / "site-packages" / "lxml"
    sp.mkdir(parents=True)
    (sp / "etree.cpython-312-darwin.so").write_bytes(MACHO_ARM64)

    plan = app.sign_plan(bundle)
    assert plan[-1] == bundle, "the enclosing bundle is signed after everything inside it"
    assert len(plan) == len(set(plan)), "nothing is signed twice"
    names = {p.name for p in plan[:-1]}
    assert names == {"Prometheus", "python3.12", "libpython3.12.dylib", "etree.cpython-312-darwin.so"}
    assert plan.index(py / "lib" / "libpython3.12.dylib") < len(plan) - 1


def test_the_interpreter_gets_its_own_identifier(app):
    argv = app.codesign_argv("/a/Resources/python/bin/python3.12", FINGERPRINT, identifier="com.oaralabs.prometheus.python")
    assert argv[argv.index("--identifier") + 1] == "com.oaralabs.prometheus.python"


# ── the licenses a distributed bundle must carry ─────────────────────────────

def _dist(sp: Path, name: str, version: str, license_line: str, files: dict[str, str] | None = None) -> None:
    d = sp / f"{name}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{license_line}\n")
    for rel, body in (files or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(body)


def test_license_index_lists_every_distribution_and_flags_copyleft(app, tmp_path):
    _dist(tmp_path, "httpx", "0.28.1", "License-Expression: BSD-3-Clause")
    _dist(tmp_path, "PyMuPDF", "1.27.2.3", "License: Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License")
    _dist(tmp_path, "python-telegram-bot", "22.7", "License-Expression: LGPL-3.0-only")
    _dist(tmp_path, "certifi", "2026.2.25", "License-Expression: MPL-2.0")
    _dist(tmp_path, "mystery", "1.0", "Summary: no license field at all")
    rows = {r["name"]: r for r in app.license_index(tmp_path)}
    assert set(rows) == {"httpx", "PyMuPDF", "python-telegram-bot", "certifi", "mystery"}
    assert rows["httpx"]["copyleft"] is False
    assert rows["PyMuPDF"]["copyleft"] is True
    assert rows["python-telegram-bot"]["copyleft"] is True
    assert rows["certifi"]["copyleft"] is True            # MPL-2.0 is file-level copyleft: still named
    assert rows["mystery"]["license"] == "" and rows["mystery"]["copyleft"] is None, \
        "a missing license is unknown, not permissive"


def test_the_index_text_names_the_copyleft_ones_first(app, tmp_path):
    _dist(tmp_path, "httpx", "0.28.1", "License-Expression: BSD-3-Clause")
    _dist(tmp_path, "PyMuPDF", "1.27.2.3", "License: GNU AFFERO GPL 3.0")
    text = app.license_index_text(app.license_index(tmp_path))
    assert text.index("PyMuPDF") < text.index("httpx")
    assert "Copyleft" in text


# ── notarization credentials ─────────────────────────────────────────────────

def test_notarytool_takes_a_keychain_profile_or_the_apple_id_trio(app):
    assert app.notarytool_credential_args("prometheus", {}) == ["--keychain-profile", "prometheus"]
    env = {"APPLE_ID": " me@example.com ", "APPLE_APP_SPECIFIC_PASSWORD": "abcd-efgh-ijkl-mnop\n",
           "APPLE_TEAM_ID": "53JM8W47RL"}
    assert app.notarytool_credential_args("", env) == [
        "--apple-id", "me@example.com", "--password", "abcd-efgh-ijkl-mnop", "--team-id", "53JM8W47RL"]
    # A profile wins: it keeps the password out of argv, which is why the local route uses one.
    assert app.notarytool_credential_args("prometheus", env) == ["--keychain-profile", "prometheus"]


def test_missing_notarization_credentials_say_which_are_missing(app):
    with pytest.raises(app.BuildError, match="APPLE_APP_SPECIFIC_PASSWORD"):
        app.notarytool_credential_args("", {"APPLE_ID": "me@example.com", "APPLE_TEAM_ID": "53JM8W47RL"})
    with pytest.raises(app.BuildError, match="credential"):
        app.notarytool_credential_args("", {})


def test_a_failed_submission_never_echoes_the_password(app, tmp_path):
    password = "abcd-efgh-ijkl-mnop"
    env = {"APPLE_ID": "me@example.com", "APPLE_APP_SPECIFIC_PASSWORD": password, "APPLE_TEAM_ID": "53JM8W47RL"}
    runner = _Runner([(("xcrun", "notarytool", "submit"), 1, f"Error: invalid credentials {password}")])
    with pytest.raises(app.BuildError) as excinfo:
        app.notarize(tmp_path / "Prometheus.app", "", tmp_path, runner=runner, env=env)
    assert password not in str(excinfo.value)


# ── what the release leaves out, and what it looks like ──────────────────────

def test_pymupdf_is_left_out_unless_someone_asks_for_it(app):
    """PyMuPDF is AGPL-3.0. Will's call (2026-10-08): the app ships without it. Leaving it out is the default,
    so a release built by CI or by hand cannot include it by forgetting a flag."""
    assert app.DEFAULT_WITHOUT == ("pymupdf",)
    assert app.parse_args(["--identity", FINGERPRINT]).without == ("pymupdf",)
    assert app.parse_args(["--identity", FINGERPRINT, "--include-pymupdf"]).without == ()
    assert app.parse_args(["--identity", FINGERPRINT, "--without", "pymupdf,lxml"]).without == ("pymupdf", "lxml")


def test_the_manifest_says_what_the_default_build_left_out(app):
    ns = app.parse_args(["--identity", FINGERPRINT])
    m = app.manifest(version="1.2.3", asset="a.zip", sha256="00" * 32, size=1, extras=ns.extras,
                     lock_sha256="11" * 32, without=ns.without)
    assert m["without"] == ["pymupdf"]


def test_the_bundle_names_its_icon(app):
    assert app.info_plist("1.2.3")["CFBundleIconFile"] == "Prometheus"


def test_the_iconset_never_asks_for_more_pixels_than_the_artwork_has(app):
    sizes = app.ICONSET
    assert sizes["icon_16x16.png"] == 16 and sizes["icon_16x16@2x.png"] == 32
    assert sizes["icon_128x128.png"] == 128 and sizes["icon_128x128@2x.png"] == 256
    assert sizes["icon_512x512.png"] == 512
    # The source mark is 512 px. A 1024 px slot would be an upscale, so it is not produced.
    assert "icon_512x512@2x.png" not in sizes
    assert max(sizes.values()) <= app.ICON_SOURCE_PIXELS == 512
    for name, pixels in sizes.items():
        base = int(name.split("_")[1].split("x")[0])
        assert pixels == base * (2 if "@2x" in name else 1), name


def test_the_icon_comes_from_the_checked_in_mark_and_ends_as_an_icns(app, tmp_path):
    runner = _Runner()
    out = app.build_icon(tmp_path, runner=runner)
    assert out == tmp_path / "Prometheus.icns"
    joined = [" ".join(c) for c in runner.calls]
    assert any("make_iconset.swift" in c and "mark-512.png" in c for c in joined), joined
    converts = [c for c in runner.calls if c[0] == "iconutil"]
    assert len(converts) == 1, joined
    assert converts[0][1:3] == ["-c", "icns"] and converts[0][-1] == str(tmp_path / "Prometheus.icns"), converts
    assert (Path(app.__file__).parent / "icon" / "mark-512.png").is_file()
    assert (Path(app.__file__).parent / "icon" / "make_iconset.swift").is_file()
