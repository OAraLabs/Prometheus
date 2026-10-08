#!/usr/bin/env python3
"""Build Prometheus.app: a relocatable Python, the pinned wheels, a native launcher, signed.

    python packaging/macos/build_app.py --identity <SHA-1 of a Developer ID Application cert> \\
        [--extras anthropic,mcp,slack,discord] [--without pymupdf] [--notarize --notary-profile NAME]

Run it on an Apple-silicon Mac from a CLEAN checkout of the commit being released. The result is
``<out>/Prometheus-<version>-arm64.zip`` (a stapled app inside, when notarized), its stable alias
``Prometheus-mac-arm64.zip`` and ``prometheus-mac.json``. Design and contract with Beacon:
docs/design/macos-app-installer.md.

Everything that decides WHAT is built is a plain function with an injected runner, and is tested
without a Mac in tests/test_macos_app_build.py. The orchestration at the bottom (download, install,
compile, sign, notarize) is verified by running it, and ``verify_app.py`` checks the result.

The decisions worth knowing before changing anything here:

* The interpreter is python-build-standalone, pinned by release tag AND sha256. A floating "latest"
  would make two builds of one commit different programs.
* Dependencies come from ``uv export --frozen --hashes`` of uv.lock and install with
  ``--require-hashes --only-binary :all:``: nothing compiles on the build machine and nothing is
  resolved fresh. Prometheus itself is the one wheel built from the checkout.
* Bytecode is compiled BEFORE signing, as ``unchecked-hash``: a signed bundle must never be written
  into at runtime, so it cannot cache bytecode itself, and a sealed file's mtime means nothing.
* Everything is signed by SHA-1 fingerprint, never by name: two Developer ID Application certificates
  in this keychain share one name, and codesign calls a name that matches two "ambiguous".
* Hardened runtime, secure timestamp, and NO entitlements until a failing run proves one is needed.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

BUNDLE_ID = "com.oaralabs.prometheus"
APP_NAME = "Prometheus"
TEAM_ID = "53JM8W47RL"
MIN_MACOS = "13.0"
ALIAS = "Prometheus-mac-arm64.zip"
INTERPRETER_IDENTIFIER = f"{BUNDLE_ID}.python"

# python-build-standalone. Bump all four together and rebuild; the sha256 is the release's published
# SHA256SUMS entry for PBS_ASSET.
PBS_RELEASE = "20260325"
PBS_PYTHON = "3.12.13"
PBS_ASSET = f"cpython-{PBS_PYTHON}+{PBS_RELEASE}-aarch64-apple-darwin-install_only_stripped.tar.gz"
PBS_SHA256 = "c33a34853ae48d54fbac15cbb84ad67ccd8a639ce2cef866ecf474ebd02f1286"
PBS_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PBS_RELEASE}/{urllib.parse.quote(PBS_ASSET)}"
)

DEFAULT_EXTRAS = ("anthropic", "mcp", "slack", "discord")
# Paths that decide whether a tree is "the commit being released".
SOURCE_PATHSPECS = ("src", "config", "templates", "pyproject.toml", "uv.lock", "README.md", "packaging/macos")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class BuildError(RuntimeError):
    """The build cannot continue, and the message says why in terms of what to do next."""


def _run(argv: Sequence[str | Path], runner: Runner = subprocess.run, **kwargs: Any) -> str:
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    result = runner([str(a) for a in argv], **kwargs)
    if result.returncode != 0:
        tail = ((result.stderr or "") + (result.stdout or "")).strip()[-600:]
        raise BuildError(f"{' '.join(str(a) for a in argv[:3])} failed ({result.returncode}): {tail}")
    return result.stdout or ""


# ── identity, version, plists ────────────────────────────────────────────────

def read_version(repo: Path = REPO) -> str:
    text = (repo / "src" / "prometheus" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise BuildError("src/prometheus/__init__.py has no __version__")
    return match.group(1)


def _launchd() -> Any:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from prometheus.cli import launchd

    return launchd


def info_plist(version: str, build: str | None = None) -> dict[str, Any]:
    launchd = _launchd()
    return {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": build or version,
        "LSMinimumSystemVersion": MIN_MACOS,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        # A background service with a small window: no Dock icon, no menu bar.
        "LSUIElement": True,
        "NSHumanReadableCopyright": "Copyright (c) OAra Labs. MIT License.",
        # The launcher probes http://127.0.0.1 itself; say so rather than rely on ATS exempting IP literals.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        # Which agent plist the launcher registers; read from here so a development bundle with
        # other identifiers runs the identical launcher.
        "PrometheusAgentLabel": launchd.LABEL,
        "PrometheusAgentPlist": f"{launchd.LABEL}.plist",
    }


def agent_plist() -> bytes:
    """The LaunchAgent plist bundled at Contents/Library/LaunchAgents.

    Static: SMAppService reads it from inside the app, so it cannot carry the user's home directory.
    Logs and the working directory are set by the launcher at ``--run``. The program is the LAUNCHER,
    because Login Items names an agent after its program file ("Prometheus", not "python3.12").
    """
    launchd = _launchd()
    return launchd.render_plist(
        bundle_program=f"Contents/MacOS/{APP_NAME}",
        program_arguments=[APP_NAME, "--run"],
        associated_bundle_ids=[BUNDLE_ID],
    )


def manifest(
    *, version: str, asset: str, sha256: str, size: int, extras: Iterable[str], lock_sha256: str,
    without: Iterable[str],
) -> dict[str, Any]:
    """What Beacon and the website read. ``team_id`` is informational: Beacon pins its own."""
    return {
        "version": version,
        "asset": asset,
        "alias": ALIAS,
        "sha256": sha256,
        "size": size,
        "team_id": TEAM_ID,
        "bundle_id": BUNDLE_ID,
        "min_macos": MIN_MACOS,
        "arch": "arm64",
        "python": {"version": PBS_PYTHON, "release": PBS_RELEASE, "sha256": PBS_SHA256},
        "extras": list(extras),
        "without": list(without),
        "lock_sha256": lock_sha256,
    }


# ── the interpreter ──────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, str(dest))  # noqa: S310 - https URL pinned above


def fetch_python(cache: Path, download: Callable[[str, Path], None] | None = None) -> Path:
    """The pinned standalone Python archive, downloaded once and verified every time it is used."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / PBS_ASSET
    if not archive.exists():
        (download or _default_download)(PBS_URL, archive)
    actual = sha256_file(archive)
    if actual != PBS_SHA256:
        archive.unlink(missing_ok=True)
        raise BuildError(
            f"{PBS_ASSET}: sha256 {actual} is not the pinned {PBS_SHA256}; the file was removed. "
            "If python-build-standalone republished it, check its SHA256SUMS before changing the pin."
        )
    return archive


def extract_python(archive: Path, dest: Path) -> Path:
    """Unpack to ``dest/python`` (the archive's own top-level directory)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:  # pragma: no cover - Python without PEP 706
            tar.extractall(dest)
    root = dest / "python"
    if not (root / "bin" / "python3.12").exists():
        raise BuildError(f"{archive.name} did not contain python/bin/python3.12")
    return root


# ── requirements ─────────────────────────────────────────────────────────────

_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_blocks(text: str) -> list[list[str]]:
    """Split a ``uv export`` file into blocks: each requirement with its indented hash/comment lines.
    Lines that open no requirement (the header comments) form blocks of their own."""
    blocks: list[list[str]] = []
    for line in text.splitlines(keepends=True):
        if line[:1] in (" ", "\t") and blocks:
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


def filter_requirements(text: str, exclude: Iterable[str]) -> str:
    """Drop whole requirements by (pip-normalised) name, with their --hash lines."""
    drop = {_normalise(n) for n in exclude}
    kept: list[str] = []
    for block in _requirement_blocks(text):
        match = _NAME.match(block[0])
        if match and not block[0].lstrip().startswith("#") and _normalise(match.group(1)) in drop:
            continue
        kept.extend(block)
    return "".join(kept)


def require_hashes(text: str) -> None:
    """Every requirement must carry at least one --hash, or ``--require-hashes`` is a lie."""
    for block in _requirement_blocks(text):
        first = block[0]
        if not _NAME.match(first) or first.lstrip().startswith("#"):
            continue
        if not any("--hash=" in line for line in block):
            raise BuildError(f"requirement without a hash: {first.strip()[:80]}")


# ── a dirty tree does not ship ───────────────────────────────────────────────

def require_clean_tree(repo: Path = REPO, allow_dirty: bool = False, runner: Runner = subprocess.run) -> None:
    """Refuse to build from a tree that is not the commit: the wheel is built from the files on disk."""
    if allow_dirty:
        return
    out = _run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_PATHSPECS], runner)
    if out.strip():
        shown = "\n  ".join(out.strip().splitlines()[:12])
        raise BuildError(
            "the working tree is not clean where the wheel is built from; commit or remove:\n  " + shown
            + "\n(--allow-dirty overrides, for a throwaway local build only)"
        )


# ── Mach-O ───────────────────────────────────────────────────────────────────

_THIN_MAGICS = {
    bytes.fromhex("feedface"), bytes.fromhex("cefaedfe"),
    bytes.fromhex("feedfacf"), bytes.fromhex("cffaedfe"),
}
_FAT_MAGICS = {
    bytes.fromhex("cafebabe"), bytes.fromhex("bebafeca"),
    bytes.fromhex("cafebabf"), bytes.fromhex("bfbafeca"),
}


def _is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    if len(head) < 8:
        return False
    magic = head[:4]
    if magic in _THIN_MAGICS:
        return True
    if magic in _FAT_MAGICS:
        # A Java class file shares 0xcafebabe; its next word is a version (>= 45), a fat header's is
        # an architecture count.
        count = int.from_bytes(head[4:8], "big")
        return 0 < count < 20
    return False


def find_macho(root: Path) -> list[Path]:
    """Every Mach-O file under root, found by content. Symlinks are not code to sign."""
    found: list[Path] = []
    for directory, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink() and _is_macho(path):
                found.append(path)
    return sorted(found)


def _archs(path: Path, runner: Runner) -> list[str]:
    return _run(["lipo", "-archs", path], runner).split()


def thin_to_arm64(files: Iterable[Path], runner: Runner = subprocess.run) -> list[Path]:
    """Strip the x86_64 slice from universal files (lxml ships one). Returns what was thinned."""
    thinned: list[Path] = []
    for path in files:
        archs = _archs(path, runner)
        if "x86_64" in archs and "arm64" in archs:
            temp = path.with_name(path.name + ".thin")
            _run(["lipo", "-thin", "arm64", path, "-output", temp], runner)
            temp.replace(path)
            thinned.append(path)
    return thinned


def require_arm64_only(files: Iterable[Path], runner: Runner = subprocess.run) -> None:
    for path in files:
        archs = _archs(path, runner)
        if archs != ["arm64"]:
            raise BuildError(f"{path.name} is {' '.join(archs) or 'unreadable'}, not arm64 only: {path}")


# ── pruning ──────────────────────────────────────────────────────────────────

_PRUNE_STDLIB = (
    "ensurepip", "idlelib", "tkinter", "turtledemo", "lib2to3", "config-3.12-darwin",
    "site-packages/pip", "site-packages/setuptools", "site-packages/_distutils_hack",
)
_PRUNE_TOP = ("include", "share")
_PRUNE_LIB_GLOBS = ("tcl*", "tk*", "itcl*", "tdbc*", "thread*", "Tix*")
_PRUNE_BIN_GLOBS = ("idle3*", "pip*", "2to3*", "pydoc*", "python3-config", "python3.12-config")


def prune_python(root: Path) -> None:
    """Remove what a daemon never runs: IDLE, Tk, pip, headers, man pages, and all bytecode (it is
    recompiled after pruning, so nothing stale from the archive is signed)."""
    lib = root / "lib" / "python3.12"
    for rel in _PRUNE_STDLIB:
        shutil.rmtree(lib / rel, ignore_errors=True)
    for dist_info in (lib / "site-packages").glob("pip-*.dist-info"):
        shutil.rmtree(dist_info, ignore_errors=True)
    for dist_info in (lib / "site-packages").glob("setuptools-*.dist-info"):
        shutil.rmtree(dist_info, ignore_errors=True)
    for ext in (lib / "lib-dynload").glob("_tkinter*.so"):
        ext.unlink()
    for rel in _PRUNE_TOP:
        shutil.rmtree(root / rel, ignore_errors=True)
    for pattern in _PRUNE_LIB_GLOBS:
        for path in (root / "lib").glob(pattern):
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
    for pattern in _PRUNE_BIN_GLOBS:
        for path in (root / "bin").glob(pattern):
            path.unlink()
    for cache in list(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)


def strip_build_traces(site_packages: Path) -> int:
    """Remove ``direct_url.json`` files: uv records the builder's local wheel path in one, which
    would put a path from the build machine into every copy of the app. Returns how many."""
    removed = 0
    for path in site_packages.glob("*.dist-info/direct_url.json"):
        path.unlink()
        removed += 1
    return removed


# ── signing ──────────────────────────────────────────────────────────────────

_FINGERPRINT = re.compile(r"^[0-9A-Fa-f]{40}$")


def codesign_argv(
    path: str | Path, fingerprint: str, *, identifier: str | None = None, entitlements: Path | None = None,
) -> list[str]:
    if not _FINGERPRINT.match(fingerprint or ""):
        raise BuildError(
            "sign by the certificate's SHA-1 fingerprint (40 hex characters), not by name: two Developer ID "
            "Application certificates share one name here, and codesign calls that ambiguous. "
            "`security find-identity -v -p codesigning` lists them."
        )
    argv = ["codesign", "--force", "--sign", fingerprint, "--options", "runtime", "--timestamp"]
    if identifier:
        argv += ["--identifier", identifier]
    if entitlements is not None:
        argv += ["--entitlements", str(entitlements)]
    argv.append(str(path))
    return argv


def sign_plan(bundle: Path) -> list[Path]:
    """Every Mach-O inside the bundle, deepest first, then the bundle itself, each exactly once."""
    inside = sorted(set(find_macho(bundle)), key=lambda p: (-len(p.parts), str(p)))
    return [*inside, bundle]


def sign_all(
    bundle: Path, fingerprint: str, *, entitlements: Path | None = None, runner: Runner = subprocess.run,
) -> int:
    count = 0
    for path in sign_plan(bundle):
        identifier = None
        if path.name == "python3.12" and path.parent.name == "bin":
            identifier = INTERPRETER_IDENTIFIER
        ents = entitlements if path == bundle else None
        _run(codesign_argv(path, fingerprint, identifier=identifier, entitlements=ents), runner)
        count += 1
    return count


# ── licenses ─────────────────────────────────────────────────────────────────

_COPYLEFT = re.compile(
    r"AGPL|AFFERO|LGPL|LESSER GENERAL|\bGPL|GENERAL PUBLIC|\bMPL\b|MOZILLA PUBLIC|\bEPL\b|ECLIPSE PUBLIC|CDDL|SSPL",
    re.I,
)


def license_index(site_packages: Path) -> list[dict[str, Any]]:
    """One row per installed distribution. ``copyleft`` is True/False, or None when the metadata
    names no license at all: unknown is never reported as permissive."""
    rows: list[dict[str, Any]] = []
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        metadata = dist_info / "METADATA"
        if not metadata.is_file():
            continue
        msg = email.parser.Parser().parsestr(metadata.read_text(encoding="utf-8", errors="replace"))
        expression = (msg.get("License-Expression") or "").strip()
        legacy = (msg.get("License") or "").strip().splitlines()
        license_text = expression or (legacy[0][:200] if legacy else "")
        classifiers = [c.split("::")[-1].strip() for c in msg.get_all("Classifier") or [] if c.startswith("License ::")]
        haystack = " ".join([license_text, *classifiers])
        rows.append({
            "name": msg.get("Name") or dist_info.name.split("-")[0],
            "version": msg.get("Version") or "",
            "license": license_text,
            "classifiers": classifiers,
            "copyleft": (True if _COPYLEFT.search(haystack) else False) if haystack.strip() else None,
        })
    return rows


def license_index_text(rows: Sequence[dict[str, Any]]) -> str:
    """The human-readable index shipped in the bundle: copyleft first, then unknown, then the rest."""
    order = {True: 0, None: 1, False: 2}
    lines = [
        "Third-party software in Prometheus.app",
        "=======================================",
        "",
        "Each package's own license text is kept in its .dist-info directory under",
        "Contents/Resources/python/lib/python3.12/site-packages/. Prometheus itself is MIT (LICENSE).",
        "",
    ]
    for row in sorted(rows, key=lambda r: (order[r["copyleft"]], r["name"].lower())):
        flag = {True: "Copyleft", None: "Unknown ", False: ""}[row["copyleft"]]
        lines.append(f"{flag:<9}{row['name']} {row['version']}  {row['license'] or '(no license in metadata)'}")
    return "\n".join(lines) + "\n"


# ── orchestration (verified by running it) ───────────────────────────────────

def _extras_args(extras: Iterable[str]) -> list[str]:
    return [a for e in extras for a in ("--extra", e)]


def assemble(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - needs a Mac
    work = Path(args.work).resolve()
    out = Path(args.out).resolve()
    version = read_version()
    require_clean_tree(REPO, allow_dirty=args.allow_dirty)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    archive = fetch_python(work / "cache")
    py_root = extract_python(archive, work / "runtime")
    py = py_root / "bin" / "python3.12"
    site = py_root / "lib" / "python3.12" / "site-packages"

    wheel_dir = work / "wheel"
    shutil.rmtree(wheel_dir, ignore_errors=True)
    _run(["uv", "build", "--wheel", "--out-dir", wheel_dir], cwd=REPO)
    wheels = sorted(wheel_dir.glob("oara_prometheus-*.whl"))
    if len(wheels) != 1:
        raise BuildError(f"expected one oara_prometheus wheel in {wheel_dir}, found {len(wheels)}")

    requirements = _run(
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", *_extras_args(args.extras)], cwd=REPO,
    )
    requirements = filter_requirements(requirements, args.without)
    require_hashes(requirements)
    req_file = work / "requirements.txt"
    req_file.write_text(requirements, encoding="utf-8")
    _run(["uv", "pip", "install", "--python", py, "--target", site, "--no-deps", "--require-hashes",
          "--only-binary", ":all:", "-r", req_file])
    _run(["uv", "pip", "install", "--python", py, "--target", site, "--no-deps", wheels[0]])
    strip_build_traces(site)

    prune_python(py_root)
    machos = find_macho(py_root)
    thin_to_arm64(machos)
    require_arm64_only(find_macho(py_root))

    rows = license_index(site)
    # Bytecode last, after every file that will be sealed exists. The path stripped from and prepended
    # to the code objects keeps the builder's directory out of every copy of the app; the import system
    # rewrites co_filename to the real location when it loads a pyc, so tracebacks are unaffected.
    _run([py, "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash",
          "-s", py_root / "lib" / "python3.12", "-p", "/Prometheus.app/Contents/Resources/python/lib/python3.12",
          py_root / "lib" / "python3.12"], env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"})

    bundle = work / f"{APP_NAME}.app"
    shutil.rmtree(bundle, ignore_errors=True)
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    (bundle / "Contents" / "Resources").mkdir()
    (bundle / "Contents" / "Library" / "LaunchAgents").mkdir(parents=True)
    shutil.copytree(py_root, bundle / "Contents" / "Resources" / "python", symlinks=True)

    launcher = bundle / "Contents" / "MacOS" / APP_NAME
    _run(["xcrun", "swiftc", "-O", "-target", "arm64-apple-macos11.0",
          REPO / "packaging" / "macos" / "launcher" / "main.swift", "-o", launcher])
    linked = _run(["otool", "-L", launcher])
    if "@rpath" in linked:
        raise BuildError("the launcher links a library it would have to ship:\n" + linked)

    with (bundle / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(info_plist(version), handle)
    launchd = _launchd()
    (bundle / "Contents" / "Library" / "LaunchAgents" / f"{launchd.LABEL}.plist").write_bytes(agent_plist())
    for name in ("LICENSE", "NOTICE"):
        if (REPO / name).is_file():
            shutil.copy2(REPO / name, bundle / "Contents" / "Resources" / name)
    (bundle / "Contents" / "Resources" / "THIRD-PARTY-LICENSES.txt").write_text(
        license_index_text(rows), encoding="utf-8")

    sign_all(bundle, args.identity)

    asset = f"{APP_NAME}-{version}-arm64.zip"
    if args.notarize:
        notarize(bundle, args.notary_profile, work)
    zip_path = out / asset
    zip_path.unlink(missing_ok=True)
    _run(["ditto", "-c", "-k", "--keepParent", bundle, zip_path])
    shutil.copy2(zip_path, out / ALIAS)
    digest = sha256_file(zip_path)
    meta = manifest(
        version=version, asset=asset, sha256=digest, size=zip_path.stat().st_size, extras=args.extras,
        lock_sha256=sha256_file(REPO / "uv.lock"), without=args.without,
    )
    (out / "prometheus-mac.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "app": str(bundle), "zip": str(zip_path), "alias": str(out / ALIAS), "sha256": digest,
        "size": zip_path.stat().st_size, "installed_bytes": sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file()),
        "files": sum(1 for f in bundle.rglob("*") if f.is_file()), "mach_o": len(find_macho(bundle)),
        "notarized": bool(args.notarize), "copyleft": [r["name"] for r in rows if r["copyleft"]],
    }


def notarize(bundle: Path, profile: str, work: Path, runner: Runner = subprocess.run) -> None:  # pragma: no cover
    """Submit, wait, staple, validate. A stalled notary is reported, never worked around."""
    if not profile:
        raise BuildError("--notarize needs --notary-profile (xcrun notarytool store-credentials ...)")
    submit_zip = work / f"{bundle.stem}-submit.zip"
    submit_zip.unlink(missing_ok=True)
    _run(["ditto", "-c", "-k", "--keepParent", bundle, submit_zip], runner)
    result = _run(["xcrun", "notarytool", "submit", submit_zip, "--keychain-profile", profile, "--wait",
                   "--output-format", "json"], runner)
    info = json.loads(result)
    if info.get("status") != "Accepted":
        log = _run(["xcrun", "notarytool", "log", info.get("id", ""), "--keychain-profile", profile], runner)
        raise BuildError(f"notarization {info.get('status')}: {log[-1500:]}")
    _run(["xcrun", "stapler", "staple", bundle], runner)
    _run(["xcrun", "stapler", "validate", bundle], runner)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a Mac
    parser = argparse.ArgumentParser(description="Build Prometheus.app (arm64).")
    parser.add_argument("--identity", required=True, help="SHA-1 fingerprint of the Developer ID Application certificate")
    parser.add_argument("--extras", default=",".join(DEFAULT_EXTRAS), type=lambda s: tuple(x for x in s.split(",") if x))
    parser.add_argument("--without", default="", type=lambda s: tuple(x for x in s.split(",") if x),
                        help="distributions to leave out of the bundle (e.g. pymupdf)")
    parser.add_argument("--work", default=str(REPO / "dist" / "macos" / "work"))
    parser.add_argument("--out", default=str(REPO / "dist" / "macos"))
    parser.add_argument("--notarize", action="store_true")
    parser.add_argument("--notary-profile", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = assemble(args)
    except BuildError as exc:
        print(f"build_app: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
