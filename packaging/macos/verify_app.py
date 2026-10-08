#!/usr/bin/env python3
"""Verify a built Prometheus.app (or the zip of one) before anyone is asked to trust it.

    python packaging/macos/verify_app.py dist/macos/Prometheus-0.9.7-arm64.zip --notarized

Beacon downloads the app with Node, so the file carries no quarantine flag and Gatekeeper never
assesses it: Beacon's own checks are the only gate. These are the same questions, asked at build time:

* codesign --verify --deep --strict passes;
* the bundle and EVERY Mach-O inside it: Developer ID Application, the team, hardened runtime, a secure
  timestamp, arm64 only, and no entitlements (until a failing run proves one is needed);
* the Info.plist and the bundled LaunchAgent plist agree, and the agent runs the launcher;
* with --notarized: the ticket is stapled (``stapler validate``), ``syspolicy_check distribution`` passes
  (it lists the CDHash of every nested item), and ``spctl --assess`` on the BUNDLE says
  "Notarized Developer ID".

``spctl -t exec`` on nested executables is deliberately never asked: on Beacon's own notarized release it
rejected helpers that Apple had accepted, so its answer is wrong, not strict.

Exit 0 and the word OK when there is nothing to report; otherwise one problem per line and exit 1.
"""

from __future__ import annotations

import argparse
import plistlib
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_app  # noqa: E402  (sibling module, same directory)

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

USER_PATH_KEYS = ("StandardOutPath", "StandardErrorPath", "WorkingDirectory")


def read_plist(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = plistlib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dictionary plist")
    return data


# ── codesign ─────────────────────────────────────────────────────────────────

def parse_codesign_display(text: str) -> dict[str, Any]:
    """Read ``codesign -dvv`` output. Everything absent is reported as absent, never defaulted."""
    identifier = re.search(r"^Identifier=(.+)$", text, re.M)
    team = re.search(r"^TeamIdentifier=(.+)$", text, re.M)
    flags_match = re.search(r"flags=0x[0-9a-fA-F]+\(([^)]*)\)", text)
    team_id = team.group(1).strip() if team else None
    return {
        "identifier": identifier.group(1).strip() if identifier else None,
        "team_id": None if team_id in (None, "not set") else team_id,
        "authorities": re.findall(r"^Authority=(.+)$", text, re.M),
        "flags": set(flags_match.group(1).split(",")) if flags_match else set(),
        "timestamp": bool(re.search(r"^Timestamp=", text, re.M)),
    }


def signature_problems(facts: dict[str, Any], *, team_id: str, identifier: str | None = None) -> list[str]:
    problems: list[str] = []
    authorities = facts["authorities"]
    if not authorities or not authorities[0].startswith("Developer ID Application:"):
        shown = authorities[0] if authorities else "none (ad-hoc or unsigned)"
        problems.append(f"not signed by a Developer ID Application certificate (authority: {shown})")
    if "runtime" not in facts["flags"]:
        problems.append("the hardened runtime flag is missing")
    if not facts["timestamp"]:
        problems.append("no secure timestamp")
    if facts["team_id"] is None:
        problems.append("no team identifier")
    elif facts["team_id"] != team_id:
        problems.append(f"team {facts['team_id']} is not {team_id}")
    if identifier is not None and facts["identifier"] != identifier:
        problems.append(f"signing identifier {facts['identifier']} is not {identifier}")
    return problems


def entitlement_problems(dump: str, allowed: Iterable[str] = ()) -> list[str]:
    """``codesign -d --entitlements -`` prints only the Executable line when there are none."""
    permitted = set(allowed)
    return [
        f"entitlement {key} (none are allowed until a failing run proves one is needed)"
        for key in re.findall(r"<key>([^<]+)</key>", dump)
        if key not in permitted
    ]


# ── the bundle's own metadata ────────────────────────────────────────────────

def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def metadata_problems(
    info: dict[str, Any], agent: dict[str, Any], *, bundle_id: str, min_macos: str = build_app.MIN_MACOS,
) -> list[str]:
    problems: list[str] = []
    if info.get("CFBundleIdentifier") != bundle_id:
        problems.append(f"bundle id {info.get('CFBundleIdentifier')} is not {bundle_id}")
    minimum = str(info.get("LSMinimumSystemVersion", "0"))
    if _version_tuple(minimum) < _version_tuple(min_macos):
        problems.append(f"LSMinimumSystemVersion {minimum} is below {min_macos} (SMAppService needs macOS 13)")
    if agent.get("Label") != info.get("PrometheusAgentLabel"):
        problems.append(
            f"agent plist Label {agent.get('Label')} is not the PrometheusAgentLabel the launcher "
            f"registers ({info.get('PrometheusAgentLabel')})"
        )
    launcher = f"Contents/MacOS/{info.get('CFBundleExecutable', '')}"
    if agent.get("BundleProgram") != launcher:
        problems.append(
            f"the agent runs {agent.get('BundleProgram')}, not the launcher {launcher}: Login Items would "
            "name it after the program file"
        )
    if bundle_id not in (agent.get("AssociatedBundleIdentifiers") or []):
        problems.append("AssociatedBundleIdentifiers does not name the app, so Login Items cannot group it")
    for key in USER_PATH_KEYS:
        if key in agent:
            problems.append(f"{key} is set in a static bundled plist; the launcher sets it at start")
    for key, value in agent.items():
        if isinstance(value, str) and ("/Users/" in value or value.startswith("~")):
            problems.append(f"{key} carries a user path: {value}")
    return problems


def icon_problems(app: Path, info: dict[str, Any]) -> list[str]:
    """The Info.plist names an icon, and the bundle holds a real .icns of that name."""
    name = info.get("CFBundleIconFile")
    if not name:
        return ["Info.plist has no CFBundleIconFile: Login Items and notifications would show a generic icon"]
    path = app / "Contents" / "Resources" / (name if str(name).endswith(".icns") else f"{name}.icns")
    if not path.is_file():
        return [f"{path.relative_to(app)} is missing (CFBundleIconFile is {name})"]
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"icns":
        return [f"{path.relative_to(app)} is not an icns file"]
    return []


# ── the zip Beacon downloads ─────────────────────────────────────────────────

def zip_problems(names: Sequence[str]) -> list[str]:
    problems: list[str] = []
    tops = {n.split("/", 1)[0] for n in names if n}
    if "__MACOSX" in tops:
        problems.append("the zip contains __MACOSX resource-fork entries (use ditto -c -k --keepParent)")
    apps = sorted(t for t in tops if t.endswith(".app"))
    if len(apps) != 1:
        problems.append(f"the zip must hold exactly one app, found {len(apps)}: {apps}")
    for top in sorted(tops - set(apps) - {"__MACOSX"}):
        problems.append(f"entry outside the app: {top}")
    return problems


# ── notarization ─────────────────────────────────────────────────────────────

def _capture(argv: Sequence[str | Path], runner: Runner) -> tuple[int, str]:
    try:
        result = runner([str(a) for a in argv], capture_output=True, text=True)
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def notarization_problems(app: Path, *, runner: Runner = subprocess.run) -> list[str]:
    problems: list[str] = []
    code, out = _capture(["xcrun", "stapler", "validate", app], runner)
    if code != 0:
        problems.append(f"stapler validate: no valid ticket is stapled to the app (exit {code}): {out.strip()[-200:]}")
    code, out = _capture(["syspolicy_check", "distribution", app], runner)
    if code != 0:
        problems.append(f"syspolicy_check distribution failed (exit {code}): {out.strip()[-300:]}")
    # The BUNDLE only. Per-executable spctl gives wrong answers for notarized helpers.
    code, out = _capture(["spctl", "--assess", "--type", "execute", "--verbose=4", app], runner)
    if code != 0 or "source=Notarized Developer ID" not in out:
        problems.append(f"spctl does not accept the app as Notarized Developer ID (exit {code}): {out.strip()[-200:]}")
    return problems


# ── a whole bundle ───────────────────────────────────────────────────────────

def verify_app(
    app: Path, *, team_id: str = build_app.TEAM_ID, bundle_id: str = build_app.BUNDLE_ID, notarized: bool = False,
    allowed_entitlements: Iterable[str] = (), runner: Runner = subprocess.run,
) -> list[str]:
    problems: list[str] = []
    info_path = app / "Contents" / "Info.plist"
    if not info_path.is_file():
        return [f"{app} has no Contents/Info.plist"]
    info = read_plist(info_path)
    agent_name = info.get("PrometheusAgentPlist", "")
    agent_path = app / "Contents" / "Library" / "LaunchAgents" / agent_name
    if not agent_path.is_file():
        problems.append(f"the agent plist {agent_name or '(PrometheusAgentPlist unset)'} is not in Contents/Library/LaunchAgents")
    else:
        problems += metadata_problems(info, read_plist(agent_path), bundle_id=bundle_id)

    problems += icon_problems(app, info)

    launcher = app / "Contents" / "MacOS" / str(info.get("CFBundleExecutable", ""))
    interpreter = app / "Contents" / "Resources" / "python" / "bin" / "python3.12"
    for path in (launcher, interpreter):
        if not path.is_file():
            problems.append(f"missing {path.relative_to(app)}")

    code, out = _capture(["codesign", "--verify", "--deep", "--strict", "--verbose=2", app], runner)
    if code != 0:
        problems.append(f"codesign --verify --deep --strict failed (exit {code}): {out.strip()[-300:]}")
    _code, display = _capture(["codesign", "-dvv", app], runner)
    problems += [f"bundle: {p}" for p in signature_problems(
        parse_codesign_display(display), team_id=team_id, identifier=bundle_id)]

    for path in build_app.find_macho(app):
        rel = path.relative_to(app)
        _c, shown = _capture(["codesign", "-dvv", path], runner)
        problems += [f"{rel}: {p}" for p in signature_problems(parse_codesign_display(shown), team_id=team_id)]
        _c, ents = _capture(["codesign", "-d", "--entitlements", "-", path], runner)
        problems += [f"{rel}: {p}" for p in entitlement_problems(ents, allowed_entitlements)]
        _c, archs = _capture(["lipo", "-archs", path], runner)
        if archs.split() != ["arm64"]:
            problems.append(f"{rel}: architectures are {archs.strip() or 'unreadable'}, not arm64 only")

    site = app / "Contents" / "Resources" / "python" / "lib" / "python3.12" / "site-packages"
    if site.is_dir():
        if not (site / "prometheus" / "__pycache__").is_dir():
            problems.append("no precompiled bytecode: the first start would write into the sealed bundle")
        for leftover in site.glob("*.dist-info/direct_url.json"):
            problems.append(f"{leftover.relative_to(app)} records a path from the build machine")

    if notarized:
        problems += notarization_problems(app, runner=runner)
    return problems


def verify_zip(zip_path: Path, **kwargs: Any) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        problems = zip_problems(archive.namelist())
    if problems:
        return problems
    with tempfile.TemporaryDirectory(prefix="verify-app-") as temp:
        subprocess.run(["ditto", "-x", "-k", str(zip_path), temp], check=True)
        apps = sorted(Path(temp).glob("*.app"))
        return verify_app(apps[0], **kwargs)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a built app
    parser = argparse.ArgumentParser(description="Verify a built Prometheus.app or its zip.")
    parser.add_argument("path", type=Path, help="Prometheus.app or the zip that holds one")
    parser.add_argument("--notarized", action="store_true", help="also require a stapled, accepted notarization")
    parser.add_argument("--team", default=build_app.TEAM_ID)
    parser.add_argument("--bundle-id", default=build_app.BUNDLE_ID)
    parser.add_argument("--allow-entitlement", action="append", default=[], metavar="KEY")
    args = parser.parse_args(argv)
    options = dict(team_id=args.team, bundle_id=args.bundle_id, notarized=args.notarized,
                   allowed_entitlements=args.allow_entitlement)
    problems = verify_zip(args.path, **options) if args.path.suffix == ".zip" else verify_app(args.path, **options)
    if problems:
        print("\n".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
