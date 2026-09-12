"""The gate manifest — what a test run's composition was decided BY.

WHY THIS EXISTS
---------------
Two full-suite runs on the same tree produced different pass/skip splits
(7781/404 vs 7742/441, collected total constant at 8186). Nothing was added or
dropped — 39 tests changed category because the gates that decide them probe
LIVE HOST STATE at collection time, and that state differed between the two
runs. The root filesystem was measured as ``ro`` in one context and ``rw`` in
the other on the same machine, which flips the bwrap write-floor probe, which
flips ~37 security-floor tests between "ran and passed" and "skipped".

The probes are RIGHT to depend on the environment. A bwrap floor test that
executed on a host where bwrap cannot work would be worse than one that skips:
the skip reason says so explicitly ("SKIPPED IS NOT PASSED: on this machine
bash can write anywhere and these are live holes"). Chasing determinism in the
probe is either impossible or a lie.

What was missing is that NEITHER RUN RECORDED WHAT GATED IT. So "no regression
vs pristine main" quietly became an unearned claim: two runs were compared
without establishing they measured the same thing. This module makes the
provenance visible so an incomparable comparison refuses instead of lying.

THE COMPARISON RULE
-------------------
Same gate manifest FIRST, then same failure sets. When the manifests differ,
the comparison is **void** — not passed, not failed, UNMEASURED.
``scripts/compare_gate_manifests.py`` enforces that mechanically.

This is the recurring-failures §4f lesson one level up: don't force the value
to be stable, make the provenance visible so an incomparable comparison refuses
instead of reporting a verdict it cannot support.

DESIGN CONSTRAINTS
------------------
* The manifest calls the SAME probe functions the ``skipif`` markers call, not
  a parallel reimplementation. A second probe would drift from the first and
  report a gate state that did not decide anything.
* Volatile gates are probed, declared gates are read — both are recorded,
  because a comparison across environments (local vs CI) differs on the
  declared ones too (CI installs web+anthropic+mcp only, so every
  ``importorskip`` for slack_bolt/discord skips there and runs here).
* NOTHING here emits an infrastructure identifier: no hostname, no address, no
  resolved binary path (a path can carry a username), no account id. Binary
  gates record PRESENCE as a bool, not where it was found. The manifest goes to
  stdout and into files that get pasted into issues and PRs, and stdout is
  persisted state (the lesson behind #474).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

# Bump only on a change to what is COLLECTED (a gate added/removed/renamed),
# never on a change of host state. Two manifests of different schema versions
# are incomparable by definition and the comparison tool says so.
#
# 2 — `root_writable` (an `os.access("/", W_OK)` uid probe that reported false on
#     both a ro and an rw root) renamed and re-derived as `root_mount_ro`, the
#     `ro` flag of `/proc/self/mounts`. Schema 1 manifests exist from this
#     branch's own test runs and must compare VOID against schema 2, which is
#     what this bump buys.
# 1 — first version.
SCHEMA = 2

# Hostname of the DNS probe the network gate uses. A public name, not an
# infrastructure identifier; recorded because the gate is meaningless without
# saying what it resolved.
NETWORK_PROBE_HOST = "duckduckgo.com"

# Binaries a `skipif(shutil.which(...))` consults. Presence only — never the
# resolved path, which would persist a username-bearing absolute path.
PATH_GATES: tuple[str, ...] = (
    "aa-exec", "bash", "bwrap", "git", "node", "uv", "yt-dlp",
)

# Modules an `importorskip(...)` consults. Declared, not volatile — but they
# differ between environments, which is exactly what a cross-env comparison
# must not silently absorb.
MODULE_GATES: tuple[str, ...] = (
    "cryptography", "discord", "fastapi", "httpx", "mcp", "numpy",
    "PIL.Image", "skimage", "slack_bolt", "telegram", "websockets",
)


# Absolute paths are the one thing a probe's failure reason can carry that is an
# infrastructure identifier: a path names a username, and on a CI runner it names
# the runner's layout. `str(OSError)` is the concrete case — it interpolates the
# offending path verbatim:
#
#     OSError: [Errno 30] Read-only file system: '/home/will/tmpcmze1bu3'
#
# Measured, not assumed: reading a missing file under a home directory puts that
# home directory in `str(exc)`. Three of this module's `unprobeable` details
# interpolated `{exc}` straight through, and the probe details are pass-through
# stderr from bwrap/aa-exec, which can echo a bind path too.
#
# Redacted at ONE chokepoint (every detail, on its way out) rather than at each
# call site, for the same reason the render env is an allowlist: a per-site
# fix rots the moment someone adds a probe whose detail interpolates an
# exception. The pattern names the path CLASS, not a denylist of bad paths.
_PATH_RE = re.compile(
    r"/(?:home|Users|root|tmp|var|mnt|media|opt|srv|run|etc|usr|dev/shm)"
    r"(?:/[^\s'\"]*)?"
)


def _redact_paths(text: str) -> str:
    """Replace absolute paths with ``<path>`` in a probe's reason text.

    A lone ``/`` is deliberately NOT matched — ``bwrap: Failed to make / slave``
    is the single most useful sentence in the whole manifest, and the pattern
    requires a known top-level directory after the slash so it survives intact.
    """
    return _PATH_RE.sub("<path>", text)


@dataclass(frozen=True)
class Gate:
    """One condition a test's execution was gated on.

    ``value`` is a normalized string so the manifest is diffable line-by-line
    and hashable without type surprises. ``detail`` carries the probe's own
    reason text — for the volatile gates that is the string the skip reason
    would have shown, which is the part that tells you WHY, not just THAT.
    """

    name: str
    value: str
    detail: str = ""

    def __post_init__(self) -> None:
        """Redact any absolute path out of the detail, at construction.

        Done HERE rather than at each call site or in ``as_dict`` so no Gate can
        carry a username-bearing path no matter who builds it — a probe whose
        failure reason interpolates ``str(exc)`` (which carries the path) is
        cleaned by construction, and a future gate inherits the guarantee rather
        than having to remember it. ``value`` is left alone: values are a fixed
        vocabulary this module controls (present/absent/read-only/available/…),
        never free-form probe text, and the hash covers values only.
        """
        if self.detail:
            object.__setattr__(self, "detail", _redact_paths(self.detail))


def _root_mount_ro_verdict(options: str) -> str:
    """The ``ro`` flag of a mount-options string, as a gate verdict.

    PURE — a function of the string only, no host access. That is deliberate:
    the verdict must be testable against both mount states on a host that only
    ever presents one of them. This host has ``/`` mounted ``ro`` for most
    contexts, so any test that derives its expectation from the live mount
    silently agrees with a broken implementation and proves nothing. (Measured:
    a regression to ``os.access("/", W_OK)`` passed the whole suite here,
    because as a non-root user that call returns false on a ro root too — the
    two implementations agree exactly where they cannot be told apart.)

    An unreadable or absent mount table yields ``unknown``, NOT a guessed
    ``read-write``: this whole module exists so a run does not report a verdict
    it cannot support, and "the mount state could not be read" is not evidence
    of writability. ``unknown`` also means two runs that both failed to read the
    table compare equal, which is correct — they were gated the same way, even
    if neither of us knows how.
    """
    if not options or options == "absent" or options.startswith("unreadable"):
        return "unknown"
    return "read-only" if "ro" in set(options.split(",")) else "read-write"


def _root_mount_gate() -> tuple[str, str]:
    """Is ``/`` mounted read-only? The state that decides the bwrap floor probe.

    Read from ``/proc/self/mounts`` rather than by shelling out to ``findmnt``:
    a gate that records why another gate flipped must not itself depend on a
    binary being present, or it goes blank in exactly the degraded environments
    where it is most useful.

    THE VALUE IS THE ``ro`` FLAG, NOT THE WHOLE OPTIONS STRING. The full string
    goes in the detail, which the hash excludes by design: options like
    ``relatime``/``noatime`` or ``nosuid`` can vary between two runs without
    changing which tests run, and hashing them would void comparisons that are
    actually admissible. Only ``ro`` flips the bwrap probe (it cannot make ``/``
    slave), so only ``ro`` belongs in the verdict.

    This replaced a ``root_writable`` gate that called ``os.access("/", W_OK)``.
    Measured across two real artifacts, it reported ``false`` in both — one with
    ``/`` mounted ``ro`` and one with ``rw`` — because the suite runs as a
    non-root user, for whom ``/`` is never writable whatever the mount says. A
    gate that cannot vary is noise in the hash, and one that reads as if it
    measures the mount while measuring the uid is worse than absent.
    """
    options = _root_mount_options()
    return _root_mount_ro_verdict(options), options


def _root_mount_options() -> str:
    """Raw mount options of ``/``, e.g. ``ro,nosuid,nodev,relatime``."""
    try:
        for line in Path("/proc/self/mounts").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "/":
                return parts[3]
    except OSError as exc:  # pragma: no cover - platform without /proc
        return f"unreadable: {exc.__class__.__name__}"
    return "absent"


def _network_gate() -> tuple[str, str]:
    """The same DNS probe ``tests/test_web_tools.py::_has_network`` runs."""
    try:
        socket.gethostbyname(NETWORK_PROBE_HOST)
        return "up", f"resolved {NETWORK_PROBE_HOST}"
    except OSError as exc:
        return "down", f"{NETWORK_PROBE_HOST} did not resolve: {exc.__class__.__name__}"


def _probe_quietly(fn, ok_label: str, fail_label: str) -> tuple[str, str]:
    """Run a confinement probe without its own ERROR log duplicating the manifest.

    ``write_preflight``/``preflight`` log at ERROR when the floor is absent,
    which is correct for the daemon and noise here: the manifest is now the
    authoritative record of the gate state, and printing it twice — once as a
    bare log line, once inside the block that explains it — buries the part
    that matters. The level is restored unconditionally, including on an
    exception, so a probe that blows up cannot leave logging muted for the run.
    """
    import logging

    logger = logging.getLogger("prometheus.permissions.confinement")
    saved = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        ok, detail = fn()
    finally:
        logger.setLevel(saved)
    return (ok_label if ok else fail_label), detail


def _write_floor_gate() -> tuple[str, str]:
    """The bwrap write floor — the gate that flipped 37 tests between runs.

    Calls ``confinement.write_preflight(force=True)`` and resets the cache
    afterwards, exactly as ``tests/test_bash_write_floor.py::_floor_available``
    does, so the manifest reports the verdict that ACTUALLY decided the skip
    and leaves no cached state behind for later probes to inherit.
    """
    try:
        from prometheus.permissions import confinement as C

        def probe() -> tuple[bool, str]:
            ok, detail = C.write_preflight(force=True)
            C.reset_write_cache()
            return ok, detail

        return _probe_quietly(probe, "available", "unavailable")
    except Exception as exc:  # import or probe blew up — that IS the gate state
        return "unprobeable", f"{exc.__class__.__name__}: {exc}"[:200]


def _read_floor_gate() -> tuple[str, str]:
    """The AppArmor read floor — ``tests/test_bash_confinement.py::_profile_loaded``."""
    try:
        from prometheus.permissions import confinement as C

        def probe() -> tuple[bool, str]:
            ok, detail = C.preflight(C.PROFILE, force=True)
            C.reset_cache()
            return ok, detail

        return _probe_quietly(probe, "available", "unavailable")
    except Exception as exc:
        return "unprobeable", f"{exc.__class__.__name__}: {exc}"[:200]


def _live_config_gate() -> tuple[str, str]:
    """``tests/test_config_drift.py`` skips without a live config file.

    Presence only — the path is not emitted, and neither is any content. A
    developer's checkout has this file; a worktree and CI do not, which is why
    the suite composition differs between them.
    """
    try:
        from prometheus.config.defaults import REPO_CONFIG_PATH

        return ("present" if Path(REPO_CONFIG_PATH).exists() else "absent"), ""
    except Exception as exc:  # pragma: no cover
        return "unprobeable", f"{exc.__class__.__name__}: {exc}"[:200]


def _symlink_gate() -> tuple[str, str]:
    """Several API tests skip when the filesystem will not make a symlink."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="prom-gate-probe-") as tmp:
        target = Path(tmp) / "target"
        link = Path(tmp) / "link"
        target.write_text("x", encoding="utf-8")
        try:
            os.symlink(target, link)
            return "supported", ""
        except (OSError, NotImplementedError, AttributeError) as exc:
            return "unsupported", exc.__class__.__name__


def _running_as_root_gate() -> tuple[str, str]:
    """``test_api_wiki_read`` skips when it cannot make a file unreadable.

    Recorded as a capability, not a uid: what the test needs is "can I revoke
    read permission and have it be enforced", which is what actually gates it.
    """
    import os
    import stat
    import tempfile

    if os.name != "posix":
        return "unknown", "non-posix"
    with tempfile.TemporaryDirectory(prefix="prom-gate-probe-") as tmp:
        f = Path(tmp) / "f"
        f.write_text("x", encoding="utf-8")
        try:
            f.chmod(stat.S_IRUSR)
            can_read = os.access(f, os.R_OK)
            return ("cannot_revoke" if can_read else "can_revoke"), ""
        except OSError as exc:  # pragma: no cover
            return "unknown", exc.__class__.__name__


def collect_gates() -> tuple[Gate, ...]:
    """Every condition the suite's composition depends on, probed once.

    ORDER IS STABLE (sorted by name at the end) so two manifests of the same
    host state are byte-identical and the hash is reproducible.
    """
    gates: list[Gate] = []

    # ── volatile: live host state, the reason composition moved ────────────
    val, det = _write_floor_gate()
    gates.append(Gate("bwrap_write_floor", val, det))
    val, det = _read_floor_gate()
    gates.append(Gate("apparmor_read_floor", val, det))
    val, det = _root_mount_gate()
    gates.append(Gate("root_mount_ro", val, det))
    val, det = _network_gate()
    gates.append(Gate("network_dns", val, det))
    val, det = _symlink_gate()
    gates.append(Gate("symlinks", val, det))
    val, det = _running_as_root_gate()
    gates.append(Gate("read_permission_revocable", val, det))

    # ── declared: differ between environments (local vs CI), not run to run ─
    val, det = _live_config_gate()
    gates.append(Gate("live_repo_config", val, det))
    for name in PATH_GATES:
        gates.append(Gate(f"bin:{name}", "present" if shutil.which(name) else "absent"))
    for name in MODULE_GATES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        gates.append(Gate(f"mod:{name}", "present" if found else "absent"))

    return tuple(sorted(gates, key=lambda g: g.name))


def manifest_hash(gates: tuple[Gate, ...]) -> str:
    """A short stable digest of the GATE VALUES ONLY.

    Details are deliberately excluded: the reason text carries probe output
    (paths, errno strings) that can vary without the gate's verdict varying.
    Hashing values alone means two runs that gated identically compare equal
    even when the prose differs — which is the question that matters. A run
    whose gates differ is void regardless of how similar the prose looks.

    The schema is folded in, so manifests collected by different versions of
    this module never hash-equal each other.
    """
    lines = [f"schema={SCHEMA}"] + [f"{g.name}={g.value}" for g in gates]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def as_dict(gates: tuple[Gate, ...] | None = None) -> dict:
    """The manifest as a JSON-serializable dict.

    ``collected_at`` sits OUTSIDE the hashed fields: a timestamp must not make
    two identical gate states look incomparable.
    """
    import datetime

    if gates is None:
        gates = collect_gates()
    return {
        "schema": SCHEMA,
        "hash": manifest_hash(gates),
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "gates": {g.name: {"value": g.value, "detail": g.detail} for g in gates},
    }


def render(manifest: dict) -> str:
    """Human-readable block for the terminal — session start and summary."""
    lines = [
        "══ GATE MANIFEST ══════════════════════════════════════════════════",
        f"  hash: {manifest['hash']}   (schema {manifest['schema']})",
        "  What this run's composition was decided by. Volatile gates probe",
        "  live host state; a run is only comparable to another run whose",
        "  manifest hash MATCHES. Different hash → the comparison is VOID,",
        "  not passed and not failed: unmeasured.",
        "",
    ]
    for name, g in manifest["gates"].items():
        detail = f"  — {g['detail']}" if g["detail"] else ""
        lines.append(f"  {name}: {g['value']}{detail}")
    lines.append("═" * 68)
    return "\n".join(lines)


def write_manifest(path: str | Path) -> dict:
    """Write the manifest as JSON for a later two-run comparison."""
    manifest = as_dict()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
