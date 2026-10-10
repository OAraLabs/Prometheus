"""The one-time pairing secret for a Beacon on the SAME Mac as the daemon.

Why a file
----------
Prometheus.app's daemon is started by launchd, so nobody sees the six-digit code that setup mode prints
at a terminal, and a secret in the daemon's environment cannot work either (launchd starts it, not
Beacon). Beacon on the same Mac can read a file that only its user can read, which is the property that
matters: any process running as this user can already read ``~/.config/prometheus/env``, so this
widens nothing, and a process running as anyone else can read neither.

The file
--------
``<pairing dir>/pair.secret``: 32 random bytes, base64url without padding (43 characters), one line.
The directory is ``0700`` and the file ``0600``, both owned by this user. Beacon reads it and sends it to
the daemon's loopback pairing route; the daemon compares against the file on EVERY attempt (it is the
source of truth, so ``Prometheus --pair`` can write a fresh one while the daemon runs) and deletes it on
first successful use.

The fingerprint beside it
-------------------------
``<pairing dir>/pair.fp``: the 16 hex characters this daemon's ``GET /api/hello`` advertises as ``fp``, one
line, with the secret's protections (0700 directory, 0600 file, never through a link). A client reads it and
compares it with hello's ``fp`` BEFORE it sends the secret, so a secret is not handed to something else on the
port. The full daemon writes it at boot from the key it has just made or loaded; setup mode writes it only when
a key already exists (setup mode creates no ``~/.prometheus`` state, and a key is state), and otherwise REMOVES
an earlier one, so it never vouches for an identity this daemon does not have. It is not proof: ``fp`` is public,
and another local account that once read it could serve the same value. Key-backed proof is the TLS change's.

Decisions that look odd and are not
-----------------------------------
* **It never expires.** A person may open Beacon an hour, or a day, after installing. Permissions are
  the authority here, not a timer, and the six-digit code's 15-minute window is a property of something
  printed where anyone can see it. An unused secret also survives a daemon restart: minting twice
  returns the first.
* **A file others could read is replaced, not trusted.** If the mode was loosened, the secret may have
  been read; the next mint writes a fresh one and the old value stops matching.
* **Symlinks are refused** (the directory) and never followed (the file): this directory is a trust
  boundary, and a link into somewhere an attacker controls would hand them the choice of the secret.
* **Nothing here logs a value**, and failures say what is wrong with the PATH, not what was presented.
* **Use is exactly once, and ``unlink`` is NOT how that is enforced.** Measured on APFS (macOS 15.6):
  with eight threads unlinking one path, up to seven calls RETURNED SUCCESS in a single trial, so
  "whoever's unlink succeeds wins" lets one secret be used several times. ``rename`` and
  ``open(O_CREAT|O_EXCL)`` gave exactly one winner in 300 of 300 trials. ``consume_if_matches`` claims
  the file by renaming it to a private name, then checks that what it claimed really is what it was
  shown, then deletes it.

This module knows nothing about HTTP. Loopback checks and the response belong to the route.
"""

from __future__ import annotations

import errno
import hmac
import logging
import os
import re
import secrets
import stat
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_KIND = "PROMETHEUS_INSTALL_KIND"
ENV_DIR = "PROMETHEUS_LOCAL_PAIRING_DIR"
SECRET_FILE = "pair.secret"
FINGERPRINT_FILE = "pair.fp"

# What a minted secret looks like, and the widest shape read back. The width is deliberate slack so a
# longer secret in a future version is still read by an older daemon rather than silently ignored.
_SHAPE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_SIX_DIGITS = re.compile(r"^\d{6}$")
# config/instance_key.py's fingerprint: the first FINGERPRINT_HEX_CHARS (16) of a lowercase SHA-256 hexdigest.
_FP_SHAPE = re.compile(r"^[0-9a-f]{16}$")
_MAX_READ = 256


class PairingFileError(RuntimeError):
    """The pairing directory is not one we can trust, and why."""


def _platform() -> str:
    return sys.platform


def enabled() -> bool:
    """True for the app install (the launcher sets PROMETHEUS_INSTALL_KIND=app), or when a directory
    is named explicitly (tests, and anyone wiring this up by hand)."""
    return os.environ.get(ENV_KIND) == "app" or bool(os.environ.get(ENV_DIR))


def pairing_dir() -> Path:
    override = os.environ.get(ENV_DIR)
    if override:
        return Path(override).expanduser()
    if _platform() == "darwin":
        return Path.home() / "Library" / "Application Support" / "Prometheus" / "pairing"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "prometheus" / "pairing"


def secret_path(directory: Path | None = None) -> Path:
    return (directory or pairing_dir()) / SECRET_FILE


def fingerprint_path(directory: Path | None = None) -> Path:
    return (directory or pairing_dir()) / FINGERPRINT_FILE


def is_secret_shaped(value: str) -> bool:
    """Anything that is not exactly six digits is not a typed code. Routes use this to decide which
    path a presented value takes, so a wrong six-digit guess never touches this file's state."""
    return _SIX_DIGITS.fullmatch(value) is None


# ── the directory ────────────────────────────────────────────────────────────

def _directory_is_safe(directory: Path) -> bool:
    """A real directory (not a link), ours, and closed to everyone else."""
    try:
        st = os.lstat(directory)
    except OSError:
        return False
    return (
        stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode)
        and st.st_uid == os.geteuid() and not (st.st_mode & 0o077)
    )


def _ensure_directory(directory: Path) -> None:
    parent = directory.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(directory)
    if stat.S_ISLNK(st.st_mode):
        raise PairingFileError(f"{directory} is a symlink; refusing to put a secret behind it")
    if not stat.S_ISDIR(st.st_mode):
        raise PairingFileError(f"{directory} exists and is not a directory")
    if st.st_uid != os.geteuid():
        raise PairingFileError(f"{directory} is owned by another user")
    if st.st_mode & 0o077:
        os.chmod(directory, 0o700)
        logger.warning("pairing directory %s was open to others; tightened to 0700", directory)


# ── reading ──────────────────────────────────────────────────────────────────

def _read_file(path: Path, shape: re.Pattern[str] = _SHAPE) -> str | None:
    """Read and validate one pairing file. The file must be regular, ours, closed to others, never a link."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            logger.warning("pairing file %s is a symlink; ignoring it", path)
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or (st.st_mode & 0o077):
            logger.warning("pairing file %s has unsafe ownership or permissions; ignoring it", path)
            return None
        raw = os.read(fd, _MAX_READ)
    finally:
        os.close(fd)
    text = raw.decode("utf-8", errors="replace").strip()
    return text if shape.fullmatch(text) else None


def read_secret(directory: Path | None = None) -> str | None:
    """The current secret, or None if there is none or it cannot be trusted (loose mode, wrong
    owner, a symlink, or content that is not a secret). Never raises for a bad file."""
    directory = directory or pairing_dir()
    if not _directory_is_safe(directory):
        return None
    return _read_file(directory / SECRET_FILE)


def check_secret(presented: object, directory: Path | None = None) -> bool:
    """Whether ``presented`` is exactly the current secret. Constant-time; False when there is none."""
    if not isinstance(presented, str) or not _SHAPE.fullmatch(presented):
        return False
    current = read_secret(directory)
    if current is None:
        return False
    return hmac.compare_digest(presented.encode(), current.encode())


# ── writing ──────────────────────────────────────────────────────────────────

def _write(directory: Path, text: str, name: str = SECRET_FILE) -> None:
    """Write ``text`` to ``name`` via a private temp file and a rename, so a reader sees all of it or none. The
    rename replaces a symlink at ``name`` rather than writing through it."""
    temp = directory / f".{name}.{os.getpid()}.{secrets.token_hex(4)}"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, (text + "\n").encode())
    finally:
        os.close(fd)
    try:
        os.replace(temp, directory / name)
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def mint_secret(directory: Path | None = None) -> str:
    """The secret to hand out: the existing one if it is intact, otherwise a fresh one.

    Idempotent on purpose. A daemon that restarts before Beacon has paired must not invalidate a
    secret Beacon may already be holding.
    """
    directory = directory or pairing_dir()
    _ensure_directory(directory)
    existing = read_secret(directory)
    if existing is not None:
        return existing
    return replace_secret(directory)


def replace_secret(directory: Path | None = None) -> str:
    """Write a fresh secret over any earlier one (re-pairing after a Beacon reinstall)."""
    directory = directory or pairing_dir()
    _ensure_directory(directory)
    secret = secrets.token_urlsafe(32)
    _write(directory, secret)
    logger.info("pairing secret written to %s (value never logged)", directory / SECRET_FILE)
    return secret


def consume_if_matches(presented: object, directory: Path | None = None) -> bool:
    """Use the secret once: True for exactly one caller, and only if its value matched.

    A wrong guess returns before the file is touched. A right one claims the file by renaming it (the
    atomic step; see the module docstring for why not ``unlink``), then compares what it claimed with what
    it was shown, because between the check and the claim a re-pair may have replaced the file. If the
    claimed file is not the presented secret it is put back, unless a newer one already exists.
    """
    directory = directory or pairing_dir()
    if not check_secret(presented, directory):
        return False
    assert isinstance(presented, str)
    path = directory / SECRET_FILE
    claimed = directory / f".{SECRET_FILE}.claimed.{os.getpid()}.{secrets.token_hex(4)}"
    try:
        os.rename(path, claimed)
    except FileNotFoundError:
        return False   # another request claimed it first
    won = False
    try:
        text = _read_file(claimed)
        won = text is not None and hmac.compare_digest(text.encode(), presented.encode())
        if not won and text is not None:
            try:
                os.link(claimed, path)    # fails if a newer secret is already in place: that one wins
            except OSError:
                pass
    finally:
        claimed.unlink(missing_ok=True)
    return won


# ── the fingerprint beside it ────────────────────────────────────────────────

def read_fingerprint(directory: Path | None = None) -> str | None:
    """The fingerprint in ``pair.fp``, or None if there is none or it cannot be trusted. Never raises."""
    directory = directory or pairing_dir()
    if not _directory_is_safe(directory):
        return None
    return _read_file(directory / FINGERPRINT_FILE, _FP_SHAPE)


def write_fingerprint(fp: str, directory: Path | None = None) -> bool:
    """Make ``pair.fp`` say ``fp``: what this daemon's ``GET /api/hello`` advertises. True if a file is in place.

    ``""`` (no instance key) removes an earlier file and creates nothing. Anything that is not a fingerprint is a
    ValueError; an untrustworthy directory is a PairingFileError, as for the secret.
    """
    directory = directory or pairing_dir()
    if not fp:
        if _directory_is_safe(directory):
            (directory / FINGERPRINT_FILE).unlink(missing_ok=True)
        return False
    if not _FP_SHAPE.fullmatch(fp):
        raise ValueError("not an instance fingerprint (16 lowercase hex characters)")
    _ensure_directory(directory)
    _write(directory, fp, FINGERPRINT_FILE)
    return True


def publish_fingerprint(fp: str, directory: Path | None = None) -> None:
    """The boot step, on the app install only: ``pair.fp`` says ``fp``. Never raises; a failure costs only the
    client's check before it sends the secret, and says so."""
    if not enabled():
        return
    try:
        written = write_fingerprint(fp, directory)
    except (PairingFileError, OSError, ValueError) as exc:
        logger.warning("could not write %s (%s); a client cannot check this daemon's fingerprint before pairing",
                       fingerprint_path(directory), exc)
        return
    if written:
        logger.info("pairing fingerprint written to %s", fingerprint_path(directory))
