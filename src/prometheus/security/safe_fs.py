"""A file tool opens exactly the path the gate checked, and nothing a link could put in its place.

WHY. The file tools (``write_file``, ``read_file``, ``edit_file`` and the rest) run inside the daemon, outside the
shell floors. The loop asks the gate about a tool's path, resolved; the tool used to resolve the argument again and
open it by name, following links. A folder in the workspace swapped for a link to a protected folder between the two
(by a shell or task running at the same time) sent the write or read to the protected file. A hard link to a protected
file inside the workspace passed the gate with no race at all (GHSA-wgcv-xw9m-9q9q).

WHAT.

* **The decided path.** The loop puts the path the gate ruled on in the tool's context (``metadata["decided_path"]``).
  :func:`resolve_tool_path` resolves the tool's argument as before and refuses when the two differ: a link moved
  between the check and the tool. A tool with a long wait before it writes (a download, an image or speech engine)
  resolves once at the start and keeps that :class:`~pathlib.Path`.
* **No link is followed.** The path is resolved, so it has no links in it. :func:`_open_dir` walks it from ``/`` one
  component at a time with ``O_NOFOLLOW`` and ``dir_fd`` (``O_SEARCH`` or ``O_PATH`` where the platform has one, so a
  folder that is only searchable still works), and the file itself is opened with ``O_NOFOLLOW``: a link that
  appears anywhere along it fails the open instead of being followed.
* **Writes replace.** :func:`atomic_writer` refuses an existing file this user may not write (as an open for writing
  would), writes an ``O_CREAT | O_EXCL | O_NOFOLLOW`` temporary file in the target's folder (by its fd), keeps an
  existing file's permission bits, and ``os.replace``-s it in with ``dir_fd``: a rename replaces a link rather than
  following it, and a reader never sees half a file. Where a rename cannot replace the file (a single file
  bind-mounted into a container answers EBUSY), it is rewritten in place through an ``O_NOFOLLOW`` fd instead.
  Missing folders are created one level at a time, never through a link.
* **Hard links, for reads.** A file with more than one link is not read when it is one of the files the gate refuses
  to read (:func:`register_read_floor`, which the gate calls with its own list). Writes need no such check: a
  replace puts a new file under the name and never writes through the other names' inode.

A link that already exists when the gate checks still works: the gate rules on its target, and the target is what is
opened. Only a path that CHANGES after the check is refused.
"""

from __future__ import annotations

import contextlib
import errno
import glob as _glob
import os
import stat
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any


class PathChanged(PermissionError):
    """The path is not the one the gate checked, or a link appeared along it after the check. Nothing was opened."""


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
#: Intermediate folders need only search permission: ``O_SEARCH`` (macOS) or ``O_PATH`` (Linux) where available.
_SEARCH = getattr(os, "O_SEARCH", 0) or getattr(os, "O_PATH", 0) or os.O_RDONLY
_DIR_FLAGS = _SEARCH | _DIRECTORY | _NOFOLLOW | _CLOEXEC


def _no_op(path: Path) -> None:
    return None


#: Called between the decision and the open, with the path about to be opened. A no-op in production; tests replace it
#: to change the filesystem at exactly that moment.
before_open: Callable[[Path], None] = _no_op


def resolve_tool_path(candidate: str, base: Path, metadata: dict[str, Any] | None = None) -> Path:
    """The tool's argument resolved as it always was (``~``, relative to *base*), checked against the gate's decision.

    When the loop recorded the path the gate ruled on (``metadata["decided_path"]``), a different resolution means a
    link along the path changed after the check, and the call is refused.
    """
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    decided = (metadata or {}).get("decided_path")
    if decided is not None and Path(decided) != resolved:
        raise PathChanged(
            f"{candidate} now resolves to {resolved}, not {decided}, which is the path that was checked: a link along "
            f"it changed after the check, so nothing was opened")
    return resolved


def _refusal(path: Path, at: int, exc: OSError) -> OSError:
    """What a failed step of the walk means: a link where the checked path had a folder (refused), or an ordinary
    error (a file where a folder should be, no permission), which is raised as it is."""
    if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
        partial = Path("/", *path.parts[1:at + 1])
        with contextlib.suppress(OSError):
            if stat.S_ISLNK(os.lstat(partial).st_mode):
                return PathChanged(f"{path}: {partial} became a link after the path was checked; nothing was opened")
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            return PathChanged(f"{path}: a link appeared along this path after it was checked; nothing was opened")
    return exc


def _open_dir(path: Path, *, create: bool = False) -> int:
    """An fd on the directory *path* (absolute, resolved), reached without following a link. With *create*, missing
    components are made (mode 0o777 less the umask), each in the fd of the one above it."""
    if not path.is_absolute():
        raise ValueError(f"not absolute: {path}")
    fd = os.open("/", _DIR_FLAGS)
    parts = path.parts[1:]
    try:
        for at, part in enumerate(parts, start=1):
            try:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o777, dir_fd=fd)
                try:
                    nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
                except OSError as exc:
                    raise _refusal(path, at, exc) from exc
            except OSError as exc:
                raise _refusal(path, at, exc) from exc
            os.close(fd)
            fd = nxt
    except BaseException:
        os.close(fd)
        raise
    return fd


def _leaf_refusal(path: Path, exc: OSError) -> OSError:
    if exc.errno in (errno.ELOOP, errno.EMLINK):
        return PathChanged(f"{path} became a link after it was checked; nothing was opened")
    return exc


@contextlib.contextmanager
def open_read(path: Path) -> Iterator[int]:
    """An fd to read the regular file *path* (resolved), opened without following a link."""
    before_open(path)
    dfd = _open_dir(path.parent)
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC | getattr(os, "O_NONBLOCK", 0), dir_fd=dfd)
        except OSError as exc:
            raise _leaf_refusal(path, exc) from exc
    finally:
        os.close(dfd)
    try:
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", str(path))
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        if st.st_nlink > 1 and (st.st_dev, st.st_ino) in _read_floor_identities():
            raise PathChanged(f"{path} is another name (a hard link) for a file Prometheus never reads for a tool; "
                              f"nothing was read")
        yield fd
    finally:
        os.close(fd)


def read_bytes(path: Path) -> bytes:
    with open_read(path) as fd, os.fdopen(os.dup(fd), "rb") as fh:
        return fh.read()


def read_text(path: Path, encoding: str = "utf-8", errors: str | None = None) -> str:
    return read_bytes(path).decode(encoding, errors or "strict")


def exists(path: Path) -> bool:
    """Whether *path* names something now, without following a link at its end."""
    try:
        dfd = _open_dir(path.parent)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except PathChanged:
        return True   # something is there, through a link: the open that follows refuses it with the reason
    try:
        os.stat(path.name, dir_fd=dfd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    finally:
        os.close(dfd)


@contextlib.contextmanager
def atomic_writer(path: Path, *, create_parents: bool = True) -> Iterator[Any]:
    """A binary file object; on a clean exit its bytes replace *path* (resolved) atomically, through no link."""
    before_open(path)
    dfd = _open_dir(path.parent, create=create_parents)
    tmp_name = f".{path.name[:200]}.{os.urandom(4).hex()}.tmp"
    tmp_made = False
    try:
        try:
            st: os.stat_result | None = os.stat(path.name, dir_fd=dfd, follow_symlinks=False)
        except FileNotFoundError:
            st = None
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise PathChanged(f"{path} became a link after it was checked; nothing was written")
            if stat.S_ISDIR(st.st_mode):
                raise IsADirectoryError(errno.EISDIR, "Is a directory", str(path))
            # As an open for writing would: a file this user may not write is not replaced just because its folder is
            # writable.
            if not os.access(path.name, os.W_OK, dir_fd=dfd, follow_symlinks=False):
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
        # 0o666 less the umask, as the kernel applies it; an existing file's own bits are kept below.
        fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC, 0o666, dir_fd=dfd)
        tmp_made = True
        with os.fdopen(fd, "wb") as fh:
            yield fh
            fh.flush()
            if st is not None:
                os.fchmod(fh.fileno(), stat.S_IMODE(st.st_mode) & 0o777)
        try:
            os.replace(tmp_name, path.name, src_dir_fd=dfd, dst_dir_fd=dfd)
            tmp_made = False
        except OSError as exc:
            if exc.errno not in (errno.EBUSY, errno.EXDEV) or st is None:
                raise
            _rewrite_in_place(dfd, tmp_name, path)
    finally:
        if tmp_made:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name, dir_fd=dfd)
        os.close(dfd)


def _rewrite_in_place(dfd: int, tmp_name: str, path: Path) -> None:
    """A file that cannot be replaced (a bind mount): copy the new bytes over it, through an fd opened without
    following a link."""
    try:
        out = os.open(path.name, os.O_WRONLY | os.O_TRUNC | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)
    except OSError as exc:
        raise _leaf_refusal(path, exc) from exc
    src = os.open(tmp_name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)
    with os.fdopen(src, "rb") as fin, os.fdopen(out, "wb") as fout:
        while chunk := fin.read(1 << 20):
            fout.write(chunk)


def write_bytes(path: Path, data: bytes, *, create_parents: bool = True) -> None:
    with atomic_writer(path, create_parents=create_parents) as fh:
        fh.write(data)


def write_text(path: Path, text: str, encoding: str = "utf-8", *, create_parents: bool = True) -> None:
    write_bytes(path, text.encode(encoding), create_parents=create_parents)


def place_file(src: Path, dest: Path, *, create_parents: bool = True) -> None:
    """Copy the file another program wrote at *src* (in a private temp folder) into *dest*, through no link."""
    with open(src, "rb") as fin, atomic_writer(dest, create_parents=create_parents) as fout:
        while chunk := fin.read(1 << 20):
            fout.write(chunk)


# ── the read floor, by identity ──────────────────────────────────────────────

#: The gate's read floor, as entries (globs and literals). Each gate registers its own list when it is built, so the
#: configured ``denied_paths`` and the directories this process moved are covered exactly as the gate covers them.
_read_floor: list[str] = []

#: A folder under the read floor holding more files than this is not walked further (none of the shipped ones does).
_WALK_CAP = 5000
_IDENTITY_TTL = 5.0
_identity_cache: tuple[float, tuple[str, ...], set[tuple[int, int]]] | None = None


def register_read_floor(entries: Iterable[str]) -> None:
    """Add the gate's read floor (``SecurityGate._denied_paths``) to the files a hard link may not reach."""
    global _identity_cache
    for entry in entries:
        if entry not in _read_floor:
            _read_floor.append(entry)
            _identity_cache = None


def _read_floor_entries() -> list[str]:
    """What every gate built in this process refuses to read, and this process's own floor as the grep prune reads
    it (which needs no gate: a script, a coding run's child)."""
    from prometheus.tools.denied_prune import resolve_denied

    return list(dict.fromkeys([*_read_floor, *resolve_denied(None)]))


def _read_floor_identities() -> set[tuple[int, int]]:
    """``(st_dev, st_ino)`` of every file under the read floor that exists now. Only asked for a file with more than
    one link; kept for a few seconds, so a grep over a uv venv (hard-linked from its cache) walks the floor once."""
    global _identity_cache
    entries = tuple(_read_floor_entries())
    now = time.monotonic()
    if _identity_cache is not None and _identity_cache[1] == entries and now - _identity_cache[0] < _IDENTITY_TTL:
        return _identity_cache[2]
    ids = _walk_identities(entries)
    _identity_cache = (now, entries, ids)
    return ids


def _walk_identities(entries: Iterable[str]) -> set[tuple[int, int]]:
    home = str(Path.home().resolve())
    ids: set[tuple[int, int]] = set()
    for entry in entries:
        pattern = entry.replace("/*/", home + "/", 1) if entry.startswith("/*/") else entry
        for hit in _glob.glob(pattern):
            try:
                st = os.lstat(hit)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                ids.add((st.st_dev, st.st_ino))
            elif stat.S_ISDIR(st.st_mode):
                seen = 0
                for dirpath, _dirs, files in os.walk(hit, followlinks=False):
                    for name in files:
                        seen += 1
                        with contextlib.suppress(OSError):
                            fst = os.lstat(os.path.join(dirpath, name))
                            if stat.S_ISREG(fst.st_mode):
                                ids.add((fst.st_dev, fst.st_ino))
                    if seen > _WALK_CAP:
                        break
    return ids
