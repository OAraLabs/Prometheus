"""A file tool opens exactly the path the gate checked, and nothing a link could put in its place.

WHY. The file tools (``write_file``, ``read_file``, ``edit_file`` and the rest) run inside the daemon, outside the
shell floors. The loop asks the gate about a tool's path, resolved; the tool used to resolve the argument again and
open it by name, following links. A folder in the workspace swapped for a link to a protected folder between the two
(by a shell or task running at the same time) sent the write or read to the protected file. A hard link to a protected
file inside the workspace passed the gate with no race at all.

WHAT.

* **The decided path.** The loop puts the path the gate ruled on, resolved, in the tool's context
  (``metadata["decided_path"]``). :func:`resolve_tool_path` resolves the tool's argument as before and refuses when
  the two differ: a link moved between the check and the tool.
* **No link is followed.** The path is resolved, so it has no links in it. :func:`_open_dir` walks it from ``/`` one
  component at a time with ``O_DIRECTORY | O_NOFOLLOW`` and ``dir_fd``, and the file itself is opened with
  ``O_NOFOLLOW``: a link that appears anywhere along it now fails the open (ELOOP, ENOTDIR) instead of being followed.
* **Writes replace.** :func:`write_bytes` writes an ``O_CREAT | O_EXCL | O_NOFOLLOW`` temporary file in the target's
  folder (by its fd), keeps an existing file's permission bits, and ``os.replace``-s it in with ``dir_fd``: a rename
  replaces a link rather than following it, and a reader never sees half a file. Missing folders are created one level
  at a time, never through a link.
* **Hard links.** A file with more than one link is refused when it is one of Prometheus's protected files
  (:func:`_protected_identities`), for reads and writes.

A link that already exists when the gate checks still works: the gate rules on its target, and the target is what is
opened. Only a path that CHANGES after the check is refused.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

#: Raised (as the tool's error) when the path is not the one the gate checked, or a link appeared along it.
class PathChanged(PermissionError):
    pass


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
_LINK_ERRNOS = {errno.ELOOP, errno.ENOTDIR, getattr(errno, "EMLINK", errno.ELOOP)}

#: Called between the decision and the open, with the path about to be opened. A no-op in production; tests replace it
#: to change the filesystem at exactly that moment.
def _no_op(path: Path) -> None:
    return None


before_open: Callable[[Path], None] = _no_op


def supported() -> bool:
    """Whether this platform can open by ``dir_fd`` without following links (macOS and Linux can)."""
    return bool(_NOFOLLOW and _DIRECTORY) and os.open in os.supports_dir_fd


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


def _changed(path: Path, exc: OSError) -> PathChanged:
    return PathChanged(f"{path}: a link appeared along this path after it was checked ({exc.strerror}); nothing was "
                       f"opened")


def _open_dir(path: Path, *, create: bool = False) -> int:
    """An fd on the directory *path* (absolute, resolved), reached without following a link. With *create*, missing
    components are made (mode 0o777, minus the umask), each in the fd of the one above it."""
    if not path.is_absolute():
        raise ValueError(f"not absolute: {path}")
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            try:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o777, dir_fd=fd)
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except OSError as exc:
        os.close(fd)
        if exc.errno in _LINK_ERRNOS:
            raise _changed(path, exc) from exc
        raise
    return fd


def _check_identity(path: Path, st: os.stat_result) -> None:
    if st.st_nlink > 1 and (st.st_dev, st.st_ino) in _protected_identities():
        raise PathChanged(f"{path} is another name (a hard link) for one of Prometheus's protected files; nothing was "
                          f"opened")


@contextlib.contextmanager
def open_read(path: Path) -> Iterator[int]:
    """An fd to read the regular file *path* (resolved), opened without following a link."""
    before_open(path)
    dfd = _open_dir(path.parent)
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=dfd)
        except OSError as exc:
            if exc.errno in _LINK_ERRNOS:
                raise _changed(path, exc) from exc
            raise
    finally:
        os.close(dfd)
    try:
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", str(path))
        _check_identity(path, st)
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
    except FileNotFoundError:
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
    tmp_name = f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    try:
        mode = 0o666 & ~_umask()
        try:
            st = os.stat(path.name, dir_fd=dfd, follow_symlinks=False)
        except FileNotFoundError:
            st = None
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise _changed(path, OSError(errno.ELOOP, "a link is where the file was"))
            if stat.S_ISDIR(st.st_mode):
                raise IsADirectoryError(errno.EISDIR, "Is a directory", str(path))
            _check_identity(path, st)
            mode = stat.S_IMODE(st.st_mode) & 0o777
        fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC, 0o600, dir_fd=dfd)
        try:
            with os.fdopen(fd, "wb") as fh:
                yield fh
                fh.flush()
                os.fchmod(fh.fileno(), mode)
                os.fsync(fh.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name, dir_fd=dfd)
            raise
        os.replace(tmp_name, path.name, src_dir_fd=dfd, dst_dir_fd=dfd)
    finally:
        os.close(dfd)


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


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


# ── the protected files, by identity ────────────────────────────────────────

#: A folder under the floor holding more files than this is not walked for hard links: the install trees (venvs, a
#: deploy clone) and the checkpoint blobs. The kernel floors refuse a shell's `link` into them.
_WALK_CAP = 2000


def _protected_roots() -> list[Path]:
    """Every existing path the floor names for this process: the read-and-write floor, the write floor and what
    starts Prometheus, with ``/*/`` read as this user's home."""
    from prometheus.permissions import checker as C

    home = str(Path.home().resolve())
    out: list[Path] = []
    entries = [*C._ALWAYS_DENIED_PATHS, *C._ALWAYS_WRITE_DENIED_PATHS, *C._STARTUP_PATHS,
               *C._this_process_state(writes=False), *C._this_process_state(writes=True),
               *C._this_process_startup()]
    try:
        from prometheus.permissions.confinement import TRUSTED_DIRS, TRUSTED_FILES, loaded_configs
        from prometheus.config.paths import config_dir_path

        cfg = config_dir_path().expanduser().resolve()
        entries += [str(cfg / n) for n in (*TRUSTED_FILES, *TRUSTED_DIRS)]
        entries += list(loaded_configs())
    except Exception:  # noqa: BLE001 — the identity set is defence in depth; the floor lists above are enough
        pass
    import glob as _glob

    for entry in entries:
        pattern = entry.replace("/*/", home + "/", 1) if entry.startswith("/*/") else entry
        for hit in _glob.glob(pattern):
            out.append(Path(hit))
    return out


def _protected_identities() -> set[tuple[int, int]]:
    """``(st_dev, st_ino)`` of every protected file that exists now. Computed per call: it is only asked for a file
    with more than one link, which is rare."""
    ids: set[tuple[int, int]] = set()
    for root in _protected_roots():
        try:
            st = os.lstat(root)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            ids.add((st.st_dev, st.st_ino))
        elif stat.S_ISDIR(st.st_mode):
            seen = 0
            for dirpath, _dirs, files in os.walk(root, followlinks=False):
                for name in files:
                    seen += 1
                    if seen > _WALK_CAP:
                        break
                    with contextlib.suppress(OSError):
                        fst = os.lstat(os.path.join(dirpath, name))
                        if stat.S_ISREG(fst.st_mode):
                            ids.add((fst.st_dev, fst.st_ino))
                if seen > _WALK_CAP:
                    break
    return ids
