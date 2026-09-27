"""How skill files are written: never over one, never half, archived before replaced.

Shared by the writers of ``skills/auto/``: ``SkillCreator.persist_skill_content``
(the auto path, record-a-skill, an accepted draft, teacher escalation) and GEPA's
promotion. One convention, so an accept that replaces a live skill archives it
exactly the way a GEPA promotion does.

- :func:`create_exclusive` makes a NEW file (``O_CREAT | O_EXCL``, as the memory
  store's snapshots do since #594/#601): a name that is taken raises
  ``FileExistsError`` instead of being overwritten.
- :func:`atomic_write` replaces a file in one step (a temp file, then
  ``os.replace``), so a reader never sees half of it. For a deliberate
  replacement only.
- :func:`archive_copy` keeps the version being replaced, as
  ``archive/<stem>_<unixtime>[-N].md``, created exclusively: an archive is never
  overwritten, even by two replacements in one second.

Temp files never end in ``.md``: the loader serves every ``*.md`` in
``skills/auto/``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


def create_exclusive(path: Path, text: str) -> None:
    """Write *text* to a NEW file at *path*; ``FileExistsError`` if the name is taken.

    A failed write removes what it created, so nothing half-written is left
    where the loader would serve it.
    """
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    try:
        with fh:
            fh.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def atomic_write(path: Path, text: str) -> None:
    """Replace *path* with *text* in one step: a temp file beside it, then ``os.replace``."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def archive_copy(path: Path, archive_dir: Path, *, text: str | None = None) -> Path:
    """Keep *path*'s content (or *text*) as ``<stem>_<unixtime>[-N].md`` in *archive_dir*.

    Created exclusively: a taken name moves to the next free ``-N``, so two
    archives of one skill in one second both survive (the #594 collision).
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    body = path.read_text(encoding="utf-8") if text is None else text
    stamp = int(time.time())
    for n in range(1, 1000):
        dst = archive_dir / f"{path.stem}_{stamp}{'' if n == 1 else f'-{n}'}.md"
        try:
            create_exclusive(dst, body)
            return dst
        except FileExistsError:
            continue
    raise FileExistsError(f"no free archive name for {path.name} in {archive_dir}")
