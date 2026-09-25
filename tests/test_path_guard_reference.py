"""The shipped deny-list matcher decides exactly what a plain one decides.

``path_guard.denying_entry`` is fast because it screens: one C-level test per
pass (a ``startswith`` tuple, one combined ``fnmatch`` regex behind a keyword
search), a skip when nothing needs folding, and a folded-string filter in
front of the identity pass. Every one of those is an optimisation that must
never change a decision: a screen that says "no" where the full walk would
have denied is a hole in the deny list, and it would look like a speed-up.

So this file keeps the decision written out plainly — every pass a simple
loop, identity asked of every absolute entry and every spelling, no screen,
no combined regex, no fast path — and requires the shipped matcher to agree
with it, entry for entry, on a fixed corpus: case variants, NFC and NFD
names, symlinks, glob-looking names, absent directories, firmlink spellings
where the host has them; against the shipped deny list and a large one.

If you change the prefilter or a screen and this fails, the screen skipped a
path the full matcher denies (or denied one it doesn't). Fix the screen; do
not edit the reference to match it.
"""

from __future__ import annotations

import fnmatch
import unicodedata
from pathlib import Path

import pytest

from prometheus.config.shipped_defaults import SHIPPED_DENIED_PATHS
from prometheus.security import path_guard as pg

ALWAYS_DENIED = ("/*/.ssh", "/*/.gnupg", "/*/.config/*/*env")


# ── the reference: the decision, plainly ────────────────────────────────────

def _fold(text: str) -> str:
    return "/".join(pg.fold_component(part) for part in text.split("/"))


def _under(text: str, entry: str) -> bool:
    return text == entry or text.startswith(entry.rstrip("/") + "/")


def _rest_matches(rel: str, rest: str) -> bool:
    return fnmatch.fnmatch(rel, rest) or fnmatch.fnmatch(rel, rest.rstrip("/") + "/*")


def _nearest_existing_folds_case(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if pg.file_identity(candidate) is not None:
            return pg.volume_folds_case(candidate)
    return True


def reference_denying_entry(resolved_path, entries) -> str | None:
    """Every pass of ``denying_entry``, as simple loops over everything."""
    text = str(resolved_path)
    path = Path(text)
    entries = [e for e in entries if e]
    for entry in entries:                                   # 1. as given
        if pg.denied_entry_matches(text, entry):
            return entry
    for entry in entries:                                   # 2. a glob as a literal path
        if pg.is_glob_pattern(entry) and _under(text, entry):
            return entry
    for entry in entries:                                   # 3. the other spellings
        for spelling in pg.entry_spellings(entry)[1:]:
            if pg.denied_entry_matches(text, spelling) or (
                    pg.is_glob_pattern(spelling) and _under(text, spelling)):
                return entry
    literal: dict[Path, str] = {}                           # 4. identity, unfiltered
    globs: list[tuple[str, str]] = []
    for entry in entries:
        if not Path(entry).is_absolute():
            continue
        for spelling in pg.entry_spellings(entry):
            if pg.is_glob_pattern(spelling):
                globs.append((entry, spelling))
            literal.setdefault(Path(spelling), entry)
    hit = pg.inside_by_identity(path, literal, absent_only_where_case_folds=True)
    if hit is not None:
        return literal[hit]
    for entry, spelling in globs:
        prefix, rest = pg.split_literal_prefix(spelling)
        key = pg.file_identity(prefix)
        if key is None:
            continue
        for ancestor in path.parents:
            if pg.file_identity(ancestor) != key:
                continue
            rel = path.relative_to(ancestor).as_posix()
            if _rest_matches(rel, rest):
                return entry
            if _nearest_existing_folds_case(path) and _rest_matches(_fold(rel), _fold(rest)):
                return entry
            respelled = pg._as_the_pattern_spells_it(ancestor, rel, rest)
            if respelled != rel and _rest_matches(respelled, rest):
                return entry
    return None


# ── the corpus ──────────────────────────────────────────────────────────────

NFC = unicodedata.normalize("NFC", "café")
NFD = unicodedata.normalize("NFD", "café")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    home = root / "home"
    monkeypatch.setenv("HOME", str(home))
    files = [
        "home/.ssh/id_rsa", "home/.gnupg/pubring.kbx", "home/.config/app/.env",
        "home/.config/app/settings.json", "home/.config/tool/xenv", "home/Documents/notes.txt",
        "real/secret/key", "real/public/readme", "[old]/f.md", "Photos [2024]/p.jpg",
        "keys/a.pem", "keys/b.txt", "data/x", f"{NFC}/menu", "ws/build/out.o",
        "nested/deep/a/b/c.txt", "star*dir/q", "q?a/z",
    ]
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n")
    (root / "link").symlink_to(root / "real", target_is_directory=True)
    (root / "oldlink").symlink_to(root / "[old]", target_is_directory=True)
    (root / "homelink").symlink_to(home, target_is_directory=True)

    paths: set[str] = set()
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix()
        paths.add(rel)
        parts = rel.split("/")
        for i, part in enumerate(parts):  # a case variant of each component in turn
            for variant in {part.upper(), part.swapcase(), part.title()} - {part}:
                paths.add("/".join(parts[:i] + [variant] + parts[i + 1:]))
    paths |= {
        "link/secret/key", "link/SECRET/new", "oldlink/f.md", "OLDLINK/F.MD",
        f"{NFD}/menu", f"{NFD.upper()}/menu",
        "home/.SSH/new", "home/.Ssh/id_rsa", "home/nope/.ssh/k", "home/.config/new-app/.ENV",
        "homelink/.ssh/id_rsa", "homelink/.CONFIG/app/.env",
        "absent/dir/file", "Data/y", "DATA/x", "keys/new.PEM", "[OLD]/f.md", "photos [2024]/p.jpg",
    }
    candidates = {str(Path(root / p).resolve()) for p in paths}
    candidates |= {str(Path(p).resolve()) for p in (
        "/etc/hosts", "/private/etc/hosts", "/tmp/x.refcorpus", "/private/tmp/y.REFCORPUS",
        "/System/Volumes/Data/private/etc/hosts", "/system/volumes/data/private/etc/hosts")}
    return root, home, sorted(candidates)


def _large_list(root: Path) -> list[str]:
    entries = list(SHIPPED_DENIED_PATHS) + list(ALWAYS_DENIED) + [
        str(root / "link" / "secret"), f"{root}/link/*.key", str(root / "[old]"),
        str(root / "oldlink"), f"{root}/keys/*.pem", str(root / "Data"), str(root / NFC),
        str(root / NFD), "~/Documents", "/tmp/*.refcorpus", f"{root}/nested/*/a/*/c.txt",
        str(root / "Photos [2024]"), str(root / "star*dir"), str(root / "absent"),
    ]
    for i in range(12):
        entries += [f"/opt/none{i}", f"~/.cache/*/tok{i}", f"{root}/filler{i}/*.bin"]
    return [str(Path(e).expanduser()) if e.startswith("~") else e for e in entries]


@pytest.mark.parametrize("which", ["shipped", "large"])
def test_the_shipped_matcher_decides_exactly_what_the_plain_one_decides(corpus, which):
    root, _, candidates = corpus
    entries = (list(SHIPPED_DENIED_PATHS) + list(ALWAYS_DENIED)) if which == "shipped" \
        else _large_list(root)

    mismatches = []
    denied = 0
    for text in candidates:
        want = reference_denying_entry(text, entries)
        got = pg.denying_entry(text, entries)
        denied += want is not None
        if got != want:
            mismatches.append(f"{text}: shipped {got!r}, reference {want!r}")

    assert not mismatches, (
        f"{len(mismatches)} of {len(candidates)} decisions differ — a screen skipped "
        "(or added) what the full matcher decides:\n  " + "\n  ".join(mismatches[:20]))
    assert len(candidates) > 150
    assert denied >= (8 if which == "shipped" else 30), f"corpus too thin: {denied} denied"


def test_the_corpus_reaches_the_identity_pass_where_the_volume_folds(corpus):
    """Not vacuous: where the volume folds case, some corpus paths are denied
    only by identity (no spelling of any entry names them), so the pass the
    screens guard is really exercised. Where it doesn't (ext4), identity adds
    nothing once a path is resolved, and none may be denied that way."""
    root, home, candidates = corpus
    entries = _large_list(root)
    identity_only = [
        t for t in candidates
        if reference_denying_entry(t, entries) is not None
        and not any(pg.denied_entry_matches(t, s) or _under(t, s)
                    for e in entries for s in pg.entry_spellings(e))
    ]
    if _probe_folds(root):
        assert len(identity_only) >= 5, identity_only
    else:
        assert identity_only == []


def _probe_folds(directory: Path) -> bool:
    probe = directory / "FoldProbe"
    probe.mkdir()
    try:
        return (directory / "foldprobe").exists()
    finally:
        probe.rmdir()
