"""path_guard — shared write-boundary helper.

A single-function utility for asserting that a candidate path resolves
under one of an allow-listed set of roots. Used by autonomous components
that should not be able to write to arbitrary paths (MemoryExtractor's
ObsidianWriter, future self-improvement writers, etc.).

The path is resolved BEFORE the prefix check, so a ``../`` traversal that
lands outside the allow-list is rejected even when the literal input
string starts with an allowed prefix. This mirrors the GraftEngine pattern
documented in PROMETHEUS.md ("Path Traversal Defense").

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import fnmatch
import functools
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, NamedTuple


def assert_path_under_roots(
    candidate: str | Path,
    allowed_roots: Iterable[str | Path],
) -> Path:
    """Resolve ``candidate`` and verify it lives under one of ``allowed_roots``.

    Returns the resolved path on success. Raises ``ValueError`` if the
    resolved path is not under any allowed root, or if the candidate
    cannot be resolved at all.

    Both the candidate and each root are expanduser+resolve'd before
    comparison, so the call is robust to ``~`` and ``../`` in the input.
    A path equal to a root counts as "under" the root.

    Example:
        >>> assert_path_under_roots(
        ...     "~/.prometheus/wiki/people/alice.md",
        ...     [Path.home() / ".prometheus"],
        ... )
        PosixPath('/home/.../.prometheus/wiki/people/alice.md')
    """
    try:
        target = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve candidate path {candidate!r}: {exc}")

    resolved_roots: list[Path] = []
    for root in allowed_roots:
        try:
            resolved_roots.append(Path(root).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    if not resolved_roots:
        raise ValueError("allowed_roots is empty after resolution")

    for root in resolved_roots:
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target

    roots_str = ", ".join(str(r) for r in resolved_roots)
    raise ValueError(
        f"Path {target} is not under any allowed root ({roots_str})"
    )


def is_path_under_roots(
    candidate: str | Path,
    allowed_roots: Iterable[str | Path],
) -> bool:
    """Boolean variant of :func:`assert_path_under_roots`. Never raises."""
    try:
        assert_path_under_roots(candidate, allowed_roots)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Denied-path matching — ONE matcher, used by every layer that reads the list.
# ---------------------------------------------------------------------------

GLOB_CHARS = "*?["


def is_glob_pattern(entry: str) -> bool:
    """True when a ``denied_paths`` entry carries a wildcard."""
    return any(c in entry for c in GLOB_CHARS)


def denied_entry_matches(resolved_path: str, entry: str) -> bool:
    """Whether an absolute, already-resolved path is denied by ONE entry.

    THE ONE MATCHER. Two layers read ``security.denied_paths`` — the SecurityGate
    (which refuses a denied ROOT) and the grep/glob prune layer (which withholds
    denied paths from a legitimate root that contains them). They used to match
    with DIFFERENT semantics:

      * the gate used ``fnmatch``, where ``*`` spans ``/``;
      * the prune layer expanded entries with ``Path.glob``, where ``*`` matches
        ONE component.

    So the shipped credential floor ``/*/.ssh`` matched ``/home/<user>/.ssh`` at
    the gate and resolved to NOTHING in the prune layer — ``Path('/').glob
    ('*/.ssh')`` is ``[]`` because it only matches at depth one. Measured on a
    real box: ``resolve_denied(SHIPPED_DENIED_PATHS)`` returned
    ``(/etc, /sys, /boot)`` and ``is_denied(~/.ssh/id_rsa)`` was False, while the
    gate denied the same path. ``grep --root ~ 'PRIVATE KEY'`` then returned the
    key lines with no ``[N paths withheld]`` note, so the model read the result as
    complete. That is the #214 guarantee, silently absent.

    ``fnmatch``'s ``*`` spanning ``/`` is broader than a shell glob — deliberately:
    broader means MORE denied, and this is a deny list.

    A glob entry matches the path itself OR anything under it, so denying a
    directory denies its subtree without the entry having to say so. A literal
    entry matches on PATH COMPONENTS, not raw string prefix: a bare
    ``startswith`` denied ``/etcetera/notes`` for the entry ``/etc``, which is
    over-refusal that never announced itself.
    """
    if is_glob_pattern(entry):
        if fnmatch.fnmatch(resolved_path, entry):
            return True
        return fnmatch.fnmatch(resolved_path, entry.rstrip("/") + "/*")
    return (
        resolved_path == entry
        or resolved_path.startswith(entry.rstrip("/") + os.sep)
    )


def matches_any_denied(path: str | Path, entries: Iterable[str]) -> bool:
    """Whether *path* is denied by ANY entry.

    Resolves the path FIRST (so ``../`` cannot step around an entry) and fails
    CLOSED on a path that cannot be resolved — a broken symlink, a loop, or a
    path that is simply not there is not something to hand back from inside a
    search that may be rooted anywhere.

    ⚠ ``strict=True`` IS THE GUARANTEE, NOT A DETAIL. This read
    ``.resolve()`` — non-strict — and the fail-closed promise above was
    therefore delivered by accident and only for one of the three shapes it
    names. Non-strict resolve raises for a symlink CYCLE but returns the path
    unchanged for a DANGLING symlink and for a path that does not exist, so
    the ``except`` never fired for either and the guard answered on the
    path's SPELLING. Measured on 3.12.3, the version we ship: a dangling
    symlink and a nonexistent path both came back allowed. 3.13 then removed
    the last case that raised, which is how CI found it — but it was never a
    3.13 defect; 3.13 only took away the accident.

    Safe to be strict, and checked before relying on it: every caller reaches
    here through ``denied_prune.is_denied``, whose only two call sites prune
    the RESULTS of a filesystem walk (``glob.py`` filters ``root.glob(...)``;
    ``grep.py`` filters the same after ``is_file()`` has already passed). No
    caller asks about a path that has yet to be created, so nothing here can
    refuse a write target. A path arriving here that does not exist means
    reality disagrees with the caller — a file removed mid-scan — which is
    exactly when a boundary should refuse rather than guess.
    """
    try:
        resolved = str(Path(path).expanduser().resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return True
    return denying_entry(resolved, entries) is not None


# ---------------------------------------------------------------------------
# Path identity: compare what a path IS, not how it is spelled (WP-X.23/27).
# ---------------------------------------------------------------------------
#
# A guard that resolves the path it is asked about but not the entry it
# compares against never fires where the two spellings differ: on macOS
# ``/etc``, ``/tmp`` and ``/var`` are symlinks into ``/private``, and on any
# host a ``~`` entry or a configured path can run through a symlink. And
# ``resolve()`` folds neither case nor firmlinks, so on a case-insensitive
# volume ``~/.SSH`` IS ``~/.ssh`` while no string says so. The download
# tool's guard was fixed this way first (#574); these are its helpers, shared.


def fold_component(name: str) -> str:
    """A path component as a case-insensitive, normalisation-insensitive
    volume compares it (APFS folds case, and ``ſ`` as ``s``)."""
    return unicodedata.normalize("NFC", name).casefold()


def file_identity(path: str | Path, *, follow_symlinks: bool = True) -> tuple[int, int] | None:
    """``(device, inode)`` of what ``path`` names, or None if it can't be stat'ed."""
    try:
        st = os.stat(path, follow_symlinks=follow_symlinks)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino)


#: ``_PC_CASE_SENSITIVE`` in Darwin's <sys/unistd.h>; Python has no name for it.
_DARWIN_PC_CASE_SENSITIVE = 11


def volume_folds_case(directory: str | Path) -> bool:
    """Whether the volume holding ``directory`` treats names differing only
    in case as one name.

    Asked of that path's own volume on every call, never once at start: a
    Mac can mount a case-sensitive disk next to its case-insensitive one.
    FAILS CLOSED: on macOS, an answer the volume can't give (the path is
    gone, no permission, any value but 0 or 1) counts as folding, which only
    ever denies more. Linux has no such query; its filesystems here (ext4)
    are case-sensitive, and that is what it answers.
    """
    if sys.platform != "darwin":
        return False
    try:
        answer = os.pathconf(directory, _DARWIN_PC_CASE_SENSITIVE)
    except (OSError, ValueError):
        return True
    return answer != 1  # 1: case-sensitive; 0: folds; anything else: can't tell


def inside_by_identity(
    candidate: Path,
    prefixes: Iterable[Path],
    *,
    absent_only_where_case_folds: bool = False,
) -> Path | None:
    """The protected directory ``candidate`` is inside, by file identity.

    ``resolve()`` follows symlinks but folds neither case nor firmlinks. On
    macOS's case-insensitive volumes ``~/.SSH`` IS ``~/.ssh``, and
    ``/System/Volumes/Data/private/etc`` IS ``/private/etc``; no string
    comparison sees that. So each existing ancestor of the destination is
    compared with each protected directory by (device, inode).

    A protected directory that does not exist yet (a fresh account's
    ``~/.ssh``) has no identity to compare, and writing ``~/.SSH/...`` would
    CREATE it, as the directory sshd then reads as ``~/.ssh``. So for those,
    the ancestor that IS its parent (by identity) is found, and the next
    component of the destination is compared with the protected name folded
    for case. That refuses ``~/.SSH`` on a case-sensitive volume too, where
    it is merely a confusing name: the download guard's choice (#574).
    ``absent_only_where_case_folds`` keeps that comparison to volumes that
    fold case, for readers of a configured deny list, where on a
    case-sensitive volume the other name is a different, existing directory
    the operator never denied (WP-X.27).
    """
    existing: dict[tuple[int, int], Path] = {}
    absent: dict[tuple[int, int], list[Path]] = {}
    for prefix in prefixes:
        key = file_identity(prefix)
        if key is not None:
            existing.setdefault(key, prefix)
            continue
        parent = file_identity(prefix.parent)
        if parent is None:
            continue  # neither it nor its parent exists (/proc on macOS)
        absent.setdefault(parent, []).append(prefix)
    chain = (candidate, *candidate.parents)
    for i, ancestor in enumerate(chain):
        key = file_identity(ancestor)
        if key is None:
            continue  # not created yet, or not reachable: its parents still are
        hit = existing.get(key)
        if hit is not None:
            return hit
        if i > 0 and key in absent:  # the component of the destination directly below `ancestor`
            if absent_only_where_case_folds and not volume_folds_case(ancestor):
                continue
            below = fold_component(chain[i - 1].name)
            for prefix in absent[key]:
                if below == fold_component(prefix.name):
                    return prefix
    return None


def split_literal_prefix(entry: str) -> tuple[str, str]:
    """``entry``'s leading directories that hold no wildcard, and the rest.

    ``/tmp/*.secret`` -> ``("/tmp", "*.secret")``; ``/*/.ssh`` ->
    ``("/", "*/.ssh")``; a literal entry is all prefix: ``("/etc", "")``.
    """
    if not is_glob_pattern(entry):
        return entry, ""
    parts = entry.split("/")
    cut = next(i for i, part in enumerate(parts) if is_glob_pattern(part))
    prefix = "/".join(parts[:cut]) or "/"
    return prefix, "/".join(parts[cut:])


@functools.lru_cache(maxsize=1024)
def entry_spellings(entry: str) -> tuple[str, ...]:
    """A deny-list entry as written, and with its literal directories resolved.

    ``/etc`` -> ``("/etc", "/private/etc")`` on macOS; ``/tmp/*.secret`` ->
    ``("/tmp/*.secret", "/private/tmp/*.secret")``. Only the literal part is
    resolved: ``Path.resolve()`` on a wildcard would treat it as a name. A
    relative entry is returned as it is and never resolved, because resolving
    it against the daemon's working directory is the defect
    ``checker._normalise_denied_path`` refuses to start on.

    Cached for the life of the process, as the gate already resolves its
    literal entries once at start: a symlink in an entry that is repointed
    later keeps its old target denied too (never less denied) until restart.
    """
    prefix, rest = split_literal_prefix(entry)
    if not Path(prefix).is_absolute():
        return (entry,)
    try:
        resolved = str(Path(prefix).resolve())
    except (OSError, RuntimeError, ValueError):
        return (entry,)
    other = resolved if not rest else resolved.rstrip("/") + "/" + rest
    return (entry,) if other == entry else (entry, other)


#: The Data volume's own spelling of a firmlinked directory on macOS
#: (``/System/Volumes/Data/private/etc`` IS ``/private/etc``), folded: the
#: volume folds the prefix's case too, so it is stripped after folding.
_DATA_VOLUME = "/system/volumes/data/"


def _loose(text: str) -> str:
    """A path as a case- and normalisation-folding volume might equate it,
    firmlinks included. Only a filter: identity decides."""
    if text.isascii():  # NFC is the identity and casefold is lower() for ASCII
        folded = text.lower()
    else:
        folded = "/".join(fold_component(part) for part in text.split("/"))
    if folded.startswith(_DATA_VOLUME):
        folded = folded[len(_DATA_VOLUME) - 1:]
    return folded


@functools.lru_cache(maxsize=1024)
def _entry_forms(entry: str) -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    """``(spelling, loose spelling, literal words)`` for each spelling of an
    absolute entry; nothing for a relative one, which is never resolved.

    ``literal words`` is None for a literal entry. For a glob, it is the
    loose form of each wildcard-free component: a path that lacks any of
    them cannot match, which rules out nearly every path before fnmatch.
    """
    if not Path(entry).is_absolute():
        return ()
    forms: list[tuple[str, str, tuple[str, ...] | None]] = []
    for spelling in entry_spellings(entry):
        folded = _loose(spelling)
        if is_glob_pattern(spelling):
            words = tuple(w for w in folded.split("/") if w and not is_glob_pattern(w))
            forms.append((spelling, folded, words))
        # Every spelling is also a literal path: a directory named ``[old]``
        # (or a link resolving to one) is compared by identity as a name too.
        forms.append((spelling, folded, None))
    return tuple(forms)


def _literal_under(text: str, entry: str) -> bool:
    """``text`` is ``entry`` or under it, comparing whole components."""
    return text == entry or text.startswith(entry.rstrip("/") + "/")


def _glob_regex(patterns: Iterable[str]) -> re.Pattern[str] | None:
    """One regex matching whatever ``denied_entry_matches`` matches for any of
    ``patterns`` (the pattern itself, or anything under it). fnmatch compares
    through ``os.path.normcase``, which is the identity on POSIX."""
    parts = [fnmatch.translate(p) for pat in patterns for p in (pat, pat.rstrip("/") + "/*")]
    return re.compile("|".join(parts)) if parts else None


class _Screen(NamedTuple):
    """Could any of a set of spellings match? Literal ones by component
    (``_literal_under``), glob ones by pattern. A superset test: when it says
    no, no spelling matches; when it says yes, the caller walks the list."""
    exact: frozenset[str]
    prefixes: tuple[str, ...]
    globs: re.Pattern[str] | None
    #: A word each glob must contain, searched for before the patterns.
    words: re.Pattern[str] | None

    def hits(self, text: str) -> bool:
        if text in self.exact or text.startswith(self.prefixes):
            return True
        if self.globs is None:
            return False
        if self.words is not None and self.words.search(text) is None:
            return False
        return self.globs.match(text) is not None


def _screen(literals: Iterable[str], globs: Iterable[str] = ()) -> _Screen:
    literals = tuple(literals)
    globs = tuple(globs)
    return _Screen(
        frozenset(literals), tuple(x.rstrip("/") + "/" for x in literals),
        _glob_regex(globs),
        _word_regex(tuple(w for w in g.split("/") if w and not is_glob_pattern(w)) for g in globs),
    )


def _word_regex(word_sets: Iterable[tuple[str, ...]]) -> re.Pattern[str] | None:
    """One search for a word every glob must contain: each glob's literal
    component nearest the leaf (the one that tells paths apart). None when
    some glob has no literal component, so every path must be matched."""
    keys = []
    for words in word_sets:
        if not words:
            return None
        keys.append(words[-1])
    return re.compile("|".join(re.escape(k) for k in sorted(set(keys)))) if keys else None


class _Plan(NamedTuple):
    entries: tuple[str, ...]
    glob_entries: tuple[str, ...]
    other_spellings: tuple[tuple[str, str, bool], ...]
    #: Every form's loose spelling equals its spelling (nothing to fold).
    fold_free: bool
    literal_exact: frozenset[str]
    literal_prefixes: tuple[str, ...]
    literal_forms: tuple[tuple[str, str, str], ...]
    glob_forms: tuple[tuple[str, str, str, tuple[str, ...] | None], ...]  # words never None here
    # One C-level test per pass; only a hit walks the list, in order, so the
    # entry named is the same one the walk alone would name.
    screen_1: _Screen
    screen_2: _Screen
    screen_3: _Screen
    screen_4_globs: _Screen


@functools.lru_cache(maxsize=64)
def _plan(entries: tuple[str, ...]) -> _Plan:
    """A deny list compiled once: readers pass the same list on every call,
    and the prune layer calls once per search result."""
    kept = tuple(e for e in entries if e)
    forms = [(e, *form) for e in kept for form in _entry_forms(e)]
    literal = tuple((e, s, f) for e, s, f, words in forms if words is None)
    glob_forms = tuple(form for form in forms if form[3] is not None)
    return _Plan(
        entries=kept,
        glob_entries=tuple(e for e in kept if is_glob_pattern(e)),
        other_spellings=tuple(
            (e, s, is_glob_pattern(s)) for e in kept for s in entry_spellings(e)[1:]),
        fold_free=all(f == s for _, s, f, _ in forms),
        literal_exact=frozenset(f for _, _, f in literal),
        literal_prefixes=tuple(f.rstrip("/") + "/" for _, _, f in literal),
        literal_forms=literal,
        glob_forms=glob_forms,
        screen_1=_screen((e for e in kept if not is_glob_pattern(e)),
                         (e for e in kept if is_glob_pattern(e))),
        screen_2=_screen(e for e in kept if is_glob_pattern(e)),
        screen_3=_screen(
            (s for e in kept for s in entry_spellings(e)[1:] if not is_glob_pattern(s)),
            (s for e in kept for s in entry_spellings(e)[1:] if is_glob_pattern(s))),
        screen_4_globs=_screen((), (f for _, _, f, _ in glob_forms)),
    )


def denying_entry(resolved_path: str | Path, entries: Iterable[str]) -> str | None:
    """The first entry that denies an already-resolved path, or None.

    THE decision for every reader of a deny list (the gate, workspace binding,
    the grep/glob prune layer, the coding sandboxes). Its passes only ever
    add denials, and whatever an earlier pass denies names the same entry it
    did before:

    1. each entry exactly as given: ``denied_entry_matches``, main's
       comparison (a glob matches the path or anything under it; a literal
       matches whole components);
    2. a glob entry as a literal path too: a directory named ``[old]`` holds
       glob characters and was compared as a name by the sandboxes;
    3. each entry with its literal directories resolved
       (:func:`entry_spellings`), for an entry that runs through a symlink;
    4. by identity, for what no spelling shows: case and Unicode
       normalisation on a volume that folds them, and macOS firmlinks. Only
       entries whose folded spelling could name the path are asked, so a path
       nowhere near a denied one costs no ``stat``. A literal entry goes
       through :func:`inside_by_identity`; a glob entry is matched with each
       path component spelled as the pattern spells it, wherever the volume
       says both names are one entry.
    """
    text = str(resolved_path)
    plan = _plan(tuple(entries))
    if plan.screen_1.hits(text):  # main's comparison, walked only when it can match
        for entry in plan.entries:
            if denied_entry_matches(text, entry):
                return entry
    if plan.screen_2.hits(text):
        for entry in plan.glob_entries:
            if _literal_under(text, entry):
                return entry
    if plan.screen_3.hits(text):
        for entry, spelling, glob in plan.other_spellings:
            if denied_entry_matches(text, spelling) or (glob and _literal_under(text, spelling)):
                return entry
    loose = _loose(text)
    if loose == text and plan.fold_free:
        # Nothing to fold on either side, so the filter below would ask
        # exactly what passes 1-3 just answered: nothing can pass it.
        return None
    literal: dict[Path, str] = {}
    if loose in plan.literal_exact or loose.startswith(plan.literal_prefixes):
        for entry, spelling, folded in plan.literal_forms:
            if _literal_under(loose, folded):  # also covers an absent entry's variant
                literal.setdefault(Path(spelling), entry)
    globs: list[tuple[str, str]] = []
    if plan.screen_4_globs.hits(loose):
        globs = [(entry, spelling) for entry, spelling, folded, words in plan.glob_forms
                 if denied_entry_matches(loose, folded)]
    if not literal and not globs:
        return None
    path = Path(text)
    if literal:
        hit = inside_by_identity(path, literal, absent_only_where_case_folds=True)
        if hit is not None:
            return literal[hit]
    for entry, spelling in globs:
        prefix, rest = split_literal_prefix(spelling)
        key = file_identity(prefix)
        if key is None:
            continue
        for ancestor in path.parents:
            if file_identity(ancestor) != key:
                continue
            rel = path.relative_to(ancestor).as_posix()
            if _glob_rest_matches(rel, rest):
                return entry
            # On a volume that folds case, ``.ENV`` IS ``.env``: match the
            # wildcard part folded as well, as the volume compares names.
            if _nearest_folds_case(path) and _glob_rest_matches(_loose(rel), _loose(rest)):
                return entry
            canonical = _as_the_pattern_spells_it(ancestor, rel, rest)
            if canonical != rel and _glob_rest_matches(canonical, rest):
                return entry
    return None


def _glob_rest_matches(rel: str, rest: str) -> bool:
    return fnmatch.fnmatch(rel, rest) or fnmatch.fnmatch(rel, rest.rstrip("/") + "/*")


def _as_the_pattern_spells_it(ancestor: Path, rel: str, rest: str) -> str:
    """``rel`` with each component that differs from one of the pattern's
    literal components only by case spelled as the pattern spells it, where
    that is the same directory entry.

    Same entry means both names exist and are one file, or, on a volume that
    folds case, neither exists yet (creating one creates the other: #574's
    rule for a protected directory that isn't there yet). A name beside a
    differently cased one on a case-sensitive volume is its own entry and is
    left alone, as are components with a wildcard in them.
    """
    literals = {
        fold_component(part): part
        for part in rest.split("/")
        if part and not is_glob_pattern(part)
    }
    if not literals:
        return rel
    out: list[str] = []
    parent = ancestor
    for part in rel.split("/"):
        spelled = literals.get(fold_component(part))
        if spelled is not None and spelled != part:
            as_given = file_identity(parent / part, follow_symlinks=False)
            as_spelled = file_identity(parent / spelled, follow_symlinks=False)
            if as_given is not None and as_given == as_spelled:
                part = spelled
            elif as_given is None and as_spelled is None and _nearest_folds_case(parent):
                part = spelled
        out.append(part)
        parent = parent / part
    return "/".join(out)


def _nearest_folds_case(directory: Path) -> bool:
    """:func:`volume_folds_case` for the volume ``directory`` is (or would be)
    on: that of its nearest existing ancestor. Nothing exists: can't tell, fold."""
    for candidate in (directory, *directory.parents):
        if file_identity(candidate) is not None:
            return volume_folds_case(candidate)
    return True
