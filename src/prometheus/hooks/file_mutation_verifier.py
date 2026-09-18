"""SPRINT-2 WS2 — File-mutation verifier.

Catches silent failures where a tool *claims* a write succeeded but the
side effect didn't land on disk: Gemma saying "wrote 47 lines to foo.py"
while the editor returns success but the file is unchanged
bash exiting
0 without the side effect; permission-denied surfacing as "success" in a
buggy tool wrapper. These are the Adapter Layer's blind spot — the
*response shape* was fine, but the bytes on disk disagree.

How it works:
  - Pre-tool-use: for any FS-touching tool call, ``os.stat`` the target
    path (or each path the bash lexer extracts) and stash the result on the
    in-flight turn record.
  - Post-tool-use: ``os.stat`` again. Diff with the snapshot. Tag the
    mutation as ``created``, ``modified``, ``deleted``, ``failed``, or
    ``no_change`` (claimed write but disk unchanged — the load-bearing
    case).
  - Post-turn: if any mutations accumulated, emit a one-block summary as
    a synthetic injected turn so the model sees it on its NEXT turn. Same
    channel as PeriodicNudge, but tagged ``provenance="file_mutation_
    verifier"`` rather than masquerading as something the user typed.

TURN SCOPING: ``run_daemon`` builds exactly ONE verifier and hands it to
every surface — telegram, CLI, cron, and (since this change) web/Beacon.
State is therefore keyed by ``turn_key``: one key per ``run_loop``
invocation, minted by the loop itself. Without it a single flat
accumulator is shared by every concurrent turn, so the first turn to
finish drains the other's mutations and reports them as its own while
the second reports nothing — which inverts a feature whose entire job is
checking that the writes YOU claimed actually landed.

No Hermes precedent: their hooks docs explicitly state file-mutation
verification "isn't provided as a ready-made feature" (see
``website/docs/user-guide/features/hooks.md`` on the upstream). Built
native — note in commit message.

Config:
  hooks:
    file_mutation_verifier:
      enabled: true              # opt-out, on by default
      truncate_after_n_mutations: 20
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

# Callers that don't mint a turn key (unit tests, duck-typed embedders, any
# future single-threaded driver) share this one. Safe there BECAUSE they are
# single-threaded; every concurrent surface goes through run_loop, which always
# passes a real key.
DEFAULT_TURN_KEY = "__unscoped__"

# Backstop against records that never get drained: a turn that ends by raising,
# or one whose run_loop generator is abandoned without being closed, leaves its
# record behind. run_loop discards in a ``finally`` so this should stay cold —
# it exists so a caller that forgets cannot leak without bound. Evicting the
# least-recently-touched turn degrades to "no summary for that turn", never to
# cross-turn contamination.
MAX_LIVE_TURNS = 64


# Tool names that touch the filesystem. Path extraction is per-tool —
# see ``_extract_paths`` below.
_FS_TOOLS = frozenset({
    "file_write", "write_file",       # writes
    "file_edit", "edit_file",
    "notebook_edit",
})

# Bash command words whose side effect is a path mutation, mapped to the
# claimed-action tag reported for them.
#
# Keyed by the clause's COMMAND WORD — the token the shell would actually
# execute — not matched anywhere in the clause text. `echo touch $FOO` mutates
# nothing, but a pattern table matching the verb wherever it appeared read the
# unresolved token after it as a lost destination and reported the clause blind.
# #486 pinned that as a known false positive rather than fixing it, because the
# structural fix needs a token stream. This is that fix.
#
# The value says which operands carry the mutation:
#   "last" — mv/cp: the destination is the new home.
#   "all"  — rm/touch/mkdir: every non-flag operand is a target.
_MUTATION_VERBS: dict[str, tuple[str, str]] = {
    "mv":    ("move",   "last"),
    "cp":    ("copy",   "last"),
    "rm":    ("delete", "all"),
    "touch": ("touch",  "all"),
    "mkdir": ("mkdir",  "all"),
}

# Wrappers that keep the FOLLOWING word in command position: `sudo rm -f x`
# deletes, and so does `env FOO=1 touch x`.
#
# READ THIS BEFORE EXTENDING IT. The rewrite issue (#484) says "a token stream
# says which token is the command word; nothing has to guess", and for finding
# token 0 that is true. It is NOT true for wrappers: `sudo rm $X` and
# `echo rm $X` are structurally identical token streams — verb at index 1,
# another word at index 0 — and no amount of lexing distinguishes them. Only
# knowing that `sudo` execs its argument and `echo` prints it does. So a list is
# unavoidable here, and pretending otherwise would be the fifth repeat of this
# file's documented failure mode.
#
# What makes it safe is the DIRECTION it fails in. An unlisted wrapper
# (`nice rm -f $X`) leaves the verb in argument position, so the clause reports
# nothing — a false negative, which is this file's accepted direction and the
# same outcome as any command it does not model. An over-long list is the
# dangerous direction: adding `echo` here would invent mutations. Add a name only
# when it genuinely execs a following command word.
_COMMAND_WRAPPERS = frozenset({
    "sudo", "doas", "env", "time", "nohup", "command", "exec", "builtin",
    "nice", "ionice", "stdbuf", "setsid", "xargs", "timeout",
    # shell keywords that precede a command word rather than being one
    "do", "then", "else", "elif", "if", "while", "until", "!", "{",
})


@dataclass
class _Snapshot:
    """os.stat result captured before a tool runs (None = file absent)."""
    exists: bool
    size: int = 0
    mtime: float = 0.0
    mode: int = 0


def _changed(before: _Snapshot, after: _Snapshot) -> bool:
    """True when the filesystem actually moved under this path.

    Creation, deletion, or any change to size/mtime. Content is not captured
    (see ``_Snapshot``), so an in-place rewrite of identical length within the
    same mtime granularity is invisible — a known false negative, and the
    reason this layer is described as detection rather than containment.
    """
    if before.exists != after.exists:
        return True
    if not after.exists:
        return False
    return (before.size, before.mtime) != (after.size, after.mtime)


@dataclass
class _Mutation:
    """One tracked filesystem touch this turn."""
    tool: str
    path: str
    claimed_action: str           # per-path: "delete", "redirect_write", "write", …
    before: _Snapshot
    after: _Snapshot
    error: str | None = None      # populated when the tool itself reported failure


@dataclass
class _TurnRecord:
    """Per-turn accumulator. Created on first touch, dropped on PostTurn."""
    mutations: list[_Mutation] = field(default_factory=list)
    # Commands that redirected somewhere this hook could not name. Not mutations —
    # the opposite: the places it knows it could not look. post_turn speaks up on
    # these so silence never has to mean two different things (issue #275).
    blind: list[str] = field(default_factory=list)
    # Commands this hook could not TOKENIZE at all — an unbalanced quote, usually.
    # Kept apart from ``blind`` because they are different admissions: "I parsed this
    # and could not name the destination" versus "I could not parse this, so I do not
    # know whether it wrote anything." Folding the second into the first would
    # overstate what was measured (issue #484, criterion 3).
    unaudited: list[str] = field(default_factory=list)
    # Map turn-scoped pre-snapshots by (tool_use_id, path) so post_tool_use
    # can pair them up even when one tool call touches multiple paths.
    _pending: dict[tuple[str, str], _Snapshot] = field(default_factory=dict)


def _expand_user(path: str) -> str:
    """Expand a leading ``~`` the way the shell already did.

    The shell expands ``~/.ssh/x`` before the write lands
    ``os.stat`` does
    not. Snapshotting the unexpanded literal stats a path that can never
    exist, so before/after is absent->absent, ``_classify`` returns
    "missing", and a mutation that really happened is reported as nothing at
    all. Three writes under the denied-path floor landed this way with zero
    lines emitted.

    Expanding here also matters downstream: ``landed_paths()`` is handed to
    the permission gate by ``agent_loop._boundary_escapes``, and the floor
    globs (``/*/.ssh``) cannot match a ``~``-prefixed literal.

    Only a leading ``~`` is touched. Relative paths are deliberately left
    alone: a bash clause can ``cd`` first, so resolving them against the
    daemon's cwd would invent a path the command never used.
    """
    if not path.startswith("~"):
        return path
    return os.path.expanduser(path)


def _snapshot(path: str) -> _Snapshot:
    """Cheap os.stat wrapper. Returns an absent-marker on any error."""
    try:
        st = os.stat(path)
        return _Snapshot(
            exists=True,
            size=int(st.st_size),
            mtime=float(st.st_mtime),
            mode=int(st.st_mode),
        )
    except (OSError, ValueError):
        return _Snapshot(exists=False)


def _classify(before: _Snapshot, after: _Snapshot) -> str:
    """Compare before/after snapshots and assign a status tag."""
    if before.exists and not after.exists:
        return "deleted"
    if not before.exists and after.exists:
        return "created"
    if not before.exists and not after.exists:
        return "missing"      # claimed something but path never existed
    # both exist — compare
    if before.size != after.size or before.mtime != after.mtime:
        return "modified"
    return "no_change"        # the load-bearing silent-failure case


def _extract_paths(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """Best-effort extraction of paths from tool input."""
    out: list[str] = []
    # file_write / file_edit / notebook_edit all use ``file_path`` (or
    # ``path``) — the Prometheus convention.
    #
    # Every key, not the first one found: this used to ``break`` on the first hit, so
    # a tool call carrying more than one path key was only PARTLY audited and the
    # unaudited half was silent — the same failure this file exists to remove, one
    # layer up from the bash extraction. Order preserved, duplicates dropped, so a
    # call naming the same path twice does not snapshot it twice.
    for key in ("file_path", "path", "notebook_path"):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            expanded = _expand_user(val)
            if expanded not in out:
                out.append(expanded)
    return out


# ---------------------------------------------------------------------------
# Tokenizer — the extraction floor
# ---------------------------------------------------------------------------
#
# WHAT THIS REPLACED AND WHY. Extraction used to be a table of regexes run over
# text whose quoted spans had first been blanked to spaces. That mechanism
# discarded the data it existed to read: `touch "my file.txt"` — a path written the
# only way a path with a space CAN be written — was blanked along with the prose the
# blanking existed to protect against, and the patterns then groped around the hole.
# `rm -f "$P"` reported a claimed write to a file named `-f`, because the operand had
# been blanked and `(\S+)` reached past it to the flag.
#
# Four patches (#198, #274, #483, #485) each removed the shape that had been reported
# and left the next one open, because a filter extended against known-bad shapes can
# only ever be complete with respect to shapes someone has already met. A lexer is not
# a fifth filter: clause structure, flags, command words and redirect operators stop
# being approximated by lookbehinds and become properties of the token stream.
#
# THE INVARIANT this exists to establish:
#
#     No write form may be both untracked and unreported. Every clause containing a
#     SHELL-LEVEL redirect or mutation verb either resolves to a tracked path, or
#     produces a blindness row. Silence is a failure of the contract.
#
# "SHELL-LEVEL" is load-bearing and is a correction to the way #484 states it. An
# unqualified reading is not achievable by any lexer: `python -c "os.remove(p)"`,
# `make`, `npm ci` and `git checkout` all mutate the filesystem with no shell operator
# anywhere in the command. Scoping the contract to shell syntax is the difference
# between a contract with an undocumented hole and one with a stated boundary — and
# inside that boundary it is held absolutely, with the single exception of a device
# sink, where nothing reaches disk at all.
#
# WHY posix=False. The lexer must be able to tell a quoted `'>'` from an operator. In
# posix mode shlex resolves quoting during tokenization, so `grep -rn '>' src/ > out`
# arrives with a bare `>` where the search pattern was, the following `src/` is read as
# a redirect target, and a directory the command only READ is reported as written.
# The same erasure turns a quoted `'#'` into a comment introducer and a quoted `"|"`
# into a clause separator. Non-posix keeps the quote characters ON the token, so
# quoting is a fact the walker can consult; :func:`_unquote` resolves it afterwards,
# which is what keeps `touch "my file.txt"` tracked.

_LEX_PUNCTUATION = "();<>|&\n"

# Characters that, alone or in a run, end a clause. shlex merges consecutive
# punctuation into ONE token, so `;` before a newline arrives as `;\n` and a blank line
# arrives as `\n\n`. Membership testing against a fixed set of spellings missed every
# one of those: the clause did not split, and for a `which == "all"` verb every
# surviving token became a claimed path — `rm -f a.txt;\nrm -f b.txt` reported deleting
# a file literally named `;\n`. Classify by CHARACTER CONTENT, not by spelling.
_SEPARATOR_CHARS = frozenset(";&|()\n")

# The characters a redirect operator can be built from. `=` is deliberately NOT a
# punctuation char: making it one does merge `=>` into a single harmless token, but it
# also shatters every operand containing `=` (`rm -f a=b.txt` became three claimed
# paths) and, worse, `FOO= rm -f x` merged the assignment with the VERB, so the
# command word became `FOO=rm` and a real delete went silent. `=>` and `>=` are
# neutralised in :func:`_preprocess` instead, where quote state is known.
_OPERATOR_CHARS = frozenset("<>&|")

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DIGITS = re.compile(r"^\d+$")

# A token that is not a path. Under the lexer this is a much smaller job than it was:
# operators and quoted spans no longer leak into operands, so what is left is a flag
# (`-f`, `--recursive`) and the stray fd digit forms. Rejected by PREFIX rather than by
# a whitelist of what a path may contain, because a path may legitimately contain
# almost anything. The cost is a real file whose name begins with `-`, which needs `--`
# to address in shell anyway.
_NOT_A_PATH = re.compile(r"^(?:-|\d*[<>&])")


class _Unlexable(Exception):
    """The command could not be tokenized — an unbalanced quote, usually.

    NOT suppressed. `shlex` raising here is a feature: a command this instrument
    cannot parse must be reported as *unaudited*, exactly the way an unnameable
    redirect target is reported blind. A parse that fails loudly beats a regex that
    succeeds wrongly, which is `docs/audits/RECURRING.md` 4j applied to the file 4j
    was written about.
    """


_SUBSTITUTION_MARKER = "$__prometheus_subst__"


def _preprocess(command: str) -> str:
    """Quote-aware pre-pass, run before the lexer sees the text.

    Every transform here MUST track quote state. An earlier draft did these with
    regexes over raw text and each one reintroduced exactly the defect the rewrite
    exists to retire — a quote-blind scan:

      * `gh pr create --body "use <<EOF for stdin"` matched a heredoc start inside
        quoted prose, found no terminator, and swallowed the REST OF THE COMMAND,
        including an `rm -rf` on the next line. Silent, and catastrophic.
      * ``echo '```' >> a.md`` paired the first backtick of a markdown fence with one
        three lines later and deleted everything between them.

    Three transforms, in one pass:

    1. LINE CONTINUATIONS. `\\` before a newline is deleted by the shell, so the
       command is one clause. shlex in non-posix mode leaves both characters, and the
       newline then splits the clause: `rm -rf \\<nl> build` lost `build` entirely.
    2. HEREDOC BODIES. A body is DATA — the shell never executes it. Left in the
       stream it is read as commands, and not merely as noise: `cat <<EOF` followed by
       `echo hi > /tmp/x` extracted `/tmp/x` as a claimed write and emitted a
       permanent false "CLAIMED but FILE ABSENT" about a file nobody named.
    3. SUBSTITUTIONS. `$(…)`, backticks and process substitutions `<(…)` / `>(…)`
       collapse to one opaque token. `cp a.txt /tmp/bak-$(date +%s).txt` otherwise
       lexes to `/tmp/bak-$(date` and `+%s).txt`, and the last-operand rule would
       track a file literally named `+%s).txt` — a fabricated path in an audit trail.
       The marker matches :data:`_UNRESOLVED`, so the clause routes to the blindness
       contract instead, which is the honest answer.
    """
    out: list[str] = []
    i, n = 0, len(command or "")
    quote: str | None = None
    pending_heredocs: list[tuple[str, bool]] = []   # (delimiter, strip_tabs)

    while i < n:
        ch = command[i]

        if quote is not None:
            out.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                out.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            if command[i + 1] == "\n":
                i += 2
                continue                    # line continuation: delete both
            out.append(ch)
            out.append(command[i + 1])
            i += 2
            continue

        if ch == "\n":
            out.append(ch)
            i += 1
            # Consume the body of every heredoc opened on the line just ended.
            while pending_heredocs:
                delimiter, strip_tabs = pending_heredocs.pop(0)
                i, terminated = _skip_heredoc_body(command, i, delimiter, strip_tabs)
                if not terminated:
                    # The body ran to the end of the command without its terminator,
                    # so everything after the `<<` is data of unknown extent and any
                    # command in it is unreadable. Bash accepts an indented terminator
                    # only for `<<-`, so `  EOF` closing a plain `<<EOF` lands here.
                    # Reporting this as a clean turn would hide whatever followed;
                    # `unaudited` says what is true — this could not be parsed.
                    raise _Unlexable(f"unterminated heredoc: {delimiter}")
            continue

        two = command[i:i + 2]
        if two in ("=>", ">="):
            # Not shell. `(s) => /^smoke:/.test(s)` is the JS/TS arrow that #483
            # existed to stop claiming a file, and `>=` is a comparison. Both put a
            # bare `>` in front of an operand, which any redirect rule must then read
            # as a write. Neutralised HERE, outside quotes, so the token stream never
            # carries a `>` that was not an operator — rather than by a lookbehind
            # that has to guess from the preceding character.
            #
            # ACCEPTED FALSE NEGATIVE, stated rather than discovered later: bash does
            # parse `a=>b` as the assignment `a=` plus a redirect to `b`, and that
            # redirect is lost here. `main` loses it too (its lookbehind refuses a `>`
            # preceded by `=`), so this is not a regression, and no such form has been
            # observed in agent-issued bash.
            out.append("  ")
            i += 2
            continue

        if ch == "`":
            close = command.find("`", i + 1)
            if close == -1:
                out.append(ch)
                i += 1
                continue
            out.append(_SUBSTITUTION_MARKER)
            i = close + 1
            continue

        two = command[i:i + 2]
        if two in ("$(", "<(", ">(") :
            end = _match_paren(command, i + 1)
            if end is None:
                out.append(ch)
                i += 1
                continue
            out.append(_SUBSTITUTION_MARKER)
            i = end + 1
            continue

        if command[i:i + 3] == "<<<":
            # A herestring, not a heredoc. Consumed whole: emitting one `<` and
            # re-entering left `<<yes` looking like a heredoc opened with the
            # delimiter `yes`, which never terminates — so everything after it,
            # including the next line's `rm -rf`, was swallowed as body.
            out.append("<<<")
            i += 3
            continue

        if two == "<<":
            j = i + 2
            strip_tabs = j < n and command[j] == "-"
            if strip_tabs:
                j += 1
            while j < n and command[j] in " \t":
                j += 1
            opened, j = _read_heredoc_delimiter(command, j)
            if opened is None:
                out.append(ch)
                i += 1
                continue
            out.append(two)
            if strip_tabs:
                out.append("-")
            out.append(" " + opened)
            pending_heredocs.append((opened, strip_tabs))
            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _match_paren(text: str, open_index: int) -> int | None:
    """Index of the `)` matching the `(` at ``open_index``, or None."""
    depth, j, n = 0, open_index, len(text)
    while j < n:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return None


def _read_heredoc_delimiter(text: str, j: int) -> tuple[str | None, int]:
    """Read a heredoc delimiter at ``j``. Handles `EOF`, `'EOF'`, `"EOF"`, `\\EOF`."""
    n = len(text)
    if j < n and text[j] in "'\"":
        quote = text[j]
        close = text.find(quote, j + 1)
        if close == -1:
            return None, j
        return text[j + 1:close], close + 1
    if j < n and text[j] == "\\":
        j += 1
    start = j
    while j < n and (text[j].isalnum() or text[j] == "_"):
        j += 1
    return (text[start:j], j) if j > start else (None, j)


def _skip_heredoc_body(
    text: str, i: int, delimiter: str, strip_tabs: bool
) -> tuple[int, bool]:
    """Return (index just past the body and its terminator, was it terminated).

    The terminator must be the delimiter ALONE on its line. Bash allows leading tabs
    only for the `<<-` form
    accepting arbitrary indentation for a plain `<<` ends the
    body early and spills the rest back into the command stream.
    """
    n = len(text)
    while i < n:
        line_end = text.find("\n", i)
        stop = n if line_end == -1 else line_end
        line = text[i:stop]
        candidate = line.lstrip("\t") if strip_tabs else line
        if candidate == delimiter:
            return (n if line_end == -1 else line_end + 1), True
        i = n if line_end == -1 else line_end + 1
    return n, False


def _lex(command: str) -> list[str]:
    """Tokenize one bash command. Raises :class:`_Unlexable` if it cannot.

    ``commenters`` is cleared deliberately. `#` is legal in a filename and shlex's
    comment handling is not word-anchored: with it on, `touch a#b.txt` lexes to
    `['touch', 'a']` — a claim about a different, plausible file, which is worse than
    any comment gap. Comments are dropped in :func:`_clause_token_lists`, where a `#`
    that begins its own UNQUOTED token can be told from one inside a path.
    """
    lexer = shlex.shlex(
        _preprocess(command or ""),
        posix=False,
        punctuation_chars=_LEX_PUNCTUATION,
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.whitespace = " \t\r"          # NOT \n — it is a clause separator token
    try:
        return list(lexer)
    except ValueError as exc:           # "No closing quotation"
        raise _Unlexable(str(exc)) from exc


def _is_quoted(token: str) -> bool:
    """True when the token carries its own quotes, so it is a WORD whatever it spells.

    This is the whole reason for ``posix=False``. A quoted token can spell `>` or `|`
    or `#` and none of them are operators.
    """
    return len(token) >= 2 and token[0] == token[-1] and token[0] in "'\""


def _unquote(token: str) -> str:
    """Resolve quoting for a token the walker has decided is an operand.

    Done HERE rather than by the lexer so the operator/word decision is made while the
    quotes are still visible. `touch "my file.txt"` therefore yields `my file.txt` —
    the floor row #484 exists to raise — without a quoted `'>'` ever being mistaken for
    a redirect.
    """
    out, i, n = [], 0, len(token)
    while i < n:
        ch = token[i]
        if ch in "'\"":
            close = token.find(ch, i + 1)
            if close == -1:
                out.append(token[i + 1:])
                break
            out.append(token[i + 1:close])
            i = close + 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(token[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_separator(token: str) -> bool:
    """True for `;`, `&&`, `|`, `(`, `)`, a newline — and any RUN of them.

    Classified by character content because shlex merges consecutive punctuation:
    `
    \\n`, `&&\\n`, `\\n\\n`, `
    \\n` and `|&` are all single tokens, and a set of
    literal spellings missed every one.
    """
    return (
        not _is_quoted(token)
        and bool(token)
        and all(ch in _SEPARATOR_CHARS for ch in token)
    )


def _redirect_kind(token: str) -> str | None:
    """Classify a token as an output redirect operator, or None."""
    if not token or _is_quoted(token) or _is_separator(token):
        return None
    if any(ch not in _OPERATOR_CHARS for ch in token):
        return None                     # `=>`, `<=`, `a>b`, any word
    if ">" not in token:
        return None                     # `<`, `<<` — input, not a write
    return "redirect_append" if ">>" in token else "redirect_write"


def _clause_token_lists(tokens: list[str]) -> list[list[str]]:
    """Split a token stream into clauses on separator TOKENS, dropping comments.

    Quote-blind `re.split` on `;` is what turned `echo "… does not touch them."` into a
    claimed `touch` of a file named `them.`: the split cut the quoted span in half,
    each half carried an unbalanced quote, and the prose survived as bare shell text. A
    separator that is a token cannot be inside a quoted span, so that cascade is not
    merely fixed — it is unreachable.
    """
    clauses: list[list[str]] = []
    current: list[str] = []
    skipping = False
    for token in tokens:
        newline = not _is_quoted(token) and "\n" in token
        if skipping and not newline:
            continue
        skipping = False
        if _is_separator(token):
            if current:
                clauses.append(current)
            current = []
            continue
        if token == "#":                # an UNQUOTED bare `#` opens a comment
            skipping = True
            if current:
                clauses.append(current)
            current = []
            continue
        current.append(token)
    if current:
        clauses.append(current)
    return clauses


def _command_word_index(words: list[str]) -> tuple[int | None, bool]:
    """Index of the token the shell would execute, and whether a wrapper was skipped.

    Skips `VAR=value` assignment prefixes and the wrappers in
    :data:`_COMMAND_WRAPPERS`. The second return value is what makes an UNRESOLVED
    wrapper safe: `sudo -u postgres rm -rf /x` skips `sudo`, then `-u` as a flag, and
    lands on `postgres` — a flag ARGUMENT mistaken for the command word, because
    nothing here knows which flags take one. Rather than grow a list of those, the
    caller is told a wrapper was involved and reports blindness when a mutation verb
    is sitting in the clause unexplained.
    """
    i, n, saw_wrapper = 0, len(words), False
    while i < n:
        word = words[i]
        if _ASSIGNMENT.match(word) and not _is_quoted(word):
            i += 1
            continue
        if word in _COMMAND_WRAPPERS:
            saw_wrapper = True
            i += 1
            continue
        if word.startswith("-"):
            saw_wrapper = True          # a flag here belongs to a wrapper we skipped
            i += 1
            continue
        return i, saw_wrapper
    return None, saw_wrapper


# `find … -exec rm …` and `find … -delete` run a mutation the clause's command word
# does not name, and `-t DIR` inverts cp/mv's destination-last argument order. Neither
# can be resolved to a path here, so both route to the blindness contract rather than
# being guessed at or ignored.
_DEFERRED_EXECUTORS = frozenset({"-exec", "-execdir", "-delete"})
_TARGET_DIRECTORY_FLAGS = frozenset({"-t", "--target-directory"})
_SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


@dataclass(frozen=True)
class _BashAudit:
    """What one bash command yielded: tracked paths, blindness, auditability."""
    tracked: list[tuple[str, str]]
    blind: bool
    unaudited: str | None = None        # reason, when the command could not be lexed


def _analyze_clause(tokens: list[str], depth: int = 0) -> tuple[list[tuple[str, str]], bool]:
    """One clause -> (tracked paths, blind).

    The invariant is enforced HERE, as a disjunction rather than a list of cases: every
    redirect target and every mutation operand either becomes a tracked path or sets
    ``blind``. The one documented exception is a device sink (`/dev/null`, an fd
    duplicate), where nothing reaches disk — the absence of a write, not silence
    about one.
    """
    tracked: list[tuple[str, str]] = []
    blind = False
    words: list[str] = []

    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        kind = _redirect_kind(token)
        if kind is None:
            words.append(token)
            i += 1
            continue
        target = tokens[i + 1] if i + 1 < n else None
        # `2> err.log` lexes the fd as its own word. Only a SINGLE digit is treated as
        # one: `mkdir -p 2024 > /dev/null` popped `2024` as a phantom fd and lost the
        # directory, and on a destination-last verb `cp a.txt 42 > /dev/null` shifted
        # the claim onto the SOURCE file.
        if words and len(words[-1]) == 1 and words[-1].isdigit():
            words.pop()
        if token.endswith("&") and target is not None and _DIGITS.match(target):
            # `2>&1` — an fd duplicate. Not a file, and correctly not blindness:
            # nothing was written anywhere this hook could look. The digit check is
            # what keeps `make >& build.log` — a real write — out of this branch.
            i += 2
            continue
        if target is None or _redirect_kind(target) is not None or _is_separator(target):
            blind = True                # a redirect with nothing to name
        else:
            resolved = _expand_user(_unquote(target))
            if _is_device_sink(resolved):
                pass
            elif _is_trackable(resolved):
                tracked.append((resolved, kind))
            else:
                blind = True
        i += 2

    bare = [w for w in words if not _is_quoted(w)]
    index, saw_wrapper = _command_word_index(words)
    if index is None:
        return tracked, blind
    verb = os.path.basename(_unquote(words[index]))

    # `sh -c '<script>'` hides an entire command inside one operand. Audit it rather
    # than reporting the clause as clean: the script is ordinary shell, so the same
    # walker applies. Depth-bounded so a pathological nesting cannot recurse away.
    if verb in _SHELL_INTERPRETERS and depth < 3 and "-c" in bare:
        after = words[index + 1:]
        for j, word in enumerate(after):
            if word == "-c" and j + 1 < len(after):
                inner = _unquote(after[j + 1])
                try:
                    for clause in _clause_token_lists(_lex(inner)):
                        inner_tracked, inner_blind = _analyze_clause(clause, depth + 1)
                        tracked.extend(inner_tracked)
                        blind = blind or inner_blind
                except _Unlexable:
                    blind = True
                return tracked, blind

    entry = _MUTATION_VERBS.get(verb)
    if entry is None:
        # Not a mutation command. `echo touch $FOO` lands here: the verb is in ARGUMENT
        # position, so nothing is mutated and the clause is not blind. #486 reported
        # exactly this shape blind and pinned it as a known false positive; the command
        # word is what closes it.
        if _DEFERRED_EXECUTORS & set(bare):
            return tracked, True        # find -exec / -delete: a mutation we cannot name
        if saw_wrapper and any(os.path.basename(w) in _MUTATION_VERBS for w in bare):
            # A wrapper whose own flags we could not parse is standing between us and a
            # mutation verb. Report it rather than letting the verb hide behind an
            # unrecognised flag argument.
            return tracked, True
        return tracked, blind

    action, which = entry
    operands = [w for w in words[index + 1:] if _is_quoted(w) or not w.startswith("-")]
    if not operands:
        # `find … | xargs rm -f` — the targets arrive on stdin, so the verb mutates
        # something this hook cannot see. Silence here would be the invariant's
        # purest break: a delete with no row at all.
        return tracked, True
    if which == "last" and (_TARGET_DIRECTORY_FLAGS & set(bare)):
        return tracked, True            # -t DIR inverts the argument order
    for operand in (operands[-1:] if which == "last" else operands):
        resolved = _expand_user(_unquote(operand))
        if _is_device_sink(resolved):
            continue
        if _is_trackable(resolved):
            tracked.append((resolved, action))
        else:
            blind = True
    return tracked, blind


def audit_bash_command(command: str) -> _BashAudit:
    """Extract tracked paths and decide auditability for one bash command."""
    try:
        tokens = _lex(command)
    except _Unlexable as exc:
        return _BashAudit(tracked=[], blind=False, unaudited=str(exc))
    tracked: list[tuple[str, str]] = []
    blind = False
    for clause in _clause_token_lists(tokens):
        clause_tracked, clause_blind = _analyze_clause(clause)
        tracked.extend(clause_tracked)
        blind = blind or clause_blind
    return _BashAudit(tracked=tracked, blind=blind)


def _extract_bash_paths(command: str) -> list[tuple[str, str]]:
    """Return ``(path, claimed_action)`` tuples extracted from a bash line."""
    return audit_bash_command(command).tracked


def _is_device_sink(target: str) -> bool:
    """True for redirect targets that are kernel device sinks, not files.

    ``> /dev/null`` (or /dev/zero, /dev/stdout, /dev/urandom, …) never
    changes on disk, so snapshotting it yields a guaranteed "CLAIMED but
    NO CHANGE ON DISK" — a false positive that poisons the summary. The
    exception is ``/dev/shm/``: a real tmpfs where files genuinely land.

    ``>&1`` / ``2>&1`` (fd duplicates) capture ``&1``-style targets here
    too — same class: not a path, never on disk, guaranteed false
    "CLAIMED but FILE ABSENT".
    """
    if re.fullmatch(r"&\d+", target):
        return True
    return target.startswith("/dev/") and not target.startswith("/dev/shm/")


# An unexpanded parameter or command substitution: `$VAR`, `${VAR}`, `$(…)`, and
# the positional/special parameters `$1`, `$@`, `$?`, `$$`.
#
# The positional forms were EXCLUDED under the pattern table, on measured grounds:
# across 7,244 real commands, 239 contained one somewhere and 0 had one tracked as a
# phantom path, so widening the regex would have been a blocklist growing without a
# reason. Under the lexer the reason arrives: `$1` in an operand position is a token
# the stream identifies as an operand, so including it costs no new machinery and
# removes three standing false positives — `echo x > $1` used to track a file
# literally named `$1` and emit a permanent `⚠ CLAIMED but FILE ABSENT` about it.
# A phantom path replaced by an honest blindness row is the trade this file is built
# on. NOT a bare `$` at end of string, which names nothing.
_UNRESOLVED = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|\$\(|\$[0-9@*#?$!-]")


def _is_unresolved(target: str) -> bool:
    """True when a redirect/operand target still carries a shell substitution.

    ``> $LOG`` names a real destination this instrument cannot resolve. Stat'd as the
    literal string it can never exist, so TRACKING it emits a permanent
    ``⚠ CLAIMED but FILE ABSENT``: a false row, forever, about a file never named.

    QUOTING IS NOT AN INPUT HERE, deliberately. The pattern table blanked quoted spans
    before matching, so ``> "$LOG"`` vanished entirely while ``> $LOG`` survived — two
    spellings of one write got opposite treatment, and the quoted one was reported as a
    CLEAN TURN. Under the lexer both arrive as the operand ``$LOG`` and both are blind.
    A predicate that answered differently for them would carry an exception whose only
    evidence is typography.

    This is NOT "a shape to filter out". ``echo x > $LOG`` is **a write whose
    destination this instrument cannot name** — which is precisely what the #275
    blindness contract exists for. So the predicate has two call sites and the
    second is the point:

      * :func:`_is_trackable` stops TRACKING it (no false FILE ABSENT row), and
      * :func:`command_has_unnameable_target` stops counting it as a NAMEABLE target, so
        a clause whose only destination is unresolved has an operator and nothing
        to name, and the existing contract emits a blindness row.

    The result is one honest row saying *"a write happened here and I could not
    name the destination"* in place of a false row saying *"this file is missing."*
    The first is true and actionable
    the second is noise that trains the reader to
    skim.

    Adding ``$`` to ``_NOT_A_PATH`` instead would have dropped the target SILENTLY —
    the same failure mode as the no-space redirect, freshly reproduced, and the
    fourth extension of a blocklist that #484 documents as having a floor.
    """
    return bool(_UNRESOLVED.search(target or ""))


def _is_trackable(target: str) -> bool:
    """True when a captured operand is worth snapshotting.

    Rejected: shell syntax (flags, fd digits, redirect chars), device sinks, and
    targets still carrying an unexpanded substitution — see :func:`_is_unresolved`.
    """
    if not target or _NOT_A_PATH.match(target):
        return False
    if _is_unresolved(target):
        return False
    return not _is_device_sink(target)


def command_has_unnameable_target(command: str) -> bool:
    """Does this command touch something whose destination we could not name?

    True when a clause carries a redirect or mutation operator but yields no
    trackable target — an unresolved substitution, a form the lexer cannot resolve
    to a filename — or when the command could not be tokenized at all. It is the
    signal that the verifier may be BLIND on this turn rather than that the turn was
    clean, and ``post_turn`` speaks up on it (issue #275).

    WHAT THE LEXER CHANGED HERE. This used to walk clause TEXT, re-run the pattern
    table to recover the operands the tracker had discarded, and then reason about
    operator position with a lookbehind. Every one of those steps was a place for the
    next unmet shape to slip through, and three did:

      * `echo hi>out.txt` was neither tracked nor reported, because the lookbehind
        that keeps `=>` from claiming a file also refuses to see an operator after
        `i`. The operator is now a token, so the shape is simply gone (#484's
        headline break).
      * `echo touch $FOO` was reported blind although nothing is mutated, because a
        verb matched wherever it sat in the clause. The command word decides now.
      * heredoc prose was read as commands, so `cat <<EOF ... echo x > /tmp/p ... EOF`
        claimed `/tmp/p` as a written file. The body is stripped before lexing.

    `2>&1` alone is still not blindness: an fd duplicate is correctly not-a-file, and
    a clause whose only redirect is one has nothing to audit. That is a different case
    from `> $LOG` and `mkdir -p $HOME/x`, which ARE blindness — something was touched
    and the name was lost.

    An UNPARSEABLE command counts as unnameable too. That is criterion 3 of #484: a
    command the instrument cannot tokenize must not read as a clean turn. ``post_turn``
    renders it under its own heading, since "I could not parse this" and "I parsed this
    and could not name the target" are different admissions.
    """
    audit = audit_bash_command(command)
    return audit.blind or audit.unaudited is not None

class FileMutationVerifier:
    """Per-turn tracker for claimed vs actual filesystem mutations.

    ONE instance is shared process-wide (see ``run_daemon``), so every entry
    point takes a ``turn_key`` identifying which in-flight turn it belongs to.
    Keys are minted per ``run_loop`` invocation
    omitting one falls back to
    :data:`DEFAULT_TURN_KEY`, which is correct only for single-threaded callers.

    Lifecycle:
      pre_tool_use(tool_name, tool_input, tool_use_id, turn_key=...)
        snapshots the affected path(s) before execution.
      post_tool_use(tool_name, tool_input, tool_use_id, output, is_error,
                    turn_key=...)
        snapshots again and records the diff.
      post_turn(turn_key=...) -> str | None
        returns a summary string when that turn has mutations pending,
        ``None`` otherwise, and drops the turn's record. The caller
        (agent_loop) decides where the summary goes (default: append as an
        injected turn so the model sees it on its next turn).
      discard_turn(turn_key=...)
        drops a turn's record without rendering — the cleanup path for turns
        that end early (iteration cap, circuit breaker, interrupt).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        truncate_after_n_mutations: int = 20,
    ) -> None:
        self._enabled = bool(enabled)
        self._truncate_n = max(1, int(truncate_after_n_mutations))
        # turn_key -> record, least-recently-touched first. Guarded by
        # ``_lock``: turns are driven by asyncio and interleave at every
        # ``await``, and gateways are free to drive the loop from a worker
        # thread, so map mutation must not race.
        self._turns: OrderedDict[str, _TurnRecord] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def new_turn_key(session_id: str | None = None) -> str:
        """Mint a key for one turn. Unique per call — a session id alone is
        NOT enough, since a session can have more than one turn in flight."""
        return f"{session_id or 'anon'}:{uuid4().hex}"

    @property
    def live_turns(self) -> int:
        """Number of turns currently holding state. Diagnostics only; a
        number that keeps climbing means a caller isn't draining."""
        with self._lock:
            return len(self._turns)

    # ------------------------------------------------------------------
    # Hook entry points — called by agent_loop
    # ------------------------------------------------------------------

    def pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
        *,
        turn_key: str | None = None,
    ) -> None:
        """Snapshot every path the tool is expected to touch."""
        if not self._enabled:
            return
        try:
            paths = self._paths_for(tool_name, tool_input)
            # A bash command that redirects somewhere unnameable gets recorded even
            # when nothing is trackable — that is EXACTLY the case that used to leave
            # no row at all, indistinguishable from a clean turn.
            if tool_name == "bash":
                command = str(tool_input.get("command", ""))
                audit = audit_bash_command(command)
                if audit.unaudited is not None:
                    with self._lock:
                        self._record(turn_key).unaudited.append(command[:200])
                elif audit.blind:
                    with self._lock:
                        self._record(turn_key).blind.append(command[:200])
            if not paths:
                # Don't materialise a record for a tool that touches nothing —
                # otherwise every bash `ls` allocates a turn slot.
                return
            snaps = {(tool_use_id, p): _snapshot(p) for p in paths}
            with self._lock:
                self._record(turn_key)._pending.update(snaps)
        except Exception:
            log.debug("FileMutationVerifier.pre_tool_use raised", exc_info=True)

    def post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
        *,
        output: str | None = None,
        is_error: bool = False,
        turn_key: str | None = None,
    ) -> None:
        """Diff snapshots and record one ``_Mutation`` per tracked path."""
        if not self._enabled:
            return
        try:
            paths = self._paths_for(tool_name, tool_input)
            if not paths:
                return
            claim = self._claim_from(tool_name, tool_input)
            per_path = self._claim_map(tool_name, tool_input)
            err = (output or "")[:200] if is_error else None
            with self._lock:
                turn = self._record(turn_key)
                for p in paths:
                    before = turn._pending.pop((tool_use_id, p), None)
                    if before is None:
                        # No pre-snapshot — happens if pre_tool_use raised or
                        # the post_tool_use receives a path the pre couldn't
                        # extract. Treat ``before`` as absent.
                        before = _Snapshot(exists=False)
                    turn.mutations.append(_Mutation(
                        tool=tool_name,
                        path=p,
                        claimed_action=per_path.get(p, claim),
                        before=before,
                        after=_snapshot(p),
                        error=err,
                    ))
        except Exception:
            log.debug(
                "FileMutationVerifier.post_tool_use raised", exc_info=True,
            )

    def landed_paths(self, *, turn_key: str | None = None) -> list[str]:
        """Paths this turn ACTUALLY changed on disk. Non-draining.

        The teeth of the outcome layer read this, so it is deliberately
        narrower than what :meth:`post_turn` renders: a mutation is included
        only when the before/after ``os.stat`` diff shows a real change. A
        tool that claimed a write and produced none is a REPORTING matter, not
        a boundary violation — nothing escaped.

        Ground truth, unlike anything available before dispatch. For ``bash``
        the pre-execution path guess comes from the lexer in
        :func:`audit_bash_command`, which is deliberately incomplete — it models
        SHELL syntax, so a mutation performed by a program the shell invokes
        (``make``, ``npm ci``, ``python -c "os.remove(p)"``) is outside what any
        lexer can see. This is the after-the-fact diff, so a write the walker DID
        name is confirmed by bytes rather than by parse.

        ⚠ HONEST LIMIT, and it decides what the caller may do with this:
        ``_Snapshot`` holds ``exists``/``size``/``mtime``/``mode`` and NO
        CONTENT. A caller can learn that a file was overwritten. It can never
        put the old bytes back. Detection, never containment.
        """
        with self._lock:
            turn = self._turns.get(self._key(turn_key))
            if turn is None:
                return []
            return [
                m.path for m in turn.mutations
                if _changed(m.before, m.after)
            ]

    def post_turn(self, *, turn_key: str | None = None) -> str | None:
        """Render THIS turn's summary and drop its record. Returns ``None``
        when the turn tracked nothing.

        Dropping the record also discards unmatched pre-snapshots — those came
        from a tool that failed before execution, or one the pre handler
        didn't recognise, and must not leak into a later turn.
        """
        with self._lock:
            turn = self._turns.pop(self._key(turn_key), None)
        if turn is None or not (turn.mutations or turn.blind or turn.unaudited):
            return None
        return self._format_summary(turn.mutations, turn.blind, turn.unaudited)

    def discard_turn(self, *, turn_key: str | None = None) -> None:
        """Drop a turn's record without rendering. Idempotent — safe to call
        after ``post_turn`` has already drained it."""
        with self._lock:
            self._turns.pop(self._key(turn_key), None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(turn_key: str | None) -> str:
        return turn_key or DEFAULT_TURN_KEY

    def _record(self, turn_key: str | None) -> _TurnRecord:
        """Get-or-create this turn's record. Caller holds ``_lock``."""
        key = self._key(turn_key)
        turn = self._turns.get(key)
        if turn is None:
            turn = _TurnRecord()
            self._turns[key] = turn
            while len(self._turns) > MAX_LIVE_TURNS:
                evicted, _ = self._turns.popitem(last=False)
                log.warning(
                    "FileMutationVerifier: evicted undrained turn %r "
                    "(>%d live) — that turn gets no summary; a caller is "
                    "not calling post_turn/discard_turn",
                    evicted, MAX_LIVE_TURNS,
                )
        else:
            self._turns.move_to_end(key)
        return turn

    def _paths_for(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> list[str]:
        """Return every path this tool call may touch."""
        if tool_name in _FS_TOOLS:
            return _extract_paths(tool_name, tool_input)
        if tool_name == "bash":
            command = str(tool_input.get("command", ""))
            return [p for p, _ in _extract_bash_paths(command)]
        return []

    def _claim_from(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> str:
        """Human-readable claim of what the tool said it would do."""
        if tool_name == "file_write":
            return "write"
        if tool_name == "file_edit":
            return "edit"
        if tool_name == "notebook_edit":
            return "edit_notebook"
        if tool_name == "bash":
            command = str(tool_input.get("command", ""))
            actions = [a for _, a in _extract_bash_paths(command)]
            if not actions:
                return "bash"
            return "/".join(sorted(set(actions)))
        return tool_name

    def _claim_map(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> dict[str, str]:
        """Per-path action, so one bash call is not described by one joined verb.

        ``_claim_from`` collapses every action in a command into one string, which
        is right for a file tool (one path, one verb) and wrong for bash. A single
        ``rm -f stale.txt && echo x > out.txt`` claims a DELETE of one path and a
        WRITE of another; stamping both with ``"delete/redirect_write"`` makes the
        deletion rule below undecidable, because the tag cannot tell whether the
        absent-file case it is looking at was a delete or a write.
        """
        if tool_name != "bash":
            return {}
        command = str(tool_input.get("command", ""))
        out: dict[str, str] = {}
        for path, action in _extract_bash_paths(command):
            # LAST write wins, not first. A path can be claimed twice in one
            # command — `rm -f x && echo y > x` yields delete then redirect_write
            # for the same x — and the final on-disk state is the outcome of the
            # LAST operation against it. Judging that state against the first
            # claim is how a failed write gets laundered into a clean ✓:
            #
            #   delete + absent->absent        -> "deleted (no-op)", ✓
            #   redirect_write + absent->absent -> "CLAIMED but FILE ABSENT", ⚠
            #
            # so `setdefault` here turned a write that never landed into a
            # successful deletion. Overwriting keeps the claim that the
            # observed state should actually be measured against.
            out[path] = action
        return out

    def _format_summary(
        self,
        muts: list[_Mutation],
        blind: list[str] | None = None,
        unaudited: list[str] | None = None,
    ) -> str:
        """Render the per-turn list into a single string. Truncates.

        A turn with no mutations still renders when something was BLIND: silence used
        to mean both "nothing was written" and "the writes were never extracted", and
        an audit that cannot tell those apart is counted on for a guarantee it is not
        making (issue #275).
        """
        blind = blind or []
        unaudited = unaudited or []
        if not muts and (blind or unaudited):
            lines = ["[FILE MUTATION VERIFIER]", "Nothing trackable this turn, but NOT a clean audit:"]
            lines.extend(self._blindness_lines(blind, unaudited))
            lines.append("   (this turn may have written files this hook could not see)")
            return "\n".join(lines)
        lines = ["[FILE MUTATION VERIFIER]", "Files touched this turn:"]
        shown = muts[: self._truncate_n]
        for m in shown:
            tag, badge = self._tag(m)
            size_note = ""
            if m.after.exists and tag in {"modified", "created"}:
                delta = m.after.size - m.before.size
                size_note = (
                    f" (+{delta} bytes)" if delta > 0
                    else f" ({delta} bytes)" if delta < 0
                    else f" ({m.after.size} bytes)"
                )
            error_note = f" — {m.error}" if m.error else ""
            lines.append(
                f"   {badge} {m.path} — {m.claimed_action}: "
                f"{tag}{size_note}{error_note}"
            )
        if len(muts) > self._truncate_n:
            lines.append(
                f"   ... and {len(muts) - self._truncate_n} more "
                f"(truncated at {self._truncate_n})"
            )
        # A partial audit must say it is partial. Listing what WAS seen while
        # staying quiet about what could not be is the same misleading silence,
        # just harder to notice because the row looks complete.
        lines.extend(self._blindness_lines(blind, unaudited))
        return "\n".join(lines)

    def _blindness_lines(self, blind: list[str], unaudited: list[str]) -> list[str]:
        """Render the two not-a-clean-audit kinds, each under its own wording.

        The blindness row no longer says "redirect": the contract was generalised
        past redirects in #486 and again by the lexer, so `mkdir -p $HOME/x` reaches
        it with no redirect anywhere in the clause. The phrase "no nameable target"
        is kept verbatim because it is the contract's public wording.
        """
        lines = []
        for cmd, count in _dedupe_with_counts(blind)[: self._truncate_n]:
            suffix = f" (×{count})" if count > 1 else ""
            lines.append(f"   ? no nameable target — {cmd}{suffix}")
        for cmd, count in _dedupe_with_counts(unaudited)[: self._truncate_n]:
            suffix = f" (×{count})" if count > 1 else ""
            lines.append(f"   ? UNAUDITED, could not parse this command — {cmd}{suffix}")
        return lines

    @staticmethod
    def _tag(m: _Mutation) -> tuple[str, str]:
        """Map a _Mutation into a (status, unicode-badge) pair."""
        status = _classify(m.before, m.after)
        if m.error:
            return status, "✗"
        if status == "missing" and m.claimed_action == "delete":
            # A DELETE whose file was already absent is a SUCCESS, not a failure:
            # absence is the outcome a deletion asks for, so the post-state already
            # matches the claim. Reporting it as "CLAIMED but FILE ABSENT" read the
            # result of the operation as evidence the operation had failed — exactly
            # inverted. `rm -f x` on a missing `x` exited 0 and did what was asked.
            #
            # Scoped to deletes only. For a write, edit or redirect, absent-before
            # plus absent-after is still the real defect this verifier exists for, and
            # still warns below.
            #
            # An `rm` WITHOUT `-f` on a missing file exits non-zero, so it arrives
            # with ``m.error`` set and is caught by the branch above as ✗ — this
            # cannot mask a deletion that actually refused to happen.
            return "deleted (no-op: already absent)", "✓"
        if status == "no_change":
            # The load-bearing silent-failure case: claimed write, no disk change.
            return "CLAIMED but NO CHANGE ON DISK", "⚠"
        if status == "missing":
            return "CLAIMED but FILE ABSENT", "⚠"
        return status, "✓"


def _dedupe_with_counts(commands: list[str]) -> list[tuple[str, int]]:
    """Collapse repeats of the same command into one row with a count.

    The volume of blindness rows went UP with the lexer: a mutation operand that could
    not be named now reports whether or not it was quoted, and `rm -f "$PROBE"` is a
    shape a probe loop can issue dozens of times in one turn. Nothing deduplicated
    before — a turn that retried one probe eight times rendered eight identical
    200-character lines, which was already true and is worse now.

    Deduplicating HERE rather than in the predicate is the point. The contract governs
    the RECORD — every unnameable write produces one — and the renderer governs the
    LINE. Suppressing the record to keep the report short would be trading a true
    signal for a quiet one, which is the #275 defect with better manners. Order of
    first appearance is preserved so the summary still reads chronologically.
    """
    counts: OrderedDict[str, int] = OrderedDict()
    for command in commands:
        counts[command] = counts.get(command, 0) + 1
    return list(counts.items())


def make_default_verifier(config: dict[str, Any] | None = None) -> "FileMutationVerifier":
    """Build a verifier from a (possibly partial) config block.

    Honours the spec:
        hooks:
          file_mutation_verifier:
            enabled: true
            truncate_after_n_mutations: 20

    ``show_in_telegram`` was specified in SPRINT-2 WS2 and implemented as far
    as an attribute, but no code ever read it — the summary has always been
    model-facing only, on every surface. It is gone rather than left as a
    setting that silently does nothing
    an unrecognised key here is ignored,
    so a config that still carries it keeps loading.
    """
    cfg = (
        ((config or {}).get("hooks") or {}).get("file_mutation_verifier")
        or {}
    )
    return FileMutationVerifier(
        enabled=cfg.get("enabled", True),
        truncate_after_n_mutations=cfg.get("truncate_after_n_mutations", 20),
    )
