"""A streaming reply cut into phrases a voice can speak (docs/design/speech-route.md, section 5).

The reply arrives as text deltas. :class:`PhraseCutter` gives back each phrase the moment it can be spoken, so the
voice box synthesises phrase 1 while the model is still writing phrase 3. Pure and synchronous: text in, phrases out,
no clock and no I/O.

Where a phrase ends, in order:

* **a sentence end**: ``.`` ``!`` ``?`` ``…`` (with any closing quote or bracket) and then whitespace. A decimal has no
  whitespace after its dot, so it never cuts. A few common abbreviations (``e.g.``, ``Mr.``) and a list number at the
  start of a line (``1.``) do not cut either;
* **a line break** after text, so a heading or a list item is a phrase of its own;
* **a clause end** (``,`` ``;`` ``:`` and whitespace), once the phrase has enough words: few for the very first
  phrase, because the time to the first word is what the listener feels, more after that;
* **the maximum length**: the last whitespace before ``max_chars``, or ``max_chars`` itself in one long token.

The caller adds two more: :meth:`PhraseCutter.boundary` when a tool call starts (what was said before it is spoken
while it runs), and :meth:`PhraseCutter.finish` when the reply ends.

**Code is never read and never cut.** A fenced block (```` ``` ```` or ``~~~``) is a phrase of its own, spoken as
:data:`CODE_SKIPPED`; blocks with only blank space between them are said once. No cut lands inside a fence, and an
unclosed one holds everything after it until it closes or the reply ends.

**What is spoken** is the phrase without its markdown (:func:`speakable`). **At most** ``spoken_cap`` characters are
spoken; then one :data:`TOO_LONG` phrase, and nothing more.

Each :class:`Phrase` carries ``start``/``end``: offsets into the reply's own text, in code points (the unit of
``chat_discard``'s ``keep_chars``), so a client knows which words a phrase covers and where unspoken text starts.

Source: novel code for Prometheus, 2026-11-13, modelled on OAra Voice's phrase chunker (same boundary order and
fence masking; line breaks, tool calls, abbreviations, offsets and the cap are new here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Spoken in place of a code block.
CODE_SKIPPED = "Code block skipped."
#: Spoken once when the cap is reached.
TOO_LONG = "The rest is too long to read aloud."

MAX_CHARS = 200
SPOKEN_CAP = 5000
FIRST_MIN_WORDS = 3
CLAUSE_MIN_WORDS = 6

_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s")
_PUNCT_RUN = re.compile(r"[.!?…]+")
_CLAUSE_END = re.compile(r"[,;:]\s")
# A fence opens with a run of three or more backticks or tildes, and closes with a run of the same character at least
# as long (so a ``` inside a ```` block does not close it).
_FENCE_RUN = re.compile(r"`{3,}|~{3,}")
# One or two fence characters at the very end of the buffer may be the start of a fence still arriving.
_PARTIAL_FENCE = re.compile(r"[`~]{1,2}\Z")
# Compared lower-case, without the final dot.
_ABBREVIATIONS = frozenset({"e.g", "i.e", "mr", "mrs", "ms", "dr", "vs", "st"})
#: A bare URL longer than a phrase is still one phrase (it speaks as "link"), up to this length.
URL_MAX = 2000
_URL_START = re.compile(r"https?://")

_SPEAKABLE = (
    (re.compile(r"^\s*([-*_=])(?:\s*\1){2,}\s*$", re.M), ""),              # a rule line
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),                          # heading marks
    (re.compile(r"^\s*>\s?", re.M), ""),                                   # quote marks
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),                               # bullets
    (re.compile(r"^\s*\d{1,3}[.)]\s+", re.M), ""),                         # list numbers (a year is not one)
    (re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)"), r"\1"),                   # [text](url) is its text
    (re.compile(r"https?://[^\s<>()]*[^\s<>().,;:!?'\"]"), "link"),        # a bare URL
    (re.compile(r"\*\*|__"), ""),                                          # strong
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),     # *emphasis*, in pairs only
    (re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)"), r"\1"),             # _emphasis_, never inside snake_case
    (re.compile(r"`+"), ""),                                               # inline code marks
)


def speakable(text: str) -> str:
    """*text* as it should be spoken: markdown taken out, whitespace collapsed. ``""`` when nothing is left."""
    for pattern, replacement in _SPEAKABLE:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Phrase:
    """One phrase: the reply's text from ``start`` to ``end`` (code points), spoken as ``text``."""

    start: int
    end: int
    text: str
    code: bool = False




def _run_end(buf: str, at: int) -> int:
    """Where the run of ``buf[at]`` starting at *at* ends."""
    end = at
    while end < len(buf) and buf[end] == buf[at]:
        end += 1
    return end


class PhraseCutter:
    """Cuts one reply. Feed it deltas; it returns phrases as they become speakable.

    Work per call is bounded by the phrase length, not by the size of what is pending: cuts are searched only in the
    first ``max_chars + 1`` characters, and inside an open fence only the unsearched tail is kept (code is never
    spoken, so its text is not needed, only where it starts and ends).
    """

    def __init__(
        self,
        *,
        max_chars: int = MAX_CHARS,
        first_min_words: int = FIRST_MIN_WORDS,
        clause_min_words: int = CLAUSE_MIN_WORDS,
        spoken_cap: int = SPOKEN_CAP,
    ) -> None:
        if max_chars < 20 or first_min_words < 1 or clause_min_words < 1 or spoken_cap < 1:
            raise ValueError("phrase cutter limits out of range")
        self.max_chars = max_chars
        self.first_min_words = first_min_words
        self.clause_min_words = clause_min_words
        self.spoken_cap = spoken_cap
        #: True once the cap stopped the reply; the cutter then takes nothing more (until a discard before it).
        self.truncated = False
        self._trunc_at = 0
        self._buf = ""
        self._base = 0          # offset of _buf[0] in the reply's text
        self._spoken = 0        # characters spoken so far
        self._emitted = 0       # phrases spoken so far (drives the first-phrase rule)
        self._after_code = False
        # Inside an open fence: (fence character, opening run length, where the fence starts).
        self._fence: tuple[str, int, int] | None = None
        # (end, spoken characters, is code) of each phrase given out, so a discard can take back what it drops.
        self._ledger: list[tuple[int, int, bool]] = []

    @property
    def released(self) -> int:
        """Where the reply's text not yet given out as a phrase starts."""
        return self._fence[2] if self._fence is not None else self._base

    # ------------------------------------------------------------------ api
    def feed(self, delta: str) -> list[Phrase]:
        """Add *delta* to the reply; every phrase that became complete."""
        if self.truncated or not delta:
            return []
        self._buf += delta
        return self._drain()

    def boundary(self) -> list[Phrase]:
        """A tool call started: what is pending is a phrase now (never inside a fence, never a fence half-arrived)."""
        out = self._drain()
        if self.truncated or self._fence is not None or not self._buf:
            return out
        keep = len(self._buf)
        if (fence := _FENCE_RUN.search(self._buf)) is not None:
            keep = fence.start()  # an opener still growing at the end; after _drain it is at the head
        elif (partial := _PARTIAL_FENCE.search(self._buf)) is not None:
            keep = partial.start()
        if self._buf[:keep].strip():
            out += self._take(keep)
        return out

    def finish(self) -> list[Phrase]:
        """The reply ended: what remains is the last phrase (an open fence is skipped like a closed one)."""
        out = self._drain()
        if not self.truncated:
            if self._fence is not None:
                out += self._code(self._fence[2], self._base + len(self._buf.rstrip()))
            elif self._buf:
                fence = _FENCE_RUN.match(self._buf)
                if fence is not None:
                    out += self._code(self._base, self._base + len(self._buf.rstrip()))
                else:
                    out += self._take(len(self._buf))
        self._buf = ""
        self._fence = None
        return out

    def discard(self, keep_chars: int) -> None:
        """The reply's text was cut back to its first *keep_chars* characters (a dropped round).

        A client drops the phrases it has whose ``end`` is past *keep_chars*, so they are taken back here too: their
        characters no longer count toward the cap, and a cap reached past *keep_chars* is lifted.
        """
        keep_chars = max(0, keep_chars)
        while self._ledger and self._ledger[-1][0] > keep_chars:
            _end, spoken, _code = self._ledger.pop()
            self._spoken -= spoken
            self._emitted -= 1
        self._after_code = bool(self._ledger) and self._ledger[-1][2]
        if self.truncated:
            if keep_chars >= self._trunc_at:
                return
            self.truncated = False
        if self._fence is not None and keep_chars <= self._fence[2]:
            self._fence = None
        if keep_chars >= self._base + len(self._buf):
            return
        if keep_chars >= self._base:
            self._buf = self._buf[: keep_chars - self._base]
        else:
            # Phrases past keep_chars were already given out; the caller drops the ones it has not sent.
            self._buf = ""
            self._base = keep_chars

    # ------------------------------------------------------------- internals
    def _drain(self) -> list[Phrase]:
        out: list[Phrase] = []
        while not self.truncated:
            if self._fence is not None:
                closed = self._scan_fence()
                if closed is None:
                    break
                out += closed
                continue
            stripped = self._buf.lstrip()
            self._base += len(self._buf) - len(stripped)
            self._buf = stripped
            if not self._buf:
                break
            window = self._buf[: self.max_chars + 1]
            fence = _FENCE_RUN.search(window)
            if fence is not None and fence.start() == 0:
                run = _run_end(self._buf, 0)
                if run == len(self._buf):
                    break  # the opening run may still be growing
                self._fence = (self._buf[0], run, self._base)
                self._buf = self._buf[run:]
                self._base += run
                continue
            cut = self._next_cut(window, fence.start() if fence is not None else None)
            if cut is None:
                break
            out += self._take(cut)
        return out

    def _scan_fence(self) -> list[Phrase] | None:
        """Inside an open fence: its code phrase once it closes, else None (keeping only what a closer could need)."""
        assert self._fence is not None
        char, length, start = self._fence
        at = self._buf.find(char * length)
        if at < 0:
            drop = max(0, len(self._buf) - (length - 1))
            self._buf = self._buf[drop:]
            self._base += drop
            return None
        end = _run_end(self._buf, at)
        self._buf = self._buf[end:]
        self._base += end
        self._fence = None
        return self._code(start, self._base)

    def _next_cut(self, window: str, fence_at: int | None) -> int | None:
        """Where the next text phrase ends in the buffer (no leading whitespace, no fence at its head), or None."""
        end = len(self._buf) if fence_at is None else fence_at
        if fence_at is None and (partial := _PARTIAL_FENCE.search(self._buf)) is not None:
            end = partial.start()
        region = window[: min(end, len(window))]
        cut = self._sentence_or_line_cut(region)
        if cut is None:
            cut = self._clause_cut(region)
        if cut is None and fence_at is not None and fence_at <= self.max_chars:
            cut = fence_at  # text before a fence is a phrase of its own
        if cut is None and end >= self.max_chars:
            cut = self._hard_cut(region)
        return cut

    def _sentence_or_line_cut(self, region: str) -> int | None:
        cuts = []
        for m in _SENTENCE_END.finditer(region):
            if not self._false_sentence_end(region, m.start()):
                cuts.append(m.end())
                break
        newline = region.find("\n")
        if newline > 0:  # the buffer has no leading whitespace, so text comes before it
            cuts.append(newline + 1)
        return min(cuts) if cuts else None

    @staticmethod
    def _false_sentence_end(region: str, at: int) -> bool:
        """A dot that ends an abbreviation or a list number, not a sentence."""
        punct = _PUNCT_RUN.match(region, at)
        if punct is None or punct.group() != ".":
            return False
        space = max(region.rfind(" ", 0, at), region.rfind("\n", 0, at), region.rfind("\t", 0, at))
        word = region[space + 1:at].lstrip("(\"'“‘")
        if word.lower() in _ABBREVIATIONS:
            return True
        line_start = region.rfind("\n", 0, at) + 1
        return word.isdigit() and len(word) <= 3 and region[line_start:at].strip() == word

    def _clause_cut(self, region: str) -> int | None:
        need = self.first_min_words if self._emitted == 0 else self.clause_min_words
        for m in _CLAUSE_END.finditer(region):
            if len(region[: m.end()].split()) >= need:
                return m.end()
        return None

    def _hard_cut(self, region: str) -> int | None:
        window = region[: self.max_chars]
        for i in range(len(window) - 1, 0, -1):
            if window[i].isspace():
                return i + 1
        if _URL_START.match(self._buf):
            # One long URL: one phrase, which speaks as "link", rather than its characters read out.
            limit = self._buf[:URL_MAX + 1]
            space = next((i for i, ch in enumerate(limit) if ch.isspace()), -1)
            if space > 0:
                return space
            if len(self._buf) <= URL_MAX:
                return None  # the URL is still arriving
        return self.max_chars

    def _take(self, cut: int) -> list[Phrase]:
        """Give out ``_buf[:cut]`` as a text phrase (none when it speaks as nothing)."""
        raw = self._buf[:cut].rstrip()
        start = self._base
        self._buf = self._buf[cut:]
        self._base += cut
        text = speakable(raw)
        if not text:
            return []
        self._after_code = False
        return self._spend(Phrase(start, start + len(raw), text))

    def _code(self, start: int, end: int) -> list[Phrase]:
        """A code block from *start* to *end*: said once for a run of blocks."""
        if self._after_code:
            return []
        self._after_code = True
        return self._spend(Phrase(start, end, CODE_SKIPPED, code=True))

    def _spend(self, phrase: Phrase) -> list[Phrase]:
        if self._spoken + len(phrase.text) > self.spoken_cap:
            self.truncated = True
            self._trunc_at = phrase.start
            self._buf = ""
            self._fence = None
            self._base = phrase.start
            return [Phrase(phrase.start, phrase.start, TOO_LONG)]
        self._spoken += len(phrase.text)
        self._emitted += 1
        self._ledger.append((phrase.end, len(phrase.text), phrase.code))
        return [phrase]
