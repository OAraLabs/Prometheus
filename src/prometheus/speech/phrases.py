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
_FENCES = ("```", "~~~")
# One or two fence characters at the very end of the buffer may be the start of a fence still arriving.
_PARTIAL_FENCE = re.compile(r"[`~]{1,2}\Z")
# Compared lower-case, without the final dot.
_ABBREVIATIONS = frozenset({"e.g", "i.e", "mr", "mrs", "ms", "dr", "vs", "st"})

_SPEAKABLE = (
    (re.compile(r"^\s*([-*_=])(?:\s*\1){2,}\s*$", re.M), ""),           # a rule line
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),                       # heading marks
    (re.compile(r"^\s*>\s?", re.M), ""),                                # quote marks
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),                            # bullets
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),                          # list numbers
    (re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)"), r"\1"),                # [text](url) is its text
    (re.compile(r"https?://[^\s<>()]*[^\s<>().,;:!?'\"]"), "link"),     # a bare URL
    (re.compile(r"\*\*|__"), ""),                                       # strong
    (re.compile(r"(?<!\w)[*_]|[*_](?!\w)"), ""),                        # emphasis, never inside snake_case
    (re.compile(r"`+"), ""),                                            # inline code marks
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


def _first_fence(buf: str) -> tuple[int, str] | None:
    """Where the first fence marker in *buf* starts, and which marker it is."""
    found = [(i, marker) for marker in _FENCES if (i := buf.find(marker)) >= 0]
    return min(found) if found else None


class PhraseCutter:
    """Cuts one reply. Feed it deltas; it returns phrases as they become speakable."""

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
        #: True once the cap stopped the reply; the cutter then takes nothing more.
        self.truncated = False
        self._buf = ""
        self._base = 0          # offset of _buf[0] in the reply's text
        self._spoken = 0        # characters spoken so far
        self._emitted = 0       # phrases spoken so far (drives the first-phrase rule)
        self._after_code = False

    @property
    def released(self) -> int:
        """Where the reply's text not yet given out as a phrase starts."""
        return self._base

    # ------------------------------------------------------------------ api
    def feed(self, delta: str) -> list[Phrase]:
        """Add *delta* to the reply; every phrase that became complete."""
        if self.truncated or not delta:
            return []
        self._buf += delta
        return self._drain()

    def boundary(self) -> list[Phrase]:
        """A tool call started: what is pending is a phrase now (unless it is inside an open fence)."""
        out = self._drain()
        if self.truncated or not self._buf or _first_fence(self._buf) is not None:
            return out
        return out + self._take(len(self._buf), code=False)

    def finish(self) -> list[Phrase]:
        """The reply ended: what remains is the last phrase (an open fence is skipped like a closed one)."""
        out = self._drain()
        if not self.truncated and self._buf:
            # After _drain, a fence can only be at the head of what remains.
            out += self._take(len(self._buf), code=_first_fence(self._buf) is not None)
        self._buf = ""
        return out

    def discard(self, keep_chars: int) -> None:
        """The reply's text was cut back to its first *keep_chars* characters (a dropped round)."""
        keep_chars = max(0, keep_chars)
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
            stripped = self._buf.lstrip()
            self._base += len(self._buf) - len(stripped)
            self._buf = stripped
            cut = self._next_cut()
            if cut is None:
                break
            out += self._take(*cut)
        return out

    def _next_cut(self) -> tuple[int, bool] | None:
        """``(where, is_code)`` for the next phrase in the buffer (which has no leading whitespace), or None."""
        buf = self._buf
        if not buf:
            return None
        fence = _first_fence(buf)
        if fence is not None and fence[0] == 0:
            marker = fence[1]
            close = buf.find(marker, len(marker))
            return None if close < 0 else (close + len(marker), True)
        end = len(buf)
        if fence is not None:
            end = fence[0]
        elif partial := _PARTIAL_FENCE.search(buf):
            end = partial.start()
        region = buf[:end]
        cut = self._sentence_or_line_cut(region)
        if cut is None:
            cut = self._clause_cut(region)
        if cut is None and fence is not None:
            cut = end  # text before a fence is a phrase of its own
        if cut is None and len(region) >= self.max_chars:
            cut = self._hard_cut(region)
        return None if cut is None else (cut, False)

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
        words = region[:at].split()
        if not words:
            return False
        word = words[-1].lstrip("(\"'“‘")
        if word.lower() in _ABBREVIATIONS:
            return True
        line_start = region.rfind("\n", 0, at) + 1
        return word.isdigit() and region[line_start:at].strip() == word

    def _clause_cut(self, region: str) -> int | None:
        need = self.first_min_words if self._emitted == 0 else self.clause_min_words
        for m in _CLAUSE_END.finditer(region):
            if len(region[: m.end()].split()) >= need:
                return m.end()
        return None

    def _hard_cut(self, region: str) -> int:
        window = region[: self.max_chars]
        for i in range(len(window) - 1, 0, -1):
            if window[i].isspace():
                return i + 1
        return self.max_chars

    def _take(self, cut: int, code: bool) -> list[Phrase]:
        """Give out ``_buf[:cut]`` as a phrase (none when it speaks as nothing)."""
        raw = self._buf[:cut].rstrip()
        start = self._base
        self._buf = self._buf[cut:]
        self._base += cut
        if code:
            if self._after_code:
                return []
            self._after_code = True
            text = CODE_SKIPPED
        else:
            text = speakable(raw)
            if not text:
                return []
            self._after_code = False
        if self._spoken + len(text) > self.spoken_cap:
            self.truncated = True
            self._buf = ""
            self._base = start
            return [Phrase(start, start, TOO_LONG)]
        self._spoken += len(text)
        self._emitted += 1
        return [Phrase(start, start + len(raw), text, code)]
