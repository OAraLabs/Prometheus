"""FTS5 query sanitisation utilities for LCM stores.

SQLite FTS5 has a query syntax that treats certain characters as operators.
These helpers escape user-provided strings so they can be used safely in
MATCH expressions and indexing operations.
"""

from __future__ import annotations

import re

# Word tokens (unicode letters/digits/underscore). Everything else —
# operators, punctuation, quotes — is a separator. Whitelist-extraction
# instead of operator-blocklisting: the old blocklist regex missed ``.``
# (and ``?``, ``'``, ``,`` …), so model queries containing a filename or a
# question mark raised ``fts5: syntax error near "."`` at MATCH time
# (3 live lcm_expand_query failures, 0% tool success).
_WORD_TOKEN = re.compile(r"\w+", re.UNICODE)

# Collapse whitespace runs into a single space.
_WHITESPACE_RUN = re.compile(r"\s+")


def sanitize_fts5_query(query: str) -> str:
    """Render an arbitrary string safe for an FTS5 MATCH clause.

    Extracts word tokens and double-quotes each one (quoted FTS5 strings
    carry no operator meaning), joined by spaces — i.e. implicit AND over
    literal tokens, matching the old sanitiser's semantics for plain
    queries while being immune to any punctuation by construction.

    Returns an empty string if the input is blank or entirely punctuation,
    which callers should interpret as "no match filter".
    """
    if not query:
        return ""
    tokens = _WORD_TOKEN.findall(query)
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


def tokenize_for_fts5(text: str) -> str:
    """Produce a simple whitespace-normalised form suitable for FTS5 indexing.

    Strips the same special characters that *sanitize_fts5_query* removes,
    lower-cases everything, and collapses runs of whitespace.  The result is
    appropriate for inserting into an FTS5 content table.
    """
    if not text:
        return ""
    cleaned = _FTS5_SPECIAL.sub(" ", text)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    return cleaned.lower()
