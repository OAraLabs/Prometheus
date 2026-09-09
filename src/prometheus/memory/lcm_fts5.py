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

    Keeps the same characters *sanitize_fts5_query* keeps, lower-cases
    everything, and collapses runs of whitespace. The result is appropriate
    for inserting into an FTS5 content table.

    ⚠ THIS FUNCTION RAISED ``NameError`` ON EVERY CALL until 2026-09-09. It
    referenced ``_FTS5_SPECIAL``, a blocklist regex deleted when the sibling
    sanitiser was rewritten to whitelist-extraction — the rewrite updated the
    function that had failures to fix and left this one pointing at a name
    that no longer existed. It had no callers and no test, so nothing noticed;
    `ruff`'s F821 is what found it. Rewritten here on the same whitelist basis,
    so the two cannot drift apart again, and now tested.
    """
    if not text:
        return ""
    tokens = _WORD_TOKEN.findall(text)
    return " ".join(tokens).lower()
