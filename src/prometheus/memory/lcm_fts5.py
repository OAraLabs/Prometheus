"""FTS5 query sanitisation utilities for LCM stores.

SQLite FTS5 has a query syntax that treats certain characters as operators.
These helpers escape user-provided strings so they can be used safely in
MATCH expressions and indexing operations.
"""

from __future__ import annotations

import re

# Characters with special meaning inside FTS5 match expressions.
_FTS5_SPECIAL = re.compile(r'["\*\(\)\-\+\^:{}~@#]')

# Collapse whitespace runs into a single space.
_WHITESPACE_RUN = re.compile(r"\s+")


def sanitize_fts5_query(query: str) -> str:
    """Escape an arbitrary string for use in an FTS5 MATCH clause.

    * Removes special FTS5 operator characters.
    * Collapses whitespace.
    * Returns an empty string if the input is blank or entirely punctuation,
      which callers should interpret as "no match filter".
    """
    if not query:
        return ""
    cleaned = _FTS5_SPECIAL.sub(" ", query)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    return cleaned


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
