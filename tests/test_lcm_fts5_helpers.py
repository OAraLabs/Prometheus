"""The two FTS5 helpers keep the same definition of a token.

`tokenize_for_fts5` raised `NameError` on EVERY call until 2026-09-09. It
referenced `_FTS5_SPECIAL`, a blocklist regex deleted when `sanitize_fts5_query`
was rewritten to whitelist-extraction: the rewrite fixed the function that had
live failures and left this one pointing at a name that no longer existed.

Nothing noticed because it had no callers and no test — it was found by ruff's
F821 while wiring the lint gate, which is the argument for the gate in one line.

These tests exist so the pair cannot drift apart a second time.
"""

from __future__ import annotations

import pytest

from prometheus.memory.lcm_fts5 import sanitize_fts5_query, tokenize_for_fts5


@pytest.mark.parametrize("text, expected", [
    ("Hello, world.txt!", "hello world txt"),
    ("  MiXeD   Case  ", "mixed case"),
    ("a-b_c", "a b_c"),          # underscore is a word char, hyphen is not
    ("", ""),
    ("...", ""),                  # punctuation only -> nothing to index
    ("café Ünicode", "café ünicode"),
])
def test_tokenize_produces_lowercased_word_tokens(text, expected):
    assert tokenize_for_fts5(text) == expected


def test_tokenize_does_not_raise_on_anything_the_sanitiser_accepts():
    """The regression itself: any call at all used to be a NameError."""
    for text in ("plain", "with.punctuation", "sym!@#$%", "", "   "):
        tokenize_for_fts5(text)


@pytest.mark.parametrize("text", [
    "Hello, world.txt!", "a-b_c", "café Ünicode", "one two", "...",
])
def test_the_two_helpers_agree_on_what_a_token_is(text):
    """THE POINT. Both are whitelist-extraction over the same regex, so the
    indexed form and the queried form cannot disagree about tokenisation —
    which is exactly what the deleted blocklist constant caused.
    """
    indexed = tokenize_for_fts5(text).split()
    queried = [t.strip('"').lower() for t in sanitize_fts5_query(text).split()]
    assert indexed == queried
