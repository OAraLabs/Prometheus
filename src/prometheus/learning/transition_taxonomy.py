"""Transition-type taxonomy for repair pairs (harvest-breadth sprint).

pairs/run measures CONTEXT diversity; what a LoRA actually learns is the
TRANSITION — the kind of repair (dict-unwrap vs fuzzy-rename vs …). A corpus
that is 1,000 dict-unwraps with different nouns teaches one lesson 1,000 times
and Goodharts the gym. This module buckets a (rejected → chosen) repair into
the transition class it exercises, so breadth can be measured, not assumed.

A pair is ``{"name": str, "input": dict}`` for both ``rejected`` and
``chosen`` (the shapes pair_capture stores). Classification is structural and
deterministic — no model, no I/O.
"""

from __future__ import annotations

from typing import Any

# Canonical buckets (also the histogram's key order).
TRANSITION_TYPES = [
    "dict_wrap_unwrap",        # {p: {p: v}} → {p: v}  (phantom self/foreign nesting)
    "fuzzy_rename",            # wrong tool name → corrected (levenshtein)
    "json_stuffed_string",     # a param is a JSON blob string → parsed object
    "missing_discriminator",   # chosen adds a `type`/discriminator the rejected lacked
    "type_coercion",           # a scalar param changed type (str↔int↔bool)
    "other",
]


def _looks_like_json(s: str) -> bool:
    t = s.strip()
    return len(t) >= 2 and t[0] in "{[" and t[-1] in "}]"


def classify_transition(
    pair_source: str | None,
    rejected: dict[str, Any] | None,
    chosen: dict[str, Any] | None,
) -> str:
    """Return the transition bucket a repair pair exercises.

    Order matters — the first structural signature that matches wins:
    fuzzy_rename (name change) → json_stuffed_string (blob param) →
    dict_wrap_unwrap (a dict param the chosen unwraps) → missing_discriminator
    → type_coercion → other.
    """
    if not isinstance(rejected, dict) or not isinstance(chosen, dict):
        return "other"
    r_name, c_name = rejected.get("name"), chosen.get("name")
    r_in = rejected.get("input") if isinstance(rejected.get("input"), dict) else {}
    c_in = chosen.get("input") if isinstance(chosen.get("input"), dict) else {}

    # Levenshtein / fuzzy tool-name repair.
    if r_name and c_name and r_name != c_name:
        return "fuzzy_rename"

    # A rejected param that's a JSON blob string the chosen form parses.
    for k, v in r_in.items():
        if isinstance(v, str) and _looks_like_json(v):
            cv = c_in.get(k)
            if not (isinstance(cv, str) and _looks_like_json(cv)):  # chosen parsed it
                return "json_stuffed_string"

    # Phantom dict nesting the chosen form unwraps (self-keyed {p:{p:v}} or
    # foreign {p:{q:…}}). The signature: a rejected param is a dict, and the
    # chosen value for it is no longer that same dict.
    for k, v in r_in.items():
        if isinstance(v, dict):
            cv = c_in.get(k)
            if not isinstance(cv, dict) or cv != v:
                return "dict_wrap_unwrap"

    # A discriminator (task_create.type, …) the rejected omitted and chosen supplied.
    for disc in ("type", "kind", "mode"):
        if disc in c_in and disc not in r_in:
            return "missing_discriminator"

    # A scalar param whose type changed (string "5" → 5, "true" → True, …).
    for k in set(r_in) & set(c_in):
        rv, cv = r_in[k], c_in[k]
        if not isinstance(rv, (dict, list)) and not isinstance(cv, (dict, list)):
            if type(rv) is not type(cv):
                return "type_coercion"

    return "other"


def histogram(pairs: list[tuple[str | None, dict | None, dict | None]]) -> dict[str, int]:
    """Bucket a list of (pair_source, rejected, chosen) into the taxonomy.
    Returns a dict over every TRANSITION_TYPES key (zeros included)."""
    counts = {t: 0 for t in TRANSITION_TYPES}
    for src, rej, cho in pairs:
        counts[classify_transition(src, rej, cho)] += 1
    return counts
