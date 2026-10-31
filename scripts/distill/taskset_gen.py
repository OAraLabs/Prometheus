#!/usr/bin/env python3
"""Gym taskset generator: varied teacher inputs built by code from real user requests.

Runs on the host that holds the traces (stdlib only; ``tokenizers`` is used
for prompt token counts when present). It writes the input file the local
teacher (``teacher_gen.py``) runs through the gym.

    nice -n 19 ionice -c3 python3 taskset_gen.py build --name real-v1 [--seed N] [--dry-run] \\
        [--traces '~/.prometheus/trajectories/golden_traces_*.jsonl'] \\
        [--index ~/distill/phase1/trace_index.jsonl] [--lcm-snapshot ~/distill/snapshot/lcm.db] \\
        [--holdout ~/distill/phase3/build-v2/eval-tools.jsonl --holdout .../val-cases.jsonl] \\
        [--injection-manifest ~/distill/phase3/evalsets/eval-injection.manifest.json] \\
        [--repo ~/Prometheus] [--wiki ~/.prometheus/wiki] [--wiki-sections topics] \\
        [--pack ~/distill/phase3/request_pack.json] [--cron-jobs ~/.prometheus/data/cron_jobs.json] \\
        [--out ~/distill/phase2b/tasksets] [--exclude-sessions ~/distill/policy/exclude-sessions.txt]
    python3 taskset_gen.py validate ~/distill/phase2b/tasksets/real-v1.jsonl
    # Drop every row built from a prompt of a listed session (DISTILL-OPEN-QUESTIONS #6: the
    # 2026-10-13 incident session is in no training data) -> real-v1-excl.jsonl + .exclusion.json;
    # --check writes nothing and exits 1 while the taskset still holds such a row:
    python3 taskset_gen.py exclude --tasks ~/distill/phase2b/tasksets/real-v1.jsonl \\
        --exclude-sessions ~/distill/policy/exclude-sessions.txt [--check] \\
        [--traces '~/.prometheus/trajectories/golden_traces_*.jsonl'] [--lcm-snapshot ~/distill/snapshot/lcm.db]

Excluded sessions: a row's prompts are its ``source_pattern_hash`` and every
``sources[].pattern_hash`` (a recombined row's partner too). ``exclude`` drops a
row when any of them is the task proxy of a user message the listed sessions
hold, in their golden traces (every user message, any surface) or their
snapshot rows (however labelled). ``build --exclude-sessions`` drops those
prompts before mining. Only counts and input hashes are written, never the ids.

The rule this file exists to keep
---------------------------------
No model writes task text. Every prompt is assembled by code from:

* spans of real user prompts (sentences, and the text between slots),
* typed slot values from real inventories on the host: git-tracked repo
  paths, file names, directories, doc headings and doc URLs (``git ls-files``,
  files with local edits skipped), wiki page titles, section headings and
  paths (``--wiki-sections``, read only), tool names from the registry
  (``--pack`` / ``--tool-schemas``), job names (``src/prometheus/jobs`` and
  the cron store's job *names*), package names under ``src/prometheus``,
  commit ids (``git log``),
* generated values (numbers, versions, ISO and month-name dates, weekdays,
  clock times, cron expressions, issue numbers), and
* a newline between two recombined pieces when the first has no trailing
  whitespace of its own.

Each output row records which real spans (character offsets into each source
prompt) and which slot values made it. ``verify_provenance`` rebuilds every
prompt from exactly those parts and checks every value against its pool or
generator; it runs on every row before anything is written, and the build
fails on any mismatch.

Inputs (all read only)
----------------------
* Golden traces (``--traces``): the last user message before each teacher
  call is a real prompt; prompts group by the Phase 1 task proxy
  ``h(norm(prompt))`` (ported from ``build.py``; a test holds them equal),
  which is also ``source_pattern_hash``. Only sessions on a human surface
  (``--surfaces``, default ``build.py``'s) count: probe, smoke and verify
  sessions may hold machine-written text.
* Phase 1 ``trace_index.jsonl`` (``--index``): per-trace category, joined by
  (file, ordinal) like ``build.py``; ``h_prompt`` disagreements are counted.
* ``--lcm-snapshot``: a snapshot of the conversation store, opened
  ``mode=ro``; never the live store (refused). Its user-typed rows
  (provenance ``user``, trusted, no tool-result blocks) on human surfaces add
  the requests that never reached a golden trace. The golden exporter drops
  provenance, so a turn the daemon injected as ``user`` (a task or worker
  report, a model-written re-engagement prompt, a retry notice) looks like a
  human prompt there: a real prompt whose normalized text matches any
  snapshot user row with another provenance (or untrusted) is excluded
  (``injected_turn``), and with a snapshot a golden prompt is kept only when
  the snapshot holds it as a user-typed row (``golden_not_user_typed``).
  Legacy rows read back as ``user``, so known injected openers are also
  refused on any line (``system_injected``), as is text that speaks of "the
  user" / "the owner" in the third person (``model_written``).
* ``--holdout`` (default: build-v2 ``eval-tools.jsonl`` and ``val-cases.jsonl``):
  every ``meta.task_id`` / ``meta.group_id`` and every user message in those
  rows. A real prompt or a variant is excluded when its task proxy or
  normalized text equals one of them (``holdout_task``), or when at least
  ``--holdout-containment`` of its word 5-grams appear in one held-out user
  message or that much of one held-out message appears in it; a held-out
  message under 5 words counts only when it is most of the text
  (``holdout_near_dup``). The same test runs again with every slot span (of
  the text and of each held-out message) masked to its kind, so a prompt that
  differs from a held-out one only in a path, name, number or date is
  excluded too (``holdout_slot_variant``).
* ``--injection-manifest``: ``build_eval_sets.InjectionManifest``; a prompt or
  variant sharing an 8-token shingle, a marker or a whole instruction with an
  eval-injection case is excluded.
* Person names: titles of the wiki's ``people`` / ``clients`` pages, used only
  as a deny list (never written anywhere, not even counted per name).

Mining
------
1. Filter real prompts: too short or long, system-injected (daemon / worker /
   compaction markers, a leading ``[``, ``<`` or ``{``), pasted markup (fenced
   code, headings, bold, bullet lists: likely pasted model output), code or log
   pastes, prompts naming another assistant, personal data (emails, user@host,
   phone numbers, keys and tokens, credentials, URLs carrying a secret in a
   query key or a token-shaped path segment, webhook URLs, IPs other than
   loopback, MAC addresses, card and SSN shapes, tailnet hosts), known person
   names, and a home-directory path anywhere (quoted code, ``host:`` or
   ``file://`` forms included) that is not a refillable path slot
   (``home_path_literal``). A template with personal data is dropped, never
   filled.
2. Extract typed slots (non-overlapping, by priority): url, quoted span (typed
   by content: path/file/url/tool/job/topic; code-like quotes are protected
   literal text), path, file, ISO date, month-name date, cron expression,
   clock time, version, commit id, issue number, registry tool name, job name,
   wiki page title, multi-word doc/wiki heading (topic), ``src/prometheus``
   package name (component), weekday, number. A title, heading or package
   name becomes a slot only when it holds a distinctive word (``Lexicon``:
   absent from ``--dictionary``, or a digit, ``_``, ``-`` or inner capital),
   so everyday words like "memory" stay literal; any 8+ character title
   counts once the prompt mentions the wiki. A package name that is common
   tech vocabulary (``config``, ``cli``, ``mcp``, ``utils``...) is a slot only
   when written as a code unit (``prometheus.<name>``, ``<name> module`` /
   ``package``). A lowercase ``may`` is a month only with an ordinal or
   ``of`` ("3rd of may"), never the modal verb.
   A path under a home directory must be refilled (the user name never
   survives).
3. Sentences split on ``.?!`` + whitespace or a newline, never inside a slot.
4. Intent (summarize, remember, schedule, plan, explain, research, fix, run,
   create, edit, find, check, question, request) from the request after its
   polite opener: the leading verb first (every rule's anchored form), then
   a keyword anywhere, then a leading auxiliary ("is", "can"...), so "Delete
   the old summary emails" is an edit, not a summary. Subcategory
   (scheduling, memory, wiki, daemon-ops, research, coding, general) by
   keyword score, falling back to the Phase 1 category; a Phase 1 ``coding``
   label adds a coding vote and wins ties, so coding work is not relabelled
   past ``--coding-cap``.
5. Near-duplicate real prompts (word-set Jaccard >= ``--source-merge``) merge
   into one pattern; its most-used member is the template.

Variants (seeded; ``--per-pattern`` per pattern, ``--max-uses`` per real prompt)
---------------------------------------------------------------------------------
* ``slot_fill``: every slot redrawn (inventory or generator).
* ``slot_swap``: one content slot redrawn, the rest keep their real values.
* ``recombine``: the template's leading sentences followed by another real
  template's trailing sentences, both of the same intent (same subcategory
  preferred); content slots redrawn with p=0.5, numbers always.

A candidate is kept only if it is new against every real prompt read
(normalized text: a variant ``build.py`` would file under a real task is
dropped), new against every kept variant (normalized text and word 4-gram
Jaccard < ``--variant-dup``), free of personal data and person names, clear
of the holdout and the injection manifest, and within the length limits.

Modes, categories, flags
------------------------
* ``mode: chat`` (``flags.suppress_tools``) for ``--chat-share`` of the tasks
  whose intent is explain/summarize/plan/question, that name no path, file or
  URL and do not ask about live state; everything else is ``agentic``.
* ``category`` uses ``build.py``'s vocabulary so ``teacher_gen.py`` can copy it
  to ``_meta.category``: ``chat-mode`` for chat tasks, otherwise memory/wiki ->
  ``memory/wiki``, scheduling/daemon-ops -> ``sentinel/daemon``, coding ->
  ``coding``, research/general -> ``other``. ``subcategory`` keeps the detail.
* Agentic ``coding`` tasks are capped at ``--coding-cap`` of the output
  (round-robin over patterns, so the cap keeps coverage).
* ``flags.inject_errors`` on exactly ``--error-share`` of agentic tasks and
  ``flags.force_compaction`` on exactly ``--compaction-share`` of all tasks,
  chosen by seeded hash rank.

Output: ``<out>/<name>.jsonl`` (mode 0600; one task per line, round-robin
over patterns so any prefix covers as many patterns as possible)::

    {"id": "ts-<16 hex>",                 # "ts-" + h(norm(prompt)): unique
     "taskset": "<name>",
     "category": "chat-mode|memory/wiki|sentinel/daemon|coding|other",
     "subcategory": "scheduling|memory|wiki|daemon-ops|research|coding|general",
     "intent": "<intent>",
     "mode": "chat|agentic",
     "prompt": "<the task text>",
     "slots": [{"name": "path_1", "type": "path", "value": "...",
                "from": "repo|repo-docs|wiki|registry|jobs|cron|git|generated|original",
                "source": 0, "span": [start, end]}],   # offsets in that source's prompt
     "template_id": "tpl-<16 hex>",       # sources[0]
     "source_pattern_hash": "<16 hex>",   # sources[0]; Phase 1 task proxy
     "task_group": "tg-<16 hex>",         # "tg-" + source_pattern_hash: shared
                                          #  by every variant of one real prompt
     "transform": "slot_fill|slot_swap|recombine",
     "sources": [{"template_id": ..., "pattern_hash": ..., "origin": "golden|lcm",
                  "span": [start, end]}],
     "flags": {"suppress_tools": bool, "inject_errors": bool, "force_compaction": bool},
     "provenance": "real-pattern-v1"}

and ``<out>/<name>.report.json``: aggregates only (no text, ids or hashes):
inputs, inventory sizes, exclusions by reason, patterns, slot types, counts
per category / mode / subcategory / intent / transform / flag, variants per
pattern, prompt lengths, diversity (distinct-1/2/3 and sampled self-BLEU-4 of
the output and of the real prompts it came from) and the checks
(``holdout_overlaps`` / ``injection_overlaps`` are ``null``, with
``holdout_used`` / ``injection_manifest_used`` false, when that check was
switched off: an unchecked taskset never reads as a clean one).

``teacher_gen.py`` contract
---------------------------
Read rows with ``load_taskset`` (it runs ``validate_task``). Per row: send
``prompt`` as the user turn; with ``flags.suppress_tools`` send no tools
(``tool_choice: none``); ``inject_errors`` / ``force_compaction`` are
execution conditions for the gym run. Tag every trace row ``_meta.source =
"qwen38-27b-gym"``, ``_meta.task_id = task_group`` (not ``id``: ``build.py``
unions rows on ``_meta.task_id``, so every variant of one real prompt lands
in one split; with ``id`` two slot-fill siblings could split across train
and eval-tools), ``_meta.category = category``, ``_meta.suppress_tools``
(chat), ``_meta.compacted`` / ``_meta.error_recovery`` when the condition
actually fired, and carry ``provenance`` through. A recombined row joins its
primary's group only: ``build.py`` reads one ``_meta.task_id``, so the
partner's group (``"tg-" + sources[1].pattern_hash``) is not joined today;
carry it as ``_meta.partner_task_id`` for when ``build.py`` unions on it.

Determinism: one ``--seed``; every draw comes from ``random.Random`` streams
keyed by the seed and a template id, inputs are read in sorted order, so the
same seed over the same inputs gives a byte-identical task file.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


def _load_sibling(name: str, filename: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} next to {Path(__file__).name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The eval-set builder's hashing (injection manifest, shingles, tracked files),
# so disjointness is judged exactly the way the eval sets were built.
BES = _load_sibling("distill_build_eval_sets", "build_eval_sets.py")

TASKSET_SCHEMA = "distill-taskset/1"
REPORT_SCHEMA = "distill-taskset-report/1"
PROVENANCE = "real-pattern-v1"
GENERATOR_VERSION = "1"
DEFAULT_SEED = 20261031
DEFAULT_OUT = "~/distill/phase2b/tasksets"
DEFAULT_TRACES = "~/.prometheus/trajectories/golden_traces_*.jsonl"
DEFAULT_INDEX = "~/distill/phase1/trace_index.jsonl"
DEFAULT_LCM = "~/distill/snapshot/lcm.db"
DEFAULT_HOLDOUT = ("~/distill/phase3/build-v2/eval-tools.jsonl",
                   "~/distill/phase3/build-v2/val-cases.jsonl")
DEFAULT_MANIFEST = "~/distill/phase3/evalsets/eval-injection.manifest.json"
DEFAULT_REPO = "~/Prometheus"
DEFAULT_WIKI = "~/.prometheus/wiki"
DEFAULT_WIKI_SECTIONS = ("topics",)
PERSON_SECTIONS = ("people", "clients")
DEFAULT_PACK = "~/distill/phase3/request_pack.json"
DEFAULT_CRON = "~/.prometheus/data/cron_jobs.json"
LIVE_STORE = "~/.prometheus"
HUMAN_SURFACES = ("beacon:", "telegram:", "slack:", "discord:")  # build.py's

CATEGORIES = ("chat-mode", "memory/wiki", "sentinel/daemon", "coding", "other")
SUBCATEGORIES = ("scheduling", "memory", "wiki", "daemon-ops", "research", "coding", "general")
SUBCAT_TO_CATEGORY = {"scheduling": "sentinel/daemon", "daemon-ops": "sentinel/daemon",
                      "memory": "memory/wiki", "wiki": "memory/wiki", "coding": "coding",
                      "research": "other", "general": "other"}
MODES = ("chat", "agentic")
TRANSFORMS = ("slot_fill", "slot_swap", "recombine")
INTENTS = ("summarize", "remember", "schedule", "plan", "explain", "research", "fix", "run",
           "create", "edit", "find", "check", "question", "request")
CHAT_INTENTS = frozenset({"explain", "summarize", "plan", "question"})
FROM_VALUES = ("repo", "repo-docs", "wiki", "registry", "jobs", "cron", "git", "generated",
               "original")


class TasksetError(Exception):
    pass


# ---------------------------------------------------------------------------
# Phase 1 / build.py functions (ported unchanged; a test holds them equal)
# ---------------------------------------------------------------------------

_HEX = re.compile(r"\b[0-9a-f]{6,}\b", re.I)
_NUM = re.compile(r"\d+")
_WS = re.compile(r"\s+")
DAEMON_PROMPT = re.compile(
    r"^\s*(A worker you started|\[Restored after a restart|\[SENTINEL|SENTINEL\b|\[cron|"
    r"\[Scheduled|Scheduled (job|task|prompt)|Heartbeat|\[heartbeat|\[Task result|"
    r"Task: .*\(t[a-z0-9]+\))", re.I)
COMPACTION = re.compile(r"\[End of compressed context\]|\[Compacted summary of turns|"
                        r"\[microcompacted\]|\[content pruned|tool result trimmed by the compactor")


def norm(s: str) -> str:
    s = (s or "").lower()
    s = _HEX.sub("h", s)
    s = _NUM.sub("0", s)
    return _WS.sub(" ", s).strip()


def h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def rank(seed: int, key: str, ident: str) -> float:
    """A seeded uniform draw in [0, 1) fixed by (seed, key, ident)."""
    d = hashlib.sha256(f"{seed}\x1f{key}\x1f{ident}".encode()).hexdigest()
    return int(d[:13], 16) / float(16 ** 13)


def tokens(text: str) -> list[str]:
    return BES.norm_tokens(text)


def ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    if len(toks) < n:
        return [tuple(toks)] if toks else []
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _ihash(parts: tuple[str, ...]) -> int:
    return int.from_bytes(hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
# Personal data and prompt filters
# ---------------------------------------------------------------------------

_PII_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*(?:PRIVATE KEY|CERTIFICATE)")),
    ("email", re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
    ("user_at_host", re.compile(r"(?<![\w.+-])[A-Za-z_][\w.-]*@[A-Za-z][\w-]*")),
    ("api_key", re.compile(
        r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_\w{20,}"
        r"|\bxox[abprs]-[\w-]{10,}|\bAKIA[0-9A-Z]{16}\b|\bAIza[\w-]{30,}|\bhf_[A-Za-z0-9]{20,}"
        r"|\beyJ[\w-]{8,}\.[\w-]{8,}")),
    ("credential", re.compile(
        # a bare key name, or a compound one ending in it (access_token, client-secret, x-api-key)
        r"(?i)\b(?:[\w-]*[_-])?(?:api[_-]?key|secret|token|password|passwd|pwd|bearer|passcode)\b"
        r"\s*[:=]\s*\S{4,}"
        r"|(?i:\b(?:[\w-]*[_-])?(?:key|sig|signature)\s*=\s*\S{4,})"
        r"|(?i:\b(?:password|passcode|pin)\s+is\s+\S+)")),
    ("phone", re.compile(
        r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-]|\.\d)"
        r"|(?<![\w+])\+\d{1,3}(?:[\s.-]?\d){8,13}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("mac", re.compile(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b")),
    ("tailnet_host", re.compile(r"(?i)\b[\w-]+(?:\.[\w-]+)*\.ts\.net\b")),
    ("hex_secret", re.compile(r"(?i)(?<![\w-])[0-9a-f]{41,}(?![\w-])")),
)
_IPV4 = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\w|\.\d)")
_IPV6 = re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])")
_RANDOM_TOKEN = re.compile(r"(?<![\w/.-])[A-Za-z0-9+/_=-]{32,}(?![\w/.-])")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_ANY_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"'`)\]}]+")
_SECRET_QUERY = re.compile(r"(?i)[?&](?:[\w-]*(?:token|key|sig|secret|auth)[\w-]*)=")
#: URLs that are themselves a credential (anyone holding one can post)
_WEBHOOK_URL = re.compile(
    r"(?i)\b(?:hooks\.slack\.com/(?:services|workflows|triggers)/|discord(?:app)?\.com/api/webhooks/"
    r"|[\w-]+\.webhook\.office\.com/|outlook\.office(?:365)?\.com/webhook|hooks\.zapier\.com/hooks/"
    r"|api\.telegram\.org/(?:file/)?bot\d)")
#: a home directory and the user-name component after it, wherever it appears
#: (a bare path, quoted code, ``host:/home/x``, ``file:///Users/x``)
HOME_ANY = re.compile(r"(?:^|(?<=[^\w]))(?:/home|/Users|/root|/var/home)/(?P<user>[^/\s`\"']+)")


def _token_shaped(s: str, min_len: int = 20) -> bool:
    return (len(s) >= min_len and sum(c.isupper() for c in s) >= 3 and sum(c.islower() for c in s) >= 3
            and sum(c.isdigit() for c in s) >= 3)


def secret_url(url: str) -> bool:
    """*url* carries a credential: a secret-named query key, a webhook shape,
    or a path / query segment shaped like a random token."""
    if _SECRET_QUERY.search(url) or _WEBHOOK_URL.search(url):
        return True
    rest = url.split("://", 1)[-1]
    return any(_token_shaped(seg) for seg in re.split(r"[/?&=#:;,]", rest))


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def pii_kinds(text: str) -> list[str]:
    """The kinds of personal data or secrets in *text* (empty: none found)."""
    found = {name for name, rx in _PII_RES if rx.search(text)}
    for m in _IPV4.finditer(text):
        octets = [int(x) for x in m.groups()]
        if all(o <= 255 for o in octets) and octets[0] != 127 and octets != [0, 0, 0, 0]:
            found.add("ip")
            break
    for m in _IPV6.finditer(text):
        s = m.group(0)
        groups = [g for g in s.split(":") if g]
        if s in ("::", "::1") or not groups:
            continue
        if ("::" in s and len(groups) >= 2 and len(s) >= 6) or len(groups) >= 5:
            found.add("ip")
            break
    for m in _RANDOM_TOKEN.finditer(text):
        if _token_shaped(m.group(0), 32):
            found.add("token")
            break
    if _WEBHOOK_URL.search(text) or any(secret_url(m.group(0)) for m in _ANY_URL.finditer(text)):
        found.add("secret_url")
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.add("card")
            break
    return sorted(found)


_NAME_STOP = frozenset("""
the and for inc llc ltd corp company group client clients people person team project projects
notes note page index about home new old main test misc general contact contacts info
""".split())


#: words that put a name in a person's place ("ask jordan", "email jordan", "jordan's")
_PERSON_CUE = (r"ask|tell|call|email|e-mail|text|message|msg|ping|dm|remind|cc|meet|meeting with|invite|"
               r"thank|thanks|with|from|by|for|to|and")


class PersonNames:
    """A deny list of person and client names (wiki ``people`` / ``clients``
    page titles). Multi-word names match case-insensitively. A single name
    token matches capitalized ("Jordan") or in capitals ("JORDAN"); in
    lowercase only next to a person cue ("ask jordan", "jordan's"), so an
    ordinary word ("the jordan river") stays a word. A false hit costs one
    real prompt; a miss puts a contact's name in task text."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        phrases: set[str] = set()
        toks: set[str] = set()
        for name in names:
            parts = [p for p in re.split(r"[\s_.-]+", name.strip()) if p]
            if len(parts) >= 2:
                phrases.add(" ".join(p.lower() for p in parts))
            for p in parts:
                if len(p) >= 3 and p.isalpha() and p.lower() not in _NAME_STOP:
                    toks.add(p[:1].upper() + p[1:].lower())
        self.count = len(phrases) + len(toks)
        self._phrase = (re.compile(r"(?i)(?<![\w])(?:" + "|".join(
            r"\s+".join(map(re.escape, p.split())) for p in sorted(phrases, key=len, reverse=True))
            + r")(?![\w])") if phrases else None)
        ordered = sorted(toks, key=len, reverse=True)
        alt = "|".join(map(re.escape, ordered))
        self._tok = (re.compile(r"(?<![\w])(?:" + alt + "|" + "|".join(re.escape(t.upper()) for t in ordered)
                                + r")(?![\w])") if toks else None)
        self._tok_cue = (re.compile(
            rf"(?i)(?<![\w])(?:(?:{_PERSON_CUE})\s+(?:{alt})(?![\w])|(?:{alt})['’]s(?![\w]))")
            if toks else None)

    @classmethod
    def from_wiki(cls, wiki: Path | None, sections: Iterable[str] = PERSON_SECTIONS) -> "PersonNames":
        names: list[str] = []
        for section in sections:
            d = (wiki / section) if wiki else None
            if d is None or not d.is_dir():
                continue
            for p in sorted(d.rglob("*.md")):
                names.append(p.stem)
                title = _first_heading(p)
                if title:
                    names.append(title)
        return cls(names)

    def mentions(self, text: str) -> bool:
        return bool((self._phrase and self._phrase.search(text))
                    or (self._tok and self._tok.search(text))
                    or (self._tok_cue and self._tok_cue.search(text)))


_MARKUP_LINE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", re.M)
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s+\S", re.M)
_CODE_CHARS = frozenset("{}()[];=<>$|\\")
_OTHER_ASSISTANT = re.compile(r"(?i)\b(?:claude|anthropic|chatgpt|codex)\b")
_INJECTED_START = re.compile(r"^\s*[\[<{]")
#: openers of turns the daemon injects as role ``user`` (task and worker
#: reports, retry notices), on any line: the golden exporter keeps no
#: provenance, and legacy store rows read back as ``user``
_INJECTED_MARKERS = re.compile(
    r"^\s*(?:A (?:background task|worker) you started (?:has finished|did not finish)"
    r"|Your previous response contained\b"
    r"|(?:Task|Worker): .*\(t[a-z0-9]{4,}\)"
    r"|Status: \w+ \(exit code\b"
    r"|-{3} (?:its )?output \(tail"
    r"|Available tools: )", re.M)
#: an instruction about "the user" / "the owner" in the third person: a
#: model-written re-engagement prompt, not something a person types to the agent
_MODEL_WRITTEN = re.compile(
    r"(?i)\b(?:tell|notify|inform|let|ping|update|message|warn|alert|ask|remind|report (?:back )?to)"
    r" the (?:user|owner)\b")


def prompt_problem(text: str, s: "Settings") -> str | None:
    """Why a real prompt cannot be a template (None: it can)."""
    if len(text) < s.min_chars or len(text.split()) < s.min_words:
        return "too_short"
    if len(text) > s.max_prompt_chars:
        return "too_long"
    if (_INJECTED_START.search(text) or DAEMON_PROMPT.search(text) or COMPACTION.search(text)
            or _INJECTED_MARKERS.search(text)):
        return "system_injected"
    if _MODEL_WRITTEN.search(text):
        return "model_written"
    if ("```" in text or len(_MARKUP_LINE.findall(text)) >= 3 or _HEADING_LINE.search(text)
            or text.count("**") >= 2):
        return "pasted_markup"
    if text.count("\n") + 1 > s.max_lines:
        return "multi_line_paste"
    if sum(1 for c in text if c in _CODE_CHARS) / len(text) > 0.06:
        return "code_or_log_paste"
    if _OTHER_ASSISTANT.search(text):
        return "names_other_assistant"
    return None


# ---------------------------------------------------------------------------
# Intent and subcategory
# ---------------------------------------------------------------------------

_POLITE = re.compile(
    r"^(?:(?:hey|hi|hello|ok(?:ay)?|so|now|alright|please|pls|also|and|then|great|thanks|"
    r"thank you|cool|nice|perfect|yes|yeah|yep|no|nope|sure|right|well)\b[\s,.!:-]*"
    r"|(?:can|could|would|will) you\s+(?:please\s+)?|i (?:need|want|would like) you to\s+"
    r"|i(?:'d| would) like (?:you )?to\s+|let'?s\s+|go ahead and\s+|help me\s+)+", re.I)
#: (intent, what the request starts with, a keyword anywhere). classify_intent
#: tries every leading form first, then every keyword, then a leading
#: auxiliary: the verb a request opens with says what it asks for ("Delete the
#: old summary emails" is an edit; "summary" there is only its object).
_INTENT_RULES: tuple[tuple[str, re.Pattern[str] | None, re.Pattern[str] | None], ...] = tuple(
    (name, re.compile(lead, re.I) if lead else None, re.compile(anywhere, re.I) if anywhere else None)
    for name, lead, anywhere in (
        ("summarize", r"^(?:summari[sz]e|recap|sum up|tl;?dr|give me (?:a |an )?(?:summary|recap|overview|"
                      r"tl;?dr))\b", r"\bsummar(?:y|ize|ise)\b"),
        ("remember", r"^(?:remember|memori[sz]e|forget|note (?:that|this|down)|save (?:this|that) (?:to|in)|"
                     r"add (?:this|that|it) to (?:your )?(?:memory|notes))\b",
                     r"\b(?:remember that|don'?t forget|add (?:this|that|it) to (?:your )?(?:memory|notes))\b"),
        ("schedule", r"^(?:schedule|remind|set (?:up )?(?:a )?(?:reminder|cron|timer|schedule))\b",
                     r"\b(?:remind me|every (?:day|morning|evening|night|week|hour|weekday|monday|tuesday|"
                     r"wednesday|thursday|friday|saturday|sunday)|on a schedule)\b"),
        ("plan", r"^(?:plan|design|propose|outline|brainstorm|how should (?:we|i)|what(?:'s| is) the best "
                 r"(?:way|approach)|make a plan|come up with a plan)\b",
                 r"\b(?:how should (?:we|i)|what(?:'s| is) the best (?:way|approach)|make a plan|"
                 r"come up with a plan|strategy|roadmap|next steps)\b"),
        ("explain", r"^(?:explain|describe|walk me through|tell me (?:about|how|why|what)|why\b|"
                    r"how (?:does|do|is|are|did|can|could|would)\b|what (?:is|are|does|do|was|were)\b)", None),
        ("research", r"^(?:research|look up|find out|investigate|compare)\b",
                     r"\b(?:search (?:the web|online)|on the web|online|latest news|what'?s new)\b"),
        ("fix", r"^(?:fix|debug|repair|resolve|troubleshoot)\b",
                r"\b(?:is broken|isn'?t working|not working|doesn'?t work|fails?|failing|crash(?:es|ed|ing)?)\b"),
        ("run", r"^(?:run|restart|start|stop|kill|deploy|install|upgrade|rebuild|reload|execute|launch|"
                r"retry|re-?run)\b", None),
        ("create", r"^(?:add|create|write|make|build|implement|generate|draft|set up|setup|scaffold)\b", None),
        ("edit", r"^(?:change|edit|rename|move|update|modify|refactor|replace|clean ?up|remove|delete|"
                 r"rewrite|convert|merge|commit|push)\b", None),
        ("find", r"^(?:find|search|look for|locate|grep|list|show|get|pull|fetch|read|open|look at|"
                 r"go through)\b", None),
        ("check", r"^(?:check|verify|confirm|make sure|ensure|test|monitor|see if|look into|review|audit|"
                  r"inspect)\b", r"\b(?:status|is it (?:running|up|working|done)|healthy)\b"),
        ("question", None, r"\?\s*$"),
    ))
#: a leading auxiliary or wh-word: a question, but weaker than any keyword
#: ("could we set up a reminder every morning" is a schedule request)
_QUESTION_LEAD = re.compile(r"^(?:is|are|can|could|does|do|did|should|would|will|was|were|has|have|which|"
                            r"who|where|when|what|how)\b", re.I)
_NEEDS_STATE = re.compile(
    r"(?i)\b(?:status|running|logs?|right now|currently|uptime|disk space|fail\w*|error\w*|broken|"
    r"crash\w*|this (?:file|page|branch|output)|that (?:file|output)|the (?:diff|output))\b")
_SUBCAT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple((name, re.compile(rx, re.I)) for name, rx in (
    ("scheduling", r"\b(?:schedul\w*|cron\w*|remind\w*|every (?:day|morning|evening|night|week|weekday|"
                   r"hour|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|daily|weekly|"
                   r"recurring|briefing|timer)\b"),
    ("memory", r"\b(?:memor(?:y|ies|ize|ise)|remember|forget|recall|MEMORY\.md|USER\.md|"
               r"note (?:that|this|down))\b"),
    ("wiki", r"\b(?:wiki|vault|knowledge base)\b"),
    ("daemon-ops", r"\b(?:daemon|systemctl|journalctl|services?|restart\w*|reboot|uptime|logs?|gpu|vram|"
                   r"nvidia|llama[- ]?server|ollama|docker|tailscale|sentinel|config(?:uration)?|"
                   r"prometheus\.ya?ml|disk|cpu|health\w*|heartbeat|deploy\w*|server|ssh|ports?|"
                   r"memory (?:usage|use|leak|pressure)|ram|oom)\b"),
    ("research", r"\b(?:research|look up|find out|search (?:the web|online)|web|internet|news|latest|"
                 r"compare|articles?|papers?|sources?|google|recommend\w*|pricing|prices?|reviews?)\b"),
    ("coding", r"\b(?:code|coding|function|method|class|tests?|testing|pytest|bug|repo(?:sitory)?|commit|"
               r"branch|pull request|PR|merge|refactor|lint|ruff|mypy|compile|build|stack ?trace|"
               r"traceback|exception|implement\w*|script|module|import|api|endpoint|python|typescript|"
               r"javascript|rust|swift)\b"),
))
_INDEX_TO_SUBCAT = {"coding": "coding", "memory/wiki": "memory", "sentinel/daemon": "daemon-ops",
                    "other": "general", "chat-mode": "general"}


def classify_intent(text: str) -> str:
    rest = _POLITE.sub("", text.strip()).strip()
    for k in (1, 2):
        for rule in _INTENT_RULES:
            rx = rule[k]
            if rx is not None and rx.search(rest):
                return rule[0]
    return "question" if _QUESTION_LEAD.search(rest) else "request"


def classify_subcategory(text: str, slots: Iterable["Slot"] = (), index_category: str | None = None,
                         intent: str | None = None) -> str:
    # what the request asks for outranks what it mentions ("remember that the
    # server restarts on Friday" is a memory task)
    if intent == "remember":
        return "memory"
    if intent == "schedule":
        return "scheduling"
    scores: collections.Counter = collections.Counter()
    for name, rx in _SUBCAT_RULES:
        hits = {m.group(0).lower() for m in rx.finditer(text)}
        if hits:
            scores[name] += len(hits)
    for sl in slots:
        if sl.cls == "wiki" or sl.type == "wiki_page":
            scores["wiki"] += 1
        elif sl.cls == "code":
            scores["coding"] += 1
        elif sl.type in ("cron", "job"):
            scores["scheduling"] += 1
        elif sl.type == "url":
            scores["research"] += 1
    # Phase 1 saw the teacher do coding work on this prompt: one vote, and the
    # ties ("Fix the bug in the server config loader"), so coding work is
    # not relabelled out from under --coding-cap
    coding_first = index_category == "coding"
    if coding_first:
        scores["coding"] += 1
    if scores:
        best = max(scores.values())
        tie_order = ("scheduling", "memory", "wiki", "daemon-ops", "research", "coding")
        if coding_first:
            tie_order = ("coding",) + tie_order[:-1]
        for name in tie_order:
            if scores[name] == best:
                if name == "memory" and scores["daemon-ops"] and re.search(
                        r"(?i)memory (?:usage|use|leak|pressure)", text):
                    return "daemon-ops"
                return name
    if index_category:
        return _INDEX_TO_SUBCAT.get(index_category, "general")
    return "general"


# ---------------------------------------------------------------------------
# Inventories (real slot values on the host)
# ---------------------------------------------------------------------------

EXT_CLASS = {
    **{e: "doc" for e in (".md", ".rst", ".txt")},
    **{e: "code" for e in (".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs", ".go", ".swift",
                           ".kt", ".java", ".c", ".h", ".cc", ".cpp", ".sh", ".css", ".html", ".jinja",
                           ".sql", ".rb", ".lua")},
    **{e: "config" for e in (".yaml", ".yml", ".toml", ".json", ".jsonl", ".ini", ".cfg", ".conf",
                             ".service", ".timer", ".plist", ".lock", ".env")},
}
_EXT_ALT = "|".join(sorted((e[1:] for e in EXT_CLASS), key=len, reverse=True))
_LABEL = re.compile(r"[A-Za-z][\w ,'&/:()+.-]{1,59}")
_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_DOC_URL = re.compile(r"https://[^\s)<>\"'`\]]+")


def _first_heading(path: Path, max_lines: int = 20) -> str | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                m = re.match(r"^#\s+(.+?)\s*$", line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


def _label_ok(v: str, persons: PersonNames) -> bool:
    v = v.strip()
    if not _LABEL.fullmatch(v) or len(v.split()) > 8 or "http" in v.lower():
        return False
    chars = [c for c in v if not c.isspace()]
    if sum(c.isalpha() for c in chars) < max(3, len(chars) // 2):
        return False
    return not pii_kinds(v) and not persons.mentions(v)


def _value_ok(v: str, persons: PersonNames) -> bool:
    return bool(v) and len(v) <= 160 and "\n" not in v and not pii_kinds(v) and not persons.mentions(v)


class Lexicon:
    """Tells a distinctive name (``symbiote``, ``LoRA``) from an ordinary word
    (``memory``), so an inventory name that is also everyday English is not
    turned into a slot wherever the word occurs. With a word list (one word
    per line, e.g. ``/usr/share/dict/words``) a word is distinctive when the
    list lacks it; without one, only when it carries a digit, ``_``, ``-`` or
    an inner capital."""

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.words = {w.strip().lower() for w in words} if words is not None else None

    @classmethod
    def load(cls, path: Path | None) -> "Lexicon":
        if path is None or not path.exists():
            return cls(None)
        with path.open(encoding="utf-8", errors="replace") as fh:
            return cls(line for line in fh if line.strip())

    def distinctive(self, word: str) -> bool:
        if not re.search(r"[A-Za-z]{2}", word):
            return False
        if re.search(r"\d|_|-", word) or re.search(r"[A-Z]", word[1:]):
            return True
        if self.words is None:
            return False
        return word.lower() not in self.words

    def label_distinctive(self, label: str) -> bool:
        return any(self.distinctive(w) for w in re.findall(r"[\w-]+", label))


@dataclass
class Inventory:
    """Slot value pools: key -> sorted unique (value, from)."""

    pools: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pools = {k: sorted(set((str(v), str(f)) for v, f in vals)) for k, vals in self.pools.items()}
        self._index = {k: {v for v, _ in vals} for k, vals in self.pools.items()}
        self._pairs = {k: set(vals) for k, vals in self.pools.items()}

    def values(self, key: str) -> list[str]:
        return [v for v, _ in self.pools.get(key, [])]

    def contains(self, key: str, value: str) -> bool:
        return value in self._index.get(key, ())

    def has(self, key: str, value: str, frm: str) -> bool:
        """*value* is in pool *key* and came from *frm*."""
        return (value, frm) in self._pairs.get(key, ())

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.pools.items())}

    def pick(self, keys: Iterable[str], rng: random.Random, avoid: str) -> tuple[str, str] | None:
        for key in keys:
            pool = self.pools.get(key) or []
            if not pool or (len(pool) == 1 and pool[0][0] == avoid):
                continue
            for _ in range(8):
                v = pool[rng.randrange(len(pool))]
                if v[0] != avoid:
                    return v
        return None

    @classmethod
    def build(cls, *, repo: Path | None, wiki: Path | None, wiki_sections: Iterable[str],
              tool_names: Iterable[str], cron_names: Iterable[str], persons: PersonNames,
              stats: collections.Counter) -> "Inventory":
        pools: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        if repo is not None:
            _repo_pools(repo, persons, pools, stats)
        if wiki is not None:
            _wiki_pools(wiki, wiki_sections, persons, pools, stats)
        for name in tool_names:
            if re.fullmatch(r"[a-z][a-z0-9_]{1,60}", name):
                pools["tool"].append((name, "registry"))
                if "_" in name:
                    pools["tool_snake"].append((name, "registry"))
        for name in cron_names:
            name = str(name).strip()
            if 4 <= len(name) <= 60 and _value_ok(name, persons) and re.search(r"[A-Za-z]{3}", name):
                pools["job"].append((name, "cron"))
            else:
                stats["inventory_cron_name_dropped"] += 1
        return cls(dict(pools))


def _repo_pools(repo: Path, persons: PersonNames, pools: dict[str, list[tuple[str, str]]],
                stats: collections.Counter) -> None:
    tracked = sorted(BES.git_tracked(repo, (".",)))
    stats["repo_tracked_files"] = len(tracked)
    dirs: set[str] = set()
    for rel in tracked:
        if rel.startswith("tests/fixtures/") or len(rel) > 120 or not _value_ok(rel, persons):
            stats["inventory_repo_path_dropped"] += 1
            continue
        p = Path(rel)
        for parent in list(p.parents)[:-1]:
            dirs.add(parent.as_posix())
        if rel.startswith("src/prometheus/") and rel.count("/") >= 3:
            comp = rel.split("/")[2]
            if re.fullmatch(r"[a-z][a-z0-9_]{1,40}", comp):
                pools["component"].append((comp, "repo"))
        cls_ = EXT_CLASS.get(p.suffix.lower())
        if cls_ is None or p.name == "__init__.py":  # a package marker is no task's target
            continue
        pools[f"path:{cls_}"].append((rel, "repo"))
        pools["path:any"].append((rel, "repo"))
        pools[f"file:{cls_}"].append((p.name, "repo"))
        pools["file:any"].append((p.name, "repo"))
        if rel.startswith("src/prometheus/jobs/") and p.suffix == ".py" and p.stem != "__init__" \
                and rel.count("/") == 3:
            pools["job"].append((p.stem, "jobs"))
        if p.suffix == ".md" and (rel.startswith("docs/") or "/" not in rel) and "CHANGELOG" not in rel:
            _doc_labels(repo / rel, persons, pools, stats)
    for d in sorted(dirs):
        pools["dir"].append((d, "repo"))
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        res = subprocess.run(["git", "-C", str(repo), "log", "--format=%H", "-n", "500"],
                             capture_output=True, env=env, check=False, timeout=60)
        if res.returncode == 0:
            for sha in res.stdout.decode().split():
                if re.fullmatch(r"[0-9a-f]{40}", sha):
                    pools["sha"].append((sha, "git"))
    except (OSError, subprocess.TimeoutExpired):
        stats["git_log_failed"] += 1


def _doc_labels(path: Path, persons: PersonNames, pools: dict[str, list[tuple[str, str]]],
                stats: collections.Counter) -> None:
    try:
        if path.stat().st_size > 1_000_000:
            return
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stats["repo_doc_unreadable"] += 1
        return
    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            label = re.sub(r"[`*_]", "", m.group(2)).strip()
            if _label_ok(label, persons):
                pools["topic"].append((label, "repo-docs"))
    for m in _DOC_URL.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        if len(url) <= 120 and not secret_url(url) and _value_ok(url, persons):
            pools["url"].append((url, "repo-docs"))


def _wiki_pools(wiki: Path, sections: Iterable[str], persons: PersonNames,
                pools: dict[str, list[tuple[str, str]]], stats: collections.Counter) -> None:
    for section in sections:
        if section in PERSON_SECTIONS:
            raise TasksetError(f"wiki section {section!r} holds people; it is a deny list only")
        d = wiki / section
        if not d.is_dir():
            stats["wiki_section_missing"] += 1
            continue
        for p in sorted(d.rglob("*.md")):
            rel = "wiki/" + p.relative_to(wiki).as_posix()
            stem_label = re.sub(r"[-_]+", " ", p.stem).strip()
            if _value_ok(rel, persons) and _label_ok(stem_label, persons):
                pools["wiki_path"].append((rel, "wiki"))
            else:
                stats["inventory_wiki_page_dropped"] += 1
                continue
            title = _first_heading(p) or stem_label
            title = re.sub(r"[`*_]", "", title).strip()
            if _label_ok(title, persons):
                pools["wiki_page"].append((title, "wiki"))
                pools["topic"].append((title, "wiki"))
            try:
                body = p.read_text(encoding="utf-8", errors="replace") if p.stat().st_size <= 1_000_000 else ""
            except OSError:
                body = ""
            for line in body.splitlines():
                m = re.match(r"^#{2,3}\s+(.+?)\s*#*\s*$", line)
                if m:
                    label = re.sub(r"[`*_]", "", m.group(1)).strip()
                    if _label_ok(label, persons):
                        pools["topic"].append((label, "wiki"))


def tool_names_from(pack: Path | None, schemas: Path | None) -> list[str]:
    names: list[str] = []
    if pack is not None:
        data = json.loads(pack.read_text(encoding="utf-8"))
        tools = data.get("tools") or {}
        if isinstance(tools, dict):
            lst = tools.get("full") or tools.get("stubs") or []
        else:
            lst = tools
        names += [str(t.get("name")) for t in lst if isinstance(t, dict) and t.get("name")]
    if schemas is not None and not names:
        data = json.loads(schemas.read_text(encoding="utf-8"))
        for t in data.get("tools") or []:
            fn = t.get("function") if isinstance(t, dict) else None
            name = (fn or t).get("name") if isinstance(fn or t, dict) else None
            if name:
                names.append(str(name))
    return sorted(set(names))


def cron_names_from(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data if isinstance(data, list) else data.get("jobs", []) if isinstance(data, dict) else []
    return sorted({str(j.get("name")) for j in jobs if isinstance(j, dict) and j.get("name")})


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec")
_MONTH_FULL = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

RE_URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")
RE_DQUOTE = re.compile(r'"([^"\n]{2,80})"')
RE_SMARTQUOTE = re.compile("\u201c([^\u201d\n]{2,80})\u201d")
RE_BACKTICK = re.compile(r"`([^`\n]{1,100})`")
RE_PATH = re.compile(r"(?<![\w@:/.~-])(?:~/|\.{1,2}/|/)?(?:[\w.+@-]+/)+[\w.+@-]*")
RE_FILE = re.compile(rf"(?i)(?<![\w/.@~-])[\w-]+(?:\.[\w-]+)*\.(?:{_EXT_ALT})(?![\w-]|\.\w)")
RE_ISO_DATE = re.compile(r"(?<![\w-])\d{4}-\d{2}-\d{2}(?![\w-])")
RE_MONTH_DATE_A = re.compile(rf"(?i)\b(?P<mon>{_MONTHS})(?P<dot>\.?)(?P<sp>\s+)(?P<day>\d{{1,2}})"
                             r"(?P<suf>st|nd|rd|th)?\b")
RE_MONTH_DATE_B = re.compile(rf"(?i)\b(?P<day>\d{{1,2}})(?P<suf>st|nd|rd|th)?(?P<sp>\s+)(?P<of>of\s+)?"
                             rf"(?P<mon>{_MONTHS})\b")
RE_CRON = re.compile(r"(?<!\S)(?:(?:\*|\d{1,2}(?:-\d{1,2})?)(?:/\d{1,2})?(?:,\d{1,2})*\s+){4}"
                     r"(?:\*|\d(?:-\d)?(?:,\d)*)(?:/\d{1,2})?(?!\S)")
RE_TIME_12 = re.compile(r"(?i)(?<![\w:])(?P<h>\d{1,2})(?::(?P<m>\d{2}))?(?P<sp>\s?)"
                        r"(?P<ap>am|pm|a\.m\.|p\.m\.)(?!\w)")
RE_TIME_24 = re.compile(r"(?<![\w:.])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?::(?P<s>[0-5]\d))?(?![\w:]|\.\d)")
RE_VERSION = re.compile(r"(?<![\w.])[vV]?\d+\.\d+\.\d+(?:\.\d+)?(?![\w]|\.\d)")
RE_SHA = re.compile(r"(?<![\w-])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}(?![\w-])")
RE_ISSUE = re.compile(r"(?<![\w&#])#\d{1,6}\b")
RE_WEEKDAY = re.compile(r"(?i)\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b")
RE_NUMBER = re.compile(r"(?<![\w.#-])\d+(?:\.\d+)?(?!\w|\.\d)")
_WIKI_CONTEXT = re.compile(r"(?i)\b(?:wiki|vault|notes?|page)\b")
_HOME_PATH = re.compile(r"^(?:/home|/Users|/root|/var/home)/")
_CODE_LIKE = re.compile(r"[()=$;{}<>|\\\[\]/]|^[-.]|\w\.\w|  ")
#: product names shaped like file names: literal text, never a file slot
_TECH_NAMES = frozenset({"node.js", "next.js", "vue.js", "react.js", "three.js", "d3.js", "express.js",
                         "nuxt.js", "chart.js", "socket.io"})
#: package names that are common tech words (often missing from word lists):
#: a component slot only when written as a code unit, never as a plain word
COMMON_TECH_WORDS = frozenset("""
api apis cli config configs configuration util utils mcp infra eval evals lsp kanban web docs doc tests
test lib libs core common misc tools jobs tasks hooks skills packs plugins server client sdk ui app apps
auth db cache logs log
""".split())
_CODE_UNIT_BEFORE = re.compile(r"(?i)(?:\bprometheus[./]|\b(?:module|package|subpackage|pkg)\s+)$")
_CODE_UNIT_AFTER = re.compile(r"(?i)^(?:\.py\b|/|\s+(?:module|package|subpackage|pkg)\b)")

#: types whose new value changes the normalized text (a real variant)
VARYING = frozenset({"path", "file", "url", "tool", "job", "wiki_page", "topic", "component", "weekday",
                     "month_date"})
GENERATED = frozenset({"number", "version", "iso_date", "month_date", "weekday", "time", "cron",
                       "issue"})


@dataclass(frozen=True)
class Slot:
    start: int
    end: int
    type: str
    cls: str = ""
    must_fill: bool = False


def _strip_end(text: str, s: int, e: int, chars: str) -> int:
    while e > s and text[e - 1] in chars:
        e -= 1
    return e


class SlotExtractor:
    """Typed slot spans in a prompt, by priority, never overlapping."""

    def __init__(self, inv: Inventory, lexicon: Lexicon | None = None) -> None:
        self.inv = inv
        lex = lexicon or Lexicon(None)
        self.known = {v.rstrip("/") for k in ("dir", "path:any", "wiki_path") for v in inv.values(k)}
        self.dirs = set(inv.values("dir"))
        self.tools = set(inv.values("tool"))
        self.jobs = {j.lower() for j in inv.values("job")}
        self.tool_rx = _alternation([t for t in self.tools if "_" in t], r"[\w-]", flags=0)
        self.job_rx = _alternation([j for j in inv.values("job") if len(j) >= 5], r"[\w-]", flags=re.I)
        titles = inv.values("wiki_page")
        # anywhere: a multi-word title with a distinctive word; with the wiki
        # mentioned: any title of 8+ characters or more than one word
        self.wiki_any_rx = _alternation([t for t in titles if " " in t and lex.label_distinctive(t)],
                                        r"\w", flags=re.I)
        self.wiki_rx = _alternation([t for t in titles if len(t) >= 8 or " " in t], r"\w", flags=re.I)
        self.topic_rx = _alternation([t for t in inv.values("topic") if " " in t and len(t) >= 10
                                      and lex.label_distinctive(t)], r"\w", flags=re.I)
        self.component_rx = _alternation([c for c in inv.values("component") if len(c) >= 3
                                          and lex.distinctive(c)], r"[\w/-]", flags=re.I)

    # -- typing helpers --------------------------------------------------------

    def path_ok(self, p: str) -> bool:
        core = p.rstrip("/")
        if not core or not re.search(r"[A-Za-z]", p):
            return False
        if p.startswith(("~/", "./", "../", "/")):
            return True
        if Path(core).suffix.lower() in EXT_CLASS:
            return True
        if core.count("/") >= 2:
            return True
        return core in self.known

    def path_cls(self, p: str) -> str:
        if re.search(r"(?:^|/)wiki/", p):
            return "wiki"
        if p.endswith("/"):
            return "dir"
        cls_ = EXT_CLASS.get(Path(p).suffix.lower())
        if cls_:
            return cls_
        return "dir" if p in self.dirs or "." not in Path(p).name else "other"

    def type_inner(self, inner: str) -> tuple[str, str] | None:
        if inner != inner.strip() or not inner:
            return None
        if RE_URL.fullmatch(inner):
            return "url", ""
        if RE_PATH.fullmatch(inner) and self.path_ok(inner):
            return "path", self.path_cls(inner)
        if RE_FILE.fullmatch(inner):
            return "file", EXT_CLASS.get(Path(inner).suffix.lower(), "other")
        if inner in self.tools:
            return "tool", ""
        if inner.lower() in self.jobs:
            return "job", ""
        if _CODE_LIKE.search(inner) or "_" in inner:
            return None
        words = inner.split()
        if 1 <= len(words) <= 8 and re.search(r"[A-Za-z]{2}", inner) and all(
                re.fullmatch(r"[A-Za-z][\w'&+.-]*|\d+", w) for w in words):
            return "topic", ""
        return None

    # -- extraction --------------------------------------------------------------

    def extract(self, text: str) -> tuple[list[Slot], list[tuple[int, int]]]:
        """(slots in text order, protected spans: quoted code kept literal)."""
        taken: list[tuple[int, int]] = []
        slots: list[Slot] = []
        protected: list[tuple[int, int]] = []

        def free(s: int, e: int) -> bool:
            return e > s and all(e <= ts or s >= te for ts, te in taken)

        def add(s: int, e: int, typ: str, cls_: str = "", must: bool = False) -> None:
            if free(s, e):
                taken.append((s, e))
                slots.append(Slot(s, e, typ, cls_, must))

        for m in RE_URL.finditer(text):
            add(m.start(), _strip_end(text, m.start(), m.end(), ".,;:!?'\""), "url")
        for rx in (RE_DQUOTE, RE_SMARTQUOTE, RE_BACKTICK):
            for m in rx.finditer(text):
                if not free(m.start(), m.end()):
                    continue
                typed = self.type_inner(m.group(1))
                if typed is None:
                    taken.append((m.start(), m.end()))
                    protected.append((m.start(), m.end()))
                else:
                    s, e = m.span(1)
                    add(s, e, typed[0], typed[1], typed[0] == "path" and bool(_HOME_PATH.match(m.group(1))))
        for m in RE_PATH.finditer(text):
            e = _strip_end(text, m.start(), m.end(), ".,;:!?)'\"")
            p = text[m.start():e]
            if self.path_ok(p):
                add(m.start(), e, "path", self.path_cls(p), bool(_HOME_PATH.match(p)))
        for m in RE_FILE.finditer(text):
            if m.group(0).lower() in _TECH_NAMES:
                continue
            add(m.start(), m.end(), "file", EXT_CLASS.get(Path(m.group(0)).suffix.lower(), "other"))
        for m in RE_ISO_DATE.finditer(text):
            try:
                date.fromisoformat(m.group(0))
            except ValueError:
                continue
            add(m.start(), m.end(), "iso_date")
        for rx in (RE_MONTH_DATE_A, RE_MONTH_DATE_B):
            for m in rx.finditer(text):
                # a lowercase "may" is the modal verb ("these 2 may conflict")
                # unless an ordinal or "of" makes it a date ("3rd of may")
                if m.group("mon") == "may" and not (m.group("suf") or m.groupdict().get("of")):
                    continue
                if 1 <= int(m.group("day")) <= 31:
                    add(m.start(), m.end(), "month_date")
        for m in RE_CRON.finditer(text):
            if "*" in m.group(0):
                add(m.start(), m.end(), "cron")
        for m in RE_TIME_12.finditer(text):
            if 1 <= int(m.group("h")) <= 12:
                add(m.start(), m.end(), "time")
        for m in RE_TIME_24.finditer(text):
            add(m.start(), m.end(), "time")
        for m in RE_VERSION.finditer(text):
            add(m.start(), m.end(), "version")
        for m in RE_SHA.finditer(text):
            add(m.start(), m.end(), "sha")
        for m in RE_ISSUE.finditer(text):
            add(m.start(), m.end(), "issue")
        for rx, typ in ((self.tool_rx, "tool"), (self.job_rx, "job")):
            if rx is not None:
                for m in rx.finditer(text):
                    add(m.start(), m.end(), typ)
        wiki_rx = self.wiki_rx if _WIKI_CONTEXT.search(text) else self.wiki_any_rx
        if wiki_rx is not None:
            for m in wiki_rx.finditer(text):
                add(m.start(), m.end(), "wiki_page")
        if self.topic_rx is not None:
            for m in self.topic_rx.finditer(text):
                add(m.start(), m.end(), "topic")
        if self.component_rx is not None:
            for m in self.component_rx.finditer(text):
                if m.group(0).lower() in COMMON_TECH_WORDS and not (
                        _CODE_UNIT_BEFORE.search(text[max(0, m.start() - 24):m.start()])
                        or _CODE_UNIT_AFTER.search(text[m.end():m.end() + 24])):
                    continue  # "check the config", "the MCP server": a word, not a package
                add(m.start(), m.end(), "component")
        for m in RE_WEEKDAY.finditer(text):
            add(m.start(), m.end(), "weekday")
        for m in RE_NUMBER.finditer(text):
            add(m.start(), m.end(), "number")
        slots.sort(key=lambda sl: sl.start)
        return slots, protected


def home_path_literal(text: str, slots: Iterable[Slot] = ()) -> bool:
    """A home directory's user name in *text* outside every must-fill slot
    (quoted code, ``host:/home/x``, ``file:///Users/x``: places no path slot
    covers, so the name would be copied into variants verbatim)."""
    must = [(sl.start, sl.end) for sl in slots if sl.must_fill]
    for m in HOME_ANY.finditer(text):
        s, e = m.span("user")
        if not any(a <= s and e <= b for a, b in must):
            return True
    return False


#: the placeholder token a slot span becomes when text is masked
#: (``tokens`` keeps letters and digits only, so it stays one token)
_MASK_KIND = {"path": "path", "file": "path", "url": "url", "tool": "name", "job": "name",
              "wiki_page": "name", "topic": "name", "component": "name"}


def mask_token(typ: str) -> str:
    return f"qqslot{_MASK_KIND.get(typ, 'value')}qq"


def _alternation(words: Iterable[str], boundary: str, flags: int) -> re.Pattern[str] | None:
    ws = sorted({w for w in words if w}, key=lambda w: (-len(w), w))
    if not ws:
        return None
    body = "|".join(r"\s+".join(map(re.escape, w.split())) for w in ws)
    return re.compile(rf"(?<!{boundary})(?:{body})(?!{boundary})", flags)


_SENT_BOUND = re.compile(r"[.?!][\"')\]]*(\s+)(?=\S)|\n\s*(?=\S)")


def sentences(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Contiguous (start, end) sentence spans covering *text*; an end includes
    the whitespace after it. Never splits inside a *blocked* span."""
    cuts: list[int] = []
    for m in _SENT_BOUND.finditer(text):
        if any(s <= m.start() < e for s, e in blocked):
            continue
        cuts.append(m.end())
    out: list[tuple[int, int]] = []
    start = 0
    for c in cuts:
        if c > start:
            out.append((start, c))
            start = c
    out.append((start, len(text)))
    return out


# ---------------------------------------------------------------------------
# Generated slot values
# ---------------------------------------------------------------------------


def _case_like(word: str, model: str) -> str:
    if model.isupper() and len(model) > 1:
        return word.upper()
    if model[:1].isupper():
        return word[:1].upper() + word[1:].lower()
    return word.lower()


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def gen_value(typ: str, orig: str, rng: random.Random, base: date) -> str:
    """A generated value of *typ* shaped like *orig*."""
    if typ == "number":
        if "." in orig:
            whole, frac = orig.split(".", 1)
            return f"{rng.randint(0, max(1, 10 ** len(whole) - 1))}.{rng.randint(0, 10 ** len(frac) - 1):0{len(frac)}d}"
        d = len(orig)
        if d == 1:
            return str(rng.randint(1, 9))
        if orig.startswith("0"):
            return str(rng.randint(0, 10 ** d - 1)).zfill(d)
        return str(rng.randint(10 ** (d - 1), 10 ** d - 1))
    if typ == "version":
        prefix = orig[0] if orig[:1] in ("v", "V") else ""
        parts = orig.lstrip("vV").split(".")
        return prefix + ".".join([str(rng.randint(0, 3))] + [str(rng.randint(0, 20)) for _ in parts[1:]])
    if typ == "iso_date":
        return (base + timedelta(days=rng.randint(-120, 120))).isoformat()
    if typ == "month_date":
        month = rng.randrange(12)
        day = rng.randint(1, 28)
        for rx in (RE_MONTH_DATE_A, RE_MONTH_DATE_B):
            m = rx.fullmatch(orig)
            if not m:
                continue
            mon = m.group("mon")
            name = _MONTH_FULL[month] if len(mon) > 4 or mon.lower() == "may" else _MONTH_FULL[month][:3]
            name = _case_like(name, mon)
            suf = _ordinal(day) if m.group("suf") else ""
            if suf and m.group("suf").isupper():
                suf = suf.upper()
            if rx is RE_MONTH_DATE_A:
                return f"{name}{m.group('dot')}{m.group('sp')}{day}{suf}"
            return f"{day}{suf}{m.group('sp')}{m.group('of') or ''}{name}"
        return f"{_MONTH_FULL[month]} {day}"
    if typ == "weekday":
        return _case_like(_WEEKDAYS[rng.randrange(7)], orig)
    if typ == "time":
        m = RE_TIME_12.fullmatch(orig)
        if m:
            hh = rng.randint(1, 12)
            mm = f":{rng.choice(('00', '15', '30', '45'))}" if m.group("m") else ""
            ap = m.group("ap")
            new = rng.choice(("am", "pm"))
            if "." in ap:
                new = new[0] + "." + new[1] + "."
            return f"{hh}{mm}{m.group('sp')}{new.upper() if ap.isupper() else new}"
        m = RE_TIME_24.fullmatch(orig)
        hh = rng.randint(0, 23)
        width = 2 if m and len(m.group("h")) == 2 else 1
        out = f"{hh:0{width}d}:{rng.randrange(0, 60, 5):02d}"
        if m and m.group("s"):
            out += ":00"
        return out
    if typ == "cron":
        minute = rng.choice((0, 5, 10, 15, 30, 45))
        hour = rng.choice((str(rng.randint(0, 23)), f"*/{rng.choice((2, 3, 4, 6))}"))
        dow = rng.choice(("*", "1-5", "0", "6", "1"))
        return f"{minute} {hour} * * {dow}"
    if typ == "issue":
        d = max(1, len(orig) - 1)
        return "#" + str(rng.randint(10 ** (d - 1) if d > 1 else 1, 10 ** d - 1))
    if typ == "sha":
        return "".join(rng.choice("0123456789abcdef") for _ in orig)
    raise TasksetError(f"no generator for slot type {typ!r}")


_GEN_SHAPES = {
    "number": re.compile(r"\d+(?:\.\d+)?"),
    "version": re.compile(r"[vV]?\d+(?:\.\d+)+"),
    "iso_date": RE_ISO_DATE,
    "weekday": RE_WEEKDAY,
    "cron": RE_CRON,
    "issue": re.compile(r"#\d+"),
    "sha": re.compile(r"[0-9a-f]{7,40}"),
}


def generated_shape_ok(typ: str, value: str) -> bool:
    if typ == "month_date":
        return bool(RE_MONTH_DATE_A.fullmatch(value) or RE_MONTH_DATE_B.fullmatch(value))
    if typ == "time":
        return bool(RE_TIME_12.fullmatch(value) or RE_TIME_24.fullmatch(value))
    rx = _GEN_SHAPES.get(typ)
    if rx is None or not rx.fullmatch(value):
        return False
    if typ == "iso_date":
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
    return True


def pool_keys(sl: Slot, orig: str) -> tuple[str, ...]:
    if sl.type == "path":
        if sl.cls == "wiki":
            return ("wiki_path",)
        if sl.cls == "dir":
            return ("dir",)
        return (f"path:{sl.cls}", "path:any")
    if sl.type == "file":
        return (f"file:{sl.cls}", "file:any")
    if sl.type == "tool":
        return ("tool_snake",) if "_" in orig else ("tool",)
    if sl.type in ("url", "job", "wiki_page", "topic", "component", "sha"):
        return (sl.type,)
    return ()


# ---------------------------------------------------------------------------
# Real prompts and templates
# ---------------------------------------------------------------------------


@dataclass
class RealPrompt:
    text: str
    pattern_hash: str
    origin: str  # golden | lcm
    n_traces: int = 0
    categories: collections.Counter = field(default_factory=collections.Counter)
    in_lcm: bool = False

    @property
    def index_category(self) -> str | None:
        if not self.categories:
            return None
        order = {c: i for i, c in enumerate(("coding", "sentinel/daemon", "memory/wiki", "other", "chat-mode"))}
        return sorted(self.categories.items(), key=lambda kv: (-kv[1], order.get(kv[0], 9), kv[0]))[0][0]


@dataclass
class Template:
    id: str
    pattern_hash: str
    origin: str
    text: str
    slots: tuple[Slot, ...]
    sentences: tuple[tuple[int, int], ...]
    intent: str
    subcategory: str
    chat_ok: bool
    n_traces: int
    token_set: frozenset[str]
    members: int = 1

    @property
    def varying(self) -> list[int]:
        return [i for i, sl in enumerate(self.slots) if sl.type in VARYING]


def load_index(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Phase 1 index rows keyed by (file, ordinal within the file, 1-based)."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if path is None:
        return out
    seen: collections.Counter = collections.Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                seen[row["file"]] += 1
                out[(row["file"], seen[row["file"]])] = row
    return out


def load_golden(paths: list[Path], index: dict[tuple[str, int], dict[str, Any]], surfaces: tuple[str, ...],
                stats: collections.Counter) -> dict[str, RealPrompt]:
    out: dict[str, RealPrompt] = {}
    for p in paths:
        ordinal = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ordinal += 1
                stats["golden_traces_read"] += 1
                try:
                    trace = json.loads(line)
                    msgs = trace["messages"]
                except (ValueError, KeyError, TypeError):
                    stats["golden_unparseable"] += 1
                    continue
                meta = trace.get("_meta") or {}
                sid = str(meta.get("session_id") or "")
                if not sid.startswith(surfaces):
                    stats["golden_not_human_surface"] += 1
                    continue
                users = [content_text(m.get("content")) for m in msgs[:-1] if m.get("role") == "user"]
                prompt = users[-1] if users else ""
                if not prompt.strip():
                    stats["golden_no_prompt"] += 1
                    continue
                ph = h(norm(prompt))
                idx = index.get((p.name, ordinal))
                if idx is not None and idx.get("h_prompt") and idx["h_prompt"] != ph:
                    stats["index_h_prompt_disagreements"] += 1
                rp = out.get(ph)
                if rp is None:
                    rp = out[ph] = RealPrompt(prompt.strip(), ph, "golden")
                rp.n_traces += 1
                cat = (idx or {}).get("category")
                if cat:
                    rp.categories[str(cat)] += 1
    return out


def load_lcm(path: Path, surfaces: tuple[str, ...], stats: collections.Counter,
             injected: set[str] | None = None) -> dict[str, RealPrompt]:
    """User-typed prompts in a snapshot of the conversation store, by task
    proxy. *injected* collects the task proxy of every role-``user`` row the
    daemon wrote (provenance other than ``user``, or an untrusted text turn),
    on any surface: the golden exporter drops provenance, so a golden prompt
    with one of these hashes is a machine-written turn."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(lcm_messages)")}
        prov = "provenance" if "provenance" in cols else "'user'"
        trusted = "is_trusted" if "is_trusted" in cols else "1"
        rows = con.execute(
            f"SELECT session_id, content, content_json, {prov}, {trusted} FROM lcm_messages"
            " WHERE role = 'user' ORDER BY session_id, turn_index, rowid")
        out: dict[str, RealPrompt] = {}

        def collect(content: str | None, blocks: Any) -> None:
            if injected is not None:
                texts = [content or ""]
                if isinstance(blocks, list):  # the exporter may join the text blocks instead
                    texts.append(content_text(blocks))
                injected.update(h(norm(t)) for t in texts if t.strip())

        for sid, content, cj, provenance, is_trusted in rows:
            stats["lcm_user_rows"] += 1
            try:
                blocks = json.loads(cj) if cj else None
            except ValueError:
                blocks = None
            if provenance != "user":
                stats["lcm_not_user_provenance"] += 1
                collect(content, blocks)
                continue
            if cj:
                types = {b.get("type") for b in blocks if isinstance(b, dict)} if isinstance(blocks, list) else set()
                if types - {"text"}:  # a tool result or an attachment: never a typed prompt
                    stats["lcm_non_text_rows"] += 1
                    continue
            if not is_trusted:  # a text turn marked untrusted: not the owner typing
                stats["lcm_untrusted_user_rows"] += 1
                collect(content, blocks)
                continue
            if not str(sid or "").startswith(surfaces):
                stats["lcm_not_human_surface"] += 1
                continue
            text = (content or "").strip()
            if not text:
                continue
            ph = h(norm(text))
            if ph not in out:
                out[ph] = RealPrompt(text, ph, "lcm", in_lcm=True)
        return out
    finally:
        con.close()


class Holdout:
    """Held-out tasks: task ids and every user message of the held-out rows.

    With a masker (``use_masker``: the generator's slot extractor), each
    held-out message that holds a slot is indexed a second time with every
    slot span replaced by its kind, and ``check`` masks the text the same
    way: one changed path, name or number leaves the raw 5-gram containment
    of a short prompt under the threshold, the masked one does not."""

    MIN_LITERAL = 3  # a masked message needs this many words that are not placeholders

    def __init__(self, k: int = 5, containment: float = 0.6) -> None:
        self.k = k
        self.containment = containment
        self.task_ids: set[str] = set()
        self.norms: set[str] = set()
        self.short: set[str] = set()
        self.index: dict[int, list[int]] = collections.defaultdict(list)
        self.sizes: list[int] = []
        self.rows = 0
        self.texts: list[str] = []
        self._masker: Any = None
        self.masked_norms: set[str] = set()
        self.masked_index: dict[int, list[int]] = collections.defaultdict(list)
        self.masked_sizes: list[int] = []

    def use_masker(self, masker: Any) -> None:
        """Index every held-out message's slot-masked form (``masker(text)``
        returns the text with each slot span replaced by ``mask_token``)."""
        self._masker = masker
        self.masked_norms.clear()
        self.masked_index.clear()
        self.masked_sizes.clear()
        for text in self.texts:
            self._add_masked(text)

    def _masked_tokens(self, text: str) -> tuple[str, list[str]] | None:
        masked = self._masker(text)
        if masked == text:
            return None  # no slot: the raw check is the whole test
        toks = tokens(masked)
        if sum(1 for t in toks if not (t.startswith("qqslot") and t.endswith("qq"))) < self.MIN_LITERAL:
            return None  # mostly placeholders: would match every short command
        return masked, toks

    def _add_masked(self, text: str) -> None:
        got = self._masked_tokens(text)
        if got is None:
            return
        masked, toks = got
        self.masked_norms.add(norm(masked))
        if len(toks) < self.k:
            return
        doc = len(self.masked_sizes)
        grams = {_ihash(g) for g in ngrams(toks, self.k)}
        self.masked_sizes.append(len(grams))
        for g in grams:
            self.masked_index[g].append(doc)

    def load(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows += 1
                meta = row.get("meta") or {}
                for key in ("task_id", "group_id"):
                    if meta.get(key):
                        self.task_ids.add(str(meta[key]))
                for m in row.get("messages") or []:
                    if m.get("role") == "user":
                        self.add_text(content_text(m.get("content")))

    def add_text(self, text: str) -> None:
        n = norm(text)
        if not n or n in self.norms:
            return
        self.norms.add(n)
        self.texts.append(text)
        self.task_ids.add(h(n))
        if self._masker is not None:
            self._add_masked(text)
        toks = tokens(text)
        if len(toks) < self.k:
            self.short.add(" ".join(toks))
            return
        doc = len(self.sizes)
        grams = {_ihash(g) for g in ngrams(toks, self.k)}
        self.sizes.append(len(grams))
        for g in grams:
            self.index[g].append(doc)

    def _contained(self, toks: list[str], index: dict[int, list[int]], sizes: list[int]) -> bool:
        grams = {_ihash(g) for g in ngrams(toks, self.k)}
        hits: collections.Counter = collections.Counter()
        for g in grams:
            for doc in index.get(g, ()):
                hits[doc] += 1
        # both ways: the text mostly inside one held-out message, or one held-out
        # message mostly inside the text (a real prompt that quotes a held-out
        # request and adds a sentence would otherwise carry it into variants)
        return any(n / len(grams) >= self.containment or n / sizes[doc] >= self.containment
                   for doc, n in hits.items())

    def check(self, text: str) -> str | None:
        n = norm(text)
        if h(n) in self.task_ids or n in self.norms:
            return "holdout_task"
        toks = tokens(text)
        if len(toks) < self.k:
            if " ".join(toks) in self.short:
                return "holdout_task"
        elif self._contained(toks, self.index, self.sizes):
            return "holdout_near_dup"
        else:
            # a short held-out message (3+ words, under k) only rules out a text that
            # is mostly that message: "continue" or "yes do it" must not knock out
            # every prompt that happens to contain the words
            joined = " ".join(toks)
            for short in self.short:
                n_short = short.count(" ") + 1
                if n_short >= 3 and n_short / len(toks) >= self.containment and re.search(
                        r"(?<!\S)" + re.escape(short) + r"(?!\S)", joined):
                    return "holdout_near_dup"
        if self._masker is not None and self.masked_norms:
            got = self._masked_tokens(text)
            if got is not None:
                masked, mtoks = got
                if norm(masked) in self.masked_norms or (
                        len(mtoks) >= self.k and self._contained(mtoks, self.masked_index, self.masked_sizes)):
                    return "holdout_slot_variant"
        return None


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def distinct_n(docs: list[list[str]], n: int) -> float | None:
    total = 0
    uniq: set[tuple[str, ...]] = set()
    for d in docs:
        if len(d) < n:
            continue
        grams = [tuple(d[i:i + n]) for i in range(len(d) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return round(len(uniq) / total, 4) if total else None


def self_bleu(docs: list[list[str]], *, sample: int, seed: int, max_n: int = 4, eps: float = 0.1) -> float | None:
    """Mean BLEU-4 of up to *sample* docs, each against all the others
    (clipped by the max count in any other doc; brevity penalty from the
    closest other length; epsilon smoothing for empty orders). Lower = more
    diverse."""
    docs = [d for d in docs if d]
    if len(docs) < 2:
        return None
    tops: list[dict[tuple[str, ...], list[int]]] = []
    for n in range(1, max_n + 1):
        best: dict[tuple[str, ...], list[int]] = {}
        for i, d in enumerate(docs):
            for g, c in collections.Counter(tuple(d[j:j + n]) for j in range(len(d) - n + 1)).items():
                b = best.get(g)
                if b is None:
                    best[g] = [c, i, 0]
                elif c > b[0]:
                    best[g] = [c, i, b[0]]
                elif c > b[2]:
                    b[2] = c
        tops.append(best)
    lengths = sorted(len(d) for d in docs)
    idxs = sorted(random.Random(f"{seed}|self-bleu").sample(range(len(docs)), min(sample, len(docs))))
    scores = []
    for i in idxs:
        d = docs[i]
        orders = min(max_n, len(d))
        logp = 0.0
        for n in range(1, orders + 1):
            grams = collections.Counter(tuple(d[j:j + n]) for j in range(len(d) - n + 1))
            total = sum(grams.values())
            best = tops[n - 1]
            clipped = 0
            for g, c in grams.items():
                b = best[g]
                clipped += min(c, b[2] if b[1] == i else b[0])
            logp += math.log((clipped if clipped else eps) / total)
        c = len(d)
        k = bisect.bisect_left(lengths, c)
        cands = []
        for j in (k - 1, k, k + 1):
            if 0 <= j < len(lengths):
                cands.append(lengths[j])
        if lengths.count(c) < 2 and c in cands:
            cands.remove(c)
        r = min(cands, key=lambda x: (abs(x - c), x)) if cands else c
        bp = 1.0 if c > r else math.exp(1 - r / c)
        scores.append(bp * math.exp(logp / orders))
    return round(sum(scores) / len(scores), 4)


def diversity(texts: list[str], seed: int, sample: int = 300) -> dict[str, Any]:
    docs = [tokens(t) for t in texts]
    return {"docs": len(docs), "distinct_1": distinct_n(docs, 1), "distinct_2": distinct_n(docs, 2),
            "distinct_3": distinct_n(docs, 3), "self_bleu4": self_bleu(docs, sample=sample, seed=seed),
            "self_bleu_sample": min(sample, len(docs))}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    seed: int = DEFAULT_SEED
    per_pattern: int = 8
    max_uses: int = 8  # appearances of one real template across all variants (primary or partner)
    coding_cap: float = 0.15
    chat_share: float = 0.85
    error_share: float = 0.2
    compaction_share: float = 0.2
    max_total: int | None = None
    min_chars: int = 12
    min_words: int = 3
    max_prompt_chars: int = 800
    max_lines: int = 12
    max_variant_chars: int = 1200
    max_variant_sentences: int = 6
    source_merge: float = 0.85
    variant_dup: float = 0.8
    dup_ngram: int = 4
    holdout_k: int = 5
    holdout_containment: float = 0.6
    attempts_per_variant: int = 6
    surfaces: tuple[str, ...] = HUMAN_SURFACES
    base_date: str = "2026-11-01"

    def check(self) -> None:
        for name in ("coding_cap", "chat_share", "error_share", "compaction_share", "source_merge",
                     "variant_dup", "holdout_containment"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise TasksetError(f"{name} must be in [0, 1] (got {v})")
        if self.per_pattern < 1 or self.max_uses < 1:
            raise TasksetError("per_pattern and max_uses must be >= 1")
        if self.max_total is not None and self.max_total < 1:
            raise TasksetError("max_total must be >= 1")
        date.fromisoformat(self.base_date)


@dataclass
class Candidate:
    prompt: str
    parts: list[tuple[Template, int, int]]
    values: dict[tuple[int, int], tuple[str, str]]  # (part, slot index) -> (value, from)
    transform: str
    pattern: str


def render(parts: list[tuple[Template, int, int]], values: dict[tuple[int, int], tuple[str, str]]) -> str:
    out: list[str] = []
    for k, (tpl, s, e) in enumerate(parts):
        piece: list[str] = []
        pos = s
        for i, sl in enumerate(tpl.slots):
            if sl.start < s or sl.end > e:
                continue
            piece.append(tpl.text[pos:sl.start])
            got = values.get((k, i))
            piece.append(got[0] if got else tpl.text[sl.start:sl.end])
            pos = sl.end
        piece.append(tpl.text[pos:e])
        text = "".join(piece)
        if out and not out[-1][-1:].isspace():
            out.append("\n")
        out.append(text)
    return "".join(out).strip()


class Generator:
    def __init__(self, settings: Settings, inventory: Inventory, *, holdout: Holdout | None = None,
                 manifest: Any = None, persons: PersonNames | None = None,
                 lexicon: Lexicon | None = None, injected: Iterable[str] = (),
                 require_user_row: bool = False) -> None:
        """*injected*: task proxies of daemon-written user turns (``load_lcm``);
        *require_user_row*: keep a golden prompt only when the snapshot holds
        it as a user-typed row (``RealPrompt.in_lcm``)."""
        settings.check()
        self.s = settings
        self.inv = inventory
        self.holdout = holdout
        self.manifest = manifest
        self.persons = persons or PersonNames()
        self.extractor = SlotExtractor(inventory, lexicon)
        self.injected = frozenset(injected)
        self.require_user_row = require_user_row
        if holdout is not None:
            holdout.use_masker(self.mask)
        self.base = date.fromisoformat(settings.base_date)
        self.stats: collections.Counter = collections.Counter()
        self.excluded: collections.Counter = collections.Counter()
        self.rejected: collections.Counter = collections.Counter()
        self.source_norms: set[str] = set()
        self.templates: list[Template] = []
        self.by_id: dict[str, Template] = {}
        self.origins: collections.Counter = collections.Counter()

    # -- templates -------------------------------------------------------------

    def add_prompts(self, prompts: Iterable[RealPrompt]) -> None:
        kept: list[Template] = []
        for rp in sorted(prompts, key=lambda r: r.pattern_hash):
            self.stats["real_prompts"] += 1
            self.origins[rp.origin] += 1
            self.source_norms.add(norm(rp.text))
            reason = self.template_problem(rp)
            if reason:
                self.excluded[reason] += 1
                continue
            kept.append(self.make_template(rp))
        self.templates = self.merge_near_duplicates(kept)
        self.by_id = {t.id: t for t in self.templates}

    def mask(self, text: str) -> str:
        """*text* with every slot span replaced by its kind's placeholder."""
        slots, _ = self.extractor.extract(text)
        out: list[str] = []
        pos = 0
        for sl in slots:
            out += [text[pos:sl.start], " " + mask_token(sl.type) + " "]
            pos = sl.end
        out.append(text[pos:])
        return "".join(out) if slots else text

    def template_problem(self, rp: RealPrompt) -> str | None:
        text = rp.text
        if rp.pattern_hash in self.injected:
            return "injected_turn"
        if self.require_user_row and rp.origin == "golden" and not rp.in_lcm:
            return "golden_not_user_typed"
        if self.holdout is not None:
            r = self.holdout.check(text)
            if r:
                return r
        if self.manifest is not None and self.manifest.overlaps(text):
            return "injection_overlap"
        kinds = pii_kinds(text)
        if kinds:
            return "pii:" + kinds[0]
        if self.persons.mentions(text):
            return "person_name"
        if HOME_ANY.search(text) and home_path_literal(text, self.extractor.extract(text)[0]):
            return "home_path_literal"
        return prompt_problem(text, self.s)

    def make_template(self, rp: RealPrompt) -> Template:
        slots, protected = self.extractor.extract(rp.text)
        sents = sentences(rp.text, [(sl.start, sl.end) for sl in slots] + protected)
        intent = classify_intent(rp.text)
        sub = classify_subcategory(rp.text, slots, rp.index_category, intent)
        chat_ok = (intent in CHAT_INTENTS and not _NEEDS_STATE.search(rp.text)
                   and not any(sl.type in ("path", "file", "url") for sl in slots))
        return Template(id="tpl-" + h(rp.text), pattern_hash=rp.pattern_hash, origin=rp.origin,
                        text=rp.text, slots=tuple(slots), sentences=tuple(sents), intent=intent,
                        subcategory=sub, chat_ok=chat_ok, n_traces=rp.n_traces,
                        token_set=frozenset(tokens(rp.text)))

    def merge_near_duplicates(self, tpls: list[Template]) -> list[Template]:
        """One template per group of near-identical real prompts (word-set
        Jaccard >= source_merge): the most-used member, then golden first,
        then the shortest."""
        order = sorted(tpls, key=lambda t: (-t.n_traces, t.origin != "golden", len(t.text), t.id))
        reps: list[Template] = []
        by_token: dict[str, list[int]] = collections.defaultdict(list)
        for t in order:
            if not t.token_set:
                continue
            shared: collections.Counter = collections.Counter()
            for tok in t.token_set:
                for r in by_token.get(tok, ()):
                    shared[r] += 1
            dup = None
            for r, n in shared.items():
                union = len(t.token_set) + len(reps[r].token_set) - n
                if union and n / union >= self.s.source_merge:
                    dup = r
                    break
            if dup is not None:
                reps[dup].members += 1
                self.stats["near_duplicate_prompts_merged"] += 1
                continue
            for tok in t.token_set:
                by_token[tok].append(len(reps))
            reps.append(t)
        return sorted(reps, key=lambda t: t.id)

    # -- values ----------------------------------------------------------------

    def draw(self, tpl: Template, i: int, rng: random.Random) -> tuple[str, str]:
        sl = tpl.slots[i]
        orig = tpl.text[sl.start:sl.end]
        keys = pool_keys(sl, orig)
        if keys:
            got = self.inv.pick(keys, rng, avoid=orig.rstrip("/"))
            if got is not None:
                value, frm = got
                if sl.type == "sha":
                    value = value[:len(orig)]
                if sl.type == "path" and orig.endswith("/") and not value.endswith("/"):
                    value += "/"
                return value, frm
            if sl.type == "sha":
                return gen_value("sha", orig, rng, self.base), "generated"
            return "", ""
        if sl.type in GENERATED:
            return gen_value(sl.type, orig, rng, self.base), "generated"
        return "", ""

    def fill(self, tpl: Template, part: int, idxs: Iterable[int], rng: random.Random,
             values: dict[tuple[int, int], tuple[str, str]]) -> bool:
        for i in idxs:
            v, frm = self.draw(tpl, i, rng)
            if not v:
                if tpl.slots[i].must_fill:
                    return False
                continue
            values[(part, i)] = (v, frm)
        return True

    # -- transforms --------------------------------------------------------------

    def partners(self, tpl: Template) -> tuple[list[Template], list[Template]]:
        same = self._by_intent.get(tpl.intent, [])
        others = [t for t in same if t.id != tpl.id and self._uses[t.id] < self.s.max_uses
                  and frozenset((tpl.id, t.id)) not in self._pairs]
        return [t for t in others if t.subcategory == tpl.subcategory], others

    def candidate(self, tpl: Template, rng: random.Random) -> Candidate | None:
        opts: list[tuple[str, float]] = []
        if tpl.varying:
            opts.append(("slot_fill", 0.45))
            if len(tpl.slots) >= 2:
                opts.append(("slot_swap", 0.2))
        near, anyp = self.partners(tpl)
        if anyp:
            opts.append(("recombine", 0.35))
        if not opts:
            return None
        r = rng.random() * sum(w for _, w in opts)
        kind = opts[-1][0]
        for name, w in opts:
            if r < w:
                kind = name
                break
            r -= w
        values: dict[tuple[int, int], tuple[str, str]] = {}
        must = [i for i, sl in enumerate(tpl.slots) if sl.must_fill]
        if kind == "slot_fill":
            if not self.fill(tpl, 0, range(len(tpl.slots)), rng, values):
                return None
            parts = [(tpl, 0, len(tpl.text))]
        elif kind == "slot_swap":
            pick = tpl.varying[rng.randrange(len(tpl.varying))]
            if not self.fill(tpl, 0, sorted({pick, *must}), rng, values):
                return None
            parts = [(tpl, 0, len(tpl.text))]
        else:
            pool = near if (near and rng.random() < 0.7) else anyp
            other = pool[rng.randrange(len(pool))]
            na, nb = len(tpl.sentences), len(other.sentences)
            i = na if na == 1 or rng.random() < 0.6 else rng.randint(1, na - 1)
            j = 0 if nb == 1 or rng.random() < 0.5 else rng.randint(1, nb - 1)
            if i + (nb - j) > self.s.max_variant_sentences:
                return None
            parts = [(tpl, 0, tpl.sentences[i - 1][1]), (other, other.sentences[j][0], len(other.text))]
            for k, (t, s, e) in enumerate(parts):
                idxs = [x for x, sl in enumerate(t.slots) if s <= sl.start and sl.end <= e
                        and (sl.must_fill or sl.type not in VARYING or rng.random() < 0.5)]
                if not self.fill(t, k, idxs, rng, values):
                    return None
        prompt = render(parts, values)
        return Candidate(prompt, parts, values, kind, tpl.pattern_hash)

    # -- acceptance ----------------------------------------------------------------

    def problem(self, c: Candidate) -> str | None:
        if len(c.prompt) > self.s.max_variant_chars:
            return "variant_too_long"
        n = norm(c.prompt)
        if n in self.source_norms:
            return "same_as_real_prompt"
        if n in self._accepted_norms:
            return "duplicate_norm"
        if pii_kinds(c.prompt):
            return "pii"
        if self.persons.mentions(c.prompt):
            return "person_name"
        if home_path_literal(c.prompt):  # must-fill slots were refilled: any home path left is a leak
            return "home_path"
        if self.holdout is not None:
            r = self.holdout.check(c.prompt)
            if r:
                return r
        if self.manifest is not None and self.manifest.overlaps(c.prompt):
            return "injection_overlap"
        if c.transform == "recombine" and frozenset(t.id for t, _, _ in c.parts) in self._pairs:
            return "duplicate_pair"  # one recombination per unordered pair of real prompts
        if self._near_dup(c.prompt) is not None:
            return "near_duplicate"
        return None

    def _grams(self, text: str) -> set[int]:
        return {_ihash(g) for g in ngrams(tokens(text), self.s.dup_ngram)}

    def _near_dup(self, text: str) -> int | None:
        grams = self._grams(text)
        if not grams:
            return None
        shared: collections.Counter = collections.Counter()
        for g in grams:
            for doc in self._dup_index.get(g, ()):
                shared[doc] += 1
        for doc, n in shared.items():
            union = len(grams) + self._dup_sizes[doc] - n
            if union and n / union >= self.s.variant_dup:
                return doc
        return None

    def _accept(self, c: Candidate) -> None:
        for t, _, _ in c.parts:
            self._uses[t.id] += 1
        if c.transform == "recombine":
            self._pairs.add(frozenset(t.id for t, _, _ in c.parts))
        self._accepted_norms.add(norm(c.prompt))
        grams = self._grams(c.prompt)
        doc = len(self._dup_sizes)
        self._dup_sizes.append(len(grams))
        for g in grams:
            self._dup_index[g].append(doc)

    # -- the run -------------------------------------------------------------------

    def generate(self, name: str) -> list[dict[str, Any]]:
        self._by_intent: dict[str, list[Template]] = collections.defaultdict(list)
        for t in self.templates:
            self._by_intent[t.intent].append(t)
        self._accepted_norms: set[str] = set()
        self._dup_index: dict[int, list[int]] = collections.defaultdict(list)
        self._dup_sizes: list[int] = []
        self._uses: collections.Counter = collections.Counter()
        self._pairs: set[frozenset[str]] = set()
        order = sorted(self.templates, key=lambda t: rank(self.s.seed, "pattern", t.pattern_hash))
        by_pattern: dict[str, list[Candidate]] = {}
        for tpl in order:
            rng = random.Random(f"{self.s.seed}|{tpl.id}")
            got: list[Candidate] = []
            for _ in range(self.s.per_pattern * self.s.attempts_per_variant):
                if len(got) >= self.s.per_pattern or self._uses[tpl.id] >= self.s.max_uses:
                    break
                c = self.candidate(tpl, rng)
                self.stats["candidates"] += 1
                if c is None:  # no transform left, a refill impossible, or too many sentences
                    self.rejected["no_candidate"] += 1
                    continue
                why = self.problem(c)
                if why:
                    self.rejected[why] += 1
                    continue
                self._accept(c)
                got.append(c)
            if got:
                by_pattern[tpl.pattern_hash] = got
            else:
                self.stats["patterns_without_variants"] += 1
        rows = self.assign(by_pattern, [t.pattern_hash for t in order if t.pattern_hash in by_pattern], name)
        for row in rows:
            problems = validate_task(row) + verify_provenance(row, self.by_id, self.inv)
            if problems:
                raise TasksetError(f"internal: row failed its own checks ({problems[0]})")
        return rows

    def assign(self, by_pattern: dict[str, list[Candidate]], order: list[str], name: str
               ) -> list[dict[str, Any]]:
        rows_by_pattern: dict[str, list[dict[str, Any]]] = {}
        for ph in order:
            rows_by_pattern[ph] = [self.row(c, name) for c in by_pattern[ph]]
        n_coding = sum(1 for rs in rows_by_pattern.values() for r in rs if r["category"] == "coding")
        n_other = sum(len(rs) for rs in rows_by_pattern.values()) - n_coding
        cap = self.s.coding_cap
        coding_quota = n_coding if cap >= 1 else math.floor(cap * n_other / (1 - cap) + 1e-9)
        total_quota = self.s.max_total
        if total_quota is not None:
            coding_quota = min(coding_quota, math.floor(cap * total_quota + 1e-9))
        picked: list[dict[str, Any]] = []
        coding = 0
        depth = 0
        while True:
            progressed = False
            for ph in order:
                rs = rows_by_pattern[ph]
                if depth >= len(rs):
                    continue
                progressed = True
                r = rs[depth]
                if total_quota is not None and len(picked) >= total_quota:
                    self.stats["max_total_dropped"] += 1
                    continue
                if r["category"] == "coding":
                    if coding >= coding_quota:
                        self.stats["coding_cap_dropped"] += 1
                        continue
                    coding += 1
                picked.append(r)
            if not progressed:
                break
            depth += 1
        if cap < 1:
            # --max-total can leave fewer rows than its quota: hold the share anyway
            limit = math.floor(cap * (len(picked) - coding) / (1 - cap) + 1e-9)
            while coding > limit:
                last = max(i for i, r in enumerate(picked) if r["category"] == "coding")
                picked.pop(last)
                coding -= 1
                self.stats["coding_cap_dropped"] += 1
        self.assign_flags(picked)
        return picked

    def row(self, c: Candidate, name: str) -> dict[str, Any]:
        primary = c.parts[0][0]
        rid = "ts-" + h(norm(c.prompt))
        # every part must be chat-eligible (a chat intent, no path/file/url, no live state)
        chat_ok = all(t.chat_ok for t, _, _ in c.parts)
        mode = "chat" if chat_ok and rank(self.s.seed, "chat", rid) < self.s.chat_share else "agentic"
        category = "chat-mode" if mode == "chat" else SUBCAT_TO_CATEGORY[primary.subcategory]
        slots: list[dict[str, Any]] = []
        counter: collections.Counter = collections.Counter()
        for k, (t, s, e) in enumerate(c.parts):
            for i, sl in enumerate(t.slots):
                if sl.start < s or sl.end > e:
                    continue
                counter[sl.type] += 1
                got = c.values.get((k, i))
                value, frm = got if got else (t.text[sl.start:sl.end], "original")
                slots.append({"name": f"{sl.type}_{counter[sl.type]}", "type": sl.type, "value": value,
                              "from": frm, "source": k, "span": [sl.start, sl.end]})
        return {
            "id": rid, "taskset": name, "category": category, "subcategory": primary.subcategory,
            "intent": primary.intent, "mode": mode, "prompt": c.prompt, "slots": slots,
            "template_id": primary.id, "source_pattern_hash": primary.pattern_hash,
            "task_group": "tg-" + primary.pattern_hash, "transform": c.transform,
            "sources": [{"template_id": t.id, "pattern_hash": t.pattern_hash, "origin": t.origin,
                         "span": [s, e]} for t, s, e in c.parts],
            "flags": {"suppress_tools": mode == "chat", "inject_errors": False, "force_compaction": False},
            "provenance": PROVENANCE,
        }

    def assign_flags(self, rows: list[dict[str, Any]]) -> None:
        def take(eligible: list[dict[str, Any]], share: float, key: str) -> None:
            ranked = sorted(eligible, key=lambda r: rank(self.s.seed, key, r["id"]))
            for r in ranked[:round(share * len(eligible))]:
                r["flags"][key] = True

        take([r for r in rows if r["mode"] == "agentic"], self.s.error_share, "inject_errors")
        take(rows, self.s.compaction_share, "force_compaction")


# ---------------------------------------------------------------------------
# Row checks (also the reader teacher_gen.py uses)
# ---------------------------------------------------------------------------

_ID = re.compile(r"ts-[0-9a-f]{16}")
_TPL = re.compile(r"tpl-[0-9a-f]{16}")
_HASH = re.compile(r"[0-9a-f]{16}")


def validate_task(row: Any) -> list[str]:
    """Schema problems of one taskset row (empty: valid)."""
    if not isinstance(row, dict):
        return ["row is not an object"]
    p: list[str] = []
    want = {"id": str, "taskset": str, "category": str, "subcategory": str, "intent": str, "mode": str,
            "prompt": str, "slots": list, "template_id": str, "source_pattern_hash": str,
            "task_group": str, "transform": str, "sources": list, "flags": dict, "provenance": str}
    for k, typ in want.items():
        if not isinstance(row.get(k), typ):
            p.append(f"{k}: missing or not {typ.__name__}")
    extra = set(row) - set(want)
    if extra:
        p.append(f"unknown keys {sorted(extra)}")
    if p:
        return p
    if not _ID.fullmatch(row["id"]):
        p.append("id: bad format")
    elif row["id"] != "ts-" + h(norm(row["prompt"])):
        p.append("id: does not match the prompt")
    if not row["prompt"].strip():
        p.append("prompt: empty")
    for k, allowed in (("category", CATEGORIES), ("subcategory", SUBCATEGORIES), ("intent", INTENTS),
                       ("mode", MODES), ("transform", TRANSFORMS)):
        if row[k] not in allowed:
            p.append(f"{k}: {row[k]!r} not allowed")
    if row["provenance"] != PROVENANCE:
        p.append("provenance: not " + PROVENANCE)
    flags = row["flags"]
    if set(flags) != {"suppress_tools", "inject_errors", "force_compaction"} or not all(
            isinstance(v, bool) for v in flags.values()):
        p.append("flags: need exactly suppress_tools/inject_errors/force_compaction booleans")
    else:
        if flags["suppress_tools"] != (row["mode"] == "chat"):
            p.append("flags.suppress_tools must equal mode == chat")
        if flags["inject_errors"] and row["mode"] != "agentic":
            p.append("flags.inject_errors on a chat task")
    if (row["category"] == "chat-mode") != (row["mode"] == "chat"):
        p.append("category chat-mode iff mode chat")
    if not _TPL.fullmatch(row["template_id"]) or not _HASH.fullmatch(row["source_pattern_hash"]):
        p.append("template_id/source_pattern_hash: bad format")
    elif row["task_group"] != "tg-" + row["source_pattern_hash"]:
        p.append("task_group must be 'tg-' + source_pattern_hash")
    srcs = row["sources"]
    if not srcs or (row["transform"] == "recombine") != (len(srcs) == 2) or len(srcs) > 2:
        p.append("sources: one per template (two for recombine)")
    for src in srcs:
        if not (isinstance(src, dict) and _TPL.fullmatch(str(src.get("template_id")))
                and _HASH.fullmatch(str(src.get("pattern_hash"))) and src.get("origin") in ("golden", "lcm")
                and isinstance(src.get("span"), list) and len(src["span"]) == 2
                and all(isinstance(x, int) for x in src["span"])):
            p.append("sources: bad entry")
            break
    else:
        if srcs and (srcs[0]["template_id"] != row["template_id"]
                     or srcs[0]["pattern_hash"] != row["source_pattern_hash"]):
            p.append("template_id/source_pattern_hash must name sources[0]")
    for sl in row["slots"]:
        if not (isinstance(sl, dict) and isinstance(sl.get("name"), str) and isinstance(sl.get("type"), str)
                and isinstance(sl.get("value"), str) and sl.get("from") in FROM_VALUES
                and isinstance(sl.get("source"), int) and 0 <= sl["source"] < len(srcs)
                and isinstance(sl.get("span"), list) and len(sl["span"]) == 2):
            p.append("slots: bad entry")
            break
    return p


def verify_provenance(row: dict[str, Any], templates: dict[str, Template], inv: Inventory) -> list[str]:
    """Rebuild the prompt from its sources' real text and its slot values; check
    every value against its pool or generator (empty: traceable)."""
    p: list[str] = []
    parts: list[tuple[Template, int, int]] = []
    for src in row["sources"]:
        t = templates.get(src["template_id"])
        if t is None or t.pattern_hash != src["pattern_hash"]:
            return ["source template unknown"]
        s, e = src["span"]
        if not (0 <= s < e <= len(t.text)) or (s, e) != _sentence_span(t, s, e):
            return ["source span is not a sentence range"]
        parts.append((t, s, e))
    values: dict[tuple[int, int], tuple[str, str]] = {}
    for sl in row["slots"]:
        t = parts[sl["source"]][0]
        idx = next((i for i, x in enumerate(t.slots) if [x.start, x.end] == sl["span"]), None)
        if idx is None:
            p.append("slot span is not a template slot")
            continue
        slot = t.slots[idx]
        orig = t.text[slot.start:slot.end]
        if slot.type != sl["type"]:
            p.append("slot type differs from the template")
        frm, v = sl["from"], sl["value"]
        if frm == "original":
            if v != orig or slot.must_fill:
                p.append("original value differs or must be refilled")
            continue
        if frm == "generated":
            if not generated_shape_ok(slot.type, v) or (slot.type not in GENERATED and slot.type != "sha"):
                p.append(f"generated {slot.type} value has the wrong shape")
        else:
            keys = pool_keys(slot, orig)
            probe = v.rstrip("/") if slot.type == "path" else v
            if slot.type == "sha":
                ok = any(x.startswith(v) for x in inv.values("sha")) and len(v) == len(orig)
            else:
                ok = any(inv.has(k, probe, frm) for k in keys)
            if not ok:
                p.append(f"{slot.type} value is not in its pool")
        values[(sl["source"], idx)] = (v, frm)
    covered = {(k, i) for k, (t, s, e) in enumerate(parts) for i, x in enumerate(t.slots)
               if s <= x.start and x.end <= e}
    listed = {(sl["source"], next((i for i, x in enumerate(parts[sl["source"]][0].slots)
                                   if [x.start, x.end] == sl["span"]), -1)) for sl in row["slots"]}
    if covered != listed or len(row["slots"]) != len(covered):
        p.append("slots do not list every template slot in the spans exactly once")
    if render(parts, values) != row["prompt"]:
        p.append("prompt does not rebuild from its sources and slots")
    return p


def _sentence_span(t: Template, s: int, e: int) -> tuple[int, int]:
    starts = {a for a, _ in t.sentences}
    ends = {b for _, b in t.sentences}
    return (s, e) if s in starts and e in ends else (-1, -1)


def load_taskset(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a taskset file (raises TasksetError on any bad row)."""
    rows = []
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            problems = validate_task(row)
            if problems:
                raise TasksetError(f"line {n}: {problems[0]}")
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(xs: list[int], q: float) -> int | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def token_stats(texts: list[str], tokenizer_dir: Path | None) -> dict[str, Any] | None:
    if tokenizer_dir is None:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except ImportError:
        return {"error": "tokenizers not installed"}
    tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    counts = [len(tok.encode(t).ids) for t in texts]
    return {"p50": _pct(counts, 0.5), "p90": _pct(counts, 0.9), "max": max(counts) if counts else None}


def build_report(gen: Generator, rows: list[dict[str, Any]], *, name: str, inputs: dict[str, Any],
                 elapsed: float, tokenizer_dir: Path | None) -> dict[str, Any]:
    c = collections.Counter
    per_pattern = c(r["source_pattern_hash"] for r in rows)
    reuse = c(src["template_id"] for r in rows for src in r["sources"])
    used_sources = sorted(reuse)
    slot_types = c(sl.type for t in gen.templates for sl in t.slots)
    out_rows = {
        "tasks": len(rows),
        "by_category": dict(sorted(c(r["category"] for r in rows).items())),
        "by_mode": dict(sorted(c(r["mode"] for r in rows).items())),
        "by_category_mode": dict(sorted(c(f"{r['category']}|{r['mode']}" for r in rows).items())),
        "by_subcategory": dict(sorted(c(r["subcategory"] for r in rows).items())),
        "by_intent": dict(sorted(c(r["intent"] for r in rows).items())),
        "by_transform": dict(sorted(c(r["transform"] for r in rows).items())),
        "by_source_origin": dict(sorted(c(r["sources"][0]["origin"] for r in rows).items())),
        "flags": {k: sum(1 for r in rows if r["flags"][k]) for k in
                  ("suppress_tools", "inject_errors", "force_compaction")},
        "coding_share": round(sum(1 for r in rows if r["category"] == "coding") / len(rows), 4) if rows else None,
        "patterns_total": len(gen.templates),
        "patterns_used": len(per_pattern),
        "variants_per_pattern": {"min": min(per_pattern.values()) if per_pattern else None,
                                 "p50": _pct(list(per_pattern.values()), 0.5),
                                 "p90": _pct(list(per_pattern.values()), 0.9),
                                 "max": max(per_pattern.values()) if per_pattern else None},
        "prompt_chars": {"p50": _pct([len(r["prompt"]) for r in rows], 0.5),
                         "p90": _pct([len(r["prompt"]) for r in rows], 0.9),
                         "max": max((len(r["prompt"]) for r in rows), default=None)},
        "prompt_tokens": token_stats([r["prompt"] for r in rows], tokenizer_dir),
        "slot_fills_by_type": dict(sorted(c(sl["type"] for r in rows for sl in r["slots"]
                                             if sl["from"] != "original").items())),
        "slot_values_from": dict(sorted(c(sl["from"] for r in rows for sl in r["slots"]).items())),
        "real_prompt_reuse": {"p50": _pct(list(reuse.values()), 0.5), "p90": _pct(list(reuse.values()), 0.9),
                              "max": max(reuse.values()) if reuse else None,
                              "real_prompts_used": len(reuse)},
        "distinct_recombined_pairs": len({tuple(s["template_id"] for s in r["sources"])
                                          for r in rows if r["transform"] == "recombine"}),
    }
    return {
        "schema": REPORT_SCHEMA, "taskset_schema": TASKSET_SCHEMA, "generator_version": GENERATOR_VERSION,
        "taskset": name, "provenance": PROVENANCE,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1), "settings": {k: (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in asdict(gen.s).items()},
        "inputs": inputs,
        "inventory": gen.inv.sizes(),
        "sources": {
            "real_prompts": gen.stats["real_prompts"],
            "by_origin": dict(sorted(gen.origins.items())),
            "excluded": dict(sorted(gen.excluded.items())),
            "near_duplicate_prompts_merged": gen.stats["near_duplicate_prompts_merged"],
            "patterns": len(gen.templates),
            "patterns_with_content_slots": sum(1 for t in gen.templates if t.varying),
            "patterns_multi_sentence": sum(1 for t in gen.templates if len(t.sentences) > 1),
            "patterns_chat_eligible": sum(1 for t in gen.templates if t.chat_ok),
            "slot_types": dict(sorted(slot_types.items())),
            "intents": dict(sorted(c(t.intent for t in gen.templates).items())),
            "subcategories": dict(sorted(c(t.subcategory for t in gen.templates).items())),
        },
        "generation": {
            "candidates": gen.stats["candidates"],
            "rejected": dict(sorted(gen.rejected.items())),
            "patterns_without_variants": gen.stats["patterns_without_variants"],
            "coding_cap_dropped": gen.stats["coding_cap_dropped"],
            "max_total_dropped": gen.stats["max_total_dropped"],
        },
        "output": out_rows,
        "diversity": {
            "output": diversity([r["prompt"] for r in rows], gen.s.seed),
            "source_prompts_used": diversity([gen.by_id[i].text for i in used_sources], gen.s.seed),
        },
        "checks": {
            "rows_validated": len(rows), "provenance_verified": len(rows),
            "distinct_normalized_prompts": len({norm(r["prompt"]) for r in rows}),
            "same_as_a_real_prompt": sum(1 for r in rows if norm(r["prompt"]) in gen.source_norms),
            # None, not 0, when the check was off (--no-holdout / --no-injection-manifest):
            # an unchecked taskset must not read as a clean one
            "holdout_used": gen.holdout is not None,
            "injection_manifest_used": gen.manifest is not None,
            "holdout_overlaps": (sum(1 for r in rows if gen.holdout.check(r["prompt"]))
                                 if gen.holdout is not None else None),
            "injection_overlaps": (sum(1 for r in rows if gen.manifest.overlaps(r["prompt"]))
                                   if gen.manifest is not None else None),
            "pii_hits": sum(1 for r in rows if pii_kinds(r["prompt"])),
            "home_path_hits": sum(1 for r in rows if home_path_literal(r["prompt"])),
            "snapshot_user_rows_required": gen.require_user_row,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand(p: str | None) -> Path | None:
    return Path(p).expanduser().resolve() if p else None


def _inside_git_tree(path: Path) -> bool:
    for d in [path, *path.parents]:
        if (d / ".git").exists():
            return True
    return False


def _refuse_live(path: Path) -> None:
    live = Path(LIVE_STORE).expanduser().resolve()
    if path == live or live in path.parents:
        raise TasksetError(f"{path} is in the live store; pass a snapshot (e.g. {DEFAULT_LCM})")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Excluded sessions (DISTILL-OPEN-QUESTIONS #6: the 2026-10-13 Token-Plan incident session)
# ---------------------------------------------------------------------------

EXCLUSION_SCHEMA = "distill-taskset-exclusion/1"


def read_session_list(path: Path) -> set[str]:
    """Session ids, one per line (blank lines and ``#`` comments skipped: build.py's format). A file that
    lists none is an error, so an empty list cannot stand in for the exclusion."""
    ids = {line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()}
    ids.discard("")
    if not ids:
        raise TasksetError(f"{path} lists no session id")
    return ids


def excluded_pattern_hashes(sessions: set[str], trace_paths: list[Path], lcm_path: Path | None,
                            stats: collections.Counter) -> set[str]:
    """The task proxies (``h(norm(text))``, the taskset's ``pattern_hash``) of every user message a listed
    session holds: each user message of its golden traces (any surface) and each of its user rows in the
    snapshot (the text and the joined text blocks), however labelled. Wider than what ``build`` mines on
    purpose: no prompt of a listed session can source a task."""
    out: set[str] = set()
    for p in trace_paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    trace = json.loads(line)
                    meta = trace.get("_meta") or {}
                    msgs = trace["messages"]
                except (ValueError, KeyError, TypeError, AttributeError):
                    stats["golden_unparseable"] += 1
                    continue
                if str(meta.get("session_id") or "").strip() not in sessions:
                    continue
                stats["golden_rows_in_listed_sessions"] += 1
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "user":
                        text = content_text(m.get("content"))
                        if text.strip():
                            out.add(h(norm(text)))
    if lcm_path is not None:
        con = sqlite3.connect(f"file:{lcm_path}?mode=ro", uri=True)
        try:
            marks = ",".join("?" * len(sessions))
            for content, cj in con.execute(f"SELECT content, content_json FROM lcm_messages WHERE role = 'user' "
                                           f"AND session_id IN ({marks})", sorted(sessions)):
                stats["lcm_rows_in_listed_sessions"] += 1
                texts = [content or ""]
                try:
                    blocks = json.loads(cj) if cj else None
                except ValueError:
                    blocks = None
                if isinstance(blocks, list):
                    texts.append(content_text(blocks))
                out.update(h(norm(t)) for t in texts if t.strip())
        finally:
            con.close()
    return out


def row_patterns(row: dict[str, Any]) -> set[str]:
    """Every real prompt a taskset row was built from (its own and a recombined partner's)."""
    return {str(row.get("source_pattern_hash"))} | {str(s.get("pattern_hash")) for s in row.get("sources") or []
                                                     if isinstance(s, dict)}


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def run_exclude(a: argparse.Namespace) -> int:
    """``exclude``: drop every taskset row built from a prompt of a listed session (any source); writes
    ``<out>`` (default ``<stem>-excl.jsonl`` beside the input) and ``<out stem>.exclusion.json`` (counts and
    hashes of the inputs, never the ids or any text). ``--check`` writes nothing: exit 0 when no row would
    be dropped, 1 when some would (gen_runner.py's preflight)."""
    tasks = _expand(a.tasks)
    sessions_path = _expand(a.exclude_sessions)
    assert tasks is not None and sessions_path is not None
    sessions = read_session_list(sessions_path)
    rows = load_taskset(tasks)
    paths = sorted({Path(p) for pat in (a.traces or [DEFAULT_TRACES])
                    for p in glob.glob(os.path.expanduser(pat))}, key=lambda p: p.name)
    if not paths:
        raise TasksetError("no golden trace file matched (--traces): the listed sessions' prompts cannot be found")
    lcm = None
    if a.lcm_snapshot:
        lcm = _expand(a.lcm_snapshot)
        assert lcm is not None
        _refuse_live(lcm)
        if not lcm.exists():
            raise TasksetError(f"lcm snapshot {lcm} not found ('' to skip)")
    stats: collections.Counter = collections.Counter()
    excluded = excluded_pattern_hashes(sessions, paths, lcm, stats)
    keep = [r for r in rows if not row_patterns(r) & excluded]
    dropped = [r for r in rows if row_patterns(r) & excluded]
    summary: dict[str, Any] = {
        "schema": EXCLUSION_SCHEMA, "taskset": sorted({str(r.get("taskset")) for r in rows}),
        "input": tasks.name, "input_sha256": _sha256(tasks), "sessions_listed": len(sessions),
        "session_list_sha256": _sha256(sessions_path), "golden_files": len(paths), "lcm_snapshot": bool(lcm),
        "golden_rows_in_listed_sessions": stats["golden_rows_in_listed_sessions"],
        "lcm_rows_in_listed_sessions": stats["lcm_rows_in_listed_sessions"],
        "patterns_excluded": len(excluded), "rows_in": len(rows), "rows_out": len(keep),
        "rows_dropped": len(dropped), "task_groups_dropped": len({r["task_group"] for r in dropped}),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if a.check:
        summary["check"] = "clean" if not dropped else "holds rows of listed sessions"
        print(json.dumps(summary, sort_keys=True))
        return 0 if not dropped else 1
    out = _expand(a.out) if a.out else tasks.with_name(f"{tasks.stem}-excl.jsonl")
    assert out is not None
    if out == tasks:
        raise TasksetError("--out must not be the input taskset")
    if _inside_git_tree(out.parent):
        raise TasksetError(f"{out} is inside a git work tree; task text never goes near a repo")
    os.makedirs(out.parent, mode=0o700, exist_ok=True)
    _write_atomic(out, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in keep))
    summary.update(output=out.name, output_sha256=_sha256(out))
    _write_atomic(out.with_name(f"{out.stem}.exclusion.json"), json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_build(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.monotonic()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", a.name):
        raise TasksetError("--name: letters, digits, dot, dash, underscore")
    s = Settings(seed=a.seed, per_pattern=a.per_pattern, max_uses=a.max_uses, coding_cap=a.coding_cap,
                 chat_share=a.chat_share,
                 error_share=a.error_share, compaction_share=a.compaction_share, max_total=a.max_total,
                 max_prompt_chars=a.max_prompt_chars, holdout_containment=a.holdout_containment,
                 surfaces=tuple(x.strip() for x in a.surfaces.split(",") if x.strip()),
                 base_date=a.base_date)
    s.check()
    out = _expand(a.out)
    assert out is not None
    if _inside_git_tree(out):
        raise TasksetError(f"{out} is inside a git work tree; task text never goes near a repo")
    stats: collections.Counter = collections.Counter()
    inputs: dict[str, Any] = {}

    persons = PersonNames.from_wiki(_expand(a.wiki)) if a.wiki else PersonNames()
    inputs["person_deny_list_entries"] = persons.count
    repo = _expand(a.repo)
    tools = tool_names_from(_expand(a.pack) if a.pack and Path(a.pack).expanduser().exists() else None,
                            _expand(a.tool_schemas))
    inv = Inventory.build(repo=repo, wiki=_expand(a.wiki), wiki_sections=a.wiki_sections.split(","),
                          tool_names=tools, cron_names=cron_names_from(_expand(a.cron_jobs)),
                          persons=persons, stats=stats)

    holdout = None
    if not a.no_holdout:
        holdout = Holdout(s.holdout_k, s.holdout_containment)
        for hp in a.holdout or list(DEFAULT_HOLDOUT):
            path = _expand(hp)
            assert path is not None
            if not path.exists():
                raise TasksetError(f"holdout file {path} not found (--no-holdout to run without)")
            holdout.load(path)
        inputs["holdout"] = {"files": len(a.holdout or DEFAULT_HOLDOUT), "rows": holdout.rows,
                             "task_ids": len(holdout.task_ids), "user_messages": len(holdout.norms)}
    manifest = None
    if not a.no_injection_manifest:
        mp = _expand(a.injection_manifest)
        assert mp is not None
        if not mp.exists():
            raise TasksetError(f"injection manifest {mp} not found (--no-injection-manifest to run without)")
        manifest = BES.InjectionManifest.load(mp)
        inputs["injection_manifest_cases"] = len(manifest.data.get("cases") or [])

    paths = sorted({Path(p) for pat in (a.traces or [DEFAULT_TRACES])
                    for p in glob.glob(os.path.expanduser(pat))}, key=lambda p: p.name)
    index = load_index(_expand(a.index) if a.index and Path(a.index).expanduser().exists() else None)
    golden = load_golden(paths, index, s.surfaces, stats)
    lcm: dict[str, RealPrompt] = {}
    injected: set[str] = set()
    if a.lcm_snapshot:
        lp = _expand(a.lcm_snapshot)
        assert lp is not None
        _refuse_live(lp)
        if not lp.exists():
            raise TasksetError(f"lcm snapshot {lp} not found")
        lcm = load_lcm(lp, s.surfaces, stats, injected)
    prompts = dict(golden)
    for ph, rp in lcm.items():
        if ph in prompts:
            prompts[ph].in_lcm = True
        else:
            prompts[ph] = rp
    if a.exclude_sessions:
        sp = _expand(a.exclude_sessions)
        assert sp is not None
        sessions = read_session_list(sp)
        excluded = excluded_pattern_hashes(sessions, paths, _expand(a.lcm_snapshot) if a.lcm_snapshot else None,
                                           stats)
        gone = [ph for ph in prompts if ph in excluded]
        for ph in gone:
            del prompts[ph]
        inputs["excluded_sessions"] = {"listed": len(sessions), "prompts_dropped": len(gone)}
    inputs.update({"golden_files": len(paths), "index_rows": len(index),
                   "golden_prompts": len(golden), "lcm_prompts": len(lcm),
                   "golden_and_lcm": sum(1 for rp in golden.values() if rp.in_lcm),
                   "lcm_injected_texts": len(injected),
                   "read_stats": dict(sorted(stats.items()))})
    if not prompts:
        raise TasksetError("no real prompts read")

    lexicon = Lexicon.load(_expand(a.dictionary)) if a.dictionary else Lexicon(None)
    inputs["dictionary_words"] = len(lexicon.words) if lexicon.words is not None else None
    # with a snapshot, a golden prompt must be a user-typed row there (fail closed:
    # the golden exporter keeps no provenance, so an injected turn looks typed)
    gen = Generator(s, inv, holdout=holdout, manifest=manifest, persons=persons, lexicon=lexicon,
                    injected=injected, require_user_row=bool(a.lcm_snapshot))
    gen.add_prompts(prompts.values())
    rows = gen.generate(a.name)
    report = build_report(gen, rows, name=a.name, inputs=inputs, elapsed=time.monotonic() - t0,
                          tokenizer_dir=_expand(a.tokenizer_dir))
    report["dry_run"] = bool(a.dry_run)
    os.makedirs(out, mode=0o700, exist_ok=True)
    if not a.dry_run:
        _write_atomic(out / f"{a.name}.jsonl",
                      "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    _write_atomic(out / f"{a.name}.report.json", json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="mine real prompts and write a taskset (host only; inputs read only)")
    b.add_argument("--name", required=True)
    b.add_argument("--out", default=DEFAULT_OUT)
    b.add_argument("--traces", action="append", help=f"glob, repeatable (default {DEFAULT_TRACES})")
    b.add_argument("--index", default=DEFAULT_INDEX)
    b.add_argument("--lcm-snapshot", default=DEFAULT_LCM, help="'' to skip")
    b.add_argument("--holdout", action="append", help="held-out case JSONL, repeatable (default: build-v2 "
                   "eval-tools + val-cases)")
    b.add_argument("--no-holdout", action="store_true")
    b.add_argument("--injection-manifest", default=DEFAULT_MANIFEST)
    b.add_argument("--no-injection-manifest", action="store_true")
    b.add_argument("--repo", default=DEFAULT_REPO)
    b.add_argument("--wiki", default=DEFAULT_WIKI)
    b.add_argument("--wiki-sections", default=",".join(DEFAULT_WIKI_SECTIONS))
    b.add_argument("--pack", default=DEFAULT_PACK)
    b.add_argument("--tool-schemas", default=str(Path(__file__).with_name("eval_tool_schemas.json")))
    b.add_argument("--cron-jobs", default=DEFAULT_CRON)
    b.add_argument("--tokenizer-dir", default=None)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--per-pattern", type=int, default=Settings.per_pattern)
    b.add_argument("--max-uses", type=int, default=Settings.max_uses,
                   help="appearances of one real prompt across all variants")
    b.add_argument("--dictionary", default="/usr/share/dict/words",
                   help="word list: names that are ordinary words never become slots ('' for none)")
    b.add_argument("--coding-cap", type=float, default=Settings.coding_cap)
    b.add_argument("--chat-share", type=float, default=Settings.chat_share)
    b.add_argument("--error-share", type=float, default=Settings.error_share)
    b.add_argument("--compaction-share", type=float, default=Settings.compaction_share)
    b.add_argument("--max-total", type=int, default=None)
    b.add_argument("--max-prompt-chars", type=int, default=Settings.max_prompt_chars)
    b.add_argument("--holdout-containment", type=float, default=Settings.holdout_containment)
    b.add_argument("--surfaces", default=",".join(HUMAN_SURFACES))
    b.add_argument("--base-date", default=Settings.base_date)
    b.add_argument("--dry-run", action="store_true", help="write the report only")
    b.add_argument("--exclude-sessions", metavar="FILE",
                   help="session ids, one per line: no prompt of a listed session sources a task")
    v = sub.add_parser("validate", help="schema-check a taskset file")
    v.add_argument("path")
    x = sub.add_parser("exclude", help="drop the rows built from a listed session's prompts (or --check for any)")
    x.add_argument("--tasks", required=True, help="the taskset to filter")
    x.add_argument("--exclude-sessions", required=True, metavar="FILE", help="session ids, one per line")
    x.add_argument("--traces", action="append", help=f"glob, repeatable (default {DEFAULT_TRACES})")
    x.add_argument("--lcm-snapshot", default=DEFAULT_LCM, help="'' to skip")
    x.add_argument("--out", help="default: <stem>-excl.jsonl beside --tasks")
    x.add_argument("--check", action="store_true",
                   help="write nothing; exit 0 when no row comes from a listed session, 1 when some do")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        if a.cmd == "validate":
            rows = load_taskset(a.path)
            print(json.dumps({"rows": len(rows), "valid": True}))
            return 0
        if a.cmd == "exclude":
            return run_exclude(a)
        report = run_build(a)
    except (TasksetError, BES.BuildError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"taskset_gen: {exc}", file=sys.stderr)
        return 2
    summary = {k: report[k] for k in ("taskset", "dry_run")}
    summary.update({"tasks": report["output"]["tasks"], "by_category_mode": report["output"]["by_category_mode"],
                    "patterns_used": report["output"]["patterns_used"],
                    "patterns_total": report["output"]["patterns_total"],
                    "diversity": report["diversity"]["output"]})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
