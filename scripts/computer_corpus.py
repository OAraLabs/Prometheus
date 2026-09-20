"""Read the candidate-table corpus, and record what the correct answer was.

WHY THIS SCRIPT EXISTS
----------------------
This repo has no post-hoc human-annotation path of any kind. Every human label
in it is authored UP FRONT — gym taskset ``score:`` predicates, escalation
fixtures, golden ``SKILL.md`` files. None of them is a judgment made about a run
that already happened, which is exactly what a corpus needs.

So the interface had to be built regardless of what the storage was, which is
part of why the storage is a database rather than a text file: editing JSONL by
hand was never going to be the annotation path either.

    list        what has been captured, and how much is unreviewed
    show        one table, as the chooser saw it
    annotate    record the correct answer
    score       score a chooser's recorded answer against the judgments

THE ANNOTATION IS NOT THE EXECUTION
------------------------------------
``executed_candidate_id`` is what ran. ``correct_candidate_id`` is what should
have run. They differ exactly when the run was wrong, which is the case the
corpus exists to capture, so ``annotate`` never defaults to what was executed —
it would quietly turn every mistake into a correct label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prometheus.computer.corpus import (  # noqa: E402
    ANNOTATION_NONE_CORRECT,
    ANNOTATION_TABLE_UNUSABLE,
    ACCURACY_GRADE_SOURCES,
    CorpusStore,
    IncompleteCorpus,
    assert_corpus_complete,
    load_corpus,
)
from prometheus.config.paths import get_computer_corpus_db_path  # noqa: E402


def _store(args: argparse.Namespace) -> CorpusStore:
    return CorpusStore(args.db or get_computer_corpus_db_path())


def _cmd_list(args: argparse.Namespace) -> int:
    store = _store(args)
    corpus = load_corpus(store)
    print(corpus.summary())
    print()
    print(f"{'record_id':<34} {'app':<22} {'N':>3} {'status':<10} {'correct':<16}")
    print("-" * 92)

    def _rows(bucket, label):
        for r in bucket:
            print(f"{r['record_id']:<34} {r['app'][:22]:<22} "
                  f"{r['candidate_count']:>3} {r['status'][:10]:<10} {label(r):<16}")

    _rows(corpus.unannotated, lambda r: "— UNREVIEWED")
    _rows(corpus.answered, lambda r: r["correct_candidate_id"])
    _rows(corpus.none_correct, lambda r: ANNOTATION_NONE_CORRECT)
    _rows(corpus.unusable, lambda r: ANNOTATION_TABLE_UNUSABLE)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    store = _store(args)
    row = store.get_table(args.record_id)
    if row is None:
        print(f"no record {args.record_id!r}", file=sys.stderr)
        return 1
    import json

    candidates = json.loads(row["candidates_json"])
    print(f"record   : {row['record_id']}")
    print(f"app      : {row['app']}  window {row['window_id']}  target {row['target']}")
    print(f"goal     : {row['goal']}")
    print(f"status   : {row['status']}   {row['reason']}")
    print(f"deterministic : {row['deterministic_id']!r} "
          f"(source {row['deterministic_chooser']!r}, "
          f"confidence {row['deterministic_confidence']}, "
          f"abstained={bool(row['deterministic_abstained'])})")
    print(f"shadow        : {row['shadow_id']!r} "
          f"({row['shadow_chooser'] or 'none configured'}, "
          f"{row['shadow_latency_ms']}ms)")
    print(f"gate          : {row['gate_decision']!r} "
          f"approval required={row['gate_approval_required']} "
          f"granted={row['gate_approval_granted']}")
    print(f"executed : {row['executed_candidate_id']!r}")
    ann = store.latest_annotations().get(args.record_id)
    print(f"correct  : {ann['correct_candidate_id']!r} (by {ann['annotated_by']})"
          if ann else "correct  : — NOT YET REVIEWED")
    print()
    print("the table, as the chooser saw it:")
    for c in candidates:
        print(f"  {c['id']:<14} {c['description']}")
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        store.record_annotation(
            args.record_id, args.answer,
            annotated_by=args.by, label_source=args.label_source,
            note=args.note,
        )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"recorded: {args.record_id} -> {args.answer}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Score the chooser whose answer is stored on each row.

    Reports the two populations separately, because they answer different
    questions and a single blended accuracy hides both: on rows where the
    deterministic chooser ABSTAINED there is nothing to pair against and the
    question is coverage, while on rows where it answered the question is
    whether a replacement regresses.
    """
    corpus = load_corpus(_store(args))
    scorable = corpus.scorable
    if not scorable:
        print(corpus.summary())
        print("\nnothing scorable yet — annotate some rows first")
        return 1

    abstained = [r for r in scorable if r["deterministic_abstained"]]
    answered = [r for r in scorable if not r["deterministic_abstained"]]

    def _right(r) -> bool:
        if r["correct_candidate_id"] == ANNOTATION_NONE_CORRECT:
            return r["deterministic_abstained"]
        return r["deterministic_id"] == r["correct_candidate_id"]

    print(corpus.summary())
    print()
    for label, bucket in (
        ("chooser ANSWERED", answered), ("chooser ABSTAINED", abstained)
    ):
        if not bucket:
            print(f"{label:<20} 0 rows")
            continue
        hits = sum(1 for r in bucket if _right(r))
        sizes = [r["candidate_count"] for r in bucket]
        chance = sum(1 / n for n in sizes if n) / len(bucket)
        print(f"{label:<20} {hits}/{len(bucket)} matching the label "
              f"({100 * hits // len(bucket)}%) — chance floor "
              f"{100 * chance:.1f}%, median table {sorted(sizes)[len(sizes) // 2]}")
    print()
    print(f"WHAT THESE NUMBERS ARE: {corpus.score_noun()}")
    if set(corpus.label_mix()) - ACCURACY_GRADE_SOURCES:
        print("  ⚠ Some labels were PROPOSED BY A MODEL. A number over those rows")
        print("    measures how alike two models are — it is agreement, not")
        print("    accuracy, and it cannot be reported as accuracy.")
    print()
    print("⚠ The two populations above are not comparable to each other, and a")
    print("  blended number over both would be meaningless. See")
    print("  docs/computer-use-corpus.md.")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """HARVEST EXIT CRITERIA. Run this before annotating anything.

    A column empty in every row cannot be repaired afterwards — the value was
    never captured — so the harvest has to be fixed and re-run. Finding that
    out after a human has annotated 100 rows is the expensive version.
    """
    try:
        assert_corpus_complete(_store(args), mode=args.mode)
    except IncompleteCorpus as exc:
        print(f"INCOMPLETE\n{exc}", file=sys.stderr)
        return 1
    print("complete — every declared column is populated in at least one row")
    return 0


def _cmd_abstains(args: argparse.Namespace) -> int:
    """Print the ABSTAIN rows as one question each. Diagnostic, not labelling.

    One question per row: was a correct candidate present at all, and if not,
    why not. That measures THE TABLE. It is not `correct_id`, it does not touch
    the calibration set, and it is stored in its own table so it can never be
    counted as ground truth.
    """
    import json

    store = _store(args)
    rows = [r for r in store.all_tables() if r["deterministic_abstained"]]
    done = store.abstain_diagnostics()
    todo = [r for r in rows if r["record_id"] not in done]

    print(f"{len(rows)} abstain row(s); {len(todo)} unanswered\n")
    print("For each: was ANY candidate below the right thing to do?")
    print("  yes -> the TABLE was fine, the chooser missed it")
    print("  no  -> say which: verb_not_offered | key_not_offered |")
    print("                    element_not_in_tree | not_achievable_here\n")

    for i, r in enumerate(todo, 1):
        cands = r["candidates"]  # all_tables() already decoded it
        print("=" * 78)
        print(f"[{i}/{len(todo)}]  {r['record_id']}")
        print(f"  app  : {r['app']}   ({r['candidate_count']} candidates)")
        print(f"  GOAL : {r['goal']}")
        print("  table:")
        for c in cands[: args.max_candidates]:
            print(f"    {c['id']:<12} {c['description'][:72]}")
        if len(cands) > args.max_candidates:
            print(f"    … {len(cands) - args.max_candidates} more")
        print()
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Record one abstain diagnostic. NOT a label."""
    store = _store(args)
    try:
        store.record_abstain_diagnostic(
            args.record_id,
            correct_present=(args.answer == "yes"),
            reason=None if args.answer == "yes" else args.reason,
            answered_by=args.by, note=args.note,
        )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"recorded (diagnostic, not a label): {args.record_id} -> {args.answer}"
          + (f" / {args.reason}" if args.reason else ""))
    return 0


def _cmd_reasons(args: argparse.Namespace) -> int:
    """THE SPLIT that decides whether build_candidates is fixed first."""
    store = _store(args)
    diags = store.abstain_diagnostics()
    if not diags:
        print("no abstain diagnostics recorded yet — run `abstains` first")
        return 1
    present = sum(1 for d in diags.values() if d["correct_present"])
    by_reason: dict[str, int] = {}
    for d in diags.values():
        if not d["correct_present"]:
            by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1

    n = len(diags)
    print(f"{n} abstain row(s) diagnosed\n")
    print(f"  table was FINE, chooser missed it : {present}  ({100*present//n}%)")
    print(f"  no correct candidate existed      : {n-present}  ({100*(n-present)//n}%)")
    for k, v in sorted(by_reason.items()):
        print(f"      {k:<22} {v}")
    table_defects = sum(
        v for k, v in by_reason.items()
        if k in ("verb_not_offered", "key_not_offered")
    )
    print()
    if table_defects:
        print(f"  ⚠ {table_defects}/{n} ({100*table_defects//n}%) are TABLE DEFECTS —")
        print("    build_candidates never offers the verb or key the goal needs.")
        print("    No chooser can improve these. Fix build_candidates and")
        print("    re-measure BEFORE judging any classifier.")
    else:
        print("  No table defects. The abstain region is a genuine chooser gap.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=None, help="corpus database (default: the resolved one)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="what has been captured").set_defaults(fn=_cmd_list)

    s = sub.add_parser("show", help="one table, as the chooser saw it")
    s.add_argument("record_id")
    s.set_defaults(fn=_cmd_show)

    a = sub.add_parser("annotate", help="record the correct answer")
    a.add_argument("record_id")
    a.add_argument(
        "answer",
        help=(f"a candidate id, {ANNOTATION_NONE_CORRECT} (the table was fine "
              f"but nothing in it was right), or {ANNOTATION_TABLE_UNUSABLE} "
              f"(the table should not have existed)"),
    )
    a.add_argument("--by", required=True, help="who is asserting this")
    a.add_argument(
        "--label-source", required=True,
        choices=("human", "model", "model_confirmed"),
        help=("WHO decided. No default: a score over model-proposed labels is "
              "AGREEMENT WITH A MODEL, not accuracy, and it cannot be "
              "separated out afterwards if nobody recorded it."),
    )
    a.add_argument("--note", default="")
    a.set_defaults(fn=_cmd_annotate)

    sub.add_parser("score", help="score the recorded chooser answers").set_defaults(
        fn=_cmd_score
    )

    ck = sub.add_parser(
        "check", help="harvest exit criteria — no declared column universally empty"
    )
    ck.add_argument(
        "--mode", choices=("full", "capture"), default="full",
        help=("'capture' also excludes the five action-outcome columns, for a "
              "harvest that observed windows without executing anything. "
              "'full' is the default and holds a corpus of real runs to "
              "every column."),
    )
    ck.set_defaults(fn=_cmd_check)

    ab = sub.add_parser("abstains", help="print the ABSTAIN rows as questions")
    ab.add_argument("--max-candidates", type=int, default=40)
    ab.set_defaults(fn=_cmd_abstains)

    dg = sub.add_parser("diagnose", help="answer one abstain row (NOT a label)")
    dg.add_argument("record_id")
    dg.add_argument("answer", choices=("yes", "no"),
                    help="was a correct candidate present in the table?")
    dg.add_argument("--reason", default=None,
                    choices=("verb_not_offered", "key_not_offered",
                             "element_not_in_tree", "not_achievable_here"))
    dg.add_argument("--by", required=True)
    dg.add_argument("--note", default="")
    dg.set_defaults(fn=_cmd_diagnose)

    sub.add_parser("reasons", help="the abstain split").set_defaults(fn=_cmd_reasons)

    ms = sub.add_parser("mark-session", help="record a session as pilot or real")
    ms.add_argument("session")
    ms.add_argument("status", choices=("pilot", "real"))
    ms.add_argument("--note", default="")
    ms.set_defaults(fn=lambda a: (_store(a).mark_session(a.session, a.status, a.note),
                                  print(f"{a.session} -> {a.status}"))[1] or 0)

    args = p.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
