#!/usr/bin/env bash
# Phase 2 judging on Qwen Max through the PAY-PER-TOKEN key (QWEN_PAYG_KEY), never the Token Plan key.
#
#   judge_run.sh [--execute] [--rejudge] [--view outcome|blind] [--no-blind-fail]
#                [--budget-usd 20] [--blind-budget-usd 5] [--rejudge-budget-usd 40]
#                [--allow-missing 0] [--concurrency 2]
#
# DRY RUN by default: the preflight runs and the projected cost is printed. Nothing is sent.
# --execute: sample (when there is no queue yet), run, the blind fail-side check, report.
#   The blind fail-side check (--view blind --only fail) is its own ledger phase (2-judge-blind)
#   with its own --blind-budget-usd, so it is not left with what the outcome run did not spend.
# --allow-missing N: up to N sampled items may stay without a verdict (an item the API always
#   rejects, or the judge always answers unparseably). The run then counts as finished, the blind
#   check and the re-judge go ahead, and the report's gate is decided with N missing allowed.
# --rejudge (with --execute): when the report's gate says re-judge, run-all in REJUDGE_WORKDIR
#   under its own budget (the worst-agreement categories first).
#
# The preflight runs in every mode, and any failure stops before anything else runs. It checks
# that the key is present, that its prefix matches its host, and that it is not the Token Plan's
# key or host. It checks that the STOP file is absent, that the price table has the model in the
# key's region, that the ledger is readable and below the $140 alert, that the notify command
# exists, and that Prometheus imports (judge.py needs it).
#
# The key never appears on a command line or in the output: spend.py and judge.py read it
# themselves, from the environment or the 0600 env file. Every paid call goes through the one
# spend ledger, $HOME/distill/spend-ledger.csv (not configurable: a second ledger would be a
# second $150): a $150 hard cap, and at $140 the STOP file plus the notify.
#
# Environment (all optional): PY (python3), DISTILL (~/distill), WORKDIR ($DISTILL/phase2),
# REJUDGE_WORKDIR ($DISTILL/phase2-rejudge), NOTIFY ($DISTILL/notify.sh), MODEL (qwen3.8-max),
# PROMETHEUS_ENV_FILE (the env file). PYTHONPATH must make Prometheus importable.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
DISTILL=${DISTILL:-$HOME/distill}
WORKDIR=${WORKDIR:-$DISTILL/phase2}
REJUDGE_WORKDIR=${REJUDGE_WORKDIR:-$DISTILL/phase2-rejudge}
NOTIFY=${NOTIFY:-$DISTILL/notify.sh}
MODEL=${MODEL:-qwen3.8-max}

if [ -n "${LEDGER:-}" ] && [ "$LEDGER" != "$HOME/distill/spend-ledger.csv" ]; then
  echo "judge_run: refused: LEDGER is set, but the spend ledger is fixed at ~/distill/spend-ledger.csv" >&2
  exit 2
fi

execute=0
rejudge=0
blind_fail=1
view=outcome
budget=20
blind_budget=5
rejudge_budget=40
allow_missing=0
concurrency=2
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) execute=1 ;;
    --rejudge) rejudge=1 ;;
    --no-blind-fail) blind_fail=0 ;;
    --view) view=${2:?--view needs a value}; shift ;;
    --budget-usd) budget=${2:?--budget-usd needs a value}; shift ;;
    --blind-budget-usd) blind_budget=${2:?--blind-budget-usd needs a value}; shift ;;
    --rejudge-budget-usd) rejudge_budget=${2:?--rejudge-budget-usd needs a value}; shift ;;
    --allow-missing) allow_missing=${2:?--allow-missing needs a value}; shift ;;
    --concurrency) concurrency=${2:?--concurrency needs a value}; shift ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "judge_run: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$allow_missing" in
  ''|*[!0-9]*) echo "judge_run: --allow-missing needs a whole number" >&2; exit 2 ;;
esac

say() { printf 'judge_run: %s\n' "$*"; }
run_spend() { "$PY" "$here/spend.py" "$@"; }
run_judge() { "$PY" "$here/judge.py" "$@"; }
paid=(--provider max-payg --notify-cmd "$NOTIFY" --judge-model "$MODEL" --concurrency "$concurrency")
# A run that ended "incomplete" (exit 6) counts as finished when --allow-missing allows some items
# to stay without a verdict; the report's gate then says whether few enough are missing.
finished() { [ "$1" -eq 0 ] || { [ "$1" -eq 6 ] && [ "$allow_missing" -gt 0 ]; }; }
gate_of() {
  "$PY" -c 'import json, sys
try:
    print((json.load(open(sys.argv[1], encoding="utf-8")).get("gate") or {}).get("decision") or "")
except (OSError, ValueError):
    print("")' "$1"
}

say "preflight (key, host, Token Plan, STOP file, price, ledger, notify)"
run_spend preflight --notify-cmd "$NOTIFY" --model "$MODEL"
if ! "$PY" -c 'import prometheus.gym.ladder.record' >/dev/null 2>&1; then
  echo "judge_run: refused: Prometheus is not importable (PYTHONPATH=<deployed tree>/src)" >&2
  exit 2
fi

if [ ! -s "$WORKDIR/judge-queue.jsonl" ]; then
  if [ "$execute" -eq 0 ]; then
    say "dry run: no queue in $WORKDIR yet; --execute would sample 300 pass + 100 fail there,"
    say "dry run: then estimate, run (--budget-usd $budget), the blind fail-side check"
    say "dry run: (--blind-budget-usd $blind_budget) and report"
    say "dry run: nothing was sent"
    exit 0
  fi
  say "sample"
  run_judge --workdir "$WORKDIR" sample
fi

say "projected cost of what is pending"
run_judge --workdir "$WORKDIR" estimate "${paid[@]}" --view "$view"

if [ "$execute" -eq 0 ]; then
  say "dry run: would run judge.py run --provider max-payg --budget-usd $budget --view $view"
  if [ "$blind_fail" -eq 1 ] && [ "$view" = outcome ]; then
    say "dry run: then the blind fail-side check (--view blind --only fail, phase 2-judge-blind,"
    say "dry run: its own --budget-usd $blind_budget)"
  fi
  say "dry run: then report (--allow-missing $allow_missing)"
  if [ "$rejudge" -eq 1 ]; then
    say "dry run: then, if the gate says re-judge, run-all in $REJUDGE_WORKDIR (--budget-usd $rejudge_budget)"
  fi
  say "dry run: nothing was sent; pass --execute to run"
  exit 0
fi

rc=0
run_judge --workdir "$WORKDIR" run "${paid[@]}" --budget-usd "$budget" --view "$view" || rc=$?
if finished "$rc" && [ "$blind_fail" -eq 1 ] && [ "$view" = outcome ]; then
  say "blind fail-side check"
  blind_rc=0
  run_judge --workdir "$WORKDIR" run "${paid[@]}" --budget-usd "$blind_budget" --view blind --only fail \
    || blind_rc=$?
  if ! finished "$blind_rc"; then rc=$blind_rc; fi
fi
say "report"
report_rc=0
run_judge --workdir "$WORKDIR" report --view "$view" --allow-missing "$allow_missing" \
  --markdown-out "$WORKDIR/judge-report.md" || report_rc=$?
if ! finished "$rc"; then
  say "stopped (exit $rc); see $WORKDIR/judge-status.json"
  exit "$rc"
fi
decision=""
if [ "$report_rc" -eq 0 ]; then decision=$(gate_of "$WORKDIR/judge-report.json"); fi
case "$decision" in
  trust-labels|re-judge) say "gate: $decision"; rc=0 ;;
  incomplete)
    say "gate: incomplete (more than $allow_missing item(s) without a verdict); run again, or raise --allow-missing"
    exit 6 ;;
  *)
    say "the report did not decide the gate (exit $report_rc); see $WORKDIR/judge-report.md"
    exit 5 ;;
esac

if [ "$rejudge" -eq 1 ]; then
  say "re-judge (run-all) if the gate asks for it"
  run_judge --workdir "$REJUDGE_WORKDIR" run-all --rejudge-from "$WORKDIR" "${paid[@]}" \
    --budget-usd "$rejudge_budget" || rc=$?
  if [ -s "$REJUDGE_WORKDIR/judge-queue.jsonl" ]; then
    run_judge --workdir "$REJUDGE_WORKDIR" report --allow-missing "$allow_missing" \
      --markdown-out "$REJUDGE_WORKDIR/judge-report.md" || true
  fi
fi
exit "$rc"
