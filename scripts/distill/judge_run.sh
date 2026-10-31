#!/usr/bin/env bash
# Phase 2 judging on Qwen Max: the PAY-PER-TOKEN key (QWEN_PAYG_KEY) when it is present, else the Token
# Plan key (QWEN_API_KEY; the operator's informed override of 2026-10-31).
#
#   judge_run.sh [--execute] [--provider auto|payg|token-plan] [--trial] [--limit N]
#                [--rejudge] [--view outcome|blind] [--no-blind-fail]
#                [--budget-usd 20] [--blind-budget-usd 5] [--rejudge-budget-usd 40]
#                [--allow-missing 0] [--concurrency 2|8]
#
# --provider auto (default): `spend.py which-key` picks payg when QWEN_PAYG_KEY is present (it always
#   wins), else token-plan when QWEN_API_KEY is. token-plan: $0 per call, no --budget-usd (the budget
#   flags are ignored), concurrency 8 workers by default (the client's adaptive limit starts at 4); its
#   preflight is `spend.py tokenplan-preflight` (exit 3: QWEN_PAYG_KEY is present; 4: the quota pause).
# --trial (with --execute): the 50-call trial, the first Phase 2 judge calls: sample if needed, judge 50
#   items (outcome view), no blind check, no re-judge, then `spend.py tokenplan-status` (trial clean?).
#   It is a report, not a gate. Exit 0 when the 50 ran (items remaining is expected).
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
concurrency=""
provider=auto
trial=0
limit=""
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) execute=1 ;;
    --provider) provider=${2:?--provider needs a value}; shift ;;
    --trial) trial=1 ;;
    --limit) limit=${2:?--limit needs a value}; shift ;;
    --rejudge) rejudge=1 ;;
    --no-blind-fail) blind_fail=0 ;;
    --view) view=${2:?--view needs a value}; shift ;;
    --budget-usd) budget=${2:?--budget-usd needs a value}; shift ;;
    --blind-budget-usd) blind_budget=${2:?--blind-budget-usd needs a value}; shift ;;
    --rejudge-budget-usd) rejudge_budget=${2:?--rejudge-budget-usd needs a value}; shift ;;
    --allow-missing) allow_missing=${2:?--allow-missing needs a value}; shift ;;
    --concurrency) concurrency=${2:?--concurrency needs a value}; shift ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "judge_run: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$allow_missing" in
  ''|*[!0-9]*) echo "judge_run: --allow-missing needs a whole number" >&2; exit 2 ;;
esac
case "$provider" in
  auto|payg|token-plan) ;;
  *) echo "judge_run: --provider must be auto, payg or token-plan" >&2; exit 2 ;;
esac
if [ "$trial" -eq 1 ]; then
  limit=${limit:-50}
  blind_fail=0
  rejudge=0
  view=outcome
fi
case "$limit" in
  ''|*[!0-9]*) [ -z "$limit" ] || { echo "judge_run: --limit needs a whole number" >&2; exit 2; } ;;
esac

say() { printf 'judge_run: %s\n' "$*"; }
run_spend() { "$PY" "$here/spend.py" "$@"; }
run_judge() { "$PY" "$here/judge.py" "$@"; }

if [ "$provider" = auto ]; then
  # presence only (QWEN_PAYG_KEY wins); no value is read into this script
  provider=$(run_spend which-key) || { echo "judge_run: refused: no QWEN_PAYG_KEY or QWEN_API_KEY" >&2; exit 2; }
fi
lim=()
if [ -n "$limit" ]; then lim=(--limit "$limit"); fi
if [ "$provider" = token-plan ]; then
  concurrency=${concurrency:-8}
  paid=(--provider token-plan --notify-cmd "$NOTIFY" --judge-model "$MODEL" --concurrency "$concurrency")
else
  concurrency=${concurrency:-2}
  paid=(--provider max-payg --notify-cmd "$NOTIFY" --judge-model "$MODEL" --concurrency "$concurrency")
fi
# The --budget-usd flag (pay-per-token only) in the array _b: empty on the Token Plan.
_b=()
budget_args() { _b=(); if [ "$provider" != token-plan ]; then _b=(--budget-usd "$1"); fi; }
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

if [ "$provider" = token-plan ]; then
  say "route: Token Plan (QWEN_API_KEY; operator override 2026-10-31): \$0 per call, adaptive concurrency"
  say "preflight (key, Token Plan host, QWEN_PAYG_KEY absent, STOP file, quota pause, key fingerprint, notify)"
  pre=0
  run_spend tokenplan-preflight --notify-cmd "$NOTIFY" --model "$MODEL" || pre=$?
  case "$pre" in
    0) ;;
    3) echo "judge_run: refused: QWEN_PAYG_KEY is present: run with --provider auto (it uses that key)" >&2
       exit 3 ;;
    4) say "the Token Plan quota pause stands: nothing was sent"; exit 4 ;;
    *) exit 2 ;;
  esac
else
  say "route: pay-per-token (QWEN_PAYG_KEY)"
  say "preflight (key, host, Token Plan, STOP file, price, ledger, notify)"
  run_spend preflight --notify-cmd "$NOTIFY" --model "$MODEL"
fi
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
  if [ "$trial" -eq 1 ]; then
    say "dry run: the trial would judge $limit items (${paid[1]}), then print spend.py tokenplan-status"
    say "dry run: nothing was sent; pass --execute to run"
    exit 0
  fi
  say "dry run: would run judge.py run ${paid[1]} --view $view${limit:+ --limit $limit}" \
    "$([ "$provider" = token-plan ] || printf -- '--budget-usd %s' "$budget")"
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
budget_args "$budget"
run_judge --workdir "$WORKDIR" run "${paid[@]}" ${_b[@]+"${_b[@]}"} --view "$view" ${lim[@]+"${lim[@]}"} || rc=$?
if [ "$trial" -eq 1 ]; then
  say "trial: judge.py exit $rc ($([ "$rc" -eq 6 ] && echo 'items remain, as expected' || echo 'see judge-status.json'))"
  if [ "$provider" = token-plan ]; then run_spend tokenplan-status --notify-cmd "$NOTIFY" || true; fi
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 6 ]; then exit 0; fi
  exit "$rc"
fi
if finished "$rc" && [ "$blind_fail" -eq 1 ] && [ "$view" = outcome ]; then
  say "blind fail-side check"
  blind_rc=0
  budget_args "$blind_budget"
  run_judge --workdir "$WORKDIR" run "${paid[@]}" ${_b[@]+"${_b[@]}"} --view blind --only fail \
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
  budget_args "$rejudge_budget"
  run_judge --workdir "$REJUDGE_WORKDIR" run-all --rejudge-from "$WORKDIR" "${paid[@]}" \
    ${_b[@]+"${_b[@]}"} || rc=$?
  if [ -s "$REJUDGE_WORKDIR/judge-queue.jsonl" ]; then
    run_judge --workdir "$REJUDGE_WORKDIR" report --allow-missing "$allow_missing" \
      --markdown-out "$REJUDGE_WORKDIR/judge-report.md" || true
  fi
fi
exit "$rc"
