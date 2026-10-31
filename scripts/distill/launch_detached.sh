#!/usr/bin/env bash
# Launch train_qlora.py detached, as a transient systemd user unit.
#
#   scripts/distill/launch_detached.sh [--execute] [--resume] [--python PATH] [--ts STAMP] \
#       -- --data ~/distill/phase3/train.jsonl --tokenizer-dir ~/distill/models/qwen3.5-9b-tokenizer \
#          --out data/distill/runs/9b-r32 [--stop-at 2026-11-01T06:45-05:00] [trainer flags...]
#
# Print-only by default: it prints the systemd-run command and exits 0. Only
# --execute runs it. The unit is distill-train-<UTC stamp>, so a
# `setups.py apply business` restore stops it like every distill-* job: the
# trainer gets SIGTERM, abandons the accumulation group in progress and
# checkpoints the last completed step (TimeoutStopSec leaves it the time).
# A CUDA OOM re-executes the trainer inside the same unit at the next shorter
# sequence length. Logs go to data/distill/logs/<unit>/: train.log and
# metrics.jsonl from the trainer, unit.log for the unit's stdout/stderr.
# To continue a stopped run, launch again with --resume and the same trainer
# flags (same --out): it starts from the newest complete checkpoint.
#
# The trainer itself refuses Max-derived data without the operator's approval
# flag (exit 3, written to unit.log): see DISTILL-OPEN-QUESTIONS #6.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${DISTILL_PYTHON:-python3}"
execute=0
resume=0
ts=""
train_args=()

usage() {
  sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (($#)); do
  case "$1" in
    --execute) execute=1 ;;
    --resume) resume=1 ;;
    --python) python="$2"; shift ;;
    --ts) ts="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; train_args=("$@"); break ;;
    *) echo "unknown option: $1 (trainer flags go after --)" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ((${#train_args[@]} == 0)); then
  echo "no trainer flags: pass them after --" >&2
  exit 2
fi
case " ${train_args[*]} " in
  *" --out "*|*" --out="*) ;;
  *) echo "the trainer needs --out (the run directory)" >&2; exit 2 ;;
esac
case " ${train_args[*]} " in
  *" --dry-run "*) echo "--dry-run needs no unit: run train_qlora.py directly" >&2; exit 2 ;;
esac

ts="${ts:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$ts" =~ ^[0-9A-Za-z]+$ ]]; then
  echo "--ts must be alphanumeric" >&2
  exit 2
fi
unit="distill-train-${ts}"
log_dir="${repo}/data/distill/logs/${unit}"

trainer=("$python" "${repo}/scripts/distill/train_qlora.py" --log-dir "$log_dir" "${train_args[@]}")
if ((resume)); then
  trainer+=(--resume)
fi

cmd=(systemd-run --user "--unit=${unit}" --collect "--working-directory=${repo}"
  -p KillSignal=SIGTERM -p TimeoutStopSec=110 -p Nice=5
  -p "StandardOutput=append:${log_dir}/unit.log" -p "StandardError=append:${log_dir}/unit.log"
  --setenv=PYTHONUNBUFFERED=1
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "${trainer[@]}")

printf '%q ' "${cmd[@]}"
printf '\n'
echo "# unit: ${unit}   logs: ${log_dir}/"
echo "# follow: tail -f ${log_dir}/train.log   stop (checkpoints first): systemctl --user stop ${unit}"

if ((!execute)); then
  echo "# print-only: nothing was started (pass --execute to launch)"
  exit 0
fi
if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run is not available on this host" >&2
  exit 2
fi
mkdir -p "$log_dir"
"${cmd[@]}"
