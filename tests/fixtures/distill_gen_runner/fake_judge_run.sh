#!/bin/bash
# Stand-in for scripts/distill/judge_run.sh: logs its arguments and what its environment names, works
# FAKE_JUDGE_S seconds, exits FAKE_JUDGE_RC (default 0).
printf '%s | PY=%s PYTHONPATH=%s NOTIFY=%s MODEL=%s\n' "$*" "${PY:-}" "${PYTHONPATH:-}" "${NOTIFY:-}" "${MODEL:-}" \
  >> "$FAKE_STATE/judge-calls.log"
echo $$ > "$FAKE_STATE/judge-pid"
sleep "${FAKE_JUDGE_S:-0.3}"
exit "${FAKE_JUDGE_RC:-0}"
