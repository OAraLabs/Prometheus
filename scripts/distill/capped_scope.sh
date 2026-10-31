#!/usr/bin/env bash
# capped_scope.sh NAME LOG -- CMD...
#
# Run CMD under a memory cap on a host whose services must keep their RAM (the 3090 Ti host keeps
# at least 8 GB free for them): a transient systemd user scope distill-p6-NAME-<epoch> with
# MemoryMax (page cache included) and no swap, nice 19, ionice idle, oom_score_adj 1000. The cap is
# MemAvailable - FLOOR_GIB - 1 GiB at the start, at most CAP_MAX_GIB; under 2 GiB it does not start
# (exit 99). A watchdog stops the scope (SIGTERM: quantize.py pauses with exit 75, and the same
# command resumes) if MemAvailable still falls under FLOOR_GIB. A step that needs more than the
# cap is killed inside the scope (only it), and systemd stops the scope (exit 75 again).
# Every 10 s it appends MemAvailable, the scope's memory and the free disk to LOG.monitor; at the
# end the peaks. A restore that stops distill-* units stops this scope too.
#
#   FLOOR_GIB=8 CAP_MAX_GIB=12 capped_scope.sh merge logs/merge.log -- python3 quantize.py merge ...
set -uo pipefail
if [ $# -lt 4 ] || [ "$3" != "--" ]; then
  echo "usage: capped_scope.sh NAME LOG -- CMD..." >&2
  exit 2
fi
NAME=$1; LOG=$2; shift 3
FLOOR_GIB=${FLOOR_GIB:-8}
CAP_MAX_GIB=${CAP_MAX_GIB:-12}
avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
A=$(avail_kb)
CAP_KB=$(( A - (FLOOR_GIB + 1) * 1048576 ))
MAXKB=$(( CAP_MAX_GIB * 1048576 ))
[ "$CAP_KB" -gt "$MAXKB" ] && CAP_KB=$MAXKB
mkdir -p "$(dirname "$LOG")"
MON="$LOG.monitor"
if [ "$CAP_KB" -lt $(( 2 * 1048576 )) ]; then
  echo "$(date -Is) capped_scope: MemAvailable $((A / 1024)) MiB leaves under 2 GiB above the" \
       "${FLOOR_GIB} GiB floor; not starting $NAME" | tee -a "$MON" >&2
  exit 99
fi
UNIT="distill-p6-${NAME}-$(date +%s)"
echo "$(date -Is) unit $UNIT cap $((CAP_KB / 1024)) MiB (MemAvailable $((A / 1024)) MiB," \
     "floor ${FLOOR_GIB} GiB): $*" | tee -a "$MON"
systemd-run --user --scope --quiet --collect --unit="$UNIT" -p MemoryMax="${CAP_KB}K" -p MemorySwapMax=0 \
  -- bash -c 'echo 1000 > /proc/self/oom_score_adj; exec nice -n 19 ionice -c3 "$@"' _ "$@" \
  >>"$LOG" 2>&1 &
PID=$!
CG=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 0.5
  REL=$(systemctl --user show -p ControlGroup --value "$UNIT.scope" 2>/dev/null)
  [ -n "$REL" ] && CG="/sys/fs/cgroup$REL" && break
done
[ -z "$CG" ] && echo "$(date -Is) WARNING: no cgroup for $UNIT.scope; its memory is not sampled" >>"$MON"
PEAK=0; MINAV=$A; MINDISK=$(df --output=avail -k "$HOME" | tail -1); STOPPED=0; N=0
while kill -0 "$PID" 2>/dev/null; do
  AV=$(avail_kb); [ "$AV" -lt "$MINAV" ] && MINAV=$AV
  CUR=0; ANON=0
  if [ -n "$CG" ]; then
    CUR=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
    P2=$(cat "$CG/memory.peak" 2>/dev/null || echo 0)
    [ "$CUR" -gt "$PEAK" ] && PEAK=$CUR
    [ "$P2" -gt "$PEAK" ] && PEAK=$P2
    ANON=$(awk '/^anon / {print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
  fi
  DF=$(df --output=avail -k "$HOME" | tail -1); [ "$DF" -lt "$MINDISK" ] && MINDISK=$DF
  if [ $((N % 10)) -eq 0 ]; then
    echo "$(date -Is) avail $((AV / 1024)) MiB scope $((CUR / 1048576)) MiB anon" \
         "$((ANON / 1048576)) MiB disk_free $((DF / 1048576)) GiB" >>"$MON"
  fi
  if [ "$AV" -lt $(( FLOOR_GIB * 1048576 )) ] && [ $STOPPED -eq 0 ]; then
    echo "$(date -Is) WATCHDOG: MemAvailable $((AV / 1024)) MiB < ${FLOOR_GIB} GiB: stopping $UNIT" \
      | tee -a "$MON" >>"$LOG"
    systemctl --user stop "$UNIT.scope"; STOPPED=1
  fi
  N=$((N + 1)); sleep 1
done
wait "$PID"; RC=$?
echo "$(date -Is) exit $RC; scope peak $((PEAK / 1048576)) MiB (cap $((CAP_KB / 1024)) MiB);" \
     "host MemAvailable min $((MINAV / 1024)) MiB; disk free min $((MINDISK / 1048576)) GiB;" \
     "watchdog_stopped $STOPPED" | tee -a "$MON"
exit $RC
