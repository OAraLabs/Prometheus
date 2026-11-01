#!/usr/bin/env bash
# round2_tower.sh {start|end|status|plan} [options] [--dry-run]
#   (the 4090 tower; installed as ~/distill/window/round2/round2_tower.sh beside the window machinery)
#
# Round 2 (operator 2026-11-01 ~01:20 ET): the 4090 serves the 27B teacher (serve_teacher.sh) for the 3090 Ti
# host's sandboxed gym in every 4090 window: the continuous window's tail (Sun after the post-train pipeline and
# the capacity block, until Mon 05:20 EST) and then every nightly 01:00-07:00 ET window. The 3090 Ti host
# drives it over ssh (scripts/distill/mini_window/round2_chain.py: `start` before its generation, nothing after:
# the end is a timer HERE), so a lost link never leaves the business server down past the window machinery's
# own end, dead-man and watcher. Nothing in this script runs a tool command: this host has no jail (OQ #11).
#
#   start --mode cont|night --profiles "SxC [SxC...]" --end-utc "YYYY-MM-DD HH:MM:SS UTC" [--req ID]
#         [--preempt-prefix distill-cap-]
#       Under the window lock (window_start's gates, from lib.sh): no NO-SWAPS flag, preflight, the hours
#       (night: 01:00-06:00 ET and never while the continuous window is open; cont: only while it is open, up to
#       30 min before its runner stop), disk, the window end and the dead-man armed (ensure_window_end,
#       ensure_deadman), a rehearsal PASS for the installed bundle (night: required, as window_start; cont: only
#       logged, as the post-train pipeline). The GPU is taken the window machinery's way: business healthy ->
#       `setups apply flywheel` (exactly "stop llama-server", checked first) + VRAM freed; business already
#       down (cont only: the capacity block left it down) -> no other distill-* unit and the GPU idle, so no
#       extra down/up flap. --preempt-prefix stops the active distill-* units with that prefix by name first
#       (the round-2 chain's capacity-block timeout). Then the teacher, as the template instance
#       distill-jobs-r2teacher@<stamp>-<n>.service (distill-jobs-* on purpose: window_end, window_watch and
#       continuous_end stop and restore it like a job runner; a restore stops every distill-* unit), one
#       profile after the other (SLOTSxCTX_PER_SLOT; serve_teacher.sh refuses a shape that does not fit) until
#       one verifies (key enforced, slots, no tools/MCP/web UI). Recorded in state/jobs.unit (window_start at
#       01:20 then skips: "a window is already open"), a watcher (window_watch.sh: restores as soon as the
#       teacher ends early) and a one-shot end timer flywheel-r2-end at --end-utc running `end`. Telegram
#       "<BIZ_NAME> going down: 4090 teacher generation until <end>" ("stays down" when it already was).
#       No profile verifies, or anything fails after the GPU was taken: business restored (lib.sh EXIT trap).
#       Result: state/r2-start.json ({"status": ok|refused|failed, "why", "unit", "slots", "ctx", "req", ...}).
#       Exit 0 ok, 2 refused (nothing changed), 1 failed (business restored).
#   end
#       Under the window lock: stops the recorded round-2 teacher (and any other active r2teacher instance) by
#       name, then ONE restore (apply business + tower_verify: llama-server healthy on the tunnel's origin, the
#       connector's LOCAL /ready, VRAM back; the public hostname is never contacted), state/window-<date>.closed,
#       Telegram "<BIZ_NAME> back". A failed restore: NO-SWAPS flag + a loud Telegram; the dead-man retries.
#       A job runner recorded instead (not ours) is left to window_end. Nothing recorded and business healthy:
#       nothing to do.
#   swap --profiles "[MODEL:]SxC ..." [--req ID]
#       Under the window lock, while a round-2 teacher is the recorded window job: stops its watcher and the
#       teacher, starts the first profile that verifies (the teacher A/B's IQ3 arm, then the generation shape),
#       records it, starts a new watcher. Business stays down (no flap). Nothing verifies: business restored.
#   status   read-only key=value lines (the round-2 chain and the heartbeat read them)
#   plan     read-only: serve_teacher.sh plan for each --profiles shape
# A profile is [MODEL:]SLOTSxCTX_PER_SLOT; MODEL is q4 (the pinned UD-Q4_K_XL, the default) or iq3 (the 3090 Ti
# teacher's IQ3_XXS, copied here by the capacity block: the teacher A/B's same-hardware arm A).
# --dry-run prints every action (read-only checks still run) and changes nothing.
#
# Install (from the repo machine; not part of window.conf BUNDLE_FILES, so no new rehearsal is needed):
#   ssh <4090> 'install -d -m 700 ~/distill/window/round2 ~/distill/window/round2/env'
#   install round2_tower.sh as ~/distill/window/round2/round2_tower.sh (0700) and
#   distill-jobs-r2teacher@.service as ~/.config/systemd/user/ (0644); systemctl --user daemon-reload
# Business names appear only through window.conf's BIZ_NAME (Telegram only).
CMD="${1:-}"
[ $# -gt 0 ] && shift
MODE=""
PROFILES=""
END_UTC=""
REQ=""
PREEMPT=""
_args=()
usage() { echo "usage: $0 {start|swap|end|status|plan} [--mode cont|night] [--profiles '[MODEL:]SxC ...'] [--end-utc T] [--req ID] [--preempt-prefix P] [--dry-run]" >&2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 || { usage; exit 64; } ;;
    --profiles) PROFILES="${2:-}"; shift 2 || { usage; exit 64; } ;;
    --end-utc) END_UTC="${2:-}"; shift 2 || { usage; exit 64; } ;;
    --req) REQ="${2:-}"; shift 2 || { usage; exit 64; } ;;
    --preempt-prefix) PREEMPT="${2:-}"; shift 2 || { usage; exit 64; } ;;
    --dry-run|-h|--help) _args+=("$1"); shift ;;
    *) usage; exit 64 ;;
  esac
done
case "$CMD" in start|swap|end|status|plan) ;; *) usage; exit 64 ;; esac
SCRIPT_NAME="round2_$CMD"
# shellcheck source=/dev/null
. "${WIN_DIR:-$HOME/distill/window}/lib.sh"
parse_args ${_args[@]+"${_args[@]}"}

R2_DIR="$WIN_DIR/round2"
R2_ENV_DIR="$R2_DIR/env"
R2_TEMPLATE="distill-jobs-r2teacher@"
R2_END_UNIT=flywheel-r2-end
SERVE="${R2_SERVE:-$HOME/distill/teacher-serve/serve_teacher.sh}"
RESULT="$STATE/r2-start.json"

# every active round-2 teacher instance, one per line
r2_units() {
  systemctl --user list-units --plain --no-legend --no-pager --state=active,activating,deactivating,reloading \
    "${R2_TEMPLATE}*" 2>/dev/null | awk '{print $1}' || true
}
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
json_str() { local s=${1//\\/\/}; s=${s//\"/\'}; printf '%s' "${s//$'\n'/ }"; }

R_UNIT=""; R_SLOTS=""; R_CTX=""; R_BIZ=""; R_MODEL=""
result() {  # $1 ok|refused|failed, $2 why
  log "result: $1: $2"
  [ "$DRY_RUN" = 1 ] && return 0
  local tmp="$STATE/.r2-start.$$"
  printf '{"status": "%s", "why": "%s", "req": "%s", "cmd": "%s", "mode": "%s", "unit": "%s", "model": "%s", "slots": "%s", "ctx": "%s", "business": "%s", "end_utc": "%s", "at": "%s"}\n' \
    "$1" "$(json_str "$2")" "$(json_str "$REQ")" "$CMD" "$MODE" "$R_UNIT" "$R_MODEL" "$R_SLOTS" "$R_CTX" "$R_BIZ" \
    "$(json_str "$END_UTC")" "$(date -u +%FT%TZ)" > "$tmp"
  mv -f "$tmp" "$RESULT"
}
refuse() {  # nothing changed: exit 2 (a dry run logs and goes on)
  if [ "$DRY_RUN" = 1 ]; then log "[dry-run] a real run would refuse here: $1"; return 0; fi
  result refused "$1"
  exit 2
}

# The served model of a profile: q4 = serve_teacher.sh's own pinned defaults; iq3 = the 3090 Ti teacher's
# IQ3_XXS (state.json teacher_model; the capacity block's copy here, sha256-verified against its manifest).
IQ3_FILE="$HOME/distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
IQ3_BYTES=10094357632
IQ3_SHA256=fdfcb6a29b11188956dfbfd904223588a6c1b77eb250c3e8a36e1bd269df91f7
model_env() {  # $1 model -> MENV (KEY=VALUE words for the env file and the plan)
  case "$1" in
    q4) MENV=() ;;
    iq3) MENV=("TEACHER_MODEL=$IQ3_FILE" "TEACHER_MODEL_BYTES=$IQ3_BYTES" "TEACHER_SHA256=$IQ3_SHA256"
               "TEACHER_MANIFEST=$HOME/distill/models/teacher/MANIFEST.json"
               "TEACHER_ALIAS=qwen3.8-27b-gsq-rco-iq3_xxs" "TEACHER_WEIGHTS_MIB=9000") ;;
    *) return 1 ;;
  esac
}
# "[MODEL:]SxC" -> P_MODEL, P_SLOTS and P_CTX; refuses anything serve_teacher.sh would (1..8 slots, ctx a
# multiple of 1024 >= 8192) and an unknown model
parse_profile() {
  local p=$1 s c
  P_MODEL=q4
  case "$p" in *:*) P_MODEL=${p%%:*}; p=${p#*:} ;; esac
  model_env "$P_MODEL" || return 1
  case "$p" in *x*) ;; *) return 1 ;; esac
  s=${p%%x*}; c=${p#*x}
  case "$s$c" in ''|*[!0-9]*) return 1 ;; esac
  [ "$s" -ge 1 ] && [ "$s" -le 8 ] && [ "$c" -ge 8192 ] && [ $(( c % 1024 )) -eq 0 ] || return 1
  P_SLOTS=$s; P_CTX=$c
}
# serve_teacher.sh's own VRAM plan says this shape fits this card (read-only)
profile_fits() {  # $1 slots, $2 ctx, $3 model   (output captured first: grep -q on a pipe would SIGPIPE under pipefail)
  local out
  model_env "${3:-q4}" || return 1
  out=$(env ${MENV[@]+"${MENV[@]}"} TEACHER_SLOTS="$1" TEACHER_CTX_PER_SLOT="$2" "$SERVE" plan 2>/dev/null) || return 1
  printf '%s\n' "$out" | grep -E "^slots $1 x ctx $2: .*: fits$" >/dev/null || return 1
  if [ "${3:-q4}" = iq3 ] && [ "$(file_size "$IQ3_FILE" 2>/dev/null || echo 0)" != "$IQ3_BYTES" ]; then return 1; fi
}
file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

cmd_status() {
  local u st
  u=$(cat "$STATE/jobs.unit" 2>/dev/null || true)
  echo "jobs_unit=${u:-none}"
  if [ -n "$u" ]; then st=$(systemctl --user is-active "$u" 2>/dev/null || true); echo "jobs_unit_state=${st:-unknown}"; fi
  echo "r2_units=$(r2_units | tr '\n' ' ')"
  echo "distill_units=$(distill_units | tr '\n' ' ')"
  echo "business=$(systemctl --user is-active "$LLAMA_UNIT" 2>/dev/null || true)"
  echo "vram_used_mib=$(gpu_used_mib)"
  echo "cont_on=$(cont_on && echo 1 || echo 0)"
  if [ -e "$FLAG" ]; then echo "flag=$(flag_reason)"; else echo "flag=none"; fi
  echo "start_result=$(head -1 "$RESULT" 2>/dev/null || echo none)"
  if [ "$(file_size "$IQ3_FILE" 2>/dev/null || echo 0)" = "$IQ3_BYTES" ]; then echo "iq3_file=ok"; else echo "iq3_file=missing"; fi
  echo "end_last=$(head -1 "$STATE/r2-end.last" 2>/dev/null || echo none)"
  echo "verify_last=$(verify_line)"
  for t in "$R2_END_UNIT" flywheel-window-start "$END_TIMER" "${CONT_END_UNIT:-flywheel-window-cont-end}" \
           "$DEADMAN_UNIT" "${CONT_REARM_UNIT:-flywheel-window-rearm}"; do
    if timer_read "$t.timer"; then echo "timer $t=$TIMER_NEXT"; else echo "timer $t=-"; fi
  done
  echo "failed_units=$(systemctl --user list-units --plain --no-legend --no-pager --state=failed 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"
}

cmd_plan() {
  local p
  for p in $PROFILES; do
    if ! parse_profile "$p"; then echo "profile $p: not [q4:|iq3:]SLOTSxCTX (1..8 slots, ctx a multiple of 1024 >= 8192)"; continue; fi
    if profile_fits "$P_SLOTS" "$P_CTX" "$P_MODEL"; then echo "profile $p: fits"; else echo "profile $p: does NOT fit or its model file is missing"; fi
  done
  TEACHER_SLOTS=${P_SLOTS:-4} TEACHER_CTX_PER_SLOT=${P_CTX:-32768} "$SERVE" plan 2>&1 | sed 's/^/    /' || true
}

# the one-shot end timer (state/jobs.unit names the teacher; `end` stops it and restores)
arm_end() {  # $1 = END_UTC
  local target
  target=$(utc_epoch "$1") || return 1
  clear_unit "$R2_END_UNIT"
  act systemd-run --user --unit="$R2_END_UNIT" --on-calendar="$1" --timer-property=AccuracySec=1s \
    -p KillMode=mixed -p TimeoutStopSec="$UNIT_STOP_TIMEOUT_S" --expand-environment=no \
    --description="flywheel round 2: end the 4090 teacher, restore business" \
    /bin/bash "$R2_DIR/round2_tower.sh" end || return 1
  [ "$DRY_RUN" = 1 ] && return 0
  timer_read "$R2_END_UNIT.timer" && [ "$TIMER_NEXT" -le "$((target + 120))" ]
}

cmd_start() {
  local t_end latest now llama st used left u pid stamp down_word
  case "$MODE" in cont|night) ;; *) usage; exit 64 ;; esac
  [ -n "$PROFILES" ] || { usage; exit 64; }
  [ -n "$END_UTC" ] || { usage; exit 64; }
  take_lock 1800 || refuse "another window script held the lock for 30 min"
  t_end=$(utc_epoch "$END_UTC" 2>/dev/null) || refuse "bad --end-utc '$END_UTC'"
  now=$(date +%s)
  if [ "$MODE" = night ]; then
    if cont_on; then refuse "the continuous window is still open (until $CONT_UNTIL_UTC): no nightly start"; fi
    latest=$(win_end_at)
  else
    if ! cont_on; then refuse "no continuous window is open (CONT_UNTIL_UTC ${CONT_UNTIL_UTC:-unset})"; fi
    latest=$(utc_epoch "$CONT_END_UTC")
  fi
  [ "$t_end" -gt "$((now + 1200))" ] || refuse "--end-utc $END_UTC is less than 20 min away"
  [ "$t_end" -le "$latest" ] || refuse "--end-utc $END_UTC is after the window end ($(when_label "$latest"))"
  # one window at a time (the nightly timer can fire twice when the clocks go back)
  u=$(cat "$STATE/jobs.unit" 2>/dev/null || true)
  if [ -n "$u" ]; then
    st=$(systemctl --user is-active "$u" 2>/dev/null || true)
    case "$st" in
      active|activating|reloading|deactivating)
        case "$u" in
          "$R2_TEMPLATE"*) R_UNIT=$u; result ok "already running ($u)"; return 0 ;;
          *) refuse "a window is already open ($u runs)" ;;
        esac ;;
    esac
    refuse "an earlier window was never closed (state/jobs.unit names $u): run window_end.sh first"
  fi
  if [ -e "$FLAG" ]; then refuse "NO-SWAPS flag is set ($(flag_reason))"; fi
  REHEARSAL_WHY=""
  if ! rehearsal_ok; then
    if [ "$MODE" = night ]; then refuse "$REHEARSAL_WHY"; fi
    log "note: $REHEARSAL_WHY (the continuous window's start does not require it, as the post-train pipeline)"
  fi
  preflight || refuse "preflight failed (see log)"
  window_check || refuse "outside the swap hours"
  disk_check || refuse "less than ${MIN_FREE_DISK_GB} GB free on disk"
  [ -x "$SERVE" ] || refuse "no $SERVE"
  fitting
  [ "${#FIT[@]}" -gt 0 ] || refuse "no profile of '$PROFILES' fits"
  ensure_window_end || refuse "the window end ($(end_label)) is not scheduled and could not be armed"
  ensure_deadman || refuse "the dead-man restore could not be armed"

  if [ -n "$PREEMPT" ]; then
    case "$PREEMPT" in distill-[a-z]*) ;; *) refuse "--preempt-prefix must start with distill-" ;; esac
    while IFS= read -r u; do
      [ -n "$u" ] || continue
      case "$u" in "$PREEMPT"*) log "preempt: $u (the capacity block's time is up)"; act systemctl --user stop "$u" || log "stop $u exited $?" ;; esac
    done < <(distill_units)
  fi
  left=$(distill_units | grep -v "^$R2_TEMPLATE" | tr '\n' ' ' || true)
  [ -z "${left// /}" ] || refuse "the GPU is still owned by: $left"

  llama=$(systemctl --user is-active "$LLAMA_UNIT" 2>/dev/null || true)
  if [ "$llama" = active ]; then
    verify || refuse "business not healthy before the swap: $(verify_line)"
    log "flywheel plan (dry run):"
    plan_check_flywheel || refuse "flywheel plan is not exactly 'stop $LLAMA_UNIT'"
    R_BIZ=stopped; down_word="going down"
  else
    [ "$MODE" = cont ] || refuse "$LLAMA_UNIT is '${llama:-?}' at the nightly start: business must be up first"
    used=$(gpu_used_mib)
    case "$used" in ''|*[!0-9]*) refuse "nvidia-smi did not report the VRAM in use" ;; esac
    [ "$used" -le "$IDLE_VRAM_MAX_MIB" ] || refuse "business is down but the GPU holds $used MiB (another job?)"
    log "business is already down ($LLAMA_UNIT '${llama:-?}', VRAM $used MiB): no extra down/up flap"
    R_BIZ=was-down; down_word="stays down"
  fi

  stamp=$(et +%Y%m%d-%H%M)
  SWAP_PENDING=1
  [ "$DRY_RUN" = 1 ] && SWAP_PENDING=0
  if [ "$R_BIZ" = stopped ]; then
    pid=$(systemctl --user show -p MainPID --value "$LLAMA_UNIT")
    fly_rc=0
    act setups apply flywheel --execute || fly_rc=$?
    if [ "$fly_rc" -ne 0 ] || ! confirm_freed "$pid"; then
      result failed "the flywheel swap did not complete (rc $fly_rc)"
      exit 1   # EXIT trap restores + notifies
    fi
  fi
  start_first "$stamp" || { result failed "no profile verified (tried ${TRIED[*]})"; exit 1; }   # EXIT trap restores
  record_job "$R_UNIT" 1
  SWAP_PENDING=0   # handed over: the end timer, the watcher, the window end and the dead-man own it now
  endnote="restore $(when_label "$t_end")"
  arm_end "$END_UTC" || endnote="the end timer did NOT arm: restore at $(end_label)"
  watch=started
  start_watch "$stamp" || watch="FAILED to start"
  result ok "teacher $R_UNIT $R_MODEL ${R_SLOTS}x$R_CTX verified ($R_BIZ)"
  printf '%s %s: 4090 teacher generation until %s.\nTeacher %s (%s), %s slots x %s verified; %s; watcher %s.\nBackstops: window end %s, dead-man %s.\n' \
    "${BIZ_NAME:-business}" "$down_word" "$(when_label "$t_end")" "$R_UNIT" "$R_MODEL" "$R_SLOTS" "$R_CTX" "$endnote" \
    "$watch" "$(end_label)" "${DEADMAN_AT:-$(deadman_label)}" | notify
}

# The profiles of $PROFILES that serve_teacher.sh's VRAM plan accepts (and whose model file is here): FIT
fitting() {
  local p
  FIT=()
  for p in $PROFILES; do
    if ! parse_profile "$p"; then log "profile '$p' skipped: not [q4:|iq3:]SLOTSxCTX"; continue; fi
    if profile_fits "$P_SLOTS" "$P_CTX" "$P_MODEL"; then FIT+=("$P_MODEL:$P_SLOTS:$P_CTX"); log "profile $p: fits"
    else log "profile $p skipped: serve_teacher.sh's VRAM plan refuses it or its model file is missing"; fi
  done
}

# Start the first of FIT that verifies as ${R2_TEMPLATE}<stamp>-<n>.service; sets R_UNIT R_MODEL R_SLOTS R_CTX.
start_first() {  # $1 stamp
  local p i=0 u used
  TRIED=()
  [ "$DRY_RUN" = 1 ] || install -d -m 700 "$R2_ENV_DIR"
  for p in "${FIT[@]}"; do
    i=$((i + 1))
    IFS=: read -r P_MODEL P_SLOTS P_CTX <<< "$p"
    model_env "$P_MODEL"
    u="${R2_TEMPLATE}$1-$i.service"
    TRIED+=("$P_MODEL:${P_SLOTS}x$P_CTX")
    if [ "$DRY_RUN" = 1 ]; then
      log "[dry-run] would write $R2_ENV_DIR/$1-$i.env (TEACHER_SLOTS=$P_SLOTS TEACHER_CTX_PER_SLOT=$P_CTX ${MENV[*]:-})"
      act systemctl --user start "$u"
      R_UNIT=$u; R_MODEL=$P_MODEL; R_SLOTS=$P_SLOTS; R_CTX=$P_CTX
      return 0
    fi
    { printf 'TEACHER_SLOTS=%s\nTEACHER_CTX_PER_SLOT=%s\n' "$P_SLOTS" "$P_CTX"
      [ "${#MENV[@]}" -eq 0 ] || printf '%s\n' "${MENV[@]}"; } > "$R2_ENV_DIR/$1-$i.env"
    chmod 600 "$R2_ENV_DIR/$1-$i.env"
    log "starting $u ($P_MODEL, $P_SLOTS x $P_CTX; returns once serve_teacher.sh verify passed)"
    if act systemctl --user start "$u"; then
      R_UNIT=$u; R_MODEL=$P_MODEL; R_SLOTS=$P_SLOTS; R_CTX=$P_CTX
      return 0
    fi
    log "$u did not verify: stopping it by name"
    systemctl --user stop "$u" 2>/dev/null || true
    systemctl --user reset-failed "$u" 2>/dev/null || true
    for _ in $(seq 1 18); do
      used=$(gpu_used_mib); [ "${used:-99999}" -le "$IDLE_VRAM_MAX_MIB" ] && break; sleep 5
    done
  done
  return 1
}

# The teacher as the window's job (window_end, window_watch, continuous_end and window_start read it)
record_job() {  # $1 unit, $2 1 = a new window (the started/closed markers)
  local pid
  [ "$DRY_RUN" = 0 ] || return 0
  printf '%s\n' "$1" > "$STATE/jobs.unit"
  pid=$(systemctl --user show -p MainPID --value "$1" 2>/dev/null || echo 0)
  printf '%s %s\n' "$pid" "$1" > "$STATE/jobs.pid"
  if [ "${2:-0}" = 1 ]; then
    rm -f "$STATE/window-$(today).closed"
    printf '%s %s\n' "$(et '+%F %T %Z')" "$1" > "$STATE/window-$(today).started"
  fi
}

# window_watch.sh as a recorded unit (a swap stops it first: it would restore business between the two teachers)
start_watch() {  # $1 stamp
  local w="$WATCH_PREFIX-r2-$1"
  act systemd-run --user --unit="$w" --collect \
    --description="flywheel window: restore as soon as the round-2 teacher ends" \
    -p KillMode=mixed -p TimeoutStopSec="$UNIT_STOP_TIMEOUT_S" --expand-environment=no \
    /bin/bash "$WIN_DIR/window_watch.sh" || return 1
  [ "$DRY_RUN" = 1 ] || printf '%s.service\n' "$w" > "$STATE/r2-watch.unit"
}

cmd_swap() {
  local u st w stamp used
  [ -n "$PROFILES" ] || { usage; exit 64; }
  take_lock 1800 || refuse "another window script held the lock for 30 min"
  u=$(cat "$STATE/jobs.unit" 2>/dev/null || true)
  case "$u" in "$R2_TEMPLATE"*) ;; *) refuse "no round-2 teacher is the window's job (state/jobs.unit: ${u:-none})" ;; esac
  st=$(systemctl --user is-active "$u" 2>/dev/null || true)
  [ "$st" = active ] || refuse "$u is '${st:-?}', not active"
  if [ -e "$FLAG" ]; then refuse "NO-SWAPS flag is set ($(flag_reason))"; fi
  if cont_on; then MODE=cont; else MODE=night; fi
  fitting
  [ "${#FIT[@]}" -gt 0 ] || refuse "no profile of '$PROFILES' fits"
  R_BIZ=was-down
  stamp=$(et +%Y%m%d-%H%M%S)
  w=$(cat "$STATE/r2-watch.unit" 2>/dev/null || true)
  case "$w" in
    "$WATCH_PREFIX"-r2-*.service) log "stopping the watcher $w (it would restore business between the teachers)"
                                   act systemctl --user stop "$w" || log "stop $w exited $?" ;;
    *) log "no round-2 watcher recorded" ;;
  esac
  SWAP_PENDING=1   # from here a failure restores business (lib.sh EXIT trap)
  [ "$DRY_RUN" = 1 ] && SWAP_PENDING=0
  act systemctl --user stop "$u" || log "stop $u exited $?"
  [ "$DRY_RUN" = 1 ] || systemctl --user reset-failed "$u" 2>/dev/null || true
  for _ in $(seq 1 18); do
    [ "$DRY_RUN" = 1 ] && break
    used=$(gpu_used_mib); [ "${used:-99999}" -le "$IDLE_VRAM_MAX_MIB" ] && break; sleep 5
  done
  start_first "$stamp" || { result failed "swap: no profile verified (tried ${TRIED[*]})"; exit 1; }
  record_job "$R_UNIT" 0
  SWAP_PENDING=0
  start_watch "$stamp" || log "the new watcher did not start: the window end restores"
  result ok "swapped $u -> $R_UNIT $R_MODEL ${R_SLOTS}x$R_CTX"
}

cmd_end() {
  local u st ran="" others closed rc=0
  take_lock 1800 || log "lock still held after 1800 s: restoring anyway"
  closed="$STATE/window-$(today).closed"
  u=$(cat "$STATE/jobs.unit" 2>/dev/null || true)
  case "$u" in
    '') ;;
    "$R2_TEMPLATE"*) ;;
    *) log "the recorded unit $u is not a round-2 teacher: window_end owns it"; exit 0 ;;
  esac
  others=$(r2_units | grep -vxF "${u:-none}" || true)
  if [ -z "$u" ] && [ -z "$others" ]; then
    if verify; then log "no round-2 teacher and business verifies healthy: nothing to do"; exit 0; fi
    log "no round-2 teacher recorded, but business is not healthy: restoring"
    ran="no teacher was recorded"
  fi
  if [ -n "$u" ]; then
    st=$(systemctl --user is-active "$u" 2>/dev/null || true)
    case "$st" in
      active|activating|reloading|deactivating)
        act systemctl --user stop "$u" || log "stop $u exited $?"
        ran="$u stopped at $(et +%H:%M)" ;;
      *) ran="$u had ended (${st:-gone})" ;;
    esac
    if [ "$DRY_RUN" = 0 ]; then
      systemctl --user reset-failed "$u" 2>/dev/null || true
      mv -f "$STATE/jobs.unit" "$STATE/jobs.unit.done" 2>/dev/null || true
      mv -f "$STATE/jobs.pid" "$STATE/jobs.pid.done" 2>/dev/null || true
    fi
  fi
  while IFS= read -r o; do
    [ -n "$o" ] || continue
    log "another round-2 teacher is running: $o: stopping it"
    act systemctl --user stop "$o" || log "stop $o exited $?"
    ran="${ran:+$ran; }$o stopped"
  done <<< "$others"
  SWAP_PENDING=1
  [ "$DRY_RUN" = 1 ] && SWAP_PENDING=0
  restore || rc=$?
  SWAP_PENDING=0
  if [ "$rc" -eq 0 ]; then
    if [ "$DRY_RUN" = 0 ]; then
      printf 'OK %s %s\n' "$(et '+%F %T %Z')" "$RESTORE_LINE" > "$closed"
      printf 'OK %s %s; %s\n' "$(et '+%F %T %Z')" "$ran" "$RESTORE_LINE" > "$STATE/r2-end.last"
    fi
    printf '%s back: 4090 teacher generation ended %s; business RESTORED + verified.\n%s\n%s\n' \
      "${BIZ_NAME:-business}" "$(et '+%a %H:%M %Z')" "$ran" "$RESTORE_LINE" | notify
  else
    if [ "$DRY_RUN" = 0 ]; then
      printf 'FAIL %s %s\n' "$(et '+%F %T %Z')" "$RESTORE_LINE" > "$closed"
      printf 'FAIL %s %s; %s\n' "$(et '+%F %T %Z')" "$ran" "$RESTORE_LINE" > "$STATE/r2-end.last"
    fi
    write_flag "round2_end: restore failed"
    printf '4090 RESTORE FAILED after teacher generation: %s may be DOWN.\n%s\n%s\nNO-SWAPS flag set. Dead-man %s retries apply business.\n' \
      "${BIZ_NAME:-business}" "$RESTORE_LINE" "$ran" "$(deadman_label)" | notify
    exit 1
  fi
}

case "$CMD" in
  status) cmd_status ;;
  plan) cmd_plan ;;
  start) start_log; cmd_start ;;
  swap) start_log; cmd_swap ;;
  end) start_log; cmd_end ;;
esac
