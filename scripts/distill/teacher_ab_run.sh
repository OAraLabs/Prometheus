#!/usr/bin/env bash
# Teacher quant A/B on one GPU host: one llama-server at a time on 127.0.0.1:18080.
#
#   teacher_ab_run.sh --tasks TASKSET.jsonl [--out DIR] [--until HH:MM|ISO] [--execute]
#       [--iq3 GGUF] [--q4 GGUF] [--server LLAMA_SERVER] [--port 18080] [--ngl 99]
#       [--py PYTHON] [--code DIR] [--repo DIR] [--n 30] [--seed teacher-ab-v1] [--pack PACK.json]
#       [--iq3-sha256 HEX] [--q4-sha256 HEX] [--min-free-mib-iq3 13000] [--min-free-mib-q4 21000]
#       [--load-timeout 900] [--min-phase-seconds 600] [--max-hours 8] [--chat-template-file FILE]
#       [--step-retries 2] [--retry-pause 30] [--notify-cmd CMD]
#   teacher_ab_run.sh --stop [--out DIR]      # stop a server this script recorded (by its PID)
#
# The plan (teacher_ab.py does the work; every step resumes where a previous run stopped):
#   1. select: 30 tasks from the real-pattern taskset, stratified and seeded; the request
#      settings both arms use, sampling included, are fixed here
#   2. start the IQ3 server -> arm A (no judging) -> stop it
#   3. start the Q4_K_M server -> arm B (no judging) -> judge all traces of BOTH arms on it
#      (one judge, the bigger quant) -> stop it
#   4. report: ab_report.json + ab_summary.txt (5 lines; piped to --notify-cmd when given)
# A step whose output is complete is skipped (a server is only started when its arm or the
# judge still has work). Both servers get the same pinned flags: 2 slots with 16,384 tokens
# of context EACH (-c 32768 --parallel 2 --no-kv-unified: a unified cache would let the two
# requests in flight run each other out of room, and llama-server then fails both), no
# vision projector and the same explicit sampling; the same sampling goes on every request
# too (a GGUF's own metadata can change a server's defaults: the Q4 file carries no min_p).
# Each arm records its server's /props (model file, per-slot context, chat template, sampler
# defaults) and teacher_ab.py refuses a server that differs from the other arm's. Before any
# server starts, `teacher_ab.py models` checks the two files: different files, Q4 the bigger
# one, the same embedded chat template (--chat-template-file FILE pins one for both). A
# force_compaction task pads with --repo's own docs (default --code): a --repo without
# docs/, wiki/ or README.md is refused.
#
# DRY RUN by default: checks the files and prints every command; starts nothing, stops
# nothing, sends nothing. --execute runs it and needs --until (a hard end). --until is
# resolved ONCE at start to an absolute time (an HH:MM is its next occurrence) and that
# time is passed to every step: an HH:MM read again after it has passed would mean
# tomorrow. An --until already past, or more than --max-hours ahead, is refused. Nothing
# starts after it, and no server starts with less than --min-phase-seconds left. A step
# that leaves work for a reason other than the time box (transport errors) is rerun up to
# --step-retries times. A server is started only by this script, its PID and start time are
# recorded in <out>/server.pid before anything else can run, and it is stopped only by that
# recorded PID after checking the process is still that binary on that model, or this run's
# own child (SIGTERM, SIGKILL after 60 s). Nothing is ever stopped by name. On exit, error or
# a signal the running step and the recorded server are stopped. Before a start it refuses a
# busy port and, when nvidia-smi is there, a GPU with less free memory than the model needs
# (the window must already have freed it; this never stops another server).
#
# Exit: 0 a decision was made, 1 not finished (time box) or no decision, 2 usage or input
# error, 3 a server would not start or answer, or a step failed.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${TEACHER_AB_CODE:-$(cd "$here/../.." && pwd)}"
PY="${TEACHER_AB_PY:-python3}"
OUT="$HOME/distill/teacher-ab"
TASKS=""
UNTIL=""
EXECUTE=0
STOP_ONLY=0
SERVER="$HOME/llama.cpp/build/bin/llama-server"
IQ3="$HOME/distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
Q4="$HOME/models/Qwen3.8-27B-regular-Q4_K_M-bartowski.gguf"
IQ3_SHA=""
Q4_SHA=""
PORT=18080
NGL=99
SLOTS=2
CTX_SLOT=16384   # tokens per slot; the server gets CTX_SLOT * SLOTS with no unified cache
N=30
SEED="teacher-ab-v1"
PACK=""
REPO=""
TEMPLATE=""
MIN_FREE_IQ3=13000
MIN_FREE_Q4=21000
LOAD_TIMEOUT=900
MIN_PHASE_SECONDS=600
MAX_HOURS=8
STEP_RETRIES=2
RETRY_PAUSE=30
NOTIFY=""
# Explicit sampling: the server flags and teacher_ab.py's request fields are the same values.
TEMP=0.7
TOP_K=20
TOP_P=0.8
MIN_P=0.0

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --tasks) TASKS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --until) UNTIL="$2"; shift 2 ;;
        --execute) EXECUTE=1; shift ;;
        --stop) STOP_ONLY=1; shift ;;
        --server) SERVER="$2"; shift 2 ;;
        --iq3) IQ3="$2"; shift 2 ;;
        --q4) Q4="$2"; shift 2 ;;
        --iq3-sha256) IQ3_SHA="$2"; shift 2 ;;
        --q4-sha256) Q4_SHA="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ngl) NGL="$2"; shift 2 ;;
        --py) PY="$2"; shift 2 ;;
        --code) CODE="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --pack) PACK="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --chat-template-file) TEMPLATE="$2"; shift 2 ;;
        --min-free-mib-iq3) MIN_FREE_IQ3="$2"; shift 2 ;;
        --min-free-mib-q4) MIN_FREE_Q4="$2"; shift 2 ;;
        --load-timeout) LOAD_TIMEOUT="$2"; shift 2 ;;
        --min-phase-seconds) MIN_PHASE_SECONDS="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --step-retries) STEP_RETRIES="$2"; shift 2 ;;
        --retry-pause) RETRY_PAUSE="$2"; shift 2 ;;
        --notify-cmd) NOTIFY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
# Absolute paths: the recorded server is matched against them before it is ever signalled.
OUT="$(abspath "$OUT")"; SERVER="$(abspath "$SERVER")"; IQ3="$(abspath "$IQ3")"; Q4="$(abspath "$Q4")"
CODE="$(abspath "$CODE")"
[ -z "$TASKS" ] || TASKS="$(abspath "$TASKS")"
[ -z "$PACK" ] || PACK="$(abspath "$PACK")"
[ -z "$TEMPLATE" ] || TEMPLATE="$(abspath "$TEMPLATE")"
URL="http://127.0.0.1:$PORT"
REPO="$(abspath "${REPO:-$CODE}")"
PIDFILE="$OUT/server.pid"
LOCKDIR="$OUT/run.lock.d"
STEPFILE="$OUT/step.json"
AB="$CODE/scripts/distill/teacher_ab.py"

log() { local line; line="$(date '+%Y-%m-%dT%H:%M:%S%z') $*"; echo "$line" >&2;
        if [ "$EXECUTE" = 1 ] && [ -d "$OUT" ]; then echo "$line" >> "$OUT/run.log"; fi; }
die() { local rc="$1"; shift; log "error: $*"; exit "$rc"; }
show() { printf '  $'; printf ' %q' "$@"; printf '\n'; }

# The pinned server command for one model (one array, used for printing and for running).
server_cmd() {  # $1 model path, $2 alias
    SERVER_ARGV=("$SERVER" -m "$1" --alias "$2" --host 127.0.0.1 --port "$PORT"
        -c "$((CTX_SLOT * SLOTS))" --parallel "$SLOTS" --no-kv-unified -ngl "$NGL" --flash-attn on --jinja
        --no-mmproj --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P")
    [ -z "$TEMPLATE" ] || SERVER_ARGV+=(--chat-template-file "$TEMPLATE")
}

ab() {  # teacher_ab.py with the code tree on PYTHONPATH
    PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$AB" "$@"
}

AB_PID=""
ab_step() {  # a long teacher_ab.py step in the background, so a signal reaches our trap at once.
    # exec: $! is the Python process itself (not a subshell around it), so cleanup's SIGTERM
    # ends the step. Its JSON result goes to $STEPFILE (and run.log).
    ( exec env PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$AB" "$@" > "$STEPFILE" ) &
    AB_PID=$!
    local rc=0
    wait "$AB_PID" || rc=$?
    AB_PID=""
    cat "$STEPFILE" >> "$OUT/run.log" 2>/dev/null || true
    return "$rc"
}

STEP_STOP=""
run_step() {  # $1 label, then teacher_ab.py args: a step left with work for any reason but the
    # time box (transport errors drop a task from that run's queue) is rerun, at most STEP_RETRIES times
    local label="$1" rc attempt=0
    shift
    while :; do
        rc=0
        : > "$STEPFILE"
        ab_step "$@" || rc=$?
        STEP_STOP="$(sed -n 's/.*"stop": *"\([a-z_]*\)".*/\1/p' "$STEPFILE" | tail -1)"
        if [ "$rc" = 1 ] && [ "$STEP_STOP" != until ] && [ "$attempt" -lt "$STEP_RETRIES" ]; then
            attempt=$((attempt + 1))
            log "$label left work (${STEP_STOP:-no result}); rerunning it ($attempt of $STEP_RETRIES) in ${RETRY_PAUSE}s"
            sleep "$RETRY_PAUSE"
            continue
        fi
        return "$rc"
    done
}

step_failed() {  # $1 label, $2 rc: exit as the step's outcome says
    if [ "$2" = 1 ]; then
        if [ "$STEP_STOP" = until ]; then log "$1 not finished (time box)"; exit 1; fi
        log "$1 not finished: work left after $STEP_RETRIES reruns (${STEP_STOP:-no result}; see $OUT/run.log)"
        exit 1
    fi
    die 3 "$1 failed ($2)"
}

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

proc_cmdline() {  # the command line of a PID, or nothing
    if [ -r "/proc/$1/cmdline" ]; then tr '\0' ' ' < "/proc/$1/cmdline"; else ps -p "$1" -o command= 2>/dev/null || true; fi
}
proc_start() { ps -p "$1" -o lstart= 2>/dev/null | sed 's/  */ /g;s/^ //;s/ $//' || true; }
proc_ppid() { ps -p "$1" -o ppid= 2>/dev/null | tr -dc '0-9' || true; }

stop_server() {  # stops ONLY the server recorded in the pidfile, after checking it is ours
    [ -f "$PIDFILE" ] || return 0
    local pid start model now cmd i ours=0
    pid="$(sed -n 1p "$PIDFILE")"; start="$(sed -n 2p "$PIDFILE")"; model="$(sed -n 3p "$PIDFILE")"
    case "$pid" in ''|*[!0-9]*) log "pidfile $PIDFILE holds no PID; removing it"; rm -f "$PIDFILE"; return 0 ;; esac
    if ! kill -0 "$pid" 2>/dev/null; then log "server $pid already gone"; rm -f "$PIDFILE"; return 0; fi
    now="$(proc_start "$pid")"; cmd="$(proc_cmdline "$pid")"
    if [ "$(proc_ppid "$pid")" = "$$" ] && { [ -z "$start" ] || [ "$now" = "$start" ]; }; then
        ours=1  # this run's own child (it may not have exec'd the server binary yet)
    elif [ -n "$start" ] && [ "$now" = "$start" ] && [[ "$cmd" == *"$SERVER"* ]] && [[ "$cmd" == *"$model"* ]]; then
        ours=1
    fi
    if [ "$ours" != 1 ]; then
        log "PID $pid is no longer the server this script started (start '$now' vs '$start'); leaving it"
        rm -f "$PIDFILE"; return 1
    fi
    start="${start:-$now}"
    log "stopping server $pid ($(basename "$model"))"
    kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null && [ "$(proc_start "$pid")" = "$start" ]; then
        log "server $pid ignored SIGTERM for 60 s; SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PIDFILE"
}

free_mib() {  # free memory of GPU 0 in MiB, or nothing when nvidia-smi is not there
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'
}

IN_SPAWN=0
SIGNALLED=0
start_server() {  # $1 model, $2 alias, $3 min free MiB, $4 log name
    local model="$1" alias="$2" need="$3" name="$4" pid start free i code
    [ ! -f "$PIDFILE" ] || die 3 "a server is already recorded in $PIDFILE (run --stop first)"
    port_busy && die 3 "127.0.0.1:$PORT is already in use; refusing to start a second server"
    free="$(free_mib)"
    if [ -n "$free" ] && [ "$need" -gt 0 ] && [ "$free" -lt "$need" ]; then
        die 3 "GPU has $free MiB free, $(basename "$model") needs ~$need MiB (is the window's GPU free?)"
    fi
    server_cmd "$model" "$alias"
    log "starting $(basename "$model") on $URL"
    # A signal between the start and the pidfile write would leave a server nobody stops:
    # on_signal only notes it here, and acts once the PID is recorded.
    IN_SPAWN=1
    "${SERVER_ARGV[@]}" >> "$OUT/server-$name.log" 2>&1 < /dev/null &
    pid=$!
    start="$(proc_start "$pid")"
    printf '%s\n%s\n%s\n' "$pid" "$start" "$model" > "$PIDFILE"
    IN_SPAWN=0
    if [ "$SIGNALLED" = 1 ]; then log "signal: stopping"; exit 143; fi
    for i in $(seq 1 "$LOAD_TIMEOUT"); do
        sleep 1
        kill -0 "$pid" 2>/dev/null || die 3 "the server exited while loading (see $OUT/server-$name.log)"
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL/health" 2>/dev/null || true)"
        [ "$code" = 200 ] && { log "server $pid ready after ${i}s"; return 0; }
    done
    die 3 "the server did not become healthy in ${LOAD_TIMEOUT}s"
}

check_sha() {  # $1 file, $2 expected sha256 (empty: skip)
    [ -n "$2" ] || return 0
    local got
    if command -v sha256sum >/dev/null 2>&1; then got="$(sha256sum "$1" | cut -d' ' -f1)"
    else got="$(shasum -a 256 "$1" | cut -d' ' -f1)"; fi
    [ "$got" = "$2" ] || die 2 "$1: sha256 $got, expected $2"
}

release_lock() { rm -rf "$LOCKDIR"; }

resolve_until() {  # once, at start: --until -> an absolute ISO time, passed to every step after
    local rc=0 iso
    iso="$(ab time-left --until "$UNTIL" --max-seconds "$((MAX_HOURS * 3600))" --iso)" || rc=$?
    case "$rc" in
        0) log "--until $UNTIL is $iso"; UNTIL="$iso" ;;
        1) log "--until $UNTIL ($iso) has already passed: nothing to do"; exit 1 ;;
        *) die 2 "--until $UNTIL: not HH:MM or an ISO time, or more than ${MAX_HOURS} h ahead (--max-hours)" ;;
    esac
}

enough_time() {  # before a server start: MIN_PHASE_SECONDS are left before the resolved --until
    local rc=0
    ab time-left --until "$UNTIL" --min-seconds "$MIN_PHASE_SECONDS" > /dev/null || rc=$?
    case "$rc" in
        0) return 0 ;;
        1) log "less than ${MIN_PHASE_SECONDS}s left before $UNTIL: not starting $1"; exit 1 ;;
        *) die 2 "--until $UNTIL is not HH:MM or an ISO time" ;;
    esac
}
cleanup() {  # on any exit or signal: end a running step, stop the recorded server, drop the lock
    local rc=$?
    trap - EXIT INT TERM
    if [ -n "$AB_PID" ] && kill -0 "$AB_PID" 2>/dev/null; then
        kill -TERM "$AB_PID" 2>/dev/null || true
        wait "$AB_PID" 2>/dev/null || true
    fi
    stop_server || true
    release_lock
    exit "$rc"
}
on_signal() {
    if [ "$IN_SPAWN" = 1 ]; then SIGNALLED=1; return 0; fi
    log "signal: stopping"; exit 143
}

# --- --stop: stop a recorded server and leave ---------------------------------
if [ "$STOP_ONLY" = 1 ]; then
    if [ "$EXECUTE" != 1 ]; then
        if [ -f "$PIDFILE" ]; then echo "would stop PID $(sed -n 1p "$PIDFILE") recorded in $PIDFILE (dry run; add --execute)"
        else echo "no server recorded in $PIDFILE"; fi
        exit 0
    fi
    stop_server; exit $?
fi

# --- checks (both modes) ------------------------------------------------------
[ -n "$TASKS" ] || { echo "error: --tasks is required" >&2; usage >&2; exit 2; }
problems=()
[ -r "$TASKS" ] || problems+=("taskset not readable: $TASKS")
[ -f "$AB" ] || problems+=("teacher_ab.py not found under --code: $AB")
[ -x "$SERVER" ] || problems+=("llama-server not executable: $SERVER")
[ -r "$IQ3" ] || problems+=("IQ3 model not readable: $IQ3")
[ -r "$Q4" ] || problems+=("Q4_K_M model not readable: $Q4")
[ -z "$PACK" ] || [ -r "$PACK" ] || problems+=("pack not readable: $PACK")
[ -z "$TEMPLATE" ] || [ -r "$TEMPLATE" ] || problems+=("chat template not readable: $TEMPLATE")
[ -d "$REPO/docs" ] || [ -d "$REPO/wiki" ] || [ -f "$REPO/README.md" ] \
    || problems+=("--repo $REPO has no docs/, wiki/ or README.md: force_compaction tasks pad with real repo docs")
case "$PORT" in ''|*[!0-9]*) problems+=("bad --port $PORT") ;; esac
for v in MAX_HOURS STEP_RETRIES RETRY_PAUSE MIN_PHASE_SECONDS LOAD_TIMEOUT; do
    case "${!v}" in ''|*[!0-9]*) problems+=("bad $v ${!v} (a whole number)") ;; esac
done
if [ -r "$IQ3" ] && [ -r "$Q4" ] && [ -f "$AB" ]; then
    # two different files, Q4 the bigger, the same embedded chat template (unless pinned)
    MODELS_ARGS=(models --a "$IQ3" --b "$Q4")
    [ -z "$TEMPLATE" ] || MODELS_ARGS+=(--template-pinned)
    models_err=""
    models_err="$(ab "${MODELS_ARGS[@]}" 2>&1 >/dev/null)" || {
        while IFS= read -r ln; do [ -z "$ln" ] || problems+=("models: ${ln#error: }"); done <<< "$models_err"
        [ -n "$models_err" ] || problems+=("models: the GGUF check failed ($PY)")
    }
fi
SELECT_ARGS=(select --out "$OUT" --tasks "$TASKS" --n "$N" --seed "$SEED" --repo "$REPO"
             --temperature "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P" --concurrency "$SLOTS"
             --label-a IQ3_XXS --label-b Q4_K_M)
[ -z "$PACK" ] || SELECT_ARGS+=(--pack "$PACK")

if [ "$EXECUTE" != 1 ]; then
    echo "DRY RUN (nothing is started, stopped or sent; add --execute)"
    for p in "${problems[@]+"${problems[@]}"}"; do echo "  problem: $p"; done
    UNTIL_ARGS=()
    if [ -n "$UNTIL" ]; then
        until_now=""
        until_now="$(ab time-left --until "$UNTIL" --iso 2>/dev/null)" || true
        if [ -n "$until_now" ]; then
            echo "--until $UNTIL is resolved once when --execute starts (now it would be $until_now)"
            UNTIL_ARGS=(--until "$until_now")
        else
            echo "--until $UNTIL is resolved once when --execute starts (now it does not resolve: an HH:MM" \
                 "more than 12 h ahead, or not a time)"
            UNTIL_ARGS=(--until "$UNTIL")
        fi
    fi
    echo "1. select:"; show env PYTHONPATH="$CODE/src" "$PY" "$AB" "${SELECT_ARGS[@]}"
    echo "2. arm A (IQ3), unless complete:"
    server_cmd "$IQ3" qwen3.8-27b-iq3_xxs; show "${SERVER_ARGV[@]}"
    echo "   pid -> $PIDFILE; wait for $URL/health"
    show env PYTHONPATH="$CODE/src" "$PY" "$AB" arm --out "$OUT" --arm A --url "$URL" "${UNTIL_ARGS[@]+"${UNTIL_ARGS[@]}"}"
    echo "   stop the recorded PID (SIGTERM, SIGKILL after 60 s)"
    echo "3. arm B (Q4_K_M) and the judge, unless complete:"
    server_cmd "$Q4" qwen3.8-27b-q4_k_m; show "${SERVER_ARGV[@]}"
    show env PYTHONPATH="$CODE/src" "$PY" "$AB" arm --out "$OUT" --arm B --url "$URL" "${UNTIL_ARGS[@]+"${UNTIL_ARGS[@]}"}"
    show env PYTHONPATH="$CODE/src" "$PY" "$AB" judge --out "$OUT" --judge-url "$URL" "${UNTIL_ARGS[@]+"${UNTIL_ARGS[@]}"}"
    echo "   stop the recorded PID"
    echo "   (a step that leaves work for a reason other than the time box is rerun up to $STEP_RETRIES times)"
    echo "4. report:"; show env PYTHONPATH="$CODE/src" "$PY" "$AB" report --out "$OUT"
    [ -z "$NOTIFY" ] || { echo "   then: $NOTIFY < $OUT/ab_summary.txt"; }
    [ -z "$UNTIL" ] && echo "  note: --execute needs --until (a hard end)"
    if [ ${#problems[@]} -gt 0 ]; then exit 2; fi
    exit 0
fi

# --- execute ------------------------------------------------------------------
[ ${#problems[@]} -eq 0 ] || { for p in "${problems[@]}"; do echo "error: $p" >&2; done; exit 2; }
[ -n "$UNTIL" ] || { echo "error: --execute needs --until (a hard end)" >&2; exit 2; }
mkdir -p "$OUT"
chmod 700 "$OUT"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    other="$(cat "$LOCKDIR/pid" 2>/dev/null || true)"
    if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then die 2 "another run ($other) holds $LOCKDIR"; fi
    log "taking over a stale lock (${other:-no pid})"
    rm -rf "$LOCKDIR"; mkdir "$LOCKDIR"
fi
echo $$ > "$LOCKDIR/pid"
trap cleanup EXIT
trap on_signal INT TERM
[ ! -f "$PIDFILE" ] || stop_server || die 3 "a recorded server could not be stopped"

PYTHONPATH="$CODE/src" "$PY" -c 'import httpx, prometheus' 2>/dev/null \
    || die 2 "$PY cannot import prometheus from $CODE/src"
resolve_until
check_sha "$IQ3" "$IQ3_SHA"
check_sha "$Q4" "$Q4_SHA"
log "teacher A/B: out=$OUT until=$UNTIL port=$PORT"

ab "${SELECT_ARGS[@]}" >> "$OUT/run.log" || die 2 "select failed"
enough_time "anything"

phase_rc=0
if ! ab status --out "$OUT" --need arm-A >/dev/null; then
    enough_time "the IQ3 server"
    start_server "$IQ3" qwen3.8-27b-iq3_xxs "$MIN_FREE_IQ3" iq3
    run_step "arm A" arm --out "$OUT" --arm A --url "$URL" --until "$UNTIL" || phase_rc=$?
    stop_server
    [ "$phase_rc" = 0 ] || step_failed "arm A" "$phase_rc"
fi
if ! ab status --out "$OUT" --need arm-B >/dev/null || ! ab status --out "$OUT" --need judge >/dev/null; then
    enough_time "the Q4_K_M server"
    start_server "$Q4" qwen3.8-27b-q4_k_m "$MIN_FREE_Q4" q4
    if ! ab status --out "$OUT" --need arm-B >/dev/null; then
        run_step "arm B" arm --out "$OUT" --arm B --url "$URL" --until "$UNTIL" || phase_rc=$?
        [ "$phase_rc" = 0 ] || { stop_server; step_failed "arm B" "$phase_rc"; }
    fi
    run_step "judging" judge --out "$OUT" --judge-url "$URL" --until "$UNTIL" || phase_rc=$?
    stop_server
    [ "$phase_rc" = 0 ] || step_failed "judging" "$phase_rc"
fi

report_rc=0
ab report --out "$OUT" > /dev/null || report_rc=$?
cat "$OUT/ab_summary.txt" >&2
if [ -n "$NOTIFY" ]; then
    $NOTIFY < "$OUT/ab_summary.txt" || log "notify failed"
fi
exit "$report_rc"
