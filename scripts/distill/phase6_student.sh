#!/usr/bin/env bash
# Phase 6 for one trained adapter, in one command, run on the 3090 Ti host ("here"):
#
#   TOWER=<ssh target of the 4090 host> scripts/distill/phase6_student.sh <adapter dir there> [--execute]
#   e.g. TOWER=<user>@<4090 host> scripts/distill/phase6_student.sh \
#            distill/phase5/runs/9b-r32-v4-max-derived/adapter-max-derived --execute
#
# Without --execute it checks what it can and prints the plan. Stages (each resumes; --only runs one):
#   pull      the adapter and its trainer manifest.json from the 4090 host, sha256-checked at both
#             ends, into ~/distill/phase5/runs/<run>/ here; the train split the manifest names must
#             be here with the same sha256 (it calibrates the imatrix).
#   merge     quantize.py merge --merge-mode stream, here, on the CPU: the official 16-bit base lives
#             here, and the stream merge holds one weight at a time (peak ~1.3 GB RSS; PEFT's merge
#             would hold ~19 GB, more than this host may use). -> ~/distill/phase6/<run>/merged
#   calib     quantize.py calibrate, here (the splits and eval sets live here): the train-only imatrix
#             text, checked against every eval file, the Phase 4 eval-tools set included.
#   push      merged/ and calib/ to ~/distill/phase6/<run>/ on the 4090 host, sha256-checked. First the
#             4090 disk preflight (disk_preflight.py there, the shared check of every 4090 job): only what
#             the REMAINING stages still have to write counts (merged/, the BF16 GGUF, the imatrix's Q8_0
#             copy, each quant: a file already there counts only for its missing part), plus 1 GiB fixed
#             and the 10 GiB floor the post-train pipeline keeps; it prints the plan. Too little: it logs,
#             Telegrams the operator and this exits 28 (nothing changed).
#   quantize  on the 4090 host: quantize.py run --no-adapter --base <merged> (convert, imatrix on its GPU
#             on a Q8_0 copy of the BF16 GGUF (the BF16 cuBLAS path aborted on 2026-11-01: quantize.py
#             step 5), Q4_K_M + Q5_K_M from the BF16 GGUF, perplexity and metadata sanity, manifest), as the user unit
#             distill-quantize-p6-<run> (a setups.py restore stops it; the same command resumes).
#             It needs the GPU free (training over, judge and business servers down): checked first.
#   fetch     the GGUFs, imatrix and manifest back here, sha256-checked against the manifest.
#   cleanup   the 16-bit intermediates on both hosts (merged/, base-view/, the BF16 GGUF); the
#             quantized GGUFs stay on both (the 4090 host scores them).
#
# Why the 4090 host does the middle: here, the GGUF converter peaks at 8.3 GB RSS and an imatrix on
# the CPU runs ~18 s per 512 tokens (~5 h for the 524k-token calibration); the host keeps >= 8 GB
# free for its services and its GPU is not ours. There, convert takes ~2 min (CPU) and the imatrix
# minutes (GPU). --local runs everything here on the CPU instead (quantize.py run --merge-mode
# stream --ngl 0 under capped_scope.sh, imatrix and perplexity at 512 tokens per chunk so their
# buffers fit): it pauses (exit 75) whenever a step needs more memory than the floor leaves, and
# resumes when run again.
#
# Options:
#   --execute            run (default: check and print the plan)
#   --only STAGE         one stage: pull merge calib push quantize fetch cleanup
#   --local              everything here on the CPU (see above)
#   --tower-cpu          the 4090 step on its CPU (CUDA hidden, --ngl 0, no GPU check): rehearsal,
#                        or a fallback (its imatrix then runs ~11 s per 512 tokens: ~3 h at 524k)
#   --tower-ngl N        layers on the 4090 GPU (default 99; lower it if the BF16 imatrix runs out
#                        of VRAM: 17.9 GB of weights plus buffers on a 24 GB card)
#   --quants LIST        default Q4_K_M,Q5_K_M (Q4_K_M is what base-9b was scored at)
#   --calib-tokens N --imatrix-ctx N --ppl-ctx N --ppl-chunks N   passed to quantize.py (rehearsal)
#   --stop-at ISO        passed to quantize.py on the 4090 host (with a UTC offset): it stops before
#                        then and exits 75 (paused); the same command resumes
#   --keep-intermediates skip cleanup
#   --keep-bf16          cleanup keeps the BF16 GGUF on the 4090 host (it is scored there as a
#                        reference); merged/ and base-view/ are still removed
# Environment: TOWER (the 4090 host's ssh target, required), FLOOR_GIB (default 8).
# Exit: 0 done, 2 error, 28 the 4090 disk preflight refused (logged and Telegrammed there).
#
# Rehearsed 2026-10-31 with the 9B smoke adapter (the 4090 step with --tower-cpu, its GPU being
# busy): pull 1 s, merge 3.7 min (here, 4.3 GB cap), calib 4 s, push 3.7 min, 4090 step 11.6 min
# on its CPU (convert 2 min, imatrix 6 min for 16k tokens, Q4_K_M + Q5_K_M, sanity), fetch 2.8 min.
#
# Scoring (after this, on the 4090 GPU, the server and settings base-9b was scored with; ~1.5-2 h
# for the four sets at 256k, as base-9b took 94 min). L=<run>-Q4_K_M, R=~/distill/phase6/<run>/score:
#   systemd-run --user --unit=distill-score-$L --collect ~/llama.cpp/build/bin/llama-server \
#     -m ~/distill/phase6/<run>/gguf/$L.gguf -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja \
#     -ngl 99 -fit off -b 2048 -ub 2048 --host 127.0.0.1 --port 18080 --alias $L
#   for s in sanity:600 eval-tools:900 eval-injection:900 eval-longctx:1800; do
#     ~/distill/venv-score/bin/python ~/distill/code/scripts/distill/score.py run \
#       --base-url http://127.0.0.1:18080 --label $L --cases ~/distill/evalsets/${s%%:*}.jsonl \
#       --out $R/${s%%:*}.jsonl --concurrency 1 --timeout ${s##*:}; done
#   then (this checkout's score.py, as the head-4ef1dfe baselines were summarized): summarize the
#   concatenated rows (--cases the same four files) into $R/$L.summary.json, and
#   score.py compare --base ~/distill/phase4/results/base-9b/baseline/base-9b.head-4ef1dfe.summary.json
#     --candidate $R/$L.summary.json --also saluki.head-4ef1dfe.summary.json (3090 Ti host,
#     ~/distill/phase4/results/saluki) --markdown-out $R/gate.md --json-out $R/gate.json
#
# Max-derived (DISTILL-OPEN-QUESTIONS Q6b): the run's name carries the tag, so every output does;
# the imatrix text is Max-derived, approved by the operator's Q6b ruling (the flag below).
# Nothing here uploads anything; files move between our two hosts only.
set -euo pipefail

die() { echo "phase6: $*" >&2; exit 2; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

[ $# -ge 1 ] || die "usage: phase6_student.sh <adapter dir on the 4090 host> [--execute] [options]"
# relative to the 4090 host's home: "~/x", "x" and "/home/<user>/x" all name x
ADAPTER_REMOTE=$(printf '%s' "${1%/}" | sed -E 's#^~/##; s#^/home/[^/]+/##')
shift
EXECUTE=0; ONLY=""; LOCAL=0; TOWER_CPU=0; TOWER_NGL=99; QUANTS=Q4_K_M,Q5_K_M; KEEP=0; KEEP_BF16=0; STOP_AT=""; SIZES=()
while [ $# -gt 0 ]; do
  case $1 in
    --execute) EXECUTE=1 ;;
    --only) ONLY=$2; shift ;;
    --local) LOCAL=1 ;;
    --tower-cpu) TOWER_CPU=1 ;;
    --tower-ngl) TOWER_NGL=$2; shift ;;
    --quants) QUANTS=$2; shift ;;
    --calib-tokens|--imatrix-ctx|--ppl-ctx|--ppl-chunks) SIZES+=("$1" "$2"); shift ;;
    --keep-intermediates) KEEP=1 ;;
    --keep-bf16) KEEP_BF16=1 ;;
    --stop-at) STOP_AT=$2; shift ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

TOWER=${TOWER:-}
[ -n "$TOWER" ] || die "set TOWER to the 4090 host's ssh target (user@host)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)       # this checkout's scripts/distill
TOOLS=$HOME/distill/phase6/work                            # here: mvenv, qvenv, CPU llama.cpp
PY=$TOOLS/mvenv/bin/python                                 # torch (stream merge), stdlib rest
LLAMA_CPU=$TOOLS/llama.cpp-9d57ce456-cpu
QPY=$TOOLS/qvenv/bin/python                                # the converter's requirements
T_TOOLS=distill/phase6-rehearsal                           # there (relative to its home)
T_LLAMA=$T_TOOLS/llama.cpp-9d57ce456-cuda
T_QPY=$T_TOOLS/qvenv/bin/python
BASE=$HOME/distill/models/qwen3.5-9b-official
TEMPLATE_REL=distill/models/qwen3.5-9b-tokenizer
EVALSETS=$HOME/distill/phase3/evalsets
SCORED_TOOLS=$HOME/distill/phase3/build-v2/eval-tools.jsonl   # the eval-tools Phase 4 scored
APPROVAL=--i-have-operator-approval-for-max-derived-data      # Q6b ruling, 2026-10-30

ANAME=$(basename "$ADAPTER_REMOTE")
RUN=$(basename "$(dirname "$ADAPTER_REMOTE")")
RUNS=$HOME/distill/phase5/runs/$RUN
ADAPTER=$RUNS/$ANAME
P6REL=distill/phase6/$RUN
OUT=$HOME/$P6REL
LOGS=$OUT/p6-logs
case $RUN in *max-derived*|*max_derived*) ;; *) die "run $RUN does not carry the max-derived tag" ;; esac
for t in "$PY" "$QPY" "$LLAMA_CPU/build/bin/llama-quantize" "$BASE/config.json" "$CODE/quantize.py"; do
  [ -e "$t" ] || die "missing $t (set up by the 2026-10-31 Phase 6 rehearsal under ~/distill/phase6/work)"
done
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
run_capped() {  # NAME -- CMD...
  local name=$1; shift 2
  FLOOR_GIB=${FLOOR_GIB:-8} "$CODE/capped_scope.sh" "$name" "$LOGS/$name.log" -- "$@"
}
tower() { "${SSH[@]}" "$TOWER" "$@"; }

manifest_field() {  # FIELD -> from the trainer manifest here
  "$PY" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); v=d
for k in sys.argv[2].split("."): v=v.get(k) if isinstance(v, dict) else None
print("" if v is None else v)' "$RUNS/manifest.json" "$1"
}

say "run $RUN (adapter $ANAME); out here $OUT; 4090 host $TOWER:~/$P6REL"
[ "$EXECUTE" = 1 ] || say "PLAN ONLY (add --execute)"
if [ "$EXECUTE" = 1 ]; then mkdir -p -m 700 "$RUNS" "$OUT" "$LOGS"; fi

# -- pull ---------------------------------------------------------------------------------------
if want pull; then
  say "pull: sha256 on the 4090 host"
  REMOTE_SUMS=$(tower "cd ~/$ADAPTER_REMOTE/.. && sha256sum $ANAME/adapter_config.json $ANAME/adapter_model.safetensors manifest.json")
  echo "$REMOTE_SUMS"
  if [ "$EXECUTE" = 1 ]; then
    rsync -a --chmod=Du=rwx,Dgo=,Fu=rw,Fgo= -e "${SSH[*]}" "$TOWER:$ADAPTER_REMOTE" \
      "$TOWER:$(dirname "$ADAPTER_REMOTE")/manifest.json" "$RUNS/"
    (cd "$RUNS" && echo "$REMOTE_SUMS" | sha256sum -c --quiet) || die "adapter sha256 mismatch after the copy"
    say "pull: adapter sha256 verified here"
  fi
fi
if [ -f "$RUNS/manifest.json" ]; then
  STATUS=$(manifest_field status); TRAIN_SHA=$(manifest_field data.sha256)
  TRAIN_REMOTE=$(manifest_field data.path)
  BUILD=$HOME/distill/${TRAIN_REMOTE#*/distill/}; BUILD=$(dirname "$BUILD")
  say "trainer manifest: status $STATUS; train split $BUILD (sha256 ${TRAIN_SHA:0:16}...)"
  case $STATUS in failed|diverged) die "the trainer manifest says $STATUS: nothing to quantize" ;; esac
  if [ "$LOCAL" = 1 ] || want calib; then
    [ -f "$BUILD/train.jsonl" ] || die "$BUILD/train.jsonl is not here"
    HAVE=$(sha256sum "$BUILD/train.jsonl" | cut -d' ' -f1)
    [ "$HAVE" = "$TRAIN_SHA" ] || die "$BUILD/train.jsonl here is $HAVE, not the split the adapter trained on"
  fi
elif [ "$EXECUTE" = 1 ]; then
  die "no trainer manifest at $RUNS/manifest.json (run --only pull first)"
fi

# -- local: everything here on the CPU --------------------------------------------------------
if [ "$LOCAL" = 1 ]; then
  [ -f "$BUILD/train.jsonl" ] || die "no train split here"
  CMD=("$PY" -I "$CODE/quantize.py" run --adapter "$ADAPTER" --base "$BASE"
       --template-dir "$HOME/$TEMPLATE_REL" --data "$BUILD" --evalsets "$EVALSETS"
       --eval "$SCORED_TOOLS" --merge-mode stream --llama-cpp-dir "$LLAMA_CPU"
       --convert-python "$QPY" --ngl 0 --quants "$QUANTS" --out "$OUT"
       --imatrix-ctx 512 --ppl-ctx 512 "${SIZES[@]}" "$APPROVAL")
  if [ "$EXECUTE" = 1 ]; then
    run_capped local -- "${CMD[@]}" --execute; exit $?
  fi
  "${CMD[@]}"; exit $?
fi

# -- merge (here, CPU) --------------------------------------------------------------------------
if want merge; then
  CMD=("$PY" -I "$CODE/quantize.py" merge --adapter "$ADAPTER" --base "$BASE"
       --template-dir "$HOME/$TEMPLATE_REL" --out "$OUT/merged" --merge-mode stream)
  if [ "$EXECUTE" = 1 ]; then
    if [ -f "$OUT/merged/merge-info.json" ]; then "${CMD[@]}"; else run_capped merge -- "${CMD[@]}" --execute; fi
  else
    "${CMD[@]}" || true
  fi
fi

# -- calib (here) ---------------------------------------------------------------------------------
if want calib; then
  CMD=("$PY" -I "$CODE/quantize.py" calibrate --data "$BUILD" --evalsets "$EVALSETS"
       --eval "$SCORED_TOOLS" --out "$OUT" "${SIZES[@]}")
  if [ "$EXECUTE" = 1 ]; then run_capped calib -- "${CMD[@]}" --execute >/dev/null; tail -3 "$LOGS/calib.log.monitor"
  else "${CMD[@]}"; fi
fi

# -- push (here -> 4090 host) ---------------------------------------------------------------------
TCODE=$P6REL/code
# The 4090 disk preflight for the stages still to run there: LABEL:GIB:PATH items for disk_preflight.py (a
# file or dir already there counts only for what it still lacks). Sizes from the merge here (the BF16 GGUF is
# ~1.02x the merged weights; each quant its quantize.py SIZE_RATIO of the BF16 file).
disk_items() {
  local mb m bf q r items have_imatrix
  mb=$(du -sb "$OUT/merged" 2>/dev/null | cut -f1)
  [ -n "$mb" ] || mb=$(tower "du -sb ~/$P6REL/merged 2>/dev/null | cut -f1")
  [ -n "$mb" ] || die "no merged/ here or on the 4090 host to size the plan from"
  m=$(awk -v b="$mb" 'BEGIN{printf "%.2f", b/1073741824}')
  bf=$(awk -v m="$m" 'BEGIN{printf "%.2f", m*1.03}')
  items="--item merged:$m:$HOME_T/$P6REL/merged --item BF16:$bf:$HOME_T/$P6REL/gguf/$RUN-BF16.gguf"
  have_imatrix=$(tower "test -e ~/$P6REL/gguf/$RUN.imatrix.gguf && echo yes || echo no")
  if [ "$have_imatrix" != yes ]; then
    items="$items --item imatrix-Q8_0-copy:$(awk -v b="$bf" 'BEGIN{printf "%.2f", b*0.54}'):$HOME_T/$P6REL/gguf/$RUN.imatrix-src-Q8_0.gguf"
  fi
  IFS=, read -ra QS <<< "$QUANTS"
  for q in "${QS[@]}"; do
    case $q in Q4_K_S) r=0.29 ;; Q4_K_M) r=0.31 ;; Q5_K_S) r=0.35 ;; Q5_K_M) r=0.36 ;; Q6_K) r=0.41 ;; Q8_0) r=0.54 ;; *) r=0.6 ;; esac
    items="$items --item $q:$(awk -v b="$bf" -v r="$r" 'BEGIN{printf "%.2f", b*r}'):$HOME_T/$P6REL/gguf/$RUN-$q.gguf"
  done
  echo "$items"
}
disk_preflight() {  # $1 = --no-notify (the plan: report only)
  local rc=0 out items
  HOME_T=$(tower 'echo $HOME')
  items=$(disk_items) || die "could not size the 4090 plan"
  out=$(tower "python3 -I ~/$TCODE/disk_preflight.py check --job 'Phase 6 $RUN' --path ~ --need-gib 1 --floor-gib 10 $items --detail 'Phase 6 stages still to run on the 4090: push, convert, imatrix (Q8_0 copy), quantize.' ${1:-}") || rc=$?
  echo "  $out"
  [ -z "${1:-}" ] || return 0
  if [ "$rc" = 28 ]; then echo "phase6: the 4090 disk preflight refused: $out" >&2; exit 28; fi
  [ "$rc" = 0 ] || die "the 4090 disk preflight did not run (exit $rc): $out"
}
if want push && [ "$EXECUTE" != 1 ] && tower "test -f ~/$TCODE/disk_preflight.py"; then
  say "push (plan): the 4090 disk preflight with the code there now (report only)"
  disk_preflight --no-notify || true
fi
if want push && [ "$EXECUTE" = 1 ]; then
  tower "mkdir -p -m 700 ~/$P6REL ~/$TCODE"
  rsync -a --delete --exclude __pycache__ -e "${SSH[*]}" "$CODE/" "$TOWER:$TCODE/"
  say "push: disk preflight on the 4090 host (only what the remaining stages still write counts)"
  disk_preflight
  say "push: merged/ + calib/ (~19 GB; rsync skips what is already there)"
  rsync -a --chmod=Du=rwx,Dgo=,Fu=rw,Fgo= -e "${SSH[*]}" "$OUT/merged" "$OUT/calib" "$TOWER:$P6REL/"
  (cd "$OUT" && sha256sum merged/*.safetensors calib/*) > "$LOGS/push.sha256"
  tower "cd ~/$P6REL && sha256sum -c --quiet" < "$LOGS/push.sha256" || die "sha256 mismatch on the 4090 host"
  say "push: sha256 verified on the 4090 host"
fi

# -- quantize (4090 host) -------------------------------------------------------------------------
# One unit per run, named distill-quantize-p6-<run> (a setups.py restore stops distill-* units;
# quantize.py then pauses with exit 75 and this stage, run again, resumes it). If the unit is
# already running (an earlier call lost its ssh), this waits for it instead of starting another.
if want quantize; then
  THOME=$(tower 'echo $HOME')
  T=$THOME/$P6REL
  UNIT=distill-quantize-p6-$(echo "$RUN" | tr -c 'A-Za-z0-9-\n' '-')
  NGL=$TOWER_NGL; SETENV=""  # python -I ignores PYTHON* variables: -u unbuffers instead
  if [ "$TOWER_CPU" = 1 ]; then NGL=0; SETENV="$SETENV --setenv=CUDA_VISIBLE_DEVICES="; fi
  STOPS=()
  if [ -n "$STOP_AT" ]; then
    case $STOP_AT in *[!0-9T:+-]*) die "--stop-at takes an ISO time with a UTC offset, e.g. 2026-11-02T05:00:00-05:00" ;; esac
    STOPS=(--stop-at "$STOP_AT")
  fi
  QCMD=("$THOME/$T_QPY" -I -u "$THOME/$TCODE/quantize.py" run --no-adapter --base "$T/merged"
        --label "$RUN" --out "$T" --data "$T/nodata" --evalsets= --train-sha256 "${TRAIN_SHA:-x}"
        --template-dir "$THOME/$TEMPLATE_REL" --llama-cpp-dir "$THOME/$T_LLAMA"
        --convert-python "$THOME/$T_QPY" --ngl "$NGL" --quants "$QUANTS" "${SIZES[@]}"
        "${STOPS[@]}" "$APPROVAL")
  if [ "$EXECUTE" = 1 ]; then
    if tower "systemctl --user is-active --quiet $UNIT"; then
      say "quantize: $UNIT is already running on the 4090 host; waiting for it"
    else
      if [ "$TOWER_CPU" != 1 ]; then
        USED=$(tower "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
        [ "${USED:-99999}" -lt 1500 ] || die "the 4090 GPU holds ${USED} MiB (training, the judge or business is on it); run --only quantize again when it is free"
      fi
      tower "mkdir -p $T/nodata $T/logs && rm -f $T/logs/$UNIT.rc && systemd-run --user --quiet --collect --unit=$UNIT --working-directory=$T -p KillSignal=SIGTERM -p TimeoutStopSec=90 -p Nice=5 -p MemoryMax=40G $SETENV -p StandardOutput=append:$T/logs/$UNIT.log -p StandardError=append:$T/logs/$UNIT.log -- bash -c '${QCMD[*]} --execute; echo \$? > $T/logs/$UNIT.rc'"
      say "quantize: started $UNIT on the 4090 host (log $T/logs/$UNIT.log)"
    fi
    while tower "systemctl --user is-active --quiet $UNIT"; do sleep 30; done
    RC=$(tower "cat $T/logs/$UNIT.rc 2>/dev/null || echo stopped")
    tower "tail -n 30 $T/logs/$UNIT.log"
    [ "$RC" = 0 ] || die "quantize on the 4090 host ended: exit $RC (75 = paused, 1 = a sanity check failed, 2 = error; the same command resumes)"
    say "quantize: PASS"
  else
    echo "  on the 4090 host, as unit $UNIT: ${QCMD[*]} --execute"
    tower "${QCMD[*]}" 2>&1 | tail -n 30 || true
  fi
fi

# -- fetch (4090 host -> here) ----------------------------------------------------------------------
if want fetch && [ "$EXECUTE" = 1 ]; then
  mkdir -p "$OUT/gguf"
  FILES=()
  IFS=, read -ra QS <<< "$QUANTS"
  for q in "${QS[@]}"; do FILES+=("$TOWER:$P6REL/gguf/$RUN-$q.gguf"); done
  FILES+=("$TOWER:$P6REL/gguf/$RUN.imatrix.gguf")
  rsync -a --chmod=Fu=rw,Fgo= -e "${SSH[*]}" "${FILES[@]}" "$OUT/gguf/"
  rsync -a -e "${SSH[*]}" "$TOWER:$P6REL/quantize-manifest.json" "$TOWER:$P6REL/run.json" "$OUT/"
  rsync -a -e "${SSH[*]}" "$TOWER:$P6REL/logs/" "$OUT/logs/"
  "$PY" -I - "$OUT" <<'EOF' || die "fetched files do not match the manifest"
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1]); m = json.loads((out / "quantize-manifest.json").read_text())
bad = []
for a in m["artifacts"]:
    p = out / "gguf" / a["name"]
    if a["kind"] == "gguf" and a.get("quant") in ("BF16", "F16"):
        continue  # the 16-bit GGUF stays on the 4090 host until cleanup
    if a["kind"] not in ("gguf", "imatrix"):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    ok = h.hexdigest() == a["sha256"]
    print(f"  {a['name']}  {a['bytes']:,} B  sha256 {a['sha256'][:16]}  {'ok' if ok else 'MISMATCH'}")
    bad += [] if ok else [a["name"]]
print(f"  result {m.get('result')}; max_derived {m.get('max_derived')}; private {m.get('private')}")
sys.exit(1 if bad or m.get("result") != "PASS" else 0)
EOF
  say "fetch: GGUFs here, sha256 match the manifest"
fi

# -- cleanup --------------------------------------------------------------------------------------
if want cleanup && [ "$EXECUTE" = 1 ] && [ "$KEEP" = 0 ]; then
  [ -f "$OUT/quantize-manifest.json" ] || die "no fetched manifest here: not cleaning up"
  if [ "$KEEP_BF16" = 1 ]; then
    tower "rm -rf ~/$P6REL/merged ~/$P6REL/base-view"
    rm -rf "$OUT/merged"
    say "cleanup: merged/ removed; the BF16 GGUF stays on the 4090 host (--keep-bf16)"
  else
    tower "rm -rf ~/$P6REL/merged ~/$P6REL/base-view ~/$P6REL/gguf/$RUN-BF16.gguf"
    rm -rf "$OUT/merged"
    say "cleanup: merged/ and the BF16 GGUF removed (the quantized GGUFs stay on both hosts)"
  fi
fi
say "done"
