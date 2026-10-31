#!/usr/bin/env bash
# serve_teacher.sh {plan|keygen|check|exec}   (the 4090 tower; installed as ~/distill/teacher-serve/serve_teacher.sh)
#
# The distillation teacher served over the tailnet for gym generation on the 3090 Ti host ("option 3",
# operator 2026-10-31 ~11:20 ET; DISTILL-OPEN-QUESTIONS #11, #12): this host's Qwen3.8-27B UD-Q4_K_XL in
# llama-server, text only (no mmproj), fully offloaded, several slots, an API key, bound ONLY to this host's
# tailnet address. Every tool command the teacher asks for runs on the 3090 Ti host inside its bwrap jail
# (teacher_gen.py --teacher-key-file); NOTHING executes here: this host has no jail (OQ #11).
#
#   plan    read-only: the VRAM plan for TEACHER_SLOTS x TEACHER_CTX_PER_SLOT (and the neighbours), the flags
#   keygen  writes TEACHER_KEY_FILE (0600, 64 hex chars from /dev/urandom) if it is missing; prints only a
#           fingerprint (sha256 prefix), never the key. The 3090 Ti host PULLS it over ssh into its own 0600 file
#           (gen_runner.py remote-setup): this host has no ssh login to the 3090 Ti host.
#   check   read-only: every precondition of `exec`; one line per problem, exit 1 on any
#   exec    check, then exec llama-server (the ExecStart of distill-teacher-serve.service). SERVE_TEACHER_DRY=1
#           prints the argv instead of exec'ing it.
#
# Started and stopped only as the user unit distill-teacher-serve.service (template beside this file), by the
# 4090 window machinery: it stops the business llama-server first (and sends the "down" Telegram); `exec`
# refuses while llama-server.service is active, so starting the teacher never takes the business server down
# by itself. The unit is named distill-*, so a business restore (`setups.py apply business`: the window end,
# the dead-man at Mon 06:00 EST) stops it before the business server starts. No [Install]: never at boot.
#
# Security:
# * --host is the tailnet address only: an IPv4 in 100.64.0.0/10 assigned to TEACHER_IFACE (tailscale0) on
#   this host; 0.0.0.0, ::, loopback, a LAN address or one this host does not hold is refused.
# * --api-key-file (this llama.cpp build has it): the key is never in this process's argv, the unit, a log or
#   ps output. The file must be a regular 0600 file owned by this user (not a symlink). llama-server then
#   refuses every endpoint without the key except its public /health and /v1/models.
# * --no-slots: /slots would expose per-request state. No --props (POST /props stays off), no --metrics.
#
# Model and template (pinned; checked without reading the 17 GB file: size + the manifest's recorded sha):
# * TEACHER_MODEL: unsloth/Qwen3.8-27B-GGUF @ fe1e2a23 Qwen3.8-27B-UD-Q4_K_XL.gguf, 17,923,394,624 B,
#   sha256 TEACHER_SHA256 (recorded in ~/distill/models/qwen38-27b-gguf/MANIFEST.json, 2026-10-31).
#   SERVE_TEACHER_VERIFY_SHA=1 re-hashes the file once (read-only, about a minute).
# * --chat-template-file TEACHER_TEMPLATE: the BASE model's chat_template.jinja (Qwen/Qwen3.8-27B; sha256
#   TEACHER_TEMPLATE_SHA256, 8,952 B), the template the 3090 Ti's IQ3 teacher has embedded byte for byte.
#   The UD-Q4_K_XL file embeds Unsloth's own template (9,993 chars, sha256 12827f24...): different from the
#   local teacher's, so it is NOT used; both teachers render a request identically. Copy it from the 3090 Ti
#   host's ~/distill/models/teacher/provenance/base-chat_template.jinja (gen_runner.py remote-setup does).
#   TEACHER_TEMPLATE=embedded uses the GGUF's own instead (recorded as a difference by teacher_ab.py).
#
# Flags against the 3090 Ti's local teacher (scripts/distill/units/distill-teacher.service), same llama.cpp
# commit (9d57ce456; this build is sm_89): the same --jinja, chat template (above), --reasoning-budget 2048
# with the same budget message, sampling defaults (temp 1.0, top-k 20, top-p 0.95, min-p 0; teacher_gen also
# sends its sampling explicitly), flash attention, q8_0 K and V cache, -b 2048 -ub 512, --no-mmproj,
# --no-slots, -ngl 99 --fit off. Different on purpose: the quant (UD-Q4_K_XL, all on the GPU, vs IQ3_XXS),
# TEACHER_SLOTS slots of TEACHER_CTX_PER_SLOT each (--no-kv-unified: every slot keeps its own 32,768; the local
# teacher shares one 57,344 pool between 2 slots with the daemon), the tailnet bind, the API key, the alias.
#
# VRAM plan (24,564 MiB card, nothing else resident once the business server is stopped):
#   weights on the GPU   16,400.6 MiB  (17,082.6 MiB of tensors - 682.0 MiB token embeddings, which llama.cpp
#                                       keeps on the CPU; incl. the Q6_K output 994.6 and the MTP block 272.6)
#   KV, q8_0             34,816 B/token (16 attention layers x 4 KV heads x 256 x K+V x 1.0625 B; measured on
#                                       the 3090 Ti's teacher, same architecture and build; +6.25% if the MTP
#                                       layer's attention got a cache too: 17 layers)
#   recurrent state      149.6 MiB per slot (48 gated-delta-net layers, f32; measured on the 3090 Ti)
#   compute + context    1,200 MiB     (the 3090 Ti's teacher measured 1,191.9 at -ub 512 WITH a vision encoder)
#   4 slots x 32,768:    16,400.6 + 4,352.0 + 598.5 + 1,200 = 22,551 MiB -> ~2,013 MiB spare (1,741 at 17 layers)
#   5 slots x 32,768:    23,517 MiB -> ~1,047 spare (~707 at 17 layers): too close for a first start
# (`plan` and the check count whole MiB, rounded up: 22,553 for 4 slots, so the check wants 23,153 free.)
# The gym's requests: 10.5K tokens per call on average, p90 12.7K, the largest task's mean 15.4K (the local
# run's ledger); training rows are cut to 16,384. 32,768 per slot holds them with room; a call that would not
# fit is a context_overflow rejection (as on the local teacher). `exec` refuses unless nvidia-smi shows the
# projection + TEACHER_MARGIN_MIB free.
set -euo pipefail

: "${TEACHER_LLAMA_SERVER:=$HOME/llama.cpp/build/bin/llama-server}"
: "${TEACHER_MODEL:=$HOME/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
: "${TEACHER_MODEL_BYTES:=17923394624}"
: "${TEACHER_SHA256:=bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372}"
: "${TEACHER_MANIFEST:=$HOME/distill/models/qwen38-27b-gguf/MANIFEST.json}"
: "${TEACHER_TEMPLATE:=$HOME/distill/teacher-serve/chat_template.jinja}"
: "${TEACHER_TEMPLATE_SHA256:=c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041}"
: "${TEACHER_KEY_FILE:=$HOME/distill/secrets/teacher-serve.key}"
: "${TEACHER_ALIAS:=qwen3.8-27b-ud-q4_k_xl}"
: "${TEACHER_PORT:=18094}"
: "${TEACHER_IFACE:=tailscale0}"
: "${TEACHER_BIND:=}"
: "${TEACHER_SLOTS:=4}"
: "${TEACHER_CTX_PER_SLOT:=32768}"
: "${TEACHER_BUSINESS_UNIT:=llama-server.service}"
: "${TEACHER_MARGIN_MIB:=600}"
# the VRAM plan's constants (MiB, except the per-token KV in bytes)
: "${TEACHER_WEIGHTS_MIB:=16401}"
: "${TEACHER_KV_BYTES_PER_TOKEN:=34816}"
: "${TEACHER_RS_MIB_PER_SLOT:=150}"
: "${TEACHER_BUFFERS_MIB:=1200}"
: "${TEACHER_CARD_MIB:=24564}"

KEY_MIN_LEN=32
die() { echo "serve_teacher: $*" >&2; exit 1; }
problems=()
problem() { problems+=("$1"); }

file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }
file_mode() { stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1"; }
file_uid() { stat -c %u "$1" 2>/dev/null || stat -f %u "$1"; }
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
is_uint() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }

# projected MiB for $1 slots of $2 tokens each
projection() {
  local slots=$1 ctx=$2
  echo $(( TEACHER_WEIGHTS_MIB + (slots * ctx * TEACHER_KV_BYTES_PER_TOKEN + 1048575) / 1048576
           + slots * TEACHER_RS_MIB_PER_SLOT + TEACHER_BUFFERS_MIB ))
}

# the tailnet address to bind: TEACHER_BIND, else this host's on TEACHER_IFACE
bind_address() {
  if [ -n "$TEACHER_BIND" ]; then echo "$TEACHER_BIND"; return; fi
  ip -o -4 addr show dev "$TEACHER_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1
}

# refuses anything but an IPv4 in 100.64.0.0/10 held by TEACHER_IFACE on this host
check_bind() {
  local a=$1 o1 o2 o3 o4 extra
  case "$a" in
    '') problem "no tailnet address on $TEACHER_IFACE (is tailscale up?)"; return ;;
    0.0.0.0|::|'[::]'|localhost|127.*|::1) problem "bind $a refused: the teacher binds the tailnet address only"; return ;;
  esac
  IFS=. read -r o1 o2 o3 o4 extra <<<"$a"
  if [ -n "${extra:-}" ] || ! is_uint "$o1" || ! is_uint "$o2" || ! is_uint "$o3" || ! is_uint "$o4" \
      || [ "$o1" -gt 255 ] || [ "$o2" -gt 255 ] || [ "$o3" -gt 255 ] || [ "$o4" -gt 255 ]; then
    problem "bind $a refused: not an IPv4 address"; return
  fi
  if [ "$o1" -ne 100 ] || [ "$o2" -lt 64 ] || [ "$o2" -gt 127 ]; then
    problem "bind $a refused: not a tailnet address (100.64.0.0/10)"; return
  fi
  if ! ip -o -4 addr show dev "$TEACHER_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -qxF "$a"; then
    problem "bind $a refused: $TEACHER_IFACE on this host does not hold it"
  fi
}

check_key() {
  local f=$TEACHER_KEY_FILE key
  if [ -L "$f" ]; then problem "$f is a symlink (refused)"; return; fi
  if [ ! -f "$f" ]; then problem "no key file $f (serve_teacher.sh keygen)"; return; fi
  [ "$(file_uid "$f")" = "$(id -u)" ] || problem "$f is not owned by this user"
  case "$(file_mode "$f")" in 600|400) ;; *) problem "$f has mode $(file_mode "$f"); a key file must be 0600" ;; esac
  key=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$f" | head -n 1 | tr -d '[:space:]')
  [ "${#key}" -ge "$KEY_MIN_LEN" ] || problem "$f holds no key of at least $KEY_MIN_LEN characters"
  key=
}

manifest_sha() {  # the sha256 the manifest records for the model file (empty when it has none)
  [ -f "$TEACHER_MANIFEST" ] || return 0
  grep -o '"sha256": *"[0-9a-f]\{64\}"' "$TEACHER_MANIFEST" | head -n 1 | grep -o '[0-9a-f]\{64\}' || true
}

check_model() {
  [ -x "$TEACHER_LLAMA_SERVER" ] || problem "no llama-server at $TEACHER_LLAMA_SERVER"
  if [ ! -f "$TEACHER_MODEL" ]; then problem "no model file $TEACHER_MODEL"; return; fi
  local size; size=$(file_size "$TEACHER_MODEL")
  [ "$size" = "$TEACHER_MODEL_BYTES" ] || problem "$TEACHER_MODEL is $size B, not the pinned $TEACHER_MODEL_BYTES"
  local m; m=$(manifest_sha)
  if [ -z "$m" ]; then problem "$TEACHER_MANIFEST records no sha256 for the model"
  elif [ "$m" != "$TEACHER_SHA256" ]; then problem "the manifest's sha256 ${m:0:12}... is not the pinned ${TEACHER_SHA256:0:12}..."
  fi
  if [ "${SERVE_TEACHER_VERIFY_SHA:-0}" = 1 ] && [ "$(sha256_of "$TEACHER_MODEL")" != "$TEACHER_SHA256" ]; then
    problem "$TEACHER_MODEL does not hash to the pinned sha256"
  fi
  if [ "$TEACHER_TEMPLATE" != embedded ]; then
    if [ ! -f "$TEACHER_TEMPLATE" ]; then problem "no chat template $TEACHER_TEMPLATE (copy the base template; or TEACHER_TEMPLATE=embedded)"
    elif [ "$(sha256_of "$TEACHER_TEMPLATE")" != "$TEACHER_TEMPLATE_SHA256" ]; then problem "$TEACHER_TEMPLATE is not the pinned base template"
    fi
  fi
}

check_shape() {
  is_uint "$TEACHER_SLOTS" && [ "$TEACHER_SLOTS" -ge 1 ] && [ "$TEACHER_SLOTS" -le 8 ] \
    || problem "TEACHER_SLOTS=$TEACHER_SLOTS: 1 .. 8"
  is_uint "$TEACHER_CTX_PER_SLOT" && [ "$TEACHER_CTX_PER_SLOT" -ge 8192 ] && [ $(( TEACHER_CTX_PER_SLOT % 1024 )) -eq 0 ] \
    || problem "TEACHER_CTX_PER_SLOT=$TEACHER_CTX_PER_SLOT: a multiple of 1024, at least 8192"
  is_uint "$TEACHER_PORT" && [ "$TEACHER_PORT" -ne 8080 ] || problem "TEACHER_PORT=$TEACHER_PORT (never the business port 8080)"
}

check_live() {
  if systemctl --user is-active --quiet "$TEACHER_BUSINESS_UNIT" 2>/dev/null; then
    problem "$TEACHER_BUSINESS_UNIT is active: the window machinery stops the business server first"
  fi
  local free need
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
  if is_uint "$TEACHER_SLOTS" && is_uint "$TEACHER_CTX_PER_SLOT"; then
    need=$(( $(projection "$TEACHER_SLOTS" "$TEACHER_CTX_PER_SLOT") + TEACHER_MARGIN_MIB ))
    if ! is_uint "${free:-}"; then problem "nvidia-smi did not report free VRAM"
    elif [ "$free" -lt "$need" ]; then problem "only $free MiB VRAM free, needs $need (projection + $TEACHER_MARGIN_MIB)"
    fi
  fi
}

run_checks() {
  check_shape
  check_bind "$(bind_address)"
  check_key
  check_model
  check_live
}

server_argv() {
  local bind; bind=$(bind_address)
  ARGV=("$TEACHER_LLAMA_SERVER" -m "$TEACHER_MODEL" --no-mmproj --alias "$TEACHER_ALIAS"
        --host "$bind" --port "$TEACHER_PORT" --api-key-file "$TEACHER_KEY_FILE" --no-slots
        -ngl 99 --fit off -c $(( TEACHER_SLOTS * TEACHER_CTX_PER_SLOT )) --parallel "$TEACHER_SLOTS" --no-kv-unified
        --flash-attn on -ctk q8_0 -ctv q8_0 --jinja --batch-size 2048 --ubatch-size 512
        --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0
        --reasoning-budget 2048 --reasoning-budget-message "[REASONING_BUDGET_EXHAUSTED]")
  if [ "$TEACHER_TEMPLATE" != embedded ]; then ARGV+=(--chat-template-file "$TEACHER_TEMPLATE"); fi
}

report_problems() {
  local p
  for p in ${problems[@]+"${problems[@]}"}; do echo "serve_teacher: REFUSED: $p" >&2; done
}

cmd="${1:-}"
case "$cmd" in
  plan)
    echo "bind: $(bind_address || true):$TEACHER_PORT on $TEACHER_IFACE (tailnet only)"
    for s in $(( TEACHER_SLOTS > 1 ? TEACHER_SLOTS - 1 : 1 )) "$TEACHER_SLOTS" $(( TEACHER_SLOTS + 1 )); do
      p=$(projection "$s" "$TEACHER_CTX_PER_SLOT")
      echo "slots $s x ctx $TEACHER_CTX_PER_SLOT: projected $p MiB of $TEACHER_CARD_MIB (spare $(( TEACHER_CARD_MIB - p )))"
    done
    server_argv
    printf 'argv:'; printf ' %q' "${ARGV[@]}"; echo
    ;;
  keygen)
    umask 077
    dir=$(dirname "$TEACHER_KEY_FILE")
    mkdir -p "$dir"; chmod 700 "$dir"
    if [ -e "$TEACHER_KEY_FILE" ] || [ -L "$TEACHER_KEY_FILE" ]; then
      echo "serve_teacher: $TEACHER_KEY_FILE exists; kept (fingerprint $(sha256_of "$TEACHER_KEY_FILE" | cut -c1-12))"
    else
      tmp=$(mktemp "$dir/.teacher-key.XXXXXX")
      od -An -tx1 -N32 /dev/urandom | tr -d ' \n' >"$tmp"
      echo >>"$tmp"
      chmod 600 "$tmp"
      mv -f "$tmp" "$TEACHER_KEY_FILE"
      echo "serve_teacher: wrote $TEACHER_KEY_FILE (0600; fingerprint $(sha256_of "$TEACHER_KEY_FILE" | cut -c1-12))"
    fi
    ;;
  check)
    run_checks
    if [ "${#problems[@]}" -gt 0 ]; then report_problems; exit 1; fi
    echo "serve_teacher: OK: $(bind_address):$TEACHER_PORT, $TEACHER_SLOTS slots x $TEACHER_CTX_PER_SLOT," \
         "projected $(projection "$TEACHER_SLOTS" "$TEACHER_CTX_PER_SLOT") MiB"
    ;;
  exec)
    run_checks
    if [ "${#problems[@]}" -gt 0 ]; then report_problems; exit 1; fi
    server_argv
    if [ "${SERVE_TEACHER_DRY:-0}" = 1 ]; then printf '%q ' "${ARGV[@]}"; echo; exit 0; fi
    echo "serve_teacher: starting on $(bind_address):$TEACHER_PORT ($TEACHER_SLOTS slots x $TEACHER_CTX_PER_SLOT)" >&2
    exec "${ARGV[@]}"
    ;;
  *)
    echo "usage: serve_teacher.sh {plan|keygen|check|exec}" >&2
    exit 64
    ;;
esac
