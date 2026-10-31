#!/usr/bin/env bash
# serve_teacher.sh {plan|keygen|check|exec|verify}   (the 4090 tower; installed as ~/distill/teacher-serve/serve_teacher.sh)
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
#   verify  the unit's ExecStartPost: waits for /health (up to TEACHER_VERIFY_WAIT_S), then checks the RUNNING
#           server: the key enforced on /props, ours accepted, the slots as asked, the MCP proxy, web UI and
#           extra endpoints off in /props, and GET /tools refused (403: no built-in tool and no MCP server).
#           Any failure exits 1, so systemd fails the unit and stops llama-server. The key is read by python
#           from the key file and goes only into the Authorization header (never argv or output).
#
# The same server is also the local 27B JUDGE for judge.py --provider local --judge-url (the 3090 Ti host
# reaches it with the same key): one model, one shape, nothing else changes.
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
# * --no-slots: /slots would expose per-request state. No --props (POST /props stays off), no --metrics,
#   --no-webui (nothing here needs the web UI's public pages).
# * llama-server's built-in tools (read_file, write_file, edit_file, exec_shell_command, ...), its agent mode,
#   MCP servers and the web UI's MCP proxy would run UNJAILED on this host for anyone holding the key. This
#   build (9d57ce456, common/arg.cpp) enables them from three places, each closed here:
#   - the command line: `--no-agent` (clears the tool list and the MCP proxy), `--no-ui-mcp-proxy`,
#     `--mcp-servers-config ''` and `--mcp-servers-json ''`. llama.cpp applies the command line LAST, after
#     its config files and its environment, so these win over both;
#   - the environment (LLAMA_ARG_TOOLS=all, LLAMA_ARG_AGENT=1, LLAMA_ARG_MCP_SERVERS_*, LLAMA_ARG_TOOLS_RUNTIME,
#     LLAMA_ARG_UI_MCP_PROXY; LLAMA_API_KEY, which has no command-line override and would ADD a second key
#     beside the 0600 file; LLAMA_ARG_STATIC_PATH, LLAMA_ARG_MEDIA_PATH, LLAMA_ARG_ENDPOINT_*; AIP_* moves the
#     port and the routes): every LLAMA_*, AIP_* and HF_* variable is unset at the top (`scrub_env`; the unit
#     inherits the user manager's environment, so this does not rely on it being clean) and `exec` starts
#     llama-server with `env -i` and only SERVER_ENV_KEEP (HOME, PATH, the locale, CUDA's device selection)
#     plus XDG_CONFIG_HOME. `check` names (never prints the values of) what it cleared;
#   - config files (/etc/llama.cpp/config.ini, then $XDG_CONFIG_HOME/llama.cpp/config.ini; they can set the
#     same options, api-key included): XDG_CONFIG_HOME is TEACHER_CONFIG_HOME, an empty directory of ours,
#     and the check refuses while a system-wide config (TEACHER_SYSTEM_LLAMA_CONFIG) or anything under
#     TEACHER_CONFIG_HOME/llama.cpp exists.
#   Then `check_no_tools` REFUSES to start (check and exec) unless the final argv carries every off switch
#   (TOOL_OFF_FLAGS, and both MCP options present and empty) and none of TOOL_ON_FLAGS (tools, agent, MCP proxy,
#   web UI, an inline key, /props POST, /metrics, /slots, a static or media path), and the environment it
#   passes holds no LLAMA_*/AIP_*/HF_* variable. A later edit of server_argv that turns any of it on fails
#   here, not on the host. After the start, `verify` (ExecStartPost) checks the running server the same way.
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
# VRAM plan (24,564 MiB card, of which the driver reserves 491 (nvidia-smi memory.reserved, never shown as
# free): 24,073 MiB usable, nothing else resident once the business server is stopped):
#   weights on the GPU   16,400.6 MiB  (17,082.6 MiB of tensors - 682.0 MiB token embeddings, which llama.cpp
#                                       keeps on the CPU; incl. the Q6_K output 994.6 and the MTP block 272.6)
#   KV, q8_0             34,816 B/token (16 attention layers x 4 KV heads x 256 x K+V x 1.0625 B; measured on
#                                       the 3090 Ti's teacher, same architecture and build; +6.25% if the MTP
#                                       layer's attention got a cache too: 17 layers)
#   recurrent state      149.6 MiB per slot (48 gated-delta-net layers, f32; measured on the 3090 Ti)
#   compute + context    1,200 MiB     (the 3090 Ti's teacher measured 1,191.9 at -ub 512 WITH a vision encoder)
#   4 slots x 32,768:    16,400.6 + 4,352.0 + 598.5 + 1,200 = 22,551 MiB -> ~1,520 MiB of the usable 24,073
#                        spare (~1,250 at 17 layers)
#   5 slots x 32,768:    16,400.6 + 5,440.0 + 748.1 + 1,200 = 23,789 MiB: with TEACHER_MARGIN_MIB it needs 24,391
#                        free, more than this card ever reports: it does NOT fit (the check refuses it). Five
#                        slots fit only with less context each: 5 x 24,576 = 22,431 MiB (~1,640 spare).
# (`plan` and the check count whole MiB, rounded up: 22,553 for 4 slots, so the check wants 23,153 free; `plan`
# reports spare against the usable VRAM: nvidia-smi's total - reserved, else TEACHER_USABLE_MIB.)
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
: "${TEACHER_USABLE_MIB:=24073}"   # 24,564 total - 491 reserved by the driver (the 4090, 2026-10-31)
# llama.cpp's config files (header: Security)
: "${TEACHER_CONFIG_HOME:=$HOME/distill/teacher-serve/llama-config-empty}"
: "${TEACHER_SYSTEM_LLAMA_CONFIG:=/etc/llama.cpp/config.ini}"
# verify (ExecStartPost): how long the model may take to load, and the python that checks the running server
: "${TEACHER_VERIFY_WAIT_S:=420}"
: "${TEACHER_PYTHON:=python3}"

KEY_MIN_LEN=32
die() { echo "serve_teacher: $*" >&2; exit 1; }
problems=()
problem() { problems+=("$1"); }

# llama-server's own environment, cleared before anything else (header: Security). Names only are kept.
SCRUBBED=()
scrub_env() {
  local v
  for v in $(compgen -e); do
    case "$v" in
      LLAMA_*|AIP_*|HF_*|HUGGING_FACE_*) SCRUBBED+=("$v"); unset "$v" ;;
    esac
  done
}
scrub_env
# the only variables llama-server is started with (`env -i`), besides XDG_CONFIG_HOME
SERVER_ENV_KEEP=(HOME PATH LANG LC_ALL TZ USER LOGNAME CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER LD_LIBRARY_PATH)
# llama-server's off switches the argv must carry, and the options it must never carry (this build: arg.cpp)
TOOL_OFF_FLAGS=(--no-agent --no-ui-mcp-proxy --no-webui --no-slots)
TOOL_ON_FLAGS=(--tools --tools-runtime -ag --agent --ui-mcp-proxy --webui-mcp-proxy --ui --webui
               --api-key --props --metrics --slots --path --media-path --api-prefix --rpc)

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
  # never a silent exit under set -e (no such interface, say): an empty answer is refused by check_bind
  ip -o -4 addr show dev "$TEACHER_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1 || true
}

# the VRAM llama-server can use: nvidia-smi's memory.total - memory.reserved, else TEACHER_USABLE_MIB
usable_mib() {
  local line total reserved
  line=$(nvidia-smi --query-gpu=memory.total,memory.reserved --format=csv,noheader,nounits 2>/dev/null \
         | head -n 1 | tr -d ' ') || line=
  total=${line%%,*}; reserved=${line#*,}
  if [ -n "$line" ] && is_uint "$total" && is_uint "$reserved" && [ "$total" -gt "$reserved" ]; then
    echo $(( total - reserved ))
  else
    echo "$TEACHER_USABLE_MIB"
  fi
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
  # a file of comments only makes grep exit 1: reported below, never a silent exit under set -e
  key=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$f" | head -n 1 | tr -d '[:space:]') || key=
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
  local free need usable
  # a failing nvidia-smi (a driver hiccup right after the business server stops) is reported below, never a
  # silent exit under set -e
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ') || free=
  if is_uint "$TEACHER_SLOTS" && is_uint "$TEACHER_CTX_PER_SLOT"; then
    need=$(( $(projection "$TEACHER_SLOTS" "$TEACHER_CTX_PER_SLOT") + TEACHER_MARGIN_MIB ))
    usable=$(usable_mib)
    if [ "$need" -gt "$usable" ]; then
      problem "$TEACHER_SLOTS slots x $TEACHER_CTX_PER_SLOT need $need MiB (projection + $TEACHER_MARGIN_MIB), more than this card's usable $usable: fewer slots or less context per slot"
    elif ! is_uint "${free:-}"; then problem "nvidia-smi did not report free VRAM"
    elif [ "$free" -lt "$need" ]; then problem "only $free MiB VRAM free, needs $need (projection + $TEACHER_MARGIN_MIB)"
    fi
  fi
}

# llama.cpp reads config.ini from /etc/llama.cpp and from $XDG_CONFIG_HOME/llama.cpp: neither may exist
check_config() {
  if [ -e "$TEACHER_SYSTEM_LLAMA_CONFIG" ] || [ -L "$TEACHER_SYSTEM_LLAMA_CONFIG" ]; then
    problem "$TEACHER_SYSTEM_LLAMA_CONFIG exists: llama.cpp's system-wide config could enable tools or add a key"
  fi
  if [ -e "$TEACHER_CONFIG_HOME/llama.cpp" ] || [ -L "$TEACHER_CONFIG_HOME/llama.cpp" ]; then
    problem "$TEACHER_CONFIG_HOME/llama.cpp exists: the teacher's config directory must stay empty"
  fi
}

run_checks() {
  check_config
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
        --no-webui --no-agent --no-ui-mcp-proxy --mcp-servers-config '' --mcp-servers-json ''
        --flash-attn on -ctk q8_0 -ctv q8_0 --jinja --batch-size 2048 --ubatch-size 512
        --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0
        --reasoning-budget 2048 --reasoning-budget-message "[REASONING_BUDGET_EXHAUSTED]")
  if [ "$TEACHER_TEMPLATE" != embedded ]; then ARGV+=(--chat-template-file "$TEACHER_TEMPLATE"); fi
}

# the whole environment llama-server is started with (`env -i`): XDG_CONFIG_HOME + SERVER_ENV_KEEP
server_env() {
  local v
  ENVV=(XDG_CONFIG_HOME="$TEACHER_CONFIG_HOME")
  for v in "${SERVER_ENV_KEEP[@]}"; do
    if [ -n "${!v:-}" ]; then ENVV+=("$v=${!v}"); fi
  done
}

# REFUSES (a problem) an argv or environment that could turn on llama-server's built-in tools, agent mode, MCP
# servers, the MCP proxy, the web UI, a second key or an extra endpoint (header: Security). Runs on the final
# ARGV and ENVV, so a later edit of server_argv or SERVER_ENV_KEEP cannot slip one in.
check_no_tools() {
  local a f i n=${#ARGV[@]} found name
  for f in "${TOOL_OFF_FLAGS[@]}"; do
    found=0
    for a in "${ARGV[@]}"; do if [ "$a" = "$f" ]; then found=1; fi; done
    [ "$found" = 1 ] || problem "llama-server's argv lacks $f: its tools, agent, MCP and web UI are turned off explicitly"
  done
  for f in --mcp-servers-config --mcp-servers-json; do
    found=0
    i=0
    while [ "$i" -lt "$n" ]; do
      if [ "${ARGV[$i]}" = "$f" ]; then
        found=1
        if [ -n "${ARGV[$((i + 1))]:-}" ]; then problem "llama-server's argv sets $f: MCP servers stay off"; fi
      fi
      i=$((i + 1))
    done
    [ "$found" = 1 ] || problem "llama-server's argv lacks $f '' (MCP servers are turned off explicitly)"
  done
  for a in "${ARGV[@]}"; do
    for f in "${TOOL_ON_FLAGS[@]}"; do
      if [ "$a" = "$f" ] || [ "${a#"$f"=}" != "$a" ]; then
        problem "llama-server's argv holds $f: built-in tools, agent, MCP, web UI, extra keys and endpoints stay off"
      fi
    done
    case "$a" in
      --mcp-servers-config=?*|--mcp-servers-json=?*) problem "llama-server's argv sets ${a%%=*}: MCP servers stay off" ;;
    esac
  done
  for a in "${ENVV[@]}"; do
    name=${a%%=*}
    case "$name" in
      LLAMA_*|AIP_*|HF_*|HUGGING_FACE_*) problem "llama-server's environment would hold $name (it is never passed)" ;;
    esac
  done
}

# the running server, checked from this host (the unit's ExecStartPost): exit 1 on any problem
verify_running() {
  local bind; bind=$(bind_address)
  if [ -z "$bind" ]; then echo "serve_teacher: VERIFY FAILED: no tailnet address on $TEACHER_IFACE" >&2; return 1; fi
  "$TEACHER_PYTHON" -I - "http://$bind:$TEACHER_PORT" "$TEACHER_KEY_FILE" "$TEACHER_SLOTS" "$TEACHER_VERIFY_WAIT_S" <<'PY'
import json, sys, time, urllib.error, urllib.request

base, key_path, slots, wait_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy


def get(path, key=None):
    headers = {"Authorization": "Bearer " + key} if key else {}
    try:
        with opener.open(urllib.request.Request(base + path, headers=headers), timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception:  # refused, reset, timeout: not up yet
        return None, b""


with open(key_path, encoding="ascii") as fh:
    key = next((ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")), "")
end = time.monotonic() + wait_s
while True:
    code, _ = get("/health")
    if code == 200:
        break
    if time.monotonic() >= end:
        print(f"serve_teacher: VERIFY FAILED: /health did not answer 200 within {wait_s:.0f} s (last: {code})",
              file=sys.stderr)
        sys.exit(1)
    time.sleep(2)
problems = []
code, _ = get("/props")
if code not in (401, 403):
    problems.append(f"/props answers without the key (HTTP {code}): the key is not enforced")
code, body = get("/props", key)
props = {}
if code != 200:
    problems.append(f"/props refused this host's own key (HTTP {code})")
else:
    try:
        props = json.loads(body)
    except ValueError:
        problems.append("/props is not JSON")
for name in ("cors_proxy_enabled", "ui", "endpoint_slots", "endpoint_props", "endpoint_metrics"):
    if props.get(name) is True:
        problems.append(f"/props says {name}=true: it must be off")
if props and props.get("total_slots") != slots:
    problems.append(f"/props says {props.get('total_slots')} slots, not the {slots} asked for")
code, _ = get("/tools", key)
if code == 200:
    problems.append("GET /tools answers 200: llama-server's built-in tools or MCP servers are ON")
elif code not in (403, 404):
    problems.append(f"GET /tools answered HTTP {code} (403 expected: no tools)")
for p in problems:
    print(f"serve_teacher: VERIFY FAILED: {p}", file=sys.stderr)
if problems:
    sys.exit(1)
print(f"serve_teacher: verified: key enforced, {slots} slots, no tools, no MCP, no web UI")
PY
}

report_problems() {
  local p
  for p in ${problems[@]+"${problems[@]}"}; do echo "serve_teacher: REFUSED: $p" >&2; done
}

report_scrubbed() {
  if [ "${#SCRUBBED[@]}" -gt 0 ]; then
    echo "serve_teacher: NOTE: cleared from llama-server's environment (never applied): ${SCRUBBED[*]}" >&2
  fi
}

cmd="${1:-}"
case "$cmd" in
  plan)
    report_scrubbed
    echo "bind: $(bind_address || true):$TEACHER_PORT on $TEACHER_IFACE (tailnet only)"
    usable=$(usable_mib)
    for s in $(( TEACHER_SLOTS > 1 ? TEACHER_SLOTS - 1 : 1 )) "$TEACHER_SLOTS" $(( TEACHER_SLOTS + 1 )); do
      p=$(projection "$s" "$TEACHER_CTX_PER_SLOT")
      if [ $(( p + TEACHER_MARGIN_MIB )) -le "$usable" ]; then fits=fits; else fits="does NOT fit (the check refuses it)"; fi
      echo "slots $s x ctx $TEACHER_CTX_PER_SLOT: projected $p MiB of $usable usable (spare $(( usable - p ))): $fits"
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
    report_scrubbed
    run_checks
    server_argv
    server_env
    check_no_tools
    if [ "${#problems[@]}" -gt 0 ]; then report_problems; exit 1; fi
    echo "serve_teacher: OK: $(bind_address):$TEACHER_PORT, $TEACHER_SLOTS slots x $TEACHER_CTX_PER_SLOT," \
         "projected $(projection "$TEACHER_SLOTS" "$TEACHER_CTX_PER_SLOT") MiB"
    ;;
  exec)
    report_scrubbed
    run_checks
    server_argv
    server_env
    check_no_tools
    if [ "${#problems[@]}" -gt 0 ]; then report_problems; exit 1; fi
    if [ "${SERVE_TEACHER_DRY:-0}" = 1 ]; then printf '%q ' "${ARGV[@]}"; echo; exit 0; fi
    mkdir -p "$TEACHER_CONFIG_HOME"; chmod 700 "$TEACHER_CONFIG_HOME"
    echo "serve_teacher: starting on $(bind_address):$TEACHER_PORT ($TEACHER_SLOTS slots x $TEACHER_CTX_PER_SLOT)" >&2
    exec env -i "${ENVV[@]}" "${ARGV[@]}"
    ;;
  verify)
    verify_running
    ;;
  *)
    echo "usage: serve_teacher.sh {plan|keygen|check|exec|verify}" >&2
    exit 64
    ;;
esac
