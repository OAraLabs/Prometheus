#!/usr/bin/env bash
# Put the `oara` command on PATH for a deploy that does not install the package.
#
# WHY THIS EXISTS
# ---------------
# pip, pipx, uv tool and Homebrew install the package, and the package's entry point (`oara`) comes with it. The deploy
# in scripts/deploy.sh does not: it builds a venv with `uv sync --no-install-project` and runs the code from a git
# checkout, so the venv has no `oara` console script and nothing else puts one on PATH. On the machine that runs the
# daemon, the owner step for a second computer (`oara pair --owner`) was "command not found", and so was `oara doctor`.
#
# WHAT IT WRITES
# --------------
# One small launcher, <bin-dir>/oara (default ~/.local/bin, which a login shell has on PATH on the usual Linux
# setups). It runs the deployed venv's interpreter on the deployed code with the service's own PROMETHEUS_* settings, so
# the command sees the same config, data and logs as the daemon. Point <venv> at the `current` link deploy.sh moves and
# the command follows every deploy and every rollback with no second step.
#
#   scripts/install_oara_command.sh [--bin-dir DIR] [--unit NAME] [--env NAME=VALUE]... <venv> <src-dir>
#
#     <venv>      the deployed venv (normally $ROOT/current): its bin/python is what runs
#     <src-dir>   the directory that holds the `prometheus` package (normally <clone>/src)
#     --bin-dir   where to write the launcher                     [$PROMETHEUS_BIN_DIR or ~/.local/bin]
#     --unit      a systemd --user unit whose PROMETHEUS_* settings the command should share (deploy.sh passes it)
#     --env       one more PROMETHEUS_* setting to bake in (may repeat)
#
# SAFETY
#   * It never overwrites an `oara` it did not write (a real entry point, a link, anything without its marker).
#   * It never writes a secret: a setting whose name has TOKEN, SECRET, KEY or PASSWORD in it is refused. The API
#     token stays in the env file, which the command reads itself.
#   * It writes only inside <bin-dir>, never asks for root, and replaces its own launcher atomically.
#   * Exit 0 installed or left alone, 1 the venv or code is not usable, 2 bad usage, 3 installed but did not run.
set -euo pipefail

MARK='prometheus oara launcher v1'
BIN="${PROMETHEUS_BIN_DIR:-$HOME/.local/bin}"
UNIT=""
EXTRA=()

say() { printf 'install_oara_command: %s\n' "$*"; }
die() { printf 'install_oara_command: %s\n' "$*" >&2; exit "${2:-1}"; }
usage() { sed -n '/^#   scripts\/install_oara_command.sh/,/^#     --env/p' "$0" | sed 's/^# \{0,1\}//' >&2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir) [ $# -ge 2 ] || { usage; exit 2; }; BIN="$2"; shift 2 ;;
        --unit)    [ $# -ge 2 ] || { usage; exit 2; }; UNIT="$2"; shift 2 ;;
        --env)     [ $# -ge 2 ] || { usage; exit 2; }; EXTRA+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; break ;;
        -*)        usage; exit 2 ;;
        *)         break ;;
    esac
done
[ $# -eq 2 ] || { usage; exit 2; }
VENV="$1"
SRC="$2"

# A setting is PROMETHEUS_<NAME>=<plain value>. The value is plain on purpose: it is written into a script, so nothing
# that a shell would treat as syntax gets in, and nothing that looks like a secret.
SETTING_RE='^PROMETHEUS_[A-Z0-9_]+=[A-Za-z0-9_./:@%+,=-]*$'
secret_name() { case "${1%%=*}" in *TOKEN*|*SECRET*|*KEY*|*PASSWORD*|*PASSPHRASE*) return 0 ;; esac; return 1; }

# POSIX single-quoting, so a path with a space or a quote in it survives being written into a script.
sq() { local s=${1//\'/\'\\\'\'}; printf "'%s'" "$s"; }

case "$VENV" in /*) ;; *) die "<venv> must be an absolute path: $VENV" ;; esac
case "$SRC" in /*) ;; *) die "<src-dir> must be an absolute path: $SRC" ;; esac
[ -x "$VENV/bin/python" ] || die "$VENV/bin/python is not an executable file; nothing was written" 1
[ -f "$SRC/prometheus/__init__.py" ] || die "$SRC has no prometheus package; nothing was written" 1

# The service's own settings, when it is asked for and systemctl can say. Only PROMETHEUS_* ones, never the venv (this
# launcher's is the one that matters) and never a secret.
SETTINGS=()
if [ -n "$UNIT" ] && command -v systemctl >/dev/null 2>&1; then
    raw="$(systemctl --user show -p Environment --value "$UNIT" 2>/dev/null || true)"
    set -f
    for token in $raw; do
        [[ "$token" =~ $SETTING_RE ]] || continue
        [ "${token%%=*}" = "PROMETHEUS_VENV" ] && continue
        secret_name "$token" && continue
        SETTINGS+=("$token")
    done
    set +f
fi
for token in ${EXTRA[@]+"${EXTRA[@]}"}; do
    [[ "$token" =~ $SETTING_RE ]] || die "refused --env '$token': it must be PROMETHEUS_NAME=plain-value" 2
    secret_name "$token" && die "refused --env '${token%%=*}': a launcher is world-readable, so no secret goes in it" 2
    SETTINGS+=("$token")
done

TARGET="$BIN/oara"
if [ -L "$TARGET" ]; then
    say "$TARGET is a link to somewhere else, so I left it alone"
    exit 0
fi
if [ -e "$TARGET" ] && ! grep -q "$MARK" "$TARGET" 2>/dev/null; then
    say "$TARGET exists and I did not write it, so I left it alone"
    exit 0
fi

mkdir -p "$BIN"
TMP="$(mktemp "$BIN/.oara.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
{
    printf '#!/bin/sh\n'
    printf '# oara: the Prometheus command. Written by scripts/install_oara_command.sh (%s).\n' "$MARK"
    printf '# It runs the deployed venv'"'"'s interpreter on the deployed code with the service'"'"'s settings.\n'
    printf '# A deploy rewrites this file, so edit nothing here.\n'
    printf 'PROMETHEUS_VENV=%s\n' "$(sq "$VENV")"
    printf 'PROMETHEUS_SRC=%s\n' "$(sq "$SRC")"
    printf 'export PROMETHEUS_VENV\n'
    printf 'export PYTHONNOUSERSITE=1\n'
    printf 'PYTHONPATH="$PROMETHEUS_SRC${PYTHONPATH:+:$PYTHONPATH}"\n'
    printf 'export PYTHONPATH\n'
    for token in ${SETTINGS[@]+"${SETTINGS[@]}"}; do
        printf 'export %s=%s\n' "${token%%=*}" "$(sq "${token#*=}")"
    done
    printf 'exec "$PROMETHEUS_VENV/bin/python" -m prometheus "$@"\n'
} > "$TMP"
chmod 0755 "$TMP"
mv -f "$TMP" "$TARGET"
trap - EXIT
say "installed $TARGET (venv $VENV, code $SRC)"

case ":$PATH:" in
    *":$BIN:"*) ;;
    *)
        say "$BIN is not on your PATH, so a new shell will say 'oara: command not found'. Add it:"
        say "  echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.profile   # then open a new terminal"
        ;;
esac
other="$(command -v oara 2>/dev/null || true)"
if [ -n "$other" ] && [ "$other" != "$TARGET" ]; then
    say "note: $other comes first on PATH and will run instead of $TARGET"
fi

if ! "$TARGET" --version >/dev/null 2>&1; then
    die "installed $TARGET, but 'oara --version' did not run; check $VENV and $SRC" 3
fi
