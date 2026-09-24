#!/usr/bin/env bash
# Deploy a reviewed commit to the daemon, running it from a venv built from
# THAT commit's uv.lock — so the lock, and pip-audit over it, describe what
# actually runs.
#
# WHY THIS EXISTS
# ---------------
# Until 0.9.2 the daemon ran `/usr/bin/env python3` with the deploy clone's
# src on PYTHONPATH, so every third-party package came from the user site
# (~/.local), installed by hand. Neither uv.lock nor CI's audit described it:
# on 2026-09-24 it was running starlette 0.52.1, python-multipart 0.0.22 and
# anyio 4.12.1 — below the repo's own security floors — and starlette 0.52.1
# plus a bearer gate keyed on request.url meant a crafted Host header reached
# every /api and /v1 route without a token (CVE-2026-48710). The same user
# site also serves cron jobs and other services, so it is left alone here.
#
# USAGE
#   scripts/deploy.sh <ref> [--prepare-only]
#     <ref>           a tag or commit on origin/main, e.g. v0.9.2 — deploy
#                     exactly that, not whatever main is by then
#     --prepare-only  phase A only: build and gate the venv, touch nothing live
#
# SETTINGS (environment; defaults in brackets)
#   PROMETHEUS_DEPLOY_CLONE   deploy clone           [$HOME/prometheus-deploy]
#   PROMETHEUS_VENV_ROOT      where venvs live       [$HOME/prometheus-venvs]
#   PROMETHEUS_DEPLOY_PYTHON  interpreter            [/usr/bin/python3.12]
#   PROMETHEUS_DEPLOY_EXTRAS  extras to install      ["anthropic mcp push voice"]
#   PROMETHEUS_DEPLOY_UNIT    systemd --user unit    [prometheus.service]
#   PROMETHEUS_DEPLOY_API     the daemon's REST API  [http://127.0.0.1:8005]
#   PROMETHEUS_ENV_FILE       holds the API token    [$HOME/.config/prometheus/env]
#
# PHASE A — prepare. Nothing live changes: the code is taken with
# `git archive`, not by moving the deploy clone, because a running daemon
# lazy-imports modules and must not read half of the next release.
#   1. fetch; <ref> must be on origin/main and a fast-forward of HEAD
#   2. uv sync --locked --no-dev --no-install-project into $ROOT/<sha12>,
#      with the lock's sha256 recorded in <venv>/BUILT_FROM_UV_LOCK
#   3. gates — ANY failure stops the deploy before the daemon is touched:
#      G1 the venv's interpreter: user site off, nothing from ~/.local
#      G2 pip-audit over the exported lock for these extras: no advisories
#      G3 the target code imports in the new venv
#
# PHASE B — switch.
#   B0 record every session's model choice (a restart keeps local-backend
#      overrides; cloud ones like /claude live in memory and are lost)
#   B1 fast-forward the deploy clone to <ref>
#   B2 point $ROOT/current at the new venv ($ROOT/previous at the old one)
#   B3 write the unit drop-in: run the venv's python, set PROMETHEUS_VENV so
#      scripts/deploy_guard.sh refuses a venv built from another lock
#   B4 restart the unit
#   B5 verify: running from the venv, nothing from ~/.local, 401 without a
#      token, 401 for a crafted Host header, 200 with the token
#   B6 re-apply the recorded cloud model choices
#   If B5 fails the script stops with the rollback commands below.
#
# ROLLBACK
#   R1 runtime only (back to system python + user site):
#        rm ~/.config/systemd/user/<unit>.d/10-venv.conf
#        systemctl --user daemon-reload && systemctl --user restart <unit>
#   R3 the previous venv (only while the code still matches its lock):
#        ln -sfn "$(readlink $ROOT/previous)" $ROOT/current
#        systemctl --user restart <unit>
#   Code goes back by reverting on main and deploying that — the deploy
#   guard refuses a checkout that is not on main.

set -euo pipefail

CLONE="${PROMETHEUS_DEPLOY_CLONE:-$HOME/prometheus-deploy}"
ROOT="${PROMETHEUS_VENV_ROOT:-$HOME/prometheus-venvs}"
PY="${PROMETHEUS_DEPLOY_PYTHON:-/usr/bin/python3.12}"
EXTRAS="${PROMETHEUS_DEPLOY_EXTRAS-anthropic mcp push voice}"
UNIT="${PROMETHEUS_DEPLOY_UNIT:-prometheus.service}"
API="${PROMETHEUS_DEPLOY_API:-http://127.0.0.1:8005}"
ENV_FILE="${PROMETHEUS_ENV_FILE:-$HOME/.config/prometheus/env}"
DROPIN_DIR="$HOME/.config/systemd/user/$UNIT.d"
DROPIN="$DROPIN_DIR/10-venv.conf"

say() { printf 'deploy: %s\n' "$*" >&2; }
die() { printf 'deploy: STOP — %s\n' "$*" >&2; exit 1; }

usage() { sed -n '/^# USAGE/,/^# SETTINGS/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2; }

REF=""
PREPARE_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --prepare-only) PREPARE_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage; exit 2 ;;
        *) [ -z "$REF" ] || { usage; exit 2; }; REF="$arg" ;;
    esac
done
[ -n "$REF" ] || { usage; exit 2; }

UV="$(command -v uv || true)"
[ -n "$UV" ] || UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || die "uv not found"
[ -x "$PY" ] || die "interpreter $PY not found"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

extra_args=()
for e in $EXTRAS; do extra_args+=(--extra "$e"); done

# ── PHASE A — prepare ───────────────────────────────────────────────────────

git -C "$CLONE" fetch --quiet origin --tags
T="$(git -C "$CLONE" rev-parse --verify --quiet "$REF^{commit}")" \
    || die "$REF does not name a commit in $CLONE"
git -C "$CLONE" merge-base --is-ancestor "$T" refs/remotes/origin/main \
    || die "$REF (${T:0:12}) is not on origin/main — only reviewed code deploys"
HEAD_NOW="$(git -C "$CLONE" rev-parse HEAD)"
git -C "$CLONE" merge-base --is-ancestor "$HEAD_NOW" "$T" \
    || die "$CLONE HEAD ${HEAD_NOW:0:12} is not an ancestor of ${T:0:12} — not a fast-forward"

S="${T:0:12}"
SRC="$ROOT/src-$S"
VENV="$ROOT/$S"
say "target $REF = $T; venv $VENV; python $PY; extras: ${EXTRAS:-none}"

rm -rf "$SRC"
mkdir -p "$SRC"
git -C "$CLONE" archive "$T" pyproject.toml uv.lock README.md src | tar -x -C "$SRC"

(cd "$ROOT" && UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT="$VENV" \
    "$UV" sync --quiet --project "$SRC" --locked --no-dev --no-install-project \
    --python "$PY" ${extra_args[@]+"${extra_args[@]}"})
sha256_of "$SRC/uv.lock" > "$VENV/BUILT_FROM_UV_LOCK"
say "venv built from uv.lock $(cut -c1-12 "$VENV/BUILT_FROM_UV_LOCK")"

say "G1 interpreter"
"$VENV/bin/python" - <<'EOF' || die "G1 failed: the venv's interpreter is not isolated"
import importlib.metadata as md, site, sys
assert not site.ENABLE_USER_SITE, "user site is enabled"
leaks = [p for p in sys.path if "/.local/lib/" in p]
assert not leaks, f"user-site paths on sys.path: {leaks}"
print("deploy:   python", sys.version.split()[0], "prefix", sys.prefix, file=sys.stderr)
for d in ("fastapi", "starlette", "uvicorn", "python-multipart", "anyio"):
    print(f"deploy:   {d} {md.version(d)}", file=sys.stderr)
EOF

say "G2 pip-audit"
(cd "$ROOT" && "$UV" export --quiet --project "$SRC" --locked --no-dev --no-emit-project \
    --no-hashes ${extra_args[@]+"${extra_args[@]}"} > "$VENV/requirements.txt")
(cd "$ROOT" && "$UV" tool run --quiet pip-audit -r "$VENV/requirements.txt" \
    --no-deps --disable-pip --progress-spinner off) \
    || die "G2 failed: pip-audit found advisories in $REF's lock for these extras — the daemon was not touched"

say "G3 import smoke"
(cd /tmp && PYTHONPATH="$SRC/src" "$VENV/bin/python" -c \
    'import prometheus, prometheus.daemon, prometheus.web.server; print("deploy:   prometheus", prometheus.__version__, "imports", file=__import__("sys").stderr)') \
    || die "G3 failed: $REF does not import in the new venv"

say "phase A passed"
if [ "$PREPARE_ONLY" = 1 ]; then
    say "--prepare-only: nothing live was changed"
    exit 0
fi

# ── PHASE B — switch ────────────────────────────────────────────────────────

TOKEN="$(sed -n 's/^\(export \)\{0,1\}PROMETHEUS_API_TOKEN=//p' "$ENV_FILE" 2>/dev/null \
    | tail -1 | tr -d "\"'")"
[ -n "$TOKEN" ] || die "no PROMETHEUS_API_TOKEN in $ENV_FILE — cannot record or verify"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CHOICES="$ROOT/model-choices-$STAMP.json"

say "B0 record model choices -> $CHOICES"
API="$API" TOKEN="$TOKEN" python3 - "$CHOICES" <<'EOF' || say "B0: could not record (daemon down?) — continuing"
import json, os, sys, urllib.request
api, token = os.environ["API"], os.environ["TOKEN"]
def get(path):
    req = urllib.request.Request(api + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)
rows = []
for s in get("/api/sessions"):
    sid = s.get("session_id")
    if sid:
        m = get(f"/api/sessions/{urllib.request.quote(sid, safe='')}/model")
        rows.append({"session_id": sid, **{k: m.get(k) for k in ("key", "provider", "model", "backend", "is_default")}})
json.dump(rows, open(sys.argv[1], "w"), indent=1)
cloud = [r for r in rows if not r["is_default"] and not r["backend"]]
print(f"deploy:   {len(rows)} sessions, {len(cloud)} with a cloud model choice", file=sys.stderr)
EOF

say "B1 fast-forward $CLONE to ${T:0:12}"
[ "$(git -C "$CLONE" symbolic-ref --short -q HEAD)" = "main" ] || die "$CLONE is not on main"
git -C "$CLONE" diff-index --quiet HEAD -- || die "$CLONE has tracked changes"
git -C "$CLONE" merge --quiet --ff-only "$T"

say "B2 $ROOT/current -> $S"
if [ -L "$ROOT/current" ]; then
    ln -sfn "$(readlink "$ROOT/current")" "$ROOT/previous"
fi
ln -sfn "$S" "$ROOT/current.tmp"
mv -Tf "$ROOT/current.tmp" "$ROOT/current"

say "B3 drop-in $DROPIN"
mkdir -p "$DROPIN_DIR"
cat > "$DROPIN" <<EOF
# Written by scripts/deploy.sh — run the daemon from the venv built from the
# deployed commit's uv.lock. Remove this file to roll back to the base unit.
[Service]
ExecStart=
ExecStart=$ROOT/current/bin/python scripts/daemon.py --config config/prometheus.yaml
Environment=PROMETHEUS_VENV=$ROOT/current
Environment=PYTHONNOUSERSITE=1
EOF
systemctl --user daemon-reload

say "B4 restart $UNIT"
systemctl --user restart "$UNIT"

say "B5 verify"
fail() {
    say "VERIFY FAILED — $*"
    say "  R1 (runtime only): rm $DROPIN && systemctl --user daemon-reload && systemctl --user restart $UNIT"
    [ -L "$ROOT/previous" ] && say "  R3 (previous venv): ln -sfn \"\$(readlink $ROOT/previous)\" $ROOT/current && systemctl --user restart $UNIT"
    exit 1
}
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$@" || true; }
for _ in $(seq 1 60); do
    [ "$(code "$API/api/status")" != "000" ] && break
    sleep 1
done
systemctl --user is-active --quiet "$UNIT" || fail "$UNIT is not active"
PID="$(systemctl --user show "$UNIT" -p MainPID --value)"
grep -q "$ROOT/$S/" "/proc/$PID/maps" || fail "PID $PID has nothing mapped from $VENV"
! grep -q "/.local/lib/python" "/proc/$PID/maps" || fail "PID $PID still maps user-site modules"
[ "$(code "$API/api/status")" = "401" ] || fail "/api/status without a token did not answer 401"
[ "$(code -H 'Host: x/abc?' "$API/api/status")" = "401" ] || fail "a crafted Host header did not answer 401"
# the token goes to curl on stdin, not argv, so it never shows in ps
[ "$(printf 'Authorization: Bearer %s\n' "$TOKEN" | code -H @- "$API/api/status")" = "200" ] \
    || fail "/api/status with the token did not answer 200"
say "  running from $VENV, 401 bare, 401 crafted Host, 200 with token"

say "B6 re-apply cloud model choices"
if [ -f "$CHOICES" ]; then
    API="$API" TOKEN="$TOKEN" python3 - "$CHOICES" <<'EOF'
import json, os, sys, urllib.error, urllib.request
api, token = os.environ["API"], os.environ["TOKEN"]
for r in json.load(open(sys.argv[1])):
    if r["is_default"] or r["backend"]:
        continue  # the default needs nothing; backend overrides persist on their own
    sid = r["session_id"]
    if r["key"] == "custom":
        print(f"deploy:   {sid}: a config-custom override ({r['provider']}/{r['model']}) — re-run its slash command by hand", file=sys.stderr)
        continue
    req = urllib.request.Request(
        f"{api}/api/sessions/{urllib.request.quote(sid, safe='')}/model",
        data=json.dumps({"key": r["key"]}).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
        print(f"deploy:   {sid}: {r['key']} re-applied", file=sys.stderr)
    except urllib.error.HTTPError as exc:
        print(f"deploy:   {sid}: {r['key']} NOT re-applied — {exc.code} {exc.read()[:200]!r}", file=sys.stderr)
EOF
fi
say "done: $UNIT runs $REF (${T:0:12}) from $VENV"
