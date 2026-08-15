#!/usr/bin/env bash
#
# Install local guards into the DEPLOY CLONE — the ff-only tree the daemon
# runs from (default ~/prometheus-deploy).
#
# Git hooks live in .git/hooks/, which is NOT tracked, so they do not survive
# recreating the clone. Re-run this after any fresh clone.
#
#   scripts/install-deploy-guards.sh [path-to-deploy-clone]
#
# Idempotent: overwrites the hook with the current version each run.

set -euo pipefail

DEPLOY_CLONE="${1:-$HOME/prometheus-deploy}"

if [ ! -d "$DEPLOY_CLONE/.git" ]; then
    echo "error: $DEPLOY_CLONE is not a git checkout" >&2
    echo "usage: $0 [path-to-deploy-clone]" >&2
    exit 1
fi

# Refuse to arm the guard against the dev checkout by mistake — that would
# block the very place work is supposed to happen.
if [ "$(cd "$DEPLOY_CLONE" && pwd -P)" = "$(cd "$(dirname "$0")/.." && pwd -P)" ]; then
    echo "error: refusing to install deploy guards into the DEV checkout." >&2
    echo "       $DEPLOY_CLONE is where you are meant to commit." >&2
    exit 1
fi

HOOK="$DEPLOY_CLONE/.git/hooks/pre-commit"

cat > "$HOOK" <<'HOOK_BODY'
#!/usr/bin/env bash
#
# DEPLOY CLONE — COMMITS ARE REFUSED HERE.
#
# Installed by scripts/install-deploy-guards.sh in the Prometheus repo.
# The daemon's ExecStartPre guard already refuses to boot when this clone
# diverges from origin/main — but that is silent until someone restarts,
# so the failure lands on whoever restarts next rather than whoever caused
# it. This hook fires at the moment the mistake is made instead.

cat <<'MSG' >&2

=== COMMIT REFUSED — this is the deploy clone ===

  This tree is what the DAEMON RUNS. It must stay a clean, ff-only mirror
  of origin/main. Committing here diverges it, which makes the daemon
  refuse to boot on the next restart.

  Work in the dev checkout instead:

      cd ~/Prometheus && git checkout -b <branch>

  Already made changes here by mistake? Move them, do not delete them:

      # a) working tree only
      git -C ~/prometheus-deploy diff > /tmp/wip.patch
      git -C ~/prometheus-deploy checkout -- .
      cd ~/Prometheus && git checkout -b <branch> && git apply /tmp/wip.patch

      # b) already committed here (push it somewhere safe FIRST)
      git -C ~/prometheus-deploy push origin HEAD:refs/heads/<branch>
      git -C ~/prometheus-deploy reset --hard origin/main

  To update this clone (the ONLY operation it should ever see):

      git -C ~/prometheus-deploy fetch origin \
        && git -C ~/prometheus-deploy merge --ff-only origin/main

MSG
exit 1
HOOK_BODY

chmod +x "$HOOK"

echo "Installed commit guard: $HOOK"
echo
echo "Verifying it refuses..."
if ( cd "$DEPLOY_CLONE" && git commit --allow-empty -m "guard self-test" >/dev/null 2>&1 ); then
    echo "  FAILED — the guard did not block an empty commit." >&2
    exit 1
fi
echo "  OK — commits in $DEPLOY_CLONE are refused."
