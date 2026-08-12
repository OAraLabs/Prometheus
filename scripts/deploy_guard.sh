#!/usr/bin/env bash
# Deploy guard — refuse to boot the daemon from an unmerged checkout.
#
# WHY THIS EXISTS
# ---------------
# The daemon deploys from a checkout that Claude Code sessions share, so
# `tree_head` is whatever branch that checkout happens to be on. Three times
# in eleven days it has been left on a feature branch; on 2026-08-12 the live
# reading was running_sha 19e9db4 (main) vs tree_head d8a6115
# (feat/firstlight-fl2u-absence-defaults), stale: true — a restart at that
# moment would have deployed an unmerged branch to production, and nothing in
# the merge, the CI or the PR state would have hinted at it.
#
# Standing-Principles CROSS-CUTTING §7 has documented this since 2026-08-02,
# and §13 is why prose was not enough: "writing a trap down does not stop you
# walking into it. If the failure is mechanical, the countermeasure has to be
# mechanical." This is the mechanical form.
#
# WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
#   * HEAD is on `main`             — catches the actual incident.
#   * HEAD == local refs/remotes/origin/main — catches a main that has drifted
#     ahead with unpushed commits.
#   * It does NOT fetch. Boot is the wrong time for a network call: no
#     network, a hung proxy, or a slow remote would turn a safety check into
#     an outage. Comparing against the LOCAL origin/main ref is stale-tolerant
#     by design — the guard's job is "never deploy an unmerged branch", not
#     "always deploy the newest".
#   * It does NOT look at the working tree being dirty. Uncommitted local
#     edits to a deployed main are a different (and normal) situation.
#
# FAIL DIRECTION: any uncertainty — not a git repo, git missing, detached
# HEAD, unreadable refs — REFUSES. A guard that degrades to "boot anyway"
# when its detector breaks is not a guard (CROSS-CUTTING §8: fail-closed and
# fail-open are choices; fail-by-exception is the third state nobody chose).
#
# ESCAPE HATCH, DELIBERATE AND VISIBLE
# ------------------------------------
# PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1 skips the check and logs loudly that it
# did. A guard with no override is one people disable wholesale the first
# time it blocks something legitimate — the `--no-verify` reflex. An override
# that announces itself in the journal is strictly better than a guard that
# gets commented out.
#
# HOW IT IS WIRED (~/.config/systemd/user/prometheus.service)
# ----------------------------------------------------------
#   [Unit]
#   # ExecStartPre failures are start failures, and Restart=on-failure would
#   # otherwise retry the refusal every 10s forever. Give up and land in
#   # `failed` with the reason in the journal.
#   StartLimitIntervalSec=120
#   StartLimitBurst=3
#
#   [Service]
#   # NO leading `-`: a failure here MUST abort the start.
#   ExecStartPre=/usr/bin/env bash %h/Prometheus/scripts/deploy_guard.sh %h/Prometheus
#
# The unit is NOT in this repo, so a branch cannot quietly unwire itself. A
# branch that deletes this script fails ExecStartPre and refuses — the safe
# direction either way.
#
# Read the refusal with:  journalctl --user -u prometheus.service -n 30
set -uo pipefail

REPO="${1:-$HOME/Prometheus}"
GIT=$(command -v git || echo /usr/bin/git)

log() { printf 'deploy-guard: %s\n' "$*" >&2; }

if [ "${PROMETHEUS_ALLOW_UNMERGED_DEPLOY:-0}" = "1" ]; then
    log "OVERRIDE ACTIVE (PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1) — branch check skipped."
    log "OVERRIDE the daemon may be running code that is not on main."
    exit 0
fi

if [ ! -x "$GIT" ]; then
    log "REFUSING: git not found — cannot verify the checkout is on main."
    exit 1
fi

if ! "$GIT" -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    log "REFUSING: $REPO is not a git checkout — cannot verify what would deploy."
    exit 1
fi

branch=$("$GIT" -C "$REPO" symbolic-ref --short -q HEAD || true)
# --verify --quiet: a bare `rev-parse <missing-ref>` ECHOES THE REF NAME to
# stdout and exits non-zero, so `|| true` would leave the literal string
# "refs/remotes/origin/main" in the variable and the wrong branch below
# would fire. Caught by scenario 7 in the harness — it still REFUSED, but
# with the wrong diagnosis, which is its own defect (§2b).
head=$("$GIT" -C "$REPO" rev-parse --verify --quiet HEAD || true)
origin=$("$GIT" -C "$REPO" rev-parse --verify --quiet refs/remotes/origin/main || true)

if [ -z "$head" ]; then
    log "REFUSING: cannot resolve HEAD in $REPO."
    exit 1
fi

if [ -z "$branch" ]; then
    log "REFUSING: $REPO is in DETACHED HEAD at ${head:0:7}."
    log "  A detached checkout is nobody's branch — fix with: git -C $REPO checkout main"
    exit 1
fi

if [ "$branch" != "main" ]; then
    log "REFUSING: $REPO is on branch '$branch', not main (HEAD ${head:0:7})."
    log "  Restarting now would DEPLOY AN UNMERGED BRANCH to production."
    log "  This is CROSS-CUTTING §7, and it has happened three times."
    log "  Fix:  git -C $REPO checkout main"
    log "  First: confirm the other session's work is committed AND pushed —"
    log "         local tip == remote tip, and zero dirty TRACKED files."
    log "  Override (announces itself): PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1"
    exit 1
fi

if [ -z "$origin" ]; then
    log "REFUSING: no local refs/remotes/origin/main to compare against."
    log "  Fix: git -C $REPO fetch origin"
    exit 1
fi

if [ "$head" != "$origin" ]; then
    ahead=$("$GIT" -C "$REPO" rev-list --count refs/remotes/origin/main..HEAD 2>/dev/null || echo '?')
    behind=$("$GIT" -C "$REPO" rev-list --count HEAD..refs/remotes/origin/main 2>/dev/null || echo '?')
    log "REFUSING: main is not equal to origin/main (ahead $ahead, behind $behind)."
    log "  HEAD        ${head:0:7}"
    log "  origin/main ${origin:0:7}"
    if [ "$ahead" != "0" ] && [ "$ahead" != "?" ]; then
        log "  $ahead unpushed commit(s) would deploy without ever having been reviewed."
    else
        log "  The checkout is stale. Fix: git -C $REPO pull --ff-only"
    fi
    log "  Override (announces itself): PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1"
    exit 1
fi

log "OK: $REPO on main at ${head:0:7} == origin/main."
exit 0
