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
#   * AHEAD of local refs/remotes/origin/main — REFUSES. Unpushed commits
#     would deploy without ever having been reviewed. This is the original
#     incident and the reason this file exists.
#   * BEHIND local refs/remotes/origin/main — WARNS and starts. Every commit
#     in the checkout IS on origin/main and was reviewed; the tree is old,
#     not unmerged. "Behind origin" is not a security condition, and
#     refusing there made deliberate dark-merge — merge now, deploy in a
#     chosen window — incompatible with surviving an unrelated reboot: a
#     power cut or kernel update would land the unit in `failed` with
#     StartLimitBurst exhausted, unattended. Guarded by an explicit
#     `merge-base --is-ancestor`, so a DIVERGENT history still refuses.
#   * TRACKED working-tree changes — REFUSES (see the reversal note below).
#     Untracked files are ignored: a stray scratch file must never be able
#     to cause an outage.
#   * It does NOT fetch. Boot is the wrong time for a network call: no
#     network, a hung proxy, or a slow remote would turn a safety check into
#     an outage. Comparing against the LOCAL origin/main ref is stale-tolerant
#     by design — the guard's job is "never deploy an unmerged branch", not
#     "always deploy the newest".
#
# ⚠ THE STALE-TRACKING-REF TRAP — the cost of not fetching, stated plainly
# ------------------------------------------------------------------------
# Because the guard reads the LOCAL `refs/remotes/origin/main`, that ref is
# only as fresh as the last fetch. Advance HEAD without updating it — a
# `reset --hard <sha>`, a `merge FETCH_HEAD`, a copied or rsynced tree — and
# HEAD reads as AHEAD of a ref that is merely stale. The guard then REFUSES a
# perfectly correct deploy, and the message will say "unpushed commit(s)
# would deploy without ever having been reviewed" about commits that are on
# origin and were reviewed. The diagnosis is wrong in the most misleading
# direction available: it accuses the operator of the original incident.
#
# This is accepted, not overlooked. Fetching at boot trades a rare wrong
# refusal for a routine one — DNS down, Tailscale down, a hung proxy, and the
# daemon does not start at all.
#
# The consequence for operators: **`git pull --ff-only` is the ONLY sanctioned
# way the deploy clone advances.** It fetches, so the tracking ref moves with
# HEAD and the two can never disagree. `reset --hard` reaches the same commit
# and arms the false refusal. If you see an "ahead" refusal you believe is
# wrong, `git -C <repo> fetch origin` first and re-read it before overriding.
#
# ⚠ REVERSAL, 2026-08-16 — the dirty-tree decision was the opposite until now
# --------------------------------------------------------------------------
# This file previously read: "It does NOT look at the working tree being
# dirty. Uncommitted local edits to a deployed main are a different (and
# normal) situation." That call is reversed. Uncommitted TRACKED edits mean
# the running code is not any reviewed commit, and `running_sha` will name a
# commit whose content is not what is on disk — the SHA becomes a lie. It has
# happened here: DockerSandbox was once committed directly inside the deploy
# clone. Untracked files stay ignored, deliberately: refusing on those would
# be the over-refusal direction, where a boot gate that reads as "safe" is
# really just unbootable (Standing-Principles §2c).
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
#   # The argument is the DEPLOY CLONE, not the dev checkout — ~/Prometheus is
#   # a free working tree that concurrent sessions leave on feature branches,
#   # which is the whole reason the clone exists.
#   ExecStartPre=/usr/bin/env bash %h/prometheus-deploy/scripts/deploy_guard.sh %h/prometheus-deploy
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

# TRACKED working-tree changes only. `--quiet` exits non-zero when a tracked
# file differs from HEAD; untracked files are invisible to diff-index by
# design, which is exactly the scope we want (see the reversal note above).
if ! "$GIT" -C "$REPO" diff-index --quiet HEAD -- 2>/dev/null; then
    log "REFUSING: $REPO has uncommitted changes to TRACKED files (HEAD ${head:0:7})."
    log "  The running code would not be any reviewed commit, and running_sha"
    log "  would name a commit whose content is not what is on disk."
    "$GIT" -C "$REPO" diff-index --name-only HEAD -- 2>/dev/null \
        | sed 's/^/  modified: /' >&2
    log "  Fix: commit them on a branch, or 'git -C $REPO checkout -- <paths>'."
    log "  Untracked files are NOT a refusal — only tracked modifications are."
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

    # CROSS-CUTTING §8: a detector that broke must not fall through into the
    # permissive branch. '?' means rev-list itself failed — unknown, not zero.
    if [ "$ahead" = "?" ] || [ "$behind" = "?" ]; then
        log "REFUSING: cannot count commits between HEAD and origin/main (ahead $ahead, behind $behind)."
        log "  HEAD        ${head:0:7}"
        log "  origin/main ${origin:0:7}"
        log "  rev-list failed — the comparison is UNKNOWN, which is not the same as equal."
        log "  Override (announces itself): PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1"
        exit 1
    fi

    if [ "$ahead" != "0" ]; then
        log "REFUSING: $REPO is AHEAD of origin/main (ahead $ahead, behind $behind)."
        log "  HEAD        ${head:0:7}"
        log "  origin/main ${origin:0:7}"
        log "  $ahead unpushed commit(s) would deploy without ever having been reviewed."
        log "  If you believe this is wrong, the tracking ref may be STALE:"
        log "    git -C $REPO fetch origin   # then re-read this message"
        log "  Override (announces itself): PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1"
        exit 1
    fi

    # ahead == 0: every commit here is on origin/main. Belt-and-braces — assert
    # the ancestry directly rather than inferring it from the count, so a
    # divergent history can never reach the permissive branch below.
    if ! "$GIT" -C "$REPO" merge-base --is-ancestor HEAD refs/remotes/origin/main 2>/dev/null; then
        log "REFUSING: HEAD is not an ancestor of origin/main (ahead $ahead, behind $behind)."
        log "  HEAD        ${head:0:7}"
        log "  origin/main ${origin:0:7}"
        log "  The histories have DIVERGED — this is not a fast-forward gap."
        log "  Override (announces itself): PROMETHEUS_ALLOW_UNMERGED_DEPLOY=1"
        exit 1
    fi

    # Behind only, pure fast-forward: old code, not unmerged code. Start.
    # Same "(ahead N, behind N)" shape as every refusal above, so an operator
    # or a grep does not need to learn a second format for the one line that
    # permits a start.
    log "WARNING: $REPO is BEHIND origin/main (ahead $ahead, behind $behind) — starting anyway."
    log "  HEAD        ${head:0:7}"
    log "  origin/main ${origin:0:7}"
    log "  Every commit here is on origin/main and was reviewed; the checkout is"
    log "  merely old. This is what a deliberate dark-merge looks like."
    log "  To go live with the newest: git -C $REPO pull --ff-only && restart"
    exit 0
fi

log "OK: $REPO on main at ${head:0:7} == origin/main."
exit 0
