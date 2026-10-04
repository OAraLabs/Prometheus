# Checkouts and deploy — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

## Three trees — which one you are in matters

| Path | What it is | May you commit from it? |
|---|---|---|
| `.claude/worktrees/<name>` | per-session worktree — branch, commit, PR from here | **yes** |
| `~/Prometheus` | shared dev checkout — read, review, `git worktree add` | **only with the override below** |
| `~/prometheus-deploy` | ff-only mirror of `origin/main`; **the tree the daemon runs** | **no** |

Both refusals are pre-commit hooks, and both have the same reasoning: the
mistake is invisible at the moment it is made and expensive later.

The shared checkout is a *speed bump*, not a wall — solo work, a rebase, a
hotfix are all legitimate there. Say so explicitly and it is recorded rather
than hidden:

```bash
PROMETHEUS_ALLOW_DEV_COMMIT=1 git commit ...
```

That prints a banner and adds a `Dev-checkout-override:` trailer to the
commit, so `git log --grep=Dev-checkout-override` finds every exception
afterwards. Same bargain as `PROMETHEUS_ALLOW_UNMERGED_DEPLOY` in
`scripts/deploy_guard.sh`: an override that announces itself beats a guard
people delete.

The deploy clone is updated one way only:

```bash
git -C ~/prometheus-deploy fetch origin && git -C ~/prometheus-deploy merge --ff-only origin/main
```

Once the daemon runs from a **managed venv** (its unit carries the drop-in
`scripts/deploy.sh` writes), deploy with the script instead. It fast-forwards
the clone *and* builds that commit's venv from its `uv.lock`, gated on
pip-audit. A clone moved by hand without a matching venv is refused at boot:
the guard compares the venv's `BUILT_FROM_UV_LOCK` with the checkout's lock.
The script can run from any checkout; it acts on `~/prometheus-deploy`.

```bash
scripts/deploy.sh v0.9.2 --prepare-only   # build + gate the venv; nothing live changes
scripts/deploy.sh v0.9.2                  # switch, restart, verify, restore model choices
```

**Merging a PR and deferring the deploy is two steps, not one.** The daemon's
`ExecStartPre` guard refuses to boot when that clone is not on `main`, is
AHEAD of local `origin/main`, has DIVERGED from it, or has uncommitted changes
to tracked files.

*Behind* is **not** a refusal, and this doc said it was until 2026-09-08.
Measured against a clone four commits behind:

```
deploy-guard: WARNING: <repo> is BEHIND origin/main (ahead 0, behind 4) — starting anyway.
exit 0
```

The code is right and the doc was wrong, so the doc changed. `deploy_guard.sh`
explains why: every commit in a behind checkout *is* on `origin/main` and was
reviewed, so it is old, not unmerged — and refusing there would make a
deliberate dark-merge incompatible with surviving an unrelated reboot, landing
the unit in `failed` with `StartLimitBurst` exhausted, unattended.

**What that costs you, and it is the reason the two-step rule still stands:** a
merged-but-not-deployed clone boots happily and quietly. `/health` will even
report `stale: false`, correctly — it compares the running code against the
*checked-out tree*, and those agree. Neither signal is comparing against
`origin/main`, and the guard does not fetch (deliberately: boot is the wrong
time for a network call — see the stale-tracking-ref note in the script). So
nothing tells you the clone is behind until you look.

Why this is a hard rule rather than a preference — it has failed twice:

1. **353 lines of uncommitted WIP** sat in the clone for ~10 hours. The daemon
   had booted before the edits landed, so it had never been live, and a
   restart would have deployed it silently alongside an unrelated change.
2. **A whole feature was committed there.** That diverged the clone (arming
   the boot refusal above) and the commit was unpushed and absent from
   `~/Prometheus` — one `git reset --hard` from being the only copy destroyed.

A local `pre-commit` hook in the deploy clone now refuses commits outright.
It lives in `.git/hooks/`, which is **not tracked**, so it does not survive
recreating the clone — reinstall with:

```bash
scripts/install-deploy-guards.sh
```

Before restarting the daemon, always check the clone is both clean and equal
to `origin/main` — `git -C ~/prometheus-deploy status --short` and
`git -C ~/prometheus-deploy log origin/main..HEAD`. A clone that is *diverged*
looks the same as one that is merely *behind* until the ff fails.
