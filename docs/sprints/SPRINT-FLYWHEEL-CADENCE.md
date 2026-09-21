# SPRINT: Flywheel Cadence — evals, gym, harvest, train, on a clock

**Branch:** `feat/flywheel-cadence`
**Status:** NOT STARTED (2026-09-02). Spec written from a read of `main @ ccbde1c`.
**Origin:** The fine-tuning flywheel is the one differentiator a cloud-first harness cannot
copy, and every stage of it exists as a script that a human has to remember to run:
`run_nightly_evals.py` (docstring: "designed for cron execution" — no cron runs it),
`gym_run.py`, `mine_training_pairs.py`, `export_training_pairs.py`, `lora/train_dictwrap_dpo.py`,
`lora/eval_dictwrap.py` (pre-registered bars), and `HARVEST-OVERNIGHT-RUNBOOK.md`
("STAGED, not executed"). The operator has said plainly that running these on a cadence by hand
is the thing he does not do. So the machine does it, under the loud-failure law, and a human
decides only one thing: whether to promote an adapter.

This is a **wiring and scheduling sprint**, not a research sprint. It adds no new science.
Every stage below calls a script that already exists; the sprint's whole job is the clock,
the gates, the daemon-down window, the report, and the proof that the daemon comes back.

---

## Phase 0 — survey (answer in the PR description, with line numbers)

1. Which host runs `llama-server` for the production model today, and what stops/starts it
   (`systemctl --user`? a shell? `scripts/start.sh`?). The weekly window has to own it.
2. Confirm `prometheus.jobs.daily_briefing` can send Telegram **with the daemon stopped**
   (it bridges `PROMETHEUS_TELEGRAM_TOKEN` → `TELEGRAM_BOT_TOKEN` itself). The flywheel jobs
   reuse that exact path; if it depends on the daemon in any way, say so — it is the report's
   only channel during the window.
3. `scripts/deploy_guard.sh` refuses to boot from a non-main checkout. The flywheel restarts
   the daemon in a `finally`; confirm that restart goes *through* the guard, not around it.
4. `gym.db`, `training.db`, `telemetry.db` — record their current row counts and the taskset
   SHAs in `gym/tasksets/`. These are the baseline the first weekly report compares against.
5. Current GPU memory headroom on the 4090 with `llama-server` up vs down (`nvidia-smi`), and
   the wall time of one `gym_run.py` task at each. The runbook's 38 s contended / ~17 s sole
   numbers are from June; re-measure.
6. Is there a second GPU host yet (Spark / Halo / rented)? The monthly train stage is
   parameterised on `flywheel.train_host`; Phase 3 cannot be exercised end-to-end until one
   exists. Say which, or say "none — Phase 3 lands with `--dry-run` only".

---

## The shape

Three deterministic jobs in the existing `prometheus.jobs` package (the `daily_briefing`
pattern: fixed pipeline, no agent loop, fail-loud, non-zero exit on any stage failure).
Systemd **timers on the Mini** — the Mini is the orchestrator by the standing architecture
rule; compute stages dispatch over SSH to the GPU host(s). Not the daemon's own cron:
the daemon is *down* for the weekly window, and a scheduler that dies with the thing it is
scheduling cannot restart it.

| Job | Timer | Daemon | Budget | Does |
|---|---|---|---|---|
| `flywheel_nightly` | 03:00 ET daily | up | 45 min | tier-1 evals, pytest against the live install, telemetry digest |
| `flywheel_weekly` | Sat 02:00 ET | **down** | 4 h hard | gym on the frozen taskset (the reliability index), harvest, mine, export |
| `flywheel_monthly` | 1st Sat 02:00 ET (replaces weekly that day) | down | 6 h hard | weekly + dispatch train, pull adapter, held-out eval, stage candidate |

Every job: **preflight gates → stages → report → Telegram → commit report**, in that order, and
the daemon restart in a `finally` that runs even when a gate refuses.

Reports land in `flywheel/reports/YYYY-MM-DD-<kind>.md` (+ `.json`) and `flywheel/reports/LATEST.md`,
committed to `main` by the job under the OAra Labs identity (`support@oara.ai`). The weekly
report *is* the public reliability index — that is deliberate; the repo is public and the
number is the product. A Claude scheduled task reads `LATEST.md` every Monday morning and
turns it into a one-paragraph brief plus a drafted Claude Code prompt if a gate failed.

---

## Phase 1 — `flywheel_nightly` (daemon up, contended GPU is fine)

Stages, each a function returning a `StageResult(name, ok, seconds, summary, artifact_path)`:

1. **evals** — `scripts/run_nightly_evals.py --tier 1` in-process (import its `main`, do not
   shell out; a shell-out swallows the exit code class). Judge provenance recorded as the
   evals subsystem already does.
2. **pytest** — `uv run pytest -q -m "not network" -p no:cacheprovider` in the **deployed**
   checkout with the **live** config dir. CI runs the suite on a clean image; this run is
   the one that catches the pydantic-time-bomb class (developer venv hides a break the
   install will hit). Record pass/fail/skip counts. A skip count that *rises* vs last night
   is a WARN, not a pass — "a subsystem's tests skipping is not passing" (#330).
3. **digest** — 24 h tool-call success by tool from `telemetry.db`, repair counts by
   transition class (`transition_taxonomy.classify_transition`), silent-failure table delta,
   `subsystem_runs` failures. Numbers only; no LLM call.

Gates: disk ≥ 10 GB free on the Mini; `telemetry.db` writable; deadline 45 min enforced
*between* stages (a stage past the deadline is reported as `TIMEOUT`, the job still writes its
report).

Report: one table, three rows, each with ok / seconds / one-line summary, then the digest.
Telegram: one message, ≤ 12 lines, first line `FLYWHEEL nightly · OK|WARN|FAIL`.

Config (`prometheus.yaml.default`, registered with a reader per #136 — an unread key fails
the build):

```yaml
flywheel:
  enabled: false                 # timers are installed by the operator, not by the daemon
  reports_dir: flywheel/reports
  commit_reports: true
  git_identity: "OAra Labs <support@oara.ai>"
  nightly: { budget_minutes: 45, eval_tier: 1 }
```

## Phase 2 — `flywheel_weekly` (daemon down, sole GPU)

Preflight gates, all must pass or the job exits non-zero **without touching the daemon**:

- G1 disk ≥ 30 GB free on Mini and on the GPU host (`ssh … df`)
- G2 no coding run in flight (`/api/code` shows none active) and no approval pending — a
  window that starts mid-task loses the task
- G3 frozen taskset SHA matches `gym/tasksets/<index>.yaml` recorded in the manifest;
  `runs_per_task ≥ 3` (the report refuses thin-sample verdicts, so do not schedule them)
- G4 last weekly report older than 5 days (a manual run earlier in the week is not repeated)
- G5 `llama-server` healthy on the GPU host before we stop anything

Then, inside `try/finally`:

1. **stop** — `systemctl --user stop prometheus` on the Mini; confirm the port is closed;
   record `daemon_stopped_at`. AutoDream, Telegram gateway, Beacon WS all go with it — that
   is expected and the report says so.
2. **index** — `gym_run.py` on the frozen public taskset against the production model,
   manifest `gym/experiments/index-weekly.yaml`, baseline = last week's experiment id.
   Output: the per-tool success table, raw-emission vs post-repair (the gym's dual score),
   with the model name, quant, context, GPU, and llama.cpp build recorded on every row.
   This table is the reliability index. Week 1 has no baseline; say so, do not fabricate a
   delta.
3. **harvest** — the overnight runbook's loop with its pilot gate: run 20 tasks, check
   distinct-prompt → distinct-pair, *then* run to the time budget. `runs_per_task` from the
   runbook, corpus from `gen_harvest_corpus.py`. Stop on the deadline, not on a count.
4. **mine + export** — `mine_training_pairs.py --commit`, then `export_training_pairs.py
   --since <last weekly> --out flywheel/pairs/YYYY-MM-DD.jsonl`. Record new-pair count and
   the transition-class histogram (`transition_histogram.py`); the runbook predicts ~88 %
   `dict_wrap_unwrap` — if it is not, that is a finding, put it in the report.
5. **finally: start** — `systemctl --user start prometheus` through the deploy guard;
   poll `/health` for up to 120 s; `daemon_restarted_at`. If the daemon does **not** come
   back the job exits non-zero AND sends Telegram from the job's own token AND the systemd
   `OnFailure=` unit (below) retries once. There is no state in which the window ends with
   the daemon silently down.

Budget: 4 h wall from timer fire, enforced between tasks inside the harvest loop; the
systemd unit carries `TimeoutStartSec=4h30m` as the outer wall.

```yaml
flywheel:
  weekly:
    budget_hours: 4
    window: "Sat 02:00"          # documentation; the timer is the truth
    gpu_host: oara-4090          # ssh alias; llama-server lives here
    index_taskset: gym/tasksets/index-v1.yaml
    index_manifest: gym/experiments/index-weekly.yaml
    harvest_taskset: gym/tasksets/harvest-overnight.yaml
    runs_per_task: 3
```

Systemd (Mini, `--user`), shipped under `deploy/systemd/` next to `prometheus.service`:

```ini
# prometheus-flywheel-weekly.timer
[Timer]
OnCalendar=Sat *-*-* 02:00:00 America/New_York
Persistent=false          # a missed window is skipped, not run at 09:00 on a weekday
# prometheus-flywheel-weekly.service
[Service]
Type=oneshot
TimeoutStartSec=4h30m
OnFailure=prometheus-flywheel-recover.service
ExecStart=%h/prometheus-build/Prometheus/.venv/bin/python -m prometheus.jobs.flywheel_weekly
```

`prometheus-flywheel-recover.service` does exactly two things: start the daemon through the
guard, and send one Telegram line saying the flywheel failed and the daemon was restarted by
the recovery unit. It has no other logic; it must not be clever.

## Phase 3 — `flywheel_monthly` (weekly + train + held-out eval + candidate)

Runs the weekly stages first (same window, 6 h budget), then:

6. **train gate** — new pairs since the last train ≥ `min_new_pairs` (default 500); the
   held-out taskset exists and shares no prompt with the training export (assert, do not
   assume — the runbook's dedup finding); `train_host` reachable and its GPU idle.
7. **dispatch** — `rsync` the export + `scripts/lora/train_dictwrap_dpo.py` to
   `flywheel.train_host`; run `--dry-run` first (one DPO step — validates arch + template);
   then the full train under `nohup` with its own deadline. The job polls; it does not hold
   an SSH session open for hours.
8. **pull** — adapter back to `~/.prometheus/adapters/<date>/` on the Mini with the training
   manifest (pairs count, base model SHA, LoRA config, wall time, host).
9. **held-out eval** — two harvest passes on the held-out taskset, base vs LoRA-loaded
   `llama-server` on the GPU host, then `scripts/lora/eval_dictwrap.py` with its
   **pre-registered bars unchanged** (≥ 40 % relative drop, control within 2 pts, base
   validity floor 0.20 → INCONCLUSIVE is not PASS). The bars are in code; the job does not
   take them from config, so nobody lowers them at 3 a.m.
10. **candidate** — on PASS, write `~/.prometheus/adapters/candidate` → `<date>` and put a
    **promotion request in the approval queue** (the same queue Beacon iOS shows). Nothing
    is hot-swapped. `prometheus flywheel promote` is the manual step; it is the one decision
    the operator keeps, and it fits the consent posture the rest of the daemon already has.
    Auto-promotion with rollback is v2 and needs the eval to have been right three months
    running first.

```yaml
flywheel:
  monthly:
    budget_hours: 6
    train_host: ""               # "" = train stage lands as --dry-run only, loudly
    min_new_pairs: 500
    heldout_taskset: gym/tasksets/heldout-v1.yaml
    adapters_dir: ~/.prometheus/adapters
```

**Hardware note for `train_host`** (2026-09-02): `train_dictwrap_dpo.py` records that the
26B-A4B MoE does not fit a 24 GB 4090 for DPO (bnb leaves the fused experts unquantised,
~22 GB resident, no headroom). A 128 GB unified-memory box removes the *fit* problem: a
DGX Spark is CUDA and aarch64 (torch/bnb/peft wheels exist; some packages still lag on
arm64), ~273 GB/s so a train is slow — hours-to-a-day — which is fine for a monthly job that
nobody is watching. A Strix Halo also fits 27–35B LoRA in memory but the ROCm training stack
is nightly-wheel, source-built bitsandbytes, kernel-tuned; workable, wrong for an
*unattended* pipeline. A rented A100-80 GB for 2–4 h/month is the third option: only the
train stage leaves the house, data and adapter come home. The job is the same in all three
cases; only `train_host` and the venv bootstrap script differ.

## Phase 4 — surfaces

- `GET /api/flywheel/latest` returns the last report of each kind (JSON). Beacon Status gets
  a **Flywheel** section: last nightly OK/WARN/FAIL, last index date + headline success rate,
  candidate adapter pending yes/no. iOS Status gets the same three lines. No new views.
- `prometheus flywheel status|run <kind>|promote` CLI. `run` is the same code path as the
  timer with `--now`; it is how the window is rehearsed in daylight before it runs at 02:00.
- `/flywheel` Telegram command → last report summary (shared command layer, so Slack and
  Discord get it for free).
- Weekly reviewer: a Claude scheduled task (already created, Mondays 08:00 ET) fetches
  `flywheel/reports/LATEST.md` from the public repo and writes the brief. Until Phase 1
  commits a report it says so in one line and stops.

---

## Hazards (read before Phase 2)

- **The window must fail closed on the daemon, open on everything else.** Every code path
  out of the weekly job — gate refusal, stage exception, deadline, SIGTERM from systemd —
  ends in the daemon start. Test it with a fault injected at every stage boundary.
- **Do not let the harvest loop consult the clock only at the top.** A single gym task can
  take minutes on a bad round; the deadline check runs before *each* task and the harvest
  writes its partial results before it stops.
- **`Persistent=false` on the timers.** A Mini that was off at 02:00 Saturday must not run a
  4-hour daemon-down job at 09:00 Tuesday.
- **Reports commit from a clean tree only.** If the deployed checkout has uncommitted
  changes or is not on `main` (the deploy-guard incident), write the report to disk, send
  the Telegram, and say "report not committed: tree dirty" — never `git add -A`.
- **The public index publishes model + quant + hardware, never prompts or pair contents.**
  `flywheel/pairs/` is gitignored. Only `flywheel/reports/` is committed.
- **Register every new config key** (`test_no_new_config_key_without_a_reader`). The
  register can only shrink.

## Tests (`tests/test_flywheel_jobs.py`, side-effect assertions, no live model)

- gates: each gate refuses on its condition and the job exits non-zero having called
  neither `stop` nor any stage (assert via a recording fake for `systemctl`)
- `finally` restart: inject an exception in stages 1–4 and in the deadline path; assert
  `start` was called exactly once and the report file exists with status FAIL
- deadline: a fake harvest task that sleeps past the deadline is not started; partial
  results are in the report
- report schema: `.json` validates against a pydantic model; `LATEST.md` is replaced
  atomically (tmp + rename, the vault-marker discipline)
- commit: dirty tree → report written, not committed, reason recorded
- eval bars: `eval_dictwrap` bars are imported constants; a test asserts the job passes no
  override
- registration: the three jobs are importable and runnable with `--dry-run` from a clean
  venv (the orphan-tool audit's lesson, applied to jobs)

## Non-goals

Auto-promotion of adapters. Multi-model index sweeps (v2: needs `llama-server` model swap
over SSH and a per-model time budget). Any change to the gym scoring, the eval bars, the
adapter, or the harvest corpus beyond what the runbook already specifies. A web UI for
reports beyond the three Status lines.

## Done means

Three consecutive Saturday windows have run unattended, each ending with the daemon up and
a report committed; one Monday brief has arrived from the reviewer with nothing to draft;
and the operator has not run a gym script by hand in a month.
