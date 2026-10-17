# 0.9.13 runbook: the live config

Two targeted edits, never a restore of a backup file. Both scripts sit next to this file and are guarded: each block
must occur exactly once, uncommented, where named, with exactly the content expected, or the script refuses and
writes nothing. Without `--write` a script only prints its diff.

Getting them at deploy time (the deploy checkout is on main, not this branch), into a fresh directory:

    git fetch origin docs/local-tools-state
    git show origin/docs/local-tools-state:docs/handoff/runbook-0.9.13/retire_workaround.py > <dir>/retire_workaround.py
    git show origin/docs/local-tools-state:docs/handoff/runbook-0.9.13/cleanup_dead_config.py > <dir>/cleanup_dead_config.py

Run them from the live config directory, on `prometheus.yaml`.

## At the deploy: retire the 10-16 workaround

Diff, live vs prometheus.yaml.bak-20261016T122703Z-pre-agent-cap4096: only the workaround's 4 lines.
Live was last written 10-16 12:27Z (the workaround); nothing else has changed since.

Edit (retire_workaround.py, guarded: each block must occur exactly once, else it refuses):
- remove the 27B's per-model cap under context.model_overrides (comment + GGUF-path key + max_output_tokens: 4096):
  0.9.13's llama.cpp default is 4096 (source "local" instead of "model_override"; same value).
- remove "- agent" from tools.deferred_loading.always_loaded: under local_catalog "stubs" (the default; the key is
  absent from live) the list is not consulted and agent always has its full schema. The rest of the list (the
  shipped 12 + vault_search, vault_read) stays as written: it is "pinned" (your own list) and only matters if
  local_catalog is ever set to "names".

Steps, inside the maintenance window, before the restart:
1. cp -p prometheus.yaml prometheus.yaml.bak-<UTC>-pre-retire-workaround   (this copy keeps the workaround 0.9.12 needs)
2. python3 retire_workaround.py prometheus.yaml            (dry run: exactly the 4 lines)
3. python3 retire_workaround.py prometheus.yaml --write
4. diff the pre-edit backup against live: exactly those 4 lines, nothing else
Order is not critical: on 0.9.13 the workaround is harmless (same cap; the list is not read).

After the restart:
- the boot line says the local catalog: 10 full schemas, the rest stubs; /api/tools/deferred agrees
- expected WARNINGs (unchanged by the edit): context.reserved_output (2000) is no longer read; 5 model_overrides
  that match no served model (gemma4-26b-moe, gemma4-31b, google_gemma-4-26B-A4B-it-Q4_K_M, gpt-4o, qwen3.5-32b);
  5 slash_commands.* not configured. The first two go in the day-after cleanup below; the slash_commands lines
  are left as they are.
- oara doctor: names + statuses only. "Advertised tools" reads "51 of 51 registered offered to a local model
  (10 with their full schema, 41 as stubs)" (fix 3's doctor commit).

Rollback to 0.9.12: restore the pre-edit backup from step 1 together with the code rollback (0.9.12 needs both lines).

## At the deploy: the installed service unit (after the upgrade)

0.9.13's daemon loads its env file itself at startup (#825), so the service unit should no longer pass it with `EnvironmentFile=`. A code deploy doesn't rewrite the installed unit. Do this inside the maintenance window, after the 0.9.13 restart has been verified.

- **Our boxes:** a targeted edit with `drop_unit_envfile.py`, next to this file.
  - **Don't use `oara install-service --force` here.** The installed unit is hand-written: it has `WorkingDirectory=`, the deploy guard's `ExecStartPre=`, `Environment=` lines and start limits, plus a drop-in that overrides `ExecStart=` with a path relative to that working directory. `--force` rewrites the whole file to the generic template and would break the daemon.
- **Checked, read-only, on 2026-10-17:**
  - **Where the line lives:** the main unit has exactly one `EnvironmentFile=` line; the drop-in (`10-venv.conf`) has none.
  - **The same file loads either way:** the user manager sets neither `XDG_CONFIG_HOME` nor `PROMETHEUS_ENV_FILE`, so the daemon loads the same `~/.config/prometheus/env` itself.
  - **The deploy guard is unaffected:** it reads none of that file's variables.
  - **On a copy of the live unit,** the script removed exactly one line, kept every other directive, and refused a second run.

Steps:
1. `cp -p ~/.config/systemd/user/prometheus.service ~/.config/systemd/user/prometheus.service.bak-<UTC>-pre-envfile`
2. `python3 drop_unit_envfile.py` (dry run: exactly the one `EnvironmentFile=` line)
3. `python3 drop_unit_envfile.py --write`
4. `systemctl --user daemon-reload`
5. `systemctl --user restart prometheus` (the maintenance window is still armed)
6. Verify:
   - `systemctl --user cat prometheus.service | grep -c '^EnvironmentFile'` prints `0`.
   - The daemon is active, and `/api/status` answers with the API token: the token is loaded.
   - `/api/status`'s `runtime` block reports the new `MainPID`.
   - `oara doctor` shows the same check statuses as before the edit, Telegram and the cloud providers included. Print names and statuses only.

**Rollback:** restore the step-1 backup, `daemon-reload`, restart.

**For users** (release notes and the advisory): after upgrading, a unit written by `oara install-service` is regenerated with `oara install-service --force` (the old unit is kept as `prometheus.service.bak`) and then `systemctl --user restart prometheus`. A hand-written unit gets its `EnvironmentFile=` line deleted, then `systemctl --user daemon-reload` and a restart. macOS needs nothing: its launcher already loads the env file itself.

## Day after the deploy: config cleanup (separate, not part of the deploy)

cleanup_dead_config.py, guarded the same way (each key once, uncommented, where named, with exactly the child
line expected; anything else refuses and writes nothing). Removes 11 lines:
- context.reserved_output: 2000  (0.9.13 no longer reads it: the reply room is each model's output cap)
- context.model_overrides entries no served or routed model matches: gpt-4o, qwen3.5-32b, gemma4-26b-moe,
  google_gemma-4-26B-A4B-it-Q4_K_M, gemma4-31b (each 2 lines: key + effective_limit)

Steps:
0. Precondition: the 0.9.13 boot log (journalctl --user -u prometheus.service, since the deploy restart) names
   these same five as "matches no model this daemon serves or routes to". If it names fewer, stop and take the
   others off DEAD in the script first.
1. cp -p prometheus.yaml prometheus.yaml.bak-<UTC>-pre-dead-config-cleanup
2. python3 cleanup_dead_config.py prometheus.yaml            (dry run: exactly the 11 lines)
3. python3 cleanup_dead_config.py prometheus.yaml --write
4. diff the step-1 backup against live: exactly those 11 lines removed, nothing added
No restart needed: none of these keys is read today, so nothing changes at runtime; the boot WARNINGs are gone at
the next restart. Tested on a copy: 11 lines out, parses, 0.9.13's boot report shows neither warning and the
same 32,768 window for the local model; a second run refuses.
