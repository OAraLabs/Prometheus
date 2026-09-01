# Silent-failure audit — config reads

**Status:** config-read leg CLOSED. Two halves, in the order the sequencing
below argues for: the reads made honest 2026-08-31 (#360), the path they read
fixed 2026-09-01 (#361). Superset tracking continues below.

This document is cited by `learning/skill_creator.py`, `learning/nudge.py`,
`learning/skill_refiner.py` and `learning/gepa.py` — all four say "see
docs/audits/SILENT-FAILURE-AUDIT.md Tier-1 hotfix". **It was never committed.**
Those four modules were remediated in 2026-05 and the other four sites carrying
the identical shape were not, and because the record was missing there was no
way to know that the remediation had stopped, or where. It was found in
2026-08-31's config-silence audit, by accident, two years on.

Recreating it is not bookkeeping. The reason the stopping point was unknowable
is that this file did not exist.

---

## The shape

```python
try:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    section = data.get("<section>", {})
except (OSError, Exception):
    section = {}
```

The handler converts **"I could not read your configuration"** into **"you did
not configure anything."** Those are different facts. A wrong path that raises
is found on the first boot. A wrong path behind this handler is found by
accident.

`except (OSError, Exception)` is also redundant — `Exception` subsumes
`OSError`. The belt-and-braces was the smell.

## The four states

| State | Meaning | Verdict |
|---|---|---|
| `LOADED` | file read, parsed to a mapping | fine |
| `ABSENT` | no path specified, or no default file present | legitimate — must be **recorded**, not inferred |
| `UNREADABLE` | a path WAS specified and could not be read | **error**; defaults are not a valid answer |
| `MALFORMED` | read, but did not parse to a mapping | **error** |

`MALFORMED` includes the state nobody had named: an **empty file** makes
`yaml.safe_load` return `None`, so `data.get(...)` raises `AttributeError` —
swallowed by the same bare handler, indistinguishable from a missing file.
`PARTIAL` (file loaded, key absent) is per-key and recorded by
`ConfigLoad.section` / `ConfigLoad.value`.

## Where it is implemented

`src/prometheus/config/load.py` — `load_config_file()` returns a `ConfigLoad`
carrying `data` **and** `state`. `UNREADABLE` and `MALFORMED` log at ERROR and
write a `silent_failures` ledger row; `ABSENT` and `PARTIAL` log without one.
Every message names the path attempted and the value substituted.

It never raises: these are startup paths, and taking the daemon down because an
optional section is missing would be a worse failure than the one being fixed.
What it refuses to do is stay quiet.

---

## Ledger

### Remediated 2026-05 (the original Tier-1 hotfix)

Narrowed catch + warning naming path and substitute. No ledger row — these
predate the ledger being used for config.

| Site | Subsystem |
|---|---|
| `learning/skill_creator.py` | SkillCreator |
| `learning/nudge.py` | PeriodicNudge |
| `learning/skill_refiner.py` | SkillRefiner |
| `learning/gepa.py` | GEPAOptimizer |

### Remediated 2026-08-31 (this PR) — the four it skipped

Now routed through `config/load.py`, so all four states are distinguished and
the loud ones reach the ledger.

| Site | Subsystem | Substitutes |
|---|---|---|
| `permissions/checker.py` | `security_gate` | shipped `denied_paths` floor, **no** configured `denied_commands` |
| `context/budget.py` | `token_budget` | unresolved window (`limit_source='unknown'`) |
| `context/truncation.py` | `tool_result_truncator` | `tool_result_max` default |
| `context/compression.py` | `context_compressor` | `fresh_tail_count` default |

`checker.py` is the one that mattered: its class docstring presented
`SecurityGate.from_config()` — the no-argument form — as *the* usage, and that
path yields a gate with none of the ten configured `denied_commands`. Measured:
`cat /etc/shadow` DENY → ALLOW. The docstring now shows the explicit-path form
the daemon actually uses.

### Remediated 2026-08-31 (this PR) — the same shape outside `DEFAULTS_PATH`

Found by the guard, not by hand. Each now narrows its catch and names the path
and the substitute.

| Site | Silently reported |
|---|---|
| `daemon.py` `_read_config_pins` | an unreadable pin file as "nothing pinned" |
| `cli/doctor.py` pin comparison | an unreadable config as an empty one |
| `cli/doctor.py` stranded-trace count | unknown, with no reason |
| `gateway/telegram.py` `/beacon` | an unreadable config as "web disabled" |
| `gateway/heartbeat.py` stale-state | any read error as "never nudged" |
| `web/setup_server.py` | a malformed config as "not_configured" |
| `setup_wizard.py` × 2 | a malformed config as absent — and setup may then **overwrite** it |
| `__main__.py` model registry | an unreadable registry as "no function-calling support" |
| `infra/anatomy.py` | an unreadable config as "whisper unconfigured" |
| `tools/builtin/image_generate.py` | a YAML typo as "no image_generation config" |
| `tools/builtin/video_generate.py` | a YAML typo as "no video_generation config" |
| `gateway/commands.py` `cmd_context` | a resolution crash as a resolved "unknown" |

The last one was introduced by PR #359, five days before this audit, and was
self-reported. Recency is not an exemption.

---

## Deliberately NOT changed, and why

These substitute silently but catch **narrowly**, and for each the absence of
the file *is* the answer rather than a stand-in for configuration the operator
supplied. Naming the exceptions you mean is the remediation; requiring a log
line where "missing" is the correct semantic answer would make the guard noisy,
and a noisy guard gets switched off.

| Site | Absence means |
|---|---|
| `config/ephemeral.py` | no chat is ephemeral (its docstring already reasons this out) |
| `gateway/status.py` `_read_lock` | the daemon is not running |
| `gateway/sticker_cache.py` | an empty cache |
| `sentinel/golden_trace_exporter.py` watermark | nothing exported yet |
| `symbiote/morph.py` lockfile PID | no daemon lock |
| `gateway/heartbeat.py` `FileNotFoundError` branch | never nudged |

Skill and wiki **frontmatter** parsing (`skills/loader.py`,
`sentinel/wiki_lint.py`) is out of scope: `safe_load` over a string already in
memory is not a config read. The guard requires a file read in the same `try`
body, which excludes them by construction rather than by allowlist.

---

## Closed 2026-09-01 — `DEFAULTS_PATH` (#361)

This section read "Still open: `DEFAULTS_PATH` itself is untouched", and the
sequencing it argued for is what happened. The sites had to fail honestly
first (#360, above) so that changing *what* they load could not land blind;
#361 then changed it.

What shipped, against what this section asked for:

* **The resolver.** `DEFAULTS_PATH` is gone — not repointed. A constant cannot
  express this, and that is the finding this audit had missed: the wheel
  packages `src/prometheus` only, so `config/prometheus.yaml` does not exist
  under `site-packages` at all. A repo-relative path is structurally wrong for
  pip installs, not merely off by one. `config.defaults.config_search_paths()`
  / `resolve_config_path()` carry the search order `__main__.load_config`
  always had — explicit, repo-local, `$PROMETHEUS_CONFIG_DIR` — and
  `cli/doctor` and `__main__` delegate to it rather than keeping a third and
  fourth `parents[N]`. Doctor's copy was CORRECT; two independent hop counts
  is exactly what let one of them be wrong.

* **The per-subsystem enumeration.** `tests/test_config_fallback_callers.py`
  is that work for all eight, not just the `denied_commands` table above.
  Each is covered three ways: reads the config, honours the opposite setting,
  and **absence still yields the pre-fix default** — so the "switches eight
  subsystems simultaneously" risk is bounded by assertion rather than by
  reading. Five are inert on the live config (their keys already equal the
  hardcoded defaults); `skill_refiner` and `budget` change; `permissions/
  checker.py` changes in BOTH directions, and its four gate deltas are
  enumerated in the PR.

* **The hop count is now checked against the filesystem**, not restated:
  `tests/test_config_path_resolution.py` anchors it on `pyproject.toml` and
  pins it equal to `config/template.py`. Falsified by re-breaking the source
  to `parents[1]`, `[2]` and `[4]`.

**On the "safer end state" this section proposed** — removing the module-level
default and requiring an explicit path so a forgetful caller fails loudly:
NOT taken, and the reason is worth recording. Four of the eight catch only
`(OSError, yaml.YAMLError)`, so "fails at the call site" would have meant a
`TypeError` out of `SkillRefiner.from_config` during the daemon's boot. The
resolver therefore always returns a Path, and an absent config raises
`FileNotFoundError` — an `OSError`, which all eight already handle. Loudness
came from #360's `config/load.py` instead, which is the right layer for it:
one honest read, four named states, a ledger row for the two that are errors.

### One consequence, recorded

The fix made the developer's own gitignored `config/prometheus.yaml` reachable
from tests for the first time — it exists in a checkout and NOT in a worktree
or on CI, the shape that passes here and fails there. `REPO_CONFIG_PATH` is
public and module-level so `conftest._isolated_state_dirs` can neutralise it,
alongside the `~/.prometheus` redirection that fixture already documents.

## Still open

Nothing on the config-read leg. `daemon.load_config` resolves
`Path("config/prometheus.yaml")` relative to the process CWD rather than the
source tree — the same class, deliberately left for its own change, since it
is the one path where a wrong answer means booting with the wrong gate.
systemd pins `WorkingDirectory` and passes `--config`, so it is not live.

## The guard

`tests/test_config_read_honesty_invariant.py` fails the build on a broad catch
(`except:`, `except Exception`, `except (OSError, Exception)`) around a config
file read that swallows or substitutes without recording — and on an unguarded
`.get()` over a possibly-`None` parse result, which is the fourth state. It
carries a replay set asserting the detector catches the shapes that actually
shipped, so a green run proves it is looking rather than blind.

That replay check earned its keep immediately: it caught two false-positive
rules in its own detector (an `isinstance` guard and a handler that already
logged), and the corrected detector then surfaced five sites — `__main__.py`,
`infra/anatomy.py`, `setup_wizard.py`, `image_generate.py`,
`video_generate.py` — that hand-reading the Phase 1 scan had missed.
