# Parity traces (WP-1.2)

Recorded turns that every change to `engine/`, `hooks/`, `router/` or `adapter/`
must replay **identically**. The harness is `scripts/parity/`; the CI job is
`.github/workflows/parity.yml`.

| File | What it is |
|---|---|
| `<scenario>.trace.json` | The inputs: the exact config (`{{PLACEHOLDERS}}` for ports and the token), the files, the steps, and every model exchange — the normalized request and the response exactly as the model sent it. |
| `<scenario>.expected.json` | What the replay must reproduce: step results and a normalized dump of every store the daemon wrote. Derived from the **recording run**, so a replay on unchanged code must reproduce a real run, not just itself. |

## Commands

```bash
uv run python scripts/parity_harness.py replay           # diff against the recording (CI runs this)
uv run python scripts/parity_harness.py stability        # replay twice; both runs must be identical
uv run python scripts/parity_harness.py bench --runs 10  # overhead per round, p50/p95, RSS
uv run python scripts/parity_harness.py normalizations   # every rule that hides a field, and why
```

## When a replay fails

A diff is a behavior change until shown otherwise. The report names the
category (model requests, tool calls, gate decisions, checkpoints, memory,
telemetry, final reply) and the rows. If the change is intended and no model
request changed, see [When only what is recorded changes](#when-only-what-is-recorded-changes).
Otherwise re-record the affected scenario and say in the PR what changed and why:

```bash
uv run python scripts/parity_harness.py record --scenario NAME \
    --upstream-primary <llama.cpp URL> --upstream-alt <ollama URL>
```

`hosted_route` routes its turn to a hosted provider (a `/claude` override), so
recording it also needs the hosted API and the operator's key:

```bash
env ANTHROPIC_API_KEY=<key> uv run python scripts/parity_harness.py record \
    --scenario hosted_route --upstream-primary <llama.cpp URL> \
    --upstream-alt <ollama URL> --upstream-hosted https://api.anthropic.com
```

The daemon under test never holds that key. It gets an obviously fake one from
the env file the scenario writes into its isolated HOME, and the recording proxy
puts the real key on the forwarded request in its place. Request headers are not
recorded, so no trace can contain it. A replay needs no key and no network.

Never edit `*.expected.json` by hand, and never add a normalization rule to make
a diff go away. A rule may be added only for a field that is **shown** to differ
between two runs of unchanged code. Record the evidence in the rule's `why`.

## When only what is recorded changes

Some changes alter what the daemon **records** (a telemetry column filled, a row
added) without changing a single model request. Those don't need the models
again. Every request still matches its committed exchange, so the committed
answers still answer it. Re-derive the expected files from replays of those
exchanges:

```bash
uv run python scripts/parity_harness.py rebaseline-from-exchanges --dry-run  # print the diff, write nothing
uv run python scripts/parity_harness.py rebaseline-from-exchanges            # write the changed expected files
```

It replays each scenario twice against its committed exchanges. For each one it
prints the per-column diff against the expected file. It **refuses**, exits
non-zero and writes no file at all, if any scenario:

- sends a model request that differs from its committed exchange, sends one the
  trace has no answer for, or never sends a committed one. Re-record instead;
- hits a harness error (exit 2) or a daemon step that failed;
- records something different on its second replay than on its first;
- no longer passes its own `require` check.

It writes only `*.expected.json`, and only for the scenarios that changed. It
never touches a trace. The PR that uses it shows the per-column diff for each
change. For a bundle, apply one change at a time, run `--dry-run`, and put each
report in the PR. Then commit the expected files and run `replay` and
`stability`.

A change that alters any model request is a different kind of change: re-record
it.

## What the traces must never contain

These files are public. Prompts and files are synthetic, and the recording proxy
rewrites upstream home paths and addresses before the daemon sees them.
`tests/test_parity_harness.py` runs the pre-commit hook's own patterns over
every file here, plus checks for addresses, home paths and tailnet names.
