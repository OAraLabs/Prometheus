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
telemetry, final reply) and the rows. If the change is intended, re-record the
affected scenario and say in the PR what changed and why:

```bash
uv run python scripts/parity_harness.py record --scenario NAME \
    --upstream-primary <llama.cpp URL> --upstream-alt <ollama URL>
```

Never edit `*.expected.json` by hand, and never add a normalization rule to make
a diff go away. A rule may be added only for a field that is **shown** to differ
between two runs of unchanged code. Record the evidence in the rule's `why`.

## What the traces must never contain

These files are public. Prompts and files are synthetic, and the recording proxy
rewrites upstream home paths and addresses before the daemon sees them.
`tests/test_parity_harness.py` runs the pre-commit hook's own patterns over
every file here, plus checks for addresses, home paths and tailnet names.
