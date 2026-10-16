## What this fixes

Fix 7 of the local tool-calling audit: telemetry records the tools a round SENT. Fix 4 (#816) is on main; this is the next PR in the stack.

The lucky-guess marker used to recompute "was this tool advertised?" from the loader and the adapter at call time. A run whose tools were chosen for a cloud route had been sent every tool. When the recomputation disagreed (#462's advertise-before-routing, a cloud→local fallback), an offered tool was marked a lucky guess: 25 of the 28 recorded guesses were mislabels. `tool_sets` hashed the run-start list once, whatever a round sent.

- Each round records the names it offers in a per-run ContextVar, before its calls are dispatched: its payload, or the prompt-listed tools at tier full; none on a no-tools round. The marker reads it. Outside a run there is no round and no marker.
- A round that sends a set other than the turn's writes its own `tool_sets` row (once per distinct set), and its `responses` row carries that hash.

Moves no gate check; it keeps the next audit's numbers honest.

## Tests

`tests/test_telemetry_records_what_was_sent.py` (4), through `run_loop`:
- a tool the round offered writes no marker even when a recomputation would say deferred (this one fails on the code before this PR);
- a tool the round did not offer writes one;
- a round that sends another set gets its own `tool_sets` row and hash.

The existing lucky-guess and telemetry v2 tests pass unchanged.

## Checks run

- Ruff, the mypy gate, and this PR's tests with the lucky-guess, fallback-attribution, telemetry v2 and advertisement tests plus the hermetic gate: 233 passed, 2 skipped, 7 xfailed (the gate's remaining markers). CI runs the full suite on this head.
- Parity `replay`: 12/12 PARITY (on this head). `stability`: IDENTICAL across 2 runs.
- `ruff check src/`: clean. Mypy gate: OK.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
