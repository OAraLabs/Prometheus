"""PARITY — the golden-trace harness for the daemon seams (WP-1.2).

Proves that a change to ``engine/``, ``hooks/``, ``router/`` or ``adapter/``
lost nothing and slowed nothing, by replaying recorded turns through a REAL
daemon process and diffing what it did.

How it works, in one paragraph: the daemon is booted as a subprocess with its
own HOME, data dir, config and ports — never the live daemon — and its model
endpoint points at :mod:`parity.model_server`, a stdlib HTTP server that sits
at the provider boundary. In RECORD mode that server proxies to a real model
and saves every exchange; in REPLAY mode it answers from the saved exchanges
and checks that each request the daemon sends matches the recorded one. The
harness drives the turns over the daemon's REST/WS API exactly as a client
does, then reads back every side effect the daemon persisted (tool calls, gate
decisions, checkpoints, memory writes, telemetry rows, the final reply) and
diffs it against the recording.

CONTRACT (the same one FIRSTLIGHT keeps): nothing here imports
``prometheus``. The instrument must not share code with the thing it measures,
or a regression in that code moves both sides of the comparison at once.

Not to be confused with ``telemetry.export_golden_traces`` — that is training
data for fine-tuning. These are regression baselines; the files are called
*parity traces* on disk to keep the two apart.
"""
