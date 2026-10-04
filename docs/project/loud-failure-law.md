# The loud-failure law — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

## The loud-failure law (OAra Lab-wide, 2026-07-02)

**Degraded is a state that gets announced, never absorbed. No component may
catch-and-continue silently. Every daemon writes a success heartbeat; staleness
is surfaced, spoken, and shown.**

Origin: OAra Voice's memory extractor 404'd every 30 minutes for ~3.5 months while
reporting "No new events" — fail-safe silence killed that system invisibly. Prometheus
already leans loud (silent_failure telemetry, #78 journal tracebacks); keep it that way:
any new code that catches an exception and continues silently is a bug.

Prometheus's role in enforcement: the cron job `oara_heartbeat_watcher` (every 5 min)
runs OAra's `services/watcher/heartbeat_watcher.py`; a stale Jarvis component makes the
job exit non-zero, so the outage shows up as a failed cron status in Beacon's Config → Cron
tab and as an `error` event in the Jarvis Archive. Do not "fix" that job by making it
always exit 0 — its failure IS the feature.

### Corollary (Sprint 2, 2026-07-02): the law extends to configuration

A subsystem that is expected-enabled but dark is a failure state identical to a
stale heartbeat. LCM compaction sat behind an unset `compaction.enabled` flag
since birth — 1834 messages, zero summaries, and everyone debugged the
summarizer while the flag was the outage. "Never turned on" must be a boot-time
alarm, not an archaeology finding.

Enforcement: GET /api/status now reports a `compaction` block (enabled +
lcm counters); the OAra middleware's config audit (`subsystems:` manifest,
every 5 min) treats expected-but-dark as an immediate watcher alarm. When adding
a feature behind a config flag, either default it ON in prometheus.yaml.default
or register it in the OAra manifest with `expected: false` — a flag nobody
tracks is a future archaeology dig.

### Directory ownership — one writer per vault directory (Sprint 2, 2026-07-02)

Prometheus owns the wiki root — `wiki.root`, default `~/.prometheus/wiki`
(tools/patterns/projects et al.) — and NEVER
writes `~/OAra-Brain`; the Jarvis extractor (via the OAra middleware) owns the
OAra-Brain life-note directories and never writes the wiki. Vault notes carry
`writer:` frontmatter; the hourly `oara_vault_lint` cron (detection-only)
flags any file whose claimed writer doesn't own its tree — a Prometheus wiki
file claiming `writer: oara_extractor` is a violation, and vice versa.
