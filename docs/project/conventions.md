# Key paths and conventions — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

## Key Paths
- Tools: src/prometheus/tools/builtin/
- Adapter: src/prometheus/adapter/
- Engine: src/prometheus/engine/agent_loop.py
- Providers: src/prometheus/providers/
- Memory/LCM: src/prometheus/memory/
- Gateway: src/prometheus/gateway/telegram.py
- Config: config/prometheus.yaml
- Skills: skills/

## Conventions
- FILES FOR THE USER go in ~/.prometheus/files/ (the OUTBOX). Anything saved there is published:
  indexed by GET /api/artifacts and downloadable in the user's apps (Beacon shows a download chip
  when your reply mentions the filename). Do NOT invent sibling dirs (~/.prometheus/downloads etc.)
  for deliverables — outside the outbox nothing is delivered. Working files stay in the workspace.
- New tools extend BaseTool in tools/base.py
- Security checks go through SecurityGate (permissions/)
- Tool results truncated by tool_result_max in config
- ADDITIVE ONLY: extend existing files, don't replace them

### No real infrastructure identifiers in persisted content
The test is **"does it persist?"**, not "is it committed?".

Never write real tailnet/LAN host addresses, Telegram chat ids, tokens, device
ids or account ids into: PR bodies or PR comments, commit messages, docs,
scheduled-task prompts (they are stored as `SKILL.md` on disk), audit reports,
`OAra-Brain/wiki/log.md`, memory files, or scripts under `~/.local/`.

Refer to hosts by NAME (`OAra-mini`, `oara-4090`) and sessions by NAMESPACE
(`telegram:<operator-chat-id>`). Resolve addresses at runtime from
`tailscale status` rather than writing one down. When you author a prompt for
an agent or scheduled task that will WRITE a report, put this rule in that
prompt too — it persists what it writes.

EXEMPT: range constants that are part of the logic, e.g. the CGNAT block
`100.64.0.0/10` in `security/url_guard.py` and the `100.64.0.x` literals in
`tests/test_tailnet_hop_asymmetry.py`. Those are the specification, not an
address.

WHY THIS IS A WRITTEN RULE RATHER THAN A HOOK: this repo is PUBLIC, and the
pre-commit secret hook scans **tracked files only** (there is no GHAS). Every
recurrence so far has been in something the hook structurally cannot see — a PR
body, a commit message, a task prompt, a file outside the repo. GitHub also
retains PR-body edit history, so redacting after the fact limits future exposure
but does not erase it. Get it right the first time.
