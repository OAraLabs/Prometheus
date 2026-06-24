---
name: session-logs
description: Search and analyze Prometheus session logs and telemetry data using jq and bash. Use when the user references older conversations, asks what was said before, or wants to review agent session history and costs.
---

# Session Logs

Search Prometheus conversation history stored in session log files. Use this when a user references older conversations, asks about prior context, or wants to review session telemetry.

## Trigger

Use this skill when the user asks about prior chats, previous sessions, historical context, or telemetry data that is not in memory files or wiki.

## Location

Session logs live under the Prometheus state directory:

```
~/.prometheus/sessions/
```

- **`sessions.json`** - Index mapping session keys to session IDs
- **`<session-id>.jsonl`** - Full conversation transcript per session

## Structure

Each `.jsonl` file contains messages with:

- `type`: "session" (metadata) or "message"
- `timestamp`: ISO timestamp
- `message.role`: "user", "assistant", or "toolResult"
- `message.content[]`: Text, thinking, or tool calls (filter `type=="text"` for human-readable content)
- `message.usage`: Token usage and cost data per response

## Common Queries

### List all sessions by date and size

```bash
SESSION_DIR="$HOME/.prometheus/sessions"
for f in "$SESSION_DIR"/*.jsonl; do
  date=$(head -1 "$f" | jq -r '.timestamp' | cut -dT -f1)
  size=$(ls -lh "$f" | awk '{print $5}')
  echo "$date $size $(basename $f)"
done | sort -r
```

### Find sessions from a specific day

```bash
SESSION_DIR="$HOME/.prometheus/sessions"
for f in "$SESSION_DIR"/*.jsonl; do
  head -1 "$f" | jq -r '.timestamp' | grep -q "2026-04-08" && echo "$f"
done
```

### Extract user messages from a session

```bash
jq -r 'select(.message.role == "user") | .message.content[]? | select(.type == "text") | .text' <session>.jsonl
```

### Search for keyword in assistant responses

```bash
jq -r 'select(.message.role == "assistant") | .message.content[]? | select(.type == "text") | .text' <session>.jsonl | rg -i "keyword"
```

### Get total cost for a session

```bash
jq -s '[.[] | .message.usage.cost.total // 0] | add' <session>.jsonl
```

### Daily cost summary

```bash
SESSION_DIR="$HOME/.prometheus/sessions"
for f in "$SESSION_DIR"/*.jsonl; do
  date=$(head -1 "$f" | jq -r '.timestamp' | cut -dT -f1)
  cost=$(jq -s '[.[] | .message.usage.cost.total // 0] | add' "$f")
  echo "$date $cost"
done | awk '{a[$1]+=$2} END {for(d in a) print d, "$"a[d]}' | sort -r
```

### Count messages and tokens in a session

```bash
jq -s '{
  messages: length,
  user: [.[] | select(.message.role == "user")] | length,
  assistant: [.[] | select(.message.role == "assistant")] | length,
  first: .[0].timestamp,
  last: .[-1].timestamp
}' <session>.jsonl
```

### Tool usage breakdown

```bash
jq -r '.message.content[]? | select(.type == "toolCall") | .name' <session>.jsonl | sort | uniq -c | sort -rn
```

### Search across ALL sessions for a phrase

```bash
SESSION_DIR="$HOME/.prometheus/sessions"
rg -l "phrase" "$SESSION_DIR"/*.jsonl
```

## SENTINEL Integration

Session telemetry can feed into SENTINEL for monitoring agent performance:
- Track cost per session over time
- Identify sessions with unusually high token usage
- Monitor tool usage patterns for optimization
- Flag sessions that hit circuit breaker thresholds

## Tips

- Sessions are append-only JSONL (one JSON object per line)
- Large sessions can be several MB - use `head`/`tail` for sampling
- Use `grep` tool for fast content search across session files
- Combine with wiki entries for cross-referencing session context

## Fast text-only extraction (low noise)

```bash
SESSION_DIR="$HOME/.prometheus/sessions"
jq -r 'select(.type=="message") | .message.content[]? | select(.type=="text") | .text' "$SESSION_DIR"/<id>.jsonl | rg 'keyword'
```
