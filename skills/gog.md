---
name: gog
description: Google Workspace CLI for Gmail, Calendar, Drive, Contacts, Sheets, and Docs via the gog binary. Use when the user has gog installed and needs to interact with Google services from the terminal.
version: 1.0.0
tags: [Google, Gmail, Calendar, Drive, Sheets, Docs, CLI]
---

# gog -- Google Workspace CLI

Use `gog` for Gmail, Calendar, Drive, Contacts, Sheets, and Docs operations. Requires OAuth setup. All commands run through `bash`.

## Setup (once)

```bash
gog auth credentials /path/to/client_secret.json
gog auth add you@gmail.com --services gmail,calendar,drive,contacts,docs,sheets
gog auth list
```

Set `GOG_ACCOUNT=you@gmail.com` to avoid repeating `--account`.

## Gmail

```bash
# Search threads
gog gmail search 'newer_than:7d' --max 10

# Search individual messages
gog gmail messages search "in:inbox from:ryanair.com" --max 20 --account you@example.com

# Send (plain text)
gog gmail send --to a@b.com --subject "Hi" --body "Hello"

# Send (multi-line via file)
gog gmail send --to a@b.com --subject "Hi" --body-file ./message.txt

# Send (stdin)
gog gmail send --to a@b.com --subject "Hi" --body-file -

# Send (HTML)
gog gmail send --to a@b.com --subject "Hi" --body-html "<p>Hello</p>"

# Draft
gog gmail drafts create --to a@b.com --subject "Hi" --body-file ./message.txt

# Send draft
gog gmail drafts send <draftId>

# Reply
gog gmail send --to a@b.com --subject "Re: Hi" --body "Reply" --reply-to-message-id <msgId>
```

### Email formatting notes

- Prefer plain text. Use `--body-file` for multi-paragraph messages (or `--body-file -` for stdin).
- `--body` does not unescape `\n`. For inline newlines, use a heredoc or `$'Line 1\n\nLine 2'`.
- Use `--body-html` only when you need rich formatting.

**Example (plain text via stdin):**

```bash
gog gmail send --to recipient@example.com \
  --subject "Meeting Follow-up" \
  --body-file - <<'EOF'
Hi Name,

Thanks for meeting today. Next steps:
- Item one
- Item two

Best regards,
Your Name
EOF
```

## Calendar

```bash
# List events
gog calendar events <calendarId> --from <iso> --to <iso>

# Create event
gog calendar create <calendarId> --summary "Title" --from <iso> --to <iso>

# Create with color
gog calendar create <calendarId> --summary "Title" --from <iso> --to <iso> --event-color 7

# Update event
gog calendar update <calendarId> <eventId> --summary "New Title" --event-color 4

# Show available colors
gog calendar colors
```

Event color IDs (1-11): use `gog calendar colors` to see the full list.

## Drive

```bash
gog drive search "query" --max 10
```

## Contacts

```bash
gog contacts list --max 20
```

## Sheets

```bash
# Read
gog sheets get <sheetId> "Tab!A1:D10" --json

# Write
gog sheets update <sheetId> "Tab!A1:B2" --values-json '[["A","B"],["1","2"]]' --input USER_ENTERED

# Append
gog sheets append <sheetId> "Tab!A:C" --values-json '[["x","y","z"]]' --insert INSERT_ROWS

# Clear
gog sheets clear <sheetId> "Tab!A2:Z"

# Metadata
gog sheets metadata <sheetId> --json
```

## Docs

```bash
# Export to text
gog docs export <docId> --format txt --out /tmp/doc.txt

# Print contents
gog docs cat <docId>
```

## Notes

- For scripting, prefer `--json` plus `--no-input`.
- Sheets values can be passed via `--values-json` (recommended) or as inline rows.
- Docs supports export/cat/copy. In-place edits require a separate Docs API client.
- Always confirm with the user before sending mail or creating events.
- `gog gmail search` returns one row per thread; use `gog gmail messages search` for every individual email separately.
