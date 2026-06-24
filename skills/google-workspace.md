---
name: google-workspace
description: Google Workspace integration (Gmail, Calendar, Drive, Sheets, Docs, Contacts) via local Python scripts and OAuth2. Use when the user needs Google service access and the gog CLI is not available. Falls back to direct API calls through Python.
version: 1.0.0
tags: [Google, Gmail, Calendar, Drive, Sheets, Docs, Contacts, OAuth]
---

# Google Workspace (Python API)

Gmail, Calendar, Drive, Contacts, Sheets, and Docs via Python scripts with OAuth2. No external binaries needed -- runs with Google's Python client libraries.

Note: If the user only needs email, consider using the `himalaya` skill instead (IMAP/SMTP, no Google Cloud project needed). This skill is for Calendar, Drive, Sheets, Docs, or multi-service Google access.

## Setup

### Check if already configured

```bash
python3 ~/.prometheus/skills/scripts/google-workspace/setup.py --check
```

If it prints `AUTHENTICATED`, skip to Usage.

### First-time OAuth setup

1. User creates OAuth credentials at https://console.cloud.google.com/apis/credentials
2. Enable APIs: Gmail, Calendar, Drive, Sheets, Docs, People API
3. Create OAuth 2.0 Client ID (Desktop app), download JSON

```bash
# Save the client secret
python3 ~/.prometheus/skills/scripts/google-workspace/setup.py --client-secret /path/to/client_secret.json

# Get authorization URL (send to user to open in browser)
python3 ~/.prometheus/skills/scripts/google-workspace/setup.py --auth-url

# Exchange auth code (user pastes the redirect URL back)
python3 ~/.prometheus/skills/scripts/google-workspace/setup.py --auth-code "THE_URL_OR_CODE"

# Verify
python3 ~/.prometheus/skills/scripts/google-workspace/setup.py --check
```

Token auto-refreshes after initial setup.

## Usage

Set a shorthand:

```bash
GAPI="python3 ~/.prometheus/skills/scripts/google-workspace/google_api.py"
```

### Gmail

```bash
# Search
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# Read full message
$GAPI gmail get MESSAGE_ID

# Send
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1>" --html

# Reply
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."

# Labels
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# List events (defaults to next 7 days)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (ISO 8601 with timezone required)
$GAPI calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"

# Delete event
$GAPI calendar delete EVENT_ID
```

### Drive

```bash
$GAPI drive search "quarterly report" --max 10
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# Read
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
$GAPI docs get DOC_ID
```

## Output Format

All commands return JSON. Key fields:

- **Gmail search**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**: `{status: "sent", id, threadId}`
- **Calendar list**: `[{id, summary, start, end, location, description}]`
- **Calendar create**: `{status: "created", id, summary}`
- **Drive search**: `[{id, name, mimeType, modifiedTime}]`
- **Contacts list**: `[{name, emails, phones}]`
- **Sheets get**: `[[cell, cell, ...], ...]`

## Rules

1. Never send email or create/delete events without confirming with the user first.
2. Check auth before first use -- run setup.py --check.
3. Calendar times must include timezone -- always use ISO 8601 with offset or UTC (Z).
4. Avoid rapid-fire sequential API calls; batch reads when possible.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup steps above |
| `REFRESH_FAILED` | Token expired -- redo auth steps |
| `HttpError 403: Insufficient Permission` | Missing scope -- revoke and re-auth |
| `HttpError 403: Access Not Configured` | Enable the API in Google Cloud Console |
| `ModuleNotFoundError` | `pip install google-auth google-auth-oauthlib google-api-python-client` |

## Revoking Access

```bash
python3 ~/.prometheus/skills/scripts/google-workspace/setup.py --revoke
```
