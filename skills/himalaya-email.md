---
name: himalaya-email
description: CLI email client via IMAP/SMTP using himalaya. List, read, write, reply, forward, search, and organize emails from the terminal. Supports multiple accounts and message composition with MML.
version: 1.0.0
tags: [Email, IMAP, SMTP, CLI, Communication]
---

# Himalaya Email CLI

Himalaya is a CLI email client for managing emails from the terminal using IMAP/SMTP backends. All operations run through `bash`.

## Prerequisites

1. Himalaya CLI installed (`himalaya --version` to verify)
2. Configuration file at `~/.config/himalaya/config.toml`
3. IMAP/SMTP credentials configured

### Installation

```bash
# Pre-built binary (Linux/macOS)
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh

# macOS via Homebrew
brew install himalaya

# Via cargo
cargo install himalaya --locked
```

## Configuration

Run the interactive wizard:

```bash
himalaya account configure
```

Or create `~/.config/himalaya/config.toml` manually:

```toml
[accounts.personal]
email = "you@example.com"
display-name = "Your Name"
default = true

backend.type = "imap"
backend.host = "imap.example.com"
backend.port = 993
backend.encryption.type = "tls"
backend.login = "you@example.com"
backend.auth.type = "password"
backend.auth.cmd = "pass show email/imap"

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.example.com"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "you@example.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.cmd = "pass show email/smtp"
```

## Common Operations

### List Folders

```bash
himalaya folder list
```

### List Emails

```bash
# INBOX (default)
himalaya envelope list

# Specific folder
himalaya envelope list --folder "Sent"

# With pagination
himalaya envelope list --page 1 --page-size 20
```

### Search Emails

```bash
himalaya envelope list from john@example.com subject meeting
```

### Read an Email

```bash
# By ID (plain text)
himalaya message read 42

# Export raw MIME
himalaya message export 42 --full
```

### Reply to an Email

Non-interactive reply (recommended for Prometheus):

```bash
# Get reply template, modify, and send
himalaya template reply 42 | sed 's/^$/\nYour reply text here\n/' | himalaya template send
```

Or build the reply manually:

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: sender@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>

Your reply here.
EOF
```

### Forward an Email

```bash
himalaya template forward 42 | sed 's/^To:.*/To: newrecipient@example.com/' | himalaya template send
```

### Write a New Email

Pipe the message via stdin (non-interactive):

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya!
EOF
```

### Move/Copy Emails

```bash
# Move to folder
himalaya message move 42 "Archive"

# Copy to folder
himalaya message copy 42 "Important"
```

### Delete an Email

```bash
himalaya message delete 42
```

### Manage Flags

```bash
# Mark as read
himalaya flag add 42 --flag seen

# Mark as unread
himalaya flag remove 42 --flag seen
```

## Multiple Accounts

```bash
# List accounts
himalaya account list

# Use a specific account
himalaya --account work envelope list
```

## Attachments

```bash
# Download attachments
himalaya attachment download 42

# Save to specific directory
himalaya attachment download 42 --dir ~/Downloads
```

## Output Formats

```bash
himalaya envelope list --output json
himalaya envelope list --output plain
```

## Debugging

```bash
RUST_LOG=debug himalaya envelope list
RUST_LOG=trace RUST_BACKTRACE=1 himalaya envelope list
```

## Tips

- Message IDs are relative to the current folder; re-list after folder changes.
- For rich emails with attachments, use MML syntax.
- Store passwords securely using `pass`, system keyring, or a command that outputs the password.
- Use `--output json` for structured output that is easier to parse programmatically.
- Always confirm with the user before sending emails.
