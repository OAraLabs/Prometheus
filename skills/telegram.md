---
name: telegram
description: "Telegram messaging via the message tool (channel=telegram). Use for sending notifications, status updates, and alerts through Telegram."
allowed-tools: ["message"]
---

# Telegram (Via `message`)

Use the `message` tool with `channel: "telegram"`. Prometheus has native Telegram integration for notifications and messaging.

## Musts

- Always: `channel: "telegram"`.
- Prefer explicit target IDs for channels and users.
- Keep messages concise and well-formatted.

## Guidelines

- Use Telegram-compatible Markdown (MarkdownV2 or HTML mode).
- Keep messages under 4096 characters (Telegram limit).
- Use reply markup for interactive elements when needed.

## Common Actions

### Send message:

```json
{
  "action": "send",
  "channel": "telegram",
  "target": "<chat_id>",
  "message": "hello"
}
```

### Send with media:

```json
{
  "action": "send",
  "channel": "telegram",
  "target": "<chat_id>",
  "message": "see attachment",
  "media": "file:///tmp/example.png"
}
```

### Send silent (no notification sound):

```json
{
  "action": "send",
  "channel": "telegram",
  "target": "<chat_id>",
  "message": "Status update",
  "silent": true
}
```

### Read messages:

```json
{
  "action": "read",
  "channel": "telegram",
  "target": "<chat_id>",
  "limit": 20
}
```

### Edit message:

```json
{
  "action": "edit",
  "channel": "telegram",
  "target": "<chat_id>",
  "messageId": "<message_id>",
  "message": "fixed typo"
}
```

### Delete message:

```json
{
  "action": "delete",
  "channel": "telegram",
  "target": "<chat_id>",
  "messageId": "<message_id>"
}
```

### Send poll:

```json
{
  "action": "poll",
  "channel": "telegram",
  "target": "<chat_id>",
  "pollQuestion": "Lunch?",
  "pollOption": ["Pizza", "Sushi", "Salad"],
  "pollMulti": false
}
```

### Pin message:

```json
{
  "action": "pin",
  "channel": "telegram",
  "target": "<chat_id>",
  "messageId": "<message_id>"
}
```

## Alternative: Direct Telegram Bot API via bash

If the message tool is unavailable, use curl directly:

```bash
BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
CHAT_ID="<chat_id>"

# Send a message
curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
  -H "Content-Type: application/json" \
  -d "{\"chat_id\": \"$CHAT_ID\", \"text\": \"Hello from Prometheus\", \"parse_mode\": \"MarkdownV2\"}"

# Send a photo
curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendPhoto" \
  -F "chat_id=$CHAT_ID" \
  -F "photo=@/tmp/image.png" \
  -F "caption=Status update"
```

Other Bot API methods follow the same `curl` pattern (pipe to `jq` to parse the JSON response): `sendMessage` with `reply_to_message_id` (reply), `editMessageText`, `deleteMessage`, `getUpdates` (read recent messages), `sendDocument`, `pinChatMessage` / `unpinChatMessage`, and `getChatMember`.

## Writing Style (Telegram)

- Short, conversational, low ceremony.
- Use Markdown formatting sparingly.
- Prefer inline keyboards for interactive choices.

## Prometheus Context

- Telegram is Prometheus's primary messaging channel.
- Use for SENTINEL alerts, task notifications, and status updates.
- Bot token should be in environment variables, never hardcoded.

## Secret Safety

- Never expose `TELEGRAM_BOT_TOKEN` in agent context, logs, or conversation.
- Never ask the user to paste tokens into chat.
- The user must configure the token in their environment manually.
- Do not use verbose/debug flags that might leak tokens.
