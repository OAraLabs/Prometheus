<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with:  uv run python scripts/gen_reference.py
     Pinned current by tests/test_generated_reference.py. -->

# Command reference (generated)

Slash commands as REGISTERED, read out of each gateway's own
registration calls — not from a hand-mirrored list. Slack's are
exposed as `/prometheus-<name>`; the bare name is shown here.

Enumerating the registry is deliberate: grepping for an expected
name once produced a false finding, because Slack's registrations
carry the `/prometheus-` prefix and the bare name matched nothing.

## By gateway — 55 distinct commands

The three gateways spell the same command differently. Telegram
uses the bare name, Slack prefixes every one with `/prometheus-`,
and Discord registers a single `/prometheus` root with four
sub-groups — so its invocation is `/prometheus core help`, never
`/help`. The Discord column shows the real invocation for that
reason.

| Command | Telegram | Slack | Discord |
|---|---|---|---|
| `/anatomy` | `/anatomy` | `/prometheus-anatomy` | `/prometheus core anatomy` |
| `/approve` | `/approve` | `/prometheus-approve` | `/prometheus ops approve` |
| `/audit` | `/audit` | `/prometheus-audit` | `/prometheus ops audit` |
| `/backends` | `/backends` | `/prometheus-backends` | `/prometheus provider backends` |
| `/beacon` | `/beacon` | `/prometheus-beacon` | `/prometheus core beacon` |
| `/benchmark` | `/benchmark` | `/prometheus-benchmark` | `/prometheus core benchmark` |
| `/claude` | `/claude` | `/prometheus-claude` | `/prometheus provider claude` |
| `/clear` | `/clear` | — | `/prometheus core clear` |
| `/clearsteers` | `/clearsteers` | `/prometheus-clearsteers` | `/prometheus session clearsteers` |
| `/context` | `/context` | `/prometheus-context` | `/prometheus core context` |
| `/curator` | `/curator` | `/prometheus-curator` | `/prometheus core curator` |
| `/deepseek` | `/deepseek` | `/prometheus-deepseek` | `/prometheus provider deepseek` |
| `/deny` | `/deny` | `/prometheus-deny` | `/prometheus ops deny` |
| `/doctor` | `/doctor` | `/prometheus-doctor` | `/prometheus core doctor` |
| `/ephemeral` | `/ephemeral` | — | — |
| `/escalations` | `/escalations` | `/prometheus-escalations` | `/prometheus ops escalations` |
| `/events` | `/events` | `/prometheus-events` | `/prometheus core events` |
| `/gate` | `/gate` | — | — |
| `/gemini` | `/gemini` | `/prometheus-gemini` | `/prometheus provider gemini` |
| `/gepa` | `/gepa` | `/prometheus-gepa` | `/prometheus ops gepa` |
| `/glm` | `/glm` | `/prometheus-glm` | `/prometheus provider glm` |
| `/gpt` | `/gpt` | `/prometheus-gpt` | `/prometheus provider gpt` |
| `/grants` | `/grants` | `/prometheus-grants` | `/prometheus ops grants` |
| `/grok` | `/grok` | `/prometheus-grok` | `/prometheus provider grok` |
| `/health` | `/health` | `/prometheus-health` | `/prometheus core health` |
| `/help` | `/help` | `/prometheus-help` | `/prometheus core help` |
| `/kimi` | `/kimi` | `/prometheus-kimi` | `/prometheus provider kimi` |
| `/local` | `/local` | `/prometheus-local` | `/prometheus provider local` |
| `/memory` | `/memory` | `/prometheus-memory` | `/prometheus core memory` |
| `/mimo` | `/mimo` | `/prometheus-mimo` | `/prometheus provider mimo` |
| `/model` | `/model` | `/prometheus-model` | `/prometheus core model` |
| `/note` | `/note` | `/prometheus-note` | `/prometheus core note` |
| `/notifications` | `/notifications` | `/prometheus-notifications` | `/prometheus core notifications` |
| `/pairs` | `/pairs` | `/prometheus-pairs` | `/prometheus core pairs` |
| `/pending` | `/pending` | `/prometheus-pending` | `/prometheus ops pending` |
| `/press` | `/press` | `/prometheus-press` | `/prometheus ops press` |
| `/profile` | `/profile` | `/prometheus-profile` | `/prometheus core profile` |
| `/queue` | `/queue` | `/prometheus-queue` | `/prometheus session queue` |
| `/qwen` | `/qwen` | `/prometheus-qwen` | `/prometheus provider qwen` |
| `/remember` | `/remember` | `/prometheus-remember` | `/prometheus ops remember` |
| `/reset` | `/reset` | `/prometheus-reset` | `/prometheus core reset` |
| `/revoke` | `/revoke` | `/prometheus-revoke` | `/prometheus ops revoke` |
| `/route` | `/route` | `/prometheus-route` | `/prometheus provider route` |
| `/sentinel` | `/sentinel` | `/prometheus-sentinel` | `/prometheus core sentinel` |
| `/skills` | `/skills` | `/prometheus-skills` | `/prometheus core skills` |
| `/start` | `/start` | — | `/prometheus core start` |
| `/status` | `/status` | `/prometheus-status` | `/prometheus core status` |
| `/steer` | `/steer` | `/prometheus-steer` | `/prometheus session steer` |
| `/symbiote` | `/symbiote` | `/prometheus-symbiote` | `/prometheus ops symbiote` |
| `/tools` | `/tools` | `/prometheus-tools` | `/prometheus core tools` |
| `/unqueue` | `/unqueue` | `/prometheus-unqueue` | `/prometheus session unqueue` |
| `/voice` | `/voice` | `/prometheus-voice` | `/prometheus core voice` |
| `/wiki` | `/wiki` | `/prometheus-wiki` | `/prometheus core wiki` |
| `/workspace` | `/workspace` | `/prometheus-workspace` | `/prometheus core workspace` |
| `/xai` | `/xai` | `/prometheus-xai` | `/prometheus provider xai` |

## Not available in web chat — 25

`WEB_NATIVE_ONLY` in `web/slash_router.py`. These are handled by the
chat gateways but deferred in Beacon's web chat.

| Command |
|---|
| `/approve` |
| `/audit` |
| `/benchmark` |
| `/claude` |
| `/deepseek` |
| `/deny` |
| `/escalations` |
| `/gemini` |
| `/gepa` |
| `/glm` |
| `/gpt` |
| `/grok` |
| `/kimi` |
| `/local` |
| `/mimo` |
| `/pairs` |
| `/pending` |
| `/press` |
| `/qwen` |
| `/route` |
| `/start` |
| `/symbiote` |
| `/tools` |
| `/voice` |
| `/xai` |
