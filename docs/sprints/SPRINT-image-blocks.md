# Sprint — Images reach the model as images

**Status:** Spec v1. Spec only, no code yet.
**Scope guard:** The INBOUND path only — a picture a human sends reaching the
model that answers. Not artifact rendering, not image generation, not the
Beacon gallery (Beacon roadmap #4), not voice.

---

## 0. What prompted it

A user sent a screenshot to a Beacon session routed to `qwen3.8-max`. The reply
came back minutes later, enormous, and answered a paraphrase rather than the
picture. Measured from that user's cache — the four images ever uploaded, and
what the daemon stored for each:

| when | stored as | chars | ≈ tokens |
|---|---|---|---|
| 2026-08-25 | `[Image: This image is a screenshot of a chat-based AI assistant applic…` | 5,435 | ~1,400 |
| 2026-08-18 | `[Image: This is a screenshot of a desktop application interface, likel…` | **12,394** | **~3,100** |
| 2026-08-18 | `[Image: This image displays a snippet of text, likely from a code revi…` | 7,489 | ~1,900 |
| 2026-08-15 | `[Image: This is a screenshot of a desktop application called **Beacon**…` | 11,598 | ~2,900 |

For comparison, sending the picture itself to a model that accepts one costs
`⌈w/28⌉ × ⌈h/28⌉` visual tokens — a 1920×1080 screenshot is 2,691 tokens at
full resolution, 1,560 downscaled; a 4K screenshot caps at 4,784.

**The paraphrase costs as much as the image, or more, and it is lossy.** The
12,394-char description of a screenshot cost more tokens than sending the 4K
screenshot would have. It also spends a separate vision inference pass to
produce, and — because it is stored as a normal user message — it re-enters
context on every later turn of that conversation, forever. The model cannot go
back and look at a detail the description did not happen to mention.

## 1. What exists today (verified read-only, 2026-08-25)

**The conversation model has no image type.** `engine/messages.py`:

```python
ContentBlock = Annotated[TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")]
```

`ApiMessageRequest.messages` is `list[ConversationMessage]`. Every provider
serialises from those three block types and nothing else.

**So uploads are flattened to text at the gateway.** `ws_server.py:343` →
`_describe_image` → `gateway/media_services.describe_image` → `VisionTool`, and
the result is injected as `user_text = f"[Image: {desc}]"`. Telegram, Discord
and Beacon share this one path — which is why it exists: it is what lets a
text-only local primary handle an image at all.

**The description is tuned for maximum length.** `media_services.py:47` asks
`"Describe this image in detail."` with `max_tokens=2000`
(`tools/builtin/vision.py:127`). The measurements above are that ceiling.

**There is already a vision capability flag, and it is llama-only.**
`providers/base.py:81` declares `supports_vision: bool = False`; the only
writer is `llama_cpp.py:166`, from an mmproj probe. No cloud provider sets it.

**⚠ VisionTool bypasses the type it is declared against.** It hand-builds
OpenAI-style dicts and passes them as `messages`:

```python
messages_payload = [{"role": "user", "content": [
    {"type": "text", "text": arguments.question},
    {"type": "image_url", "image_url": {"url": data_url}},
]}]
request = ApiMessageRequest(model="", messages=messages_payload, ...)
```

Those are `dict`, not `ConversationMessage`. It works only because the
OpenAI-compatible providers pass `messages` through to the wire. **The
anthropic provider iterates `msg.role` and `msg.content` on each element**
(`anthropic.py:358`), so a raw dict raises `AttributeError` — vision through
the anthropic provider cannot work today. That is a live bug this sprint
subsumes rather than patches: the fix is the typed block, not another dict.

## 2. Goal

A picture a human sends reaches a model that can see pictures, as a picture.
When the model cannot see pictures, today's description path runs unchanged.

**Non-goals (frozen):** image generation; artifact/outbound rendering; Beacon's
gallery; PDF or document blocks; changing what Telegram/Discord render; voice.

## 3. Design

### 3.1 `ImageBlock` in the conversation model

```python
class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    media_type: str          # image/png | image/jpeg | image/gif | image/webp
    data: str                # base64, no data: prefix, no newlines
    # Provenance for the fallback path and for history rendering.
    source_path: str | None = None   # the media_cache path, for re-read
    description: str | None = None   # filled iff the description path ran
```

Added to the `ContentBlock` union. Every existing consumer that pattern-matches
on block type must gain an explicit branch — a silent `else: pass` is how a
picture becomes invisible.

### 3.2 Per-provider serialisation

| provider | shape |
|---|---|
| `anthropic` | `{"type": "image", "source": {"type": "base64", "media_type": …, "data": …}}` |
| `openai_compat` (qwen, deepseek, kimi, glm, mimo, gpt) | `{"type": "image_url", "image_url": {"url": "data:<mt>;base64,<data>"}}` |
| `llama_cpp` / `ollama` | existing multimodal path, gated on the mmproj probe |

The two cloud shapes are NOT interchangeable — that difference is exactly what
makes VisionTool's raw dict wrong today.

### 3.3 Capability gating, and what happens when it says no

`supports_vision` becomes the single gate, and it must be set by every provider
rather than one. Cloud providers cannot probe, so the value comes from the
model catalog, not from the wire: `qwen3.8-max` is a text model and
`qwen-vl-max` is not, and only the preset knows which is configured.

At upload time the gateway asks the provider that will actually answer:

- **can see** → attach an `ImageBlock`; no vision pass, no `[Image: …]` text
- **cannot see** → today's path exactly, byte-identical

The check must run against the provider the TURN will use — i.e. after the
per-session override resolves, not against the process primary. Getting this
wrong is the same class of bug as #74 (`session_id` threaded per call), and it
fails in the same silent direction: an image attached for a model that cannot
read it is a dropped message, not an error.

### 3.4 History and persistence

An `ImageBlock` is stored in `content_json` like any other block, with the
base64 payload replaced by a `source_path` reference into the media cache — the
bytes live once on disk, not once per turn per conversation. On read, blocks
re-hydrate from that path; a missing file degrades to the `description` field
if one exists, else to `[Image: unavailable]`.

**Open question (Q1):** should a `description` be generated *as well* for a
vision-capable turn, so that a later turn on a text-only model can still see
something? Cheap insurance, an extra inference pass per upload. Recommendation:
no in v1 — measure how often a session changes model mid-conversation first.

### 3.5 Beacon and the wire

Beacon already renders an inline thumbnail optimistically and loses it on
reload, because the daemon had nothing to give back (`shared/parts.ts:19`).
With an `ImageBlock` in history the thumbnail becomes real: the wire carries
the block, Beacon renders it from `content_json` like any other part, and the
disappearing-screenshot behaviour goes away as a side effect rather than as a
patch. Telegram and Discord already send images; their rendering does not
change.

## 4. Phasing

**Phase 1 — the typed block, anthropic only.** `ImageBlock`, the union, the
anthropic serialiser, `supports_vision=True` for anthropic, capability gate at
the upload site, description fallback untouched for everything else. Ends with
a picture reaching Claude as a picture and every other provider byte-identical.

**Phase 2 — openai_compat.** The `image_url` shape plus per-preset vision flags
in the model catalog (`qwen-vl-*` vs `qwen3.8-max`), so the gate can be honest
per configured model.

**Phase 3 — history, Beacon, and the cache reference.** `content_json`
round-trip, `source_path` re-hydration, Beacon rendering from history.

**Phase 4 — retire the always-on description.** Only once the gate is proven
correct in the wild. Until then the description path is the default and the
image path is the exception, not the reverse.

## 5. Test plan

Each phase lands with tests that fail before it:

- a `ConversationMessage` carrying an `ImageBlock` serialises to the
  documented anthropic shape, byte-for-byte
- the same message to a `supports_vision=False` provider takes the description
  path and produces the identical `[Image: …]` text as today
- the capability check reads the provider the TURN resolved to, not
  `context.provider` — the #74 shape, asserted with a session override set
- a raw-dict `messages` payload (VisionTool's current shape) is rejected by
  type, so the bypass cannot come back
- history round-trip: block → `content_json` → block, with the bytes on disk
  once and a missing file degrading rather than raising

## 6. Open questions

- **Q1** (§3.4) description alongside the image for mid-conversation model switches
- **Q2** where the per-preset vision flag lives — `OVERRIDE_PRESETS`, or the
  models catalog `/api/models` already serves
- **Q3** size policy: downscale before send (cheaper, lossy) or send as-is and
  let the provider downscale. Anthropic caps at 8000×8000 and 10 MB base64;
  a screenshot is well inside that, so v1 sends as-is
- **Q4** whether `[Image: …]` text should remain in history for turns that
  used the image path, for grep/search (interacts with Beacon search #1)
