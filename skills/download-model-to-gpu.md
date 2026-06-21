---
name: download-model-to-gpu
description: Download a GGUF model from Hugging Face onto the GPU box (RTX 4090) as a managed background task. Use when asked to download/get/fetch/pull a model onto the 4090.
---

# Download a model to the GPU box

Models live on the GPU box `oara-4090@100.110.140.39` under `~/models/` (see [[check-local-gpu-models]] to list what's there). This machine (the mini) is NOT where they go.

## Do NOT use
- `download_file` tool — caps at 100 MB, writes to the mini, can't target the 4090.
- A bare `bash` call — bounded by the ~300s tool timeout; a multi-GB pull won't finish.

## Steps
1. **Find the file on Hugging Face.** Prefer the Unsloth GGUF repo (`unsloth/<Model>-GGUF`). For a 24 GB card pick the **UD-Q4_K_XL** quant (~22 GB, leaves KV-cache headroom). Note the exact filename and the `resolve/main` URL.
2. **Check free space first** (need filesize + margin):
   ```bash
   ssh oara-4090@100.110.140.39 'df -h ~/models | tail -1'
   ```
3. **Launch as a managed background task** (survives the turn; pings on completion). Use `task_create` with `type=local_bash` and EXACTLY this command shape:
   ```
   ssh oara-4090@100.110.140.39 'wget -c -O ~/models/<NAME>.gguf https://huggingface.co/<repo>/resolve/main/<NAME>.gguf'
   ```
   Set `on_complete='both'` so it notifies on Telegram AND re-engages you with the result. `wget -c` resumes if interrupted.
4. **Monitor / verify.** Poll `task_output` for progress; when done, confirm the byte size matches the HF file size:
   ```bash
   ssh oara-4090@100.110.140.39 'ls -l --block-size=1 ~/models/<NAME>.gguf'
   ```

## Why the command shape is strict
The SecurityGate auto-approves this download at system trust (so a background task needs no human approval) ONLY for the exact shape above: `ssh oara-4090@100.110.140.39` running a single `wget -O ~/models/<...>.gguf <huggingface.co URL ending .gguf>`. Any deviation — a different host, a non-`huggingface.co` URL, a destination outside `~/models/`, or command chaining (`;`, `&&`, `|`) — falls back to requiring approval and a background task will be blocked. Keep it to one clean `wget`.
