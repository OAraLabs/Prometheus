# Distillation + flywheel: open questions

Never blocking. Each item is skipped and work continues; the operator answers when back.

1. **4060 Ti host unreachable.** SSH to the 4060 Ti box isn't permitted yet, so its `flywheel` role (small local model for voice/extractor/fallback while the business GPUs are swapped) is skipped. Without it, the business stack has no local fallback during a swap.
2. **4090 GPU service is customer-facing.** A business tunnel on the 4090 points at its only `llama-server`, and there's no fallback while the 4060 Ti is offline. Per the runbook it stays running, so `flywheel` can't be applied on the 4090. Options: OK a planned outage of that tunnel during windows; bring the 4060 Ti online as its fallback; or move the tunnel's backend elsewhere.
3. **4090 nights are booked.** Another approved overnight job (proxy up since 2026-10-25) uses the 4090's `llama-server` 22:00–06:30. Which wins on the 4090 at night, and from when?
4. **3090 Ti backs the live daemon.** Saluki is the daemon's primary until its test ends ~2026-11-02. `flywheel` would move the daemon to the Max API, which means editing its model config and config pins on the live host. Is that allowed during windows, or should 3090 Ti work wait until the Saluki test ends?
5. **4090 disk.** 75 GB free (runbook asks ~100 GB). `~/models` is 186 GB, including several old GGUFs. Which can be removed (or moved to another disk)?
6. **Qwen3.8 Max for training data.** Before Phase 2/2b, confirm the Max plan's terms allow using its outputs (judging and teacher traces) to train a local model, and which key is meant. Until confirmed, Phase 2 judge calls and Phase 2b teacher generation are skipped; Phase 3 uses existing labels with the tool-success fallback.
