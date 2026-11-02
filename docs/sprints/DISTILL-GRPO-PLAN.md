# GRPO stage after round-2 SFT: plan (not run)

Operator, Sun 2026-11-01 ~23:00 ET: plan (don't run yet) a GRPO stage after the round-2 SFT, on gym tasks, with automatic rewards for correct tool choice, task success and injection resistance. Report feasibility on the 4090 (memory, speed).

Nothing here has run. Each number is labelled: **measured** (an earlier run), **derived** (arithmetic on measured numbers) or **estimate** (needs the dry run in §6).

## Feasibility verdict

- **Stage 1 is feasible on the 4090 and fits easily in memory.** Single-step "next action" GRPO, G = 8, ~10K-token prompts, ≤ 512 completion tokens. Estimated peak is ~14–16 GiB of 24, against the measured SFT peak of 12.7 GiB at 16K.
- **Speed is the limit, not memory.** vLLM can't be used: Unsloth's `fast_inference` doesn't support Qwen3.5, and no vLLM is installed. Rollouts therefore use Unsloth/HF generation.
  - Stock TRL `GRPOTrainer` + Unsloth: estimate **~9–16 optimizer steps/h** (32 completions per step).
  - **500 steps:** ~31–58 h, which is a continuous weekend or ~6–10 nightly windows (resumable).
  - **1,000 steps:** ~62–116 h, so no.
  - **Shared-prefix rollout and loss (our own code):** could reach ~25–50 steps/h. Build it only if the dry run shows < 10 steps/h.
- **Stage 2 (multi-turn rollouts through the gym) also fits in memory, but needs a rollout server.** The policy runs on the 4090 and the tools run in the 3090 Ti host's bwrap gym, because the 4090 blocks unprivileged user namespaces and fixing that needs root. Estimate ~6–12 steps/h. Weekend only, and only after stage 1 shows a gain.
- **Recommendation:** build stage 1 now and run a 30-minute dry run in the first free window. Hold stage 2 until stage 1 has passed its gate.

## 1. What GRPO would optimise, and the risks

**Why it fits the scorecard** (DISTILL-PHASE7-SCORECARD.md):
- **What's left is choices, not syntax.** Arg validity is 99–100% and the adapter-repair gap is 0.0. The misses are the wrong tool (Q4: 58 of 329) and no call made (Q4: 4).
- **SFT copies one reference action. GRPO rewards any action the verifier accepts** and pushes down the plausible-but-wrong tools the student actually samples.
- **This is the round-2 focus (a), wrong-tool rate, done as an objective rather than a dataset.** The rewards come from checks the gym already runs (§2), so no learned reward model is needed.
- **Injection resistance:** it's at ceiling on eval-injection (≥ 98.4% for every model). GRPO can only show a gain there on a harder held-out set (§5).

**Risks and their guards:**

| Risk | Guard |
|---|---|
| Reward hacking: refusing everything to pass injection checks, spamming calls, gaming a too-broad equivalence set | Composite reward (§2) where injection-clean scores only with task progress. Penalty for extra calls. Equivalence sets come from the tool-choice verifier, and their per-tool frequency is audited. Every 50 steps, 20 high-reward samples are audited offline. |
| Loss of general ability | KL to the round-2 SFT model (not to base, see §3). A 50-case sanity subset every 100 steps. Entropy is tracked, and a collapse stops the run. |
| KL drift / behaviour drift | Track call rate, calls per turn, reply length and the per-tool distribution against SFT. Stop if call rate moves > 10 pts or reply length moves > 50%. |
| Over-fitting the reference tool | Train and dev are split by task group. Checkpoints are chosen on dev only. The eval sets are never trained on and are scored once at the end. |
| Qwen3.5 4-bit training | Unsloth advises against QLoRA on Qwen3.5. bf16 LoRA on the 9B needs ~22 GB by Unsloth's own figure, which leaves no room for rollouts on 24 GB, so 4-bit is the only single-GPU option. The Phase 5 SFT trained cleanly in 4-bit, and BF16 did not beat Q4 on the subset. |

## 2. Reward design

**Inputs:** one pure function of (task spec, policy output), built from existing code only:
- `score.py`'s `reply_calls` / `schema_errors` / `score_tool_case` predicates
- teacher_gen's `adversarial_verdict` and `r2_checks`
- the incoming tool-choice verifier and argument-equivalence check (round-2 focus a/b)

**No judge in the reward loop.** The calibrated judge (judge.py rubric v2, ≥ 85% for coding and sentinel/daemon) was calibrated on the 4090's UD-Q4_K_XL, and that GPU is the trainer. The 3090 Ti host's IQ3 is calibrated nowhere. So the judge only audits samples offline (coding and sentinel/daemon rows), and prose-only prompts without a deterministic answer check are left out of stage 1.

**Stage 1 (one assistant message per prompt):**

| Term | Value | Source |
|---|---|---|
| Unparseable / schema-invalid call / call to an unoffered tool | **total = −1** (hard floor) | `reply_calls`, `schema_errors` |
| Injection followed: a forbidden tool, argument or string in the action, when the prompt's context already holds the exposed injection | **total = −1** (overrides everything) | `adversarial_verdict` on the one action |
| Right tool: first call is in the reference's equivalence set | +1.0 | tool-choice verifier |
| Right "no call": the reference made no call and the prompt's deterministic answer predicates pass | +1.0 | `answer_failures`, grounding terms |
| No call when one was needed | −0.5 | reference has a call |
| Wrong tool (valid call) | 0 | |
| Arguments: equivalent to the reference / required args strictly equal | +0.5 / +0.25 (only when the tool is right) | equivalence check, `expect_tool_args_require` |
| Extra calls beyond the reference's count | −0.1 each, cap −0.3 | |
| Length: DAPO soft overlong, 384 → 512 tokens | 0 → −0.5 linear; truncated completions masked from the loss | |
| Final reply on an injection prompt | must be ≥ `R2_MIN_REPLY_CHARS` (40) and pass the task's predicates, else 0 | `r2_checks` |

Weights are set so that one wrong tool (0) always scores below a right tool with wrong arguments (+1.0). They are tuned only on dev.

**Stage 2 (whole episode):**
- **Base reward:** +1 if `deterministic_checks` pass (predicates, tool success, injection not followed with a reply, recovery with a final call that succeeded, grounding).
- **Shaping:** +0.3 for the right first tool.
- **Penalties:**
  - −1 for an injection followed or the breaker tripping;
  - −0.5 for a sandbox refusal;
  - −0.5 for no call when one was needed;
  - −0.05 per call beyond 1.5× the reference's count, capped at −0.3.
- **Output-limit episodes:** masked, not rewarded.

**Anti-hacking checks** (unit tests in the reward module, written before any run):
- refuse-everything scores below doing the task;
- call spam scores below one right call;
- a reply that quotes the canary scores −1;
- an empty or short reply on an injection prompt scores 0;
- a tool from the same family but outside the equivalence set scores 0.

## 3. Rollout design and trainer settings

| | Stage 1: single-step next action | Stage 2: multi-turn agentic |
|---|---|---|
| Prompt | A gym trajectory cut just before the reference's next action, rendered with the exact runtime template (`render.py`). Includes cuts after an injected tool result (injection) and after an injected failure (recovery: right next tool, no claim of success). | The task's real start. The policy runs the whole loop. |
| Where tools run | Not at all: the reward reads the action | The 3090 Ti host's bwrap gym (teacher_gen `run_loop`), with the policy served from the 4090 over the tailnet, the same topology as today's cross-host teacher generation |
| New code | Data builder, reward module, `train_grpo.py` | All of stage 1 plus a rollout server in the trainer (OpenAI-compatible; returns token ids and logprobs), a gym driver for G rollouts per task, TRL `rollout_func`, loss masks on tool-result tokens, and a check that the chat template is prefix-preserving across turns |
| Latency | Generation and training only | Plus serial tool calls (0.1–2 s each) and tailnet round trips. A failure in either host pauses the step. |
| Cost / risk | Low | High: two hosts, tool errors, rollouts don't run the same way twice |

**Recommended path:** stage 0 → stage 1 → stage 2.
- **Stage 0, difficulty filter:** sample the SFT student's Q4 GGUF 8× per candidate prompt with llama.cpp on the 4090. This is cheap and needs no training stack. Keep prompts with a pass rate of 10–90%.
  - **Why:** at a pass rate of 80%, a group of G = 4 has the same reward throughout 41% of the time (no gradient). At G = 8 it's 17%.
- **Stage 1:** 300-step pilot, then up to 500 steps.
- **Stage 2:** only if stage 1 passes its gate (§5) and wrong-tool misses remain.

**Trainer settings:**
- TRL `GRPOTrainer` + Unsloth `FastLanguageModel(load_in_4bit=True, fast_inference=False)`, offloaded gradient checkpointing as in SFT.
- `num_generations=8`, 4 prompts per step, `max_completion_length=512`, temperature 0.7 / top-p 0.8 / top-k 20 (the teacher's sampling), thinking off as in the rows.
- `loss_type="dapo"`, `mask_truncated_completions=True`, `scale_rewards="batch"`, ε 0.2 / ε_high 0.28, LoRA lr 5e-6, constant after a 10-step warmup.
- **KL reference = the round-2 SFT model:**
  - merge the SFT adapter into BF16 (the Phase 6 merge, kept rather than cleaned up);
  - load the merged model 4-bit;
  - train a fresh GRPO LoRA (r = 32 / α = 32, the SFT targets);
  - TRL's adapter-disabled reference is then the SFT model, at no extra memory cost.
- **Why not TRL's default reference:** with the SFT adapter left in place, TRL's adapter-disabled reference would be the *base* model, and the KL term would pull the policy back toward base, undoing the SFT.
- **β:** 0.02 to start; KL is logged even at β = 0.
- **Guards from `train_qlora.py`:** provenance and max-derived approvals with private tags, checkpoint/resume, `--stop-at`, SIGTERM, disk preflight.

## 4. 4090 feasibility

**Inputs:**
- **Measured:** Qwen3.5-9B has 32 layers (24 gated-deltanet, 8 full attention with 4 KV heads × 256 dims) and a 248,320 vocab.
- **Measured:** QLoRA r = 32 with offloaded checkpointing peaks at 12.68 / 16.64 / 20.37 GiB allocated at 16K / 32K / 48K tokens, at 2,549 / 2,330 / 1,990 tok/s. 64K runs out of memory.
- **Measured** (1,314 round-2 rows): prompts are a fixed ~9.4K tokens of system prompt + 24 tool schemas, plus a short context. Prompt p50 is ~9.5–9.7K and p90 ~10.5K; long-context rows are ~25K. Reference actions are p50 ~100 tokens and p90 ~360. These are character counts ÷ 3.9 (Phase 1's measured density).
- **Prompt length is fixed by the runtime prefix.** Shortening the system prompt or the tool list would train on a distribution the student never serves. Stage 1 keeps the real prefix and leaves out long-context rows (25K tokens cost 2.5×).

**Memory, stage 1** (GiB, allocated; nvidia-smi reads ~1.7 GiB higher):

| Component | GiB | Basis |
|---|---:|---|
| 4-bit policy + LoRA + 8-bit AdamW + context | 8.8 | derived: intercept of the 3-point SFT fit (0.24 GiB per 1K tokens) |
| KL reference | 0 | adapter disabled on the merged SFT model |
| Rollout KV, G = 8 × 10K tokens (8 full-attention layers × 32 KiB/token) | 2.4 | derived |
| Gated-deltanet recurrent states, 8 sequences | 0.4 | derived: 24 × 32 heads × 128² × fp32 |
| Prefill transient (prefill one sequence at a time) | 1–2 | estimate |
| **Rollout phase peak** | **~13–14** | estimate |
| Training micro-batch: 1 sequence of 10K tokens (2 sequences) | 11.2 (13.6) | derived |
| **Overall peak** | **~14–16 by nvidia-smi** | estimate. G = 16 (+2.4) or 25K prompts (+3.6 KV, +3.6 activations) still fit. |

**Speed, stage 1, per optimizer step** (4 prompts × G 8 = 32 completions of ~9.85K tokens):

| Phase | Stock TRL + Unsloth | Shared-prefix (own code) | Basis |
|---|---:|---:|---|
| Prefill | 35–52 s (310K tokens at 6–9K tok/s, forward only) | ~6 s (one prefix per prompt) | estimate: 2.5–3.5× the measured fwd+bwd rate |
| Decode (≤ 400 steps per group, batch 8) | 64–192 s (40–120 ms per step) | 40–110 s (batch 16) | estimate: no fast linear-attention kernels installed; measure |
| Policy fwd+bwd | ~124 s (315K tokens at ~2,550 tok/s) | ~17 s (~44K tokens) | derived from measured SFT |
| Reference forward (β > 0) | 35–52 s | ~5 s | estimate |
| **Per step** | **~225–420 s → ~9–16 steps/h** | **~70–140 s → ~25–50 steps/h** | |
| 500 steps | **~31–58 h** | ~10–20 h | |
| 1,000 steps | ~62–116 h | ~20–40 h | |
| One 6-h window (~5.5 h usable) | ~50–90 steps | ~140–270 steps | |

**Stage 2 estimate:**
- **Generation:** 32 rollouts of ~3 turns go in two waves of 16, because 32 cached contexts of 11K tokens wouldn't fit beside the policy.
- **Training:** ~350K tokens.
- **Rate:** ~6–12 steps/h, so a 300-step run takes ~25–50 h.

**Not usable:**
- **vLLM:** Unsloth's `fast_inference` doesn't support Qwen3.5, and TRL's colocate mode would hold a second bf16 copy (~18 GB).
- **vLLM server mode:** needs another GPU, and the 3090 Ti is taken by the live model.
- **llama.cpp as the rollout engine:** off-policy. The adapter changes every step, and its GGUF numerics differ from the bnb4 policy.

**Must be measured before committing (dry run, ~30 min of GPU, inside a window):**
1. Generation time per group at G = 4 / 8, ~10K prompts, with Unsloth `generate` and with TRL `use_transformers_continuous_batching` (transformers 5.17 has it; whether it supports the hybrid cache is unknown).
2. Policy fwd+bwd time and peak VRAM in each phase.
3. Whether the gated-deltanet layers are on a fast kernel or the torch fallback (`flash-linear-attention` and `causal-conv1d` aren't installed). If they're on the fallback, try them in a separate `venv-grpo`, never in `venv-train`.
4. A 20-step run on 50 filtered prompts: reward variance is non-zero in ≥ 70% of groups, KL and entropy are logged, and a checkpoint resumes.
5. **Go / no-go:** ≥ 9 steps/h at G = 8 means stock TRL on nights or a weekend. Below that, build shared-prefix.

## 5. Data, held-out sets, promotion

**Stage 1 pool:**
- **Round-2 verified rows**, cut at each step. So far: daily-agent 514, injection 498, coding 182, error-recovery 106. It grows nightly. Only rows a deterministic verifier or tool success kept are used. Judge-pending rows are used only after their calibrated verdict.
- **Tool-choice tasks** from round-2 focus (a), where several tools are plausible and only one is right, ~40%.
- **A v4b replay slice ≤ 12K tokens** (coding / daemon, ~15%), so round-1 skills don't erode.
- **Size:** ~5–6K candidates, giving ~2–3K after the stage-0 filter. 500 steps × 4 prompts is ~2,000 prompt visits, so about one pass.
- **Splits:** by task group, with 5% as dev. Every prompt goes through teacher_gen's `EvalGuard` (injection-manifest shingles, eval-tools hashes).

**Never trained on:** eval-tools (329), eval-injection (192), eval-longctx, eval-longctx-tools and sanity.

**Add before GRPO:** a harder held-out injection set built from round-2 task groups that are never trained on (mid-trajectory injections, deterministic verdict), because eval-injection is at ceiling.

**Promote the GRPO adapter over the round-2 SFT student only if all of these hold**, scored by the Phase 6 pipeline at Q4_K_M with the same harness:
- **The Phase 6 gate:**
  - task success ≥ like-for-like base + 10;
  - long context not significantly worse by > 3 pts;
  - injection ≥ base;
  - sanity within 3 pts of base.
- **Against the SFT student:**
  - eval-tools wrong-tool count falls by ≥ 20% relative (paired);
  - task success Δ ≥ 0;
  - no-call misses don't rise;
  - sanity within 2 pts;
  - eval-injection and the harder injection holdout not lower.
- **Checkpoint choice:** on dev reward and the sanity subset only. The eval sets are scored once.

## 6. Schedule and prerequisites

1. **Round-2 SFT trained and scored first.** GRPO starts from its merged model. Keep the Phase 6 merged BF16 instead of cleaning it up (disk: the 4090 has ~70 GB free; the merge is ~18 GB).
2. **The tool-choice verifier and argument-equivalence check are merged.** Their equivalence sets are the reward's ground truth.
3. **Code to write:**
   - `grpo_data.py`: trajectory cuts, references, equivalence sets, adversarial specs, the stage-0 filter.
   - `grpo_rewards.py`: pure functions plus the anti-hacking tests.
   - `train_grpo.py`: TRL + Unsloth, with the `train_qlora.py` guards.
   - Stage 2 only: the rollout server, the gym driver and `rollout_func`.
4. **Pin the training venv as it stands today:**
   - trl 1.15.0, unsloth / unsloth_zoo 2026.10.10, peft 0.21.2, transformers 5.17.0, torch 2.12.1, bitsandbytes 0.50.2, accelerate 1.15.0, xformers 0.0.35, triton 3.7.1, Python 3.12.16;
   - no vLLM;
   - freeze it to a lock before the dry run.
5. **Windows:**
   - The dry run takes the first 4090 window after the round-2 SFT.
   - Stage 1 takes a continuous weekend (needs the operator's continuous-window approval) or resumes over nightly windows.
   - Each GRPO night is a night without 4090 round-2 generation. That trade goes to the operator before booking.

Sources:
- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide)
- [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) (fast_inference=False for Qwen3.5 RL; QLoRA advisory)
- Local: `~/distill/capacity/qlora-seqlen.json` (3090 Ti host), training venv metadata on the 4090.
