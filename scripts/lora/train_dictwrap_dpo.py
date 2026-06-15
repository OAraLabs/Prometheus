#!/usr/bin/env python3
"""QLoRA DPO train: a dict-wrap specialist LoRA for gemma-4-26b-a4b.

Runs on the 4090 in ~/lora-train-venv (peft/trl/datasets/bitsandbytes on
torch 2.11+cu130). The daemon/llama-server must be STOPPED first — the 24 GB is
needed for the 4-bit base + LoRA + DPO reference.

Data: the gym pairs export ({prompt: lcm_ref, chosen: <unwrapped call>,
rejected: <dict-wrapped call>}). The lcm_ref carries only the tool schema, not
the natural-language ask, so we reconstruct a faithful request from the chosen
args (per-tool) and present [system+tool, user] as the DPO prompt. The learned
signal is chosen≻rejected — identical prompt, args unwrapped vs wrapped. The
held-out eval (Phase 3), not this script, decides whether it generalized.

Target modules: attention q/k/v/o only (shared across the 128 experts → MoE-safe;
matched by suffix so nesting under a multimodal wrapper is fine).

  --dry-run  : load model, attach LoRA, run ONE DPO step. Validates the brand-new
               gemma4 arch + the chat-template rendering BEFORE the full train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# ── faithful request reconstruction (the lcm_ref dropped the NL ask) ─────────


def reconstruct_request(name: str, args: dict) -> str:
    a = args
    if name == "task_create":
        if a.get("command"):
            return f"Create a background task that runs this exact shell command: {a['command']}"
        if a.get("prompt"):
            return f"Create a background agent task whose goal is: {a['prompt']}"
        return f"Create a background task: {a.get('description','(unspecified)')}"
    if name == "task_list":
        return f"List the background tasks that currently have status '{a.get('status','')}'."
    if name == "grep":
        if a.get("root"):
            return f"Search for the pattern '{a.get('pattern','')}' under the directory {a['root']} and report matches."
        return f"Search the whole workspace for the pattern '{a.get('pattern','')}'."
    if name == "task_get":
        return f"Get the full details of background task {a.get('task_id','')}."
    if name == "cron_create":
        return f"Create a cron job named '{a.get('name','')}' on schedule '{a.get('schedule','')}' that runs: {a.get('command','')}"
    if name == "download_file":
        return f"Download the file at {a.get('url','')}."
    if name == "task_update":
        return f"Update background task {a.get('task_id','')}: set its status note to '{a.get('status_note','')}'."
    return f"Use the {name} tool to handle the request."


def render(tok, schema: dict, name: str, chosen_args: dict, rejected_args: dict):
    """Return (prompt, chosen, rejected) text via the model's own chat template,
    so the only diff between chosen and rejected is unwrapped vs wrapped args."""
    sys_msg = (
        "You are a tool-calling assistant. Call tools with arguments that match "
        "the schema exactly — scalar parameters take scalar values, never a "
        f"nested object.\nTool: {json.dumps(schema)}"
    )
    user = reconstruct_request(name, chosen_args)
    base = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]

    def call_msg(arguments):
        return {"role": "assistant",
                "tool_calls": [{"type": "function",
                                "function": {"name": name, "arguments": arguments}}]}

    try:
        prompt = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
        full_c = tok.apply_chat_template(base + [call_msg(chosen_args)], tokenize=False)
        full_r = tok.apply_chat_template(base + [call_msg(rejected_args)], tokenize=False)
        if full_c.startswith(prompt) and full_r.startswith(prompt) and full_c != full_r:
            return prompt, full_c[len(prompt):], full_r[len(prompt):]
    except Exception:
        pass
    # fallback: plain text — schema+request prompt, raw call JSON as completion
    prompt = f"{sys_msg}\n\nRequest: {user}\nTool call:"
    mk = lambda args: " " + json.dumps({"name": name, "arguments": args})
    return prompt, mk(chosen_args), mk(rejected_args)


def build_dataset(pairs_path: Path, tok) -> Dataset:
    rows = []
    skipped = 0
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("rejected") is None:
            skipped += 1
            continue
        chosen = json.loads(r["chosen"])
        rejected = json.loads(r["rejected"])
        ref = json.loads(r["prompt"]) if r["prompt"].lstrip().startswith("{") else {}
        schema = ref.get("tool_schema", {"name": chosen.get("name")})
        name = chosen.get("name") or schema.get("name")
        ca = chosen.get("input", chosen.get("arguments", {}))
        ra = rejected.get("input", rejected.get("arguments", {}))
        if ca == ra:  # not a real preference (no wrap diff) — drop
            skipped += 1
            continue
        p, c, rj = render(tok, schema, name, ca, ra)
        rows.append({"prompt": p, "chosen": c, "rejected": rj})
    print(f"[data] {len(rows)} DPO triples ({skipped} skipped)")
    return Dataset.from_list(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--base", required=True, help="HF base model dir (safetensors)")
    ap.add_argument("--out", required=True, type=Path, help="adapter output dir")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true", help="1 step — validate arch + rendering")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_dataset(args.pairs, tok)
    if len(ds) == 0:
        print("[fatal] no DPO triples — aborting")
        return 1
    # show one rendered triple so the format is auditable in the log
    print("[sample] prompt tail:", repr(ds[0]["prompt"][-160:]))
    print("[sample] chosen     :", repr(ds[0]["chosen"][:160]))
    print("[sample] rejected   :", repr(ds[0]["rejected"][:160]))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print(f"[model] loading {args.base} in 4-bit …")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      bias="none", task_type="CAUSAL_LM")

    cfg = DPOConfig(
        output_dir=str(args.out), beta=args.beta,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=1 if args.dry_run else -1,
        max_length=args.max_len, max_prompt_length=args.max_len // 2,
        logging_steps=5, save_strategy="no", bf16=True,
        gradient_checkpointing=True, report_to=[])
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    print(f"[train] {'DRY RUN (1 step)' if args.dry_run else 'full'} starting …")
    trainer.train()
    if not args.dry_run:
        trainer.save_model(str(args.out))
        tok.save_pretrained(str(args.out))
        print(f"[done] adapter → {args.out}")
    else:
        print("[dry-run OK] gemma4 loaded, LoRA attached, 1 DPO step ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
