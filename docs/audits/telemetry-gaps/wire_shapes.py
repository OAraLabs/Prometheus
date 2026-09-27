"""What each provider actually sends back about usage, from the recorded parity exchanges.

    python3 docs/audits/telemetry-gaps/wire_shapes.py [tests/fixtures/parity]

The parity traces hold every completion response exactly as the model server
sent it (synthetic prompts, public files). This scan reads the usage-bearing
chunk of each streamed completion and prints, per upstream, the usage keys and
the cache fields it carries — the evidence for which NULL cache count in
``subsystem_runs`` is a correct "the provider said nothing" and which is a
value the pipeline dropped (docs/audits/TELEMETRY-GAPS.md, T6–T8).

Upstream labels: ``primary`` is the 4090's llama.cpp, ``alt`` the mini's
Ollama, ``hosted`` the Anthropic API. Runs on any checkout; prints key names
and token counts only.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

COMPLETIONS = ("/v1/chat/completions", "/chat/completions", "/v1/messages")


def usage_chunks(body: str):
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        usage = chunk.get("usage")
        if usage is None and isinstance(chunk.get("message"), dict):
            usage = chunk["message"].get("usage")
        if isinstance(usage, dict) or isinstance(chunk.get("timings"), dict):
            yield chunk, usage or {}, chunk.get("timings") or {}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/parity")
    shapes: collections.Counter = collections.Counter()
    cached_sum: collections.Counter = collections.Counter()
    prompt_sum: collections.Counter = collections.Counter()
    anthropic: collections.Counter = collections.Counter()
    served_echo: collections.Counter = collections.Counter()
    for path in sorted(root.glob("*.trace.json")):
        trace = json.loads(path.read_text())
        for ex in trace.get("exchanges", []):
            if ex.get("path") not in COMPLETIONS or "data: " not in (ex.get("body") or ""):
                continue
            label = ex.get("upstream", "?")
            for chunk, usage, timings in usage_chunks(ex["body"]):
                details = usage.get("prompt_tokens_details")
                cache_fields = sorted(
                    k for k in ("cache_read_input_tokens", "cache_creation_input_tokens",
                                "prompt_cache_hit_tokens", "cached_tokens") if k in usage)
                if isinstance(details, dict):
                    cache_fields += [f"prompt_tokens_details.{k}" for k in sorted(details)]
                if timings:
                    cache_fields += [f"timings.{k}" for k in ("cache_n", "prompt_n") if k in timings]
                shapes[(label, ex["path"], tuple(sorted(usage)), tuple(cache_fields))] += 1
                if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
                    cached_sum[label] += details["cached_tokens"]
                    prompt_sum[label] += int(usage.get("prompt_tokens") or 0)
                if label == "hosted" and chunk.get("type") == "message_start":
                    anthropic["input_tokens"] += int(usage.get("input_tokens") or 0)
                    anthropic["cache_read_input_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
                    anthropic["cache_creation_input_tokens"] += int(
                        usage.get("cache_creation_input_tokens") or 0)
            # Does the server name the model it served? (the served_model column)
            first = next((json.loads(line[6:]) for line in ex["body"].splitlines()
                          if line.startswith("data: {")), {})
            names_model = bool(first.get("model") or (first.get("message") or {}).get("model"))
            served_echo[(label, names_model)] += 1

    print("== usage-bearing chunks, by upstream: usage keys | cache fields present")
    for (label, p, keys, cache), n in sorted(shapes.items()):
        print(f"   {n:3}  {label:8} {p:22} keys={','.join(keys)}")
        print(f"        cache fields: {', '.join(cache) or 'NONE'}")
    print("== cache counts on the wire (OpenAI shape: prompt_tokens includes the cached part)")
    for label in sorted(set(cached_sum) | set(prompt_sum)):
        share = cached_sum[label] / prompt_sum[label] if prompt_sum[label] else 0.0
        print(f"   {label:8} cached {cached_sum[label]:>8,} of {prompt_sum[label]:>8,} prompt tokens ({share:.0%})")
    if anthropic:
        whole = sum(anthropic.values())
        print("== Anthropic message_start (input_tokens EXCLUDES both cache counters)")
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            print(f"   {key:28} {anthropic[key]:>8,}")
        print(f"   whole prompt                 {whole:>8,}; input_tokens alone is "
              f"{anthropic['input_tokens'] / whole:.1%} of it")
    print("== completions whose first chunk names the served model")
    for (label, named), n in sorted(served_echo.items()):
        print(f"   {label:8} names the model: {named}  ({n} completions)")


if __name__ == "__main__":
    main()
