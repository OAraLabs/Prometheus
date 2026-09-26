# Model ladder

Accuracy = pass ÷ (pass + fail). *answer line missing* = replies without their `ANSWER:` line ÷ replies asked for one; those are scored where the bare last line still gives the answer and otherwise recorded as format misses, never as wrong.

| model | quant | tier | accuracy (95% CI) | answer line missing | qa acc / miss | single_tool acc / miss | multi_step acc / miss | file_edit acc / miss |
|---|---|---|---|---:|---|---|---|---|
| Qwen3.8-27B-UD-Q4_K_XL.gguf (`r27b-smoke-20260926`) | UD-Q4_K_XL | light | 49/51 (0.87–0.99) | 0/31 | 15/15 / 0/15 | 13/15 / 0/13 | 12/12 / 0/3 | 9/9 / — |
| Ternary-Bonsai-2-27B-PQ2_0.gguf (`r27b-pq2-smoke-v2-20260925`) | PQ2_0 | full | 40/48 (0.70–0.91) | 3/32 | 13/14 / 1/14 | 13/13 / 2/15 | 8/12 / 0/3 | 6/9 / — |
| Qwen3.5-9B-UD-Q4_K_XL.gguf (`r08b-smoke-20260926`) | UD-Q4_K_XL | light | 49/51 (0.87–0.99) | 0/32 | 14/15 / 0/15 | 14/15 / 0/14 | 12/12 / 0/3 | 9/9 / — |
| Ornith-1.5-9B-Q4_K_M.gguf (`r09b-ornith-smoke-v2-20260926`) | Q4_K_M | full | 38/48 (0.66–0.88) | 3/33 | 13/14 / 1/15 | 13/13 / 2/15 | 9/12 / 0/3 | 3/9 / — |
