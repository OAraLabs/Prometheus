---
name: batch
description: "Run a skill or prompt against multiple inputs in parallel. Use when you need to apply the same operation across many files, items, or inputs — e.g., review 5 PRs, summarize 10 articles, or process a list of URLs."
---
<!-- Provenance: whieber1/Prometheus | src/reference_data/subsystems/skills.json (batch.ts reference) | MIT -->

# Batch

Run a prompt or skill against multiple inputs efficiently.

## Triggers
- "run this on all of these"
- "apply to each file in the list"
- "batch process these items"

## When to Use
- Applying the same operation to multiple files, URLs, or items
- Running parallel reviews, summaries, or transformations
- Processing a list of inputs with the same prompt

## Steps

1. **Identify inputs**: Gather the list of items to process (files, URLs, text snippets, etc.)
2. **Define the operation**: Clarify what to do with each input (summarize, review, transform, extract, etc.)
3. **Execute in parallel**: Use subagents or background tasks for each input
   - For each item, spawn a background task with the same prompt template
   - Substitute the item-specific value into the prompt
4. **Collect results**: Wait for all tasks to complete, gather outputs
5. **Present summary**: Show results in a table or list format with per-item status

## Example

```
User: "Review all Python files in src/ for security issues"

Steps:
1. glob src/**/*.py → [file1.py, file2.py, file3.py, ...]
2. For each file, spawn: "Review {file} for security vulnerabilities"
3. Collect findings per file
4. Present: table of file → findings
```

## Output
- Per-item results (success/failure/findings)
- Summary statistics (N processed, M issues found)
- Any items that failed or timed out
