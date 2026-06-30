---
name: research-paper-writing
title: Research Paper Writing Pipeline
description: End-to-end pipeline for writing ML/AI research papers — from experiment design through analysis, drafting, revision, and submission. Covers NeurIPS, ICML, ICLR, ACL, AAAI, COLM. Integrates automated experiment monitoring, statistical analysis, iterative writing, and citation verification.
version: 1.0.0
author: Prometheus
license: MIT
dependencies: [semanticscholar, arxiv, habanero, requests, scipy, numpy, matplotlib, SciencePlots]
platforms: [linux, macos]
metadata:
  prometheus:
    tags: [Research, Paper Writing, Experiments, ML, AI, NeurIPS, ICML, ICLR, ACL, AAAI, COLM, LaTeX, Citations, Statistical Analysis]
    category: research
    related_skills: [arxiv, ocr-and-documents]
    requires_toolsets: [bash, file_read, file_write, file_edit]

---
<!-- Adapted for Prometheus from NousResearch/hermes-agent | MIT -->

# Research Paper Writing Pipeline

End-to-end pipeline for producing publication-ready ML/AI research papers targeting **NeurIPS, ICML, ICLR, ACL, AAAI, and COLM**. This skill covers the full research lifecycle: experiment design, execution, monitoring, analysis, paper writing, review, revision, and submission.

This is **not a linear pipeline** — it is an iterative loop. Results trigger new experiments. Reviews trigger new analysis. The agent must handle these feedback loops.

```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH PAPER PIPELINE                  │
│                                                             │
│  Phase 0: Project Setup ──► Phase 1: Literature Review      │
│       │                          │                          │
│       ▼                          ▼                          │
│  Phase 2: Experiment     Phase 5: Paper Drafting ◄──┐      │
│       Design                     │                   │      │
│       │                          ▼                   │      │
│       ▼                    Phase 6: Self-Review      │      │
│  Phase 3: Execution &           & Revision ──────────┘      │
│       Monitoring                 │                          │
│       │                          ▼                          │
│       ▼                    Phase 7: Submission               │
│  Phase 4: Analysis ─────► (feeds back to Phase 2 or 5)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## When To Use This Skill

Use this skill when:
- **Starting a new research paper** from an existing codebase or idea
- **Designing and running experiments** to support paper claims
- **Writing or revising** any section of a research paper
- **Preparing for submission** to a specific conference
- **Responding to reviews** with additional experiments or revisions
- **Converting** a paper between conference formats

## Core Philosophy

1. **Be proactive.** Deliver complete drafts, not questions. Scientists are busy — produce something concrete they can react to, then iterate.
2. **Never hallucinate citations.** AI-generated citations have ~40% error rate. Always fetch programmatically. Mark unverifiable citations as `[CITATION NEEDED]`.
3. **Paper is a story, not a collection of experiments.** Every paper needs one clear contribution stated in a single sentence. If you can't do that, the paper isn't ready.
4. **Experiments serve claims.** Every experiment must explicitly state which claim it supports. Never run experiments that don't connect to the paper's narrative.
5. **Commit early, commit often.** Every completed experiment batch, every paper draft update — commit with descriptive messages. Git log is the experiment history.

### Proactivity and Collaboration

**Default: Be proactive. Draft first, ask with the draft.**

| Confidence Level | Action |
|-----------------|--------|
| **High** (clear repo, obvious contribution) | Write full draft, deliver, iterate on feedback |
| **Medium** (some ambiguity) | Write draft with flagged uncertainties, continue |
| **Low** (major unknowns) | Ask 1-2 targeted questions via `clarify`, then draft |

| Section | Draft Autonomously? | Flag With Draft |
|---------|-------------------|-----------------|
| Abstract | Yes | "Framed contribution as X — adjust if needed" |
| Introduction | Yes | "Emphasized problem Y — correct if wrong" |
| Methods | Yes | "Included details A, B, C — add missing pieces" |
| Experiments | Yes | "Highlighted results 1, 2, 3 — reorder if needed" |
| Related Work | Yes | "Cited papers X, Y, Z — add any I missed" |

**Block for input only when**: target venue unclear, multiple contradictory framings, results seem incomplete, explicit request to review first.

---

## Phase 0: Project Setup

**Goal**: Establish the workspace, understand existing work, identify the contribution.

### Step 0.1: Explore and Organize

Explore the repository for README, results directories, configs, .bib files, and draft documents. Establish workspace structure: `paper/`, `experiments/`, `code/`, `results/`, `tasks/`.

### Step 0.2: Version Control

Git discipline: every completed experiment batch gets a descriptive commit message.

### Step 0.3: Identify the Contribution

Articulate **The What** (single contribution), **The Why** (evidence), **The So What** (why readers care). If you cannot state the contribution in one sentence, the paper is not ready. Track progress via a structured TODO list.

---

## Phase 1: Literature Review

**Goal**: Find related work, identify baselines, gather citations.

### Step 1.1: Search for Related Work

Start from papers referenced in the codebase (`grep` for arxiv/doi/cite). Use `web_search` for broad discovery, `web_fetch` for specific papers. Load the `arxiv` skill for structured paper discovery via arXiv REST API and Semantic Scholar.

### Step 1.2: Verify Every Citation

**NEVER generate BibTeX from memory.** 5-step process: SEARCH (Semantic Scholar), VERIFY (2+ sources), RETRIEVE (DOI content negotiation), VALIDATE (claim actually appears in paper), ADD. If any step fails, mark as `[CITATION NEEDED]`. See [references/citation-workflow.md](references/citation-workflow.md).

### Step 1.3: Organize Related Work

Group by methodology, not paper-by-paper. Good: "One line of work uses X [refs] whereas we use Y because..." Bad: "Smith et al. did X. Jones et al. did Y."

---

## Phase 2: Experiment Design

**Goal**: Design experiments that directly support paper claims. Every experiment must answer a specific question.

### Step 2.1: Map Claims to Experiments

Create an explicit mapping:

| Claim | Experiment | Expected Evidence |
|-------|-----------|-------------------|
| "Our method outperforms baselines" | Main comparison (Table 1) | Win rate, statistical significance |
| "Effect is larger for weaker models" | Model scaling study | Monotonic improvement curve |
| "Convergence requires scope constraints" | Constrained vs unconstrained | Convergence rate comparison |

**Rule**: If an experiment doesn't map to a claim, don't run it.

### Step 2.2: Design Baselines

Strong baselines are what separates accepted papers from rejected ones. Reviewers will ask: "Did they compare against X?"

Standard baseline categories:
- **Naive baseline**: Simplest possible approach
- **Strong baseline**: Best known existing method
- **Ablation baselines**: Your method minus one component
- **Compute-matched baselines**: Same compute budget, different allocation

### Step 2.3: Define Evaluation Protocol

Before running anything, specify:
- **Metrics**: What you're measuring, direction symbols (higher/lower better)
- **Aggregation**: How results are combined across runs/tasks
- **Statistical tests**: What tests will establish significance
- **Sample sizes**: How many runs/problems/tasks

### Step 2.4: Write Experiment Scripts

Key patterns: **incremental saving** (skip already-completed work on re-runs), **artifact preservation** (save all intermediate outputs), **separation of concerns** (keep generation, evaluation, and visualization in separate scripts).

See [references/experiment-patterns.md](references/experiment-patterns.md) for complete design patterns and error recovery.

---

## Phase 3: Experiment Execution & Monitoring

**Goal**: Run experiments reliably, monitor progress, recover from failures.

### Step 3.1: Launch Experiments

Use `nohup` for long-running experiments:

```bash
nohup python run_experiment.py --config config.yaml > logs/experiment_01.log 2>&1 &
echo $!  # Record the PID
```

**Parallel execution**: Run independent experiments simultaneously, but be aware of API rate limits. 4+ concurrent experiments on the same API will slow each down.

### Step 3.2: Set Up Monitoring

For long-running experiments, set up periodic status checks: check process, read logs, check results, commit if complete. Scripts should always skip completed work so re-runs are safe.

### Step 3.3: Handle Failures

| Failure | Recovery |
|---------|----------|
| API rate limit (402/429) | Wait, then re-run (scripts skip completed work) |
| Process crash | Re-run from last checkpoint |
| Timeout | Kill and skip, note in results |
| Wrong model ID | Fix ID and re-run |

### Step 3.4: Commit Completed Results

After each experiment batch completes:

```bash
git add -A
git commit -m "Add <experiment name>: <key finding in 1 line>"
git push
```

---

## Phase 4: Result Analysis

**Goal**: Extract findings, compute statistics, identify the story.

### Step 4.1: Aggregate Results

Load all result files, compute per-task and aggregate metrics, generate summary tables. Always compute error bars (std dev or std error), 95% confidence intervals, pairwise statistical tests (McNemar's), and effect sizes (Cohen's d or h). See [references/experiment-patterns.md](references/experiment-patterns.md).

### Step 4.3: Identify the Story

After analysis, explicitly answer:
1. **What is the main finding?** State it in one sentence.
2. **What surprised you?** Unexpected results often make the best papers.
3. **What failed?** Failed experiments can be the most informative. Honest reporting of failures strengthens the paper.
4. **What follow-up experiments are needed?** Results often raise new questions.

### Step 4.4: Create Figures and Tables

**Figures**:
- Use vector graphics (PDF) for all plots: `plt.savefig('fig.pdf')`
- Colorblind-safe palettes (Okabe-Ito or Paul Tol)
- Self-contained captions — reader should understand without main text
- No title inside figure — the caption serves this function

**Tables**:
- Use `booktabs` LaTeX package
- Bold best value per metric
- Include direction symbols (higher/lower better)
- Consistent decimal precision

```latex
\usepackage{booktabs}
\begin{tabular}{lcc}
\toprule
Method & Accuracy $\uparrow$ & Latency $\downarrow$ \\
\midrule
Baseline & 85.2 & 45ms \\
\textbf{Ours} & \textbf{92.1} & 38ms \\
\bottomrule
\end{tabular}
```

### Step 4.5: Decide: More Experiments or Write?

| Situation | Action |
|-----------|--------|
| Core claims supported, results significant | Move to Phase 5 (writing) |
| Results inconclusive, need more data | Back to Phase 2 (design) |
| Unexpected finding suggests new direction | Back to Phase 2 (design) |
| Missing one ablation reviewers will ask for | Run it, then Phase 5 |
| All experiments done but some failed | Note failures, move to Phase 5 |

---

## Iterative Refinement: Strategy Selection

Choose a refinement strategy based on model tier and task type:

| Situation | Strategy |
|-----------|----------|
| Mid-tier model + constrained task | Autoreason (generation-evaluation gap is widest) |
| Mid-tier model + open task | Autoreason with scope constraints |
| Frontier model + constrained task | Autoreason (still wins 2/3 constrained tasks) |
| Frontier model + unconstrained task | Critique-and-revise or single pass |
| Code with test cases | Autoreason code variant (structured failure analysis) |
| Very weak model | Single pass (model too weak for diverse candidates) |

**Autoreason loop**: Critic finds problems -> Author B revises -> Synthesizer merges -> 3 blind judges rank via Borda count -> incumbent wins k=2 consecutive passes = done. Key: k=2 convergence, CoT judges always, temperature 0.8 authors / 0.3 judges, fresh agents for every role.

**For paper drafts**: provide ground truth data to the critic, use 3+ working judges, scope-constrain revisions to specific weaknesses.

See [references/autoreason-methodology.md](references/autoreason-methodology.md) for complete details.

---

## Phase 5: Paper Drafting

**Goal**: Write a complete, publication-ready paper.

### The Narrative Principle

**The single most critical insight**: Your paper is not a collection of experiments — it's a story with one clear contribution supported by evidence.

Every successful ML paper centers on what Neel Nanda calls "the narrative": a short, rigorous, evidence-based technical story with a takeaway readers care about.

**Three Pillars (must be crystal clear by end of introduction):**

| Pillar | Description | Test |
|--------|-------------|------|
| **The What** | 1-3 specific novel claims | Can you state them in one sentence? |
| **The Why** | Rigorous empirical evidence | Do experiments distinguish your hypothesis from alternatives? |
| **The So What** | Why readers should care | Does this connect to a recognized community problem? |

**If you cannot state your contribution in one sentence, you don't yet have a paper.**

### Time Allocation

Spend approximately **equal time** on each of:
1. The abstract
2. The introduction
3. The figures
4. Everything else combined

**Why?** Most reviewers form judgments before reaching your methods. Readers encounter your paper as: title → abstract → introduction → figures → maybe the rest.

### Writing Workflow

```
Paper Writing Checklist:
- [ ] Step 1: Define the one-sentence contribution
- [ ] Step 2: Draft Figure 1 (core idea or most compelling result)
- [ ] Step 3: Draft abstract (5-sentence formula)
- [ ] Step 4: Draft introduction (1-1.5 pages max)
- [ ] Step 5: Draft methods
- [ ] Step 6: Draft experiments & results
- [ ] Step 7: Draft related work
- [ ] Step 8: Draft conclusion & discussion
- [ ] Step 9: Draft limitations (REQUIRED by all venues)
- [ ] Step 10: Plan appendix (proofs, extra experiments, details)
- [ ] Step 11: Complete paper checklist
- [ ] Step 12: Final review
```

### Step 5.0: Title

The title is the single most-read element of the paper. It determines whether anyone clicks through to the abstract.

**Good titles**:
- State the contribution or finding: "Autoreason: When Iterative LLM Refinement Works and Why It Fails"
- Highlight a surprising result: "Scaling Data-Constrained Language Models" (implies you can)
- Name the method + what it does: "DPO: Direct Preference Optimization of Language Models"

**Bad titles**:
- Too generic: "An Approach to Improving Language Model Outputs"
- Too long: anything over ~15 words
- Jargon-only: "Asymptotic Convergence of Iterative Stochastic Policy Refinement" (who is this for?)

**Rules**:
- Include your method name if you have one (for citability)
- Include 1-2 keywords reviewers will search for
- Avoid colons unless both halves carry meaning
- Test: would a reviewer know the domain and contribution from the title alone?

### Step 5.1: Abstract (5-Sentence Formula)

From Sebastian Farquhar (DeepMind):

```
1. What you achieved: "We introduce...", "We prove...", "We demonstrate..."
2. Why this is hard and important
3. How you do it (with specialist keywords for discoverability)
4. What evidence you have
5. Your most remarkable number/result
```

**Delete** generic openings like "Large language models have achieved remarkable success..."

### Step 5.2: Figure 1

Figure 1 is the second thing most readers look at. Draft it before the introduction -- it forces you to clarify the core idea. Types: method diagram, results teaser, problem illustration, conceptual diagram. Must be understandable without reading text.

### Step 5.3: Introduction (1-1.5 pages max)

Clear problem statement, brief approach overview, 2-4 bullet contribution list. Methods should start by page 2-3.

### Step 5.4: Methods, Experiments, Related Work, Limitations

- **Methods**: enable reimplementation (pseudocode, all hyperparameters, architectural details)
- **Experiments**: explicitly state which claim each experiment supports, include error bars and significance
- **Related work**: organize methodologically, not paper-by-paper. Cite generously.
- **Limitations** (REQUIRED by all venues): pre-empt criticisms, explain why they don't undermine core claims

### Writing Style

Follow Gopen & Swan's principles: subject-verb proximity, emphasis at sentence ends, context before new info, one point per paragraph. Be specific, eliminate hedging, use consistent terminology.

### LaTeX Templates and Tooling

Always copy entire template directory, verify it compiles before changes, replace content section by section. Key packages: microtype, booktabs, siunitx, cleveref. Use `latexdiff` for rebuttals, SciencePlots for matplotlib.

| Conference | Page Limit |
|------------|------------|
| NeurIPS | 9 pages |
| ICML | 8 pages |
| ICLR | 9 pages |
| ACL | 8 pages (long) |
| AAAI | 7 pages |
| COLM | 9 pages |

---

## Phase 6: Self-Review & Revision

Simulate reviews from multiple perspectives using the target venue's reviewer guidelines. Categorize feedback as critical/high/medium/low. For each critical/high issue: identify affected sections, draft fix, verify it doesn't break other claims.

**Rebuttals**: point-by-point format, address every concern, lead with strongest responses, include new results if available, use `latexdiff` for marked-up PDFs.

---

## Phase 7: Submission Preparation

### Checklists

Complete venue-specific checklists (desk rejection risk if incomplete). See [references/checklists.md](references/checklists.md).

### Anonymization

No author names/affiliations, no acknowledgments, third-person self-citations, anonymous GitHub for code, clean PDF metadata.

### Final Compilation

```bash
rm -f *.aux *.bbl *.blg *.log *.out *.pdf
latexmk -pdf main.tex
```

### Camera-Ready (Post-Acceptance)

De-anonymize, add acknowledgments, add real code URLs, address meta-reviewer revisions.

---

## Prometheus Integration

### Tools Reference

| Tool | Usage |
|------|-------|
| `bash` | LaTeX compilation, git ops, launching experiments, process checks |
| `file_read` / `file_write` / `file_edit` | Paper editing, experiment scripts, result files |
| `grep` / `glob` | Search results, find templates, locate experiment outputs |
| `web_search` / `web_fetch` | Literature discovery, citation verification |

### State Management

Use the wiki for persisting key decisions across sessions. On session startup: check wiki, review git log, check running experiments, check for new results.

---

## Reference Documents

| Document | Contents |
|----------|----------|
| [references/writing-guide.md](references/writing-guide.md) | Writing principles, word choice, figure design |
| [references/citation-workflow.md](references/citation-workflow.md) | Citation APIs, BibTeX management |
| [references/checklists.md](references/checklists.md) | Venue-specific submission checklists |
| [references/reviewer-guidelines.md](references/reviewer-guidelines.md) | Evaluation criteria, rebuttal strategies |
| [references/experiment-patterns.md](references/experiment-patterns.md) | Experiment design patterns, monitoring |
| [references/autoreason-methodology.md](references/autoreason-methodology.md) | Autoreason loop details, model selection |
