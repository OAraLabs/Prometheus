---
name: audit-context-building
description: "Enables ultra-granular, line-by-line code analysis to build deep architectural context before vulnerability or bug finding. Use when deep comprehension is needed before security auditing, architecture review, or threat modeling -- runs BEFORE the vulnerability-hunting phase."
version: 1.0.0
author: Trail of Bits
license: CC-BY-SA-4.0
---
<!-- Provenance: trailofbits/skills | plugins/audit-context-building/skills/audit-context-building/SKILL.md | CC-BY-SA-4.0 -->

# Deep Context Builder (Ultra-Granular Pure Context Mode)

## Purpose

This skill governs how to think during the context-building phase of an audit.

When active:
- Perform **line-by-line / block-by-block** code analysis by default
- Apply **First Principles**, **5 Whys**, and **5 Hows** at micro scale
- Continuously link insights across functions, modules, and the entire system
- Maintain a stable, explicit mental model that evolves with new evidence
- Identify invariants, assumptions, flows, and reasoning hazards

This skill defines a structured analysis format and runs **before** the vulnerability-hunting phase.

## When to Use

- Deep comprehension is needed before bug or vulnerability discovery
- You want bottom-up understanding instead of high-level guessing
- Reducing hallucinations, contradictions, and context loss is critical
- Preparing for security auditing, architecture review, or threat modeling

## When NOT to Use

- Vulnerability findings (this is pure context, not bug hunting)
- Fix recommendations
- Exploit reasoning
- Severity/impact rating

## Rationalizations to Reject

| Rationalization | Why It's Wrong | Required Action |
|-----------------|----------------|-----------------|
| "I get the gist" | Gist-level understanding misses edge cases | Line-by-line analysis required |
| "This function is simple" | Simple functions compose into complex bugs | Apply 5 Whys anyway |
| "I'll remember this invariant" | You won't. Context degrades. | Write it down explicitly |
| "External call is probably fine" | External = adversarial until proven otherwise | Jump into code or model as hostile |
| "I can skip this helper" | Helpers contain assumptions that propagate | Trace the full call chain |
| "This is taking too long" | Rushed context = hallucinated vulnerabilities later | Slow is fast |

## Phase 1: Initial Orientation (Bottom-Up Scan)

Before deep analysis, perform a minimal mapping:

1. Identify major modules/files/contracts
2. Note obvious public/external entrypoints
3. Identify likely actors (users, owners, relayers, oracles, other contracts)
4. Identify important storage variables, dicts, state structs, or cells
5. Build a preliminary structure without assuming behavior

This establishes anchors for detailed analysis.

## Phase 2: Ultra-Granular Function Analysis (Default Mode)

Every non-trivial function receives full micro analysis.

### Per-Function Microstructure Checklist

For each function:

**1. Purpose**
- Why the function exists and its role in the system
- Minimum 2-3 sentences

**2. Inputs and Assumptions**
- All parameters (explicit and implicit: state, sender, env)
- Preconditions and constraints
- Trust assumptions
- Each input: type, source, trust level
- Minimum 5 assumptions documented

**3. Outputs and Effects**
- Return values (or "void")
- State/storage writes
- Events/messages
- External interactions
- Postconditions
- Minimum 3 effects documented

**4. Block-by-Block / Line-by-Line Analysis**
For each logical block:
- **What**: What it does (1 sentence)
- **Why here**: Why this ordering/placement (1 sentence)
- **Assumptions**: What must be true (1+ items)
- **Depends on**: What prior state/logic this relies on
- **First Principles / 5 Whys / 5 Hows**: Apply at least ONE per block

Standards:
- Analyze ALL conditional branches, ALL external calls, ALL state modifications
- Complex blocks (>5 lines): Apply First Principles AND 5 Whys or 5 Hows
- Simple blocks (<5 lines): Minimum What + Why here + 1 Assumption

**5. Cross-Function Dependencies**
- Internal calls made (list all)
- External calls made (list all with risk analysis)
- Functions that call this function
- Shared state with other functions
- Invariant couplings
- Minimum 3 dependency relationships documented

### Quality Thresholds

A complete micro-analysis MUST identify:
- Minimum 3 invariants per function
- Minimum 5 assumptions across all sections
- Minimum 3 risk considerations (especially for external interactions)
- At least 1 application of First Principles
- At least 3 applications of 5 Whys or 5 Hows (combined)

### Cross-Function and External Flow Analysis

When encountering calls, continue the same micro-first analysis across boundaries.

**Internal Calls:**
- Jump into the callee immediately
- Perform block-by-block analysis
- Track flow: caller -> callee -> return -> caller
- Note if callee behaves differently in this specific call context

**External Calls -- Code Available:**
- Treat as internal call: jump in, continue analysis
- Propagate invariants and assumptions seamlessly

**External Calls -- No Code (True Black Box):**
- Analyze as adversarial
- Describe payload/parameters sent
- Identify assumptions about the target
- Consider all outcomes: revert, incorrect returns, state changes, reentrancy

**Continuity Rule:** Treat the entire call chain as one continuous execution flow. Never reset context. All invariants, assumptions, and data dependencies must propagate.

## Phase 3: Global System Understanding

After sufficient micro-analysis:

**1. State and Invariant Reconstruction**
- Map reads/writes of each state variable
- Derive multi-function and multi-module invariants

**2. Workflow Reconstruction**
- Identify end-to-end flows (deposit, withdraw, lifecycle, upgrades)
- Track how state transforms across flows
- Record assumptions that persist across steps

**3. Trust Boundary Mapping**
- Actor -> entrypoint -> behavior
- Identify untrusted input paths
- Privilege changes and implicit role expectations

**4. Complexity and Fragility Clustering**
- Functions with many assumptions
- High branching logic
- Multi-step dependencies
- Coupled state changes across modules

These clusters guide the vulnerability-hunting phase.

## Stability and Consistency Rules (Anti-Hallucination)

- **Never reshape evidence to fit earlier assumptions.** When contradicted: update the model and state the correction explicitly.
- **Periodically anchor key facts.** Summarize core invariants, state relationships, actor roles, workflows.
- **Avoid vague guesses.** Use: "Unclear; need to inspect X." instead of "It probably..."
- **Cross-reference constantly.** Connect new insights to previous state, flows, and invariants to maintain global coherence.

## Completeness Checklist

Before concluding micro-analysis of a function, verify:

### Structural Completeness
- [ ] Purpose section: 2+ sentences
- [ ] Inputs and Assumptions: All parameters + implicit inputs documented
- [ ] Outputs and Effects: All returns, state writes, external calls, events
- [ ] Block-by-Block Analysis: Every logical block analyzed (no gaps)
- [ ] Cross-Function Dependencies: All calls and shared state documented

### Content Depth
- [ ] Identified at least 3 invariants
- [ ] Documented at least 5 assumptions
- [ ] Applied First Principles at least once
- [ ] Applied 5 Whys or 5 Hows at least 3 times total
- [ ] Risk analysis for all external interactions

### Continuity and Integration
- [ ] Cross-references with related functions
- [ ] Propagated assumptions from callers
- [ ] Identified invariant couplings
- [ ] Tracked data flow across function boundaries

### Anti-Hallucination Verification
- [ ] All claims reference specific line numbers
- [ ] No vague statements ("probably", "might", "seems to")
- [ ] Contradictions resolved with explicit updates
- [ ] Evidence-based: every invariant/assumption tied to actual code

Analysis is complete when all checklist items are satisfied and no unresolved items remain.

## Non-Goals

While active, do NOT:
- Identify vulnerabilities
- Propose fixes
- Generate proofs-of-concept
- Model exploits
- Assign severity or impact

This is **pure context building** only.

## Output Format

Use markdown headers for major sections. Use bullet points for lists. Use code blocks for snippets. Reference line numbers (`L45`, `lines 98-102`). Separate blocks with horizontal rules for readability.
