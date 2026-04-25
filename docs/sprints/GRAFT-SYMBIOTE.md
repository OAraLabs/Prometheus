# GRAFT: GitHub Research & Code Assimilation
## Codename: SYMBIOTE

**Date:** April 21, 2026
**Purpose:** Prometheus can create skills from its own experience (SkillExtractor) and evolve them (SkillRefiner), but it can't go *outward* — it can't research GitHub for solutions to capability gaps, evaluate candidates, extract relevant code, and graft it into itself. SYMBIOTE gives Prometheus the ability to scout, harvest, and assimilate open-source code from GitHub repos, with full provenance tracking, license enforcement, and SecurityGate integration at every phase.

**Estimated Time:** 3–4 hours
**Dependencies:** Sprint 4 (SecurityGate), Sprint 11/14 (Security Audit + DangerousCodeScanner), GRAFT-BOOTSTRAP (SOUL.md, provenance conventions), GRAFT-PROFILES (profile system for symbiote profile)

**This is ADDITIVE ONLY. Do not remove or replace any existing functionality.**

(Full spec preserved verbatim — see message at start of session for the complete 17-step plan including BackupVault and MorphEngine which are Session B scope.)

## Session A scope (this sprint): Steps 0-11
- Step 0 — Survey
- Step 1 — LicenseGate
- Step 2 — GitHubSearchTool
- Step 3 — ScoutEngine
- Step 4 — HarvestEngine
- Step 5 — GraftEngine
- Step 6 — SymbioteCoordinator (Scout→Harvest→Graft phases only)
- Step 7 — Agent tools (scout/harvest/graft/status — 4 of 6)
- Step 8 — Telegram /symbiote (subcommands: <problem>, approve, graft, status, abort, history)
- Step 9 — Symbiote profile (without morph/restore tools)
- Step 10 — Config (without morph/backup sections)
- Step 11 — Tests + PROMETHEUS.md update

## Session B scope (deferred): Steps 12-17
- BackupVault, MorphEngine, swap/restore commands, blue-green deploy

See top of session for the full prose-level spec inline.
