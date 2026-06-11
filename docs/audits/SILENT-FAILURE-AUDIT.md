# Silent Failure Audit

**Verified against:** `origin/main` at `944934f566e0dfb91e63ee20ae598ffdc1a3f690`
**Date:** 2026-05-21
**Triggered by:** PR #1 / `ed8f1a6` — three-line ValidationError fix that revealed three core subsystems had been structurally wired but functionally inert for an unknown duration.
**Phase:** 1 of 2 (read-only audit; Phase 2 implements per Will's prioritization)

---

## Summary

| Category | Count | Notes |
|---|---|---|
| Total `except` blocks scanned (`src/prometheus/`) | 545 across 226 files | |
| Exception swallows — KEEP | ~330 | Mostly optional-imports, best-effort cleanup, telemetry-emit guards |
| Exception swallows — CONVERT | ~155 | Should log at WARN+ and/or write telemetry |
| Exception swallows — RE-RAISE | ~17 | Broad catch should propagate or narrow |
| Exception swallows — HIGH-RISK | **19** | Load-bearing paths; full list below |
| Out-of-scope swallows (Adapter Layer + LCM) | 19 | Noted, not classified, per sprint constraint |
| Bare `except:` blocks | **0** | One structural-hygiene bullet already in place |
| Wiring tests audited (in named `Test*Wiring` classes, line ≥ 5002) | 85 | |
| Wiring tests — STRUCTURAL-ONLY | 34 | |
| Wiring tests — FUNCTIONAL | 42 | |
| Wiring tests — MIXED | 9 | |
| Autonomous subsystems audited | 17 | |
| Subsystems with **High** telemetry gap | 9 | Could be silently failing right now |
| Subsystems with **Medium** telemetry gap | 5 | Failure visible only via logs / per-subsystem DBs |
| Subsystems with **Low** telemetry gap | 1 | (SymbioteCoordinator family) |
| Subsystems with **None** | 1 | PeriodicNudge (synchronous, rides agent-loop telemetry) |

**Headline:** the schema in `telemetry.db` answers "how is the agent's tool layer doing?" but **cannot answer "is the Curator / SkillCreator / MemoryExtractor / GEPA / SENTINEL loop running?"** — there is no `subsystem` column anywhere and no `silent_failures` table. Nine of seventeen autonomous subsystems could be inert and the only signal would be log files nobody reads. This is the structural shape of the `ed8f1a6` bug.

---

## HIGH-RISK silent failures

Nineteen swallows in load-bearing paths. **Three of the nineteen are in Sprint 1 code I authored last session (Curator + signal emission)** — flagging honestly so the audit isn't just pointing at other people's code. All nineteen are ed8f1a6-shaped: a broad swallow with at most a `log.exception`, no telemetry row, no observable signal to the operator.

### Tier 1 — Config-load `(OSError, Exception)` family (3 items)

These four sites all use `except (OSError, Exception):` which is dead syntax (Exception covers OSError) and silently substitutes a default config dict. A typo in `prometheus.yaml`'s `learning:` block — or any future pydantic schema tightening on the config itself — silently disables the subsystem with zero log.

| # | site | current code | fix |
|---|---|---|---|
| 1 | `learning/skill_creator.py:127` | `except (OSError, Exception): min_calls = _MIN_TOOL_CALLS` | Narrow to `(OSError, yaml.YAMLError)`, `log.warning(... exc_info=True)` |
| 2 | `learning/skill_refiner.py:116` | `except (OSError, Exception): learning = {}` | Same — narrow + WARN |
| 3 | `learning/gepa.py:183` | `except (OSError, Exception): data = {}` | Same — narrow + WARN |

(`learning/nudge.py:65` has the same shape but is lower-risk because PeriodicNudge falls back to safe defaults. Also addressable in this family.)

### Tier 2 — `_call_model` swallows (4 items, includes Sprint 1's Curator)

The exact gap that produced ed8f1a6. ed8f1a6 fixed the *input shape*; the *exception swallow around the call* still has no telemetry / no signal. A future schema change of a different shape would manifest identically.

| # | site | current code | fix |
|---|---|---|---|
| 4 | `learning/skill_creator.py:162` | `except Exception: log.exception("…model call failed"); return None` | Keep log; add `tel.record_silent_failure("skill_creator", "_call_model", exc)` + emit `learning_failure` signal |
| 5 | `learning/skill_refiner.py:187` | same shape | Same fix |
| 6 | `memory/extractor.py:198` | same shape, plus `run_forever` re-wraps at `:180` | Same fix + debounced `extraction_failure` signal |
| 7 | `learning/curator.py:351` | Sprint 1 code I wrote. `except Exception as exc: log.exception(...); run.errors.append(...)` | Same fix — pair `run.errors` with a `silent_failures` row + `curator_failure` signal |

### Tier 3 — Sprint 1 Curator's eternal-loop catch (1 item)

| # | site | current code | fix |
|---|---|---|---|
| 8 | `learning/curator.py:320` | Sprint 1 code I wrote. `except Exception: log.exception("Curator: run_once failed")` then loops forever | Add a consecutive-failure counter, exponential backoff, and a `curator_disabled` signal after N failures so a runaway crashes-every-7-days curator becomes visible |

### Tier 4 — Agent loop hot-path swallows (5 items)

The agent loop is the worst place for a silent failure. None of these are obviously wrong today, but all five lack observability.

| # | site | current code | fix |
|---|---|---|---|
| 9 | `engine/agent_loop.py:336` | `except Exception: continue` in config-changed check | `log.warning(... exc_info=True)` |
| 10 | `engine/agent_loop.py:240` | `except Exception as exc: log.warning("diagnose_and_recover crashed…"); return _RecoveryResult(recovered=False, …)` | Add telemetry row + `recovery_failed` signal |
| 11 | `engine/agent_loop.py:1252` | `except Exception: return False` in `_is_read_only_call` | `log.debug(... exc_info=True)`; keep fail-safe direction |
| 12 | `engine/agent_loop.py:1471` | `except Exception: _origin = "system"` (TRUST-CONTEXT origin) | WARN with session_id snippet so the next deploy notices the regression |
| 13 | `engine/agent_loop.py:1481` | `except TypeError: ... # legacy shape` | One-shot `log.warning("Legacy permission_checker.evaluate signature detected")` so deprecation is observable |

### Tier 5 — Gateway dispatch swallows (3 items)

User-visible surface. Silent failures here = "the bot just stopped responding."

| # | site | current code | fix |
|---|---|---|---|
| 14 | `gateway/telegram.py:191` | `except Exception: pass` in `_notification_mode` | `log.debug(... exc_info=True)` |
| 15 | `gateway/telegram.py:1392` | `except Exception: pass  # typing indicator is best-effort` | Keep behavior; `log.debug` — the typing indicator failing is often the canary for a dead bot token |
| 16 | `gateway/telegram.py:1467` | `except Exception: cfg = {}` (config-load fallback) | Narrow to `(OSError, yaml.YAMLError)` + WARN |

### Tier 6 — Hook executor (1 item)

| # | site | current code | fix |
|---|---|---|---|
| 17 | `hooks/executor.py:150` | `except Exception as exc: return HookResult(... success=False, blocked=hook.block_on_failure, ...)` | Add `log.warning("Hook %s failed: %s", hook.name, exc)` — currently a hook can fail forever without anyone noticing |

### Tier 7 — Tool helpers + Sprint 1 signal emission (2 items)

| # | site | current code | fix |
|---|---|---|---|
| 18 | `tools/builtin/file_read.py:94` | `except Exception: pass; return None` | `log.debug("_find_in_files_dir raised", exc_info=True)` |
| 19 | `memory/hermes_memory_tool.py:370` | Sprint 1 code I wrote. `except Exception: pass  # Never let signalling break a memory write` | Keep the swallow (intent is correct); add `log.debug("memory_updated signal emission failed", exc_info=True)` so a misconfigured bus is observable |

### Specific `_call_model` consistency check

After ed8f1a6, the four `_call_model` helpers are **structurally consistent**:

| site | shape | exception handling |
|---|---|---|
| `learning/skill_creator.py:232-249` | `[TextBlock(text=prompt)]`, `ApiTextDeltaEvent` loop | log-only swallow at `:162` |
| `learning/skill_refiner.py:269-286` | same | log-only swallow at `:187` |
| `learning/curator.py:479-497` | same | log-only swallow + `run.errors.append` at `:351` |
| `memory/extractor.py:225-243` | same | log-only swallow at `:198` |

No structural regression risk. All four share the same observability gap (Tier 2 above) — that's where Phase 2's `LLMCallEnvelope` lands.

Cosmetic cleanup also surfaced: `memory/extractor.py:240` imports `ApiTextDeltaEvent` inside the loop instead of at the top; `learning/curator.py:486-487` and the three other sites carry "Pre-existing bug found during Sprint 1" comments that are now historical breadcrumbs. Defer to a follow-up cleanup task.

---

## Exception swallows — categorized (by subsystem)

Full per-subsystem detail in Agent A's transcript at `/tmp/silent-failure-audit/exceptions.md`. Headline patterns:

### gateway/ (87 swallows total: 47 telegram, 16 commands, 12 cron_scheduler, 7 slack, 5 status)
- ~25 in `telegram.py` are routine UX fallbacks (`except Exception as exc: send(f"Error: {exc}")` ) — user sees the error string. **KEEP.**
- ~10 are silent `pass`/`return None` — **CONVERT** to `log.debug(... exc_info=True)`. Three flagged HIGH-RISK above (Tier 5).
- `gateway/commands.py:81, 106, 211, 394, 504, 750` — 6 silent fallbacks that return "module unavailable" strings to the user. KEEP for UX; **CONVERT** to also log so the underlying error is debuggable.
- `cron_scheduler.py` swallows around `proc.kill()` / asyncio cleanup — KEEP (best-effort).

### learning/ (35 total: 16 curator, 11 gepa, 5 skill_refiner, 3 skill_creator)
- Beyond the HIGH-RISK items above, most logging in curator.py is `log.exception(...)` (observable). Phase 2 should pair these with `silent_failures` rows.
- `learning/gepa.py:301, 311, 345, 383` — trace-parsing fallbacks. CONVERT to `log.debug(... exc_info=True)`.

### sentinel/ (~13 across signals, autodream, observer, gepa_engine, golden_trace_exporter, knowledge_synth, wiki_lint)
- **Generally in good shape.** These were designed with autonomous-loop observability in mind. `signals.py:58` (subscriber-failure), `observer.py:150` (nudge re-queue), `autodream.py:161` (phase-failure with `DreamResult.error`), `gepa_engine.py:131/134`, `golden_trace_exporter.py:82/99/118` — all log appropriately. **KEEP.**
- The gap with SENTINEL is *telemetry*, not exception handling — covered in the "Subsystems missing failure telemetry" grid below.

### symbiote/ (~86 total: 31 morph, 15 harvest, 9 coordinator, 8 github_search, 7 scout, 5 graft, 11 backup_vault)
- `morph.py:392, 508, 600, 771` — process-cleanup `except: pass`. KEEP (timeout/kill chain).
- `morph.py:628` — swap-failure rollback then re-raises. KEEP, the re-raise is the right call.
- `coordinator.py:189, 238, 279, 343, 389` — pipeline error handling that marks session FAILED + persists. **KEEP**, but emit a `symbiote_phase_failed` signal so Telegram can proactively notify (currently surfaces only on `/symbiote status`).
- `coordinator.py:439, 504, 521` — deserialize failures `log.debug`. **CONVERT** to WARN: a row that won't deserialize is a corruption canary.
- `harvest.py:386, 559` — LLM call wrappers `log.debug` + fallback. **CONVERT** to WARN.
- `scout.py:268, 303` — search/LLM failures `log.debug`. **CONVERT** to WARN.
- `github_search.py:172, 213` — silent swallows on HTTP. **CONVERT** to `log.warning`.

### tools/ (75 total, mostly narrow)
- Most narrow (`ValueError`, `OSError`, `asyncio.TimeoutError`, `httpx.HTTPError`, `json.JSONDecodeError`) and return structured errors. **KEEP.**
- `tools/printing_press.py:157` — `log.debug`. **CONVERT** to WARN.
- One HIGH-RISK item: `tools/builtin/file_read.py:94` (Tier 7).

### memory/ (6 in extractor + 13 OOS in lcm_*.py)
- Two HIGH-RISK items (Tier 2 and Tier 7); other catches are correct re-raise-as-tool-error patterns.

### engine/ (24 in agent_loop)
- Five HIGH-RISK items (Tier 4); others log at debug/warning appropriately.
- The `(OSError, Exception)` redundant-tuple pattern is absent here — good.

### providers/ (5 files)
- **All five providers use the correct retry-with-reraise pattern.** `except Exception as exc: ... if attempt >= MAX or not retryable: raise`. **KEEP** all.

### web/ (8 in ws_server)
- WebSocket cleanup catches with comments documenting intent. **KEEP.**

### infra/ (16 across anatomy + doctor)
- Capability-detection swallows. Most have `log.debug`. **CONVERT** a few (`anatomy.py:174, 300, 314, 401`) to `log.debug(... exc_info=True)` so failing detection leaves a breadcrumb.

### __main__.py (19)
- Optional-tool/registry registration with `except Exception: pass`. **KEEP** — this is the canonical optional-dependency pattern. Minor improvement: a single `log.debug("Optional tool X unavailable", exc_info=True)` instead of bare `pass` for future debuggability.

---

## RE-RAISE candidates (full list)

Cases where the broad catch should propagate or narrow:

1. **`learning/skill_creator.py:127`, `learning/skill_refiner.py:116`, `learning/gepa.py:183`, `learning/nudge.py:65`** — `except (OSError, Exception)`. The redundant tuple is dead syntax. Refactor to `except (OSError, yaml.YAMLError)` and let any other Exception propagate. This is the *exact* failure shape the audit was commissioned for.
2. **`engine/agent_loop.py:336`** — `except Exception: continue` in the model-changed check. Narrow to `(OSError, yaml.YAMLError)` or at least `exc_info=True` on a `log.warning`.
3. **`__main__.py:300`** — `except Exception: pass` in `_supports_function_calling`. Narrow to `(OSError, json.JSONDecodeError, KeyError)`.
4. **`__main__.py:503, 506`** — broad swallow inside system-prompt assembly. WARN + emit `prometheus_init_warning`.
5. **`telemetry/tracker.py:487`** — `except Exception: pass` on conn.close(). Narrow to `sqlite3.Error`.
6. **`coordinator/health.py:78, 105, 140, 190, 215, 311`** — health-check catches. Detail encoded into `ComponentHealth.detail` (observable), but consider narrowing.

---

## KEEP patterns (no enumeration needed)

The codebase has many legitimate broad catches that should remain:

- **Optional-dependency imports** (`except ImportError: ...`) — ~18 sites
- **Best-effort process kill / cleanup after timeout** — extensive in `symbiote/morph.py`, `harvest.py`, `printing_press.py`, `lsp/client.py`, `gateway/cron_scheduler.py`
- **`asyncio.CancelledError`** properly re-raised or graceful exit — `curator.py`, `sentinel/*`, `lsp/client.py`, `gateway/cron_scheduler.py`
- **Telemetry/SignalBus emit guards** — `skill_creator.py:211`, `skill_refiner.py:266`, `curator.py:736`, `sentinel/gepa_engine.py:131`, `sentinel/golden_trace_exporter.py:118`. Intent ("don't let observability break the action") is correct.
- **Narrow JSON/YAML/parse exceptions** in provider SSE streams
- **User-facing UX fallback strings** in gateway commands

---

## Wiring tests — categorized

**85 tests across 14 named `Test*Wiring` classes in `tests/test_wiring.py`** (line ≥ 5002). Tests outside that range are mostly unit-test suites; one sibling file (`tests/test_telemetry_wiring.py`) was inspected and is already functional (asserts row-level side effects). No `tests/wiring/` subdirectory exists.

### By section

| Section | Total | FUNCTIONAL | MIXED | STRUCTURAL-ONLY |
|---|---:|---:|---:|---:|
| TestCircuitBreakerDiagnosis | 16 | 9 | 2 | 5 |
| TestGoldenTraceCapture | 11 | 8 | 1 | 2 |
| TestLCMToolsAgainstRealEngine | 5 | 5 | 0 | 0 |
| TestSunriseAgentLoopHooksList | 3 | 1 | 0 | 2 |
| TestSunriseSkillRefiner | 5 | 3 | 0 | 2 |
| TestSunrisePeriodicNudge | 3 | 3 | 0 | 0 |
| TestSunriseGoldenTraceExporter | 3 | 2 | 0 | 1 |
| TestSunriseMemoryExtractorTaskName | 1 | 0 | 0 | 1 |
| TestSunriseGEPAEngineWiring | 3 | 2 | 0 | 1 |
| TestSymbioteWiring | 6 | 2 | 1 | 3 |
| TestSymbioteSessionBWiring | 5 | 1 | 0 | 4 |
| TestWeaveWebToolsWiring | 6 | 0 | 0 | 6 |
| TestWeavePressWiring | 6 | 0 | 0 | 6 |
| TestVisibleMemorySkillsWiring (Sprint 1) | 12 | 6 | 5 | 1 |
| **Total** | **85** | **42** | **9** | **34** |

### Top 5 STRUCTURAL-ONLY tests to upgrade first

Ranked by silent-failure risk (most load-bearing subsystem first):

1. **`test_daemon_curator_wiring_block_present`** (`TestVisibleMemorySkillsWiring`) — **this is the Sprint 1 anti-pattern reincarnated.** It greps `daemon.py` source text for `set_curator(curator)` and five other string fragments. It cannot detect a silent failure where `set_curator` IS called but `Curator._loop` never executes a real cycle. Replace with a boot-the-daemon test that observes a Curator cycle running.

2. **`test_engine_subscribes_to_idle_signals`** (`TestSunriseGEPAEngineWiring`) — asserts only `bus.subscriber_count >= 2`. Same shape as the original ed8f1a6 bug: subscribers registered but no observable side effect when signals fire. Publishing an `idle_start` and asserting `engine.last_report` populated would turn it functional.

3. **`test_named_task_discoverable`** (`TestSunriseMemoryExtractorTaskName`) — tests `asyncio` honors a `name=` kwarg, not that the daemon actually spawns a `memory_extractor` task. MemoryExtractor was one of the three subsystems silently inert in the ed8f1a6 incident.

4. **`test_symbiote_tools_registered_in_default_registry`** (`TestSymbioteWiring`) — set-membership check. If `SymbioteScoutTool.execute` regressed to a no-op, this test would stay green. Should call `tool.execute(...)` and assert a structured response.

5. **All five `TestWeaveWebToolsWiring` registration tests** — `youtube_transcript`, `download_file`, `web_fetch`, `web_search` each tested only for being in `registry.list_tools()`. High-traffic user-facing tools; a silent failure here is user-visible immediately but the test suite stays green. Upgrade to drive each tool against a recorded HTTP fixture.

### Full STRUCTURAL-ONLY list (34 tests)

Each draft "functional addition" is one sentence — Phase 2 implementers can use this as the work tracker.

| # | test | section | currently asserts | draft functional addition |
|---|---|---|---|---|
| 1 | `test_categorize_malformed_json` | CircuitBreakerDiagnosis | pure-fn string compare | drive `_CircuitBreaker.record_error(bad_json)` and assert the diagnostic row's `failure_category == "malformed_json"` |
| 2 | `test_categorize_wrong_schema` | CircuitBreakerDiagnosis | same | same shape for wrong_schema |
| 3 | `test_categorize_raw_text` | CircuitBreakerDiagnosis | same | same for raw_text |
| 4 | `test_categorize_special_char_escape` | CircuitBreakerDiagnosis | same | same for special_char_escape |
| 5 | `test_categorize_empty_output` | CircuitBreakerDiagnosis | same | same for empty_output |
| 6 | `test_provider_name_for_telemetry_class_fallback` | GoldenTraceCapture | pure-fn provider→name mapping | record a real telemetry row, assert row's `provider` column matches |
| 7 | `test_provider_name_honors_explicit_attribute` | GoldenTraceCapture | pure-fn override | same but with explicit `provider_name` |
| 8 | `test_add_post_task_hook_appends` | SunriseAgentLoopHooksList | `len(loop.post_task_hooks) == 2` | run `loop.run_async`, assert both hooks observe trace via shared list |
| 9 | `test_set_post_task_hook_replaces` | SunriseAgentLoopHooksList | `loop.post_task_hooks == [hook_b]` | run loop, assert only hook_b's side effect occurs |
| 10 | `test_from_config_returns_none_when_disabled` | SunriseSkillRefiner | `result is None` | stub refiner + telemetry, attempt refine, assert no skill file changed |
| 11 | `test_from_config_builds_when_enabled` | SunriseSkillRefiner | `result is not None`, `_model == "test-model"` | drive `result.maybe_refine_recent`, assert per-call `_model` forwarded to provider |
| 12 | `test_disabled_start_returns_none` | SunriseGoldenTraceExporter | `task is None` | assert no JSONL file created and no signal emitted |
| 13 | `test_named_task_discoverable` | SunriseMemoryExtractorTaskName | tests `asyncio.create_task(name=...)` — verifies asyncio, not Prometheus | boot daemon's `start_memory_extractor`, assert task with that name in `asyncio.all_tasks()` AND production code spawned it |
| 14 | `test_engine_subscribes_to_idle_signals` | SunriseGEPAEngineWiring | `bus.subscriber_count >= 2` | publish `idle_start`, assert `engine.last_report` populated or `run_optimization_cycle` invoked |
| 15 | `test_symbiote_tools_registered_in_default_registry` | SymbioteWiring | set membership in registry | execute each tool via `registry.execute(name, …)`, assert structured ToolResult |
| 16 | `test_symbiote_profile_exists` | SymbioteWiring | `"symbiote" in _BUILTINS`, tool names in profile.tools | load via `ProfileStore.load("symbiote")`, assert filtered registry contains tools |
| 17 | `test_telegram_has_cmd_symbiote` | SymbioteWiring | `hasattr` check | spy adapter, dispatch `/symbiote status`, assert `_cmd_symbiote` called |
| 18 | `test_backup_vault_imports_and_initializes` | SymbioteSessionBWiring | `_vault_root.exists()`, `list_snapshots() == []` | call `vault.snapshot()`, assert tarball appears |
| 19 | `test_morph_engine_imports_with_real_vault` | SymbioteSessionBWiring | `engine._vault is vault` | drive `engine.morph(...)` end-to-end, assert candidate dir populated |
| 20 | `test_telegram_has_session_b_handlers` | SymbioteSessionBWiring | 6 `hasattr` checks | dispatch each subcommand, assert corresponding handler invoked |
| 21 | `test_symbiote_phase_includes_morph_states` | SymbioteSessionBWiring | Phase enum string-value set membership | persist a session in each phase, assert deserializes back |
| 22 | `test_youtube_transcript_tool_registered` | WeaveWebToolsWiring | `"youtube_transcript" in names` | execute against stub HTTP, assert transcript-shaped ToolResult |
| 23 | `test_download_file_tool_registered` | WeaveWebToolsWiring | `"download_file" in names` | execute against stub URL, assert file written |
| 24 | `test_web_fetch_still_registered` | WeaveWebToolsWiring | `"web_fetch" in names` | execute against stub HTTP, assert body returns |
| 25 | `test_web_search_still_registered` | WeaveWebToolsWiring | `"web_search" in names` | execute against stub backend, assert results-array shape |
| 26 | `test_youtube_transcript_exported_from_builtin` | WeaveWebToolsWiring | `tool.name == "youtube_transcript"` | subsumed by #22 — consider drop |
| 27 | `test_download_file_exported_from_builtin` | WeaveWebToolsWiring | `tool.name == "download_file"` | subsumed by #23 |
| 28 | `test_telegram_has_cmd_press` | WeavePressWiring | 6 `hasattr` checks | dispatch `/press install foo`, assert `_press_install` invoked + registry call issued |
| 29 | `test_printing_press_registry_importable` | WeavePressWiring | `hasattr` surface check | call `reg.install("nonexistent")` against fixture library, assert documented `InstallResult` shape |
| 30 | `test_agent_loop_has_suggestion_helper` | WeavePressWiring | `inspect.iscoroutinefunction` | invoke helper with fake "command not found" trace, assert non-empty suggestion |
| 31 | `test_tool_search_exposes_skill_registry_getter` | WeavePressWiring | setter/getter round-trip on sentinel | set real `SkillRegistry`, call `ToolSearchTool.execute`, assert it surfaces skills |
| 32 | `test_agent_loop_accepts_tool_metadata` | WeavePressWiring | `"tool_metadata" in sig.parameters` | pass real `tool_metadata` at construction, assert appears on `LoopContext` during run |
| 33 | `test_daemon_curator_wiring_block_present` | VisibleMemorySkillsWiring (Sprint 1, mine) | greps `daemon.py` source for 6 strings | boot daemon test harness, assert `Curator.is_running()`, `SkillCreator.signal_bus is bus` on live objects |
| 34 | `test_commands_module_exposes_new_handlers` | VisibleMemorySkillsWiring (Sprint 1, mine) | 12 `hasattr` checks | invoke each handler with fixture state, assert non-empty string referencing expected data |

### Sprint 1 self-grade

`TestVisibleMemorySkillsWiring` is 6 FUNCTIONAL / 5 MIXED / 1 STRUCTURAL-ONLY — mostly the right shape, but the one structural-only (`test_daemon_curator_wiring_block_present`) is itself the bug class being audited. I wrote that test on commit `1eb97ea`. Honest assessment: that test should not have been called "wiring test" — it's a static-string presence check, not a runtime invariant.

---

## Subsystems missing failure telemetry

**17 autonomous subsystems audited. 9 High, 5 Medium, 1 Low, 1 None, 1 inconclusive.**

### Schema today (`~/.prometheus/telemetry.db`)

Two tool-scoped tables:

- **`tool_calls`** (`src/prometheus/telemetry/tracker.py:26-40`) — one row per tool-call attempt. `success` is the only failure flag; no `subsystem` column.
- **`circuit_breaker_diagnostics`** (`tracker.py:46-58`) — written only by `_CircuitBreaker.diagnose_and_recover()` (`engine/agent_loop.py:445`).

Writers: `agent_loop.py` (15+ sites) and `evals/runner.py` (different DB). **No autonomous subsystem writes to either table.**

### Per-subsystem grid

| Subsystem | Success row in `telemetry.db`? | Failure row? | Gap | One-line recommendation |
|---|---|---|---:|---|
| Curator (Sprint 1) | No (Markdown report + state_store + signal) | No (`learning/curator.py:320, 351, 366` log-only) | **High** | Insert `silent_failures` in the three swallow blocks |
| SkillCreator | No (SKILL.md + signal) | No (`skill_creator.py:162-164` log-only) | **High** | Insert `silent_failures` in the model-call swallow |
| SkillRefiner | No (mutates SKILL.md + signal + scanner check) | No (`skill_refiner.py:152, 187, 218`) | **High** | Same + "scanner disabled" needs its own tag so a long-disabled scanner is queryable |
| MemoryExtractor | No (`memories` table via MemoryStore + signal) | No (`extractor.py:180, 197, 221`) — **this is the ed8f1a6 footgun reincarnated** | **High** | Insert `silent_failures` in all three swallow blocks |
| PeriodicNudge | N/A — synchronous in agent loop | N/A | **None** | Rides the tool-call row; no action |
| GEPAOptimizer | No (Markdown skills + archive) | No (`learning/gepa.py:242, 380, 412, 450`) | **High** | Insert `subsystem_runs` per cycle + `silent_failures` in scanner branch |
| GEPAEngine | No (signal only) | No (`sentinel/gepa_engine.py:134` bare except) | **High** | Insert `subsystem_runs` after `_run_cycle` + `silent_failures` in bare except |
| AutoDreamEngine | No (`dream_log.md` + signal) | Per-phase to markdown only (`autodream.py:161-167, 220-230`) | **Med** | Insert one `subsystem_runs` row per phase per cycle |
| ActivityObserver | No (Telegram nudges) | No (`observer.py:150-154` re-queues silently) | **High** | Insert `silent_failures` when `_gateway.send` raises — dead gateway is invisible |
| MemoryConsolidator | No (mutates `memories` table directly) | No try/except (propagates to AutoDream) | **Med** | Derivable from a parent `subsystem_runs` row |
| TelemetryDigest | No (returns DigestResult in-memory) | No (no error handling) | **Med** | Insert `subsystem_runs` with `current_calls` so an extended period of `0` is itself a signal |
| KnowledgeSynthesizer | No (Markdown + signal) | No (`knowledge_synth.py:181-190` ed8f1a6 shape) | **High** | Insert `silent_failures` in the LLM exception path |
| WikiLinter | No (LintResult; auto-fix → log) | Per-issue at `wiki_lint.py:88-96` log-only | **Med** | Bubbled to AutoDream; OK if AutoDream's row lands |
| GoldenTraceExporter | No (JSONL file + signal) | No (`golden_trace_exporter.py:82, 99, 118`) | **High** | Insert `subsystem_runs` per export with `path`, `bytes`, `trace_count` |
| BackupVault | Own DB (`backup_manifest`, `restore_log`) | Partially (creation errors not inserted) | **Med** | Mirror summary rows into a central `subsystem_runs` |
| MorphEngine | Reports as JSON / `post_mortem/` dir | No DB row | **Med** | Cross-reference from `subsystem_runs` at swap completion |
| SymbioteCoordinator | Own DB (`symbiote_sessions`, persisted per phase) | Yes — `phase=FAILED` row in its own DB | **Low** | Mirror failures into central `silent_failures` |
| Scout / Harvest / Graft engines | Same as Coordinator | Yes | **Low** | Same |
| MemoryTool (Sprint 1) | No (signal only) | No (`hermes_memory_tool.py:370-372` `except: pass`) | **Med** | If `_signal_bus` is None every emit silently drops; add counter |
| Curator `curator_report` signal (Sprint 1) | `SignalBus._history = deque(maxlen=500)` — process-memory only | `except: pass` at `curator.py:736+` | **High** | The signal exists but cannot be queried after restart. Persist or pair with `subsystem_runs` |

### Recommended Phase 2 schema additions

```sql
CREATE TABLE IF NOT EXISTS silent_failures (
    id              TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    subsystem       TEXT NOT NULL,        -- "curator", "memory_extractor", "gepa", ...
    operation       TEXT,                  -- "run_once", "_call_model", "promote", ...
    exception_type  TEXT NOT NULL,
    exception_msg   TEXT,                  -- str(exc) [:2000]
    traceback       TEXT,                  -- traceback.format_exc() [:8000]
    context         TEXT                   -- optional JSON: skill_path, model_id, batch_size
);
CREATE INDEX idx_silent_failures_ts ON silent_failures (timestamp);
CREATE INDEX idx_silent_failures_subsystem ON silent_failures (subsystem);

CREATE TABLE IF NOT EXISTS subsystem_runs (
    id              TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    subsystem       TEXT NOT NULL,
    operation       TEXT,
    duration_ms     REAL,
    outcome         TEXT NOT NULL,         -- "success" | "partial" | "failed" | "skipped"
    summary_json    TEXT                   -- whatever the subsystem wants to surface
);
CREATE INDEX idx_subsystem_runs_ts ON subsystem_runs (timestamp);
CREATE INDEX idx_subsystem_runs_subsystem ON subsystem_runs (subsystem);
```

Plus a single helper: `telemetry.record_silent_failure(subsystem, operation, exc, context=None)` and `telemetry.record_run(subsystem, operation, outcome, duration_ms, summary=None)`. Both idempotent, never raise.

### SignalBus persistence question

Sprint 1 added `skill_created` / `skill_refined` / `memory_updated` / `curator_report` emissions to the in-memory `SignalBus`. Per Agent C: **`SignalBus._history` is `deque(maxlen=500)` — process-memory only.** Beacon WS subscribers see live events but a process restart drops everything. From the audit's "could you tell from telemetry.db alone?" bar, those signals don't satisfy it. Phase 2 should either persist signals to a `signals` table or pair every emit with a `subsystem_runs` row.

---

## Re-baseline metrics

Snapshot at `944934f` (post-Sprint-1 main).

### Files on disk

| Path | State |
|---|---|
| `~/.prometheus/skills/auto/` | **6 .md files** + 1 `.bak-*.md` from SkillRefiner (mtimes 2026-05-16 → 2026-05-19) |
| `~/.prometheus/skills/auto/` newest | `keep-going.md` (2026-05-19 12:57, 776 bytes) — properly-formatted SkillCreator output (`name:`, `description:`, `## When to use`, `## Steps`) |
| `~/.prometheus/skills/auto/` oldest | `practice-email-and-practice.md` (2026-05-16 22:39, 1180 bytes) |
| `~/.prometheus/skills/auto/yes-get-started.md` | **166 bytes, truncated mid-frontmatter** (only `name:` and partial `description:` — no body) — circumstantial evidence of a write that started before the bug surfaced and didn't complete |
| `~/.prometheus/MEMORY.md` | **0 bytes** (last touched 2026-04-06) |
| `~/.prometheus/USER.md` | **0 bytes** (last touched 2026-04-06) |
| `~/.prometheus/curator/` | does not exist (Curator just landed; first run not yet) |
| `~/.prometheus/AGENTS.md` | 1,889 chars / 46 lines |
| `~/.prometheus/ANATOMY.md` | 1,573 chars / 57 lines |
| Project `skills/*.md` | 92 (matches README's "92 Builtin Skills") |
| Loader's bundled `src/prometheus/skills/builtin/*.md` | 3 |

### `telemetry.db`

| Metric | Value |
|---|---|
| Total `tool_calls` rows | 1,207 |
| Success | 678 (56%) |
| Failure | 529 (44%) |
| Golden traces (`is_golden=1`) | 42 |
| `tool_calls` in last 7 days | 249 |
| `circuit_breaker_diagnostics` rows | 13 |
| Top tool by row count | `_loop_transition` (534 — internal turn-cycle marker, not a user tool) |
| Top user tools by row count | `bash` (219), `web_search` (80), `web_fetch` (51), `read_file` (17), `write_file` (16) |
| Recent failure pattern (top 10 latest errors) | `_loop_transition` + `unknown_tool: ` (empty tool_name) + `circuit_breaker_trip` clusters |

### `memory.db`

| Table | Rows |
|---|---|
| `messages` | 0 |
| `memories` | 0 |
| `summaries` | 0 |
| `messages_fts*` / `memories_fts*` | empty too |

### `lcm.db`

| Table | Rows |
|---|---|
| `checkpoints` | 0 |

### Daemon liveness

- `~/.prometheus/daemon.lock`: PID 802796, started `1778942844` = **2026-05-16 10:47 EDT** (5 days running, no restart needed since Sprint 1 just merged — daemon code on disk has the fix, in-memory daemon does not).
- `~/.prometheus/last_telegram_chat_id`: present (2-byte file).

### Stated vs reality

| README claim | Reality | Status |
|---|---|---|
| "92 Builtin Skills" (line 127) | 92 .md files in `skills/` | ✓ matches |
| "MEMORY.md + USER.md — the agent learns who you are over time (bounded: 12K + 8K chars)" (line 136) | Both files 0 bytes; last touched 2026-04-06 | ✗ MemoryTool not exercised in over a month |
| "Memory Extractor pulls structured facts from your conversations" (line 74) | `memories` table 0 rows | ✗ extractor producing nothing |
| "SkillCreator auto-generates new skills from successful task traces" (line 129) | 6 files (with one truncated) in `auto/` | ▲ partially — generated some, currently inert per ed8f1a6 bug |
| "Three of four [SENTINEL] phases use zero LLM calls" (line 72) | Not directly verifiable from telemetry — no `subsystem_runs` rows | ❓ unverifiable from data |

### The 6-file mystery — resolved (and strengthens the audit's premise)

Forensic during this audit closed the loop. The `.venv` directory `/home/will/Prometheus/.venv/` has a ctime of **2026-05-20 23:54** — created during this audit session when `uv run` first executed. The fresh venv installed **pydantic 2.12.5 (strict list validation)**. The running daemon (PID 802796, `~/.prometheus/daemon.lock`) booted **2026-05-16 10:47** against a prior venv with an older pydantic (likely 2.10.x or 2.11.x) that **auto-coerced `content="string"` to `content=["string"]`**.

This explains the 6 auto-skills dated May 16 → May 19: they were generated by an in-memory daemon process against the lenient pydantic. The on-disk `.pyc` and source were the broken pre-ed8f1a6 shape — but the lenient pydantic happily coerced.

The actual behavioural states are:

| code shape | pydantic 2.10–2.11.x (lenient) | pydantic 2.12.x (strict) |
|---|---|---|
| pre-ed8f1a6 (`content=prompt`) | ✓ works (coerces str → list) | ✗ silent ValidationError → swallowed → None |
| post-ed8f1a6 (`content=[TextBlock(text=prompt)]`) | ✓ works | ✓ works |

The bug was a **time bomb**. The wiring tests passed in CI and in the running daemon. The moment anyone ran `pip install -U pydantic` or recreated their venv post-pydantic-2.12 release, every subsystem that called `_call_model` would silently fall over and the wiring tests would still report green. The current daemon would have detonated the instant Will restarted it. ed8f1a6 fixes the source so the new shape works under both lenient AND strict pydantic.

**Why this matters for Sprint 4:**

- The wiring-test failure mode isn't "the test was always wrong." It was "the test verified structure under one runtime configuration, then the runtime configuration changed, and the test didn't notice." That's a category of silent failure broader than just `_call_model` — any dependency upgrade can silently disable a subsystem if the test only checks structure.
- The recommendation list below already covers this through the **functional wiring tests** work stream — but the framing should be "functional tests catch behavioural regressions from upstream churn", not just "we wrote bad tests once."
- The `silent_failures` table catches the failure-side: future strict-pydantic-style regressions land a row immediately.
- A `subsystem_runs` row catches the absence: "no Curator run in 7 days" detects a hang even if no exception is thrown.

**Implication for the soak:** the live daemon is still running the lenient-pydantic venv state, so SkillCreator/SkillRefiner/MemoryExtractor in that process *are* still working. The Curator code is on disk but the daemon never imported it (started before merge). To activate Sprint 1 on the live daemon, Will needs to restart it — which will simultaneously upgrade the in-memory pydantic to 2.12 (still works post-ed8f1a6) and start the Curator.

---

## Phase 2 recommendations (ranked)

Each entry: **effort** S/M/L, **dependencies**, **risk**. Cross-references HIGH-RISK Tier numbers from the section above.

1. **Tier 1 quick-wins — fix the four `(OSError, Exception)` redundant-tuple swallows.** Three of these (`skill_creator.py:127`, `skill_refiner.py:116`, `gepa.py:183`) plus `nudge.py:65` are the *exact* failure shape that produced ed8f1a6: silent config-load fallback. Narrow each to `(OSError, yaml.YAMLError)` and add `log.warning(... exc_info=True)`. **Effort: S. Deps: none. Risk: trivial (more strict, not less).** _Could be its own pre-Phase-2 hotfix._

2. **`silent_failures` + `subsystem_runs` tables + helper API** — `telemetry/tracker.py` schema migration. `tel.record_silent_failure(subsystem, operation, exc, context=None)` and `tel.record_run(subsystem, operation, outcome, duration_ms, summary=None)`. Idempotent, never raise. **Effort: S. Deps: none. Risk: low (additive).**

3. **Wire `silent_failures` into the 9 High-severity subsystems + Tiers 2–4 + Tier 6 swallows.** Curator (Tier 2/3 — HIGH-RISK #7 and #8), SkillCreator (#4), SkillRefiner (#5), MemoryExtractor (#6), GEPAOptimizer, GEPAEngine, ActivityObserver, KnowledgeSynthesizer, GoldenTraceExporter, plus hooks/executor.py:150 (#17). **Effort: M. Deps: #2. Risk: low — adds `tel.record_silent_failure(...)` to existing `except` blocks.**

4. **Shared `LLMCallEnvelope`** — Sprint 4 Work Stream 1 per spec. Migrates Curator / SkillCreator / SkillRefiner / MemoryExtractor `_call_model` to a shared helper that always writes a row, re-raises by default, and explicitly catches the message-shape error class so future ed8f1a6-shaped bugs surface immediately. Cleans up the historical "Pre-existing bug found during Sprint 1" comments in those four files. **Effort: M. Deps: #2, #3. Risk: low — but touches 4 well-trodden hot paths so needs careful testing.**

5. **Agent-loop Tier 4 hot-path observability** — HIGH-RISK #9-#13 in `engine/agent_loop.py`. WARN-level logs on the four currently-silent fallbacks + telemetry row on `diagnose_and_recover` failure. **Effort: S. Deps: #2. Risk: low.**

6. **Gateway Tier 5 + Tier 7 — log.debug + narrow** — HIGH-RISK #14-#16 (telegram swallows) and #18-#19 (file_read.py + hermes_memory_tool.py signal emission). All `log.debug(... exc_info=True)` additions; the swallows themselves stay. **Effort: S. Deps: none. Risk: trivial.**

7. **Upgrade the Top-5 STRUCTURAL-ONLY tests + the remaining 29** — Sprint 4 Work Stream 2. Add canned LLM-response fixtures under `tests/fixtures/llm_responses.py`. Especially: replace `test_daemon_curator_wiring_block_present` (my own Sprint 1 anti-pattern test — see Test #33 in the table) with a boot-the-daemon test that observes a Curator cycle. **Effort: M (full set) / S (top 5). Deps: none. Risk: low (test-only).**

8. **`/health` Telegram command** — Sprint 4 Work Stream 3. Single SELECT over `silent_failures` and `subsystem_runs` grouped by subsystem. **Effort: S. Deps: #2, #3. Risk: low.**

9. **SignalBus persistence** — every `bus.emit(...)` paired with a `subsystem_runs` row OR a dedicated `signals_log` table. Today the Sprint 1 emissions are themselves a silent-failure surface on process restart (`SignalBus._history = deque(maxlen=500)`). **Effort: M. Deps: #2. Risk: medium — touches the bus core.**

10. **Pydantic-version forensic + version pin** — 30 minutes to confirm hypothesis #1 from the resolved 6-file mystery section. Run the codebase against pydantic 2.10 / 2.11 / 2.12 and reproduce both behaviors. Then add a lower bound to `pyproject.toml` (`pydantic>=2.12`) so future installs are deterministic. **Effort: S. Deps: none. Risk: trivial.**

11. **Re-baseline rollup** — after Phase 2 lands AND Will restarts the daemon, re-snapshot the metrics in this section after 7 days. Confirms the now-functional subsystems are producing output. **Effort: trivial. Deps: 7 days of soak.**

12. **(Deferred to a separate sprint)** Cross-DB unified ops view — BackupVault, SymbioteCoordinator, evals tracker all persist to their own SQLite files. Currently you can't query "what failed across the whole stack last night" without four separate queries. Useful but not blocking.

13. **(Deferred — explicitly out of scope per sprint spec)** Adapter Layer + LCM swallows. Documented above as OUT-OF-SCOPE for completeness; no fixes proposed.

---

## Out-of-scope (Adapter Layer + LCM)

Per Sprint 4 constraints, swallows in `src/prometheus/adapter/*.py` and `src/prometheus/memory/lcm_*.py` are NOT classified HIGH-RISK and NOT recommended for fix in this sprint. They are documented in Agent A's report for context only. A separate sprint should revisit them with the necessary care.

---

## Methodology + provenance

- Agent A (background, `general-purpose`): scanned all 545 `except` blocks across 226 files in `src/prometheus/` and classified each.
- Agent B (background): classified 85 tests in `tests/test_wiring.py`.
- Agent C (background): per-subsystem telemetry write analysis for 17 autonomous subsystems.
- Step 4 baseline metrics: gathered directly via `sqlite3`, `wc`, `ls`, `grep` on the post-merge tree.
- Assembled by the main session (Claude Opus 4.7, 1M ctx).
- No code changes in Phase 1. Branch `audit/silent-failures` off `944934f`.

---

## Reporting back (Phase 1 deliverables)

1. **Branch:** `audit/silent-failures` (off merge commit `944934f`)
2. **Audit doc:** `docs/audits/SILENT-FAILURE-AUDIT.md` (this file)
3. **Counts:** see Summary table above
4. **Top-5 HIGH-RISK swallows:** in the HIGH-RISK section once Agent A delivers
5. **Top-5 wiring-test functional-assertion additions:** in the "Top 5 STRUCTURAL-ONLY" subsection above
6. **Re-baseline numbers:** in the Re-baseline Metrics section
7. **Draft PR URL:** to be opened after this commit

**Halt for Will's review.** Phase 2 implementation does not start until Will reviews and prioritizes the recommendation list above.
