BEGIN TRANSACTION;
CREATE TABLE circuit_breaker_diagnostics (
    id                TEXT PRIMARY KEY,
    timestamp         REAL NOT NULL,
    model_id          TEXT NOT NULL,
    adapter_tier      TEXT NOT NULL,
    tool_name         TEXT NOT NULL,
    failure_category  TEXT NOT NULL,
    config_drift      INTEGER NOT NULL DEFAULT 0,   -- 0 or 1
    raw_sample        TEXT,                          -- first 500 chars of failed output
    recovered         INTEGER NOT NULL DEFAULT 0,    -- 0 or 1
    recovery_method   TEXT,                          -- "tier_bump", "none", etc.
    golden_reference  TEXT                           -- Golden Trace sprint: best-match golden parsed_tool_call
);
INSERT INTO "circuit_breaker_diagnostics" VALUES('fixture-circuit_breaker_diagnostics-1',1.79081489240372037e+09,'fixture-local','light','bash','malformed',0,'sample',1,'tier_bump',NULL);
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO "schema_meta" VALUES('latency_nullable_since','0.0');
INSERT INTO "schema_meta" VALUES('billing_recorded_since','1790814892.3478322');
INSERT INTO "schema_meta" VALUES('created_at','2026-09-30 00:00:00');
INSERT INTO "schema_meta" VALUES('created_by','prometheus telemetry v3');
INSERT INTO "schema_meta" VALUES('schema_version','3');
CREATE TABLE signal_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,        -- ISO8601 UTC
    signal_type       TEXT NOT NULL,        -- ActivitySignal.kind: "skill_created", ...
    payload           TEXT NOT NULL,        -- JSON blob of ActivitySignal.payload
    source_subsystem  TEXT NOT NULL,        -- ActivitySignal.source: "SkillCreator", ...
    read_at           TEXT                  -- nullable: when surfaced to user (reserved)
);
INSERT INTO "signal_events" VALUES(1,'2026-09-30T00:00:00+00:00','idle_start','{"n": 1}','Fixture',NULL);
CREATE TABLE silent_failures (
    id              TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    subsystem       TEXT NOT NULL,        -- "curator" | "skill_creator" | ...
    operation       TEXT,                  -- "_call_model" | "run_once" | ...
    exception_type  TEXT NOT NULL,         -- type(exc).__name__
    exception_msg   TEXT,                  -- str(exc) [:2000]
    traceback       TEXT,                  -- traceback.format_exc() [:8000]
    context         TEXT                   -- optional JSON: skill_path, model_id, ...
, response_body TEXT);
INSERT INTO "silent_failures" VALUES('fixture-silent_failures-1',1.79081489240115427e+09,'fixture','op','RuntimeError','synthetic failure','Traceback (fixture)','{"k": "v"}',NULL);
CREATE TABLE subsystem_runs (
    id              TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    subsystem       TEXT NOT NULL,
    operation       TEXT,
    duration_ms     REAL,
    outcome         TEXT NOT NULL,         -- "success" | "partial" | "failed" | "skipped"
    summary_json    TEXT,                  -- arbitrary JSON the subsystem wants to surface
    -- SPRINT-loop-envelope (F1) additions (nullable for backcompat):
    input_tokens    INTEGER,               -- UsageSnapshot.input_tokens for LLM calls
    output_tokens   INTEGER,               -- UsageSnapshot.output_tokens for LLM calls
    round_index     INTEGER,               -- loop turn number (0-based) for agent_loop rows
    session_id      TEXT,                  -- LoopContext.session_id for agent_loop rows
    model           TEXT,                  -- model id the call was made with
    thinking        INTEGER                -- effective flag: 1 on, 0 suppressed, NULL unknown
, cached_input_tokens INTEGER, cache_write_tokens INTEGER, node_id TEXT, billing_mode TEXT, billing_marker TEXT);
INSERT INTO "subsystem_runs" VALUES('fixture-subsystem_runs-1',1.7908148923968935e+09,'agent_loop','loop_round',840.0,'success','{"stop_reason": "end_turn"}',1200,80,0,'desktop:fixture','fixture-local',0,NULL,NULL,NULL,'local',NULL);
INSERT INTO "subsystem_runs" VALUES('fixture-subsystem_runs-2',1.7908148923997395e+09,'curator','run_once',0.0,'skipped',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
CREATE TABLE tool_calls (
    id                TEXT PRIMARY KEY,
    timestamp         REAL NOT NULL,
    model             TEXT NOT NULL,
    tool_name         TEXT NOT NULL,
    success           INTEGER NOT NULL,   -- 0 or 1
    retries           INTEGER NOT NULL DEFAULT 0,
    -- NULLABLE SINCE SCHEMA v2. `NOT NULL DEFAULT 0.0` made "nobody
    -- measured this" and "this took zero milliseconds" the same stored
    -- value; see telemetry/latency.py. Rows written before v2 keep their
    -- 0.0 and are NOT backfilled — readers tag them `unknown` instead.
    latency_ms        REAL,
    error_type        TEXT,
    error_detail      TEXT,
    -- Golden Trace Capture sprint additions (nullable for backcompat):
    raw_model_output  TEXT,                -- raw text the model produced BEFORE adapter parsing
    parsed_tool_call  TEXT,                -- validated tool call as JSON {"name": ..., "input": {...}}
    is_golden         INTEGER NOT NULL DEFAULT 0, -- 1 = cloud + success + zero retries + captured raw
    repairs           INTEGER NOT NULL DEFAULT 0, -- M2: adapter repairs applied (fuzzy name, coercion, ...)
    -- What the server said actually served this call, echoed back in the
    -- completion response. SEPARATE from `model`, which is the name the
    -- caller REQUESTED. They disagree whenever a harness passes a config
    -- string that no longer matches the loaded model — which is why
    -- `gemma4-26b` rows kept being written months after the server moved to
    -- Qwen. NULL when the provider does not echo one.
    served_model      TEXT
, session_id TEXT, tool_schema TEXT, node_id TEXT);
INSERT INTO "tool_calls" VALUES('fixture-tool_calls-1',1.79081489236659407e+09,'fixture-local','bash',1,0,12.5,NULL,NULL,'{"name": "bash", "input": {"command": "ls"}}','{"name": "bash", "input": {"command": "ls"}}',0,0,NULL,'desktop:fixture','{"name": "bash"}',NULL);
INSERT INTO "tool_calls" VALUES('fixture-tool_calls-2',1.79081489239251971e+09,'fixture-cloud','read_file',0,1,NULL,'tool_error','no such file',NULL,NULL,0,1,NULL,'desktop:fixture',NULL,NULL);
INSERT INTO "tool_calls" VALUES('fixture-tool_calls-3',1.7908148923944087e+09,'fixture-local','_loop_transition',1,0,NULL,NULL,NULL,NULL,NULL,0,0,NULL,'desktop:fixture',NULL,NULL);
INSERT INTO "tool_calls" VALUES('fixture-tool_calls-4',1.79081489239566349e+09,'fixture-local','grep',1,0,NULL,NULL,NULL,NULL,NULL,0,0,NULL,NULL,NULL,NULL);
CREATE INDEX idx_tool_calls_model ON tool_calls (model);
CREATE INDEX idx_tool_calls_tool ON tool_calls (tool_name);
CREATE INDEX idx_tool_calls_golden ON tool_calls (is_golden);
CREATE INDEX idx_tool_calls_ts ON tool_calls (timestamp DESC);
CREATE INDEX idx_cb_diag_timestamp ON circuit_breaker_diagnostics (timestamp);
CREATE INDEX idx_cb_diag_model ON circuit_breaker_diagnostics (model_id);
CREATE INDEX idx_cb_diag_tool ON circuit_breaker_diagnostics (tool_name);
CREATE INDEX idx_silent_failures_ts ON silent_failures (timestamp);
CREATE INDEX idx_silent_failures_subsystem ON silent_failures (subsystem);
CREATE INDEX idx_subsystem_runs_ts ON subsystem_runs (timestamp);
CREATE INDEX idx_subsystem_runs_subsystem ON subsystem_runs (subsystem);
CREATE INDEX idx_subsystem_runs_session_ts ON subsystem_runs (session_id, timestamp);
CREATE INDEX idx_signal_events_type_time
    ON signal_events (signal_type, timestamp DESC);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('signal_events',1);
COMMIT;
