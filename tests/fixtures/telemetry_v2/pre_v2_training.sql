BEGIN TRANSACTION;
CREATE TABLE training_pairs (
    id            TEXT PRIMARY KEY,
    timestamp     REAL NOT NULL,
    pair_source   TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    context       TEXT,              -- JSON (see context kinds below)
    rejected      TEXT,              -- JSON {"name":..., "input":...}; NULL for cloud_golden
    chosen        TEXT NOT NULL,     -- JSON {"name":..., "input":...}
    meta          TEXT,              -- JSON: error feedback, repair log, latency
    context_hash  TEXT NOT NULL UNIQUE  -- sha256(context + rejected) — dedupe
);
INSERT INTO "training_pairs" VALUES('fixture-pair-1',1.790814892419456e+09,'retry_success','fixture-local','bash','{"messages": [{"content": "list files", "role": "user"}]}','{"input": {"cmd": "ls"}, "name": "bash"}','{"input": {"command": "ls"}, "name": "bash"}','{"note": "fixture"}','fixture-hash-1');
INSERT INTO "training_pairs" VALUES('fixture-pair-2',1.79081489242206978e+09,'cloud_golden','fixture-cloud','read_file','{"messages": []}',NULL,'{"input": {"path": "README.md"}, "name": "read_file"}','{}','fixture-hash-2');
CREATE INDEX idx_pairs_source ON training_pairs (pair_source);
CREATE INDEX idx_pairs_tool ON training_pairs (tool_name);
CREATE INDEX idx_pairs_ts ON training_pairs (timestamp);
COMMIT;
