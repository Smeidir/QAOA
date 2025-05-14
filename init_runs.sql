-- init_runs.sql  (run once)

DROP TABLE IF EXISTS runs;
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    params TEXT,
    state TEXT,
    node TEXT,
    started_at TEXT,
    finished_at TEXT,
    artefact_path TEXT,
    error_msg TEXT,
    results TEXT
);
