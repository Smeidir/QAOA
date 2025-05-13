-- init_runs.sql  (run once)
CREATE TABLE runs (
    id            INTEGER PRIMARY KEY,
    params        TEXT,          -- JSON dict incl. random seed
    state         TEXT,          -- 'pending' | 'running' | 'done' | 'error'
    node          TEXT,
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    artefact_path TEXT,
    error_msg     TEXT
);
