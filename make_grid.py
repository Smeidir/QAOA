# make_grid.py ----------------------------------------------------------
"""
Generate every combination in `settings`, add a random seed, and
INSERT one row per (combination, repetition) into qruns.db.

Run:   python make_grid.py --reps 30          # default db = qruns.db
       python make_grid.py --db alt.db --reps 10
"""

import itertools, sqlite3, json, secrets, argparse, datetime, pathlib,pickle

# ----------------------------------------------------------------------
# 1. Your original settings block
#    (edit here whenever you change an experiment)
# ----------------------------------------------------------------------
settings = {
    "backend_mode":        ["statevector"],              # or ['noisy_sampling']
    "qaoa_variant":        ["vanilla"],
    "param_initialization":["gaussian"],
    "depth":               [1, 2],
    "warm_start":          [True],
    "problem_type":        ["minvertexcover"]
}

graph_paths = json.load(open("graph_paths.json"))   # {'paper1_0': '/scratch/…'}



# ----------------------------------------------------------------------
DDL = """
CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY,
  params        TEXT,            -- JSON (settings + graph)
  state         TEXT,            -- pending | running | done | error
  node          TEXT,
  started_at    TIMESTAMP,
  finished_at   TIMESTAMP,
  artefact_path TEXT,
  error_msg     TEXT
);
"""

import itertools, json, secrets

def build_rows(reps: int):
    """
    Yield (params_json, 'pending') tuples for executemany().
    Each params_json contains
        • all hyper-parameters in `settings`
        • graph_path  – absolute / relative path to the pickled retworkx graph
    """
    # Cartesian product of the hyper-parameter grid
    keys, ranges = zip(*settings.items())          # ('backend_mode', 'qaoa_variant', ...)
    for combo in itertools.product(*ranges):
        hp = dict(zip(keys, combo))                # one concrete hyper-param set

        # Add one job row per graph label × repetition
        for g_label, g_path in graph_paths.items():     # graph_paths from graph_paths.json
            base = {
                "graph_label": g_label,               # optional, handy for logs
                "graph_path":  g_path,                # <- workers pickle.load() this
                **hp
            }
            for _ in range(reps):
                row_dict = {**base}
                yield json.dumps(row_dict), "pending"


def main(db_path: pathlib.Path, reps: int):
    with sqlite3.connect(db_path) as db:

        db.executescript(DDL)                              # guarantee schema
        cur = db.cursor()
        cur.executemany(
            "INSERT INTO runs (params, state) VALUES (?, ?)",
            build_rows(reps)
        )
        inserted = cur.rowcount
        db.commit()
    ts = datetime.datetime.now().strftime("%F %T")
    print(f"[{ts}] Inserted {inserted:,} rows into {db_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   default="qruns.db", help="SQLite file")
    ap.add_argument("--reps", type=int, default=30, help="repetitions")
    args = ap.parse_args()
    main(pathlib.Path(args.db), args.reps)
