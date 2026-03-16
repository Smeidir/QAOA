"""
Generate every combination in `settings`, add a random seed, and
INSERT one row per (combination, repetition) into qruns.db.

Run:   python make_grid.py --reps 50          # default db = qruns.db
       python make_grid.py --db alt.db --reps 10
"""

import itertools, sqlite3, json, secrets, argparse, datetime, pathlib,pickle, base64
from src.qaoa.models.MaxCutProblem import MaxCutProblem
import rustworkx as rx
problem = MaxCutProblem()

# ----------------------------------------------------------------------
# 1. Your original settings block
#    (edit here whenever you change an experiment)
# ----------------------------------------------------------------------

settings = {
    "backend_mode":        ["statevector"],
    "qaoa_variant":        ["multiangle"],
    "param_initialization":["uniform"],
    "optimizer":           ["COBYLA"],
    "depth":               [1,2,3,4,5,6,7],
    "warm_start":          [False],
    "lagrangian_multiplier": [2],
    "problem_type":        ["minvertexcover"],
    "graph_size":           [6,8,10,12],
    "graph_degree":         [3],
    "graph_weighted":        [True]
}
#Configure graphs here:



#graph_paths = json.load(open("graph_paths.json"))   # {'paper1_0': '/scratch/…'}

# Filter out specific graphs that should be excluded
#graph_paths = {k: v for k, v in graph_paths.items() if k not in ["paper1_0","paper1_3"]}


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


GRAPH_KEYS = {"graph_size", "graph_degree", "graph_weighted"}

def build_rows(reps: int, ngraphs: int):
    # Split settings into graph-defining keys vs the rest
    graph_settings = {k: v for k, v in settings.items() if k in GRAPH_KEYS}
    hp_settings    = {k: v for k, v in settings.items() if k not in GRAPH_KEYS}

    # Cartesian product over NON-graph settings
    hp_keys, hp_ranges = zip(*hp_settings.items()) if hp_settings else ((), ())

    # Cartesian product over graph parameters (often just 1x1x1, but supports multiple)
    g_keys, g_ranges = zip(*graph_settings.items())
    for g_combo in itertools.product(*g_ranges):
        g_spec = dict(zip(g_keys, g_combo))

        # Make `reps` random graphs for this graph spec
        for _ in range(ngraphs):
            g = problem.random_regular_rx(
                g_spec["graph_size"],
                g_spec["graph_degree"],
                weights=g_spec["graph_weighted"],
            )
            graph_b64 = base64.b64encode(
                pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL)
            ).decode("ascii")

            # For each graph instance, repeat all hyperparameter combos
            for hp_combo in itertools.product(*hp_ranges) if hp_ranges else [()]:
                hp = dict(zip(hp_keys, hp_combo)) if hp_keys else {}
                row = {**hp, **g_spec, "graph_pickle_b64": graph_b64}
                for _ in range(reps):
                    yield json.dumps(row), "pending"



def main(db_path: pathlib.Path, reps: int, ngraphs: int):
    with sqlite3.connect(db_path) as db:

        db.executescript(DDL)                              # guarantee schema
        cur = db.cursor()
        cur.executemany(
            "INSERT INTO runs (params, state) VALUES (?, ?)",
            build_rows(reps, ngraphs)
        )
        inserted = cur.rowcount
        db.commit()
    ts = datetime.datetime.now().strftime("%F %T")
    print(f"[{ts}] Inserted {inserted:,} rows into {db_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   default="qruns.db", help="SQLite file")
    ap.add_argument("--reps", type=int, default=20, help="repetitions")
    ap.add_argument("--ngraphs", type=int, default=50, help="numberOfGraphs")
    args = ap.parse_args()
    main(pathlib.Path(args.db), args.reps, args.ngraphs)

"""
Code blocks for paper:


settings = {
    "backend_mode":        ["statevector"],
    "qaoa_variant":        ["vanilla","multiangle"],
    "param_initialization":["uniform"],
    "depth":               [1,4,7,10],
    "warm_start":          [False],
    "lagrangian_multiplier": [2],
    "problem_type":        ["minvertexcover"]
}

Next, only for graph size 9
settings = {
    "backend_mode":        ["statevector"],
    "qaoa_variant":        ["vanilla"],
    "param_initialization":["uniform"],
    "depth":               [4],
    "warm_start":          [False],
    "lagrangian_multiplier": [1,4,10], #since 2 already exists
    "problem_type":        ["minvertexcover"]
}
settings = {
    "backend_mode":        ["statevector"],
    "qaoa_variant":        ["multiangle"],
    "param_initialization":["uniform"],
    "depth":               [1],
    "warm_start":          [False],
    "lagrangian_multiplier": [1,4,10], #since 2 already exists
    "problem_type":        ["minvertexcover"]
}
Still only size 9:
settings = {
    "backend_mode":        ["statevector"],
    "qaoa_variant":        ["vanilla"],
    "param_initialization":["uniform"],
    "depth":               [4],
    "warm_start":          [False],
    "lagrangian_multiplier": [1,2,4,10],
    "problem_type":        ["minvertexcover"],
    "warm_start":          [True],
    "hamming_dist":      [0,1,3,5],

}


"""