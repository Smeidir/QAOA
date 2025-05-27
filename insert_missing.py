# insert_missing.py  ─────────────────────────────────────────────────────
"""
Usage
-----
python insert_missing.py missing.txt --reps 50

* missing.txt  : the plain-text log containing “has X entries and needs Y more”
* --reps       : target #reps per (param combo, graph)  [default 50]

For each “needs k more” line, this adds k rows to `runs` with state='pending'.
"""

import argparse, json, re, secrets, sqlite3, pathlib
from datetime import datetime
from contextlib import closing

DB_PATH = "qruns.db"           # adjust if yours lives elsewhere

# ----------------------------------------------------------------------
# Regex that matches the log lines
PATTERN = re.compile(
    r"^For (?P<variant>\w+) and depth (?P<depth>\d+) the combination "
    r"\(warm_start=(?P<warm_start>\w+), hamming_dist (?P<hdist>\d), graph=>>graph6<<(?P<graph>[^\)]+)\) "
    r"has (?P<done>\d+) entries and needs (?P<need>\d+) more\."
)
GRAPH_MAP = {
    "Emz_":           "paper1_0.pkl",
    "HmzffJz":       "paper1_1.pkl",
    "KmzffJznl{hU":  "paper1_2.pkl",
    "Hh_iS_u":       "paper1_3.pkl",
}


def make_row(params: dict):
    """Return (params_json, state, node, started_at, finished_at, artefact_path, error_msg)."""
    return (
        json.dumps(params),
        "pending",        # state
        None,             # node
        None,             # started_at
        None,             # finished_at
        None,             # artefact_path
        None,             # error_msg
    )

def main(txt_path: pathlib.Path, target_reps: int):
    added = 0
    with open(txt_path, "r") as f, sqlite3.connect(DB_PATH) as db:
        for line in f:
            m = PATTERN.search(line.strip())
            if not m:
                continue           # skip non-matching lines

            need = int(m["need"])
            if need == 0:
                continue

            # build the static part of params
            base = {
                "backend_mode":  "noisy_sampling",
                "qaoa_variant":  m["variant"],
                "param_initialization": "gaussian",
                "depth":        int(m["depth"]),
                "warm_start":   m["warm_start"] == "True",
                "problem_type": "minvertexcover",
                "hamming_dist": int(m["hdist"]),
                "graph_path": f"graphs_paper1/{GRAPH_MAP[m['graph']]}",

            }

            # insert `need` rows
            rows = []
            for _ in range(need):
                row_params = {**base}
                rows.append(make_row(row_params))
            db.executemany("""
              INSERT INTO runs
              (params, state, node, started_at, finished_at,
               artefact_path, error_msg)
              VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            added += len(rows)

    print(f"[{datetime.now():%F %T}] Inserted {added} new rows → {DB_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Text file with 'needs N more' lines")
    ap.add_argument("--reps", type=int, default=50,
                    help="Target repetitions (informational only)")
    args = ap.parse_args()
    
    main(pathlib.Path(args.logfile), args.reps)
