# prep_graphs_paper1.py
#
# ❶ Imports your existing helper ❷ Pickles every graph ❸ Saves a JSON map
#    label → file-path   that make_grid.py can read.

import pathlib, pickle, json
from src.qaoa.models.MaxCutProblem import MaxCutProblem   # ← adjust import

OUTDIR = pathlib.Path(__file__).parent / "graphs_paper1"   # relative path
OUTDIR.mkdir(parents=True, exist_ok=True)

problem = MaxCutProblem()

graph_paths = {}                                       # label → path str
for idx, G in enumerate(problem.get_erdos_renyi_graphs_paper1()):
    label = f"paper1_{idx}"                            # paper1_0 … paper1_3
    path  = OUTDIR / f"{label}.pkl"
    with path.open("wb") as f:
        pickle.dump(G, f)
    graph_paths[label] = str(path)

with open("graph_paths.json", "w") as f:               # small JSON in repo
    json.dump(graph_paths, f, indent=2)

print(f"Saved {len(graph_paths)} graphs → {OUTDIR}")
print("Wrote graph_paths.json")
