# papergraphs_missing60.py

import ast
import pandas as pd
import ray
from QAOA import QAOArunner

import networkx as nx
import itertools
from MaxCutProblem import MaxCutProblem


problem = MaxCutProblem()
# Load previous results
df = pd.read_csv("results_papergraph_p_10_vanilla_param_initialization_{'gaussian', 'static'}_warm_start_{False, True}.csv")
df["combined"] = df["graph_name"] + "_" + df["warm_start"].astype(str) + "_" + df["param_initialization"]

# Count how many times each combination appears
comb_counts = df["combined"].value_counts()

# Define total needed per combination
total_per_combo = 50
missing_data = comb_counts[comb_counts < total_per_combo]

# Reconstruct all graphs
all_graphs = list(itertools.chain.from_iterable([problem.get_erdos_renyi_graphs([5,7,9])]))

# Generate graph6 strings
graph6_to_nx = {}
for g in all_graphs:
    G_nx = nx.Graph(list(g.edge_list()))
    g6 = nx.to_graph6_bytes(G_nx).decode('utf-8').strip()
    graph6_to_nx[g6] = g
local = True

if not local:
    with open("test_settings.txt", "r") as f:
        settings = ast.literal_eval(f.read().strip())
if local: 
    settings = "[{'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'gaussian', 'depth': 10, 'warm_start': False}, {'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'gaussian', 'depth': 10, 'warm_start': True}, {'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'static', 'depth': 10, 'warm_start': False}, {'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'static', 'depth': 10, 'warm_start': True}]"
    settings = ast.literal_eval(settings)

@ray.remote(num_cpus = 4)
def parallell_runner(parameters, graph, name):
    qaoa = QAOArunner(graph, **parameters)
    qaoa.build_circuit()
    qaoa.run()
    return { **parameters,'graph_size': len(graph.nodes()), 'graph_name' : name, #TODO: move into QAOArunner class
         'time_elapsed': qaoa.time_elapsed, 'quantum_func_evals': qaoa.fev, 'ratio' : qaoa.objective_value/qaoa.classical_objective_value,
        'quantum_solution':qaoa.solution, 'quantum_obj_value' : qaoa.objective_value, 
        'classic_solution' : qaoa.classical_solution, 'classic_value': qaoa.classical_objective_value , 
        'final_params': qaoa.final_params, 'percent_measure_optimal': qaoa.get_prob_measure_optimal()
                        }

if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')
ray.init(log_to_driver=True)


# Create specific combo jobs based on what's missing
jobs = []
for combo, count in missing_data.items():
    graph_name, warm_start, param_init = combo.split("_")
    warm_start = warm_start == 'True'
    needed = total_per_combo - count
    matching_setting = [s for s in settings if s['param_initialization'] == param_init and s['warm_start'] == warm_start][0]
    for _ in range(needed):
        jobs.append((matching_setting, graph6_to_nx[graph_name], graph_name))
print(f"Running {len(jobs)} missing jobs...")

futures = [parallell_runner.remote(p, g, name) for p, g, name in jobs]

# Run and store results
from tqdm import tqdm
results = []
with tqdm(total=len(futures)) as pbar:
    while futures:
        done, futures = ray.wait(futures, num_returns=5)
        results.extend(ray.get(done))
        pbar.update(len(done))

# Save and email
new_df = pd.DataFrame(results)
new_df.to_csv('results/results_missing_60.csv', index=False, mode='a', header=True)
print("Missing results saved.")
