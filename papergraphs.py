import itertools
import time
import params
from QAOA import QAOArunner
from datetime import date
import pandas as pd
from solver import Solver
import ray
import numpy as np
import yagmail
import logging


from MaxCutProblem import MaxCutProblem
problem = MaxCutProblem()

with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()
import ast
import networkx as nx
from tqdm import tqdm

local = True

if not local:
    with open("test_settings.txt", "r") as f:
        settings = ast.literal_eval(f.read().strip())
if local: 
    settings = "strign"

@ray.remote(num_cpus = 4)
def parallell_runner(parameters, graph, name):
    qaoa = QAOArunner(graph, simulation=True, **parameters)
    qaoa.build_circuit()
    qaoa.run()
    return { **parameters,'graph_size': len(graph.nodes()), 'graph_name' : name,
         'time_elapsed': qaoa.time_elapsed, 'quantum_func_evals': qaoa.fev, 'obj_func_evolution': qaoa.objective_func_vals,
        'quantum_solution':qaoa.solution, 'quantum_obj_value' : qaoa.objective_value, 
        'classic_solution' : qaoa.classical_solution, 'classic_value': qaoa.classical_objective_value , 'final_params': qaoa.final_params, 'percent_measure_optimal': qaoa.get_prob_most_likely_solution()
                        }

if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')
ray.init(log_to_driver=True)


graphs= [problem.get_erdos_renyi_graphs([5,7,9])]
graphs.reverse()

graphs = list(itertools.chain.from_iterable(graphs)) #should be lists from before, no?

combos = [settings, graphs] #settings should be a list of dictionaries .


# Convert graphs to networkx graphs and generate graph6 strings
graph6_strings = []
for graph in graphs: #TODO: write graph6 decoder

    graph = nx.Graph(list(graph.edge_list())) 
    graph6_string = nx.to_graph6_bytes(graph).decode('utf-8').strip()
    graph6_strings.append(graph6_string)


print('Graph6 strings', graph6_string)
all_combos = list(itertools.product(*combos))

combos_with_name = []
for liste in all_combos:
    liste2 = liste +  (graph6_strings[graphs.index(liste[1])],) #tuples are immutable
    combos_with_name.append(liste2)
all_combos = combos_with_name

n_times = 50
all_combos *= n_times



#TODO: make dictionary
print('len all_combos',len(all_combos))
print(f'performing all {n_times} times')


print('Settings:', settings)
data = []
parameter_set = []    

# Find keys with different values across the dictionaries in settings
keys_with_differences = []

keys = settings[0].keys()
for key in keys:
    values = {d[key] for d in settings}
    flag= False
    if len(values) > 1:  # If there are multiple unique values for this key
        keys_with_differences.append(key)
        flag=True
    if flag:
        keys_with_differences.append(values)

parameter_set = keys_with_differences
print('parameter set', parameter_set)


parameter_string = [str(x) + "_" for x in parameter_set]
parameter_string = "".join(parameter_string)
parameter_string = parameter_string[0:-1]

print('parameter string', parameter_string)

futures = [parallell_runner.remote(parameters, graph, name) for parameters, graph,name in all_combos]

result_ids = []
unfinished = futures

with tqdm(total=len(futures), desc="Processing tasks") as pbar:
    start_time = time.time()
    max_runtime = 3 * 24 * 60 * 60  # 3 days in seconds

    while unfinished:
        if time.time() - start_time > max_runtime:
            print("Maximum runtime exceeded. Breaking the loop.")
            break
        done, unfinished = ray.wait(unfinished, num_returns=10)
        result_ids.extend(done)
        pbar.update(len(done))

for task in unfinished:
    ray.cancel(task)

underway_df = pd.DataFrame(ray.get(result_ids))
underway_df.to_csv(f'results/results_underway.csv', mode='a', header=False)
data.extend(ray.get(result_ids))
print(f'Done with Parameters: {settings} at time: {time.time()}')


df = pd.DataFrame(data)
df.to_csv(f'results/results_papergraph_{parameter_string}.csv')

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = f'Solstorm run -papergraph -  {parameter_string}'
attachment = f'results_papergraph_{parameter_string}.csv'

yag.send( subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()

