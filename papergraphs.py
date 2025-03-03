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

with open("test_settings.txt", "r") as f:
    settings = ast.literal_eval(f.read().strip())


logging.basicConfig(level=logging.DEBUG)



@ray.remote(num_cpus = 4)
def parallell_runner(parameters, graph,name):
    timestamp = time.time()
    date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    print(f"Processing task {parameters}, {name} at time {date_time}")
    qaoa = QAOArunner(graph, simulation=True, param_initialization=parameters[0], optimizer =  parameters[1], 
    qaoa_variant=parameters[2], warm_start=parameters[3], errors = parameters[4],depth = parameters[5])
    qaoa.build_circuit()
    qaoa.run()
    solver = Solver(graph)
    bitstring, value = solver.solve()
    end_time  = time.time()
    date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f"Solved task {parameters}, {name} at time {date_time}. It took {end_time-timestamp} seconds.")
    return {'param_initialization': parameters[0], 'optimizer': parameters[1],'qaoa_variant': parameters[2], 'warm_start' : parameters[3], 
    'errors':parameters[4], 'depth' : parameters[5], 'graph_size': len(graph.nodes()), 'graph_name' : name,
        'time_elapsed': qaoa.time_elapsed, 'quantum_func_evals': qaoa.fev, 'obj_func_evolution': qaoa.objective_func_vals,
        'quantum_solution':qaoa.solution, 'quantum_obj_value' : qaoa.objective_value, 
        'classic_solution' : bitstring, 'classic_value': value , 'final_params': qaoa.final_params, 'percent_measure_optimal': qaoa.get_prob_most_likely_solution}


if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')
ray.init(log_to_driver=True)

print(settings)
data = []

name= 'F?~vw'

graphs= [problem.get_paper_graphs()]
print(graphs)
names = ['DBk', 'DK{', 'D]{']

graphs = list(itertools.chain.from_iterable(graphs))
names = list(itertools.chain.from_iterable(names))
print('Amount of graphs: ', len(graphs))

combos = [settings, graphs]
print('combos', len(combos))

all_combos = list(itertools.product(*combos))


all_combos = [combo + (names[graphs.index(combo[1])],) for combo in all_combos]
all_combos_dict = [{"parameters": combo[0], "graph": combo[1], "name": combo[2]} for combo in all_combos]


n_times = 10
all_combos *= n_times

#TODO: make dictionary
print('len all_combos',len(all_combos))
print(f'performing all {n_times} times')


print('Settings:', settings)

parameter_set = []    


for i in range(len(settings[0])):
    parameter_set += set([settings[k][i] for k in range(len(settings))])


parameter_string = [str(x) + "_" for x in parameter_set]
parameter_string = "".join(parameter_string)
parameter_string = parameter_string[0:-1]



futures = [parallell_runner.remote(parameters, graph, name) for parameters, graph, name in all_combos]

result_ids, unfinished = ray.wait(futures, timeout = 60*60*16, num_returns = len(all_combos))
for task in unfinished:
    ray.cancel(task)

underway_df = pd.DataFrame(ray.get(result_ids))
underway_df.to_csv(f'results_underway.csv', mode='a', header=False)
data.extend(ray.get(result_ids))
print(f'Done with Parameters: {settings} at time: {time.time()}')


df = pd.DataFrame(data)
df.to_csv(f'results_singlegraph_{parameter_string}.csv')

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = f'Solstorm run -singlegraph -  {parameter_string}'
attachment = f'results_singlegraph_{parameter_string}.csv'

yag.send( subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()

