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

from MaxCutProblem import MaxCutProblem
problem = MaxCutProblem()

with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()

@ray.remote
def parallell_runner(parameters, graph,name):
    qaoa = QAOArunner(graph, simulation=True, param_initialization=parameters[1],qaoa_variant=parameters[0], warm_start=parameters[2])
    qaoa.build_circuit()
    qaoa.run()
    solver = Solver(graph)
    bitstring, value = solver.solve()
    return {'param_initialization': parameters[1], 'qaoa_variant': parameters[0], 'warm_start' : parameters[2],
        'depth': params.depth, 'graph_size': len(graph.nodes()), 'graph_name' : name,
        'time_elapsed': qaoa.time_elapsed, 'quantum_func_evals': qaoa.fev, 'obj_func_evolution': qaoa.objective_func_vals,
        'quantum_solution':qaoa.solution, 'quantum_obj_value' : qaoa.objective_value, 
        'classic_solution' : bitstring, 'classic_value': value }


if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')
ray.init(num_cpus=15)

iterables = [['multiangle'], params.supported_param_inits, [True,False]] 
settings = list(itertools.product(*iterables))
print(settings)
print('Depth: ', params.depth)
data = []

for parameters in settings:
    print('Parameters:', parameters)
    graphs = []
    names = []
    for i in range(5,10):
        graphs_i, names_i = problem.get_test_graphs(i)
        graphs.append(graphs_i) #TODO: check that this works for very small values
        names.append(names_i)

    graphs = list(itertools.chain.from_iterable(graphs))
    names = list(itertools.chain.from_iterable(names))
    futures = [parallell_runner.remote(parameters, graph, name) for graph, name in zip(graphs, names)]

    result_ids, unfinished = ray.wait(futures, timeout = 60*60*4, num_returns = len(graphs))
    for task in unfinished:
        ray.cancel(task)
    underway_df = pd.DataFrame(ray.get(result_ids))
    underway_df.to_csv(f'results_underway.csv', mode='a', header=False)
    data.extend(results)
    print(f'Done with Parameters: {parameters} at time: {time.time()}')

    
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)

    yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
    recipient = "torbjorn.smed@gmail.com"
    subject = "Data from Python Script"
    body = "Here is the data you requested."
    attachment = "data.csv"
    
    yag.send( subject=subject, contents=body, attachments=attachment)
    print("Email sent successfully!")

df = pd.DataFrame(data)
df.to_csv(f'results_.csv')

