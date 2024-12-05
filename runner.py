from MaxCutProblem import MaxCutProblem
import itertools
import time
import params
from QAOA import QAOArunner
from datetime import date
import pandas as pd
from solver import Solver
import ray

problem = MaxCutProblem()


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
        'quantum_solution':qaoa.solution, 'quantum_obj_value' : qaoa.evaluate_sample(), 
        'classic_solution' : bitstring, 'classic_value': value }
if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')

ray.init()

iterables = [params.supported_qaoa_variants, params.supported_param_inits, [True]] #TODO: change to only gaussian inits
settings = list(itertools.product(*iterables))
print(settings)

#settings = [('multiangle', 'gaussian', True),('vanilla', 'uniform', True), ('vanilla', 'gaussian', True), ('multiangle', 'uniform', True),('multiangle', 'gaussian', True)] 


data = []
# Reset results_underway.csv file
with open('results_underway.csv', 'w') as f:
    f.write('index,param_initialization,qaoa_variant,warm_start,depth,graph_size,graph_name,time_elapsed,quantum_func_evals,quantum_solution,quantum_obj_value,classic_solution,classic_value,obj_func_evolution,\n')

for parameters in settings:
    print('Parameters:', parameters)
    graphs = []
    names = []
    for i in range(5,10):
        graphs_i, names_i = problem.get_test_graphs(i)
        graphs.append(graphs_i[0]) #TODO: check that this works for very small values
        names.append(names_i[0])
    graphs = list(itertools.chain.from_iterable(graphs))
    names = list(itertools.chain.from_iterable(names))
    futures = [parallell_runner.remote(parameters, graph, name) for graph, name in zip(graphs, names)]
    results = ray.get(futures)
    underway_df = pd.DataFrame(ray.get(results))
    underway_df.to_csv(f'results_underway.csv', mode='a', header=False)
    data.extend(results)
    print(f'Done with Parameters: {parameters} at time: {time.time()}')

df = pd.DataFrame(data)
df.to_csv(f'results_.csv')