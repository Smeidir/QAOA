import itertools
import time
from src.qaoa.core.QAOA import QAOArunner
import pandas as pd
import ray
import yagmail
import ast
import networkx as nx
from tqdm import tqdm
from src.qaoa.models.MaxCutProblem import MaxCutProblem


problem = MaxCutProblem()

with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()


local = False

if not local:
    with open("qaoa_settings.txt", "r") as f:
        settings = ast.literal_eval(f.read().strip())
if local: 
    settings = "[{'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'gaussian', 'depth': 10, 'warm_start': False}, {'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'gaussian', 'depth': 10, 'warm_start': True}, {'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'static', 'depth': 10, 'warm_start': False}, {'backend_mode': 'noisy_sampling', 'qaoa_variant': 'vanilla', 'param_initialization': 'static', 'depth': 10, 'warm_start': True}]"
    settings = ast.literal_eval(settings)
    print(' YOu are running without reading from qaoa_settings.txt - you should never see this message on solstorm!')

@ray.remote(num_cpus =4)
def parallell_runner(parameters, graph, name):
    qaoa = QAOArunner(graph, **parameters)
    qaoa.build_circuit()
    qaoa.run()
    return_dict = qaoa.to_dict()
    return_dict['graph_name'] = name
    return return_dict

if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')
ray.init(log_to_driver=True)


graphs= [problem.get_erdos_renyi_graphs_paper1()]
graphs.reverse() #- reverse if the largest graphs are the last!

graphs = list(itertools.chain.from_iterable(graphs)) #should be lists from before, no?

combos = [settings, graphs] #settings should be a list of dictionaries .


# Convert graphs to networkx graphs and generate graph6 strings, to have graph names. Is it better to save all edges? then weights arent lost
graph6_strings = []
for graph in graphs: #TODO: write graph6 decoder

    graph = nx.Graph(list(graph.edge_list())) 
    graph6_string = nx.to_graph6_bytes(graph).decode('utf-8').strip()
    graph6_strings.append(graph6_string)

print('Graph6 strings of the graphs tested: ', graph6_strings)
all_combos = list(itertools.product(*combos))

combos_with_name = []
for liste in all_combos:
    liste2 = liste +  (graph6_strings[graphs.index(liste[1])],) #tuples are immutable
    combos_with_name.append(liste2)
all_combos = combos_with_name

n_times = 50
all_combos *= n_times


print('Amount of runs',len(all_combos))
print(f'Wherein all instances are performed {n_times} times')


print('Instances tested: ', settings)
data = []
parameter_set = []    

# Find keys with different values across the dictionaries in settings
keys_with_differences = []

keys = settings[0].keys()
parameter_dict = {}
for key in keys:
    values = set([d[key] for d in settings])
    parameter_dict[key] = values


parameter_string = str(parameter_dict)

print('Parameter set', keys)
print('Parameter string, used for naming .csv files: ', parameter_string)

futures = [parallell_runner.remote(parameters, graph, name) for parameters, graph,name in all_combos]

result_ids = []
unfinished = futures

with tqdm(total=len(futures), desc="Processing tasks") as pbar:
    start_time = time.time()
    max_runtime = 5 * 24 * 60 * 60  # 5 days in seconds

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
print(f'Done with Parameters: {settings} at time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')


df = pd.DataFrame(data)
df.to_csv(f'results/results_papergraph_{parameter_string}.csv')

yag = yagmail.SMTP("torbjorn.solstorm@gmail.com", email_password)
recipient = "torbjorn.smed@gmail.com"
subject = "Data from Python Script"
body = f'Solstorm run -papergraph -  {parameter_string}'
attachment = f'results/results_papergraph_{parameter_string}.csv'

yag.send(subject=subject, contents=body, attachments=attachment)
print("Email sent successfully!")
yag.close()

