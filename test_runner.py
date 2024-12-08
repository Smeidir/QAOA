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
export RAY_IGNORE_UNHANDLED_ERRORS=1

from MaxCutProblem import MaxCutProblem
problem = MaxCutProblem()

with open("email_credentials.txt", "r") as f:
    email_password = f.read().strip()
import ast
import networkx as nx

with open("test_settings.txt", "r") as f:
    settings = ast.literal_eval(f.read().strip())


@ray.remote
def parallell_runner(parameters, graph,name):
    result = f"Parameters: {parameters}, Graph Name: {name}"
    return result

if ray.is_initialized():
    ray.shutdown()
    print('Shutting down old Ray instance.')
ray.init(num_cpus=48, _system_config={"worker_lease_timeout_milliseconds": 0})

print(settings)
print('Depth: ', params.depth)
data = []

graphs, names = [],[]


for i in range(9,4, -1):
    graphs_i, names_i = problem.get_test_graphs(i)
    graphs.append(graphs_i) #TODO: check that this works for very small values
    names.append(names_i)
print(len(graphs))
graphs = list(itertools.chain.from_iterable(graphs))
names = list(itertools.chain.from_iterable(names))

combos = [settings, graphs]
print('combos', len(combos))

all_combos = list(itertools.product(*combos))


all_combos = [combo + (names[graphs.index(combo[1])],) for combo in all_combos]
all_combos_dict = [{"parameters": combo[0], "graph": combo[1], "name": combo[2]} for combo in all_combos]
#TODO: make dictionary
print('len all_combos',len(all_combos))

print('Settings:', settings)

futures = [parallell_runner.remote(parameters, graph, name) for parameters, graph, name in all_combos]

result_ids, unfinished = ray.wait(futures, timeout = 60*60*12, num_returns = len(all_combos))
for task in unfinished:
    ray.cancel(task)
underway_df = pd.DataFrame(ray.get(result_ids))
underway_df.to_csv(f'results_underway.csv', mode='a', header=False)
data.extend(ray.get(result_ids))
print(f'Done with Parameters: {settings} at time: {time.time()}')


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
