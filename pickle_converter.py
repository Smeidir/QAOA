import pandas as pd


df = pd.read_csv("results/results_papergraph_depth_{2, 4, 6, 8, 10}.csv")
df.drop("obj_func_evolution", axis=1, inplace=True)
df.to_csv("results/results_papergraph_depth_{2, 4, 6, 8, 10}.csv")