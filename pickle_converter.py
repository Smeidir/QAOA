import pandas as pd


df_subset = pd.read_csv("results/results_papergraph_depth_{2, 4, 6, 8, 10}.csv", usecols=["obj_func_evolution"])
print(df_subset.memory_usage(deep=True))