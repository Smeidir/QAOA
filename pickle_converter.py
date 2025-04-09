import pandas as pd


df = pd.read_csv("results_papergraph_depth_{8,}.csv")
df = df.drop(columns=["obj_func_evals"], errors="ignore")
df.to_csv("results_papergraph_depth_{2, 4, 6}.csv")

df = pd.read_csv("results_papergraph_depth_{8, 10}.csv")
df = df.drop(columns=["obj_func_evals"], errors="ignore")
df.to_csv("results_papergraph_depth_{8, 10}.csv")