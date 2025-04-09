import pandas as pd


df = pd.read_csv("results/results_papergraph_depth_{2, 4, 6, 8, 10}.csv")
print(f"Memory usage of DataFrame: {df.memory_usage(deep=True)} bytes")
df.to_csv("results/results_papergraph_depth_{2, 4, 6, 8, 10}_size_fixed.csv")