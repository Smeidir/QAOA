import pandas as pd

df = pd.read_csv('results/results_papergraph_depth_{2, 4, 6, 8, 10}.csv')
df.to_pickle('results/results_papergraph_depth_{2, 4, 6, 8, 10}.pkl')
print(f"Pickle file size: {df.memory_usage(deep=True).sum()} bytes")