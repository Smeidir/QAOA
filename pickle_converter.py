import pandas as pd


df_subset = pd.read_csv("path/to/your/file.csv", usecols=["obj_func_evolution"])
print(df_subset.memory_usage(deep=True))