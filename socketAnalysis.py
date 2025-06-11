# %% Init
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# %% Load data
file_list = glob.glob('*.xlsx')
dfs = [pd.read_excel(file) for file in file_list]
df_concat = pd.concat(dfs, ignore_index=True)
# %% Parse time
df_concat['time'] = pd.to_datetime(df_concat['time'])
df_concat['date'] = df_concat['time'].dt.date
df_concat['hour'] = df_concat['time'].dt.hour
# %%