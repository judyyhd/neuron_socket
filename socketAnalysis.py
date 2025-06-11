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
# %%
