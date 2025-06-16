# %% Init
from utils import dataLoader, save_path, read_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
sns.set_theme(style='whitegrid')

# %% Load Data
'''
df_pc = dataLoader('pc')
df_bot = dataLoader('bot')

#save to csv
df_pc.to_csv(save_path('pc.csv'), index=False)
df_bot.to_csv(save_path('bot.csv'), index=False)
'''
# read from csv
df_pc = pd.read_csv(read_path('pc'))
df_bot = pd.read_csv(read_path('bot'))
# %% Parse time data
df_bot['time'] = pd.to_datetime(df_bot['time'])
df_bot['time'] = df_bot['time'].dt.tz_convert('Asia/Shanghai')
df_pc['time'] = pd.to_datetime(df_pc['time'])
df_pc['time'] = df_pc['time'].dt.tz_convert('Asia/Shanghai')
 # %% pivot data
df_pivot_pc= df_pc.copy()
df_pivot_bot = df_bot.copy()


df_pivot_pc = df_pivot_pc.set_index('time')
df_pivot_pc = (
    df_pivot_pc.groupby('functionType')['value']
    .resample('1T')
    .mean()
    .unstack('functionType')
    .dropna(how='all')
)

df_pivot_bot = df_pivot_bot.set_index('time')
df_pivot_bot = (
    df_pivot_bot.groupby('functionType')['value']
    .resample('1T')
    .mean()
    .unstack('functionType')
    .dropna(how='all')
)
# %% drop relay status (binary)
df_pivot_pc = df_pivot_pc.drop(columns=['RelayStatus', 'Leakage'])
df_pivot_bot = df_pivot_bot.drop(columns=['RelayStatus', 'Leakage'])
# %% Correlation
df_pivot_pc, df_pivot_bot = df_pivot_pc.align(df_pivot_bot, join='inner', axis=1)
correlations = {
    col: df_pivot_pc[col].corr(df_pivot_bot[col])
    for col in df_pivot_pc.columns if col in df_pivot_bot.columns
}
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
print("Correlation between PC and Bot data:")
print(correlation_df)

# %%
