# %% Init
from utils import dataLoader, save_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import silhouette_score
sns.set_theme(style='whitegrid')


# %% Load data

df_bot = dataLoader('bot')
# %% Parse time
df_bot['time'] = pd.to_datetime(df_bot['time'])
df_bot['time'] = df_bot['time'].dt.tz_convert('Asia/Shanghai')
df_bot['date'] = df_bot['time'].dt.date
df_bot['hour'] = df_bot['time'].dt.hour


'''# %% Determine the time interval to pivot with
df_bot['functionType'].unique()
# df_bot.to_csv('socketAnalysis.csv', index=False)
df_bot['diff'] = df_bot['time'].diff()
df_bot['diff'].value_counts().head(10)
df_bot.set_index('time').resample('59S').count()['value'].plot(title='Readings per 59 seconds')
'''

# %% Pivot data
df_pivot = df_bot.copy()
df_pivot = df_pivot.set_index('time')
df_pivot = (
    df_pivot.groupby('functionType')['value']
    .resample('1T')
    .mean()
    .unstack('functionType')
    .dropna(how='all')
)

# %% Correlation
corr_mat = df_pivot.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of functionType')
plt.savefig(save_path('correlation_matrix.png', device='bot'))
plt.show()

# %% Distribution of functionType
for col in df_bot['functionType'].unique():
    sns.histplot(df_bot[df_bot['functionType'] == col]['value'], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(save_path(f'distribution_{col}.png', device='bot'))
    plt.show()
# %%
df_bot['time'] = pd.to_datetime(df_bot['time'])
df_bot['date'] = df_bot['time'].dt.date
df_bot.groupby('date').size().plot(kind='bar', figsize=(12, 6), title='Number of Readings per Day')
# %% Time series analysis
axes = df_pivot.plot(subplots=True, figsize=(14, 2.5 * len(df_pivot.columns)), 
                     title='functionType Over Time (Subplots)', legend=False)

# Label each subplot
for ax, col in zip(axes, df_pivot.columns):
    ax.set_ylabel(col)
    ax.set_xlabel('Time')
    ax.set_title(col, loc='right')

plt.tight_layout()
plt.savefig(save_path('functionType_over_time.png', device='bot'))
plt.show()
# %% Missing values
df_pivot.isna().sum().sort_values(ascending=False).plot(kind='bar', figsize=(12, 6), title='Missing Values in functionType')
# %% PCA
pca = PCA(n_components=2)
X_scaled = StandardScaler().fit_transform(df_pivot.dropna())
X_pca = pca.fit_transform(X_scaled)


#Silhouette analysis
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=1103)
    labels = kmeans.fit_predict(X_pca)
    s_i = silhouette_score(X_pca, labels)
    silhouette_scores.append(s_i)


plt.figure(figsize=(10, 6))
sns.lineplot(x=K_range, y=silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(K_range)
plt.grid()
plt.savefig(save_path('silhouette_analysis.png', device='bot'))
plt.show()

'''
#Compare K-means clustering with different 2 and 3 clusters
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
def kM(k, ax):
    kmeans = KMeans(n_clusters=k, random_state=1103)
    labels = kmeans.fit_predict(X_pca)
    df_pca[f'cluster_{k}'] = labels
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=f'cluster_{k}', palette='viridis', ax=ax)
    ax.set_title(f'K-means Clustering with {k} Clusters')
    ax.legend(title='Cluster', loc='best')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
kM(2, axes[0])
kM(3, axes[1])
plt.tight_layout()
plt.savefig(save_path('kmeans_comparison.png', device='bot'))
plt.show()
'''

# %% k means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=1103)
kmeans.fit(X_pca)
X_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_plot['cluster'] = kmeans.labels_
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_plot, x='PC1', y='PC2', hue='cluster', palette='viridis')
plt.title('K-means Clustering of functionType')
plt.savefig(save_path('kmeans_clustering.png', device='bot'))
plt.show()

loadings = pd.DataFrame(pca.components_.T,
                        index = df_pivot.columns,
                        columns = ['PC1', 'PC2'])
print(loadings.sort_values(by='PC1', ascending=False))

# %% Disconnection detection
'''
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df_pivot.index)
outliers = df_pca[df_pca['PC1'] < -3.5]
df_outliers = df_pivot.loc[outliers.index]

for col in df_bot['functionType'].unique():
    data_all = df_bot[df_bot['functionType'] == col]['value']
    outlier_times = df_outliers.index
    outlier_values = df_bot[(df_bot['functionType'] == col) & (df_bot.index.isin(outlier_times))]['value']

    sns.histplot(data_all, bins=30, kde=True, color='blue', label='All Data')
    if not outlier_values.empty:
        sns.histplot(outlier_values, bins=30, kde=True, color='red', label='Outliers')
    plt.title(f'Distribution of {col} with Outliers Highlighted')
    plt.legend()
    plt.savefig(save_path(f'outliers_{col}.png', device='bot'))
    plt.show()

df_pivot['disconnected'] = df_pivot.index.isin(outlier_times)
df_pivot['state_extended'] = np.where(df_pivot['disconnected'], 'disconnected', df_pivot['state'])
df_pivot['state_extended'].value_counts().plot.pie(autopct='%1.1f%%', title='Extended State Distribution')
plt.ylabel('')
plt.tight_layout()
plt.savefig(save_path('state_extended_distribution.png', device='bot'))
plt.show()
disconnected_idx = df_pivot['state_extended'] == 'disconnected'
df_pivot.loc[disconnected_idx, ['Voltage', 'Current', 'Power']].describe()'''
# %% Assign states
df_pivot['cluster'] = kmeans.labels_
label_map = {0: 'charging', 1: 'activity'}
df_pivot['state'] = df_pivot['cluster'].map(label_map)
plt.figure(figsize=(6, 6))
df_pivot['state'].value_counts().plot.pie(
    autopct='%1.1f%%',
    title='Distribution of Time in States',
    ylabel=''
)
plt.tight_layout()
plt.savefig(save_path('state_distribution.png', device='bot'))
plt.show()

palette = {'charging': '#4C72B0', 'activity': '#DD8452'} 
state_palette = {
    'charging': '#4C72B0',
    'activity': '#DD8452'
}
# %% Visualize state distribution by hour and day
df_clean = df_pivot.dropna().copy()
df_clean["cluster"] = kmeans.labels_
df_clean["state"] = df_clean["cluster"].map(label_map)

df_clean["hour"] = df_clean.index.hour
sns.countplot(data=df_clean, x="hour", hue="state", palette=palette)
plt.title("State Distribution by Hour of Day")
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(save_path('state_distribution_by_hour.png', device='bot'))
plt.show()

df_clean["weekday"] = df_clean.index.day_name()
sns.countplot(data=df_clean, x="weekday", hue="state",
              order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
              palette=palette)
plt.title("State Distribution by Day of Week")
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(save_path('state_distribution_by_weekday.png', device='bot'))

df_clean["date"] = df_clean.index.date
state_trend = df_clean.groupby(["date", "state"]).size().unstack().fillna(0)

state_trend.plot(kind="area", stacked=True, figsize=(12, 6))
plt.title("Time Spent in Each State Over Days")
plt.savefig(save_path('state_trend_over_days.png', device='bot'))
plt.show()

# %% time spent in each state
df_clean = df_clean.sort_index()
df_clean['time'] = df_clean.index
df_clean['state_change'] = (df_clean['state'] != df_clean['state'].shift()).cumsum()
durations = df_clean.groupby(['state', 'state_change']).agg(
    start = ('time', 'first'),
    end = ('time', 'last'),
    duration = ('time', lambda x: (x.max() - x.min()) + pd.Timedelta(seconds=59))
    ).reset_index()
durations.groupby('state')['duration'].describe()
durations['duration_min'] = durations['duration'].dt.total_seconds() / 60
plt.figure(figsize=(10, 6))
sns.boxplot(data=durations, x='state', y='duration_min', palette=palette)
plt.title('Duration in Each State')
plt.savefig(save_path('duration_per_state.png', device='bot'))
plt.show()

# %% duration per state
plt.figure(figsize=(10, 6))
sns.histplot(data=durations, x='duration_min', hue='state', bins = 60, element='step', common_norm=False, palette=palette)
plt.title('Distribution of Duration in Each State')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.xlim(0, durations['duration_min'].quantile(0.99))  # Limit x-axis to 99th percentile
plt.savefig(save_path('duration_distribution_per_state.png', device='bot'))
plt.show()

# %% duration over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=durations, x='start', y='duration_min', hue='state', marker='o', palette=palette)
plt.title('Duration in Each State Over Time')
plt.xlabel('Time')
plt.ylabel('Duration (minutes)')
plt.ylim(0, durations['duration_min'].quantile(0.99))  # Limit y-axis to 99th percentile
plt.savefig(save_path('duration_over_time.png', device='bot'))
plt.show()

# %% Markov chain analysis
df_sorted = df_clean.reset_index(drop=True).sort_values('time')
state_sequence = df_sorted['state'].tolist()
transitions = list(zip(state_sequence[:-1], state_sequence[1:]))
transition_counts = Counter(transitions)
states = sorted(set(state_sequence))
matrix = pd.DataFrame(0, index=states, columns=states).fillna(0)
for (from_state, to_state), count in transition_counts.items():
    matrix.loc[from_state, to_state] += count
transition_matrix = matrix.div(matrix.sum(axis=1), axis=0)
print(transition_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap='Blues', square=True)
plt.title('Transition Matrix of States')
plt.savefig(save_path('transition_matrix.png', device='bot'))
plt.show()
# %% markov chain stable state
P = transition_matrix.values
A = P.T - np.eye(P.shape[0])
A = np.vstack([A, np.ones(P.shape[0])])
b = np.append(np.zeros(P.shape[0]), 1)
steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
steady_state_series = pd.Series(steady_state, index=transition_matrix.index)
print("Steady State Distribution:")
print(steady_state_series)
# %% daily usage profile by day of the week
df_clean['day_of_week'] = df_clean.index.day_name()
usage_by_day = df_clean.groupby(['day_of_week', 'state']).size().unstack(fill_value=0)
usage_by_day = usage_by_day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
usage_by_day_norm = usage_by_day.div(usage_by_day.sum(axis=1), axis=0)
plt.figure(figsize=(12, 6))
usage_by_day_norm.plot(
    kind='bar',
    stacked=True,
    color=[state_palette[state] for state in usage_by_day_norm.columns]
)
plt.title('Daily Usage Profile by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Proportion of Time in Each State')
plt.xticks(rotation=45)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(save_path('daily_usage_profile_by_day_of_week.png', device='bot'))
plt.show()

# %% usage by time of day
def ToD(t):
    hour = t.hour
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'
    
if 'time_of_day' in durations.columns:
    durations = durations.drop(columns=['time_of_day'])
durations['time_of_day'] = durations['start'].dt.time.map(lambda x: ToD(pd.Timestamp.combine(pd.Timestamp.today(), x)))


grouped = durations.groupby(['time_of_day', 'state'])['duration_min'].sum()
proportions = grouped.groupby('time_of_day', group_keys=False).apply(lambda x: x / x.sum())
time_of_day_profile = proportions.rename('proportion').reset_index()

time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
time_of_day_profile['time_of_day'] = pd.Categorical(
    time_of_day_profile['time_of_day'], 
    categories=time_order, 
    ordered=True
)
time_of_day_profile = time_of_day_profile.sort_values('time_of_day')

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=time_of_day_profile, x='time_of_day', y='proportion', hue='state')
plt.title('Usage Profile by Time of Day')
plt.ylabel('Proportion of Time in Each State')
plt.xlabel('Time of Day')
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(save_path('usage_profile_by_time_of_day.png', device='bot'))
plt.show()
# %% 
def twohr_bin(t):
    hour = t.hour
    return f"{(hour // 2) * 2:02d}:00 - {(hour // 2) * 2 + 2:02d}:00"

durations['2hr_bin'] = durations['start'].dt.time.map(
    lambda x: twohr_bin(pd.Timestamp.combine(pd.Timestamp.today(), x))
)

proportions_2hr = (
    durations
    .groupby(['2hr_bin', 'state'])['duration_min']
    .sum()
    .groupby(level=0)
    .transform(lambda x: x / x.sum())
    .reset_index(name='proportion')
)

plt.figure(figsize=(14, 6))
sns.barplot(data=proportions_2hr, x='2hr_bin', y='proportion', hue='state', palette=palette)
plt.title('Usage Profile by 2-Hour Interval')
plt.ylabel('Proportion of Time in Each State')
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Time Interval')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_path('usage_profile_by_2hr_interval.png', device='bot'))
plt.show()

# %% filtering out extreme values and re-running PCA and KMeans
'''df_filtered = df_pivot[~df_pivot['disconnected']].dropna()
df_full = df_pivot.dropna() 
def run_pca_kmeans(df, n_clusters=2):
    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Standardize
    X_scaled = StandardScaler().fit_transform(df_numeric)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=1103)
    labels = kmeans.fit_predict(X_pca)

    # Silhouette score
    silhouette = silhouette_score(X_pca, labels)

    # Output
    df_result = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)
    df_result['cluster'] = labels
    return df_result, kmeans, pca, silhouette

df_full_pca, kmeans_full, pca_full, sil_full = run_pca_kmeans(df_full)

# Filtered dataset (without disconnected outliers)
df_filtered_pca, kmeans_filtered, pca_filtered, sil_filtered = run_pca_kmeans(df_filtered)

print(f"Silhouette Score (Full): {sil_full:.3f}")
print(f"Silhouette Score (Filtered): {sil_filtered:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=df_full_pca, x='PC1', y='PC2', hue='cluster', palette='viridis', ax=axes[0])
axes[0].set_title('KMeans on Full Data (with outliers)')

sns.scatterplot(data=df_filtered_pca, x='PC1', y='PC2', hue='cluster', palette='viridis', ax=axes[1])
axes[1].set_title('KMeans on Filtered Data (no outliers)')

plt.tight_layout()
plt.savefig(save_path('kmeans_comparison_filtered_vs_full.png', device='bot'))
plt.show()'''
# %%
