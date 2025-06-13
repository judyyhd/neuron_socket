# %% Init
from utils import dataLoader, save_path
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
df_pc = dataLoader('pc')
# %% Parse time
df_pc['time'] = pd.to_datetime(df_pc['time'])
df_pc['time'] = df_pc['time'].dt.tz_convert('Asia/Shanghai')
df_pc['date'] = df_pc['time'].dt.date
df_pc['hour'] = df_pc['time'].dt.hour
# %% pivot
df_pivot = df_pc.copy()
df_pivot = df_pivot.set_index('time')
df_pivot = (
    df_pivot.groupby('functionType')['value']
    .resample('1T')
    .mean()
    .unstack('functionType')
    .dropna(how='all')
)

# %% Correlation Analysis
corr_mat = df_pivot.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of functionType')
plt.savefig(save_path('correlation_matrix.png', device='pc'))
plt.show()
# %% Distribution Analysis
for col in df_pc['functionType'].unique():
    sns.histplot(df_pc[df_pc['functionType'] == col]['value'], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(save_path(f'distribution_{col}.png', device='pc'))
    plt.show()
# %% Time Series Subplot
axes = df_pivot.plot(subplots=True, figsize=(14, 2.5 * len(df_pivot.columns)), 
                     title='functionType Over Time (Subplots)', legend=False)


for ax, col in zip(axes, df_pivot.columns):
    ax.set_ylabel(col)
    ax.set_xlabel('Time')
    ax.set_title(col, loc='right')

plt.tight_layout()
plt.savefig(save_path('functionType_subplots.png', device='pc'))
plt.show()
# %% Missing Data
df_pivot.isna().sum().sort_values(ascending=False).plot(kind='bar', figsize=(12, 6), title='Missing Values in functionType')

# %% PCA
pca = PCA(n_components=2)
X_scaled = StandardScaler().fit_transform(df_pivot.dropna())
X_pca = pca.fit_transform(X_scaled)


# %% Silhouette analysis
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
plt.savefig(save_path('silhouette_analysis.png', device='pc'))
plt.show()

kmeans = KMeans(n_clusters=2, random_state=1103)
kmeans.fit(X_pca)
X_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_plot['cluster'] = kmeans.labels_
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_plot, x='PC1', y='PC2', hue='cluster', palette='viridis')
plt.title('K-means Clustering of functionType')
plt.savefig(save_path('kmeans_clustering.png', device='pc'))
plt.show()

loadings = pd.DataFrame(pca.components_.T,
                        index = df_pivot.columns,
                        columns = ['PC1', 'PC2'])
print(loadings.sort_values(by='PC1', ascending=False))

# %% DBSCAN

# nearest neighbors for dbscan
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 9])  # 10th nearest neighbor
plt.plot(distances)
plt.title('k-distance Graph')
plt.ylabel("10th Nearest Distance")
plt.xlabel("Points sorted by distance")
plt.savefig(save_path('k_distance_graph.png', device='pc'))
plt.show()

# dbscan clustering
dbscan = DBSCAN(eps=2.5, min_samples=5)
labels = dbscan.fit_predict(X_pca)
pd.Series(labels).value_counts().plot(kind='bar', figsize=(10, 6), title='DBSCAN Cluster Sizes')
X_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
X_plot['cluster'] = labels
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_plot, x='PC1', y='PC2', hue='cluster', palette='viridis')
plt.title('DBSCAN Clustering')
plt.savefig(save_path('dbscan_clustering.png', device='pc'))
plt.show()
# %% SVM
svm = SVC(kernel='linear')
svm.fit(X_pca, kmeans.labels_)

h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1', s=40, edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Linear SVM Decision Boundary on PCA Space')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
# %% Manual clustering

m, b = -1.5, 0
x_vals = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
y_vals = m * x_vals + b
sns.scatterplot(data=X_plot, x='PC1', y='PC2', palette='viridis')
plt.plot(x_vals, y_vals, '--k', label='Manual Boundary: y = -1.5x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Manual Clustering with Custom Linear Boundary')
plt.legend()
plt.ylim(-5,5)
plt.grid(True)
plt.savefig(save_path('manual_clustering_boundary.png', device='pc'))
plt.show()

manual_clusters = np.where(X_pca[:, 1] < m * X_pca[:, 0] + b, 0, 1)
df_pivot['manual_cluster'] = manual_clusters


boundary_v = np.array([1, -1.5])
boundary_v = boundary_v / np.linalg.norm(boundary_v)
projection = X_pca @ boundary_v
sns.histplot(projection, bins=100, kde=True)
plt.title('Projection of Data Points onto Custom Boundary Vector')
plt.xlabel('Projection Value')
plt.ylabel('Frequency')
plt.axvline(0, color='black', linestyle='--', label='Boundary at 0')
plt.legend()
plt.savefig(save_path('projection_histogram.png', device='pc'))
plt.show()

threshold = -0.45

manual_clusters_shifted = np.where(projection < threshold, 0, 1)
plt.figure(figsize=(7, 5))
sns.histplot(projection, bins=100, kde=True, color='steelblue', alpha=0.6)
plt.axvline(x=threshold, color='black', linestyle='--', label=f'Boundary at {threshold}')
plt.title('Projection of Data Points onto Custom Boundary Vector')
plt.xlabel('Projection Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(save_path('projection_histogram_shifted.png', device='pc'))
plt.show()


m = -1.5
threshold = -0.45


boundary_v = np.array([1, m])
boundary_v = boundary_v / np.linalg.norm(boundary_v)
b = threshold / boundary_v[1]  # this is now the shifted y-intercept


manual_clusters_shifted = np.where(X_pca[:, 1] < m * X_pca[:, 0] + b, 0, 1)
df_pivot['manual_cluster'] = manual_clusters_shifted


x_vals = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
y_vals = m * x_vals + b

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=df_pivot['manual_cluster'], 
    palette='Set1', 
    s=20
)
plt.plot(x_vals, y_vals, '--k', label=f'Manual Boundary: y = {m}x + {b:.4f}')
plt.title("Manual Clusters in PCA Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(save_path('manual_clusters_pca.png', device='pc'))
plt.show()

# %% interpret the manual clusters
cluster_summary = df_pivot.groupby('manual_cluster').mean().T
display(cluster_summary)


label_map = {0: 'idle', 1: 'operating'}
df_pivot['state'] = df_pivot['manual_cluster'].map(label_map)
plt.figure(figsize=(6, 6))
df_pivot['state'].value_counts().plot.pie(
    autopct='%1.1f%%',
    title='Distribution of Time in States',
    ylabel=''
)
plt.tight_layout()
plt.savefig(save_path('state_distribution.png', device='pc'))
plt.show()
palette = {'charging': '#4C72B0', 'activity': '#DD8452'} 
state_palette = {
    'charging': '#4C72B0',
    'activity': '#DD8452'
}
# %%
df_clean = df_pivot.dropna().copy()

df_clean["hour"] = df_clean.index.hour
sns.countplot(data=df_clean, x="hour", hue="state")
plt.title("State Distribution by Hour of Day")
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(save_path('state_distribution_by_hour.png', device='pc'))
plt.show()
# %%
df_clean["weekday"] = df_clean.index.day_name()
sns.countplot(data=df_clean, x="weekday", hue="state",
              order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("State Distribution by Day of Week")
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(save_path('state_distribution_by_weekday.png', device='pc'))
plt.show()

# %%
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
sns.boxplot(data=durations, x='state', y='duration_min')
plt.title('Duration in Each State')
plt.savefig(save_path('duration_per_state.png', device='pc'))
plt.show()

# %% duration per state
plt.figure(figsize=(10, 6))
sns.histplot(data=durations, x='duration_min', hue='state', bins = 60, element='step', common_norm=False)
plt.title('Distribution of Duration in Each State')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.xlim(0, durations['duration_min'].quantile(0.99))  # Limit x-axis to 99th percentile
plt.show()

# %% duration over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=durations, x='start', y='duration_min', hue='state', marker='o')
plt.title('Duration in Each State Over Time')
plt.xlabel('Time')
plt.ylabel('Duration (minutes)')
plt.ylim(0, durations['duration_min'].quantile(0.99))  # Limit y-axis to 99th percentile

plt.show()


# %% daily usage profile by day of the week
df_clean['day_of_week'] = df_clean.index.day_name()
usage_by_day = df_clean.groupby(['day_of_week', 'state']).size().unstack(fill_value=0)
usage_by_day = usage_by_day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
usage_by_day_norm = usage_by_day.div(usage_by_day.sum(axis=1), axis=0)
plt.figure(figsize=(12, 6))
usage_by_day_norm.plot(kind='bar', stacked=True)
plt.title('Daily Usage Profile by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Proportion of Time in Each State')
plt.xticks(rotation=45)
plt.legend(title='State')
plt.tight_layout()
plt.savefig(save_path('daily_usage_profile.png', device='pc'))
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

plt.figure(figsize=(8, 6))
sns.barplot(data=time_of_day_profile, x='time_of_day', y='proportion', hue='state')
plt.title('Usage Profile by Time of Day')
plt.ylabel('Proportion of Time in Each State')
plt.xlabel('Time of Day')
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(save_path('usage_profile_by_time_of_day.png', device='pc'))
plt.tight_layout()
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
sns.barplot(data=proportions_2hr, x='2hr_bin', y='proportion', hue='state')
plt.title('Usage Profile by 2-Hour Interval')
plt.ylabel('Proportion of Time in Each State')
plt.xlabel('Time Interval')
plt.xticks(rotation=45)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(save_path('usage_profile_by_2hr_interval.png', device='pc'))
plt.show()

# %%
