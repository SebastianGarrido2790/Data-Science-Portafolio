import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

pd.options.display.float_format = lambda x: f"{x:20.2f}"

df = pd.read_csv("../../data/processed/online_retail_2009_2010_scaled.csv")

df = df.drop("Unnamed: 0", axis=1)

# Look at standardized data
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(df["Monetary"], df["Frequency"], df["Recency"])
ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")
ax.set_title("3D Scatter Plot of Customer Data")
plt.show()


# KMeans Clustering
def plot_elbow_silhouette(data, max_k=10):
    """
    Plots the Elbow Method and Silhouette Score to determine the optimal number of clusters for KMeans.

    Parameters:
    data (pd.DataFrame): Standardized RFM dataset.
    max_k (int): Maximum number of clusters to test.
    """
    wcss = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)  # Silhouette requires at least 2 clusters

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=1000)
        cluster_labels = kmeans.fit_predict(data)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Plot Elbow Method (WCSS)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(k_values, wcss, marker="o", linestyle="--", color="b", label="WCSS")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_title("Elbow Method and Silhouette Score")

    # Plot Silhouette Score on the same graph
    ax2 = ax1.twinx()
    ax2.plot(
        k_values,
        silhouette_scores,
        marker="s",
        linestyle="-",
        color="r",
        label="Silhouette Score",
    )
    ax2.set_ylabel("Silhouette Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.legend(loc="upper right")
    ax2.legend(loc="lower right")

    plt.savefig("../../reports/figures/elbow_silhouette(best_k).png", dpi=100)
    plt.show()


plot_elbow_silhouette(df)

# Train KMeans with optimal k
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=1000)
non_outliers_df = pd.read_csv(
    "../../data/processed/online_retail_2009_2010_without_outliers.csv",
    parse_dates=["LastInvoiceDate"],
)
cluster_labels = kmeans.fit_predict(df)

# Plot 3D Scatter Plot with Cluster Colors
non_outliers_df["Cluster"] = cluster_labels
cluster_colors = {
    0: "#1f77b4",  # Blue
    1: "#ff7f0e",  # Orange
    2: "#2ca02c",  # Green
    3: "#d62728",
}  # Red

colors = non_outliers_df["Cluster"].map(cluster_colors)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(
    non_outliers_df["Monetary"],
    non_outliers_df["Frequency"],
    non_outliers_df["Recency"],
    c=colors,  # Use mapped solid colors
    marker="o",
)

ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")
ax.set_title("3D Scatter Plot of Customer Data by Cluster")

plt.legend(handles=scatter.legend_elements()[0], labels=cluster_colors.keys())
plt.savefig("../../reports/figures/3d_scatter_plot.png", dpi=300)
plt.show()

# Save the dataset with cluster labels
non_outliers_df.to_csv(
    "../../data/processed/online_retail_2009_2010_with_clusters.csv", index=False
)
