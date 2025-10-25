import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


#
# Copied from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#
def run_silhouette_analysis(
    X: np.ndarray,
    K_list: list,
    random_state: int = 42,
):
    """
    Performs K-Means clustering across a range of n_clusters and visualizes
    the results using Silhouette Analysis plots.

    Parameters:
    - X (np.ndarray): The feature data matrix (samples x features).
    - range_n_clusters (list): A list of integers representing the number of
      clusters (k) to test.
    - random_state (int, optional): Seed for reproducibility of K-Means. Defaults to 10.
    """

    # Check if there is data to process
    if X.size == 0:
        print("Error: Input data X is empty.")
        return

    print(f"--- Running Silhouette Analysis for k in {K_list} ---")

    for n_clusters in K_list:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # --- 1st Subplot: Silhouette Plot Setup ---
        ax1.set_xlim([-0.1, 1])
        # Calculate space for the silhouette plots
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer
        # Note: Using 'n_init=10' is good practice to avoid warnings.
        try:
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = clusterer.fit_predict(X)
        except Exception as e:
            print(f"Error running KMeans for n_clusters={n_clusters}: {e}")
            continue

        # Calculate the average silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            round(silhouette_avg, 4),
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values: np.ndarray = silhouette_samples(X, cluster_labels)  # type: ignore

        # --- Plotting the Silhouettes ---
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate and sort silhouette scores for cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color: tuple = cm.nipy_spectral(float(i) / n_clusters)  # type: ignore
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples gap

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # Draw the vertical line for average silhouette score
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # --- 2nd Subplot: Actual Clusters Formed ---
        colors: np.ndarray = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)  # type: ignore
        ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

        # Labeling the cluster centers
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        # Overlay cluster numbers on the centers
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()


# 3. Run the analysis function
if __name__ == "__main__":
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )
    K_list: list[int] = [2, 3, 4, 5, 6]
    run_silhouette_analysis(X, K_list)
