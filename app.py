import logging
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


st.set_page_config(layout="wide", page_title="K-Means Clustering Analysis App")


def save_slider_value_to_session():
    st.session_state.recommended_k_value = st.session_state.k_slider_widget


def get_data_summary(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Calculates key cluster statistics (centroids) and counts."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 1. Centroids (Mean values)
    centroids = df.groupby(cluster_col)[numeric_cols].mean().reset_index()
    centroids.rename(columns={col: f"Avg_{col}" for col in numeric_cols}, inplace=True)

    # 2. Cluster size
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = [cluster_col, "Count"]

    # 3. Merge summary
    summary = pd.merge(counts, centroids, on=cluster_col)

    # Calculate percentage and format
    summary["Count (%)"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)
    summary.set_index(cluster_col, inplace=True)

    return summary.sort_values(by="Count", ascending=False)


def profile_cluster(
    cluster_label: str,
    summary: pd.DataFrame,
    sort_col: str,
    original_cols: list[str],
) -> str:
    """Performs natural language profiling based on cluster centroids."""

    # Get the cluster row by label
    cluster_row: pd.Series = summary.loc[cluster_label]  # type: ignore

    # Start profile text
    profile = [f"**Cluster {cluster_label}** (Samples: {int(cluster_row['Count']):,} | {cluster_row['Count (%)']}%):"]

    # Analyze the sorting criterion field
    avg_sort_val = cluster_row[f"Avg_{sort_col}"]
    all_avg_sort_vals = summary[f"Avg_{sort_col}"]

    if avg_sort_val == all_avg_sort_vals.max():
        profile.append(f"- **{sort_col}** is **highest** in this group (Avg: ${avg_sort_val:,.2f}).")
    elif avg_sort_val == all_avg_sort_vals.min():
        profile.append(f"- **{sort_col}** is **lowest** in this group (Avg: ${avg_sort_val:,.2f}).")
    else:
        profile.append(f"- **{sort_col}** is **moderate** (Avg: ${avg_sort_val:,.2f}).")

    # Highlight significant features in other fields
    for col in original_cols:
        if col == sort_col:
            continue

        avg_val = cluster_row[f"Avg_{col}"]
        all_avg_vals = summary[f"Avg_{col}"]

        # Compare to all cluster averages (relative position using quartiles)
        if avg_val >= all_avg_vals.quantile(0.75):
            profile.append(f"- **{col}** level is **relatively very high** (Avg: {avg_val:,.2f}).")
        elif avg_val <= all_avg_vals.quantile(0.25):
            profile.append(f"- **{col}** level is **relatively very low** (Avg: {avg_val:,.2f}).")

    return "\n".join(profile)


def _prepare_data(df_uploaded: pd.DataFrame) -> tuple:
    """Prepares data for clustering and performs scaling.

    Args:
        df_uploaded: The original uploaded DataFrame.

    Returns:
        (X_scaled_df, numeric_cols, df, X_scaled) or None
    """
    st.subheader("3. Data Scaling (Preprocessing)")
    st.info("K-Means is distance-based; scaling is essential.")

    df: pd.DataFrame = df_uploaded.copy()
    numeric_cols: list[str] = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("Error: No numeric data fields available for clustering.")
        return None, None, None, None

    X: pd.DataFrame = df[numeric_cols]

    # Check data sufficiency
    if len(X) < 20:
        st.error("Data count is too small to perform clustering (min 20 samples needed).")
        return None, None, None, None

    # Execute scaling
    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)
    X_scaled_df: pd.DataFrame = pd.DataFrame(X_scaled, columns=numeric_cols)
    st.dataframe(X_scaled_df.head())

    return X_scaled_df, numeric_cols, df, X_scaled


def _find_optimal_k(X_scaled: np.ndarray, n_samples: int) -> tuple[int, int]:
    """Finds the optimal K using the Elbow method and generates the plot."""
    st.subheader("6. Find Optimal Cluster Count (K) - Elbow Method")

    K_MAX = min(10, n_samples // 20)
    sse: dict[int, float] = {}

    # Calculate SSE
    for k in range(1, K_MAX + 1):
        kmeans: KMeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse[k] = kmeans.inertia_

    # Generate Elbow Plot
    def create_elbow_plot(sse_data):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(list(sse_data.keys()), list(sse_data.values()), marker="o")
        ax.set_title("Elbow Method for Optimal K")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Sum of Squared Errors (SSE)")
        ax.grid(True)
        return fig

    fig = create_elbow_plot(sse)
    st.pyplot(fig)

    # Automatic K estimation (simple heuristic)
    if len(sse) >= 3:
        diff: pd.Series = pd.Series(sse).diff().abs().dropna()
        diff_of_diff: pd.Series = diff.diff().abs().dropna()
        idxmin: int | str = diff_of_diff.idxmin()
        optimal_k_auto = int(idxmin) + 1
    else:
        optimal_k_auto = 3

    return optimal_k_auto, K_MAX


def _get_user_k_selection(optimal_k_auto: int, K_MAX: int) -> int:
    """Gets the K value from the user and updates the session state."""
    # Initialize session state
    if "recommended_k_value" not in st.session_state or st.session_state.recommended_k_value is None:
        st.session_state.recommended_k_value = optimal_k_auto

    # Streamlit slider widget
    recommended_k = st.select_slider(
        "**Select Optimal K (Based on Elbow curve)**",
        options=range(2, K_MAX + 1),
        value=st.session_state.recommended_k_value,
        key="k_slider_widget",
        on_change=save_slider_value_to_session,  # External function
    )
    st.success(f"Selected Optimal Cluster Count (K): **{recommended_k}**")
    return recommended_k


def _run_and_evaluate_clustering(
    X_scaled: np.ndarray,
    df: pd.DataFrame,
    recommended_k: int,
) -> pd.DataFrame:
    """Runs the final clustering model and evaluates it using the Silhouette score."""
    st.subheader(f"7. Silhouette Score Evaluation (K={recommended_k})")

    # Execute K-Means
    kmeans_final: KMeans = KMeans(n_clusters=recommended_k, random_state=42, n_init=10)
    df["Cluster_Num"] = kmeans_final.fit_predict(X_scaled)

    # Calculate Silhouette score
    silhouette_avg: float = float(silhouette_score(X_scaled, df["Cluster_Num"]))
    st.metric(
        "Average Silhouette Score",
        f"{silhouette_avg:.4f}",
        help="Closer to 1 means better clustering quality. Above 0.5 is generally good.",
    )

    # Quality comment
    explain_silhouette_score = "below 0.4: Low quality, 0.4 to 0.6: Good quality, above 0.6: ExExcellent"
    if silhouette_avg > 0.6:
        st.success(f"Excellent quality! ({explain_silhouette_score})")
    elif silhouette_avg > 0.4:
        st.info(f"Good quality. ({explain_silhouette_score})")
    else:
        st.warning(f"Low quality. Review K or variables. ({explain_silhouette_score})")

    return df


def _label_and_summarize_clusters(
    df: pd.DataFrame,
    recommended_k: int,
    label_criterion: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Assigns alphabetical labels to clusters and calculates summary statistics."""
    st.subheader(f"8. Cluster Labeling and Statistics (Based on {label_criterion})")

    # 1. Calculate and sort cluster means by the criterion field
    cluster_means: pd.Series = df.groupby("Cluster_Num")[label_criterion].mean()
    sorted_clusters: pd.Series = cluster_means.sort_values(ascending=False)

    # 2. Create alphabetical label mapping ('A' has the highest mean)
    alphabet_labels: list = [chr(65 + i) for i in range(recommended_k)]
    label_map: dict = {cluster_num: alphabet_labels[i] for i, cluster_num in enumerate(sorted_clusters.index)}

    # 3. Add the alphabetical label column
    df["Cluster_Label"] = df["Cluster_Num"].map(label_map)
    st.markdown(f"**Labeling Criterion:** Cluster mean of **`{label_criterion}`** (High-to-Low)")

    # Cluster summary statistics (Centroids)
    cluster_summary: pd.DataFrame = get_data_summary(df, "Cluster_Label")  # External function
    st.markdown("### Cluster Summary Statistics (Centroids)")

    # float formatting function
    def format_float(x):
        return f"{x:,.2f}" if isinstance(x, (int, float)) else x

    st.dataframe(cluster_summary.style.format(format_float))

    return df, cluster_summary, alphabet_labels


def _visualize_clusters(
    df: pd.DataFrame,
    label_criterion: str,
    alphabet_labels: list,
):
    """Visualizes cluster size and the distribution of the criterion field."""
    st.subheader("9. Cluster Visualization Analysis")

    col1, col2 = st.columns(2)

    # 5.1. Countplot (Cluster Size)
    with col1:
        st.markdown("#### Countplot: Cluster Size")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(
            x="Cluster_Label",
            data=df,
            order=alphabet_labels,
            ax=ax,
        )
        ax.set_title("Distribution of Samples Across Clusters")
        st.pyplot(fig)

    # 5.2. Boxplot (Criterion Field Distribution)
    with col2:
        st.markdown(f"#### Boxplot: {label_criterion} Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            x="Cluster_Label",
            y=label_criterion,
            data=df,
            order=alphabet_labels,
            ax=ax,
        )
        ax.set_title(f"Distribution of {label_criterion} by Cluster")
        st.pyplot(fig)


def _profile_clusters(
    alphabet_labels: list,
    cluster_summary: pd.DataFrame,
    label_criterion: str,
    numeric_cols: list,
):
    """Generates and outputs natural language profiling based on cluster summaries."""
    st.header("10. Natural Language Cluster Profiling")
    st.info(f"Each cluster is profiled based on the selected criterion: **`{label_criterion}`**.")

    for label in alphabet_labels:
        # profile_cluster is an external function
        profile_text = profile_cluster(label, cluster_summary, label_criterion, numeric_cols)
        st.markdown(profile_text)


def run_kmeans_analysis(
    df_uploaded: pd.DataFrame,
    label_criterion: str,
):
    """
    Executes the complete K-Means analysis steps in logical units and outputs results to Streamlit.

    Args:
        df_uploaded: The original DataFrame to be analyzed.
        label_criterion: The field name used as the basis for cluster labeling.
    """
    st.header("K-Means Clustering Analysis Results")

    # 1. Data Preparation and Scaling
    prep_results: tuple = _prepare_data(df_uploaded)
    if prep_results is None:
        return

    # Unpack results with type hints
    X_scaled_df: pd.DataFrame
    numeric_cols: list[str]
    df: pd.DataFrame
    X_scaled: np.ndarray
    X_scaled_df, numeric_cols, df, X_scaled = prep_results  # type: ignore

    # 2. Optimal K Determination (Elbow Method)
    optimal_k_auto: int
    K_MAX: int
    optimal_k_auto, K_MAX = _find_optimal_k(X_scaled, len(df))

    # 3. User K Selection
    # recommended_k: int
    recommended_k: int = _get_user_k_selection(optimal_k_auto, K_MAX)

    # 4. Clustering Execution and Evaluation (Silhouette Score)
    df = _run_and_evaluate_clustering(X_scaled, df, recommended_k)

    # 5. Cluster Labeling and Summary (Centroids)
    df, cluster_summary, alphabet_labels = _label_and_summarize_clusters(df, recommended_k, label_criterion)

    # 6. Visualization
    _visualize_clusters(df, label_criterion, alphabet_labels)

    # 7. Natural Language Profiling
    _profile_clusters(alphabet_labels, cluster_summary, label_criterion, numeric_cols)


def get_last_commit_timestamp(repo_path=".") -> str:
    """Gets the timestamp of the last Git commit."""
    try:
        # Note: This command must remain compact for the user's request.
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


def _explore_tsne_perplexity(
    X_scaled_df: pd.DataFrame,
    numeric_cols: list[str],
):
    """
    Performs t-SNE for EDA purposes to visually explore cluster potential,
    and generates charts for various Perplexity values.
    """
    st.header("4. EDA - T-SNE Perplexity Exploration")

    X: np.ndarray = X_scaled_df.values  # Convert to numpy array
    N_SAMPLES: int = len(X)

    # Adjust the list of Perplexity values to explore based on data size.
    # Generally, smaller N_SAMPLES requires exploring smaller values, and vice versa.
    # Standard values between 5 and 50 are typically used.
    perplexity_list: list[int] = [5, 10, 20, 40, 50, 75, 100, 150]

    # Adjust if Perplexity is too large compared to the number of data samples
    max_perplexity = max(5, int(N_SAMPLES / 3) - 1)
    perplexity_list = [p for p in perplexity_list if p < max_perplexity]
    if not perplexity_list:
        st.warning(f"Warning: Data sample count ({N_SAMPLES}) is too small for standard t-SNE perplexity exploration.")
        return

    st.info(f"Trying t-SNE with perplexity values: **{perplexity_list}**")

    # Create columns for chart layout (maximum 4)
    cols = st.columns(min(len(perplexity_list), 4))

    for i, p in enumerate(perplexity_list):
        with cols[i % len(cols)]:  # Layout placement
            st.markdown(f"**Perplexity: {p}**")
            with st.spinner(f"Calculating t-SNE for Perplexity={p}..."):
                # Execute t-SNE
                try:
                    tsne = TSNE(
                        n_components=2,
                        perplexity=p,
                        random_state=42,
                        n_jobs=-1,
                        init="pca",
                        learning_rate="auto",
                    )
                    X_tsne = tsne.fit_transform(X)
                except ValueError as e:
                    # t-SNE error handling (e.g., when data count is too small)
                    st.error(f"t-SNE error (p={p}): {e}")
                    continue

                # Convert results to DataFrame
                tsne_df = pd.DataFrame(data=X_tsne, columns=["TSNE_Dim_1", "TSNE_Dim_2"])

                # Visualization (Scatter Plot)
                fig, ax = plt.subplots(figsize=(6, 5))  # Adjust for smaller size
                sns.scatterplot(x="TSNE_Dim_1", y="TSNE_Dim_2", data=tsne_df, alpha=0.7, ax=ax)
                ax.set_title(f"t-SNE Plot (Perplexity={p})", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig)
                plt.close(fig)  # Close to manage memory
    st.markdown(
        "Have you found the perplexity value where the clusters are clearly visible in those charts? "
        "And have you gotten an idea of approximately how many clusters would be appropriate?"
    )


# --- Streamlit UI Main Function ---
def main():
    last_updated: str = get_last_commit_timestamp()
    st.title("K-Means Clustering App")
    st.markdown(
        "This app performs K-Means clustering on the data and visualizes the results. "
        "Repository at https://github.com/tayaee/customer-segmentation-ml-kmeans/. "
        f"Last commit: {last_updated}"
    )

    data_source_col, example_data_col = st.columns(2)
    df = None
    file_uploaded = False

    # 1. File Uploader
    with data_source_col:
        st.subheader("1a. Upload Data File (Input Option #1)")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                file_uploaded = True
            except Exception as e:
                st.error(f"File processing error: {e}")
                st.info("Ensure the file is CSV and contains numeric fields.")

    # 2. Example Data Button (Right 50%)
    with example_data_col:
        st.subheader("1b. Use Example Data (Input Option #2)")
        # Manage data loading state using Streamlit session state
        if "df_loaded" not in st.session_state:
            st.session_state["df_loaded"] = None

        if st.button("Use Example Data (Retail Customer)", key="load_example_data"):
            example_file_path = "data/retail_customer_segmentation.csv"
            try:
                df = pd.read_csv(example_file_path)
                st.session_state["df_loaded"] = df
                st.success(f"'{example_file_path}' loaded.")
            except FileNotFoundError:
                st.error(f"Error: Example data file '{example_file_path}' not found. Check file path.")
            except Exception as e:
                st.error(f"Error loading example data: {e}")

    # Check file upload or example data load state
    if st.session_state["df_loaded"] is not None and not file_uploaded:
        df = st.session_state["df_loaded"]
    elif file_uploaded:
        # If new file uploaded, reset session state and use df
        st.session_state["df_loaded"] = df
    else:
        # Cannot start analysis without data
        return

    # --- Data Processing and Analysis Execution ---
    if df is not None:
        st.subheader("2. Loaded Data Preview (Top 5)")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("Error: No numeric fields found for clustering.")
            return

        # 3. Cluster Labeling Criterion Selection
        # Default to 'Income' if present, otherwise use the first numeric field for example data.
        default_index = numeric_cols.index("Income") if "Income" in numeric_cols else 0

        label_criterion = st.selectbox(
            "**Select Field for Cluster Labeling** (Labels A, B, C... determined by cluster mean - High-to-Low)",
            options=numeric_cols,
            index=default_index,
        )

        # 4. Visually identify clusters
        prep_results: tuple = _prepare_data(df)
        if prep_results is None:
            return
        X_scaled_df: pd.DataFrame
        X_scaled_df, _, _, _ = prep_results
        _explore_tsne_perplexity(X_scaled_df, numeric_cols)

        # 5. Analysis Execution Button
        if st.button("Click to Start Analysis"):
            run_kmeans_analysis(df, label_criterion)


if __name__ == "__main__":
    main()
