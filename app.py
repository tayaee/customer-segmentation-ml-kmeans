import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
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


def profile_cluster(cluster_label: str, summary: pd.DataFrame, sort_col: str, original_cols: list[str]) -> str:
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


def run_kmeans_analysis(df_uploaded: pd.DataFrame, label_criterion: str):
    """Runs K-Means analysis step-by-step and prints to Streamlit."""

    if "recommended_k_value" not in st.session_state:
        st.session_state.recommended_k_value = None

    st.header("K-Means Clustering Analysis Results")

    # --- Data Preparation ---
    df = df_uploaded.copy()

    # Select only numeric fields for clustering
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("Error: No numeric data fields available for clustering.")
        return

    X = df[numeric_cols]

    # 1. Data Scaling
    st.subheader("1. Data Scaling (Preprocessing)")
    st.info("K-Means is distance-based; scaling is essential.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
    st.dataframe(X_scaled_df.head())

    # --- Find Optimal K (Elbow Method) ---
    st.subheader("2. Find Optimal Cluster Count (K) - Elbow Method")

    # Calculate SSE
    sse = {}
    K_MAX = min(10, len(X) // 20)  # Max K is the lesser of 10 or N/20. Too large K does not help the analysis team.
    if K_MAX < 2:
        st.error("Data count is too small to perform clustering (min 20 samples needed).")
        return

    for k in range(1, K_MAX + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse[k] = kmeans.inertia_

    # Elbow Plot
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

    # Automatic K estimation (simple heuristics)
    if len(sse) >= 3:
        # Look for the point of maximum curvature (diminishing returns)
        diff = pd.Series(sse).diff().abs().dropna()
        diff_of_diff = diff.diff().abs().dropna()
        idxmin: int = diff_of_diff.idxmin()  # type: ignore
        optimal_k_auto = idxmin + 1
    else:
        optimal_k_auto = 3  # Default if not enough data points

    if st.session_state.recommended_k_value is None:
        st.session_state.recommended_k_value = optimal_k_auto
        logger.info("Using automatic K value: %s", st.session_state.recommended_k_value)
    else:
        logger.info("Using user supplied K value: %s", st.session_state.recommended_k_value)

    # User selection for K
    recommended_k = st.select_slider(
        "**Select Optimal K (Based on Elbow curve)**",
        options=range(2, K_MAX + 1),
        value=st.session_state.recommended_k_value,
        key="k_slider_widget",
        on_change=save_slider_value_to_session,
    )
    st.success(f"Selected Optimal Cluster Count (K): **{recommended_k}**")

    # --- Silhouette Score ---
    st.subheader(f"3. Silhouette Score Evaluation (K={recommended_k})")

    kmeans_final = KMeans(n_clusters=recommended_k, random_state=42, n_init=10)
    df["Cluster_Num"] = kmeans_final.fit_predict(X_scaled)

    # Calculate Silhouette score
    silhouette_avg = silhouette_score(X_scaled, df["Cluster_Num"])
    st.metric(
        "Average Silhouette Score",
        f"{silhouette_avg:.4f}",
        help="Closer to 1 means better clustering quality. Above 0.5 is generally good.",
    )

    if silhouette_avg > 0.6:
        st.success("Excellent quality! (Above 0.6)")
    elif silhouette_avg > 0.4:
        st.info("Good quality. (0.4 to 0.6)")
    else:
        st.warning("Low quality. Review K or variables. (Below 0.4)")

    # --- Cluster Labeling ---
    st.subheader(f"4. Cluster Labeling and Statistics (Based on {label_criterion})")

    # Label clusters alphabetically (A, B, C...) based on the mean of the criterion field (High-to-Low)

    # 1. Calculate the mean of the criterion field per cluster
    cluster_means = df.groupby("Cluster_Num")[label_criterion].mean()

    # 2. Sort clusters by mean in descending order
    sorted_clusters = cluster_means.sort_values(ascending=False)

    # 3. Create alphabetical label mapping ('A' is the highest mean)
    alphabet_labels = [chr(65 + i) for i in range(recommended_k)]  # 65 = 'A'
    label_map = {cluster_num: alphabet_labels[i] for i, cluster_num in enumerate(sorted_clusters.index)}

    # 4. Create new alphabetical label column
    df["Cluster_Label"] = df["Cluster_Num"].map(label_map)
    st.markdown(f"**Labeling Criterion:** Cluster mean of **`{label_criterion}`** (High-to-Low)")
    st.dataframe(df[["Cluster_Num", "Cluster_Label"] + numeric_cols].head())

    # Cluster summary statistics (Centroids)
    cluster_summary = get_data_summary(df, "Cluster_Label")
    st.markdown("### Cluster Summary Statistics (Centroids)")

    def format_float(x):
        return f"{x:,.2f}" if isinstance(x, (int, float)) else x

    st.dataframe(cluster_summary.style.format(format_float))

    # --- Visualization ---
    # st.markdown("---")
    st.subheader("5. Cluster Visualization Analysis")

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

    # 5.2. Boxplot (Label Criterion Field)
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

    # --- Natural Language Profiling ---
    st.header("6. Natural Language Cluster Profiling")
    st.info(f"Each cluster is profiled based on the selected criterion: **`{label_criterion}`**.")

    for label in alphabet_labels:
        profile_text = profile_cluster(label, cluster_summary, label_criterion, numeric_cols)
        st.markdown(profile_text)


# --- Streamlit UI Main Function ---
def main():
    st.title("K-Means Clustering App")
    st.markdown(
        "This app performs K-Means clustering on the data and visualizes the results. "
        "Repository at https://github.com/tayaee/customer-segmentation-ml-kmeans/"
    )

    data_source_col, example_data_col = st.columns(2)
    df = None
    file_uploaded = False

    # 1. File Uploader
    with data_source_col:
        st.subheader("1. Upload Data File")
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
        st.subheader("2. Use Example Data")
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
        # st.markdown("---")
        st.subheader("Loaded Data Preview (Top 5)")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.error("Error: No numeric fields found for clustering.")
            return

        # 3. Cluster Labeling Criterion Selection
        # st.markdown("---")
        # Default to 'Income' if present, otherwise use the first numeric field
        default_index = numeric_cols.index("Income") if "Income" in numeric_cols else 0

        label_criterion = st.selectbox(
            "**Select Field for Cluster Labeling** (Labels A, B, C... determined by cluster mean - High-to-Low)",
            options=numeric_cols,
            index=default_index,
        )

        # 4. Analysis Execution Button
        if st.button("Start Analysis"):
            run_kmeans_analysis(df, label_criterion)


if __name__ == "__main__":
    main()
