import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("MLflow Metrics Dashboard")

# Set MLflow tracking URI
mlflow.set_tracking_uri("mlruns")

# Fetch all runs
def get_mlflow_data():
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=client.get_experiment_by_name("cpu-usage-experiment").experiment_id,
        order_by=["start_time DESC"]
    )
    
    data = []
    for run in runs:
        run_data = {"Run ID": run.info.run_id, "Run Name": run.data.tags.get("mlflow.runName", "N/A")}
        for metric_key, metric_value in run.data.metrics.items():
            run_data[metric_key] = metric_value
        for param_key, param_value in run.data.params.items():
            run_data[param_key] = param_value
        data.append(run_data)
    
    return pd.DataFrame(data)

df_runs = get_mlflow_data()

if not df_runs.empty:
    st.subheader("MLflow Run Visualizations")

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test R2 Score Across Runs")
        fig_r2, ax_r2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Run Name", y="test_r2", data=df_runs.sort_values(by="test_r2", ascending=False), ax=ax_r2)
        ax_r2.set_ylabel("Test R2")
        ax_r2.set_xlabel("Run Name")
        ax_r2.tick_params(axis='x', rotation=45)
        st.pyplot(fig_r2)

    with col2:
        st.subheader("Test RMSE Across Runs")
        fig_rmse, ax_rmse = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Run Name", y="test_rmse", data=df_runs.sort_values(by="test_rmse"), ax=ax_rmse)
        ax_rmse.set_ylabel("Test RMSE")
        ax_rmse.set_xlabel("Run Name")
        ax_rmse.tick_params(axis='x', rotation=45)
        st.pyplot(fig_rmse)

    # Additional plot for hyperparameter vs. metric if relevant parameters exist
    if "max_depth" in df_runs.columns and "test_rmse" in df_runs.columns:
        df_filtered = df_runs.dropna(subset=["max_depth", "test_rmse"])
        if not df_filtered.empty:
            st.subheader("Hyperparameter Analysis: Max Depth vs. Test RMSE")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x="max_depth", y="test_rmse", hue="Run Name", data=df_filtered, ax=ax_scatter)
            ax_scatter.set_ylabel("Test RMSE")
            ax_scatter.set_xlabel("Max Depth")
            st.pyplot(fig_scatter)
            
    st.subheader("All MLflow Runs Metrics and Parameters")
    st.dataframe(df_runs)

else:
    st.info("No MLflow runs found.") 