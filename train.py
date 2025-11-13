import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import json
from io import BytesIO
import os
import requests

# ------------------------------
# CONFIG MLflow Tracking (DagsHub)
# ------------------------------
DAGSHUB_OWNER = "manvip28"
DAGSHUB_REPO = "cpu-usage-mlops"

TRACKING_URI = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)

BEST_RUN_ID = "1584ae5a899e4e14b0f699322958f367"   # your best run


# ------------------------------
# Load MLflow model
# ------------------------------
@st.cache_resource
def load_model():
    model_uri = f"runs:/{BEST_RUN_ID}/model"
    return mlflow.sklearn.load_model(model_uri)

model = load_model()


# ------------------------------
# LOAD TRAINING FEATURE COLUMNS
# ------------------------------
def load_training_columns(run_id):
    """Pulls the input_example.json from MLflow which contains the exact training columns."""
    client = mlflow.tracking.MlflowClient()
    temp_path = client.download_artifacts(run_id, "model/input_example.json")

    with open(temp_path, "r") as f:
        example = json.load(f)

    return list(example.keys())


TRAIN_COLS = load_training_columns(BEST_RUN_ID)
st.sidebar.success("Training columns loaded:")


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("CPU Usage Prediction Dashboard")
st.write("This dashboard uses an MLflow model logged on DagsHub.")

st.header("Enter Input Features")

cpu_request = st.number_input("CPU Request", min_value=0.0)
mem_request = st.number_input("Memory Request (MB)", min_value=0.0)
cpu_limit = st.number_input("CPU Limit", min_value=0.0)
mem_limit = st.number_input("Memory Limit (MB)", min_value=0.0)
runtime_minutes = st.number_input("Runtime (minutes)", min_value=0.0)

controller_kind = st.selectbox(
    "Controller Kind",
    ["Job", "ReplicaSet", "ReplicationController", "StatefulSet"]
)

# Create base DataFrame
input_df = pd.DataFrame([{
    "cpu_request": cpu_request,
    "mem_request": mem_request,
    "cpu_limit": cpu_limit,
    "mem_limit": mem_limit,
    "runtime_minutes": runtime_minutes,
    "controller_kind": controller_kind
}])


# ------------------------------
# AUTO ONE-HOT ENCODE LIKE TRAINING
# ------------------------------
def preprocess_input(df, training_columns):
    """Replicates training preprocessing: one-hot encode and align columns."""
    df = pd.get_dummies(df)

    # Add missing columns from training
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Extra columns (from new user input)
    df = df[training_columns]

    return df


processed_input = preprocess_input(input_df, TRAIN_COLS)


# ------------------------------
# PREDICT
# ------------------------------
if st.button("Predict CPU Usage"):
    pred = model.predict(processed_input)[0]
    st.success(f"Predicted CPU Usage: **{pred:.4f}**")


# ------------------------------
# SHOW METRICS
# ------------------------------
st.header("Model Metrics")

def get_metrics(run_id):
    client = mlflow.tracking.MlflowClient()
    return client.get_run(run_id).data.metrics

try:
    st.json(get_metrics(BEST_RUN_ID))
except:
    st.warning("Could not load metrics.")


# ------------------------------
# SHOW ARTIFACTS
# ------------------------------
st.header("Model Plots")

def load_image(artifact_path):
    client = mlflow.tracking.MlflowClient()
    path = client.download_artifacts(BEST_RUN_ID, artifact_path)
    with open(path, "rb") as f:
        return f.read()

try:
    st.subheader("Residual Plot")
    st.image(load_image("residuals_temp.png"))

    st.subheader("Feature Importance")
    st.image(load_image("feature_importance_temp.png"))
except:
    st.info("Artifacts not found or not logged.")


st.write("---")
st.write("Powered by MLflow + DagsHub + Streamlit")
