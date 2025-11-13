import os
import sys
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    print("Please install xgboost: pip install xgboost")
    sys.exit(1)


def train_and_save(X_train, X_test, y_train, y_test, model, model_name, feature_name_type, model_path_base):
    # Train
    model.fit(X_train, y_train)

    # Predict
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics
    metrics = {
        "model": model_name,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
        "comments": f"Trained {model_name} model"
    }

    # Create folders
    model_dir = os.path.join("metrics", model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path_base), exist_ok=True)

    # Save metrics JSON
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Residual plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, test_preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual CPU Usage")
    plt.ylabel("Predicted CPU Usage")
    plt.title(f"Residuals Plot ({model_name})")
    residuals_path = os.path.join(model_dir, "residuals.png")
    plt.savefig(residuals_path)
    plt.close()

    # Feature importance / coefficients
    try:
        if model_name == "linear":
            importance = model.coef_
        else:
            importance = model.feature_importances_

        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance
        }).sort_values(by="importance", key=abs, ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(feature_importance["feature"], feature_importance["importance"])
        plt.xlabel(feature_name_type)
        plt.title(f"{feature_name_type} ({model_name})")
        feature_imp_path = os.path.join(model_dir, "feature_importance.png")
        plt.savefig(feature_imp_path)
        plt.close()
    except Exception:
        feature_imp_path = None  # not all models support feature importance

    # MLflow logging
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model", model_name)
        mlflow.log_param("data_version", os.getenv("DVC_DATA_VERSION", "unknown"))

        for k, v in metrics.items():
            if k not in ["model", "comments"]:
                mlflow.log_metric(k, v)

        # Log artifacts
        mlflow.log_artifact(residuals_path)
        if feature_imp_path:
            mlflow.log_artifact(feature_imp_path)

        # Log model
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=X_train.head(1),
        pip_requirements=[
        "scikit-learn",
        "numpy",
        "pandas",
        "xgboost",
        "joblib"
        ]
        )


        # Save run info
        run_info = {
            "mlflow_run_id": run.info.run_id,
            "dvc_data_version": os.getenv("DVC_DATA_VERSION", "unknown")
        }
        with open(os.path.join(model_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=4)

    # Save model locally for DVC
    model_file = os.path.join(model_path_base, f"{model_name}_model.pkl")
    joblib.dump(model, model_file)
    print(f"✅ Training complete. {model_name} model saved to {model_file}")


def train_all(data_path, model_path_base):
    df = pd.read_csv(data_path)
    X = df.drop("cpu_usage", axis=1)
    y = df["cpu_usage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "linear": (LinearRegression(), "Coefficient"),
        "rf": (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), "Feature Importance"),
        "xgb": (XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                             objective='reg:squarederror', random_state=42, verbosity=0), "Feature Importance")
    }

    for model_name, (model, feature_type) in models.items():
        train_and_save(X_train, X_test, y_train, y_test, model, model_name, feature_type, model_path_base)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <data_path> <models_folder>")
        sys.exit(1)

    data_path = sys.argv[1]
    model_path_base = sys.argv[2]
    train_all(data_path, model_path_base)
