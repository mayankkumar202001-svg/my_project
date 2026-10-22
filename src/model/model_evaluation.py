
import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

import dagshub

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# 1. DagsHub and MLflow Configuration
# ============================================================

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "mayankkumar202001-svg"
repo_name = "my_project"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)


# ============================================================
# 2. Logging Configuration
# ============================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_evaluation_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# 3. Load Model
# ============================================================

def load_model(file_path: str):
    """Load the trained model from a pickle file."""

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded from %s", file_path)
        return model

    except Exception:
        logger.exception("Failed to load model: %s", file_path)
        raise


# ============================================================
# 4. Load Data
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load test data from a CSV file."""

    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded from %s", file_path)
        return df

    except Exception:
        logger.exception("Failed to load data: %s", file_path)
        raise


# ============================================================
# 5. Normalize Labels
# ============================================================

def normalize_labels(labels):
    """
    Convert sentiment labels to numeric values:
    sadness = 0
    happiness = 1

    Supports string and numeric labels.
    """

    labels = (
        pd.Series(labels)
        .astype("string")
        .str.strip()
        .str.lower()
    )

    label_map = {
        "sadness": 0,
        "happiness": 1,
        "0": 0,
        "1": 1
    }

    return labels.map(label_map)


# ============================================================
# 6. Evaluate Model
# ============================================================

def evaluate_model(clf, X_test, y_test) -> dict:
    """Calculate classification evaluation metrics."""

    try:
        # Validate actual labels
        y_test = np.asarray(y_test)

        if pd.isna(y_test).any():
            raise ValueError(
                "Actual target labels (y_test) contain NaN."
            )

        # Predict class labels
        y_pred_raw = clf.predict(X_test)

        # Normalize predicted labels
        y_pred_series = normalize_labels(y_pred_raw)

        if y_pred_series.isna().any():
            invalid_predictions = pd.Series(y_pred_raw)[
                y_pred_series.isna()
            ].unique()

            raise ValueError(
                f"Unknown prediction labels: {invalid_predictions}"
            )

        y_pred = y_pred_series.astype(int).to_numpy()

        # Ensure labels are binary
        if not np.isin(y_test, [0, 1]).all():
            raise ValueError(
                "y_test must contain only 0 and 1."
            )

        # Get probabilities for class 1
        if not hasattr(clf, "predict_proba"):
            raise ValueError(
                "The model does not support predict_proba()."
            )

        probabilities = clf.predict_proba(X_test)
        classes = np.asarray(clf.classes_)

        positive_class_indices = np.where(classes == 1)[0]

        if len(positive_class_indices) != 1:
            raise ValueError(
                f"Cannot identify class 1 in model classes: {classes}"
            )

        positive_index = positive_class_indices[0]
        y_pred_proba = probabilities[:, positive_index]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        # AUC requires both classes in the actual labels
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
            logger.warning(
                "ROC AUC cannot be calculated because "
                "y_test contains only one class."
            )

        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc) if auc is not None else None
        }

        logger.info("Evaluation metrics: %s", metrics_dict)

        return metrics_dict

    except Exception:
        logger.exception("Error during model evaluation")
        raise


# ============================================================
# 7. Save Metrics
# ============================================================

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics as a JSON file."""

    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Metrics saved to %s", file_path)

    except Exception:
        logger.exception("Failed to save metrics")
        raise


# ============================================================
# 8. Save Experiment Information
# ============================================================

def save_model_info(
    run_id: str,
    model_path: str,
    file_path: str
) -> None:
    """Save MLflow run ID and model path."""

    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }

        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)

        logger.info("Model information saved to %s", file_path)

    except Exception:
        logger.exception("Failed to save model information")
        raise


# ============================================================
# 9. Main Function
# ============================================================

def main():

    # Ensure report directory exists
    os.makedirs("reports", exist_ok=True)

    mlflow.set_experiment("model_evaluation")

    with mlflow.start_run() as run:

        try:
            # Load model and test data
            clf = load_model("./models/model.pkl")

            test_data = load_data(
                "./data/processed/test_bow.csv"
            )

            if test_data.empty:
                raise ValueError("The test dataset is empty.")

            if test_data.shape[1] < 2:
                raise ValueError(
                    "Test data must contain features and a target column."
                )

            # Split features and target
            X_test = test_data.iloc[:, :-1].values
            y_raw = test_data.iloc[:, -1]

            # Display original labels for debugging
            logger.info(
                "Original target labels:\n%s",
                y_raw.value_counts(dropna=False)
            )

            # Normalize actual target labels
            y_test_series = normalize_labels(y_raw)

            # Identify invalid or missing labels
            invalid_mask = y_test_series.isna()

            if invalid_mask.any():
                invalid_labels = y_raw[invalid_mask].value_counts(
                    dropna=False
                )

                raise ValueError(
                    "Invalid or missing target labels found:\n"
                    f"{invalid_labels}"
                )

            y_test = y_test_series.astype(int).to_numpy()

            # Check feature and target row counts
            if len(X_test) != len(y_test):
                raise ValueError(
                    "Feature and target row counts do not match."
                )

            # Evaluate model
            metrics = evaluate_model(
                clf,
                X_test,
                y_test
            )

            # Save metrics locally
            save_metrics(
                metrics,
                "reports/metrics.json"
            )

            # Log available metrics to MLflow
            valid_metrics = {
                name: value
                for name, value in metrics.items()
                if value is not None
            }

            mlflow.log_metrics(valid_metrics)

            # Log model parameters
            if hasattr(clf, "get_params"):
                for param_name, param_value in clf.get_params().items():

                    if isinstance(
                        param_value,
                        (str, int, float, bool)
                    ) or param_value is None:

                        mlflow.log_param(
                            param_name,
                            param_value
                        )

            # Log and register model
            mlflow.sklearn.log_model(
                clf,
                artifact_path="model",
                registered_model_name="sentiment-analysis-model"
            )

            # Save model information
            save_model_info(
                run.info.run_id,
                "sentiment-analysis-model",
                "reports/experiment_info.json"
            )

            # Log artifacts
            mlflow.log_artifact(
                "reports/metrics.json"
            )

            mlflow.log_artifact(
                "reports/experiment_info.json"
            )

            if os.path.exists("model_evaluation_errors.log"):
                mlflow.log_artifact(
                    "model_evaluation_errors.log"
                )

            logger.info(
                "Model evaluation completed successfully."
            )

        except Exception:
            logger.exception(
                "Failed to complete model evaluation."
            )
            raise


# ============================================================
# 10. Entry Point
# ============================================================

if __name__ == "__main__":
    main()
