# model_evaluation.py

import os
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. Configuration
# ============================================================

DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER = "mayankkumar202001-svg"
REPO_NAME = "my_project"

EXPERIMENT_NAME = "model_evaluation"
REGISTERED_MODEL_NAME = "sentiment-analysis-model"

MODEL_PATH = Path("models/model.pkl")
TEST_DATA_PATH = Path("data/processed/test_bow.csv")

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_INFO_PATH = REPORTS_DIR / "experiment_info.json"
ERROR_LOG_PATH = Path("model_evaluation_errors.log")


# ============================================================
# 2. Logging configuration
# ============================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers if the module is reloaded
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(ERROR_LOG_PATH)
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# 3. DagsHub and MLflow configuration
# ============================================================

def setup_mlflow():
    """Configure DagsHub authentication and MLflow tracking."""

    dagshub_token = os.getenv("DAGSHUB_PAT")

    if not dagshub_token:
        raise EnvironmentError(
            "DAGSHUB_PAT environment variable is not set."
        )

    # Use your DagsHub personal access token.
    # Keep this token in an environment variable or CI secret.
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    tracking_uri = (
        f"{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("MLflow tracking URI configured.")


# ============================================================
# 4. Load model
# ============================================================

def load_model(file_path: Path):
    """Load the trained model from a pickle file."""

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded from %s", file_path)
        return model

    except Exception:
        logger.exception("Failed to load model from %s", file_path)
        raise


# ============================================================
# 5. Load test data
# ============================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """Load test data from a CSV file."""

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"The dataset is empty: {file_path}")

        logger.info(
            "Test data loaded from %s with shape %s",
            file_path,
            df.shape,
        )

        return df

    except Exception:
        logger.exception("Failed to load data from %s", file_path)
        raise


# ============================================================
# 6. Encode labels consistently
# ============================================================

def encode_labels(values) -> np.ndarray:
    """
    Convert sentiment labels to binary integers.

    sadness   -> 0
    happiness -> 1

    Also supports numeric labels 0 and 1.
    """

    values = pd.Series(values)

    label_map = {
        "sadness": 0,
        "happiness": 1,
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1,
    }

    encoded = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .map(label_map)
    )

    if encoded.isna().any():
        invalid_labels = (
            values[encoded.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Unknown or missing sentiment labels: "
            f"{invalid_labels[:10]}"
        )

    return encoded.astype(int).to_numpy()


# ============================================================
# 7. Prepare test features and labels
# ============================================================

def prepare_test_data(test_data: pd.DataFrame):
    """
    Assume the last CSV column contains the target label.
    Remove rows with missing target values before splitting X and y.
    """

    if test_data.shape[1] < 2:
        raise ValueError(
            "Test data must contain at least one feature column "
            "and one target column."
        )

    target_col = test_data.columns[-1]

    # Remove rows with missing target labels before creating X and y.
    test_data = test_data.dropna(subset=[target_col]).copy()

    if test_data.empty:
        raise ValueError("No valid rows remain in the test dataset.")

    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = encode_labels(test_data[target_col])

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    logger.info(
        "Prepared test data: X=%s, y=%s",
        X_test.shape,
        y_test.shape,
    )

    return X_test, y_test


# ============================================================
# 8. Evaluate model
# ============================================================

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calculate accuracy, precision, recall and ROC-AUC."""

    try:
        # Check that the test features match the model's expected input.
        if (
            hasattr(clf, "n_features_in_")
            and X_test.shape[1] != clf.n_features_in_
        ):
            raise ValueError(
                f"Model expects {clf.n_features_in_} features, "
                f"but test data contains {X_test.shape[1]}."
            )

        # Actual predictions, converted to 0 and 1.
        y_pred = encode_labels(clf.predict(X_test))

        # Both classes must be present for binary ROC-AUC.
        if len(np.unique(y_test)) != 2:
            raise ValueError(
                "ROC-AUC requires both classes in the test dataset."
            )

        # Calculate probability for the positive class: happiness (1).
        if not hasattr(clf, "predict_proba"):
            raise ValueError(
                "The model does not support predict_proba(). "
                "ROC-AUC probability evaluation cannot be performed."
            )

        if not hasattr(clf, "classes_"):
            raise ValueError(
                "The model does not expose classes_; "
                "cannot determine the probability column."
            )

        encoded_classes = encode_labels(clf.classes_)
        positive_indices = np.flatnonzero(encoded_classes == 1)

        if len(positive_indices) != 1:
            raise ValueError(
                "Could not identify the happiness class "
                "in the model's classes_."
            )

        y_pred_proba_all = clf.predict_proba(X_test)
        y_pred_proba = y_pred_proba_all[:, positive_indices[0]]

        # Calculate metrics.
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            ),
            "auc": float(
                roc_auc_score(y_test, y_pred_proba)
            ),
        }

        logger.info("Evaluation metrics: %s", metrics)

        return metrics

    except Exception:
        logger.exception("Error during model evaluation")
        raise


# ============================================================
# 9. Save metrics
# ============================================================

def save_metrics(metrics: dict, file_path: Path) -> None:
    """Save evaluation metrics to a JSON file."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Metrics saved to %s", file_path)


# ============================================================
# 10. Save MLflow model information
# ============================================================

def save_model_info(
    run_id: str,
    model_uri: str,
    registered_model_name: str,
    file_path: Path,
) -> None:
    """Save the run ID, model URI and registry name."""

    model_info = {
        "run_id": run_id,
        "model_uri": model_uri,
        "registered_model_name": registered_model_name,
    }

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    logger.info("Model information saved to %s", file_path)


# ============================================================
# 11. Log model parameters
# ============================================================

def log_model_parameters(clf) -> None:
    """Log model parameters to MLflow."""

    if not hasattr(clf, "get_params"):
        logger.warning("Model does not support get_params().")
        return

    for param_name, param_value in clf.get_params().items():
        # MLflow parameter values should be strings or simple values.
        value = str(param_value)

        # Avoid excessive parameter-value lengths.
        if len(value) > 500:
            value = value[:497] + "..."

        mlflow.log