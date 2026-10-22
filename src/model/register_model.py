
# ==========================================================
# Import Required Libraries
# ==========================================================
import os
import json
import logging

import dagshub
import mlflow
from mlflow import MlflowClient


# ==========================================================
# Global Configuration & Constants
# ==========================================================
DAGSHUB_TOKEN = os.getenv("DAGSHUB_PAT")

if not DAGSHUB_TOKEN:
    raise EnvironmentError(
        "DAGSHUB_PAT is missing. Configure it in GitHub Actions Secrets."
    )

REPO_OWNER = "mayankkumar202001-svg"
REPO_NAME = "my_project"
REGISTERED_MODEL_NAME = "sentiment-analysis-model"

MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ==========================================================
# Configure Logger
# ==========================================================
logger = logging.getLogger("model_registration")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        "model_registration_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ==========================================================
# Load Experiment Information
# ==========================================================
def load_experiment_info(file_path: str) -> dict:
    """Load metadata created during model evaluation."""

    try:
        with open(file_path, "r") as file:
            experiment_info = json.load(file)

        if not experiment_info.get("run_id"):
            raise ValueError(
                "run_id is missing from experiment_info.json"
            )

        logger.info(
            "Experiment information loaded from %s",
            file_path
        )

        return experiment_info

    except Exception:
        logger.exception(
            "Unable to load experiment information"
        )
        raise


# ==========================================================
# Promote Model Version to Staging Alias
# ==========================================================
def promote_model_to_staging(
    model_name: str,
    run_id: str
):
    """
    Find the registered model version associated with run_id
    and assign the staging alias to that version.
    """

    try:
        client = MlflowClient()

        if not run_id:
            raise ValueError("A valid MLflow run_id is required.")

        # Find registered versions for this model
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        # Match the exact MLflow run ID
        matching_versions = [
            version
            for version in versions
            if version.run_id == run_id
        ]

        if not matching_versions:
            raise ValueError(
                f"No registered version found for model "
                f"'{model_name}' with run_id '{run_id}'. "
                "Check model registration and tracking URI."
            )

        # Select the newest matching version
        target_version = max(
            matching_versions,
            key=lambda version: int(version.version)
        )

        logger.info(
            "Found model '%s', version %s",
            model_name,
            target_version.version
        )

        # Assign or update the staging alias
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=target_version.version
        )

        logger.info(
            "Alias '@staging' now points to model version %s",
            target_version.version
        )

        # Verify the alias assignment
        staging_version = client.get_model_version_by_alias(
            name=model_name,
            alias="staging"
        )

        if str(staging_version.version) != str(
            target_version.version
        ):
            raise RuntimeError(
                "Staging alias verification failed."
            )

        logger.info(
            "Verified staging alias for model version %s",
            staging_version.version
        )

        return target_version.version

    except Exception:
        logger.exception(
            "Failed to promote model to staging"
        )
        raise


# ==========================================================
# Main Pipeline Stage
# ==========================================================
def main():
    """Execute the model registration pipeline stage."""

    try:
        logger.info(
            "Starting Model Registration Pipeline Stage"
        )

        # Initialize DagsHub
       

        # Explicitly set tracking URI
        

        logger.info(
            "MLflow tracking URI configured: %s",
            MLFLOW_TRACKING_URI
        )

        # Load metadata from model evaluation stage
        experiment_info = load_experiment_info(
            "./reports/experiment_info.json"
        )

        run_id = experiment_info["run_id"]

        # Promote exact registered version
        version = promote_model_to_staging(
            model_name=REGISTERED_MODEL_NAME,
            run_id=run_id
        )

        logger.info("=" * 60)
        logger.info(
            "Model Registration Pipeline Completed Successfully"
        )
        logger.info(
            "Registered model: %s",
            REGISTERED_MODEL_NAME
        )
        logger.info("Staging version: %s", version)
        logger.info("=" * 60)

    except Exception:
        logger.exception(
            "Model Registration Pipeline Stage Failed"
        )
        raise


# ==========================================================
# Driver Code
# ==========================================================
if __name__ == "__main__":
    main()
