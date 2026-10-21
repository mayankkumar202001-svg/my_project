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
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "mayankkumar202001-svg"
repo_name = "my_project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ==========================================================
# Configure Logger
# ==========================================================

logger = logging.getLogger("model_registration")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("model_registration_errors.log")
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
    """
    Load experiment information saved during model evaluation.

    Args:
        file_path (str): Path of experiment_info.json.

    Returns:
        dict: Loaded metadata dictionary.
    """
    try:
        with open(file_path, "r") as file:
            experiment_info = json.load(file)

        logger.info("Experiment information loaded successfully from %s", file_path)
        return experiment_info

    except Exception:
        logger.exception("Unable to load experiment information from %s", file_path)
        raise


# ==========================================================
# Promote Model Version to Staging
# ==========================================================

def promote_model_to_staging(model_name: str, run_id: str):
    """
    Assign staging alias/stage to the latest registered model version for the run.

    Args:
        model_name (str): Registered model name in MLflow Registry.
        run_id (str): MLflow Run ID created during evaluation.
    """
    try:
        client = MlflowClient()

        # Search for model versions linked to this model name
        versions = client.search_model_versions(f"name='{model_name}'")

        target_version = None
        for v in versions:
            if v.run_id == run_id:
                target_version = v.version
                break

        # Fallback to the highest numeric version if run_id matching is delayed
        if not target_version and versions:
            target_version = str(max([int(v.version) for v in versions]))

        if target_version:
            logger.info("Updating stage and alias for model '%s' (Version %s)...", model_name, target_version)

            # Assign @staging alias (Modern MLflow)
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias="staging",
                    version=target_version
                )
                logger.info("Assigned alias '@staging' to model version %s.", target_version)
            except Exception as e:
                logger.warning("Could not set alias: %s", e)

            # Transition Stage to Staging (Legacy MLflow UI view compatibility)
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=target_version,
                    stage="Staging"
                )
                logger.info("Transitioned model version %s to 'Staging' stage.", target_version)
            except Exception as e:
                logger.warning("Stage transition skipped: %s", e)

        else:
            logger.warning("No registered versions found for model '%s'.", model_name)

    except Exception:
        logger.exception("Failed to promote model version.")
        raise


# ==========================================================
# Main Pipeline Stage
# ==========================================================

def main():
    """
    Execute complete model registration pipeline stage.
    """
    try:
        logger.info("Starting Model Registration Pipeline Stage...")
        logger.info("Before dagshub.init()")
        # Initialize DagsHub tracking & authentication
        dagshub.init(
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            mlflow=True
        )

        mlflow.set_tracking_uri(
            f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
        )
        
        logger.info("DagsHub initialized successfully.")

        # Load experiment metadata generated in model_evaluation
        experiment_info = load_experiment_info("./reports/experiment_info.json")

        # Promote the registered version to Staging
        promote_model_to_staging(
            model_name=REGISTERED_MODEL_NAME,
            run_id=experiment_info.get("run_id")
        )

        logger.info("=" * 60)
        logger.info("Model Registration Pipeline Completed Successfully")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Model Registration Pipeline Stage Failed.")
        raise


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()