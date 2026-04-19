# register model

import json
import mlflow
from src.logger import create_logger
import os
import dagshub
import yaml

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logger = create_logger()


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file) or {}
        logger.debug('parameters loaded from %s', params_path)
        return params
    except Exception as exc:
        logger.error('Unexpected error while loading params: %s', exc)
        raise


def setup_tracking(register_params: dict) -> None:
    tracking_uri = register_params.get('tracking_uri', 'https://dagshub.com/kevinsangani988/Capstone-MLops.mlflow')
    repo_owner = register_params.get('repo_owner', 'kevinsangani988')
    repo_name = register_params.get('repo_name', 'Capstone-MLops')

    dagshub_token = (
        os.getenv('DAGSHUB_TOKEN')
        or os.getenv('DAGSHUB_USER_TOKEN')
        or os.getenv('MLFLOW_TRACKING_PASSWORD')
    )

    if 'dagshub.com' in tracking_uri and not dagshub_token:
        raise ValueError(
            'DagsHub credentials are required for model registration. '
            'Set DAGSHUB_TOKEN (or DAGSHUB_USER_TOKEN / MLFLOW_TRACKING_PASSWORD).'
        )

    mlflow.set_tracking_uri(tracking_uri)
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def _candidate_model_uris(model_info: dict) -> list[str]:
    """Build candidate model URIs that work across MLflow versions."""
    run_id = model_info.get("run_id")
    model_path = model_info.get("model_path", "model")
    explicit_model_uri = model_info.get("model_uri")

    candidates: list[str] = []
    if explicit_model_uri:
        candidates.append(explicit_model_uri)
    if run_id:
        candidates.append(f"runs:/{run_id}/{model_path}")

    client = mlflow.tracking.MlflowClient()

    # Newer MLflow tracks logged models that can be referenced by model id.
    if run_id and hasattr(client, "search_logged_models"):
        try:
            run = client.get_run(run_id)
            experiment_id = run.info.experiment_id
            logged_models = client.search_logged_models(
                experiment_ids=[experiment_id],
                filter_string=f"source_run_id='{run_id}'",
            )
            for logged_model in logged_models:
                model_id = getattr(logged_model, "model_id", None)
                model_uri = getattr(logged_model, "model_uri", None)

                if model_uri:
                    candidates.append(model_uri)
                if model_id:
                    candidates.append(f"models:/{model_id}")
        except Exception as exc:
            logger.warning("Could not query logged models for run %s: %s", run_id, exc)

    # De-duplicate while preserving order.
    deduped: list[str] = []
    for uri in candidates:
        if uri and uri not in deduped:
            deduped.append(uri)
    return deduped

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict, stage: str = 'Staging'):
    """Register the model to the MLflow Model Registry."""
    candidate_uris = _candidate_model_uris(model_info)
    if not candidate_uris:
        raise ValueError("No valid model URI candidates found in experiment_info")

    last_error: Exception | None = None
    for model_uri in candidate_uris:
        try:
            logger.info("Trying model registration with URI: %s", model_uri)
            model_version = mlflow.register_model(model_uri, model_name)

            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )

            logger.debug(
                "Model %s version %s registered and transitioned to Staging.",
                model_name,
                model_version.version,
            )
            return
        except Exception as exc:
            last_error = exc
            logger.warning("Registration failed for URI %s: %s", model_uri, exc)

    logger.error("Error during model registration after trying all URI candidates: %s", last_error)
    raise last_error

def main():
    try:
        params = load_params('params.yaml')
        register_params = params.get('model_registration', {})

        setup_tracking(register_params)

        model_info_path = register_params.get('model_info_path', 'reports/experiment_info.json')
        model_name = register_params.get('model_name', 'my_model')
        stage = register_params.get('stage', 'Staging')

        model_info = load_model_info(model_info_path)

        register_model(model_name, model_info, stage)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

