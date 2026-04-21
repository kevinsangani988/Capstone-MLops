import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import os
import yaml
from src.logger import create_logger


logger = create_logger()

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file) or {}
        logger.debug('parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Unexpected error while loading params: %s', e)
        raise


def setup_tracking(eval_params: dict) -> None:
    tracking_uri = eval_params.get('tracking_uri', 'https://dagshub.com/kevinsangani988/Capstone-MLops.mlflow')
    repo_owner = eval_params.get('repo_owner', 'kevinsangani988')
    repo_name = eval_params.get('repo_name', 'Capstone-MLops')

    dagshub_token = (
        os.getenv('DAGSHUB_USER_TOKEN')
        or os.getenv('DAGSHUB_TOKEN')
        or os.getenv('MLFLOW_TRACKING_PASSWORD')
    )
    is_ci = os.getenv('GITHUB_ACTIONS', '').lower() == 'true'

    if dagshub_token:
        # dagshub client reliably reads DAGSHUB_USER_TOKEN.
        os.environ.setdefault('DAGSHUB_USER_TOKEN', dagshub_token)

    local_tracking_uri = 'sqlite:///mlflow.db'

    if 'dagshub.com' in tracking_uri and not dagshub_token:
        mlflow.set_tracking_uri(local_tracking_uri)
        logger.warning(
            'DagsHub credentials not found. Falling back to local MLflow tracking at %s',
            local_tracking_uri,
        )
        return

    mlflow.set_tracking_uri(tracking_uri)
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        if is_ci:
            logger.info('Running in CI with token. Using DagsHub MLflow tracking: %s', tracking_uri)
    except Exception as exc:
        mlflow.set_tracking_uri(local_tracking_uri)
        logger.warning(
            'DagsHub tracking init failed (%s). Falling back to local MLflow at %s.',
            exc,
            local_tracking_uri,
        )

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str, model_uri: str | None = None) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        if model_uri:
            model_info['model_uri'] = model_uri
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    params = load_params('params.yaml')
    eval_params = params.get('model_evaluation', {})

    setup_tracking(eval_params)

    experiment_name = eval_params.get('experiment_name', 'my-dvc-pipeline')
    model_file_path = eval_params.get('model_file_path', './model/model.pkl')
    test_data_path = eval_params.get('test_data_path', './data/processed/test.csv')
    metrics_path = eval_params.get('metrics_path', 'reports/metrics.json')
    experiment_info_path = eval_params.get('experiment_info_path', 'reports/experiment_info.json')
    mlflow_model_name = eval_params.get('mlflow_model_name', 'model')

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model(model_file_path)
            test_data = load_data(test_data_path)
            
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, metrics_path)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                serializable_params = {
                    name: value
                    for name, value in params.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
                if serializable_params:
                    mlflow.log_params(serializable_params)
            
            # Log model to MLflow
            logged_model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                name=mlflow_model_name,
                serialization_format="skops",
            )
            
            # Save model info
            save_model_info(
                run.info.run_id,
                mlflow_model_name,
                experiment_info_path,
                getattr(logged_model_info, 'model_uri', None),
            )
            
            # Log the metrics file to MLflow
            mlflow.log_artifact(metrics_path)

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
