import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import create_logger
from src.blob import blob_connection_load_data

logger = create_logger()


def _clean_connection_string(raw_value: str | None) -> str | None:
    """Normalize connection string value from env/secrets.

    Handles whitespace/quotes and values accidentally saved as
    "CONN_STRING=..." in secret managers.
    """
    if not raw_value:
        return None

    cleaned = str(raw_value).strip().strip('"').strip("'")
    if cleaned.startswith("CONN_STRING="):
        cleaned = cleaned.split("=", 1)[1].strip()
    return cleaned or None


def _get_blob_connection_string() -> tuple[str, str]:
    """Fetch blob connection string from supported environment variable names."""
    candidate_env_vars = ["CONN_STRING", "AZURE_STORAGE_CONNECTION_STRING", "CAPSTONE_TEST"]

    for env_var in candidate_env_vars:
        value = _clean_connection_string(os.getenv(env_var))
        if value:
            return value, env_var

    raise ValueError(
        "Azure blob connection string is missing. Set CONN_STRING or "
        "AZURE_STORAGE_CONNECTION_STRING in the environment/secrets."
    )


def _validate_connection_string(conn_str: str) -> None:
    """Validate shape of Azure storage connection string without logging secrets."""
    required_parts = ["DefaultEndpointsProtocol=", "AccountName=", "AccountKey=", "EndpointSuffix="]
    if not all(part in conn_str for part in required_parts):
        raise ValueError(
            "Connection string is blank or malformed. Expected Azure Storage "
            "format containing DefaultEndpointsProtocol, AccountName, AccountKey, and EndpointSuffix."
        )

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f) or {}
            logger.debug('parameters loaded from %s', params_path)
            return params
    except FileNotFoundError:
        logger.error('file not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        ingestion_params = params.get('data_ingestion', {})

        test_size = ingestion_params.get('test_size', 0.25)
        random_state = ingestion_params.get('random_state', 42)
        data_path = ingestion_params.get('data_path', './data')
        source_type = ingestion_params.get('source_type', 'blob')

        if source_type == 'csv':
            data_url = ingestion_params.get('data_url')
            df = load_data(data_url=data_url)
        else:
            blob_conn_str, env_var_name = _get_blob_connection_string()
            _validate_connection_string(blob_conn_str)
            blob_container = ingestion_params.get('container')
            blob_file_name = ingestion_params.get('blob_name')
            logger.info("Using blob connection string from env var '%s'.", env_var_name)
            df = blob_connection_load_data(
                conn_str=blob_conn_str,
                container=blob_container,
                blob_name=blob_file_name,
            )

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        save_data(train_data, test_data, data_path=data_path)
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
