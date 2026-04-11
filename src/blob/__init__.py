import os
import logging

import pandas as pd
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()


def blob_connection_load_data(conn_str:str, container:str,blob_name:str):
        try:
                if not conn_str or not container or not blob_name:
                        raise ValueError("conn_str, container, and blob_name are required.")

                logger.info("Connecting to blob container '%s'.", container)
                blob_service_client = BlobServiceClient.from_connection_string(conn_str)
                blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)

                logger.info("Downloading blob '%s'.", blob_name)
                stream = blob_client.download_blob()

                logger.info("Reading blob content into DataFrame.")
                df = pd.read_csv(stream)
                logger.info("Data loaded successfully with %d rows and %d columns.", *df.shape)
                return df

        except ValueError as exc:
                logger.error("Input validation failed: %s", exc)
                raise
        except AzureError as exc:
                logger.exception("Azure Blob operation failed: %s", exc)
                raise
        except pd.errors.EmptyDataError as exc:
                logger.exception("The blob file is empty: %s", exc)
                raise
        except pd.errors.ParserError as exc:
                logger.exception("Failed to parse CSV from blob: %s", exc)
                raise
        except Exception as exc:
                logger.exception("Unexpected error while loading data from blob: %s", exc)
                raise
    
