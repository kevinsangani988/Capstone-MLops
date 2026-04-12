import os
import pandas as pd
from src.logger import create_logger
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml


logger = create_logger()


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        logger.debug("parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error(f"unexpected error while loading params: {e}")
        raise

def load_data(path: str) -> pd.DataFrame :
    try: 
        df = pd.read_csv(path)
        logger.debug("dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"unexpected error : {e}")
        raise

def model_training(x_df: pd.DataFrame, y_df: pd.DataFrame, model_params: dict):
    try:
        logger.debug("creating model instance")

        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 300),
            max_depth=model_params.get("max_depth", 20),
            min_samples_leaf=model_params.get("min_samples_leaf", 1),
            min_samples_split=model_params.get("min_samples_split", 2),
            criterion=model_params.get("criterion", "gini"),
            max_features=model_params.get("max_features", "sqrt"),
            bootstrap=model_params.get("bootstrap", True),
            max_samples=model_params.get("max_samples", 0.8),
            class_weight=model_params.get("class_weight", None),
            random_state=model_params.get("random_state", 42),
            n_jobs=model_params.get("n_jobs", -1),
        )

        logger.debug("model training started")

        model.fit(x_df,y_df)

        logger.debug("model training successfully completed")

        return model
    except Exception as e:
        logger.error(f"unexpected error {e}")
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model,f)
        logger.debug(f"model successfully dumped to {file_path}")
    except Exception as e:
        logger.error(f"unexpected error : {e}")
        raise


def main():

    try:
        params = load_params("params.yaml")
        model_params = params.get("model_building", {})

        train_path = model_params.get("train_data_path", os.path.join("data", "processed", "train.csv"))
        model_path = model_params.get("model_output_path", os.path.join("model", "model.pkl"))
        target_col = model_params.get("target_col", "loan_status")

        df = load_data(train_path)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")

        x_df = df.drop(columns=[target_col])
        y_df = df[target_col]
        
        model = model_training(x_df, y_df, model_params)

        save_model(model=model,file_path=model_path)



    except Exception as e:
        logger.error(f"unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
