import os
import pickle

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.logger import create_logger

logger = create_logger()


def load_params(params_path: str) -> dict:
	try:
		with open(params_path, "r", encoding="utf-8") as f:
			params = yaml.safe_load(f) or {}
			logger.debug("parameters loaded from %s", params_path)
			return params
	except FileNotFoundError:
		logger.error("Params file not found: %s", params_path)
		raise
	except yaml.YAMLError as exc:
		logger.error("YAML error: %s", exc)
		raise
	except Exception as exc:
		logger.error("Unexpected error while loading params: %s", exc)
		raise


def load_data(data_path: str) -> pd.DataFrame:
	try:
		df = pd.read_csv(data_path)
		logger.info("Data loaded from %s", data_path)
		return df
	except FileNotFoundError:
		logger.error("Data file not found: %s", data_path)
		raise
	except pd.errors.ParserError as exc:
		logger.error("Failed to parse CSV file at %s: %s", data_path, exc)
		raise
	except Exception as exc:
		logger.error("Unexpected error while loading data from %s: %s", data_path, exc)
		raise


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
	obj_cols = df.select_dtypes(include=["object"]).columns
	for col in obj_cols:
		df[col] = df[col].astype(str).str.strip()
	return df


def preprocess_data(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	target_col: str,
	drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
	try:
		train_df = train_df.copy()
		test_df = test_df.copy()

		train_df.columns = train_df.columns.str.strip()
		test_df.columns = test_df.columns.str.strip()
		train_df = _strip_string_columns(train_df)
		test_df = _strip_string_columns(test_df)

		if target_col not in train_df.columns or target_col not in test_df.columns:
			raise ValueError(f"Missing target column '{target_col}' in input datasets")

		keep_drop_cols = [col for col in drop_cols if col in train_df.columns]

		X_train = train_df.drop(columns=[target_col] + keep_drop_cols)
		X_test = test_df.drop(columns=[target_col] + keep_drop_cols)
		y_train = train_df[target_col].copy()
		y_test = test_df[target_col].copy()

		categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
		numeric_features = [col for col in X_train.columns if col not in categorical_features]

		preprocessor = ColumnTransformer(
			transformers=[
				("num", StandardScaler(), numeric_features),
				("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
			]
		)

		X_train_processed = preprocessor.fit_transform(X_train)
		X_test_processed = preprocessor.transform(X_test)

		if hasattr(X_train_processed, "toarray"):
			X_train_processed = X_train_processed.toarray()
			X_test_processed = X_test_processed.toarray()

		feature_names = preprocessor.get_feature_names_out()

		X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=train_df.index)
		X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=test_df.index)

		label_encoder = LabelEncoder()
		y_train_encoded = label_encoder.fit_transform(y_train)
		y_test_encoded = label_encoder.transform(y_test)

		X_train_df[target_col] = y_train_encoded
		X_test_df[target_col] = y_test_encoded

		logger.info(
			"Preprocessing completed successfully. Train shape: %s, Test shape: %s",
			X_train_df.shape,
			X_test_df.shape,
		)
		return X_train_df, X_test_df, preprocessor
	except Exception as exc:
		logger.error("Unexpected error during preprocessing: %s", exc)
		raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
	try:
		processed_data_path = os.path.join(data_path, "processed")
		os.makedirs(processed_data_path, exist_ok=True)

		train_data.to_csv(os.path.join(processed_data_path, "train.csv"), index=False)
		test_data.to_csv(os.path.join(processed_data_path, "test.csv"), index=False)
		logger.debug("Processed train and test data saved to %s", processed_data_path)
	except Exception as exc:
		logger.error("Unexpected error while saving processed data: %s", exc)
		raise


def save_preprocessor(preprocessor: ColumnTransformer, model_path: str) -> None:
	try:
		os.makedirs(model_path, exist_ok=True)
		preprocessor_file = os.path.join(model_path, "preprocessor.pkl")
		with open(preprocessor_file, "wb") as f:
			pickle.dump(preprocessor, f)
		logger.debug("Preprocessor object saved to %s", preprocessor_file)
	except Exception as exc:
		logger.error("Unexpected error while saving preprocessor object: %s", exc)
		raise


def main() -> None:
	try:
		params = load_params(params_path="params.yaml")
		preprocessing_params = params.get("data_preprocessing", {})

		raw_data_dir = preprocessing_params.get("raw_data_path", "./data/raw")
		target_col = preprocessing_params.get("target_col", "loan_status")
		drop_cols = preprocessing_params.get("drop_cols", ["loan_id"])
		model_path = preprocessing_params.get("model_path", "./model")

		train_df = load_data(data_path=os.path.join(raw_data_dir, "train.csv"))
		test_df = load_data(data_path=os.path.join(raw_data_dir, "test.csv"))

		train_processed, test_processed, preprocessor = preprocess_data(
			train_df=train_df,
			test_df=test_df,
			target_col=target_col,
			drop_cols=drop_cols,
		)

		save_data(train_data=train_processed, test_data=test_processed, data_path="./data")
		save_preprocessor(preprocessor=preprocessor, model_path=model_path)
	except Exception as exc:
		logger.error("Failed to complete data preprocessing process: %s", exc)
		print(f"Error: {exc}")


if __name__ == "__main__":
	main()
