"""FastAPI app for loan approval inference.

This module loads a trained model (MLflow URI or local fallback), applies the
same preprocessing logic used in the training pipeline, and serves both JSON
API endpoints and an HTML form UI.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.logger import logger

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class PredictionRequest(BaseModel):
    """Request payload for batch predictions.

    Attributes:
        instances: List of raw feature dictionaries.
    """

    instances: list[dict[str, Any]] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    """Response payload returned by the prediction endpoint."""

    predictions: list[Any]
    probabilities: list[float] | None = None
    positive_class_label: str | None = None


app = FastAPI(
    title="Loan Approval Predictor",
    description="Estimate loan approval outcomes from applicant details.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

model: Any = None
preprocessor: Any = None
feature_columns: list[str] = []
processed_feature_columns: list[str] = []
numeric_columns: list[str] = []
categorical_columns: list[str] = []
categorical_options: dict[str, list[str]] = {}
app_runtime_config: dict[str, Any] = {}


def _normalize_label_mapping(mapping: Any) -> dict[str, str]:
    """Normalize a mapping object into string-key/string-value dictionary."""

    if not isinstance(mapping, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in mapping.items():
        normalized[str(key).strip()] = str(value).strip()
    return normalized


def _default_label_mapping(target_col: str) -> dict[str, str]:
    """Return sensible defaults when explicit/inferred mapping is unavailable."""

    if target_col.strip().lower() == "loan_status":
        return {
            "0": "Approved",
            "0.0": "Approved",
            "1": "Rejected",
            "1.0": "Rejected",
        }
    return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML content from disk and return an empty dict when missing."""

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON content from disk and return an empty dict when missing."""

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(path_like: str | None, fallback: str) -> Path:
    """Resolve a path to absolute form using project root for relative values."""

    raw_value = path_like or fallback
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_runtime_config() -> dict[str, Any]:
    """Build runtime config from params with safe defaults and fallbacks."""

    params = _load_yaml(PROJECT_ROOT / "params.yaml")
    serving = params.get("serving", {})
    preprocessing = params.get("data_preprocessing", {})
    building = params.get("model_building", {})
    evaluation = params.get("model_evaluation", {})

    experiment_info_path = _resolve_path(
        serving.get("experiment_info_path"),
        evaluation.get("experiment_info_path", "reports/experiment_info.json"),
    )
    experiment_info = _load_json(experiment_info_path)

    # Prefer explicit serving model URI, then experiment metadata URI.
    model_uri = serving.get("model_uri") or experiment_info.get("model_uri")
    if not model_uri:
        run_id = experiment_info.get("run_id")
        model_path = experiment_info.get("model_path", evaluation.get("mlflow_model_name", "model"))
        if run_id:
            model_uri = f"runs:/{run_id}/{model_path}"

    target_col = serving.get("target_col") or preprocessing.get("target_col", "loan_status")
    # Keep inference behavior aligned with preprocessing configuration.
    drop_cols = serving.get("drop_cols")
    if drop_cols is None:
        drop_cols = preprocessing.get("drop_cols", ["loan_id"])
    if not isinstance(drop_cols, list):
        drop_cols = [drop_cols]

    config = {
        "tracking_uri": serving.get("tracking_uri") or evaluation.get("tracking_uri"),
        "model_uri": model_uri,
        "local_model_path": _resolve_path(
            serving.get("local_model_path"),
            building.get("model_output_path", "model/model.pkl"),
        ),
        "preprocessor_path": _resolve_path(
            serving.get("preprocessor_path"),
            os.path.join(preprocessing.get("model_path", "model"), "preprocessor.pkl"),
        ),
        "target_col": target_col,
        "drop_cols": [str(col).strip() for col in drop_cols],
        "raw_train_data_path": _resolve_path(
            serving.get("raw_train_data_path"),
            "data/raw/train.csv",
        ),
        "processed_train_data_path": _resolve_path(
            serving.get("processed_train_data_path"),
            "data/processed/train.csv",
        ),
        "label_mapping": _normalize_label_mapping(serving.get("label_mapping", {})),
    }
    return config


def _load_preprocessor(preprocessor_path: Path) -> Any:
    """Load the fitted preprocessing pipeline from a local pickle file."""

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

    with open(preprocessor_path, "rb") as file:
        return pickle.load(file)


def _load_model(model_uri: str | None, local_model_path: Path) -> Any:
    """Load model from MLflow first, then local pickle as fallback."""

    if model_uri:
        try:
            logger.info("Loading MLflow model with sklearn loader: %s", model_uri)
            return mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            logger.warning("sklearn loader failed for URI %s: %s", model_uri, exc)
            try:
                logger.info("Falling back to MLflow pyfunc loader: %s", model_uri)
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as second_exc:
                logger.warning("pyfunc loader failed for URI %s: %s", model_uri, second_exc)

    if local_model_path.exists():
        logger.info("Loading local model fallback: %s", local_model_path)
        with open(local_model_path, "rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(
        "Unable to load model from MLflow URI or local model fallback. "
        f"Checked local path: {local_model_path}"
    )


def _to_list(values: Any) -> list[Any]:
    """Normalize scalar/array-like prediction outputs into a Python list."""

    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list):
        return values
    return [values]


def _extract_preprocessor_schema(preproc: Any) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    """Extract input feature schema and categorical options from preprocessor."""

    input_features: list[str]
    if hasattr(preproc, "feature_names_in_"):
        input_features = [str(col) for col in preproc.feature_names_in_.tolist()]
    else:
        input_features = []

    num_cols: list[str] = []
    cat_cols: list[str] = []
    cat_options: dict[str, list[str]] = {}

    if hasattr(preproc, "transformers_"):
        for name, transformer, cols in preproc.transformers_:
            if name == "remainder":
                continue

            resolved_cols = [str(col) for col in cols]
            if name == "num":
                num_cols.extend(resolved_cols)
            elif name == "cat":
                cat_cols.extend(resolved_cols)
                if hasattr(transformer, "categories_"):
                    for col, categories in zip(resolved_cols, transformer.categories_):
                        cat_options[col] = [str(item).strip() for item in categories]

    if not input_features:
        input_features = num_cols + cat_cols

    return input_features, num_cols, cat_cols, cat_options


def _infer_label_mapping(config: dict[str, Any]) -> dict[str, str]:
    """Infer encoded-to-label mapping by aligning raw and processed train data."""

    raw_path = config["raw_train_data_path"]
    processed_path = config["processed_train_data_path"]
    target_col = config["target_col"]

    if not raw_path.exists() or not processed_path.exists():
        return {}

    try:
        raw_df = pd.read_csv(raw_path)
        processed_df = pd.read_csv(processed_path)
    except Exception:
        return {}

    raw_df.columns = raw_df.columns.str.strip()
    processed_df.columns = processed_df.columns.str.strip()

    if target_col not in raw_df.columns or target_col not in processed_df.columns:
        return {}
    if len(raw_df) != len(processed_df):
        return {}

    mapping_frame = pd.DataFrame(
        {
            "encoded": processed_df[target_col].astype(str),
            "label": raw_df[target_col].astype(str).str.strip(),
        }
    )

    mapping: dict[str, str] = {}
    for encoded, group in mapping_frame.groupby("encoded"):
        mode = group["label"].mode()
        if not mode.empty:
            label_value = str(mode.iloc[0])
            encoded_key = str(encoded).strip()
            mapping[encoded_key] = label_value

            try:
                # Add numeric variants so lookup works for int/float-like keys.
                numeric_key = float(encoded_key)
                mapping[str(int(numeric_key))] = label_value
                mapping[str(numeric_key)] = label_value
                if numeric_key.is_integer():
                    mapping[f"{int(numeric_key)}.0"] = label_value
            except ValueError:
                continue
    return mapping


def _resolve_prediction_label(prediction: Any, label_mapping: dict[str, str]) -> Any:
    """Convert encoded prediction into human-readable label when available."""

    candidates = [str(prediction).strip()]

    try:
        numeric_value = float(prediction)
        candidates.extend(
            [
                str(int(numeric_value)),
                str(numeric_value),
                f"{int(numeric_value)}.0",
            ]
        )
    except (TypeError, ValueError):
        pass

    for key in candidates:
        if key in label_mapping:
            return label_mapping[key]
    return prediction


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace for all object columns."""

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def _get_positive_class_label() -> str:
    """Return the label name associated with probability column index 1."""

    label_mapping = app_runtime_config.get("label_mapping", {})

    classes = _to_list(getattr(model, "classes_", []))
    if len(classes) > 1:
        resolved = _resolve_prediction_label(classes[1], label_mapping)
        return str(resolved)

    if "1" in label_mapping:
        return str(label_mapping["1"])

    return "Positive class"


def _build_input_frame(instances: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare raw input records to match training-time preprocessing flow."""

    if not instances:
        raise ValueError("instances must contain at least one item")

    input_df = pd.DataFrame(instances)
    input_df.columns = input_df.columns.str.strip()
    input_df = _strip_string_columns(input_df)

    target_col = str(app_runtime_config.get("target_col", "loan_status")).strip()
    drop_cols = [str(col).strip() for col in app_runtime_config.get("drop_cols", [])]

    # Drop target and configured columns if present in inbound payload.
    cols_to_drop = [col for col in [target_col] + drop_cols if col in input_df.columns]
    if cols_to_drop:
        input_df = input_df.drop(columns=cols_to_drop)

    missing_columns = [col for col in feature_columns if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    input_df = input_df[feature_columns].copy()

    for col in categorical_columns:
        input_df[col] = input_df[col].astype(str).str.strip()

    for col in numeric_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    invalid_numeric = [col for col in numeric_columns if input_df[col].isna().any()]
    if invalid_numeric:
        raise ValueError(f"Invalid numeric values in columns: {invalid_numeric}")

    return input_df


def _transform_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted preprocessor and return transformed features as DataFrame."""

    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return pd.DataFrame(transformed, columns=processed_feature_columns)


def _predict(instances: list[dict[str, Any]]) -> PredictionResponse:
    """Run full inference pipeline: validate, preprocess, predict, format output."""

    input_df = _build_input_frame(instances)
    transformed_df = _transform_input(input_df)

    raw_predictions = model.predict(transformed_df)
    predictions_list = _to_list(raw_predictions)

    label_mapping = app_runtime_config.get("label_mapping", {})
    display_predictions = [_resolve_prediction_label(pred, label_mapping) for pred in predictions_list]

    probabilities: list[float] | None = None
    positive_class_label: str | None = None
    if hasattr(model, "predict_proba"):
        positive_class_label = _get_positive_class_label()
        probas = model.predict_proba(transformed_df)
        probas_list = _to_list(probas)
        probabilities = []
        for row in probas_list:
            if isinstance(row, list) and len(row) > 1:
                probabilities.append(float(row[1]))
            elif isinstance(row, list) and len(row) == 1:
                probabilities.append(float(row[0]))
            else:
                probabilities.append(float(row))

    return PredictionResponse(
        predictions=display_predictions,
        probabilities=probabilities,
        positive_class_label=positive_class_label,
    )


@app.on_event("startup")
def startup_event() -> None:
    """Initialize app resources at startup.

    Loads runtime config, model, preprocessor, feature schema, and label mapping.
    """

    global model
    global preprocessor
    global feature_columns
    global processed_feature_columns
    global numeric_columns
    global categorical_columns
    global categorical_options
    global app_runtime_config

    app_runtime_config = _load_runtime_config()
    tracking_uri = app_runtime_config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    preprocessor = _load_preprocessor(app_runtime_config["preprocessor_path"])
    model = _load_model(
        model_uri=app_runtime_config.get("model_uri"),
        local_model_path=app_runtime_config["local_model_path"],
    )

    (
        feature_columns,
        numeric_columns,
        categorical_columns,
        categorical_options,
    ) = _extract_preprocessor_schema(preprocessor)

    processed_feature_columns = [str(col) for col in preprocessor.get_feature_names_out()]

    inferred_mapping = _infer_label_mapping(app_runtime_config)
    configured_mapping = app_runtime_config.get("label_mapping", {})
    fallback_mapping = _default_label_mapping(str(app_runtime_config.get("target_col", "loan_status")))

    # Priority: configured mapping > inferred mapping > fallback defaults.
    app_runtime_config["label_mapping"] = {
        **fallback_mapping,
        **inferred_mapping,
        **configured_mapping,
    }

    logger.info("App startup complete with %d features.", len(feature_columns))


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint for simple liveness checks."""

    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return feature schema used by the frontend form and API clients."""

    return {
        "features": feature_columns,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "categorical_options": categorical_options,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict loan outcome from one or more input records."""

    try:
        return _predict(payload.instances)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the web UI with fields derived from the preprocessor schema."""

    fields = []
    for col in feature_columns:
        if col in numeric_columns:
            fields.append({"name": col, "type": "number", "options": []})
        else:
            fields.append({"name": col, "type": "select", "options": categorical_options.get(col, [])})

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": fields,
        },
    )
