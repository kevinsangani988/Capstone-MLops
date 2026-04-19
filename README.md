# Capstone MLOps: Loan Approval Prediction

End-to-end MLOps project for loan approval prediction using DVC, MLflow, and FastAPI.

The project supports:
- data ingestion from Azure Blob or CSV URL
- preprocessing with a saved transformer
- model training and evaluation
- MLflow logging and model registration
- a FastAPI app with API and web UI for inference

## Problem Statement

Predict whether a loan application is likely to be Approved or Rejected based on applicant and asset details.

## Tech Stack

- Python
- scikit-learn
- DVC
- MLflow
- FastAPI
- Jinja2
- Azure Blob Storage

## Repository Structure

```text
Capstone-MLops/
|-- data/
|   |-- raw/
|   `-- processed/
|-- model/
|   |-- model.pkl
|   `-- preprocessor.pkl
|-- reports/
|   |-- metrics.json
|   `-- experiment_info.json
|-- src/
|   |-- api/main.py
|   |-- blob/__init__.py
|   |-- data/data_ingestion.py
|   |-- data/data_preprocessing.py
|   |-- model/model_building.py
|   |-- model/model_evaluation.py
|   `-- model/register_model.py
|-- dvc.yaml
|-- params.yaml
|-- requirements.txt
`-- README.md
```

## Pipeline Stages (DVC)

Defined in dvc.yaml:

1. data_ingestion
    - Reads source data from Blob or CSV URL
    - Splits into train and test
    - Outputs: data/raw/train.csv, data/raw/test.csv

2. data_preprocessing
    - Applies preprocessing (scaling + one-hot encoding)
    - Encodes target variable
    - Saves preprocessor object
    - Outputs: data/processed/train.csv, data/processed/test.csv, model/preprocessor.pkl

3. model_building
    - Trains RandomForestClassifier
    - Output: model/model.pkl

4. model_evaluation
    - Evaluates model on test set
    - Logs metrics and model to MLflow
    - Outputs: reports/metrics.json, reports/experiment_info.json

5. register_model
    - Registers model in MLflow Model Registry

## Configuration

Main configuration is in params.yaml.

Important sections:
- data_ingestion
- data_preprocessing
- model_building
- model_evaluation
- model_registration
- serving

Switch ingestion source:
- source_type: blob
- source_type: csv

If using CSV source, set data_url in params.yaml.
If using Blob source, set conn_str, container, and blob_name.

## Installation

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the Training Pipeline

### Option A: Run complete pipeline with DVC

```powershell
dvc repro
```

### Option B: Run stages manually

```powershell
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

## Run the FastAPI App

```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open in browser:
- http://127.0.0.1:8000

## API Endpoints

- GET / : Web UI
- GET /health : Health check
- GET /schema : Expected feature schema
- POST /predict : Batch prediction

### Example Prediction Request

```json
{
  "instances": [
     {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 6500000,
        "loan_amount": 15000000,
        "loan_term": 12,
        "cibil_score": 780,
        "residential_assets_value": 9000000,
        "commercial_assets_value": 2500000,
        "luxury_assets_value": 14000000,
        "bank_asset_value": 4500000
     }
  ]
}
```

### Example Prediction Response

```json
{
  "predictions": ["Approved"],
  "probabilities": [0.04],
  "positive_class_label": "Rejected"
}
```

Interpretation:
- Probability of Rejected = 4%
- Therefore prediction is Approved

## Current Evaluation Metrics

From reports/metrics.json:

- accuracy: 0.9813
- precision: 0.9744
- recall: 0.9744
- auc: 0.9988

## Notes

- The web UI is schema-driven from the saved preprocessor.
- Inference preprocessing follows the same flow used in src/data/data_preprocessing.py.
- Browser requests to /favicon.ico may return 404 unless a favicon is added. This does not affect app functionality.

## Future Improvements

- Add automated tests for API endpoints
- Add containerization (Docker)
- Add CI/CD deployment workflow
- Add input drift and model monitoring

## License

This project is licensed under the MIT License.
