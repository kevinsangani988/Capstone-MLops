# Capstone MLOps: Loan Approval Prediction

An end-to-end MLOps project that trains, evaluates, registers, serves, containers, and deploys a loan approval prediction model.

This repository combines:

- data ingestion from Azure Blob Storage or a CSV URL
- DVC pipeline orchestration
- feature preprocessing and model training (scikit-learn)
- MLflow + DagsHub experiment/model tracking
- FastAPI inference API with UI
- Docker containerization
- GitHub Actions CI/CD
- AWS ECR + EKS deployment

## 1. Business Problem

Predict loan outcome (Approved or Rejected) from applicant profile and asset details.

Target column: loan_status

## 2. High-Level Flow

1. Ingest raw data from Azure Blob or CSV URL.
2. Split train and test data.
3. Preprocess features (scaling + one-hot encoding) and encode target.
4. Train RandomForest model.
5. Evaluate model and log metrics/artifacts to MLflow.
6. Register model version in model registry.
7. Serve predictions via FastAPI API and UI.
8. Containerize and deploy to AWS EKS.

## 3. Tech Stack

- Python 3.10+
- scikit-learn, pandas, numpy
- DVC
- MLflow
- DagsHub
- FastAPI + Uvicorn + Jinja2
- Azure Blob Storage SDK
- Docker
- AWS ECR/EKS, kubectl, eksctl
- GitHub Actions

## 4. Repository Structure

```text
Capstone-MLops/
|-- .github/workflows/ci.yaml
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
|   |-- api/
|   |   |-- main.py
|   |   |-- requirements.txt
|   |   |-- static/
|   |   `-- templates/
|   |-- blob/__init__.py
|   |-- data/data_ingestion.py
|   |-- data/data_preprocessing.py
|   |-- model/model_building.py
|   |-- model/model_evaluation.py
|   `-- model/register_model.py
|-- tests/
|-- deployment.yaml
|-- Dockerfile
|-- dvc.yaml
|-- params.yaml
|-- requirements.txt
`-- README.md
```

## 5. DVC Pipeline Stages

Defined in dvc.yaml.

1. data_ingestion
    - Command: python -m src.data.data_ingestion
    - Inputs: params + ingestion code
    - Outputs: data/raw/train.csv, data/raw/test.csv

2. data_preprocessing
    - Command: python -m src.data.data_preprocessing
    - Inputs: raw train/test + params
    - Outputs: data/processed/train.csv, data/processed/test.csv, model/preprocessor.pkl

3. model_building
    - Command: python -m src.model.model_building
    - Inputs: processed train + params
    - Outputs: model/model.pkl

4. model_evaluation
    - Command: python -m src.model.model_evaluation
    - Inputs: model + processed test + params
    - Outputs: reports/metrics.json, reports/experiment_info.json

5. register_model
    - Command: python -m src.model.register_model
    - Inputs: experiment info + params
    - Action: register/promote model version in MLflow registry

## 6. Configuration

All runtime config lives in params.yaml.

Main sections:

- data_ingestion
- data_preprocessing
- model_building
- model_evaluation
- model_registration
- serving

### Data source switch

In data_ingestion.source_type:

- blob -> reads from Azure Blob Storage
- csv -> reads from data_url

### Azure connection variables (for blob mode)

The ingestion code checks these env vars in order:

1. CONN_STRING
2. AZURE_STORAGE_CONNECTION_STRING
3. CAPSTONE_TEST

## 7. Local Setup

### Prerequisites

- Python 3.10+
- Git
- DVC
- Docker (optional, for container run)

### Create environment

Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run tests

```powershell
python -m pytest -q tests
```

## 8. Run Training Pipeline

### Recommended: full DVC run

```powershell
dvc repro
```

### Stage-by-stage (manual)

```powershell
python -m src.data.data_ingestion
python -m src.data.data_preprocessing
python -m src.model.model_building
python -m src.model.model_evaluation
python -m src.model.register_model
```

## 9. Model Tracking and Registry

### Evaluation tracking

- Tracking URI comes from params.yaml (model_evaluation.tracking_uri).
- If DagsHub token is present, logs go to remote MLflow.
- If token is missing, code falls back to local sqlite:///mlflow.db.

### Registration tracking

- Registration expects valid DagsHub/MLflow credentials.
- Registration logic tries multiple model URI candidates and retries transient remote errors.

### Useful env vars for MLflow/DagsHub

- DAGSHUB_TOKEN
- DAGSHUB_USER_TOKEN
- MLFLOW_TRACKING_PASSWORD

## 10. Run API Locally

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- http://127.0.0.1:8000 (UI)
- http://127.0.0.1:8000/docs (Swagger)

### API Endpoints

- GET / -> HTML UI
- GET /health -> liveness
- GET /schema -> expected feature schema
- POST /predict -> batch inference

### Example request

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

### Example response

```json
{
  "predictions": ["Approved"],
  "probabilities": [0.04],
  "positive_class_label": "Rejected"
}
```

The returned probability is the probability of the positive class label shown in positive_class_label.

## 11. Run with Docker

Build image:

```powershell
docker build -t capstone-mlops:latest -f Dockerfile .
```

Run container:

```powershell
docker run -p 8000:8000 capstone-mlops:latest
```

Access app:

- http://localhost:8000

## 12. Deploy to AWS (ECR + EKS)

### 12.1 Create/verify EKS cluster

Use a recent eksctl and explicit supported Kubernetes version:

```powershell
eksctl version
eksctl create cluster --name fastapi-app-cluster --region us-east-1 --version 1.30 --nodegroup-name flask-app-nodes --node-type t3.small --nodes 1 --nodes-min 1 --nodes-max 1 --managed
```

If you previously ran with old defaults and failed, clean up first:

```powershell
eksctl delete cluster --region us-east-1 --name fastapi-app-cluster
```

### 12.2 Push image to ECR

```powershell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t capstone-mlops:latest .
docker tag capstone-mlops:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/capstone-mlops:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/capstone-mlops:latest
```

### 12.3 Deploy Kubernetes manifests

```powershell
aws eks update-kubeconfig --region us-east-1 --name fastapi-app-cluster
kubectl create secret generic capstone-secret --from-literal=CAPSTONE_TEST=<your-value> --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get svc capstone-mlops-service
```

For LoadBalancer service, use the EXTERNAL-IP/hostname from kubectl get svc.

## 13. CI/CD (GitHub Actions)

Workflow file: .github/workflows/ci.yaml

Current pipeline does:

1. Install dependencies
2. Run tests
3. Validate blob secret format
4. Run dvc repro
5. Build and push Docker image to ECR
6. Update kubeconfig and deploy to EKS

### Required GitHub Secrets

- AZURE_STORAGE_CONNECTION_STRING
- DAGSHUB_TOKEN
- CAPSTONE_TEST
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- AWS_ACCOUNT_ID
- ECR_REPOSITORY

Important:

- Ensure AWS keys belong to an IAM user/role with EKS and ECR access.
- The workflow currently uses a hardcoded cluster name in update-kubeconfig. If your cluster name differs, update that step in ci.yaml.

## 14. Current Model Metrics

From reports/metrics.json:

- Accuracy: 0.9813
- Precision: 0.9744
- Recall: 0.9744
- AUC: 0.9988

## 15. Troubleshooting

### Error: unsupported Kubernetes version 1.25

- Cause: old eksctl defaulting to unsupported version.
- Fix: upgrade eksctl and pass --version explicitly.

### Error: UnauthorizedOperation / AccessDenied in AWS

- Verify active identity:

```powershell
aws sts get-caller-identity
```

- Ensure the intended IAM user/profile is active.
- Ensure permissions boundary/SCP allows required EKS/EC2 actions.

### API always returns Rejected

- Check input values and schema via GET /schema.
- Review returned probabilities and positive_class_label.
- Validate model and preprocessor artifacts exist.

### Docker app not reachable on localhost

- Wait for startup logs to show app startup complete.
- Check container is running and port mapping is correct.

## 16. Public Release Checklist

Before making repository public:

1. Remove hardcoded credentials/tokens (if any).
2. Keep secrets only in environment variables or GitHub Secrets.
3. Verify .gitignore excludes local virtual environments and sensitive files.
4. Ensure README instructions are reproducible.
5. Add architecture diagram and screenshots (optional but recommended).

## 17. License

This project is licensed under the MIT License.
