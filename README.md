
---

# Capstone MLOps: Loan Approval Prediction

An end-to-end MLOps project that trains, evaluates, registers, serves, containers, and deploys a loan approval prediction model.

## Overview

This project implements a complete machine learning lifecycle, including:

* Data ingestion from Azure Blob Storage or CSV
* DVC pipeline orchestration
* Feature preprocessing and model training (scikit-learn)
* MLflow + DagsHub experiment tracking
* FastAPI-based inference API with UI
* Docker containerization
* CI/CD with GitHub Actions
* Deployment on AWS ECR and EKS

---

## Business Problem

Predict loan approval outcome (**Approved** or **Rejected**) based on applicant profile and financial attributes.

* **Target column:** `loan_status`

---

## High-Level Workflow

1. Ingest raw data from Azure Blob or CSV
2. Split into training and testing datasets
3. Perform preprocessing (scaling + encoding)
4. Train a RandomForest model
5. Evaluate and log metrics to MLflow
6. Register model in the registry
7. Serve predictions via FastAPI API
8. Containerize and deploy to AWS EKS

---

## Tech Stack

* Python 3.10+
* scikit-learn, pandas, numpy
* DVC
* MLflow
* DagsHub
* FastAPI, Uvicorn, Jinja2
* Azure Blob Storage SDK
* Docker
* AWS ECR & EKS
* GitHub Actions

---

## Repository Structure

```text
Capstone-MLops/
│
├── .github/workflows/ci.yaml
├── data/
│   ├── raw/
│   └── processed/
├── model/
│   ├── model.pkl
│   └── preprocessor.pkl
├── reports/
│   ├── metrics.json
│   └── experiment_info.json
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── static/
│   │   └── templates/
│   ├── blob/
│   ├── data/
│   └── model/
├── tests/
├── deployment.yaml
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

## DVC Pipeline Stages

Defined in `dvc.yaml`:

1. **data_ingestion**

   * Outputs: raw train/test datasets

2. **data_preprocessing**

   * Outputs: processed datasets + preprocessor

3. **model_building**

   * Outputs: trained model

4. **model_evaluation**

   * Outputs: metrics and experiment metadata

5. **register_model**

   * Registers model in MLflow registry

---

## Configuration

All runtime configuration is managed in `params.yaml`.

### Data Source Options

* `blob` → Azure Blob Storage
* `csv` → External CSV URL

### Environment Variables (for blob mode)

* `CONN_STRING`
* `AZURE_STORAGE_CONNECTION_STRING`
* `CAPSTONE_TEST`

---

## Local Setup

### Prerequisites

* Python 3.10+
* Git
* DVC
* Docker (optional)

### Setup Environment

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Tests

```powershell
python -m pytest -q tests
```

---

## Training Pipeline

### Full Pipeline (Recommended)

```powershell
dvc repro
```

### Manual Execution

```powershell
python -m src.data.data_ingestion
python -m src.data.data_preprocessing
python -m src.model.model_building
python -m src.model.model_evaluation
python -m src.model.register_model
```

---

## Model Tracking & Registry

* MLflow is used for experiment tracking
* Supports both local and remote tracking
* Model registration handled via MLflow registry

---

## Run API Locally

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access

* UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoints

* `GET /` → UI
* `GET /health` → Health check
* `GET /schema` → Input schema
* `POST /predict` → Predictions
* `GET /metrics` → Prometheus metrics

---

## Example Request

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

### Example Response

```json
{
  "predictions": ["Approved"],
  "probabilities": [0.04],
  "positive_class_label": "Rejected"
}
```

---

## Docker

### Build Image

```powershell
docker build -t capstone-mlops:latest .
```

### Run Container

```powershell
docker run -p 8000:8000 capstone-mlops:latest
```

---

## Deployment (AWS ECR + EKS)

### Create Cluster

```powershell
eksctl create cluster \
  --name fastapi-app-cluster \
  --region us-east-1 \
  --version 1.30 \
  --nodegroup-name nodes \
  --node-type t3.small \
  --nodes 1
```

### Push Image

```powershell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t capstone-mlops .
docker tag capstone-mlops:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/capstone-mlops:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/capstone-mlops:latest
```

### Deploy

```powershell
aws eks update-kubeconfig --region us-east-1 --name fastapi-app-cluster
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get svc
```

---

## CI/CD

Pipeline includes:

1. Dependency installation
2. Test execution
3. DVC pipeline run
4. Docker build & push
5. Deployment to EKS

---

## Model Performance

* Accuracy: 0.9813
* Precision: 0.9744
* Recall: 0.9744
* AUC: 0.9988

---

## Troubleshooting

### Kubernetes Version Error

Use a supported version (e.g., 1.30).

### AWS Access Issues

Verify active identity:

```powershell
aws sts get-caller-identity
```

### API Issues

* Check `/schema`
* Validate input format
* Ensure model artifacts exist

### Docker Issues

* Ensure container is running
* Verify port mapping

---

## License

This project is licensed under the MIT License.

---