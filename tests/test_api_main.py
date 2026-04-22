import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main


class _PreprocessorMock:
    def transform(self, df):
        assert list(df.columns) == ["f_num", "f_cat"]
        assert df["f_num"].iloc[0] == 5
        assert df["f_cat"].iloc[0] == "A"
        return np.array([[5.0, 1.0, 0.0]])

    def get_feature_names_out(self):
        return np.array(["num__f_num", "cat__f_cat_A", "cat__f_cat_B"])


class _ModelMock:
    classes_ = np.array([0, 1])

    def predict(self, _):
        return np.array([1])

    def predict_proba(self, _):
        return np.array([[0.2, 0.8]])


def _configure_prediction_runtime():
    api_main.feature_columns = ["f_num", "f_cat"]
    api_main.numeric_columns = ["f_num"]
    api_main.categorical_columns = ["f_cat"]
    api_main.processed_feature_columns = ["num__f_num", "cat__f_cat_A", "cat__f_cat_B"]
    api_main.preprocessor = _PreprocessorMock()
    api_main.model = _ModelMock()
    api_main.app_runtime_config = {
        "target_col": "loan_status",
        "drop_cols": ["loan_id"],
        "label_mapping": {"0": "Approved", "1": "Rejected", "1.0": "Rejected"},
    }


def test_predict_pipeline_maps_label_and_probability():
    _configure_prediction_runtime()

    payload = [
        {
            "loan_id": 10,
            "loan_status": "Approved",
            "f_num": "5",
            "f_cat": " A ",
        }
    ]

    result = api_main._predict(payload)

    assert result.predictions == ["Rejected"]
    assert result.probabilities == [0.8]
    assert result.positive_class_label == "Rejected"


def test_build_input_frame_rejects_invalid_numeric_values():
    _configure_prediction_runtime()

    with pytest.raises(ValueError, match="Invalid numeric values"):
        api_main._build_input_frame([{"f_num": "abc", "f_cat": "A"}])


@pytest.fixture
def client_without_startup():
    original_startup_handlers = list(api_main.app.router.on_startup)
    api_main.app.router.on_startup.clear()
    try:
        with TestClient(api_main.app) as client:
            yield client
    finally:
        api_main.app.router.on_startup[:] = original_startup_handlers


def test_health_endpoint(client_without_startup):
    response = client_without_startup.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_response_payload(client_without_startup, monkeypatch):
    expected = api_main.PredictionResponse(
        predictions=["Approved"],
        probabilities=[0.1],
        positive_class_label="Rejected",
    )

    monkeypatch.setattr(api_main, "_predict", lambda _: expected)

    response = client_without_startup.post("/predict", json={"instances": [{"a": 1}]})

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == ["Approved"]
    assert body["probabilities"] == [0.1]
    assert body["positive_class_label"] == "Rejected"


def _extract_metric_value(metrics_text: str, metric_name: str) -> float:
    for line in metrics_text.splitlines():
        if line.startswith(f"{metric_name} "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"Metric '{metric_name}' not found in /metrics output")


def test_metrics_endpoint_exposes_prometheus_text(client_without_startup):
    response = client_without_startup.get("/metrics")

    assert response.status_code == 200
    assert "capstone_prediction_requests_total" in response.text
    assert "capstone_prediction_latency_seconds_avg" in response.text


def test_predict_updates_metrics(client_without_startup, monkeypatch):
    expected = api_main.PredictionResponse(
        predictions=["Approved"],
        probabilities=[0.1],
        positive_class_label="Rejected",
    )
    monkeypatch.setattr(api_main, "_predict", lambda _: expected)

    before = client_without_startup.get("/metrics")
    before_count = _extract_metric_value(before.text, "capstone_prediction_requests_total")

    predict_response = client_without_startup.post("/predict", json={"instances": [{"a": 1}]})
    assert predict_response.status_code == 200

    after = client_without_startup.get("/metrics")
    after_count = _extract_metric_value(after.text, "capstone_prediction_requests_total")
    latency_count = _extract_metric_value(after.text, "capstone_prediction_latency_seconds_count")

    assert after_count == before_count + 1
    assert latency_count == after_count
