from types import SimpleNamespace

import pytest

import src.model.model_evaluation as model_evaluation
import src.model.register_model as register_model


def test_candidate_model_uris_deduplicates_and_includes_logged_models(monkeypatch):
    class _Client:
        def get_run(self, _run_id):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="42"))

        def search_logged_models(self, experiment_ids, filter_string):
            assert experiment_ids == ["42"]
            assert "source_run_id='run-123'" in filter_string
            return [
                SimpleNamespace(model_id="abc", model_uri="models:/abc"),
                SimpleNamespace(model_id="abc", model_uri="models:/abc"),
            ]

    monkeypatch.setattr(register_model.mlflow.tracking, "MlflowClient", lambda: _Client())

    uris = register_model._candidate_model_uris(
        {"run_id": "run-123", "model_path": "model", "model_uri": "models:/abc"}
    )

    assert uris == ["models:/abc", "runs:/run-123/model"]


def test_register_model_tries_multiple_uris_and_transitions_stage(monkeypatch):
    monkeypatch.setattr(register_model, "_candidate_model_uris", lambda _: ["bad-uri", "good-uri"])

    calls = {}

    def _register(uri, name):
        if uri == "bad-uri":
            raise RuntimeError("cannot register bad-uri")
        calls["registered"] = (uri, name)
        return SimpleNamespace(version="7")

    class _Client:
        def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
            calls["transition"] = (name, version, stage, archive_existing_versions)

    monkeypatch.setattr(register_model.mlflow, "register_model", _register)
    monkeypatch.setattr(register_model.mlflow.tracking, "MlflowClient", lambda: _Client())

    register_model.register_model("loan_model", {"run_id": "x"}, stage="Production")

    assert calls["registered"] == ("good-uri", "loan_model")
    assert calls["transition"] == ("loan_model", "7", "Production", True)


def test_register_model_raises_when_no_candidate_uri(monkeypatch):
    monkeypatch.setattr(register_model, "_candidate_model_uris", lambda _: [])
    with pytest.raises(ValueError, match="No valid model URI"):
        register_model.register_model("loan_model", {})


def test_promote_model_version_falls_back_and_archives_old_versions():
    calls = []

    class _Client:
        def __init__(self):
            self._first_call = True

        def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
            if self._first_call and archive_existing_versions:
                self._first_call = False
                raise TypeError("archive_existing_versions not supported")
            calls.append((name, str(version), stage, archive_existing_versions))

        def get_latest_versions(self, name, stages):
            assert name == "loan_model"
            assert stages == ["Production"]
            return [SimpleNamespace(version="7"), SimpleNamespace(version="6")]

    client = _Client()
    register_model._promote_model_version(
        client=client,
        model_name="loan_model",
        version="7",
        stage="Production",
        archive_existing_versions=True,
    )

    assert ("loan_model", "7", "Production", False) in calls
    assert ("loan_model", "6", "Archived", False) in calls


def test_model_evaluation_tracking_falls_back_to_local_in_ci(monkeypatch):
    captured = {}

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("DAGSHUB_TOKEN", "token")
    monkeypatch.setattr(model_evaluation.mlflow, "set_tracking_uri", lambda uri: captured.setdefault("uri", uri))
    monkeypatch.setattr(model_evaluation.dagshub, "init", lambda **_: (_ for _ in ()).throw(AssertionError("should not call dagshub.init in CI fallback")))

    model_evaluation.setup_tracking(
        {
            "tracking_uri": "https://dagshub.com/owner/repo.mlflow",
            "repo_owner": "owner",
            "repo_name": "repo",
        }
    )

    assert captured["uri"] == "sqlite:///mlflow.db"


def test_model_evaluation_tracking_uses_remote_when_token_available(monkeypatch):
    captured = {}

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("DAGSHUB_TOKEN", "token")
    monkeypatch.setattr(model_evaluation.mlflow, "set_tracking_uri", lambda uri: captured.setdefault("uri", uri))
    monkeypatch.setattr(model_evaluation.dagshub, "init", lambda **kwargs: captured.setdefault("init", kwargs))

    model_evaluation.setup_tracking(
        {
            "tracking_uri": "https://dagshub.com/owner/repo.mlflow",
            "repo_owner": "owner",
            "repo_name": "repo",
        }
    )

    assert captured["uri"] == "https://dagshub.com/owner/repo.mlflow"
    assert captured["init"] == {"repo_owner": "owner", "repo_name": "repo", "mlflow": True}
