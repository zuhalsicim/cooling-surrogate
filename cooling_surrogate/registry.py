"""Model registry on MLflow: one tracked run per training, one registered version per candidate.

Store: a local SQLite database plus an artifact folder, both under `state/`. Pointing
`tracking_uri` at a server (or S3 for artifacts) changes nothing else in the code.

Promotion uses an MLflow alias. `production` is moved only by `promote`, and the pipeline calls
`promote` only for a version whose gate report says "promote". Each version carries its lineage as
tags: dataset version and content hash, the weights' SHA-256, and the gate decision.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")  # MLflow's INFO lines drown the pipeline's output

import mlflow
from mlflow import MlflowClient

from .data import sha256_file

MODEL_NAME = "cooling-surrogate"
PRODUCTION = "production"


class Registry:
    def __init__(self, state_dir: Path):
        state_dir = state_dir.resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        self.uri = "sqlite:///" + (state_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_registry_uri(self.uri)
        self.client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        experiment = self.client.get_experiment_by_name(MODEL_NAME)
        if experiment is None:
            artifacts = (state_dir / "artifacts").as_uri()
            self.experiment_id = self.client.create_experiment(MODEL_NAME, artifact_location=artifacts)
        else:
            self.experiment_id = experiment.experiment_id
        if not self.client.search_registered_models(f"name='{MODEL_NAME}'"):
            self.client.create_registered_model(MODEL_NAME)

    def log_run(self, name: str, params: dict, metrics: dict, tags: dict, model_dir: Path, history=()) -> str:
        """One MLflow run with the training config, the loss curve, the lineage tags and the model files."""
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=name) as run:
            mlflow.log_params({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()})
            for step, values in history:
                mlflow.log_metrics(values, step=step)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            return run.info.run_id

    def register(self, run_id: str, model_dir: Path, tags: dict) -> str:
        tags = {**tags, "weights_sha256": sha256_file(model_dir / "weights.pt")}
        mv = self.client.create_model_version(
            MODEL_NAME, source=f"runs:/{run_id}/model", run_id=run_id, tags={k: str(v) for k, v in tags.items()}
        )
        return str(mv.version)

    def tags(self, version: str) -> dict[str, str]:
        return dict(self.client.get_model_version(MODEL_NAME, version).tags)

    def set_tags(self, version: str, tags: dict) -> None:
        for key, value in tags.items():
            self.client.set_model_version_tag(MODEL_NAME, version, key, str(value))

    def download(self, version: str, dest: Path) -> Path:
        """Copy a version's model files to `dest` and check the weights against the registered hash."""
        mv = self.client.get_model_version(MODEL_NAME, version)
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(dest), tracking_uri=self.uri)
        folder = dest / "model"
        if sha256_file(folder / "weights.pt") != mv.tags["weights_sha256"]:
            raise ValueError(f"version {version}: weights do not match the registered SHA-256")
        return folder

    def promote(self, version: str) -> None:
        self.client.set_registered_model_alias(MODEL_NAME, PRODUCTION, version)

    def production_version(self) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(MODEL_NAME, PRODUCTION).version)
        except mlflow.exceptions.MlflowException:
            return None

    def versions(self) -> list[dict]:
        out = []
        for found in self.client.search_model_versions(f"name='{MODEL_NAME}'"):
            mv = self.client.get_model_version(MODEL_NAME, found.version)  # search leaves aliases empty
            out.append({"version": str(mv.version), "aliases": list(mv.aliases), "tags": dict(mv.tags)})
        return sorted(out, key=lambda v: int(v["version"]))
