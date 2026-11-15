"""Train one or all model configurations from configs/config.yaml and log to MLflow.

Usage
-----
    python -m src.train                       # trains `default_experiment`
    python -m src.train --experiment rf_default
    python -m src.train --all                 # every entry under `experiments`

Each run logs: all model hyperparameters, preprocessing settings, the data
description/version (row count + SHA-256 of the CSV), every evaluation metric on
the held-out test set, and the full preprocess+model pipeline as an artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_pipeline, format_metrics, global_feature_importance
from src.models import build_model
from src.preprocess import FEATURE_COLUMNS, load_data, preprocess_for_training

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def setup_mlflow(config: dict) -> None:
    """Point MLflow at the SQLite tracking DB and artifact folder named in the config.

    Relative paths are resolved against the repository root so the scripts work
    from any working directory.
    """
    mlflow_cfg = config["mlflow"]
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str(_resolve(tracking_uri[len("sqlite:///"):]))
    elif "://" not in tracking_uri:
        tracking_uri = _resolve(tracking_uri).as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    name = mlflow_cfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        artifact_location = _resolve(mlflow_cfg.get("artifact_location", "mlruns")).as_uri()
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def train_experiment(name: str, config: dict, save_local: bool = False) -> dict:
    """Train a single named experiment, log it to MLflow, and return a summary dict."""
    exp_cfg = config["experiments"][name]
    data_cfg = config["data"]
    data_path = _resolve(data_cfg["path"])

    df = load_data(data_path)
    preprocessor, X_train, X_test, y_train, y_test = preprocess_for_training(df, config)
    model = build_model(exp_cfg["model"], exp_cfg.get("params"), random_state=data_cfg["random_state"])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=name) as run:
        # --- parameters -------------------------------------------------- #
        mlflow.log_param("experiment", name)
        mlflow.log_param("model_type", exp_cfg["model"])
        mlflow.log_params({f"model__{k}": v for k, v in (exp_cfg.get("params") or {}).items()})
        mlflow.log_params({f"prep__{k}": v for k, v in config["preprocessing"].items()})
        mlflow.log_params(
            {
                "test_size": data_cfg["test_size"],
                "random_state": data_cfg["random_state"],
                "n_features": len(FEATURE_COLUMNS),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
        # --- data version ------------------------------------------------ #
        mlflow.set_tags(
            {
                "data.description": data_cfg["description"],
                "data.path": str(data_cfg["path"]),
                "data.sha256": _file_sha256(data_path),
                "data.rows": str(len(df)),
                "python": platform.python_version(),
            }
        )

        # --- fit + evaluate --------------------------------------------- #
        pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        importance = global_feature_importance(pipeline)
        if importance:
            mlflow.log_dict(importance, "feature_importance.json")

        # --- artifact ---------------------------------------------------- #
        # cloudpickle is needed because the pipeline contains our custom
        # DiabetesPreprocessor, which MLflow's default skops format refuses to load.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=X_test.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        summary = {"run_id": run.info.run_id, "experiment": name, "model": exp_cfg["model"], **metrics}
        print(f"[{name:16s}] {format_metrics(metrics)}  run_id={run.info.run_id}")

    if save_local:
        out = _resolve(config["selection"]["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out)
        (out.parent / "best_run.json").write_text(json.dumps(summary, indent=2))
        print(f"Saved pipeline to {out}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment", help="Name of an entry under `experiments` in the config")
    parser.add_argument("--all", action="store_true", help="Train every experiment in the config")
    parser.add_argument("--save", action="store_true", help="Also save the pipeline to selection.output_path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_mlflow(config)

    if args.all:
        names = list(config["experiments"])
    else:
        names = [args.experiment or config["default_experiment"]]

    for name in names:
        if name not in config["experiments"]:
            raise SystemExit(f"Unknown experiment {name!r}. Available: {list(config['experiments'])}")
        train_experiment(name, config, save_local=args.save and len(names) == 1)


if __name__ == "__main__":
    main()
