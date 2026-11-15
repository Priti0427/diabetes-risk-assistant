"""Compare MLflow runs with mlflow.search_runs() and export the best pipeline.

Usage
-----
    python scripts/compare_runs.py            # rank runs, save best to models/best_model.joblib
    python scripts/compare_runs.py --no-save  # just print the leaderboard

Ranking: highest `selection.primary_metric` (ROC-AUC by default), ties broken by
`selection.secondary_metric` (F1). ROC-AUC is threshold-independent and robust to
the 65/35 class imbalance; F1 keeps us honest about the positive (diabetic) class.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.train import load_config, setup_mlflow, _resolve  # noqa: E402

DISPLAY_METRICS = ["roc_auc", "f1", "accuracy", "precision", "recall"]


def leaderboard(config: dict) -> pd.DataFrame:
    """Query every finished run in the experiment and rank them."""
    primary = f"metrics.{config['selection']['primary_metric']}"
    secondary = f"metrics.{config['selection']['secondary_metric']}"
    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"{primary} DESC", f"{secondary} DESC"],
    )
    if runs.empty:
        raise SystemExit("No finished runs found. Train first: python -m src.train --all")

    cols = ["run_id", "tags.mlflow.runName", "params.model_type"] + [f"metrics.{m}" for m in DISPLAY_METRICS]
    cols = [c for c in cols if c in runs.columns]
    board = runs[cols].rename(
        columns={"tags.mlflow.runName": "run_name", "params.model_type": "model", **{f"metrics.{m}": m for m in DISPLAY_METRICS}}
    )
    board.insert(0, "rank", range(1, len(board) + 1))
    return board.reset_index(drop=True)


def export_best(board: pd.DataFrame, config: dict) -> Path:
    best = board.iloc[0]
    run_id = best["run_id"]
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    out = _resolve(config["selection"]["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    meta = {
        "run_id": run_id,
        "run_name": best["run_name"],
        "model": best["model"],
        "selected_by": config["selection"]["primary_metric"],
        **{m: float(best[m]) for m in DISPLAY_METRICS if m in best},
    }
    (out.parent / "best_run.json").write_text(json.dumps(meta, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_mlflow(config)
    board = leaderboard(config)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(board.drop(columns=["run_id"]).to_string(index=False))

    best = board.iloc[0]
    print(
        f"\nBest run: {best['run_name']} ({best['model']}) "
        f"{config['selection']['primary_metric']}={best[config['selection']['primary_metric']]:.4f}  run_id={best['run_id']}"
    )
    if not args.no_save:
        out = export_best(board, config)
        print(f"Exported best pipeline to {out}")


if __name__ == "__main__":
    main()
