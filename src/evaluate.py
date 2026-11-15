"""Evaluation utilities: metrics, confusion matrix, and feature contributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Classification metrics appropriate for an imbalanced binary task."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: float(v) for k, v in metrics.items()}


def evaluate_pipeline(pipeline, X_test, y_test) -> dict[str, float]:
    """Run a fitted pipeline on held-out data and return metrics plus confusion counts."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k in METRIC_NAMES)


def feature_contributions(pipeline, features: dict[str, float]) -> list[dict]:
    """Model-agnostic local explanation.

    For each feature, replace the user's value with the training-set median and
    measure how much the predicted probability drops. A positive contribution
    means the user's value pushes risk *up* relative to a typical patient.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    baseline = preprocessor.baseline_values_
    base_prob = float(pipeline.predict_proba(pd.DataFrame([features]))[0, 1])

    rows = []
    for name in features:
        if name not in baseline:
            continue
        counterfactual = dict(features)
        counterfactual[name] = baseline[name]
        cf_prob = float(pipeline.predict_proba(pd.DataFrame([counterfactual]))[0, 1])
        rows.append(
            {
                "feature": name,
                "value": float(features[name]),
                "typical_value": float(baseline[name]),
                "contribution": base_prob - cf_prob,
            }
        )
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


def global_feature_importance(pipeline) -> dict[str, float] | None:
    """Native importance when the model exposes it (trees or linear coefficients)."""
    model = pipeline.named_steps["model"]
    names = list(pipeline.named_steps["preprocess"].feature_names_)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel()
    else:
        return None
    total = values.sum() or 1.0
    return {n: float(v / total) for n, v in zip(names, values)}
