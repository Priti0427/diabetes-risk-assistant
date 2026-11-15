"""Model factory: builds an estimator from a name and a parameter dict (from config.yaml)."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

SUPPORTED_MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "mlp"]


def build_model(name: str, params: dict | None = None, random_state: int = 42):
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state, **params)
    if name == "xgboost":
        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=4,
            **params,
        )
    if name == "mlp":
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        return MLPClassifier(random_state=random_state, **params)
    raise ValueError(f"Unknown model {name!r}. Supported: {SUPPORTED_MODELS}")
