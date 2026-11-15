"""Shared fixtures.

The real dataset is downloaded if absent (it is not committed to Git). The model
fixture prefers the exported best pipeline in models/, and otherwise trains a quick
logistic-regression pipeline so the suite still runs on a fresh clone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import COLUMNS, DiabetesPreprocessor, load_data, split_data  # noqa: E402

DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from scripts.get_data import download

        download(DATA_PATH)
    return load_data(DATA_PATH)


@pytest.fixture(scope="session")
def splits(raw_df):
    return split_data(raw_df, test_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def pipeline(splits):
    if MODEL_PATH.exists():
        import joblib

        return joblib.load(MODEL_PATH)
    X_train, _, y_train, _ = splits
    pipe = Pipeline([("preprocess", DiabetesPreprocessor()), ("model", LogisticRegression(max_iter=1000))])
    return pipe.fit(X_train, y_train)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Small hand-built frame with impossible zeros and a spread of BMI categories."""
    rows = [
        # Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age, Outcome
        [1, 100, 70, 20, 80, 17.0, 0.3, 25, 0],
        [2, 0, 0, 0, 0, 22.0, 0.5, 30, 0],
        [3, 150, 80, 30, 120, 27.5, 0.7, 45, 1],
        [0, 180, 90, 0, 0, 35.0, 1.2, 55, 1],
        [5, 120, 75, 25, 100, 31.0, 0.4, 35, 0],
    ]
    return pd.DataFrame(rows, columns=COLUMNS).astype(float).astype({"Outcome": int, "Pregnancies": int})


@pytest.fixture
def toy_X(toy_df):
    return toy_df.drop(columns=["Outcome"])


@pytest.fixture
def rng():
    return np.random.default_rng(0)
