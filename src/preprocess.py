"""Data loading, cleaning and feature engineering for the Pima Indians Diabetes dataset.

Domain notes
------------
* The raw CSV has no header row. Column names follow the UCI documentation.
* Several measurements use ``0`` to mean "not recorded" (nobody has a glucose
  level or blood pressure of zero). Those zeros are treated as missing and imputed
  with the training-set median (robust to the skewed Insulin distribution).
* ``BloodPressure`` is *diastolic* blood pressure in mm Hg.
* A derived ordinal feature, ``BMICategoryCode`` (underweight / normal / overweight /
  obese), is added and encoded as 0-3. It is the one categorical feature in the
  pipeline and gives tree models a coarse clinical split to work with.
* All fitting (imputation statistics, scaler) happens on the training split only,
  via :class:`DiabetesPreprocessor`, so there is no leakage into the test set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET = "Outcome"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
COLUMNS = FEATURE_COLUMNS + [TARGET]

# Zero is physiologically impossible for these and encodes a missing measurement.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

BMI_CATEGORIES = ["underweight", "normal", "overweight", "obese"]
BMI_CATEGORY_CODES = {name: i for i, name in enumerate(BMI_CATEGORIES)}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_data(path: str | Path) -> pd.DataFrame:
    """Load the diabetes CSV, adding the standard header if the file has none."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/get_data.py` to download the dataset."
        )
    first_line = path.read_text().splitlines()[0]
    has_header = any(ch.isalpha() for ch in first_line)
    df = pd.read_csv(path, header=0 if has_header else None)
    if not has_header:
        df.columns = COLUMNS
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df[COLUMNS]


# --------------------------------------------------------------------------- #
# Pure, stateless transformations (each returns a new DataFrame)
# --------------------------------------------------------------------------- #
def replace_zeros_with_nan(df: pd.DataFrame, columns: Iterable[str] = ZERO_AS_MISSING) -> pd.DataFrame:
    """Return a copy where impossible zeros in ``columns`` are replaced with NaN."""
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    out[cols] = out[cols].replace(0, np.nan)
    return out


def compute_fill_values(df: pd.DataFrame, strategy: str = "median") -> dict[str, float]:
    """Compute per-column fill values (median or mean) from a *training* frame."""
    if strategy == "median":
        stats = df.median(numeric_only=True)
    elif strategy == "mean":
        stats = df.mean(numeric_only=True)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy!r}")
    return {col: float(val) for col, val in stats.items()}


def impute_missing(df: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    """Return a copy with NaNs filled using precomputed ``fill_values``."""
    out = df.copy()
    for col, val in fill_values.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    return out


def bmi_category(bmi: float) -> str:
    """WHO adult BMI category for a single value."""
    if pd.isna(bmi):
        return "normal"
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a string ``BMICategory`` column derived from BMI."""
    out = df.copy()
    out["BMICategory"] = out["BMI"].apply(bmi_category)
    return out


def encode_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``BMICategory`` ordinally encoded as ``BMICategoryCode`` (0-3).

    Ordinal encoding (not one-hot) is used because the categories are ordered.
    The original string column is dropped so the frame is fully numeric.
    """
    out = df.copy()
    if "BMICategory" not in out.columns:
        out = add_bmi_category(out)
    out["BMICategoryCode"] = out["BMICategory"].map(BMI_CATEGORY_CODES).astype(int)
    return out.drop(columns=["BMICategory"])


def make_scaler(kind: str | None):
    if kind in (None, "none"):
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {kind!r}")


def split_data(
    df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2, random_state: int = 42
):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# --------------------------------------------------------------------------- #
# Stateful, leakage-safe preprocessor (fits on train only)
# --------------------------------------------------------------------------- #
class DiabetesPreprocessor(BaseEstimator, TransformerMixin):
    """Clean -> impute -> engineer -> scale, as a scikit-learn transformer.

    Fitting learns the imputation statistics and scaler parameters from the
    training data. ``transform`` never mutates its input.
    """

    def __init__(
        self,
        zero_as_missing: list[str] | None = None,
        imputation: str = "median",
        scaler: str | None = "standard",
        bmi_category: bool = True,
    ):
        self.zero_as_missing = zero_as_missing
        self.imputation = imputation
        self.scaler = scaler
        self.bmi_category = bmi_category

    # -- helpers ---------------------------------------------------------- #
    def _zero_cols(self) -> list[str]:
        return list(self.zero_as_missing) if self.zero_as_missing is not None else list(ZERO_AS_MISSING)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        elif isinstance(X, dict):
            frame = pd.DataFrame([X])
        else:
            frame = pd.DataFrame(np.asarray(X), columns=FEATURE_COLUMNS)
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")
        return frame[FEATURE_COLUMNS].astype(float)

    def _clean_and_engineer(self, frame: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
        frame = replace_zeros_with_nan(frame, self._zero_cols())
        frame = impute_missing(frame, fill_values)
        if self.bmi_category:
            frame = encode_bmi_category(frame)
        return frame

    # -- sklearn API ------------------------------------------------------ #
    def fit(self, X, y=None):
        frame = self._as_frame(X)
        frame = replace_zeros_with_nan(frame, self._zero_cols())
        self.fill_values_ = compute_fill_values(frame, self.imputation)
        frame = self._clean_and_engineer(self._as_frame(X), self.fill_values_)
        self.feature_names_ = list(frame.columns)
        # Typical (median) raw feature values; used for defaults and what-if explanations.
        self.baseline_values_ = {c: float(v) for c, v in frame[FEATURE_COLUMNS].median().items()}
        self.scaler_ = make_scaler(self.scaler)
        if self.scaler_ is not None:
            self.scaler_.fit(frame.values)
        return self

    def transform(self, X) -> pd.DataFrame:
        frame = self._clean_and_engineer(self._as_frame(X), self.fill_values_)
        frame = frame[self.feature_names_]
        if self.scaler_ is not None:
            values = self.scaler_.transform(frame.values)
            frame = pd.DataFrame(values, columns=self.feature_names_, index=frame.index)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


def preprocess_for_training(df: pd.DataFrame, config: dict):
    """Convenience wrapper used by train.py: split, then fit the preprocessor on train only."""
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    X_train, X_test, y_train, y_test = split_data(
        df,
        target=data_cfg.get("target", TARGET),
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )
    preprocessor = DiabetesPreprocessor(
        zero_as_missing=data_cfg.get("zero_as_missing", ZERO_AS_MISSING),
        imputation=prep_cfg.get("imputation", "median"),
        scaler=prep_cfg.get("scaler", "standard"),
        bmi_category=prep_cfg.get("bmi_category", True),
    )
    return preprocessor, X_train, X_test, y_train, y_test
