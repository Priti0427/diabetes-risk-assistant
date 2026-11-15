"""Preprocessing tests: missing values, encoding, scaling, immutability, leakage-safety."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    BMI_CATEGORY_CODES,
    FEATURE_COLUMNS,
    ZERO_AS_MISSING,
    DiabetesPreprocessor,
    add_bmi_category,
    bmi_category,
    compute_fill_values,
    encode_bmi_category,
    impute_missing,
    replace_zeros_with_nan,
    split_data,
)


# ---------------------------------------------------------------- missing values
def test_zeros_become_nan_only_in_impossible_columns(toy_X):
    out = replace_zeros_with_nan(toy_X)
    # Row 1 had zeros in Glucose/BP/Skin/Insulin -> NaN
    assert out.loc[1, ["Glucose", "BloodPressure", "SkinThickness", "Insulin"]].isna().all()
    # Pregnancies = 0 is a legitimate value and must be preserved
    assert out.loc[3, "Pregnancies"] == 0
    assert out["Pregnancies"].isna().sum() == 0


def test_imputation_fills_all_missing_with_training_median(toy_X):
    with_nan = replace_zeros_with_nan(toy_X)
    fills = compute_fill_values(with_nan, "median")
    out = impute_missing(with_nan, fills)
    assert out.isna().sum().sum() == 0
    # median of the non-missing glucose values [100, 150, 180, 120] = 135
    assert fills["Glucose"] == 135.0
    assert out.loc[1, "Glucose"] == 135.0


def test_preprocessor_output_has_no_missing_values(toy_X):
    prep = DiabetesPreprocessor(scaler="none").fit(toy_X)
    out = prep.transform(toy_X)
    assert not out.isna().any().any()
    # a brand-new row with a zero insulin reading is imputed, not passed through as 0
    new = pd.DataFrame([{**dict(zip(FEATURE_COLUMNS, [1, 120, 70, 20, 0, 30.0, 0.5, 40]))}])
    assert prep.transform(new)["Insulin"].iloc[0] == prep.fill_values_["Insulin"]


# ---------------------------------------------------------------- encoding
@pytest.mark.parametrize(
    "bmi,expected",
    [(17.0, "underweight"), (18.5, "normal"), (24.9, "normal"), (25.0, "overweight"), (29.9, "overweight"), (30.0, "obese")],
)
def test_bmi_category_thresholds(bmi, expected):
    assert bmi_category(bmi) == expected


def test_bmi_category_is_ordinally_encoded(toy_X):
    with_cat = add_bmi_category(toy_X)
    assert with_cat["BMICategory"].tolist() == ["underweight", "normal", "overweight", "obese", "obese"]
    encoded = encode_bmi_category(with_cat)
    assert "BMICategory" not in encoded.columns  # string column dropped -> fully numeric
    assert encoded["BMICategoryCode"].tolist() == [0, 1, 2, 3, 3]
    assert pd.api.types.is_integer_dtype(encoded["BMICategoryCode"])
    assert set(encoded["BMICategoryCode"].unique()) <= set(BMI_CATEGORY_CODES.values())


# ---------------------------------------------------------------- scaling
def test_standard_scaler_produces_zero_mean_unit_variance(raw_df):
    X = raw_df.drop(columns=["Outcome"])
    prep = DiabetesPreprocessor(scaler="standard").fit(X)
    out = prep.transform(X)
    np.testing.assert_allclose(out.mean().values, 0.0, atol=1e-6)
    np.testing.assert_allclose(out.std(ddof=0).values, 1.0, atol=1e-6)


def test_minmax_scaler_bounds_features_to_unit_interval(raw_df):
    X = raw_df.drop(columns=["Outcome"])
    prep = DiabetesPreprocessor(scaler="minmax").fit(X)
    out = prep.transform(X)
    assert (out.min() >= -1e-9).all() and (out.max() <= 1 + 1e-9).all()
    assert out.shape == (len(X), len(FEATURE_COLUMNS) + 1)  # + BMICategoryCode


# ---------------------------------------------------------------- immutability
def test_stateless_functions_do_not_mutate_input(toy_X):
    original = toy_X.copy(deep=True)
    replace_zeros_with_nan(toy_X)
    encode_bmi_category(toy_X)
    impute_missing(replace_zeros_with_nan(toy_X), compute_fill_values(toy_X))
    pd.testing.assert_frame_equal(toy_X, original)


def test_preprocessor_fit_transform_does_not_mutate_input(toy_X):
    original = toy_X.copy(deep=True)
    prep = DiabetesPreprocessor().fit(toy_X)
    prep.transform(toy_X)
    pd.testing.assert_frame_equal(toy_X, original)


# ---------------------------------------------------------------- leakage / split
def test_split_is_stratified_and_disjoint(raw_df):
    X_train, X_test, y_train, y_test = split_data(raw_df, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(raw_df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_fill_values_come_from_training_data_only(raw_df):
    X_train, X_test, *_ = split_data(raw_df)
    prep = DiabetesPreprocessor().fit(X_train)
    expected = replace_zeros_with_nan(X_train, ZERO_AS_MISSING)["Glucose"].median()
    assert prep.fill_values_["Glucose"] == expected
