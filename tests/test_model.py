"""Model tests: prediction type/shape, minimum performance, sanity of behaviour."""
import numpy as np
import pandas as pd

from src.evaluate import compute_metrics, feature_contributions
from src.preprocess import FEATURE_COLUMNS

MIN_ACCURACY = 0.70
MIN_ROC_AUC = 0.75


def test_predictions_have_correct_type_and_shape(pipeline, splits):
    _, X_test, _, _ = splits
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_meets_minimum_performance_on_held_out_set(pipeline, splits):
    _, X_test, _, y_test = splits
    metrics = compute_metrics(y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1])
    assert metrics["accuracy"] >= MIN_ACCURACY, metrics
    assert metrics["roc_auc"] >= MIN_ROC_AUC, metrics
    assert 0 < metrics["recall"] <= 1 and 0 < metrics["precision"] <= 1


def test_known_samples_are_ordered_sensibly(pipeline):
    """A classic high-risk profile must score above a clearly low-risk one."""
    high = dict(zip(FEATURE_COLUMNS, [6, 180, 85, 35, 200, 36.0, 0.9, 55]))
    low = dict(zip(FEATURE_COLUMNS, [0, 85, 65, 20, 70, 21.0, 0.2, 23]))
    probs = pipeline.predict_proba(pd.DataFrame([high, low]))[:, 1]
    assert probs[0] > 0.5 > probs[1]
    assert probs[0] > probs[1] + 0.3


def test_single_row_and_dict_inputs_work(pipeline):
    row = dict(zip(FEATURE_COLUMNS, [1, 120, 70, 20, 80, 28.0, 0.4, 40]))
    p_df = pipeline.predict_proba(pd.DataFrame([row]))[0, 1]
    p_dict = pipeline.predict_proba(row)[0, 1]  # preprocessor accepts a dict
    assert abs(p_df - p_dict) < 1e-12


def test_feature_contributions_are_well_formed(pipeline):
    row = dict(zip(FEATURE_COLUMNS, [2, 160, 80, 30, 150, 34.0, 0.6, 50]))
    contribs = feature_contributions(pipeline, row)
    assert {c["feature"] for c in contribs} == set(FEATURE_COLUMNS)
    assert all(-1 <= c["contribution"] <= 1 for c in contribs)
    # sorted by magnitude, and glucose should be a top-3 driver for this profile
    mags = [abs(c["contribution"]) for c in contribs]
    assert mags == sorted(mags, reverse=True)
    assert "Glucose" in [c["feature"] for c in contribs[:3]]
