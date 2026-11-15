"""Interface tests: parsing accuracy, LLM-JSON handling, and graceful edge cases.

No network access is required: the LLM is replaced with a fake that returns canned
JSON, and the rule-based parser is exercised directly.
"""
import json

import pytest

from src.interface import (
    REQUIRED_FEATURES,
    DiabetesAssistant,
    clarification_message,
    parse_input,
    predict,
    rule_based_parse,
    validate_features,
)
from src.llm import extract_json


class FakeLLM:
    """Returns a scripted reply; records the messages it was sent."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def chat(self, messages, json_mode=False):
        self.calls.append(messages)
        return self.reply


class BrokenLLM:
    def chat(self, messages, json_mode=False):
        raise ConnectionError("provider down")


# ---------------------------------------------------------------- parsing accuracy
def test_rule_parser_extracts_core_features():
    parsed = rule_based_parse(
        "I'm a 35-year-old woman with a glucose level of 148 and a BMI of 33.6. What does your model predict?"
    )
    assert parsed.intent == "risk_assessment"
    assert parsed.features == {"Glucose": 148.0, "BMI": 33.6, "Age": 35.0}
    assert parsed.is_complete


def test_rule_parser_handles_units_blood_pressure_and_height_weight():
    parsed = rule_based_parse(
        "I'm 52, blood pressure 130/85, glucose 110 mg/dL, 5'4\" and 150 lbs, 2 pregnancies, insulin 94."
    )
    f = parsed.features
    assert f["Age"] == 52 and f["Glucose"] == 110 and f["Insulin"] == 94 and f["Pregnancies"] == 2
    assert f["BloodPressure"] == 85  # diastolic, not systolic
    assert f["BMI"] == pytest.approx(25.7, abs=0.1)  # computed from 5'4" / 150 lb


def test_llm_parser_uses_structured_json_from_model():
    reply = json.dumps(
        {
            "intent": "risk_assessment",
            "features": {"Pregnancies": 0, "Glucose": 130, "BloodPressure": 82, "SkinThickness": None,
                         "Insulin": None, "BMI": 29.0, "DiabetesPedigreeFunction": None, "Age": 45},
            "notes": "converted 7.2 mmol/L to 130 mg/dL",
        }
    )
    llm = FakeLLM(reply)
    parsed = parse_input("I'm a 45 year old man, glucose 7.2 mmol/L, BMI 29, BP 120/82", llm)
    assert parsed.parser == "llm"
    assert parsed.features == {"Pregnancies": 0.0, "Glucose": 130.0, "BloodPressure": 82.0, "BMI": 29.0, "Age": 45.0}
    assert "mmol" in parsed.notes
    assert len(llm.calls) == 1 and llm.calls[0][0]["role"] == "system"


def test_extract_json_tolerates_code_fences_and_chatter():
    text = 'Sure! Here you go:\n```json\n{"intent": "out_of_scope", "features": {}, "notes": ""}\n```'
    assert extract_json(text)["intent"] == "out_of_scope"


def test_llm_failure_falls_back_to_rules():
    parsed = parse_input("I am 40, glucose 120, bmi 30", BrokenLLM())
    assert parsed.parser == "rules"
    assert parsed.features == {"Glucose": 120.0, "BMI": 30.0, "Age": 40.0}
    assert "LLM parse failed" in parsed.notes


# ---------------------------------------------------------------- edge cases
def test_incomplete_input_asks_for_missing_required_features():
    parsed = rule_based_parse("I'm 45 and my BMI is 28. What's my diabetes risk?")
    assert not parsed.is_complete
    assert parsed.missing_required == ["Glucose"]
    message = clarification_message(parsed)
    assert "glucose" in message.lower() and "need" in message.lower()


def test_out_of_scope_query_is_declined_without_prediction(pipeline):
    assistant = DiabetesAssistant(pipeline=pipeline, llm=None)
    result = assistant.handle("What's the weather like in Boston today?")
    assert result["parsed"].intent == "out_of_scope"
    assert result["prediction"] is None
    assert "outside" in result["reply"].lower()


def test_invalid_values_are_flagged_not_predicted(pipeline):
    assistant = DiabetesAssistant(pipeline=pipeline, llm=None)
    result = assistant.handle("glucose 5000, bmi 30, age 40")
    assert result["prediction"] is None
    assert "Glucose" in result["parsed"].invalid
    assert "double-check" in result["reply"].lower()


def test_validate_features_rejects_non_numeric_and_out_of_range():
    valid, invalid = validate_features({"Glucose": "high", "BMI": 5, "Age": 40, "Insulin": None})
    assert valid == {"Age": 40.0}
    assert set(invalid) == {"Glucose", "BMI"}


def test_llm_out_of_scope_intent_is_respected():
    llm = FakeLLM(json.dumps({"intent": "out_of_scope", "features": {k: None for k in REQUIRED_FEATURES}, "notes": ""}))
    parsed = parse_input("Write me a Python function that reverses a string", llm)
    assert parsed.intent == "out_of_scope" and not parsed.is_complete


# ---------------------------------------------------------------- end to end
def test_complete_query_produces_prediction_and_grounded_reply(pipeline):
    assistant = DiabetesAssistant(pipeline=pipeline, llm=None)
    result = assistant.handle("I'm a 35-year-old woman with a glucose level of 148 and a BMI of 33.6.")
    pred = result["prediction"]
    assert pred is not None and 0 <= pred.probability <= 1
    assert set(pred.defaulted) == {"Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"}
    assert f"{pred.probability * 100:.0f}%" in result["reply"]
    assert "not a medical diagnosis" in result["reply"]


def test_llm_response_receives_model_numbers(pipeline):
    """The explanation LLM must be grounded in the model output, not free-styling."""
    parse_reply = json.dumps({"intent": "risk_assessment", "features": {"Glucose": 148, "BMI": 33.6, "Age": 35}, "notes": ""})
    llm = FakeLLM(parse_reply)
    assistant = DiabetesAssistant(pipeline=pipeline, llm=llm)
    result = assistant.handle("35 years old, glucose 148, BMI 33.6")
    # second call is the response-generation call; its user message carries the model output
    payload = json.loads(llm.calls[1][1]["content"])
    assert payload["predicted_probability_of_diabetes"] == round(result["prediction"].probability, 3)
    assert payload["features_used"]["Glucose"] == 148.0


def test_predict_fills_optional_features_with_training_medians(pipeline):
    pred = predict(pipeline, {"Glucose": 120.0, "BMI": 30.0, "Age": 40.0})
    baseline = pipeline.named_steps["preprocess"].baseline_values_
    assert pred.features_used["Insulin"] == baseline["Insulin"]
    assert pred.risk_band in {"low", "moderate", "elevated", "high"}
