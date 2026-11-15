"""Natural-language interface layer: parse -> validate -> predict -> explain.

The flow for one user message is:

1. **Parse.** An LLM (or, offline, a rule-based parser) turns free text into the
   eight model features plus an *intent* (risk assessment vs. out of scope).
2. **Validate.** Values are range-checked; required features that are missing
   produce a clarifying question instead of a prediction.
3. **Predict.** The best MLflow pipeline (``models/best_model.joblib``) returns a
   probability, and a what-if analysis ranks the drivers of that probability.
4. **Explain.** The LLM writes a grounded, caveated answer from the numbers in
   step 3 (never from its own medical guesses). Offline, a template is used.

Everything except the actual LLM call is deterministic and unit-tested.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from src.evaluate import feature_contributions
from src.llm import ChatClient, extract_json, get_llm_client
from src.preprocess import FEATURE_COLUMNS, bmi_category

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.joblib"

# --------------------------------------------------------------------------- #
# Feature specification
# --------------------------------------------------------------------------- #
FEATURE_SPEC: dict[str, dict] = {
    "Pregnancies": {"label": "number of pregnancies", "unit": "", "range": (0, 20), "required": False},
    "Glucose": {"label": "plasma glucose", "unit": "mg/dL", "range": (40, 400), "required": True},
    "BloodPressure": {"label": "diastolic blood pressure", "unit": "mm Hg", "range": (30, 200), "required": False},
    "SkinThickness": {"label": "triceps skin-fold thickness", "unit": "mm", "range": (5, 100), "required": False},
    "Insulin": {"label": "2-hour serum insulin", "unit": "mu U/mL", "range": (10, 900), "required": False},
    "BMI": {"label": "body mass index", "unit": "kg/m²", "range": (10, 70), "required": True},
    "DiabetesPedigreeFunction": {"label": "diabetes pedigree function", "unit": "", "range": (0.05, 3.0), "required": False},
    "Age": {"label": "age", "unit": "years", "range": (1, 120), "required": True},
}
REQUIRED_FEATURES = [f for f, s in FEATURE_SPEC.items() if s["required"]]
OPTIONAL_FEATURES = [f for f, s in FEATURE_SPEC.items() if not s["required"]]

# Questions about the tool itself (no numbers given) -> explain what we need.
ABOUT_TOOL_KEYWORDS = (
    "what do you need", "what information", "what data", "how do you work", "how does this",
    "what can you", "what does this", "how to use", "help me", "what should i", "which inputs",
)
HEALTH_KEYWORDS = (
    "diabet", "glucose", "sugar", "bmi", "insulin", "pregnan", "blood pressure", "risk",
    "weight", "height", "pedigree", "a1c", "overweight", "obese", "predict", "model",
)


@dataclass
class ParsedInput:
    intent: str = "risk_assessment"           # risk_assessment | out_of_scope | about_tool
    features: dict[str, float] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    parser: str = "rules"                      # rules | llm

    @property
    def is_complete(self) -> bool:
        return self.intent == "risk_assessment" and not self.missing_required and not self.invalid


@dataclass
class Prediction:
    probability: float
    label: int
    risk_band: str
    features_used: dict[str, float]
    defaulted: list[str]
    contributions: list[dict]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_features(raw: dict) -> tuple[dict[str, float], dict[str, str]]:
    """Coerce to float and range-check. Returns (valid_features, {feature: problem})."""
    valid, invalid = {}, {}
    for name, spec in FEATURE_SPEC.items():
        value = raw.get(name)
        if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            invalid[name] = f"could not read {value!r} as a number"
            continue
        lo, hi = spec["range"]
        if not (lo <= num <= hi):
            invalid[name] = f"{num:g} {spec['unit']} is outside the plausible range {lo}-{hi}".strip()
            continue
        valid[name] = num
    return valid, invalid


def finalize(parsed: ParsedInput, raw_features: dict) -> ParsedInput:
    parsed.features, parsed.invalid = validate_features(raw_features)
    parsed.missing_required = [f for f in REQUIRED_FEATURES if f not in parsed.features and f not in parsed.invalid]
    return parsed


# --------------------------------------------------------------------------- #
# Rule-based parser (offline fallback + deterministic test path)
# --------------------------------------------------------------------------- #
_NUM = r"(\d+(?:\.\d+)?)"

_PATTERNS: dict[str, list[str]] = {
    "Age": [
        rf"\b{_NUM}\s*-?\s*(?:years?|yrs?|y/?o)\b",
        rf"\bage(?:d)?\s*(?:is|of|:)?\s*{_NUM}",
        rf"\b(?:i'?m|i am)\s+{_NUM}\b(?!\s*(?:mg|%|kg|lbs?|pounds?|cm|ft|feet|'|bmi))",
    ],
    "Glucose": [
        rf"(?:glucose|blood sugar|sugar level)\D{{0,25}}?{_NUM}",
        rf"{_NUM}\s*(?:mg/?dl)?\s*(?:fasting\s+)?(?:glucose|blood sugar)",
    ],
    "BMI": [rf"\bbmi\D{{0,15}}?{_NUM}", rf"{_NUM}\s*bmi\b"],
    "BloodPressure": [
        rf"(?:blood pressure|\bbp\b|diastolic)\D{{0,15}}?(?:\d{{2,3}}\s*/\s*)?{_NUM}",
    ],
    "Insulin": [rf"\binsulin\D{{0,15}}?{_NUM}"],
    "SkinThickness": [rf"(?:skin\s*-?\s*fold|skin thickness|triceps)\D{{0,15}}?{_NUM}"],
    "Pregnancies": [rf"{_NUM}\s*(?:pregnanc|kids|children|times pregnant)", rf"pregnan\w*\D{{0,10}}?{_NUM}"],
    "DiabetesPedigreeFunction": [rf"pedigree\D{{0,20}}?(\d(?:\.\d+)?)"],
}


def _first_match(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _bmi_from_height_weight(text: str) -> float | None:
    weight_kg = None
    m = re.search(rf"{_NUM}\s*(kg|kilos?|kilograms?|lbs?|pounds?)\b", text, re.IGNORECASE)
    if m:
        w, unit = float(m.group(1)), m.group(2).lower()
        weight_kg = w * 0.4536 if unit.startswith(("lb", "pound")) else w
    height_m = None
    m = re.search(r"(\d)\s*(?:ft|feet|')\s*(\d{1,2})?\s*(?:in|inches|\")?", text, re.IGNORECASE)
    if m:
        height_m = (int(m.group(1)) * 12 + int(m.group(2) or 0)) * 0.0254
    else:
        m = re.search(r"(\d{3})\s*cm\b", text, re.IGNORECASE) or re.search(r"\b(1\.\d{1,2})\s*m\b", text, re.IGNORECASE)
        if m:
            h = float(m.group(1))
            height_m = h / 100 if h > 3 else h
    if weight_kg and height_m:
        return round(weight_kg / height_m**2, 1)
    return None


def rule_based_parse(text: str) -> ParsedInput:
    """Regex extraction of the model features. Good enough for well-formed inputs."""
    raw: dict[str, float] = {}
    for name, patterns in _PATTERNS.items():
        value = _first_match(text, patterns)
        if value is not None:
            raw[name] = value
    mmol = re.search(rf"{_NUM}\s*mmol\s*/?\s*l", text, re.IGNORECASE)
    if mmol and ("Glucose" not in raw or raw["Glucose"] == float(mmol.group(1))):
        raw["Glucose"] = round(float(mmol.group(1)) * 18, 1)  # mmol/L -> mg/dL
    if "BMI" not in raw:
        bmi = _bmi_from_height_weight(text)
        if bmi:
            raw["BMI"] = bmi
    if "Pregnancies" not in raw and re.search(r"never (?:been )?pregnant|no pregnanc", text, re.IGNORECASE):
        raw["Pregnancies"] = 0.0

    parsed = ParsedInput(parser="rules")
    lowered = text.lower()
    if not raw:
        if any(k in lowered for k in ABOUT_TOOL_KEYWORDS) or any(k in lowered for k in HEALTH_KEYWORDS):
            parsed.intent = "about_tool"
            return finalize(parsed, raw)
        parsed.intent = "out_of_scope"
        return parsed
    return finalize(parsed, raw)


# --------------------------------------------------------------------------- #
# LLM parser
# --------------------------------------------------------------------------- #
PARSE_SYSTEM_PROMPT = f"""You are the input parser for a diabetes-risk prediction tool.
The tool's ML model needs these features (Pima Indians Diabetes dataset definitions):
{json.dumps({k: f"{v['label']} ({v['unit'] or 'count'})" for k, v in FEATURE_SPEC.items()}, indent=2)}

Read the user's message and return ONLY a JSON object with this shape:
{{
  "intent": "risk_assessment" | "out_of_scope" | "about_tool",
  "features": {{"Pregnancies": number|null, "Glucose": number|null, "BloodPressure": number|null,
               "SkinThickness": number|null, "Insulin": number|null, "BMI": number|null,
               "DiabetesPedigreeFunction": number|null, "Age": number|null}},
  "notes": "short string"
}}

Rules:
- "risk_assessment": the user gives personal health numbers and wants a prediction.
- "about_tool": the user asks what the tool does or what information it needs, with no numbers.
- "out_of_scope": anything unrelated to diabetes risk (weather, code, other diseases, medication advice).
- Use null for anything not stated. NEVER guess or invent values.
- Glucose must be mg/dL; convert mmol/L by multiplying by 18. BloodPressure is DIASTOLIC
  (the second number in 120/80). If height and weight are given, compute BMI = kg / m^2.
- "never pregnant" => Pregnancies 0. Men => Pregnancies 0.
- Put conversions or ambiguities you noticed in "notes".
"""


def llm_parse(text: str, llm: ChatClient) -> ParsedInput:
    reply = llm.chat(
        [{"role": "system", "content": PARSE_SYSTEM_PROMPT}, {"role": "user", "content": text}],
        json_mode=True,
    )
    data = extract_json(reply)
    parsed = ParsedInput(parser="llm", notes=str(data.get("notes") or ""))
    intent = data.get("intent", "risk_assessment")
    parsed.intent = intent if intent in ("risk_assessment", "out_of_scope", "about_tool") else "risk_assessment"
    raw = data.get("features") or {}
    if parsed.intent != "risk_assessment" and not any(v is not None for v in raw.values()):
        return parsed
    parsed.intent = "risk_assessment"
    return finalize(parsed, raw)


def parse_input(text: str, llm: ChatClient | None = None) -> ParsedInput:
    """Parse with the LLM when available; fall back to rules on any failure."""
    if llm is not None:
        try:
            return llm_parse(text, llm)
        except Exception as exc:  # network error, malformed JSON, etc.
            parsed = rule_based_parse(text)
            parsed.notes = f"LLM parse failed ({type(exc).__name__}); used rule-based parser."
            return parsed
    return rule_based_parse(text)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train and export a model first:\n"
            "  python -m src.train --all && python scripts/compare_runs.py"
        )
    return joblib.load(path)


def risk_band(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.5:
        return "moderate"
    if probability < 0.75:
        return "elevated"
    return "high"


def predict(pipeline, features: dict[str, float]) -> Prediction:
    """Fill optional features with training medians, run the pipeline, explain."""
    baseline = pipeline.named_steps["preprocess"].baseline_values_
    full = {}
    defaulted = []
    for name in FEATURE_COLUMNS:
        if name in features:
            full[name] = float(features[name])
        else:
            full[name] = float(baseline[name])
            defaulted.append(name)
    frame = pd.DataFrame([full])
    proba = float(pipeline.predict_proba(frame)[0, 1])
    label = int(pipeline.predict(frame)[0])
    contributions = [c for c in feature_contributions(pipeline, full) if c["feature"] not in defaulted]
    return Prediction(
        probability=proba,
        label=label,
        risk_band=risk_band(proba),
        features_used=full,
        defaulted=defaulted,
        contributions=contributions,
    )


# --------------------------------------------------------------------------- #
# Response generation
# --------------------------------------------------------------------------- #
def _describe(name: str) -> str:
    spec = FEATURE_SPEC[name]
    return f"{spec['label']}" + (f" ({spec['unit']})" if spec["unit"] else "")


def clarification_message(parsed: ParsedInput) -> str:
    """Deterministic clarifying question for incomplete / invalid / off-topic input."""
    if parsed.intent == "out_of_scope":
        return (
            "I can only help with one thing: estimating type 2 diabetes risk from health "
            "measurements using a model trained on the Pima Indians Diabetes dataset. "
            "That question is outside what I can answer. If you'd like a risk estimate, tell me "
            "your plasma glucose (mg/dL), BMI (or height and weight), and age."
        )
    if parsed.intent == "about_tool":
        return (
            "I estimate the probability of type 2 diabetes using a machine-learning model. "
            "To run it I need at least your **glucose level (mg/dL)**, **BMI** (or height and weight) "
            "and **age**. Optionally you can add diastolic blood pressure, 2-hour insulin, "
            "triceps skin-fold thickness, number of pregnancies and the diabetes pedigree function "
            "(family-history score). Anything you leave out is filled with a typical value, which "
            "makes the estimate less personal. For example: "
            "\"I'm a 45-year-old woman, glucose 130, BMI 29, blood pressure 120/82.\""
        )
    parts = []
    if parsed.invalid:
        problems = "; ".join(f"{_describe(k)}: {v}" for k, v in parsed.invalid.items())
        parts.append(f"Some values don't look right ({problems}). Could you double-check them?")
    if parsed.missing_required:
        needed = ", ".join(_describe(f) for f in parsed.missing_required)
        have = ", ".join(f"{_describe(k)} = {v:g}" for k, v in parsed.features.items()) or "nothing yet"
        parts.append(
            f"I still need your {needed} before I can run the model. So far I have: {have}."
        )
    return " ".join(parts)


RESPONSE_SYSTEM_PROMPT = """You are the explanation layer of a diabetes-risk screening tool.
You will receive the output of a trained machine-learning model (a random-forest style
classifier trained on the Pima Indians Diabetes dataset of 768 women). Write a clear,
conversational answer for the user (120-200 words) that:

1. States the model's predicted probability of diabetes and the risk band, exactly as given.
2. Explains the 2-3 strongest drivers using the provided contribution numbers (positive
   contribution = pushes risk up vs. a typical patient; negative = pushes it down), in plain
   language with the user's actual values.
3. Mentions which inputs were filled with typical values, if any, and that this reduces precision.
4. Ends with caveats: this is a statistical screening estimate from a small historical dataset
   of Pima women, not a diagnosis; the user should confirm with a clinician (e.g. HbA1c / fasting
   glucose test). Do not give medication advice.

Use ONLY the numbers provided. Never invent additional statistics. No markdown headers."""


def template_response(parsed: ParsedInput, pred: Prediction) -> str:
    pct = pred.probability * 100
    lines = [
        f"Based on the information you provided, the model estimates a **{pct:.0f}% probability** "
        f"of diabetes, which it classifies as **{pred.risk_band} risk** "
        f"(prediction: {'positive' if pred.label else 'negative'})."
    ]
    drivers = [c for c in pred.contributions if abs(c["contribution"]) >= 0.01][:3]
    if drivers:
        bits = []
        for c in drivers:
            direction = "raises" if c["contribution"] > 0 else "lowers"
            bits.append(
                f"your {_describe(c['feature'])} of {c['value']:g} {direction} the estimate by about "
                f"{abs(c['contribution']) * 100:.0f} points versus a typical value of {c['typical_value']:g}"
            )
        lines.append("Key factors: " + "; ".join(bits) + ".")
    if "BMI" in parsed.features:
        lines.append(f"Your BMI falls in the '{bmi_category(parsed.features['BMI'])}' category.")
    if pred.defaulted:
        lines.append(
            "I did not have your " + ", ".join(_describe(f) for f in pred.defaulted)
            + ", so typical values were used; providing them would make the estimate more personal."
        )
    if parsed.notes:
        lines.append(f"Note: {parsed.notes}")
    lines.append(
        "This is a statistical screening estimate from a model trained on a small historical dataset "
        "of Pima women, not a medical diagnosis. Please confirm with a clinician, for example with an "
        "HbA1c or fasting glucose test."
    )
    return "\n\n".join(lines)


def llm_response(user_text: str, parsed: ParsedInput, pred: Prediction, llm: ChatClient) -> str:
    payload = {
        "user_message": user_text,
        "features_used": pred.features_used,
        "filled_with_typical_values": pred.defaulted,
        "parser_notes": parsed.notes,
        "predicted_probability_of_diabetes": round(pred.probability, 3),
        "risk_band": pred.risk_band,
        "bmi_category": bmi_category(pred.features_used["BMI"]),
        "feature_contributions": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in pred.contributions[:5]
        ],
    }
    return llm.chat(
        [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class DiabetesAssistant:
    """End-to-end handler: text in, answer (plus structured details) out."""

    def __init__(self, pipeline=None, llm: ChatClient | None = None, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.pipeline = pipeline if pipeline is not None else load_model(model_path)
        self.llm = llm if llm is not None else get_llm_client()

    @property
    def llm_enabled(self) -> bool:
        return self.llm is not None

    def handle(self, text: str) -> dict:
        parsed = parse_input(text, self.llm)
        if not parsed.is_complete:
            return {"reply": clarification_message(parsed), "parsed": parsed, "prediction": None}

        pred = predict(self.pipeline, parsed.features)
        if self.llm is not None:
            try:
                reply = llm_response(text, parsed, pred, self.llm)
            except Exception:
                reply = template_response(parsed, pred)
        else:
            reply = template_response(parsed, pred)
        return {"reply": reply, "parsed": parsed, "prediction": pred}
