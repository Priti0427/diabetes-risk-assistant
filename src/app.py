"""Streamlit chat UI for the Diabetes Risk Assistant.

Run from the repository root:

    streamlit run src/app.py

Set LLM_API_KEY (see .env.example) to enable LLM parsing and explanations; without
it the app runs in rule-based/offline mode so the model is still usable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Streamlit puts src/ on the path, not the repo root
    sys.path.insert(0, str(ROOT))

from src.interface import DiabetesAssistant, FEATURE_SPEC  # noqa: E402

st.set_page_config(page_title="Diabetes Risk Assistant", page_icon="🩺", layout="centered")

EXAMPLES = [
    "I'm a 35-year-old woman with a glucose level of 148 and a BMI of 33.6. What does your model predict?",
    "52 years old, glucose 110 mg/dL, blood pressure 130/85, 5'4\" and 150 lbs, two pregnancies.",
    "I'm 45 and my BMI is 28. What's my diabetes risk?",
    "What information do you need from me?",
    "What's the weather like in Boston today?",
]


@st.cache_resource(show_spinner="Loading model...")
def get_assistant() -> DiabetesAssistant:
    return DiabetesAssistant()


def best_run_info() -> dict | None:
    path = ROOT / "models" / "best_run.json"
    return json.loads(path.read_text()) if path.exists() else None


def render_details(result: dict) -> None:
    parsed, pred = result["parsed"], result["prediction"]
    with st.expander("How this answer was produced"):
        st.markdown(f"**Parser:** `{parsed.parser}` · **Intent:** `{parsed.intent}`")
        if parsed.features:
            st.markdown("**Extracted features**")
            st.table({"feature": list(parsed.features), "value": [f"{v:g}" for v in parsed.features.values()]})
        if parsed.missing_required:
            st.warning("Missing required: " + ", ".join(parsed.missing_required))
        if parsed.invalid:
            st.error("Invalid: " + "; ".join(f"{k} ({v})" for k, v in parsed.invalid.items()))
        if pred is not None:
            st.metric("Predicted probability of diabetes", f"{pred.probability:.1%}", pred.risk_band + " risk")
            if pred.defaulted:
                st.caption("Filled with typical values: " + ", ".join(pred.defaulted))
            if pred.contributions:
                st.markdown("**Feature contributions** (change in probability vs. a typical patient)")
                st.bar_chart(
                    {c["feature"]: c["contribution"] for c in pred.contributions},
                    horizontal=True,
                )


def main() -> None:
    try:
        assistant = get_assistant()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.title("🩺 Diabetes Risk Assistant")
        st.write(
            "Describe your health measurements in plain English. An LLM extracts the model's "
            "inputs, a trained classifier estimates your diabetes probability, and the LLM "
            "explains the result."
        )
        mode = "LLM-powered (Nebius / OpenAI-compatible)" if assistant.llm_enabled else "Offline rule-based (no LLM_API_KEY set)"
        st.info(f"**Mode:** {mode}")
        info = best_run_info()
        if info:
            st.markdown(
                f"**Model:** `{info['run_name']}` ({info['model']})  \n"
                f"ROC-AUC {info['roc_auc']:.3f} · F1 {info['f1']:.3f} · Recall {info['recall']:.3f}"
            )
        st.markdown("**Model inputs**")
        st.markdown(
            "\n".join(
                f"- {'**' if s['required'] else ''}{name}{'**' if s['required'] else ''}: {s['label']}"
                + (f" ({s['unit']})" if s["unit"] else "")
                for name, s in FEATURE_SPEC.items()
            )
            + "\n\nBold = required."
        )
        st.markdown("**Try an example**")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state["pending"] = ex
        if st.button("Clear conversation", type="secondary"):
            st.session_state["messages"] = []
            st.rerun()
        st.caption("Not medical advice. Model trained on the Pima Indians Diabetes dataset (768 women).")

    st.session_state.setdefault("messages", [])
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("result"):
                render_details(msg["result"])

    prompt = st.chat_input("e.g. I'm 40, glucose 125, BMI 31, blood pressure 120/80")
    prompt = st.session_state.pop("pending", None) or prompt
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = assistant.handle(prompt)
            st.markdown(result["reply"])
            render_details(result)
        st.session_state["messages"].append({"role": "assistant", "content": result["reply"], "result": result})


if __name__ == "__main__":  # Streamlit runs the script as __main__
    main()
