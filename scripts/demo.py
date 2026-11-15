"""Scripted end-to-end demo: runs representative queries (including edge cases)
through the assistant and writes a Markdown transcript to docs/demo_transcript.md.

    python scripts/demo.py            # uses the LLM if LLM_API_KEY is set, else offline mode
    python scripts/demo.py --offline  # force rule-based mode
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interface import DiabetesAssistant  # noqa: E402

QUERIES = [
    ("Complete query", "I'm a 35-year-old woman with a glucose level of 148 and a BMI of 33.6. What does your model predict?"),
    ("Rich query with units to convert", "I'm 52, blood pressure 130/85, glucose 110 mg/dL, 5'4\" and 150 lbs, two pregnancies, insulin 94."),
    ("Low-risk profile", "23 year old, never been pregnant, glucose 85, BMI 21, blood pressure 110/65."),
    ("Edge case: incomplete input", "I'm 45 and my BMI is 28. What's my diabetes risk?"),
    ("Edge case: implausible value", "glucose 5000, bmi 30, age 40"),
    ("Edge case: out of scope", "What's the weather like in Boston today?"),
    ("Edge case: asks about the tool", "What information do you need from me?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo_transcript.md"))
    args = parser.parse_args()

    assistant = DiabetesAssistant(llm=None) if args.offline else DiabetesAssistant()
    if args.offline:
        assistant.llm = None
    mode = "LLM-powered" if assistant.llm_enabled else "offline rule-based"

    lines = [
        "# Demo transcript",
        "",
        f"Generated {dt.date.today().isoformat()} by `python scripts/demo.py` in **{mode}** mode.",
        "",
    ]
    for title, query in QUERIES:
        result = assistant.handle(query)
        parsed, pred = result["parsed"], result["prediction"]
        print(f"\n=== {title} ===\nUser: {query}\nAssistant: {result['reply']}\n")
        lines += [f"## {title}", "", f"**User:** {query}", "", f"**Assistant:** {result['reply']}", ""]
        lines.append(f"<details><summary>Parsed input (parser: {parsed.parser}, intent: {parsed.intent})</summary>\n")
        lines.append(f"- features: `{parsed.features}`")
        if parsed.missing_required:
            lines.append(f"- missing required: `{parsed.missing_required}`")
        if parsed.invalid:
            lines.append(f"- invalid: `{parsed.invalid}`")
        if pred is not None:
            lines.append(f"- model probability: **{pred.probability:.3f}** ({pred.risk_band} risk), label={pred.label}")
            lines.append(f"- filled with typical values: `{pred.defaulted}`")
            top = ", ".join(f"{c['feature']} {c['contribution']:+.3f}" for c in pred.contributions[:3])
            lines.append(f"- top contributions: {top}")
        lines += ["", "</details>", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Transcript written to {out}")


if __name__ == "__main__":
    main()
