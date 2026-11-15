"""Command-line interface: an interactive loop over the same DiabetesAssistant.

    python -m src.cli                 # interactive
    python -m src.cli "I'm 45, glucose 130, BMI 29"   # single question
"""
from __future__ import annotations

import sys

from src.interface import DiabetesAssistant


def _print_result(result: dict, verbose: bool = True) -> None:
    print("\nAssistant:", result["reply"], "\n")
    if verbose:
        parsed, pred = result["parsed"], result["prediction"]
        print(f"  [parser={parsed.parser} intent={parsed.intent} features={parsed.features}"
              + (f" missing={parsed.missing_required}" if parsed.missing_required else "")
              + (f" invalid={parsed.invalid}" if parsed.invalid else "") + "]")
        if pred is not None:
            print(f"  [probability={pred.probability:.3f} band={pred.risk_band} defaulted={pred.defaulted}]")
        print()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    assistant = DiabetesAssistant()
    mode = "LLM" if assistant.llm_enabled else "offline rule-based"
    print(f"Diabetes Risk Assistant ({mode} mode). Type 'quit' to exit.")

    if argv:
        _print_result(assistant.handle(" ".join(argv)))
        return

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break
        _print_result(assistant.handle(text))


if __name__ == "__main__":
    main()
