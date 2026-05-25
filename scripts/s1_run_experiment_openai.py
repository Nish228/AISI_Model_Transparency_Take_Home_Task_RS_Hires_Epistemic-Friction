"""
STAGE 1 (OPTION A): Run the experiment trials against an OpenAI model.

Supports two modes:
  - Reasoning models (gpt-5, gpt-5.4, gpt-5.4-mini, o-series):
      Uses the Responses API and requests a reasoning summary, which the
      pipeline captures into the "Model Reasoning" column.
  - Non-reasoning models (gpt-4o, gpt-4o-mini):
      Uses the standard Chat Completions API. The reasoning column will
      be empty (expected behavior).

The script auto-detects which mode to use based on the model name.

Usage:
  export OPENAI_API_KEY="sk-..."
  export PEF_RUN=openai   # tag this run so outputs land in results/openai/
  python -m scripts.s1_run_experiment_openai
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_client import ChatClient
from src.harness import run_experiment


# ======================================================================
# >>> CONFIGURE HERE <<<
# ======================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "PASTE_YOUR_OPENAI_KEY_HERE")

# Pick the model. Options you have access to:
#   "gpt-5.4"        - flagship reasoning model. Most expensive but best.
#   "gpt-5.4-mini"   - reasoning model, cheaper, recommended for this study.
#   "gpt-5.4-nano"   - cheapest reasoning, possibly with less reasoning depth.
#   "gpt-5.3-codex"  - reasoning model tuned for code; probably not the best fit here.
#
# For methodological parity with the ByteDance Seed-OSS run, use one of
# the reasoning models. gpt-5.4-mini is a good default.
MUT_MODEL = "gpt-5.4-mini"

# Reasoning effort -- "none" / "low" / "medium" / "high" / "xhigh".
# "medium" gives substantive reasoning summaries without huge token cost.
# Use "low" if you want to economize; use "high" if you suspect the model
# is rushing through the task.
REASONING_EFFORT = "medium"


# Auto-detect: any model name containing these prefixes is reasoning-capable
# and needs the Responses API.
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(name: str) -> bool:
    n = name.lower()
    return any(n.startswith(p) for p in REASONING_MODEL_PREFIXES)


def main():
    if OPENAI_API_KEY.startswith("PASTE_"):
        raise SystemExit("Set OPENAI_API_KEY env var or edit the script.")

    use_responses = is_reasoning_model(MUT_MODEL)

    print("=" * 60)
    print("STAGE 1: Experiment trials -- OpenAI backend")
    print(f"Model under test: {MUT_MODEL}")
    print(f"API endpoint:     {'/v1/responses (reasoning)' if use_responses else '/v1/chat/completions'}")
    if use_responses:
        print(f"Reasoning effort: {REASONING_EFFORT}")
    if os.environ.get("PEF_RUN"):
        print(f"Run label:        {os.environ['PEF_RUN']}  (outputs in results/{os.environ['PEF_RUN']}/)")
    print("=" * 60)

    client = ChatClient(
        backend="openai",
        api_key=OPENAI_API_KEY,
        model=MUT_MODEL,
        use_responses_api=use_responses,
        reasoning_effort=REASONING_EFFORT,
    )

    run_experiment(client)

    print("\nStage 1 complete.")
    print("\nNext step (pick one):")
    print("  - LLM judge:  python -m scripts.s2_judge_openai")
    print("  - Hand-judge: python -m scripts.s2_judge_template --all")


if __name__ == "__main__":
    main()
