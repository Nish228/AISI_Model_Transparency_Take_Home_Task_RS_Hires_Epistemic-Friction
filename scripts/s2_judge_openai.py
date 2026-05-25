"""
STAGE 2 (OPTION A): LLM judge scoring using OpenAI.

Reads results/raw_trials.jsonl, scores each trial on the three decoupled DVs,
writes results/judged_trials.jsonl.

Resumable: if you interrupt and rerun, it picks up where it left off.

Usage:
  export OPENAI_API_KEY="sk-..."
  python -m scripts.s2_judge_openai

Skip this script entirely if you want to hand-score every trial yourself.
Use scripts.s2_judge_template instead to get a CSV to fill in by hand.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_client import ChatClient
from src.judge import score_all_trials


# ======================================================================
# >>> PLACE YOUR OPENAI KEY HERE (or set the OPENAI_API_KEY env var) <<<
# ======================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "PASTE_YOUR_OPENAI_KEY_HERE")

# Judge model. Should be DIFFERENT from your MUT to avoid same-model bias.
# gpt-4o-mini is cheap and reliable for structured scoring.
JUDGE_MODEL = "gpt-4o-mini"


def main():
    if OPENAI_API_KEY.startswith("PASTE_"):
        raise SystemExit("Set OPENAI_API_KEY env var or edit the script.")

    print("=" * 60)
    print("STAGE 2: LLM judge scoring")
    print(f"Judge model: {JUDGE_MODEL}")
    print("=" * 60)

    judge_client = ChatClient(
        backend="openai",
        api_key=OPENAI_API_KEY,
        model=JUDGE_MODEL,
    )

    score_all_trials(judge_client)

    print("\nStage 2 complete. Judged trials in results/judged_trials.jsonl")
    print("\nNext step:  python -m scripts.s3_analyze")
    print("\nOptional: also hand-score 40 trials for reliability:")
    print("    python -m scripts.s2_judge_template  (writes spotcheck CSV)")
    print("    # fill the human_* columns by hand, then:")
    print("    python -c 'from src.reliability import compute_kappa; compute_kappa()'")


if __name__ == "__main__":
    main()
