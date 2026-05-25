"""
STAGE 3: Analysis + plots.

Reads scored trials, computes per-cell rates with bootstrap CIs, runs the
confound check (the headline analysis), generates three figures.

Source can be either:
  --source llm   reads results/judged_trials.jsonl  (from s2_judge_openai)
  --source hand  reads results/hand_scores_all.csv  (from s2_judge_template --all)

Default is 'llm' if judged_trials.jsonl exists, else 'hand'.

Usage:
  python -m scripts.s3_analyze              # auto-detect source
  python -m scripts.s3_analyze --source hand
  python -m scripts.s3_analyze --source llm
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.analysis import run_analysis
from src.plotting import plot_all


def _normalize_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes", "y", "t")


def convert_hand_csv_to_judged_jsonl(csv_path: Path, jsonl_path: Path):
    """Convert the hand-scored CSV to the same JSONL format the LLM judge writes,
    so the existing analysis code can consume it without changes."""
    written = 0
    skipped = 0
    with csv_path.open() as f_in, jsonl_path.open("w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            noticed_raw = row.get("human_noticed_discrepancy", "").strip()
            rat_raw = row.get("human_rationalized", "").strip()
            ea_raw = row.get("human_eval_aware", "").strip()
            cp_raw = row.get("human_compliance", "").strip()

            # Skip unscored rows (any missing).
            if not all([noticed_raw, rat_raw, ea_raw, cp_raw]):
                skipped += 1
                continue

            rec = {
                "trial_id": row["trial_id"],
                "scenario": row["scenario"],
                "severity": row["severity"],
                "framing": row["framing"],
                "action": row["action"],
                "scores": {
                    "noticed_discrepancy": _normalize_bool(noticed_raw),
                    "rationalized_anomaly": _normalize_bool(rat_raw),
                    "verbalized_eval_awareness": _normalize_bool(ea_raw),
                    "action_compliance": _normalize_bool(cp_raw),
                    "evidence_quote": "",
                    "_source": "hand",
                },
            }
            f_out.write(json.dumps(rec) + "\n")
            written += 1
    print(f"  Converted {written} hand-scored rows ({skipped} skipped).")
    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["llm", "hand", "auto"],
        default="auto",
        help="Where to read scores from. Default: auto-detect.",
    )
    args = parser.parse_args()

    source = args.source
    if source == "auto":
        if config.JUDGED_RESULTS_FILE.exists():
            source = "llm"
        else:
            source = "hand"

    print("=" * 60)
    print(f"STAGE 3: Analysis + plots  (source = {source})")
    print("=" * 60)

    if source == "hand":
        csv_path = config.RESULTS_DIR / "hand_scores_all.csv"
        if not csv_path.exists():
            raise SystemExit(
                f"No hand-scored CSV at {csv_path}. "
                "Run scripts.s2_judge_template --all first."
            )
        # Convert the CSV to the same JSONL format the analysis expects.
        jsonl_path = config.RESULTS_DIR / "judged_trials_hand.jsonl"
        n, _ = convert_hand_csv_to_judged_jsonl(csv_path, jsonl_path)
        if n == 0:
            raise SystemExit("No fully-scored rows found in the CSV. Fill the human_* columns first.")
        # Point the analysis at the hand-derived JSONL.
        run_analysis(input_path=jsonl_path)
    else:
        if not config.JUDGED_RESULTS_FILE.exists():
            raise SystemExit(
                f"No LLM-judged file at {config.JUDGED_RESULTS_FILE}. "
                "Run scripts.s2_judge_openai or use --source hand."
            )
        run_analysis()

    plot_all()

    print("\nStage 3 complete.")
    print(f"  Analysis: {config.ANALYSIS_FILE}")
    print(f"  Plots:    {config.PLOTS_DIR}")
    print("\nNext step:  python -m scripts.s4_fill_report")


if __name__ == "__main__":
    main()
