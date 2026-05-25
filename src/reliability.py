"""
Reliability check.

After judging is done, sample N trials per cell, hand-score them in a CSV,
and run this script to compute Cohen's kappa between your scores and the
judge's. If kappa < 0.7 on the headline DV, revise the rubric and re-judge.

Usage:
  1. Run sample_for_human_review() -> writes spotcheck_template.csv
  2. Fill the human_* columns by hand.
  3. Run compute_kappa() -> prints kappas.
"""

import json
import csv
import random
from pathlib import Path

from . import config


def sample_for_human_review(out_csv: Path = None) -> Path:
    out_csv = out_csv or (config.RESULTS_DIR / "spotcheck_template.csv")

    judged = []
    with config.JUDGED_RESULTS_FILE.open() as f:
        for line in f:
            try:
                judged.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Stratified sample: HUMAN_SPOTCHECK_PER_CELL per cell.
    by_cell = {}
    for r in judged:
        key = f"{r['severity']}-{r['framing']}-{r['action']}"
        by_cell.setdefault(key, []).append(r)

    rng = random.Random(config.RANDOM_SEED)
    picks = []
    for cell, rows in by_cell.items():
        rng.shuffle(rows)
        picks.extend(rows[:config.HUMAN_SPOTCHECK_PER_CELL])

    # Pull raw transcripts so the human can read them.
    raw_lookup = {}
    with config.RAW_RESULTS_FILE.open() as f:
        for line in f:
            try:
                t = json.loads(line)
                raw_lookup[t["trial_id"]] = t
            except json.JSONDecodeError:
                continue

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "cell",
            "judge_noticed", "judge_rationalized", "judge_eval_aware", "judge_compliance",
            "human_noticed_discrepancy", "human_rationalized", "human_eval_aware", "human_compliance",
            "turn_2_text_preview",
        ])
        for r in picks:
            tid = r["trial_id"]
            preview = ""
            if tid in raw_lookup:
                preview = raw_lookup[tid]["turn_2"]["text"][:400].replace("\n", " ")
            writer.writerow([
                tid,
                f"{r['severity']}-{r['framing']}-{r['action']}",
                r["scores"].get("noticed_discrepancy"),
                r["scores"].get("rationalized_anomaly"),
                r["scores"].get("verbalized_eval_awareness"),
                r["scores"].get("action_compliance"),
                "", "", "", "",
                preview,
            ])
    print(f"Wrote {out_csv}. Fill the human_* columns by hand, then run compute_kappa().")
    return out_csv


def _kappa(a, b):
    """Cohen's kappa between two lists of equal length."""
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    cats = sorted(set(a) | set(b))
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    p_exp = 0
    for c in cats:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for x in b if x == c) / n
        p_exp += pa * pb
    if p_exp == 1:
        return 1.0
    return (p_obs - p_exp) / (1 - p_exp)


def compute_kappa(csv_path: Path = None):
    csv_path = csv_path or (config.RESULTS_DIR / "spotcheck_template.csv")
    judge_noticed, human_noticed = [], []
    judge_rat, human_rat = [], []
    judge_ea, human_ea = [], []
    judge_c, human_c = [], []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("human_eval_aware", "").strip():
                continue
            judge_noticed.append(str(row.get("judge_noticed", "")).strip().lower())
            human_noticed.append(str(row.get("human_noticed_discrepancy", "")).strip().lower())
            judge_rat.append(str(row.get("judge_rationalized", "")).strip().lower())
            human_rat.append(str(row.get("human_rationalized", "")).strip().lower())
            judge_ea.append(str(row.get("judge_eval_aware", "")).strip().lower())
            human_ea.append(str(row.get("human_eval_aware", "")).strip().lower())
            judge_c.append(str(row.get("judge_compliance", "")).strip().lower())
            human_c.append(str(row.get("human_compliance", "")).strip().lower())

    if not judge_ea:
        print("No human-scored rows found. Fill the CSV first.")
        return

    print(f"N hand-scored: {len(judge_ea)}")
    print(f"kappa(noticed_discrepancy)  = {_kappa(judge_noticed, human_noticed):.3f}")
    print(f"kappa(rationalized_anomaly) = {_kappa(judge_rat, human_rat):.3f}")
    print(f"kappa(eval_awareness)       = {_kappa(judge_ea, human_ea):.3f}")
    print(f"kappa(compliance)           = {_kappa(judge_c, human_c):.3f}")
    print("Aim for >= 0.7 on the headline DV (eval_awareness).")
