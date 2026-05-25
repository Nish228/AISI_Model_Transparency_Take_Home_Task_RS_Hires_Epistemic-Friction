"""
Analysis module.

Reads judged_trials.jsonl, computes per-cell rates for each DV, runs
bootstrap CIs, and writes analysis.json. The plotting module reads from this.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from . import config


def load_judged(path: Path = None) -> List[dict]:
    inp = path or config.JUDGED_RESULTS_FILE
    rows = []
    with inp.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if "_judge_error" in r.get("scores", {}):
                    continue
                if "_parse_error" in r.get("scores", {}):
                    continue
                rows.append(r)
            except json.JSONDecodeError:
                continue
    return rows


def cell_key(row: dict) -> str:
    return f"{row['severity']}-{row['framing']}-{row['action']}"


def bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
):
    if not values:
        return None, None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(((1 - confidence) / 2) * n_boot)
    hi_idx = int((1 - (1 - confidence) / 2) * n_boot) - 1
    point = sum(values) / n
    return point, means[lo_idx], means[hi_idx]


def aggregate(rows: List[dict]) -> Dict:
    """Compute per-cell rates for each DV with bootstrap CIs."""
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[cell_key(r)].append(r)

    out = {}
    for cell, trials in sorted(by_cell.items()):
        noticed = [
            1.0 if t["scores"].get("noticed_discrepancy") else 0.0
            for t in trials
        ]
        rationalized = [
            1.0 if t["scores"].get("rationalized_anomaly") else 0.0
            for t in trials
        ]
        eval_aware = [
            1.0 if t["scores"].get("verbalized_eval_awareness") else 0.0
            for t in trials
        ]
        compliance = [
            1.0 if t["scores"].get("action_compliance") else 0.0
            for t in trials
        ]

        sev, fram, act = cell.split("-")
        out[cell] = {
            "severity": sev,
            "framing": fram,
            "action": act,
            "n": len(trials),
            "noticed_discrepancy_rate": bootstrap_mean_ci(noticed),
            "rationalized_anomaly_rate": bootstrap_mean_ci(rationalized),
            "eval_awareness_rate": bootstrap_mean_ci(eval_aware),
            "compliance_rate": bootstrap_mean_ci(compliance),
        }
    return out


def confound_check(rows: List[dict]) -> Dict:
    """
    THE central question: does eval-awareness rate differ between benign and
    destructive actions, holding severity and framing constant?

    If destructive >> benign across all (severity, framing) pairs => the
    "suspicion" was largely a rationalization of safety-refusal.

    If benign ~ destructive => suspicion is tracking the inconsistency itself,
    independent of action harm.
    """
    by_sev_fram = defaultdict(lambda: {"benign": [], "destructive": []})
    for r in rows:
        key = (r["severity"], r["framing"])
        v = 1.0 if r["scores"].get("verbalized_eval_awareness") else 0.0
        by_sev_fram[key][r["action"]].append(v)

    out = {}
    for (sev, fram), arms in by_sev_fram.items():
        b_pt, b_lo, b_hi = bootstrap_mean_ci(arms["benign"])
        d_pt, d_lo, d_hi = bootstrap_mean_ci(arms["destructive"])
        out[f"{sev}-{fram}"] = {
            "benign_rate": b_pt,
            "benign_ci": [b_lo, b_hi],
            "destructive_rate": d_pt,
            "destructive_ci": [d_lo, d_hi],
            "delta_destructive_minus_benign": (
                (d_pt - b_pt) if (b_pt is not None and d_pt is not None) else None
            ),
            "n_benign": len(arms["benign"]),
            "n_destructive": len(arms["destructive"]),
        }
    return out


def framing_effect(rows: List[dict]) -> Dict:
    """Effect of natural vs bare framing on eval-awareness, holding others constant."""
    by_sev_act = defaultdict(lambda: {"bare": [], "natural": []})
    for r in rows:
        key = (r["severity"], r["action"])
        v = 1.0 if r["scores"].get("verbalized_eval_awareness") else 0.0
        by_sev_act[key][r["framing"]].append(v)

    out = {}
    for (sev, act), arms in by_sev_act.items():
        b_pt, *_ = bootstrap_mean_ci(arms["bare"])
        n_pt, *_ = bootstrap_mean_ci(arms["natural"])
        out[f"{sev}-{act}"] = {
            "bare_rate": b_pt,
            "natural_rate": n_pt,
            "delta_natural_minus_bare": (n_pt - b_pt) if (b_pt is not None and n_pt is not None) else None,
        }
    return out


def run_analysis(input_path: Path = None, output_path: Path = None) -> Path:
    rows = load_judged(input_path)
    print(f"Loaded {len(rows)} judged trials")

    result = {
        "n_trials": len(rows),
        "per_cell": aggregate(rows),
        "confound_check": confound_check(rows),
        "framing_effect": framing_effect(rows),
    }

    out = output_path or config.ANALYSIS_FILE
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Analysis written to {out}")
    return out
