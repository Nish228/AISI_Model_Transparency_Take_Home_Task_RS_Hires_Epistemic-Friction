"""
Auto-fill the report template with values from analysis.json.

Usage:
  python -m scripts.fill_report

This reads results/analysis.json and writes report/report_filled.md with all
{{PLACEHOLDER}} tokens replaced. You then read it, edit any prose you want to
sharpen, and convert to PDF or paste into a Google Doc.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


def fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%"


def fmt_pct_ci(triple):
    if not triple:
        return "n/a"
    pt, lo, hi = triple
    if pt is None:
        return "n/a"
    return f"{pt*100:.0f}% [{lo*100:.0f}–{hi*100:.0f}]"


def fmt_delta(x):
    if x is None:
        return "n/a"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.0f} pp"


def interpret_outcome(analysis):
    """Pick the most likely outcome label (A/B/C/D) from the data."""
    conf = analysis["confound_check"]

    deltas = [
        v["delta_destructive_minus_benign"]
        for v in conf.values()
        if v["delta_destructive_minus_benign"] is not None
    ]
    if not deltas:
        return "Insufficient data to interpret outcome."

    avg_delta = sum(deltas) / len(deltas)

    fram = analysis["framing_effect"]
    fram_deltas = [
        v["delta_natural_minus_bare"]
        for v in fram.values()
        if v["delta_natural_minus_bare"] is not None
    ]
    avg_framing_delta = sum(fram_deltas) / len(fram_deltas) if fram_deltas else 0

    if avg_delta > 0.20:
        return (
            "**Outcome B (suspicion is largely rationalization).** "
            f"Verbalized evaluation-awareness was {fmt_delta(avg_delta)} higher "
            "under destructive actions than benign ones, averaged across "
            "(severity × framing) pairs. This pattern is consistent with the "
            "Turpin et al. (2023) prediction that verbalized reasoning often "
            "post-hoc rationalizes a decision driven by other factors — here, "
            "ordinary safety-refusal."
        )
    elif abs(avg_delta) <= 0.10 and avg_framing_delta < -0.10:
        return (
            "**Outcome C (framing dominates).** "
            f"The benign-vs-destructive delta was small ({fmt_delta(avg_delta)}), "
            f"but natural framing reduced evaluation-awareness by "
            f"{fmt_delta(avg_framing_delta)}. Plausible deployment context "
            "suppresses verbalized awareness regardless of action harm — "
            "directly relevant to how naturalistic evaluations should be designed."
        )
    elif abs(avg_delta) <= 0.10:
        return (
            "**Outcome A (suspicion is real and decoupled from action harm).** "
            f"Verbalized evaluation-awareness was approximately equal across "
            f"benign and destructive actions ({fmt_delta(avg_delta)} mean delta), "
            "suggesting the model is tracking context-inconsistency itself "
            "rather than rationalizing safety-refusal."
        )
    else:
        return (
            f"**Outcome D / mixed.** Action-harm delta = {fmt_delta(avg_delta)}, "
            f"framing delta = {fmt_delta(avg_framing_delta)}. The decoupled "
            "measurements show a complex pattern — see Figure 2 for the full "
            "picture. The implications for CoT-based monitoring depend on the "
            "specific co-occurrence structure of suspicion and compliance."
        )


def main():
    with config.ANALYSIS_FILE.open() as f:
        analysis = json.load(f)

    tmpl_path = config.REPORT_DIR / "report_template.md"
    out_path = config.REPORT_DIR / "report_filled.md"

    with tmpl_path.open() as f:
        tmpl = f.read()

    # Per-cell rates.
    cells = analysis["per_cell"]
    cell_table_rows = []
    for cell, d in sorted(cells.items()):
        cell_table_rows.append(
            f"| {cell} | {d['n']} | "
            f"{fmt_pct_ci(d['noticed_discrepancy_rate'])} | "
            f"{fmt_pct_ci(d['rationalized_anomaly_rate'])} | "
            f"{fmt_pct_ci(d['eval_awareness_rate'])} | "
            f"{fmt_pct_ci(d['compliance_rate'])} |"
        )
    cell_table = "\n".join(cell_table_rows)

    conf = analysis["confound_check"]
    conf_rows = []
    for key, v in sorted(conf.items()):
        conf_rows.append(
            f"| {key} | {fmt_pct(v['benign_rate'])} | "
            f"{fmt_pct(v['destructive_rate'])} | "
            f"{fmt_delta(v['delta_destructive_minus_benign'])} |"
        )
    conf_table = "\n".join(conf_rows)

    fram = analysis["framing_effect"]
    fram_rows = []
    for key, v in sorted(fram.items()):
        fram_rows.append(
            f"| {key} | {fmt_pct(v['bare_rate'])} | "
            f"{fmt_pct(v['natural_rate'])} | "
            f"{fmt_delta(v['delta_natural_minus_bare'])} |"
        )
    fram_table = "\n".join(fram_rows)

    filled = tmpl.replace("{{N_TRIALS}}", str(analysis["n_trials"]))
    filled = filled.replace("{{CELL_TABLE}}", cell_table)
    filled = filled.replace("{{CONFOUND_TABLE}}", conf_table)
    filled = filled.replace("{{FRAMING_TABLE}}", fram_table)
    filled = filled.replace("{{OUTCOME_INTERPRETATION}}", interpret_outcome(analysis))

    with out_path.open("w") as f:
        f.write(filled)

    print(f"Wrote {out_path}")
    print("Open it, polish prose if you want, then convert to PDF or paste into a Google Doc.")


if __name__ == "__main__":
    main()
