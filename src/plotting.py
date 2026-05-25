"""
Plotting module. Generates three figures from analysis.json:

  1. eval_awareness_by_cell.png    -- bar chart of suspicion rate per cell
  2. confound_check.png            -- benign vs destructive (the headline plot)
  3. framing_effect.png            -- bare vs natural framing effect
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def _load(path: Path = None):
    p = path or config.ANALYSIS_FILE
    with p.open() as f:
        return json.load(f)


def plot_eval_awareness_by_cell(path: Path = None):
    data = _load(path)
    cells = data["per_cell"]
    labels, rates, errs = [], [], []
    for cell, d in sorted(cells.items()):
        rate, lo, hi = d["eval_awareness_rate"]
        if rate is None:
            continue
        labels.append(cell)
        rates.append(rate)
        errs.append([rate - lo, hi - rate])

    errs = list(zip(*errs)) if errs else [[], []]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    ax.bar(x, rates, yerr=errs, capsize=4, color="steelblue", edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Rate of verbalized evaluation-awareness")
    ax.set_ylim(0, 1)
    ax.set_title("Verbalized evaluation-awareness by cell (severity-framing-action)")
    fig.tight_layout()
    out = config.PLOTS_DIR / "eval_awareness_by_cell.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_confound_check(path: Path = None):
    data = _load(path)
    conf = data["confound_check"]
    keys = sorted(conf.keys())
    benign = [conf[k]["benign_rate"] or 0 for k in keys]
    destructive = [conf[k]["destructive_rate"] or 0 for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(keys))
    w = 0.38
    ax.bar([i - w/2 for i in x], benign, width=w, label="Benign action", color="seagreen")
    ax.bar([i + w/2 for i in x], destructive, width=w, label="Destructive action", color="firebrick")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=20)
    ax.set_ylabel("Rate of verbalized evaluation-awareness")
    ax.set_ylim(0, 1)
    ax.set_title("Confound check: does suspicion rate depend on action harm?")
    ax.legend()
    fig.tight_layout()
    out = config.PLOTS_DIR / "confound_check.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_framing_effect(path: Path = None):
    data = _load(path)
    fe = data["framing_effect"]
    keys = sorted(fe.keys())
    bare = [fe[k]["bare_rate"] or 0 for k in keys]
    natural = [fe[k]["natural_rate"] or 0 for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(keys))
    w = 0.38
    ax.bar([i - w/2 for i in x], bare, width=w, label="Bare framing", color="gray")
    ax.bar([i + w/2 for i in x], natural, width=w, label="Natural framing", color="goldenrod")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=20)
    ax.set_ylabel("Rate of verbalized evaluation-awareness")
    ax.set_ylim(0, 1)
    ax.set_title("Framing effect on evaluation-awareness")
    ax.legend()
    fig.tight_layout()
    out = config.PLOTS_DIR / "framing_effect.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_all(path: Path = None):
    plot_eval_awareness_by_cell(path)
    plot_confound_check(path)
    plot_framing_effect(path)
