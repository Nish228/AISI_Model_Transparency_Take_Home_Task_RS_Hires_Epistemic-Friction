"""
Central configuration for Project Epistemic Friction (PEF).

All knobs live here. Edit this file, not the others, when you want to change
sample size, model names, paths, or factor structure.

RUN_LABEL: set the PEF_RUN env var to put outputs into a per-model subdir,
e.g. PEF_RUN=bytedance -> results/bytedance/, PEF_RUN=openai -> results/openai/.
This keeps multiple model runs from clobbering each other.
"""

import os
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent

# Per-run subdirectory. If PEF_RUN is unset, falls back to results/ root
# (the original behavior, so existing scripts keep working).
RUN_LABEL = os.environ.get("PEF_RUN", "").strip()

PROMPTS_DIR = ROOT / "prompts"
_RESULTS_BASE = ROOT / "results"
RESULTS_DIR = (_RESULTS_BASE / RUN_LABEL) if RUN_LABEL else _RESULTS_BASE
PLOTS_DIR = (ROOT / "plots" / RUN_LABEL) if RUN_LABEL else (ROOT / "plots")
REPORT_DIR = ROOT / "report"

for d in (RESULTS_DIR, PLOTS_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Experiment factors (2 x 2 x 2 = 8 cells) ----------
# Note: factor names kept as-is for code compatibility, but the MEANINGS
# have shifted to the inferential design:
#
# SEVERITY = "discrepancy magnitude":
#   low  -> small violation of model's inferred rule (~5% off)
#   high -> large violation of model's inferred rule (~50%+ off)
#
# FRAMING:
#   bare    -> no explanation of why anomalies happen
#   natural -> framing prefix saying operational anomalies are routine
#
# ACTION (the critical decoupling factor):
#   benign     -> harmless follow-up action
#   destructive -> action with real-world stakes
SEVERITY_LEVELS = ["low", "high"]
FRAMING_LEVELS = ["bare", "natural"]
ACTION_LEVELS = ["benign", "destructive"]

SCENARIOS = ["tax", "logs", "inventory", "security"]

# ---------- Sample size ----------
# 8 cells x 10 = 80 trials. Reduced from 30 to fit a 4-hour sprint while
# still giving meaningful per-cell point estimates (effect sizes only;
# significance testing on n=10/cell is underpowered, report descriptively).
TRIALS_PER_CELL = 10
HUMAN_SPOTCHECK_PER_CELL = 3  # 8 x 3 = 24 trials hand-scored for kappa

# ---------- Model under test (MUT) defaults ----------
# Override per-script if needed.
MUT_TEMPERATURE = 0.7
# Reasoning models (ByteDance Seed-OSS, DeepSeek R1, o-series) emit long
# inline reasoning before their answer. 1500 was getting truncated in the
# pilot. 4000 gives headroom for both reasoning and a complete answer.
MUT_MAX_TOKENS = 4000

# ---------- LLM judge ----------
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500

# ---------- API hygiene ----------
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 120

# ---------- Output files ----------
RAW_RESULTS_FILE = RESULTS_DIR / "raw_trials.jsonl"
JUDGED_RESULTS_FILE = RESULTS_DIR / "judged_trials.jsonl"
ANALYSIS_FILE = RESULTS_DIR / "analysis.json"

# ---------- Reproducibility ----------
RANDOM_SEED = 42
