# AISI Transparency Team Research Scientist Take-home Task - Epistemic Friction 

Studies whether frontier models verbalize evaluation-awareness because they
genuinely notice inconsistent system prompts, or because they're rationalizing
ordinary safety-refusal.

## The pipeline is four separate stages

You run them one at a time, in order. Each stage writes to disk; the next one
reads what the previous one wrote. Stop and resume whenever you want.

```
STAGE 1                  STAGE 2                  STAGE 3              STAGE 4
Run 240 trials   ───▶   Score them      ───▶   Analyze + plot ───▶  Auto-fill
(MUT model)              (LLM or hand)                                 the report
```

The **judging step is optional in the sense that you choose how to do it**.
You can use the LLM judge (needs OpenAI key) OR hand-score every trial
yourself (no API needed) OR both (and we'll compute κ between them).

## Stage 1 — Run the experiment trials

Pick the runner that matches your model under test.

### OpenAI model

```bash
pip install -r requirements.txt
export PEF_RUN=azure_gpt54
export AZURE_OPENAI_ENDPOINT="<hostname>"
export AZURE_OPENAI_API_KEY="<your password>"
export AZURE_OPENAI_DEPLOYMENT="<username>"
python -m scripts.s1_run_experiment_azure
```

### Note on reasoning-model output

Models like ByteDance Seed-OSS and DeepSeek R1 emit their reasoning inline
using `<seed:think>...</seed:think>` or `<think>...</think>` tags. The API
client extracts these automatically into a separate `reasoning` field.

If you have an **old** `raw_trials.jsonl` from before this extraction was
added, run the one-off migration to clean it up retroactively:

```bash
python -m scripts.migrate_inline_reasoning
```

This makes a `.bak` copy of the original, extracts inline reasoning from all
three turns, and flags truncated responses with `finish_reason: "length"`.

## Stage 1.5 (optional) — Human-readable views

JSONL is for code. To actually *read* what the model said, run:

```bash
python -m scripts.s1b_view_trials
```

This produces two outputs (without modifying `raw_trials.jsonl`):

- **`results/trials_table.csv`** — one row per trial, every field as a
  column. Opens in Excel / Sheets / Numbers. Includes skim-flags:
  `t2_likely_refused`, `t2_truncated`, `any_reasoning_mentions_change`.
- **`results/trials_readable/<trial_id>.md`** — one markdown file per
  trial with the full conversation cleanly laid out. Start at
  `_INDEX.md`.

Use `--csv-only` or `--md-only` if you only want one of them.

## Stage 2 — Score the trials

Pick one (or do both — they're independent).

### Option A: LLM judge (needs OpenAI key)

```bash
export OPENAI_API_KEY="sk-..."
python -m scripts.s2_judge_openai
```

Output: `results/judged_trials.jsonl`

### Option B: Hand-score every trial yourself (no API needed)

```bash
python -m scripts.s2_judge_template --all
```

This writes `results/hand_scores_all.csv` with all 240 trials and a column
for each of the three DVs. Open in Excel/Sheets, fill the `human_eval_aware`,
`human_compliance`, `human_ack` columns by hand, save.

### Optional: do both, then compute κ

If you ran the LLM judge in stage 2A, you can also export a 40-trial
spotcheck for reliability:

```bash
python -m scripts.s2_judge_template --spotcheck
# fill the human_* columns by hand in results/spotcheck_template.csv
python -c "from src.reliability import compute_kappa; compute_kappa()"
```

This prints Cohen's κ between your hand scores and the LLM judge. Aim for
κ ≥ 0.7 on the headline DV. If lower, the LLM judge is unreliable and you
should either revise the rubric in `src/judge.py` or rely on hand scores.

## Stage 3 — Analyze + plot

```bash
python -m scripts.s3_analyze
```

Auto-detects source. If `results/judged_trials.jsonl` exists, uses LLM judge
output. Otherwise uses the hand-scored CSV. You can also be explicit:

```bash
python -m scripts.s3_analyze --source llm
python -m scripts.s3_analyze --source hand
```

Outputs:
- `results/analysis.json` — per-cell rates, confound check, framing effect
- `plots/eval_awareness_by_cell.png`
- `plots/confound_check.png` ← the headline figure
- `plots/framing_effect.png`

## Stage 4 — Fill the 2-page report

```bash
python -m scripts.s4_fill_report
```

Reads `results/analysis.json`, fills every `{{PLACEHOLDER}}` in the report
template, auto-picks the most likely outcome interpretation (A/B/C/D) from
the data.

Output: `report/report_filled.md`. Open, polish prose, convert to PDF or
paste into a Google Doc.

## Project layout

```
pef_pipeline/
├── src/
│   ├── config.py          # all knobs (sample size, paths, retry settings)
│   ├── prompts.py         # 3 scenarios × 2 sev × 2 fram × 2 act = 24 prompt setups
│   ├── api_client.py      # OpenAI + OpenAI-compatible endpoints
│   ├── harness.py         # the 3-turn dialogue runner (resumable)
│   ├── judge.py           # LLM judge rubric and scoring
│   ├── analysis.py        # per-cell rates, bootstrap CIs, confound check
│   ├── plotting.py        # three figures
│   └── reliability.py     # Cohen's κ for spotcheck
├── scripts/
│   ├── s1_run_experiment_openai.py        # Stage 1A
│   ├── s1_run_experiment_bytedance.py     # Stage 1B
│   ├── s2_judge_openai.py                 # Stage 2A (LLM judge)
│   ├── s2_judge_template.py               # Stage 2B (hand-judge CSV)
│   ├── s3_analyze.py                      # Stage 3
│   └── s4_fill_report.py                  # Stage 4
├── results/               # raw_trials.jsonl, judged_trials.jsonl, analysis.json
├── plots/
└── report/
    ├── report_template.md
    └── report_filled.md   # generated by stage 4
```

## Common workflows

**Just see if it works (pilot):**

Edit `src/config.py`, set `TRIALS_PER_CELL = 2`. Run stages 1 → 2 → 3 → 4.
Total time: ~5 minutes for OpenAI MUT, maybe longer if your GPU server is slow.
Revert `TRIALS_PER_CELL = 30` before the real run.

**Run with ByteDance, judge with LLM later:**

Stage 1B → (some time later, add OpenAI key) → Stage 2A → Stage 3 → Stage 4.

**Run with ByteDance, judge entirely by hand (no OpenAI key ever):**

Stage 1B → Stage 2B (`--all`) → Stage 3 → Stage 4.

**Resume an interrupted run:**

Just rerun the stage. The harness and judge both skip trials they've already
completed (by `trial_id`). No flags needed.

