"""
Experiment harness.

Runs the 3-turn dialogue for each trial. Writes one JSON line per trial to
the raw results file. Resilient: if you interrupt and rerun, it will skip
trials that are already logged (by trial_id).
"""

import json
import random
from pathlib import Path
from typing import Iterator
from itertools import product

from . import config
from .prompts import build_prompts, SCENARIO_REGISTRY
from .api_client import ChatClient


def all_cells() -> Iterator[dict]:
    """Yield one dict per (severity, framing, action) cell."""
    for sev, fram, act in product(
        config.SEVERITY_LEVELS,
        config.FRAMING_LEVELS,
        config.ACTION_LEVELS,
    ):
        yield {"severity": sev, "framing": fram, "action": act}


def cell_id(cell: dict) -> str:
    return f"{cell['severity']}-{cell['framing']}-{cell['action']}"


def trial_id(cell: dict, trial_idx: int, scenario: str) -> str:
    return f"{cell_id(cell)}__{scenario}__{trial_idx:03d}"


def already_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["trial_id"])
            except Exception:
                continue
    return done


def run_single_trial(client: ChatClient, scenario: str, cell: dict) -> dict:
    """Run one 3-turn dialogue. Returns the full record."""
    p = build_prompts(scenario, cell["severity"], cell["framing"], cell["action"])

    # ----- Turn 1 -----
    messages = [
        {"role": "system", "content": p["sys_t1"]},
        {"role": "user", "content": p["user_t1"]},
    ]
    r1 = client.chat(messages, temperature=config.MUT_TEMPERATURE,
                     max_tokens=config.MUT_MAX_TOKENS)

    # ----- Turn 2: swap system prompt, append turn-1 assistant reply, ask -----
    messages = [
        {"role": "system", "content": p["sys_t2"]},
        {"role": "user", "content": p["user_t1"]},
        {"role": "assistant", "content": r1["text"]},
        {"role": "user", "content": p["user_t2"]},
    ]
    r2 = client.chat(messages, temperature=config.MUT_TEMPERATURE,
                     max_tokens=config.MUT_MAX_TOKENS)

    # ----- Turn 3: neutral interrogation -----
    messages.append({"role": "assistant", "content": r2["text"]})
    messages.append({"role": "user", "content": p["user_t3"]})
    r3 = client.chat(messages, temperature=config.MUT_TEMPERATURE,
                     max_tokens=config.MUT_MAX_TOKENS)

    return {
        "scenario": scenario,
        "severity": cell["severity"],
        "framing": cell["framing"],
        "action": cell["action"],
        "prompts": p,
        "turn_1": {
            "text": r1["text"],
            "reasoning": r1["reasoning"],
            "finish_reason": r1.get("finish_reason", ""),
        },
        "turn_2": {
            "text": r2["text"],
            "reasoning": r2["reasoning"],
            "finish_reason": r2.get("finish_reason", ""),
        },
        "turn_3": {
            "text": r3["text"],
            "reasoning": r3["reasoning"],
            "finish_reason": r3.get("finish_reason", ""),
        },
    }


def run_experiment(client: ChatClient, output_path: Path = None) -> Path:
    """
    Run the full 2x2x2 x N_per_cell experiment.

    Trials are INTERLEAVED across cells -- we do round 1 of every cell first,
    then round 2 of every cell, etc. This way an interrupted run still gives
    representative coverage of all 8 cells rather than only the first few.

    Scenarios are rotated within each (cell, round) so all three appear
    roughly equally.
    """
    out = output_path or config.RAW_RESULTS_FILE
    done = already_done_ids(out)
    print(f"Found {len(done)} previously-completed trials. Will skip those.")

    cells = list(all_cells())

    with out.open("a") as f:
        for round_idx in range(config.TRIALS_PER_CELL):
            print(f"\n=== Round {round_idx + 1}/{config.TRIALS_PER_CELL} "
                  f"(all 8 cells) ===")
            for cell in cells:
                scenario = config.SCENARIOS[round_idx % len(config.SCENARIOS)]
                tid = trial_id(cell, round_idx, scenario)
                if tid in done:
                    continue

                print(f"  [{tid}] running...")
                try:
                    rec = run_single_trial(client, scenario, cell)
                except Exception as e:
                    print(f"  [{tid}] FAILED: {e}")
                    rec = {"error": str(e)}

                rec["trial_id"] = tid
                rec["scenario"] = scenario
                rec["severity"] = cell["severity"]
                rec["framing"] = cell["framing"]
                rec["action"] = cell["action"]

                f.write(json.dumps(rec) + "\n")
                f.flush()

    print(f"\nDone. Results in {out}")
    return out
