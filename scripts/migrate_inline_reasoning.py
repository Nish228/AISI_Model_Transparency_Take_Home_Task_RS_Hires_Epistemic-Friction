"""
One-off migration: post-process an existing raw_trials.jsonl to extract
inline reasoning tags (<seed:think>, <think>, etc.) into the reasoning field.

Useful if you ran stage 1 before the inline-reasoning fix landed in
src/api_client.py. After running this, your old data is in the same format
as new runs would produce.

Usage:
  python -m scripts.migrate_inline_reasoning
  # or specify a different file:
  python -m scripts.migrate_inline_reasoning --input some_old.jsonl --output cleaned.jsonl
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.api_client import split_inline_reasoning


def migrate_record(rec):
    """Apply inline-reasoning extraction to all three turns of a single trial."""
    for turn_key in ("turn_1", "turn_2", "turn_3"):
        turn = rec.get(turn_key)
        if not turn:
            continue
        text = turn.get("text", "")
        existing_reasoning = turn.get("reasoning", "")
        if existing_reasoning:
            # Already has a reasoning field -- leave it alone.
            continue
        clean, extracted = split_inline_reasoning(text)
        if extracted:
            turn["text"] = clean
            turn["reasoning"] = extracted
        # Flag truncation if there's no finish_reason field yet.
        if "finish_reason" not in turn:
            # Heuristic: if text doesn't end in punctuation, probably truncated.
            t = clean.rstrip()
            if t and not t[-1] in '.!?)]"`*:':
                turn["finish_reason"] = "length"
            else:
                turn["finish_reason"] = "stop"
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(config.RAW_RESULTS_FILE),
        help=f"Input JSONL (default: {config.RAW_RESULTS_FILE})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. Default: overwrite input (a .bak backup is made).",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    out = Path(args.output) if args.output else inp
    overwrite = args.output is None

    if overwrite:
        bak = inp.with_suffix(inp.suffix + ".bak")
        bak.write_bytes(inp.read_bytes())
        print(f"Backed up original to {bak}")

    records = []
    with inp.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records)} records.")
    extracted_count = 0
    truncated_count = 0
    for rec in records:
        before = sum(1 for k in ("turn_1", "turn_2", "turn_3")
                     if rec.get(k, {}).get("reasoning"))
        rec = migrate_record(rec)
        after = sum(1 for k in ("turn_1", "turn_2", "turn_3")
                    if rec.get(k, {}).get("reasoning"))
        extracted_count += (after - before)
        truncated_count += sum(
            1 for k in ("turn_1", "turn_2", "turn_3")
            if rec.get(k, {}).get("finish_reason") == "length"
        )

    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Extracted inline reasoning from {extracted_count} turns.")
    print(f"Flagged {truncated_count} turns as truncated (finish_reason='length').")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
