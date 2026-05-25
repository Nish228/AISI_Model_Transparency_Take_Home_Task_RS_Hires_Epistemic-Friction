"""
STAGE 1.5: Generate human-readable views of raw_trials.jsonl.

Produces four artifacts (none of which modify raw_trials.jsonl):

  1. results/trials_table.csv       - wide spreadsheet, one row per trial,
                                      columns ordered by conversation flow,
                                      human-readable labels ("User Prompt 1",
                                      "Model Answer 1", etc.)
  2. results/trials_readable/*.md   - one .md per trial for screen reading
  3. results/trials_pretty/*.json   - one pretty JSON per trial (easier to
                                      read than raw JSONL)
  4. results/HOW_TO_READ.md         - primer for someone seeing this for the
                                      first time, explaining what a trial is,
                                      why we have 3 turns, etc.

Usage:
  python -m scripts.s1b_view_trials                # all four
  python -m scripts.s1b_view_trials --csv-only
  python -m scripts.s1b_view_trials --md-only
  python -m scripts.s1b_view_trials --json-only
  python -m scripts.s1b_view_trials --readme-only
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viewer import (
    export_csv_table,
    export_markdown_per_trial,
    export_pretty_json_per_trial,
    export_how_to_read,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--md-only", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--readme-only", action="store_true")
    args = parser.parse_args()

    flags = [args.csv_only, args.md_only, args.json_only, args.readme_only]
    if sum(flags) > 1:
        raise SystemExit("Pick at most one --*-only flag.")

    do_all = not any(flags)

    if do_all or args.csv_only:
        export_csv_table()
    if do_all or args.md_only:
        export_markdown_per_trial()
    if do_all or args.json_only:
        export_pretty_json_per_trial()
    if do_all or args.readme_only:
        export_how_to_read()

    if do_all:
        print()
        print("All readable views generated.")
        print("Tell first-time readers to open: results/HOW_TO_READ.md")


if __name__ == "__main__":
    main()
