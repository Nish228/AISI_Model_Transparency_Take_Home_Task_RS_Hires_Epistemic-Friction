"""
STAGE 1 (OPTION B): Run the experiment trials against your ByteDance model
hosted on a GPU server via a Cloudflare tunnel.

This script ONLY runs the 240 trials. It does NOT score them.
You do NOT need an OpenAI key for this step.

Output: results/raw_trials.jsonl

Usage:
  export BYTEDANCE_BASE_URL="https://your-tunnel.trycloudflare.com/v1"
  export BYTEDANCE_API_KEY="EMPTY"     # or whatever your server expects
  python -m scripts.s1_run_experiment_bytedance

After this you can:
  - Stop here and hand-score every trial yourself, OR
  - Add OpenAI key later and run scripts.s2_judge_openai for LLM scoring, OR
  - Run scripts.s2_judge_template to get a hand-scoring CSV
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_client import ChatClient
from src.harness import run_experiment


# ======================================================================
# >>> CONFIGURE YOUR GPU SERVER ENDPOINT HERE <<<
# ======================================================================
# Your Cloudflare tunnel URL. Must include /v1 (OpenAI-compatible path prefix).
# Example: "https://my-tunnel.trycloudflare.com/v1"
BYTEDANCE_BASE_URL = os.environ.get(
    "BYTEDANCE_BASE_URL",
    "PASTE_YOUR_CLOUDFLARE_TUNNEL_URL_HERE/v1",
)

# API key your inference server expects. For vLLM with --api-key set, use that.
# For an unsecured server, use any non-empty string like "EMPTY".
BYTEDANCE_API_KEY = os.environ.get("BYTEDANCE_API_KEY", "EMPTY")

# Model name. For a vLLM server, this is the --model argument the server was
# launched with (usually a HuggingFace repo name or a local path).
BYTEDANCE_MODEL = "ByteDance-Seed/Seed-OSS-36B-Instruct"

# Cloudflare Access headers -- only if your tunnel is behind CF Access.
# Otherwise leave as empty strings.
CF_ACCESS_CLIENT_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_ACCESS_CLIENT_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")


def main():
    if BYTEDANCE_BASE_URL.startswith("PASTE_"):
        raise SystemExit("Set BYTEDANCE_BASE_URL env var or edit the script.")

    extra_headers = {}
    if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
        extra_headers["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
        extra_headers["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET

    print("=" * 60)
    print("STAGE 1: Experiment trials -- ByteDance/Cloudflare backend")
    print(f"Endpoint    : {BYTEDANCE_BASE_URL}")
    print(f"Model (MUT) : {BYTEDANCE_MODEL}")
    print("=" * 60)

    client = ChatClient(
        backend="compatible",
        api_key=BYTEDANCE_API_KEY,
        model=BYTEDANCE_MODEL,
        base_url=BYTEDANCE_BASE_URL,
        extra_headers=extra_headers if extra_headers else None,
    )

    run_experiment(client)

    print("\nStage 1 complete. Raw trials saved to results/raw_trials.jsonl")
    print("\nNext step (pick one):")
    print("  - LLM judge (needs OpenAI key):  python -m scripts.s2_judge_openai")
    print("  - Hand-judge (no key needed):    python -m scripts.s2_judge_template")


if __name__ == "__main__":
    main()
