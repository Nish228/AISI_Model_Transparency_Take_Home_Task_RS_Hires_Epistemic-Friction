"""
STAGE 1 (Azure OpenAI variant): Run the experiment against an Azure OpenAI
deployment using the Responses API.

This is for the case where a company gives you:
  - A hostname like https://<resource>.cognitiveservices.azure.com
  - A deployment name
  - An api-key

Set these env vars:
  AZURE_OPENAI_ENDPOINT - the FULL endpoint URL up to /openai/responses,
    e.g. "https://james-mihazfos-australiaeast.cognitiveservices.azure.com/openai/responses"
  AZURE_OPENAI_API_KEY  - the credential they gave you
  AZURE_OPENAI_DEPLOYMENT - the deployment name (NOT the model name; this is
    whatever the resource owner configured, e.g. "gpt-5-mini-prod")
  AZURE_OPENAI_API_VERSION - default "2025-04-01-preview" (override if their
    docs specify something else)

  PEF_RUN=azure   (recommended, so outputs go to results/azure/)

Usage:
  export PEF_RUN=azure
  export AZURE_OPENAI_ENDPOINT="https://....cognitiveservices.azure.com/openai/responses"
  export AZURE_OPENAI_API_KEY="..."
  export AZURE_OPENAI_DEPLOYMENT="<their deployment name>"
  python -m scripts.s1_run_experiment_azure
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_client import ChatClient
from src.harness import run_experiment


# ======================================================================
# >>> CONFIGURE HERE (or set env vars above) <<<
# ======================================================================
AZURE_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT",
                                   "PASTE_AZURE_ENDPOINT_HERE")
AZURE_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY",
                                   "PASTE_AZURE_KEY_HERE")
AZURE_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT",
                                   "PASTE_DEPLOYMENT_NAME_HERE")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION",
                                   "2025-04-01-preview")

# Reasoning effort. Azure honors the same set as OpenAI:
# "none" / "low" / "medium" / "high" / "xhigh".
REASONING_EFFORT = "medium"


def main():
    if AZURE_ENDPOINT.startswith("PASTE_"):
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT env var or edit the script.")
    if AZURE_API_KEY.startswith("PASTE_"):
        raise SystemExit("Set AZURE_OPENAI_API_KEY env var or edit the script.")
    if AZURE_DEPLOYMENT.startswith("PASTE_"):
        raise SystemExit("Set AZURE_OPENAI_DEPLOYMENT env var or edit the script.")

    # Strip any trailing ?api-version=... if the user pasted the full URL.
    endpoint = AZURE_ENDPOINT.split("?")[0]

    print("=" * 60)
    print("STAGE 1: Experiment trials -- Azure OpenAI backend")
    print(f"Endpoint:         {endpoint}")
    print(f"Deployment:       {AZURE_DEPLOYMENT}")
    print(f"API version:      {AZURE_API_VERSION}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    if os.environ.get("PEF_RUN"):
        print(f"Run label:        {os.environ['PEF_RUN']}  (outputs in results/{os.environ['PEF_RUN']}/)")
    print("=" * 60)

    # Azure auth is via 'api-key' header (no Bearer prefix). The deployment
    # name goes into the "model" field of the JSON body.
    client = ChatClient(
        backend="openai",                       # use OpenAI-style payload
        api_key=AZURE_API_KEY,
        model=AZURE_DEPLOYMENT,                  # deployment name, not model name
        use_responses_api=True,
        reasoning_effort=REASONING_EFFORT,
        endpoint_override=endpoint,              # use the full Azure URL
        query_params={"api-version": AZURE_API_VERSION},
        auth_header_name="api-key",
        auth_header_prefix="",                   # raw key, no "Bearer "
    )

    run_experiment(client)

    print("\nStage 1 complete.")
    print("\nNext steps:")
    print("  - Inspect: python -m scripts.s1b_view_trials")
    print("  - Score:   python -m scripts.s2_judge_openai  (or s2_judge_template --all)")


if __name__ == "__main__":
    main()
