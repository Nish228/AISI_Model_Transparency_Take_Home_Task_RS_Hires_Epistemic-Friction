"""
STAGE 1 (Azure OpenAI behind HTTP Basic Auth proxy): Run the experiment
against an Azure OpenAI endpoint that's gated by a username+password.

This is for the case where a company gives you:
  - A hostname (like https://<resource>.cognitiveservices.azure.com/openai/responses)
  - A username
  - A password
  ...per model you want to access.

The script uses HTTP Basic Auth, NOT api-key header.

Env vars:
  AZURE_PROXY_ENDPOINT   - full endpoint URL (up to and including /openai/responses)
  AZURE_PROXY_USERNAME   - username for HTTP Basic Auth
  AZURE_PROXY_PASSWORD   - password for HTTP Basic Auth
  AZURE_PROXY_API_VERSION - default "2025-04-01-preview"
  AZURE_PROXY_DEPLOYMENT  - deployment / model name to put in the request body.
                            Sometimes the proxy ignores this and uses the
                            username to determine routing -- if so, set this
                            to whatever value their docs say (often the same
                            as the username, or a model name like "gpt-5.4-mini").

  PEF_RUN=azure_<model>   (recommended; outputs go to results/azure_<model>/)

Usage:
  export PEF_RUN=azure_gpt54mini
  export AZURE_PROXY_ENDPOINT="https://james-mihazfos-australiaeast.cognitiveservices.azure.com/openai/responses"
  export AZURE_PROXY_USERNAME="<their username>"
  export AZURE_PROXY_PASSWORD="<their password>"
  export AZURE_PROXY_DEPLOYMENT="gpt-5.4-mini"    # or whatever they specify
  python -m scripts.s1_run_experiment_azure_basic
"""

import os
import sys
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_client import ChatClient
from src.harness import run_experiment


# ======================================================================
# >>> CONFIGURE HERE (or set env vars above) <<<
# ======================================================================
AZURE_ENDPOINT    = os.environ.get("AZURE_PROXY_ENDPOINT",
                                   "PASTE_ENDPOINT_HERE")
AZURE_USERNAME    = os.environ.get("AZURE_PROXY_USERNAME",
                                   "PASTE_USERNAME_HERE")
AZURE_PASSWORD    = os.environ.get("AZURE_PROXY_PASSWORD",
                                   "PASTE_PASSWORD_HERE")
AZURE_DEPLOYMENT  = os.environ.get("AZURE_PROXY_DEPLOYMENT",
                                   "PASTE_DEPLOYMENT_OR_MODEL_NAME_HERE")
AZURE_API_VERSION = os.environ.get("AZURE_PROXY_API_VERSION",
                                   "2025-04-01-preview")

REASONING_EFFORT = "medium"


def main():
    if AZURE_ENDPOINT.startswith("PASTE_"):
        raise SystemExit("Set AZURE_PROXY_ENDPOINT env var or edit the script.")
    if AZURE_USERNAME.startswith("PASTE_"):
        raise SystemExit("Set AZURE_PROXY_USERNAME env var or edit the script.")
    if AZURE_PASSWORD.startswith("PASTE_"):
        raise SystemExit("Set AZURE_PROXY_PASSWORD env var or edit the script.")
    if AZURE_DEPLOYMENT.startswith("PASTE_"):
        raise SystemExit("Set AZURE_PROXY_DEPLOYMENT env var or edit the script.")

    endpoint = AZURE_ENDPOINT.split("?")[0]

    # Build the HTTP Basic Auth header value: "Basic <base64(user:pass)>"
    creds = f"{AZURE_USERNAME}:{AZURE_PASSWORD}"
    basic_b64 = base64.b64encode(creds.encode("utf-8")).decode("ascii")

    print("=" * 60)
    print("STAGE 1: Experiment trials -- Azure OpenAI via HTTP Basic Auth proxy")
    print(f"Endpoint:         {endpoint}")
    print(f"Username:         {AZURE_USERNAME}")
    print(f"Deployment:       {AZURE_DEPLOYMENT}")
    print(f"API version:      {AZURE_API_VERSION}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    if os.environ.get("PEF_RUN"):
        print(f"Run label:        {os.environ['PEF_RUN']}  (outputs in results/{os.environ['PEF_RUN']}/)")
    print("=" * 60)

    client = ChatClient(
        backend="openai",
        api_key=basic_b64,                       # encoded creds go in the auth header
        model=AZURE_DEPLOYMENT,
        use_responses_api=True,
        reasoning_effort=REASONING_EFFORT,
        endpoint_override=endpoint,
        query_params={"api-version": AZURE_API_VERSION},
        auth_header_name="Authorization",
        auth_header_prefix="Basic ",             # HTTP Basic Auth scheme
    )

    run_experiment(client)

    print("\nStage 1 complete.")
    print("\nNext steps:")
    print("  - Inspect: python -m scripts.s1b_view_trials")
    print("  - Score:   python -m scripts.s2_judge_openai  (or s2_judge_template --all)")


if __name__ == "__main__":
    main()
