"""
Unified chat API client.

Supports three modes:

  1. backend="openai", use_responses_api=False  (default)
        Calls OpenAI's /v1/chat/completions. Good for gpt-4o, gpt-4o-mini,
        and other non-reasoning OpenAI models.

  2. backend="openai", use_responses_api=True
        Calls OpenAI's /v1/responses. Required for reasoning models (o-series,
        gpt-5, gpt-5.4, gpt-5.5) when you want a visible reasoning summary.
        Reasoning summary is requested via reasoning={"effort": ..., "summary": "auto"}
        and appears as a separate "reasoning" content block in the response.

  3. backend="compatible"
        Calls any OpenAI-compatible /v1/chat/completions endpoint, e.g.
        ByteDance Seed-OSS served via vLLM behind a Cloudflare tunnel.

All three modes return the same dict shape:
  {"text": "...", "reasoning": "...", "finish_reason": "...", "raw": {...}}

so harness/judge/analysis code is unchanged.
"""

import time
import json
import re
import requests
from typing import List, Dict, Optional

from . import config


# Models like ByteDance Seed-OSS and DeepSeek R1 emit reasoning inline using
# XML-style tags rather than a separate API field. Pull those out so
# downstream code sees a clean response and a separate reasoning trace.
_INLINE_REASONING_PATTERNS = [
    re.compile(r"<seed:think>(.*?)</seed:think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE),
]


def split_inline_reasoning(text: str):
    """If the response embeds reasoning in tags, return (clean_text, reasoning).
    Otherwise return (text, "")."""
    if not text:
        return text, ""

    extracted = []
    cleaned = text

    for pat in _INLINE_REASONING_PATTERNS:
        matches = pat.findall(cleaned)
        if matches:
            extracted.extend(m.strip() for m in matches)
            cleaned = pat.sub("", cleaned)

    # Handle truncated-mid-reasoning: opened a tag but never closed.
    for open_tag in ("<seed:think>", "<think>", "<thinking>", "<reasoning>"):
        if open_tag.lower() in cleaned.lower():
            idx = cleaned.lower().find(open_tag.lower())
            extracted.append(cleaned[idx + len(open_tag):].strip())
            cleaned = cleaned[:idx]
            break

    return cleaned.strip(), "\n\n".join(extracted).strip()


class ChatClient:
    def __init__(
        self,
        backend: str,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        auth_header_name: str = "Authorization",
        auth_header_prefix: str = "Bearer ",
        extra_headers: Optional[Dict[str, str]] = None,
        use_responses_api: bool = False,
        reasoning_effort: str = "medium",
        endpoint_override: Optional[str] = None,
        query_params: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            backend: "openai" or "compatible"
            api_key: API key (pass via env var, not hardcoded)
            model: model name (or Azure deployment name)
            base_url: override the API base. For Azure, prefer endpoint_override.
            auth_header_name, auth_header_prefix: customise auth header.
                For Azure, set to "api-key" and "" respectively.
            extra_headers: optional extra headers.
            use_responses_api: if True, use Responses endpoint and request a
                reasoning summary.
            reasoning_effort: "none" / "low" / "medium" / "high" / "xhigh".
            endpoint_override: pass the FULL endpoint URL as-is, ignoring
                base_url and the /chat-completions vs /responses logic. Use
                this for Azure (where the URL has /openai/responses and
                query params).
            query_params: dict of query-string parameters appended to every
                request. Required for Azure (api-version=...).
        """
        self.backend = backend
        self.api_key = api_key
        self.model = model
        self.use_responses_api = use_responses_api
        self.reasoning_effort = reasoning_effort
        self.query_params = query_params or {}

        if use_responses_api and backend != "openai" and endpoint_override is None:
            raise ValueError(
                "use_responses_api=True requires backend='openai' OR endpoint_override set"
            )

        if endpoint_override is not None:
            self.endpoint = endpoint_override
            self.base_url = endpoint_override  # not really used, but kept
        else:
            if base_url is None:
                if backend == "openai":
                    base_url = "https://api.openai.com/v1"
                else:
                    raise ValueError("base_url required for 'compatible' backend without endpoint_override")
            self.base_url = base_url.rstrip("/")
            if use_responses_api:
                self.endpoint = f"{self.base_url}/responses"
            else:
                self.endpoint = f"{self.base_url}/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            auth_header_name: f"{auth_header_prefix}{api_key}",
        }
        if extra_headers:
            self.headers.update(extra_headers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> Dict:
        """Send a multi-turn request. Returns:
            {"text", "reasoning", "finish_reason", "raw"}
        """
        if self.use_responses_api:
            return self._chat_responses_api(messages, temperature, max_tokens)
        else:
            return self._chat_completions(messages, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Chat Completions path (gpt-4o, ByteDance, etc.)
    # ------------------------------------------------------------------

    def _chat_completions(self, messages, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post_with_retry(payload)

        choice = data["choices"][0]
        msg = choice["message"]
        text = msg.get("content", "") or ""

        # Some compatible servers expose reasoning in a separate field.
        reasoning = (
            msg.get("reasoning_content")
            or msg.get("reasoning")
            or ""
        )
        # Otherwise look for inline <think>/<seed:think> tags.
        if not reasoning:
            text, inline = split_inline_reasoning(text)
            if inline:
                reasoning = inline

        return {
            "text": text,
            "reasoning": reasoning,
            "finish_reason": choice.get("finish_reason", ""),
            "raw": data,
        }

    # ------------------------------------------------------------------
    # Responses API path (gpt-5.4 with reasoning summary, etc.)
    # ------------------------------------------------------------------

    def _chat_responses_api(self, messages, temperature, max_tokens):
        """
        Translate our messages list into the Responses API's input format.

        Chat Completions message format:
            [{"role": "system", "content": "..."},
             {"role": "user",   "content": "..."},
             {"role": "assistant", "content": "..."}, ...]

        Responses API input format:
            instructions: "..."      (system message)
            input: [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        # Pull system messages out into a single 'instructions' field.
        # If multiple system messages exist, concatenate them.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        instructions = "\n\n".join(system_parts) if system_parts else None

        # Everything else becomes input items.
        input_items = []
        for m in messages:
            if m.get("role") == "system":
                continue
            input_items.append({
                "role": m["role"],
                "content": m["content"],
            })

        payload = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            # Reasoning configuration: request a visible summary.
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": "auto",
            },
        }
        if instructions is not None:
            payload["instructions"] = instructions

        # Note: reasoning models on the Responses API may ignore temperature
        # but we pass it for forward compatibility. Some models (e.g. gpt-5
        # family with reasoning) require temperature=1 or omit the field.
        # We only set it if not a reasoning model context, to avoid 400s.
        # Safest: just don't send temperature on the Responses API.

        data = self._post_with_retry(payload)

        # Responses API returns an "output" array with mixed block types:
        #   {"type": "reasoning", "summary": [...]}  (sometimes "summary_text")
        #   {"type": "message",   "content": [{"type": "output_text", "text": "..."}]}
        # Also may have a top-level "output_text" convenience field.
        text = data.get("output_text", "") or ""
        reasoning = ""

        output = data.get("output", []) or []
        for block in output:
            btype = block.get("type", "")
            if btype == "reasoning":
                # summary is typically a list of {type: "summary_text", text: "..."}
                summary = block.get("summary", []) or []
                if isinstance(summary, list):
                    pieces = []
                    for s in summary:
                        if isinstance(s, dict):
                            pieces.append(s.get("text", "") or s.get("summary_text", ""))
                        elif isinstance(s, str):
                            pieces.append(s)
                    reasoning = "\n\n".join(p for p in pieces if p)
                elif isinstance(summary, str):
                    reasoning = summary
            elif btype == "message" and not text:
                # Fall back to extracting text from message block if
                # the convenience output_text wasn't present.
                content = block.get("content", []) or []
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        text += c.get("text", "")

        # Translate Responses-API status into a finish_reason-like flag.
        status = data.get("status", "")
        if status == "incomplete":
            details = data.get("incomplete_details", {}) or {}
            finish_reason = details.get("reason", "incomplete")
        elif status == "completed":
            finish_reason = "stop"
        else:
            finish_reason = status or ""

        return {
            "text": text.strip(),
            "reasoning": reasoning.strip(),
            "finish_reason": finish_reason,
            "raw": data,
        }

    # ------------------------------------------------------------------
    # Shared retry/backoff helper
    # ------------------------------------------------------------------

    def _post_with_retry(self, payload):
        last_err = None
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    params=self.query_params or None,
                    json=payload,
                    timeout=config.REQUEST_TIMEOUT_SECONDS,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"retryable status {resp.status_code}: {resp.text[:200]}"
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                sleep_for = config.RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(f"  [retry {attempt+1}/{config.MAX_RETRIES}] {e} "
                      f"-> sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)

        raise RuntimeError(f"API call failed after retries: {last_err}")
