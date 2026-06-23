"""LLM abstraction. Backed by Google Gemini (free tier, no billing required).

The rest of the codebase calls `generate` / `generate_structured` and never
touches the provider SDK directly — so swapping providers later is a one-file
change. Includes simple retry/backoff because Gemini's free tier has
per-minute rate limits that a full eval run will brush against.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

from pydantic import BaseModel

from . import config


@lru_cache(maxsize=1)
def _client():
    from google import genai

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to .env. "
            "Retrieval works without it — see `python -m secrag.index query`."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _gen_config(system: str, max_tokens: int, model: str, temperature: float,
                schema: type[BaseModel] | None):
    from google.genai import types

    kwargs: dict = {
        "system_instruction": system,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    # 2.5 models "think" by default, which can eat the whole token budget and
    # return empty text. We don't need it for grounded extraction/judging.
    if model.startswith("gemini-2.5"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    if schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = schema
    return types.GenerateContentConfig(**kwargs)


_RETRYABLE = (
    "429", "resource_exhausted", "rate", "503", "unavailable", "500", "internal",
    # transient network/DNS hiccups
    "connect", "nodename", "getaddrinfo", "timeout", "temporarily", "connection reset",
)


def _call(model: str, contents: str, cfg, retries: int = 4):
    last: Exception | None = None
    for i in range(retries):
        try:
            return _client().models.generate_content(
                model=model, contents=contents, config=cfg
            )
        except Exception as e:  # noqa: BLE001 — classify by message, then retry/raise
            last = e
            if any(tok in str(e).lower() for tok in _RETRYABLE):
                time.sleep(2 * (2 ** i))
                continue
            raise
    raise last  # type: ignore[misc]


def generate(system: str, user: str, max_tokens: int = 1024,
             model: str | None = None, temperature: float = 0.2) -> str:
    """Plain text generation."""
    model = model or config.ANSWER_MODEL
    resp = _call(model, user, _gen_config(system, max_tokens, model, temperature, None))
    return (resp.text or "").strip()


def generate_structured(system: str, user: str, schema: type[BaseModel],
                        max_tokens: int = 512, model: str | None = None,
                        temperature: float = 0.0) -> BaseModel:
    """Schema-constrained generation; returns a validated pydantic instance."""
    model = model or config.JUDGE_MODEL
    resp = _call(model, user, _gen_config(system, max_tokens, model, temperature, schema))
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed
    # Fallback: parse the raw JSON text against the schema.
    return schema.model_validate_json(resp.text or "{}")


def has_key() -> bool:
    return bool(config.GEMINI_API_KEY)


if __name__ == "__main__":
    print("Model:", config.ANSWER_MODEL)
    print(generate("You are terse.", "Say 'hello from gemini' and nothing else.", max_tokens=50))
