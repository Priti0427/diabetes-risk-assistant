"""Thin LLM client over any OpenAI-compatible chat API (Nebius AI Studio by default).

Configuration is read from environment variables only (see .env.example):

    LLM_API_KEY    API key. `NEBIUS_API_KEY` / `OPENAI_API_KEY` are accepted as fallbacks.
    LLM_BASE_URL   Default: https://api.studio.nebius.com/v1/
    LLM_MODEL      Default: meta-llama/Meta-Llama-3.1-70B-Instruct

If no key is configured, :func:`get_llm_client` returns ``None`` and the
application falls back to its rule-based parser and templated responses, so the
pipeline stays usable (and testable) offline.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.studio.nebius.com/v1/"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


class ChatClient(Protocol):
    """Anything with ``chat(messages, json_mode) -> str``; lets tests inject fakes."""

    def chat(self, messages: list[dict], json_mode: bool = False) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        from openai import OpenAI  # imported lazily so tests don't need network setup

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, **kwargs
            )
        except Exception:
            if not json_mode:
                raise
            # Some providers/models reject response_format; retry relying on the prompt alone.
            resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature)
        return resp.choices[0].message.content or ""


def get_api_key() -> str | None:
    for var in ("LLM_API_KEY", "NEBIUS_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(var)
        if value:
            return value
    return None


def get_llm_client() -> OpenAICompatibleClient | None:
    """Build a client from the environment, or return None when no key is set."""
    key = get_api_key()
    if not key:
        return None
    return OpenAICompatibleClient(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
    )


def extract_json(text: str) -> dict:
    """Parse the first JSON object in an LLM reply, tolerating code fences and chatter."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in LLM reply: {text[:200]!r}")
