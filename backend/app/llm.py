"""NVIDIA NIM LLM client behind `stream(system, messages) -> Iterator[str]`."""
from __future__ import annotations

from typing import Iterator


class LLM:
    name = "none"
    generative = True

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, system: str, messages: list[dict]) -> str:
        return "".join(self.stream(system, messages))


class NvidiaLLM(LLM):
    def __init__(self, api_key: str, model: str, base_url: str, temperature: float, max_tokens: int):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=120)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"nvidia:{model}"

    def stream(self, system, messages):
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, stream=True,
            messages=[{"role": "system", "content": system}, *messages])
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_llm(settings) -> LLM:
    """Build the NVIDIA client; fail immediately when its API key is missing."""
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    return NvidiaLLM(settings.nvidia_api_key, settings.nvidia_model, settings.nvidia_base_url,
                     settings.temperature, settings.max_tokens)
