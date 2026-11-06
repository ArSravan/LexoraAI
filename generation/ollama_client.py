"""Client for a local Ollama server (https://ollama.com). No external API is ever called.

Ollama serves open models over HTTP on your own machine. We use its /api/chat endpoint with
streaming, so tokens can be shown while the answer is still being written.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from config import get_settings


class OllamaError(RuntimeError):
    """Connection, model, timeout or protocol problem. The message tells the user what to do."""


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_seconds: float = 0.0
    thinking: str = ""
    raw: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3.5:9b",
                 timeout: float = 600.0, temperature: float = 0.1, num_ctx: int = 8192,
                 num_predict: int = 1024, think: str = "off", seed: int = 42):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think = think                  # off | on | auto
        self.seed = seed
        self._no_think: set[str] = set()    # models that rejected the `think` option

    @classmethod
    def from_settings(cls) -> "OllamaClient":
        s = get_settings()
        return cls(s.ollama_base_url, s.ollama_model, s.ollama_timeout, s.ollama_temperature,
                   s.ollama_num_ctx, s.ollama_num_predict, s.ollama_think, s.seed)

    # ------------------------------------------------------------------ status
    def is_up(self, timeout: float = 3.0) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/version", timeout=timeout).ok
        except requests.RequestException:
            return False

    def version(self) -> str | None:
        try:
            return requests.get(f"{self.base_url}/api/version", timeout=3).json().get("version")
        except (requests.RequestException, ValueError):
            return None

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError) as e:
            raise OllamaError(f"Cannot list Ollama models at {self.base_url}: {e}") from e

    def has_model(self, model: str | None = None) -> bool:
        model = model or self.model
        try:
            names = self.list_models()
        except OllamaError:
            return False
        return model in names or (":" not in model and f"{model}:latest" in names)

    def status(self) -> dict:
        up = self.is_up()
        models: list[str] = []
        if up:
            try:
                models = self.list_models()
            except OllamaError:
                pass
        return {"connected": up, "base_url": self.base_url, "model": self.model,
                "model_available": self.has_model() if up else False, "models": models,
                "version": self.version() if up else None}

    # ------------------------------------------------------------------ chat
    def _payload(self, messages, fmt, options, think_override) -> dict:
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx,
                "num_predict": self.num_predict, "seed": self.seed}
        if options:
            opts.update(options)
        payload = {"model": self.model, "messages": messages, "stream": True, "options": opts}
        if fmt is not None:
            payload["format"] = fmt         # "json" or a JSON schema: structured output
        think = self.think if think_override is None else think_override
        if self.model not in self._no_think and think in ("off", "on", True, False):
            payload["think"] = think in ("on", True)
        return payload

    def _post(self, payload: dict):
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True,
                              timeout=(5, self.timeout))
        except requests.ConnectionError as e:
            raise OllamaError(f"Ollama is not reachable at {self.base_url}. Start it with `ollama serve`.") from e
        except requests.Timeout as e:
            raise OllamaError(f"Ollama timed out after {self.timeout}s.") from e
        if r.status_code == 404:
            raise OllamaError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")
        if r.status_code == 400 and "think" in payload and "think" in r.text.lower():
            self._no_think.add(self.model)  # this model has no thinking option: remember, retry without it
            return self._post({k: v for k, v in payload.items() if k != "think"})
        if not r.ok:
            raise OllamaError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r

    def chat(self, messages: list[dict], fmt=None, options: dict | None = None, think: str | None = None,
             on_token: Callable[[str], None] | None = None) -> ChatResult:
        """Run one chat completion. The reply is streamed internally; ``on_token`` sees each piece."""
        started = time.perf_counter()
        r = self._post(self._payload(messages, fmt, options, think))
        content: list[str] = []
        thinking: list[str] = []
        final: dict = {}
        try:
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise OllamaError(chunk["error"])
                msg = chunk.get("message", {})
                if msg.get("thinking"):
                    thinking.append(msg["thinking"])
                if msg.get("content"):
                    content.append(msg["content"])
                    if on_token:
                        on_token(msg["content"])
                if chunk.get("done"):
                    final = chunk
        except requests.RequestException as e:
            raise OllamaError(f"Ollama stream interrupted: {e}") from e
        return ChatResult(content="".join(content), model=self.model,
                          prompt_tokens=final.get("prompt_eval_count", 0),
                          completion_tokens=final.get("eval_count", 0),
                          total_seconds=time.perf_counter() - started,
                          thinking="".join(thinking), raw=final)
