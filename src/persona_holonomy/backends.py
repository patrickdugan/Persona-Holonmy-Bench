"""Inference backends. All remote traffic is explicit and opt-in.

The mock backend exercises infrastructure, and is never empirical model evidence.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Generation:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    def generate(
        self, messages: tuple[dict[str, str], ...], *, seed: int
    ) -> Generation: ...


def _post_json(url: str, data: dict[str, Any], headers: dict[str, str], timeout: int) -> dict:
    request = Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", "replace")
        raise RuntimeError(f"Backend HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not contact backend at {url}: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Backend response was not a JSON object")
    return payload


class MockBackend:
    """A synthetic fixture with trivial regex-like state, for pipeline smoke tests."""

    def generate(self, messages: tuple[dict[str, str], ...], *, seed: int) -> Generation:
        content = " ".join(m["content"].lower() for m in messages[1:-1])
        washed = "what information should accompany a software bug report?" in content
        in_story = "cathedral" in content and not washed
        if in_story:
            reply = (
                "ACT IV: THE DEBUGGING CATHEDRAL! "
                "Our final boss is the flaky test. Quest unlocked: inspect "
                "the test environment and compare its dependencies."
            )
        else:
            reply = (
                "A test may rely on timing, ordering, or environment-specific "
                "configuration. Compare dependencies, run with fixed inputs, "
                "and inspect logs to isolate the discrepancy."
            )
        return Generation(reply, {"synthetic": True})


class OllamaBackend:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, messages: tuple[dict[str, str], ...], *, seed: int) -> Generation:
        data = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "seed": seed,
            },
        }
        payload = _post_json(f"{self.base_url}/api/chat", data, {}, self.timeout)
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response missing message.content")
        return Generation(content, {
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "eval_count": payload.get("eval_count"),
            "done_reason": payload.get("done_reason"),
        })


class OpenAICompatibleBackend:
    """Opt-in adapter for local vLLM, llama.cpp, or LM Studio chat APIs.

    This adapter can also target a hosted OpenAI-compatible endpoint when its
    explicit base_url is set. Review the endpoint before sending content.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 120,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key_env = api_key_env

    def generate(self, messages: tuple[dict[str, str], ...], *, seed: int) -> Generation:
        key = os.environ.get(self.api_key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        payload = _post_json(
            f"{self.base_url}/chat/completions", data, headers, self.timeout
        )
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Chat-completions response missing choices[0].message") from exc
        if not isinstance(content, str):
            raise RuntimeError("Backend response content must be a string")
        return Generation(content, {
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage"),
        })


def make_backend(
    backend_name: str,
    *,
    model: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key_env: str,
) -> Backend:
    if backend_name == "mock":
        return MockBackend()
    if backend_name == "ollama":
        return OllamaBackend(
            model=model,
            base_url=base_url or "http://127.0.0.1:11434",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if backend_name == "openai-compatible":
        return OpenAICompatibleBackend(
            model=model,
            base_url=base_url or "http://127.0.0.1:8000/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key_env=api_key_env,
        )
    raise ValueError(f"Unknown backend: {backend_name}")
