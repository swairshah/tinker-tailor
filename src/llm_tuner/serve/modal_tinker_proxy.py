"""Modal endpoint that serves a Tinker checkpoint through OpenAI-compatible API.

Required Modal secrets:
  1) tinker-api-key      -> TINKER_API_KEY=...
  2) tinker-model-config -> TINKER_MODEL_PATH=tinker://.../sampler_weights/final

Deploy:
  modal deploy llm_tuner/serve/modal_tinker_proxy.py
"""

from __future__ import annotations

import os
from typing import Any

import modal

app = modal.App("qwen35-4b-tinker-proxy")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "openai>=2.28.0,<3",
    "fastapi>=0.116,<1",
)

_tinker_secret = modal.Secret.from_name("tinker-api-key", required_keys=["TINKER_API_KEY"])
_model_secret = modal.Secret.from_name(
    "tinker-model-config",
    required_keys=["TINKER_MODEL_PATH"],
)

TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

_client = None


def _lazy_init() -> None:
    global _client
    if _client is not None:
        return

    from openai import OpenAI

    _client = OpenAI(base_url=TINKER_OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"])


@app.function(
    image=image,
    cpu=1,
    timeout=120,
    secrets=[_tinker_secret, _model_secret],
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST")
def generate(payload: dict[str, Any]) -> dict[str, Any]:
    _lazy_init()

    assert _client is not None
    model_path = os.environ["TINKER_MODEL_PATH"]

    max_tokens = int(payload.get("max_tokens", 256))
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.95))

    if "messages" in payload and payload["messages"]:
        resp = _client.chat.completions.create(
            model=model_path,
            messages=payload["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        completions = [choice.message.content or "" for choice in resp.choices]
    else:
        prompt = payload.get("prompt", "Hello!")
        resp = _client.completions.create(
            model=model_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        completions = [choice.text for choice in resp.choices]

    return {
        "completions": completions,
        "model_path": model_path,
        "provider": "tinker-openai-compatible",
    }
