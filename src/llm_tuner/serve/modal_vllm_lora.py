"""Modal GPU endpoint that serves Qwen3.5-4B + Tinker LoRA adapter via vLLM.

Required Modal secrets:
  1) tinker-api-key      -> TINKER_API_KEY=...
  2) tinker-model-config -> TINKER_MODEL_PATH=tinker://.../sampler_weights/final
                            BASE_MODEL=Qwen/Qwen3.5-4B

Setup once:
  modal run llm_tuner/serve/modal_vllm_lora.py::download_adapter

Deploy endpoint:
  modal deploy llm_tuner/serve/modal_vllm_lora.py
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import modal

app = modal.App("qwen35-4b-vllm-lora")

MODEL_DIR = "/models"
ADAPTER_ROOT = f"{MODEL_DIR}/adapter"
volume = modal.Volume.from_name("qwen35-4b-rl-adapter", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tinker==0.15.0",
        "transformers==4.57.6",
        "vllm==0.8.5.post1",
        "fastapi[standard]==0.116.1",
        "requests",
        "certifi",
    )
)

_tinker_secret = modal.Secret.from_name("tinker-api-key", required_keys=["TINKER_API_KEY"])
_model_secret = modal.Secret.from_name(
    "tinker-model-config",
    required_keys=["TINKER_MODEL_PATH", "BASE_MODEL"],
)

_llm = None
_adapter_path = None


def _find_adapter_dir(root: Path) -> Path:
    for candidate in root.rglob("adapter_config.json"):
        return candidate.parent
    raise FileNotFoundError(f"Could not find adapter_config.json under {root}")


@app.function(
    image=image,
    cpu=2,
    timeout=900,
    secrets=[_tinker_secret, _model_secret],
    volumes={MODEL_DIR: volume},
)
def download_adapter() -> dict[str, str]:
    import certifi
    import requests
    import tinker

    import os

    checkpoint_path = os.environ["TINKER_MODEL_PATH"]
    root = Path(ADAPTER_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    archive_path = root / "adapter.tar"
    extract_path = root / "files"
    if extract_path.exists():
        for p in sorted(extract_path.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
        for p in sorted(extract_path.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    extract_path.mkdir(parents=True, exist_ok=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    signed = rc.get_checkpoint_archive_url_from_tinker_path(checkpoint_path).result()

    with requests.get(signed.url, stream=True, timeout=300, verify=certifi.where()) as response:
        response.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_path, filter="data")

    adapter_dir = _find_adapter_dir(extract_path)
    volume.commit()

    return {"checkpoint_path": checkpoint_path, "adapter_dir": str(adapter_dir)}


def _lazy_init_vllm() -> None:
    global _llm, _adapter_path
    if _llm is not None and _adapter_path is not None:
        return

    import os
    from vllm import LLM

    base_model = os.environ["BASE_MODEL"]
    adapter_dir = _find_adapter_dir(Path(ADAPTER_ROOT) / "files")

    _llm = LLM(
        model=base_model,
        enable_lora=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    _adapter_path = str(adapter_dir)


@app.function(
    image=image,
    gpu="L40S",
    timeout=300,
    min_containers=1,
    scaledown_window=600,
    secrets=[_model_secret],
    volumes={MODEL_DIR: volume},
)
@modal.fastapi_endpoint(method="POST")
def generate(payload: dict[str, Any]) -> dict[str, Any]:
    _lazy_init_vllm()

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    assert _llm is not None
    assert _adapter_path is not None

    prompt = payload.get("prompt")
    if not prompt:
        messages = payload.get("messages") or [{"role": "user", "content": "Hello!"}]
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

    sampling_params = SamplingParams(
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.95)),
        max_tokens=int(payload.get("max_tokens", 256)),
    )

    outputs = _llm.generate(
        [prompt],
        sampling_params,
        lora_request=LoRARequest("qwen35_rl", 1, _adapter_path),
    )

    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    import os

    return {
        "completion": text,
        "base_model": os.environ["BASE_MODEL"],
        "adapter_path": _adapter_path,
    }
