from __future__ import annotations

from dataclasses import asdict
from typing import Any

import modal

from llm_tuner.config import PipelineConfig, RLConfig, SFTConfig
from llm_tuner.pipeline import run_full_pipeline

app = modal.App("qwen35-4b-tinker-pipeline")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "tinker==0.15.0",
    "tinker-cookbook==0.1.0",
    "python-dotenv==1.2.2",
    "requests",
    "certifi",
)

_tinker_secret = modal.Secret.from_name("tinker-api-key", required_keys=["TINKER_API_KEY"])


@app.function(image=image, cpu=4, timeout=60 * 60, secrets=[_tinker_secret])
def run_remote(config_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = PipelineConfig(
        output_root=config_dict.get("output_root", "runs"),
        run_name=config_dict.get("run_name"),
        download_final_adapter=config_dict.get("download_final_adapter", True),
        sft=SFTConfig(**config_dict.get("sft", {})),
        rl=RLConfig(**config_dict.get("rl", {})),
    )
    result = run_full_pipeline(cfg)
    return asdict(result)


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3.5-4B",
    renderer_name: str = "qwen3_5_disable_thinking",
    dataset_path: str = "data/sft_conversations.jsonl",
    output_root: str = "runs",
    run_name: str | None = None,
    sft_max_steps: int = 20,
    rl_env: str = "arithmetic",
    rl_max_steps: int = 30,
):
    cfg = PipelineConfig(
        output_root=output_root,
        run_name=run_name,
        sft=SFTConfig(
            model_name=model_name,
            renderer_name=renderer_name,
            dataset_path=dataset_path,
            max_steps=sft_max_steps,
        ),
        rl=RLConfig(
            model_name=model_name,
            renderer_name=renderer_name,
            env=rl_env,
            max_steps=rl_max_steps,
        ),
    )

    print(run_remote.remote(asdict(cfg)))
