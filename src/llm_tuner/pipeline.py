from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from llm_tuner.artifacts import download_adapter_from_checkpoint
from llm_tuner.config import PipelineConfig
from llm_tuner.training import run_rl, run_sft
from llm_tuner.utils import ensure_model_available, init_env, write_json


@dataclass
class PipelineResult:
    run_dir: str
    sft_log_dir: str
    rl_log_dir: str
    sft_state_path: str
    sft_sampler_path: str
    rl_state_path: str
    rl_sampler_path: str
    adapter_archive: str | None = None
    adapter_dir: str | None = None


def build_run_dirs(config: PipelineConfig) -> tuple[Path, Path, Path]:
    run_name = config.run_name or datetime.now().strftime("qwen35_4b_%Y%m%d_%H%M%S")
    run_dir = Path(config.output_root) / run_name
    sft_log_dir = run_dir / "sft"
    rl_log_dir = run_dir / "rl"
    sft_log_dir.mkdir(parents=True, exist_ok=True)
    rl_log_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, sft_log_dir, rl_log_dir


def run_full_pipeline(config: PipelineConfig) -> PipelineResult:
    init_env()

    dataset_path = Path(config.sft.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {dataset_path}")

    ensure_model_available(config.sft.model_name, base_url=config.sft.base_url)
    run_dir, sft_log_dir, rl_log_dir = build_run_dirs(config)

    sft_result = run_sft(config.sft, sft_log_dir)
    rl_result = run_rl(config.rl, rl_log_dir, sft_result.state_path)

    adapter_archive = None
    adapter_dir = None
    if config.download_final_adapter:
        adapter_result = download_adapter_from_checkpoint(
            checkpoint_path=rl_result.sampler_path,
            archive_path=run_dir / "artifacts" / "rl_final_adapter.tar",
            extract_dir=run_dir / "artifacts" / "rl_final_adapter",
            base_url=config.rl.base_url,
        )
        adapter_archive = adapter_result.archive_path
        adapter_dir = adapter_result.extract_dir

    result = PipelineResult(
        run_dir=str(run_dir),
        sft_log_dir=sft_result.log_dir,
        rl_log_dir=rl_result.log_dir,
        sft_state_path=sft_result.state_path,
        sft_sampler_path=sft_result.sampler_path,
        rl_state_path=rl_result.state_path,
        rl_sampler_path=rl_result.sampler_path,
        adapter_archive=adapter_archive,
        adapter_dir=adapter_dir,
    )

    write_json(run_dir / "pipeline_config.json", config)
    write_json(run_dir / "pipeline_result.json", result)
    return result
