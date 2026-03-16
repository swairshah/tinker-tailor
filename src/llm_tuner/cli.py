from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from llm_tuner.artifacts import download_adapter_from_checkpoint
from llm_tuner.config import PipelineConfig, RLConfig, SFTConfig
from llm_tuner.pipeline import run_full_pipeline
from llm_tuner.training import run_rl, run_sft
from llm_tuner.utils import ensure_model_available, init_env, write_json


def _default_run_name(prefix: str) -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--renderer-name", default="qwen3_5_disable_thinking")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--base-url", default=None)
    p.add_argument("--wandb-project", default=None)


def cmd_sft(args: argparse.Namespace) -> None:
    init_env()
    ensure_model_available(args.model_name, base_url=args.base_url)

    run_dir = Path(args.output_root) / (args.run_name or _default_run_name("sft"))
    log_dir = run_dir / "sft"
    log_dir.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        model_name=args.model_name,
        renderer_name=args.renderer_name,
        dataset_path=args.dataset_path,
        lora_rank=args.lora_rank,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        save_every=args.save_every,
        base_url=args.base_url,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    result = run_sft(config, log_dir)

    write_json(run_dir / "sft_config.json", config)
    write_json(run_dir / "sft_result.json", result)
    print(json.dumps(asdict(result), indent=2))


def cmd_rl(args: argparse.Namespace) -> None:
    init_env()
    ensure_model_available(args.model_name, base_url=args.base_url)

    run_dir = Path(args.output_root) / (args.run_name or _default_run_name("rl"))
    log_dir = run_dir / "rl"
    log_dir.mkdir(parents=True, exist_ok=True)

    config = RLConfig(
        model_name=args.model_name,
        renderer_name=args.renderer_name,
        lora_rank=args.lora_rank,
        env=args.env,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        save_every=args.save_every,
        base_url=args.base_url,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    result = run_rl(config, log_dir, load_checkpoint_path=args.from_checkpoint)

    write_json(run_dir / "rl_config.json", config)
    write_json(run_dir / "rl_result.json", result)
    print(json.dumps(asdict(result), indent=2))


def cmd_adapter(args: argparse.Namespace) -> None:
    init_env()
    out_dir = Path(args.output_dir)
    result = download_adapter_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        archive_path=out_dir / "adapter.tar",
        extract_dir=out_dir / "adapter_files",
        base_url=args.base_url,
    )
    write_json(out_dir / "adapter_download_result.json", result)
    print(json.dumps(asdict(result), indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        output_root=args.output_root,
        run_name=args.run_name,
        download_final_adapter=not args.no_download_final_adapter,
        sft=SFTConfig(
            model_name=args.model_name,
            renderer_name=args.renderer_name,
            dataset_path=args.dataset_path,
            lora_rank=args.lora_rank,
            max_length=args.max_length,
            batch_size=args.sft_batch_size,
            learning_rate=args.sft_learning_rate,
            num_epochs=args.sft_num_epochs,
            max_steps=args.sft_max_steps,
            save_every=args.sft_save_every,
            base_url=args.base_url,
            wandb_project=args.wandb_project,
            wandb_name=args.sft_wandb_name,
        ),
        rl=RLConfig(
            model_name=args.model_name,
            renderer_name=args.renderer_name,
            lora_rank=args.lora_rank,
            env=args.rl_env,
            group_size=args.rl_group_size,
            groups_per_batch=args.rl_groups_per_batch,
            learning_rate=args.rl_learning_rate,
            max_tokens=args.rl_max_tokens,
            max_steps=args.rl_max_steps,
            save_every=args.rl_save_every,
            base_url=args.base_url,
            wandb_project=args.wandb_project,
            wandb_name=args.rl_wandb_name,
        ),
    )

    result = run_full_pipeline(config)
    print(json.dumps(asdict(result), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B Tinker toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    # sft
    sft = sub.add_parser("sft", help="Run SFT stage")
    _add_common_model_args(sft)
    sft.add_argument("--dataset-path", default="data/sft_conversations.jsonl")
    sft.add_argument("--max-length", type=int, default=4096)
    sft.add_argument("--batch-size", type=int, default=64)
    sft.add_argument("--learning-rate", type=float, default=2e-4)
    sft.add_argument("--num-epochs", type=int, default=1)
    sft.add_argument("--max-steps", type=int, default=20)
    sft.add_argument("--save-every", type=int, default=5)
    sft.add_argument("--wandb-name", default="qwen35-4b-sft")
    sft.add_argument("--output-root", default="runs")
    sft.add_argument("--run-name", default=None)
    sft.set_defaults(func=cmd_sft)

    # rl
    rl = sub.add_parser("rl", help="Run RL stage from an existing checkpoint")
    _add_common_model_args(rl)
    rl.add_argument("--from-checkpoint", required=True, help="SFT state checkpoint path")
    rl.add_argument("--env", default="arithmetic")
    rl.add_argument("--group-size", type=int, default=4)
    rl.add_argument("--groups-per-batch", type=int, default=32)
    rl.add_argument("--learning-rate", type=float, default=1e-5)
    rl.add_argument("--max-tokens", type=int, default=64)
    rl.add_argument("--max-steps", type=int, default=30)
    rl.add_argument("--save-every", type=int, default=5)
    rl.add_argument("--wandb-name", default="qwen35-4b-rl")
    rl.add_argument("--output-root", default="runs")
    rl.add_argument("--run-name", default=None)
    rl.set_defaults(func=cmd_rl)

    # adapter
    adapter = sub.add_parser("adapter", help="Download adapter from a sampler checkpoint")
    adapter.add_argument("--checkpoint-path", required=True)
    adapter.add_argument("--output-dir", required=True)
    adapter.add_argument("--base-url", default=None)
    adapter.set_defaults(func=cmd_adapter)

    # full pipeline
    pipe = sub.add_parser("pipeline", help="Run SFT -> RL -> optional adapter download")
    _add_common_model_args(pipe)
    pipe.add_argument("--dataset-path", default="data/sft_conversations.jsonl")
    pipe.add_argument("--max-length", type=int, default=4096)
    pipe.add_argument("--output-root", default="runs")
    pipe.add_argument("--run-name", default=None)

    pipe.add_argument("--sft-batch-size", type=int, default=64)
    pipe.add_argument("--sft-learning-rate", type=float, default=2e-4)
    pipe.add_argument("--sft-num-epochs", type=int, default=1)
    pipe.add_argument("--sft-max-steps", type=int, default=20)
    pipe.add_argument("--sft-save-every", type=int, default=5)
    pipe.add_argument("--sft-wandb-name", default="qwen35-4b-sft")

    pipe.add_argument("--rl-env", default="arithmetic")
    pipe.add_argument("--rl-group-size", type=int, default=4)
    pipe.add_argument("--rl-groups-per-batch", type=int, default=32)
    pipe.add_argument("--rl-learning-rate", type=float, default=1e-5)
    pipe.add_argument("--rl-max-tokens", type=int, default=64)
    pipe.add_argument("--rl-max-steps", type=int, default=30)
    pipe.add_argument("--rl-save-every", type=int, default=5)
    pipe.add_argument("--rl-wandb-name", default="qwen35-4b-rl")

    pipe.add_argument("--no-download-final-adapter", action="store_true")
    pipe.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
