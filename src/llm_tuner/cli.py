from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from llm_tuner.artifacts import download_adapter_from_checkpoint
from llm_tuner.config import PipelineConfig, RLConfig, SFTConfig
from llm_tuner.evals import run_mbpp_eval
from llm_tuner.pipeline import run_full_pipeline
from llm_tuner.telemetry import init_wandb_exe_logger, log_metrics_jsonl
from llm_tuner.training import run_rl, run_sft
from llm_tuner.user_config import load_user_config, normalize_sft_dataset
from llm_tuner.utils import ensure_model_available, init_env, write_json


def _default_run_name(prefix: str) -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve(value: Any, cfg: dict[str, Any], keys: Iterable[str], default: Any) -> Any:
    if value is not None:
        return value
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _load_cfg(config_path: str | None) -> dict[str, Any]:
    cfg, resolved = load_user_config(config_path)
    if resolved is not None:
        print(f"[config] loaded {resolved}")
    return cfg


def _maybe_run_mbpp_eval(

    *,
    args: argparse.Namespace,
    run_logger,
    model_path: str,
    run_dir: Path,
    base_url: str | None,
) -> None:
    if int(getattr(args, "eval_mbpp_samples", 0)) <= 0:
        return

    print(
        f"[eval] running MBPP eval: split={args.eval_mbpp_split} "
        f"samples={args.eval_mbpp_samples} model={model_path}"
    )
    eval_result = run_mbpp_eval(
        model_path=model_path,
        split=args.eval_mbpp_split,
        num_samples=args.eval_mbpp_samples,
        seed=args.eval_seed,
        max_tokens=args.eval_max_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        timeout_sec=args.eval_timeout_sec,
        base_url=base_url,
    )

    run_logger.log(
        {
            "eval/mbpp/pass_rate": eval_result.pass_rate,
            "eval/mbpp/passed": eval_result.passed,
            "eval/mbpp/num_samples": eval_result.num_samples,
            "eval/mbpp/avg_completion_chars": eval_result.avg_completion_chars,
        }
    )
    run_logger.log_evals(eval_result.eval_rows)

    write_json(
        run_dir / "eval_mbpp_summary.json",
        {
            "split": eval_result.split,
            "num_samples": eval_result.num_samples,
            "passed": eval_result.passed,
            "pass_rate": eval_result.pass_rate,
            "avg_completion_chars": eval_result.avg_completion_chars,
        },
    )
    _write_jsonl(run_dir / "eval_mbpp_rows.jsonl", eval_result.eval_rows)
    print(
        f"[eval] MBPP done: pass_rate={eval_result.pass_rate:.3f} "
        f"({eval_result.passed}/{eval_result.num_samples})"
    )


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-name", default=None)
    p.add_argument("--renderer-name", default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--wandb-project", default=None)

    # Optional wandb.exe logging (separate from tinker-cookbook's wandb integration).
    p.add_argument("--wandb-exe-project", default=None)
    p.add_argument("--wandb-exe-url", default=None)


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--eval-mbpp-samples",
        type=int,
        default=0,
        help="If > 0, run MBPP eval on the final sampler and log eval rows to wandb.exe.",
    )
    p.add_argument("--eval-mbpp-split", default="test")
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--eval-max-tokens", type=int, default=256)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)
    p.add_argument("--eval-timeout-sec", type=int, default=5)


def cmd_sft(args: argparse.Namespace) -> None:
    init_env()
    defaults = SFTConfig()
    cfg = _load_cfg(args.config)

    model_name = _resolve(args.model_name, cfg, ["model_name", "model"], defaults.model_name)
    renderer_name = _resolve(args.renderer_name, cfg, ["renderer_name"], defaults.renderer_name)
    lora_rank = int(_resolve(args.lora_rank, cfg, ["lora_rank"], defaults.lora_rank))
    base_url = _resolve(args.base_url, cfg, ["base_url"], defaults.base_url)
    wandb_project = _resolve(args.wandb_project, cfg, ["wandb_project"], defaults.wandb_project)
    wandb_name = _resolve(args.wandb_name, cfg, ["sft_wandb_name"], defaults.wandb_name)

    dataset_path = normalize_sft_dataset(
        str(
            _resolve(
                args.dataset_path,
                cfg,
                ["dataset_path", "sft_dataset"],
                defaults.dataset_path,
            )
        )
    )

    max_length = int(_resolve(args.max_length, cfg, ["sft_max_length"], defaults.max_length))
    batch_size = int(_resolve(args.batch_size, cfg, ["sft_batch_size"], defaults.batch_size))
    learning_rate = float(_resolve(args.learning_rate, cfg, ["sft_lr"], defaults.learning_rate))
    num_epochs = int(_resolve(args.num_epochs, cfg, ["sft_epochs"], defaults.num_epochs))
    max_steps = int(_resolve(args.max_steps, cfg, ["sft_max_steps"], defaults.max_steps))
    save_every = int(_resolve(args.save_every, cfg, ["sft_save_every"], defaults.save_every))

    output_root = str(_resolve(args.output_root, cfg, ["output_root", "log_dir"], "runs"))
    run_name = _resolve(args.run_name, cfg, ["run_name"], None)
    run_dir = Path(output_root) / (run_name or _default_run_name("sft"))
    log_dir = run_dir / "sft"
    log_dir.mkdir(parents=True, exist_ok=True)

    ensure_model_available(model_name, base_url=base_url)

    config = SFTConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_path=dataset_path,
        lora_rank=lora_rank,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_every=save_every,
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    wandb_exe_project = _resolve(args.wandb_exe_project, cfg, ["wandb_exe_project"], None)
    wandb_exe_url = _resolve(
        args.wandb_exe_url,
        cfg,
        ["wandb_exe_url"],
        "https://wandb.exe.xyz:8000",
    )

    run_logger = init_wandb_exe_logger(
        project=wandb_exe_project,
        run_name=f"{run_dir.name}-sft",
        config=config,
        url=wandb_exe_url,
    )

    started = time.perf_counter()
    try:
        result = run_sft(config=config, log_dir=log_dir)
        log_metrics_jsonl(run_logger, metrics_path=Path(log_dir) / "metrics.jsonl", stage_prefix="sft")
        _maybe_run_mbpp_eval(

            args=args,
            run_logger=run_logger,
            model_path=result.sampler_path,
            run_dir=run_dir,
            base_url=base_url,
        )
        duration = time.perf_counter() - started
        run_logger.log({"sft/success": 1, "sft/duration_sec": duration})
        run_logger.finish(status="finished")
    except Exception:
        duration = time.perf_counter() - started
        run_logger.log({"sft/success": 0, "sft/duration_sec": duration})
        run_logger.finish(status="failed")
        raise

    write_json(run_dir / "sft_config.json", config)
    write_json(run_dir / "sft_result.json", result)
    print(json.dumps(asdict(result), indent=2))


def cmd_rl(args: argparse.Namespace) -> None:
    init_env()
    defaults = RLConfig()
    cfg = _load_cfg(args.config)

    model_name = _resolve(args.model_name, cfg, ["model_name", "model"], defaults.model_name)
    renderer_name = _resolve(args.renderer_name, cfg, ["renderer_name"], defaults.renderer_name)
    lora_rank = int(_resolve(args.lora_rank, cfg, ["lora_rank"], defaults.lora_rank))
    base_url = _resolve(args.base_url, cfg, ["base_url"], defaults.base_url)
    wandb_project = _resolve(args.wandb_project, cfg, ["wandb_project"], defaults.wandb_project)
    wandb_name = _resolve(args.wandb_name, cfg, ["rl_wandb_name"], defaults.wandb_name)

    from_checkpoint = _resolve(
        args.from_checkpoint,
        cfg,
        ["from_checkpoint", "rl_from_checkpoint"],
        None,
    )
    if not from_checkpoint:
        raise ValueError(
            "Missing checkpoint. Provide --from-checkpoint or set from_checkpoint in config.py"
        )

    env = _resolve(args.env, cfg, ["rl_env"], defaults.env)
    group_size = int(_resolve(args.group_size, cfg, ["rl_group_size"], defaults.group_size))
    groups_per_batch = int(
        _resolve(args.groups_per_batch, cfg, ["rl_groups_per_batch"], defaults.groups_per_batch)
    )
    learning_rate = float(_resolve(args.learning_rate, cfg, ["rl_lr"], defaults.learning_rate))
    max_tokens = int(_resolve(args.max_tokens, cfg, ["rl_max_tokens"], defaults.max_tokens))
    max_steps = int(_resolve(args.max_steps, cfg, ["rl_max_steps"], defaults.max_steps))
    save_every = int(_resolve(args.save_every, cfg, ["rl_save_every"], defaults.save_every))

    output_root = str(_resolve(args.output_root, cfg, ["output_root", "log_dir"], "runs"))
    run_name = _resolve(args.run_name, cfg, ["run_name"], None)
    run_dir = Path(output_root) / (run_name or _default_run_name("rl"))
    log_dir = run_dir / "rl"
    log_dir.mkdir(parents=True, exist_ok=True)

    ensure_model_available(model_name, base_url=base_url)

    config = RLConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        lora_rank=lora_rank,
        env=env,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        max_steps=max_steps,
        save_every=save_every,
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    wandb_exe_project = _resolve(args.wandb_exe_project, cfg, ["wandb_exe_project"], None)
    wandb_exe_url = _resolve(
        args.wandb_exe_url,
        cfg,
        ["wandb_exe_url"],
        "https://wandb.exe.xyz:8000",
    )

    run_logger = init_wandb_exe_logger(
        project=wandb_exe_project,
        run_name=f"{run_dir.name}-rl",
        config={**asdict(config), "from_checkpoint": from_checkpoint},
        url=wandb_exe_url,
    )

    started = time.perf_counter()
    try:
        result = run_rl(config, log_dir, load_checkpoint_path=from_checkpoint)
        log_metrics_jsonl(run_logger, metrics_path=Path(log_dir) / "metrics.jsonl", stage_prefix="rl")
        duration = time.perf_counter() - started
        run_logger.log({"rl/success": 1, "rl/duration_sec": duration})
        run_logger.finish(status="finished")
    except Exception:
        duration = time.perf_counter() - started
        run_logger.log({"rl/success": 0, "rl/duration_sec": duration})
        run_logger.finish(status="failed")
        raise

    write_json(run_dir / "rl_config.json", config)
    write_json(run_dir / "rl_result.json", result)
    print(json.dumps(asdict(result), indent=2))


def cmd_adapter(args: argparse.Namespace) -> None:
    init_env()
    cfg = _load_cfg(args.config)
    out_dir = Path(args.output_dir)

    base_url = _resolve(args.base_url, cfg, ["base_url"], None)
    wandb_exe_project = _resolve(args.wandb_exe_project, cfg, ["wandb_exe_project"], None)
    wandb_exe_url = _resolve(
        args.wandb_exe_url,
        cfg,
        ["wandb_exe_url"],
        "https://wandb.exe.xyz:8000",
    )

    run_logger = init_wandb_exe_logger(
        project=wandb_exe_project,
        run_name=f"{out_dir.name or 'adapter'}-adapter-download",
        config={
            "checkpoint_path": args.checkpoint_path,
            "output_dir": str(out_dir),
            "base_url": base_url,
        },
        url=wandb_exe_url,
    )

    started = time.perf_counter()
    try:
        result = download_adapter_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            archive_path=out_dir / "adapter.tar",
            extract_dir=out_dir / "adapter_files",
            base_url=base_url,
        )
        duration = time.perf_counter() - started
        run_logger.log({"adapter_download/success": 1, "adapter_download/duration_sec": duration})
        run_logger.finish(status="finished")
    except Exception:
        duration = time.perf_counter() - started
        run_logger.log({"adapter_download/success": 0, "adapter_download/duration_sec": duration})
        run_logger.finish(status="failed")
        raise

    write_json(out_dir / "adapter_download_result.json", result)
    print(json.dumps(asdict(result), indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)

    sft_defaults = SFTConfig()
    rl_defaults = RLConfig()

    model_name = _resolve(args.model_name, cfg, ["model_name", "model"], sft_defaults.model_name)
    renderer_name = _resolve(args.renderer_name, cfg, ["renderer_name"], sft_defaults.renderer_name)
    lora_rank = int(_resolve(args.lora_rank, cfg, ["lora_rank"], sft_defaults.lora_rank))
    base_url = _resolve(args.base_url, cfg, ["base_url"], sft_defaults.base_url)

    output_root = str(_resolve(args.output_root, cfg, ["output_root", "log_dir"], "runs"))
    run_name = _resolve(args.run_name, cfg, ["run_name"], None)
    wandb_project = _resolve(args.wandb_project, cfg, ["wandb_project"], sft_defaults.wandb_project)

    dataset_path = normalize_sft_dataset(
        str(
            _resolve(
                args.dataset_path,
                cfg,
                ["dataset_path", "sft_dataset"],
                sft_defaults.dataset_path,
            )
        )
    )

    sft_cfg = SFTConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_path=dataset_path,
        lora_rank=lora_rank,
        max_length=int(_resolve(args.max_length, cfg, ["sft_max_length"], sft_defaults.max_length)),
        batch_size=int(_resolve(args.sft_batch_size, cfg, ["sft_batch_size"], sft_defaults.batch_size)),
        learning_rate=float(_resolve(args.sft_learning_rate, cfg, ["sft_lr"], sft_defaults.learning_rate)),
        num_epochs=int(_resolve(args.sft_num_epochs, cfg, ["sft_epochs"], sft_defaults.num_epochs)),
        max_steps=int(_resolve(args.sft_max_steps, cfg, ["sft_max_steps"], sft_defaults.max_steps)),
        save_every=int(_resolve(args.sft_save_every, cfg, ["sft_save_every"], sft_defaults.save_every)),
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=_resolve(args.sft_wandb_name, cfg, ["sft_wandb_name"], sft_defaults.wandb_name),
    )

    rl_cfg = RLConfig(
        model_name=model_name,
        renderer_name=renderer_name,
        lora_rank=lora_rank,
        env=_resolve(args.rl_env, cfg, ["rl_env"], rl_defaults.env),
        group_size=int(_resolve(args.rl_group_size, cfg, ["rl_group_size"], rl_defaults.group_size)),
        groups_per_batch=int(
            _resolve(args.rl_groups_per_batch, cfg, ["rl_groups_per_batch"], rl_defaults.groups_per_batch)
        ),
        learning_rate=float(_resolve(args.rl_learning_rate, cfg, ["rl_lr"], rl_defaults.learning_rate)),
        max_tokens=int(_resolve(args.rl_max_tokens, cfg, ["rl_max_tokens"], rl_defaults.max_tokens)),
        max_steps=int(_resolve(args.rl_max_steps, cfg, ["rl_max_steps"], rl_defaults.max_steps)),
        save_every=int(_resolve(args.rl_save_every, cfg, ["rl_save_every"], rl_defaults.save_every)),
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=_resolve(args.rl_wandb_name, cfg, ["rl_wandb_name"], rl_defaults.wandb_name),
    )

    config = PipelineConfig(
        output_root=output_root,
        run_name=run_name,
        download_final_adapter=not args.no_download_final_adapter,
        sft=sft_cfg,
        rl=rl_cfg,
    )

    pipeline_name = run_name or _default_run_name("pipeline")

    wandb_exe_project = _resolve(args.wandb_exe_project, cfg, ["wandb_exe_project"], None)
    wandb_exe_url = _resolve(
        args.wandb_exe_url,
        cfg,
        ["wandb_exe_url"],
        "https://wandb.exe.xyz:8000",
    )

    run_logger = init_wandb_exe_logger(
        project=wandb_exe_project,
        run_name=pipeline_name,
        config=config,
        url=wandb_exe_url,
    )

    started = time.perf_counter()
    try:
        result = run_full_pipeline(config)
        run_dir = Path(result.run_dir)
        log_metrics_jsonl(run_logger, metrics_path=run_dir / "sft" / "metrics.jsonl", stage_prefix="sft")
        log_metrics_jsonl(run_logger, metrics_path=run_dir / "rl" / "metrics.jsonl", stage_prefix="rl")
        _maybe_run_mbpp_eval(

            args=args,
            run_logger=run_logger,
            model_path=result.rl_sampler_path,
            run_dir=run_dir,
            base_url=base_url,
        )
        duration = time.perf_counter() - started
        run_logger.log({"pipeline/success": 1, "pipeline/duration_sec": duration})
        run_logger.finish(status="finished")
    except Exception:
        duration = time.perf_counter() - started
        run_logger.log({"pipeline/success": 0, "pipeline/duration_sec": duration})
        run_logger.finish(status="failed")
        raise

    print(json.dumps(asdict(result), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5-4B Tinker toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    # sft
    sft = sub.add_parser("sft", help="Run SFT stage")
    sft.add_argument("--config", default="config.py", help="Optional config.py path")
    _add_common_model_args(sft)
    sft.add_argument("--dataset-path", default=None)
    sft.add_argument("--max-length", type=int, default=None)
    sft.add_argument("--batch-size", type=int, default=None)
    sft.add_argument("--learning-rate", type=float, default=None)
    sft.add_argument("--num-epochs", type=int, default=None)
    sft.add_argument("--max-steps", type=int, default=None)
    sft.add_argument("--save-every", type=int, default=None)
    sft.add_argument("--wandb-name", default=None)
    sft.add_argument("--output-root", default=None)
    sft.add_argument("--run-name", default=None)
    _add_eval_args(sft)
    sft.set_defaults(func=cmd_sft)

    # rl
    rl = sub.add_parser("rl", help="Run RL stage from an existing checkpoint")
    rl.add_argument("--config", default="config.py", help="Optional config.py path")
    _add_common_model_args(rl)
    rl.add_argument("--from-checkpoint", default=None, help="SFT state checkpoint path")
    rl.add_argument("--env", default=None)
    rl.add_argument("--group-size", type=int, default=None)
    rl.add_argument("--groups-per-batch", type=int, default=None)
    rl.add_argument("--learning-rate", type=float, default=None)
    rl.add_argument("--max-tokens", type=int, default=None)
    rl.add_argument("--max-steps", type=int, default=None)
    rl.add_argument("--save-every", type=int, default=None)
    rl.add_argument("--wandb-name", default=None)
    rl.add_argument("--output-root", default=None)
    rl.add_argument("--run-name", default=None)
    rl.set_defaults(func=cmd_rl)

    # adapter
    adapter = sub.add_parser("adapter", help="Download adapter from a sampler checkpoint")
    adapter.add_argument("--config", default="config.py", help="Optional config.py path")
    adapter.add_argument("--checkpoint-path", required=True)
    adapter.add_argument("--output-dir", required=True)
    adapter.add_argument("--base-url", default=None)
    adapter.add_argument("--wandb-exe-project", default=None)
    adapter.add_argument("--wandb-exe-url", default=None)
    adapter.set_defaults(func=cmd_adapter)

    # full pipeline
    pipe = sub.add_parser("pipeline", help="Run SFT -> RL -> optional adapter download")
    pipe.add_argument("--config", default="config.py", help="Optional config.py path")
    _add_common_model_args(pipe)
    pipe.add_argument("--dataset-path", default=None)
    pipe.add_argument("--max-length", type=int, default=None)
    pipe.add_argument("--output-root", default=None)
    pipe.add_argument("--run-name", default=None)

    pipe.add_argument("--sft-batch-size", type=int, default=None)
    pipe.add_argument("--sft-learning-rate", type=float, default=None)
    pipe.add_argument("--sft-num-epochs", type=int, default=None)
    pipe.add_argument("--sft-max-steps", type=int, default=None)
    pipe.add_argument("--sft-save-every", type=int, default=None)
    pipe.add_argument("--sft-wandb-name", default=None)

    pipe.add_argument("--rl-env", default=None)
    pipe.add_argument("--rl-group-size", type=int, default=None)
    pipe.add_argument("--rl-groups-per-batch", type=int, default=None)
    pipe.add_argument("--rl-learning-rate", type=float, default=None)
    pipe.add_argument("--rl-max-tokens", type=int, default=None)
    pipe.add_argument("--rl-max-steps", type=int, default=None)
    pipe.add_argument("--rl-save-every", type=int, default=None)
    pipe.add_argument("--rl-wandb-name", default=None)

    pipe.add_argument("--no-download-final-adapter", action="store_true")
    _add_eval_args(pipe)
    pipe.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
