# llm-tuner

This repo is now organized as a standard Python package under `src/`.

## Package layout

```text
src/llm_tuner/
  config.py
  cli.py
  pipeline.py

  training/
    sft.py            # SFT stage
    rl.py             # RL stage

  artifacts/
    adapter.py        # checkpoint -> adapter tar/files

  serve/
    modal_tinker_proxy.py
    modal_vllm_lora.py
    modal_playground.py

apps/modal/
  proxy.py            # deploy wrapper
  vllm_lora.py        # deploy wrapper
  run_pipeline.py     # optional remote orchestration
```

## Install (editable)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Single CLI entrypoint

Everything is triggered from one CLI:

```bash
python -m llm_tuner.cli --help
# or after install:
llm-tuner --help
```

## `config.py` support (sft-rl-pipeline compatible)

`llm-tuner` now supports reading an optional `config.py` (like `sft-rl-pipeline`) via `--config`.
If present, those values become defaults; explicit CLI flags still win.

Example:

```bash
cp config.py.example config.py
python -m llm_tuner.cli pipeline --config config.py
python -m llm_tuner.cli sft --config config.py
python -m llm_tuner.cli rl --config config.py --from-checkpoint tinker://...
```

Supported keys include (non-exhaustive):
- `model`, `renderer_name`, `lora_rank`, `base_url`
- `sft_dataset` / `dataset_path`, `sft_lr`, `sft_batch_size`, `sft_max_steps`, ...
- `rl_env`, `rl_lr`, `rl_group_size`, `rl_groups_per_batch`, `rl_max_steps`, ...
- `wandb_project`, `output_root` (or `log_dir`), `run_name`

`no_robots` is also accepted for `sft_dataset` and maps to `data/no_robots_sft_3000.jsonl` when available.

## Manual stage-by-stage

Load env first:

```bash
source .venv/bin/activate
set -a && source .env && set +a
```

### 1) SFT

```bash
python -m llm_tuner.cli sft \
  --dataset-path data/sft_conversations.jsonl \
  --run-name my_sft_run \
  --output-root runs
```

### 2) RL (from SFT checkpoint)

```bash
python -m llm_tuner.cli rl \
  --from-checkpoint "tinker://.../weights/final" \
  --env arithmetic \
  --run-name my_rl_run \
  --output-root runs
```

### 3) Download adapter

```bash
python -m llm_tuner.cli adapter \
  --checkpoint-path "tinker://.../sampler_weights/final" \
  --output-dir runs/my_rl_run/artifacts
```

### 4) One-shot pipeline

```bash
python -m llm_tuner.cli pipeline \
  --dataset-path data/sft_conversations.jsonl \
  --run-name qwen35_full \
  --sft-max-steps 20 \
  --rl-max-steps 30
```

## Optional wandb.exe logging (if/else)

The CLI now has conditional logging to `wandb.exe`:

- if `--wandb-exe-project` is provided: logging is enabled
- else: logging is skipped automatically

When enabled, it uploads full stage metrics from:
- `runs/<run>/sft/metrics.jsonl` as `sft/*`
- `runs/<run>/rl/metrics.jsonl` as `rl/*`
then marks the run `finished`.

Install client (from your provided docs):

```bash
pip install "wandb-exe @ git+https://gitpad.exe.xyz/wandb-exe.git"
```

Example with pipeline logging:

```bash
python -m llm_tuner.cli pipeline \
  --run-name qwen35_full \
  --wandb-exe-project my-experiment \
  --wandb-exe-url https://wandb.exe.xyz:8000
```

### Evals to wandb.exe (MBPP)

`llm.txt` now supports eval rows, and the CLI can push them to the same run.

Enable MBPP eval upload with:

```bash
python -m llm_tuner.cli sft \
  --dataset-path data/codealpaca_sft_12000.jsonl \
  --run-name codealpaca_sft_v1 \
  --wandb-exe-project llm-tuner-experiments \
  --eval-mbpp-samples 100
```

This logs:
- summary metrics: `eval/mbpp/*`
- per-example eval rows in the run’s **evals** tab

## CodeAlpaca SFT dataset prep

```bash
python scripts/prepare_sft_codealpaca.py \
  --output data/codealpaca_sft_12000.jsonl \
  --num-samples 12000
```

## Serving on Modal

### Fast proxy (Modal -> Tinker inference)

```bash
modal deploy apps/modal/proxy.py
```

### Web playground (FastAPI chat UI, improved conversation UX)

```bash
modal deploy apps/modal/playground.py
```

Then open the deployed URL and chat directly in the built-in interface.
API endpoints:
- `POST /api/chat` (multi-turn chat)
- `POST /api/generate` (backward-compatible single-prompt)

### Full GPU serving (vLLM + LoRA)

```bash
modal run apps/modal/vllm_lora.py::download_adapter
modal deploy apps/modal/vllm_lora.py
```

## Current deployed endpoint

```text
https://swairshah--qwen35-4b-tinker-proxy-generate.modal.run
```
