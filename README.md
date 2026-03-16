# llm-tuner (proper package layout)

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

## Serving on Modal

### Fast proxy (Modal -> Tinker inference)

```bash
modal deploy apps/modal/proxy.py
```

### Full GPU serving (vLLM + LoRA)

```bash
modal run apps/modal/vllm_lora.py::download_adapter
modal deploy apps/modal/vllm_lora.py
```

## Current deployed endpoint

```text
https://swairshah--qwen35-4b-tinker-proxy-generate.modal.run
```
