# Project Structure (canonical)

## Canonical code

- `src/llm_tuner/training/sft.py`
  - SFT stage only
- `src/llm_tuner/training/rl.py`
  - RL stage only
- `src/llm_tuner/artifacts/adapter.py`
  - checkpoint adapter download/extract
- `src/llm_tuner/pipeline.py`
  - full SFT -> RL -> adapter orchestration
- `src/llm_tuner/serve/modal_tinker_proxy.py`
  - Modal endpoint proxying Tinker OAI API
- `src/llm_tuner/serve/modal_vllm_lora.py`
  - Modal GPU serving with vLLM + LoRA
- `src/llm_tuner/serve/modal_playground.py`
  - Modal FastAPI web playground (weblog-style UI) for hosted model testing
- `src/llm_tuner/cli.py`
  - single CLI with subcommands:
    - `sft`
    - `rl`
    - `adapter`
    - `pipeline`
- `src/llm_tuner/telemetry.py`
  - if/else-based optional logging to wandb.exe (`wandb_client`)
- `src/llm_tuner/user_config.py`
  - optional `config.py` loader (compatible with sft-rl-pipeline-style keys)

## Deployment wrappers

- `apps/modal/proxy.py`
- `apps/modal/vllm_lora.py`
- `apps/modal/playground.py`
- `apps/modal/run_pipeline.py`

These wrappers exist only so `modal deploy ...` can point to a tiny file path.
