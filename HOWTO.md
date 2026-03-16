# HOWTO: from a new dataset to a live Modal endpoint

This is a practical, copy-paste guide for running **SFT + eval** and deploying an endpoint.

You can run commands either as:
- `tinker-tailor ...` (preferred)
- `python -m llm_tuner.cli ...` (equivalent fallback)

---

## 0) Prerequisites

- Python 3.12
- Tinker API key in `.env` as `TINKER_API_KEY`
- Optional: Modal CLI (`pip install modal` + `modal setup`)
- Optional: wandb.exe client for metrics/evals upload

Install wandb.exe client:

```bash
pip install "wandb-exe @ git+https://gitpad.exe.xyz/wandb-exe.git"
```

---

## 1) Install the package

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Check CLI is available:

```bash
tinker-tailor --help
```

---

## 2) Load environment variables

Create `.env` if needed:

```bash
cat > .env <<'EOF'
TINKER_API_KEY=your_key_here
EOF
```

Load it for this shell:

```bash
set -a && source .env && set +a
```

---

## 3) Prepare your SFT dataset

### Required JSONL format
Each line must be:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### Option A: use provided prep scripts

CodeAlpaca example:

```bash
python scripts/prepare_sft_codealpaca.py \
  --output data/codealpaca_sft_12000.jsonl \
  --num-samples 12000
```

No Robots example:

```bash
python scripts/prepare_sft_no_robots.py \
  --output data/no_robots_sft_3000.jsonl \
  --num-samples 3000
```

### Option B: bring your own dataset file

Put it at e.g. `data/my_dataset.jsonl` in the required format.

Quick sanity check:

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('data/my_dataset.jsonl')
n=0
for line in p.open():
    row=json.loads(line)
    assert 'messages' in row and isinstance(row['messages'], list)
    n+=1
print('ok rows =', n)
PY
```

---

## 4) (Optional) configure defaults in `config.py`

You can copy defaults and edit once:

```bash
cp config.py.example config.py
```

Then run with config defaults:

```bash
tinker-tailor sft --config config.py --run-name my_run
```

CLI flags always override config values.

---

## 5) Run SFT + MBPP eval + wandb.exe logging

This runs training, uploads step metrics, then runs MBPP eval and uploads eval rows.

```bash
tinker-tailor sft \
  --config "" \
  --run-name exp_my_dataset_v1 \
  --dataset-path data/my_dataset.jsonl \
  --model-name Qwen/Qwen3.5-4B \
  --renderer-name qwen3_5_disable_thinking \
  --batch-size 64 \
  --max-steps 40 \
  --learning-rate 2e-4 \
  --output-root runs \
  --wandb-exe-project llm-tuner-experiments \
  --wandb-exe-url https://wandb.exe.xyz:8000 \
  --eval-mbpp-samples 100 \
  --eval-mbpp-split test \
  --eval-max-tokens 256 \
  --eval-temperature 0.0 \
  --eval-timeout-sec 5
```

Notes:
- `--config ""` disables config.py loading (fully explicit run).
- If your dataset is not code-focused, MBPP numbers may be low (that’s expected).

---

## 6) Inspect outputs

Run folder:

```bash
runs/exp_my_dataset_v1/
```

Key files:

- `runs/exp_my_dataset_v1/sft/metrics.jsonl`
- `runs/exp_my_dataset_v1/sft_result.json`
- `runs/exp_my_dataset_v1/eval_mbpp_summary.json`
- `runs/exp_my_dataset_v1/eval_mbpp_rows.jsonl`

Print checkpoint paths quickly:

```bash
python - <<'PY'
import json
r=json.load(open('runs/exp_my_dataset_v1/sft_result.json'))
print('state_path  =', r['state_path'])
print('sampler_path=', r['sampler_path'])
PY
```

---

## 7) Deploy to Modal (Tinker proxy endpoint)

### 7.1 Set Modal secrets from your final sampler checkpoint

```bash
SAMPLER_PATH=$(python - <<'PY'
import json
print(json.load(open('runs/exp_my_dataset_v1/sft_result.json'))['sampler_path'])
PY
)

bash scripts/setup_modal_secrets.sh "$SAMPLER_PATH" "Qwen/Qwen3.5-4B"
```

### 7.2 Deploy endpoint

```bash
modal deploy apps/modal/proxy.py
```

This prints a URL like:

```text
https://<workspace>--qwen35-4b-tinker-proxy-generate.modal.run
```

### 7.3 Test endpoint

```bash
curl -sS -X POST "https://<your-url>.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a Python function to reverse a string."}],"max_tokens":200,"temperature":0.2}'
```

---

## 8) Optional: deploy the web playground

```bash
modal deploy apps/modal/playground.py
```

Open the returned URL and chat in-browser.

---

## 9) Common troubleshooting

- **Model unavailable**: verify your Tinker account supports the model.
- **No wandb logs**: ensure `wandb_client` is installed and `--wandb-exe-project` is set.
- **Run stuck in running**: check that process reached `run_logger.finish()` (latest code handles finish explicitly).
- **MBPP eval errors/timeouts**: reduce `--eval-mbpp-samples`, increase `--eval-timeout-sec`, or lower `--eval-max-tokens`.
- **Modal secret errors**: re-run `scripts/setup_modal_secrets.sh` with the correct sampler checkpoint path.
