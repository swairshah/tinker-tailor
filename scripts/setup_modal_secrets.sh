#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  echo ".env not found"
  exit 1
fi

# shellcheck disable=SC1091
source .env

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY missing in .env"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TINKER_MODEL_PATH> [BASE_MODEL]"
  exit 1
fi

MODEL_PATH="$1"
BASE_MODEL="${2:-Qwen/Qwen3.5-4B}"

modal secret create tinker-api-key TINKER_API_KEY="$TINKER_API_KEY" --force
modal secret create tinker-model-config TINKER_MODEL_PATH="$MODEL_PATH" BASE_MODEL="$BASE_MODEL" --force

echo "Created/updated Modal secrets:"
echo "  - tinker-api-key"
echo "  - tinker-model-config"
