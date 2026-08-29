#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment-name> <config-name> [Hydra overrides ...]" >&2
  exit 2
fi

EXPERIMENT_NAME="$1"
CONFIG_NAME="$2"
shift 2

PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT="${EXPERIMENT_NAME}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
export WANDB_MODE=disabled

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  --config-name "${CONFIG_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "$@"
