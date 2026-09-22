#!/usr/bin/env bash
# Wait for the two baseline checkpoints and run the locked LIBERO matrices.
# This is intended for a long-lived 4090 host session.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:?set LIBERO_ASSETS_DIR}"
ACT_CHECKPOINT="${ACT_CHECKPOINT:?set ACT_CHECKPOINT}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:?set DIFFUSION_CHECKPOINT}"
ADAPTER_PATH="${ADAPTER_PATH:?set ADAPTER_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_STEPS="${MAX_STEPS:-280}"
EPISODES="${EPISODES:-30}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-$PROJECT_ROOT/artifacts/rollout/baselines_task34_real_v2}"
SMOLVLA_OUTPUT_DIR="${SMOLVLA_OUTPUT_DIR:-$PROJECT_ROOT/artifacts/rollout/current_seed123_real_v2}"
DISTILLATION_METRICS_JSON="${DISTILLATION_METRICS_JSON:-}"

wait_for_checkpoint() {
  local label="$1"
  local path="$2"
  while [[ ! -f "$path/model.safetensors" ]]; do
    printf '[%s] waiting for %s checkpoint: %s\n' "$(date -Is)" "$label" "$path"
    sleep 60
  done
  printf '[%s] found %s checkpoint: %s\n' "$(date -Is)" "$label" "$path"
}

wait_for_checkpoint ACT "$ACT_CHECKPOINT"
wait_for_checkpoint Diffusion "$DIFFUSION_CHECKPOINT"

PROJECT_ROOT="$PROJECT_ROOT" \
LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
ACT_CHECKPOINT="$ACT_CHECKPOINT" \
DIFFUSION_CHECKPOINT="$DIFFUSION_CHECKPOINT" \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_DIR="$BASELINE_OUTPUT_DIR" \
EPISODES="$EPISODES" \
MAX_STEPS="$MAX_STEPS" \
bash "$PROJECT_ROOT/scripts/run_baseline_rollout_matrix.sh"

PROJECT_ROOT="$PROJECT_ROOT" \
ADAPTER_PATH="$ADAPTER_PATH" \
LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
CHECKPOINT="${SMOLVLA_CHECKPOINT:-HuggingFaceVLA/smolvla_libero}" \
OUTPUT_DIR="$SMOLVLA_OUTPUT_DIR" \
EPISODES="$EPISODES" \
MAX_STEPS="$MAX_STEPS" \
DISTILLATION_METRICS_JSON="$DISTILLATION_METRICS_JSON" \
bash "$PROJECT_ROOT/scripts/run_acceptance_matrix.sh"

printf '[%s] all matrices completed\n' "$(date -Is)"
