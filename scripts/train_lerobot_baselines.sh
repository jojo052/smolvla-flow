#!/usr/bin/env bash
# Train official LeRobot visual baselines on the same task34 dataset.
# The resulting checkpoint directories are consumed by
# scripts/run_baseline_rollout_matrix.sh.

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:?set DATASET_REPO_ID to the task34 LeRobot dataset repo or local dataset id}"
DATASET_ARGS=("--dataset.repo_id=$DATASET_REPO_ID")
if [[ -n "${DATASET_ROOT:-}" ]]; then
  DATASET_ARGS+=("--dataset.root=$DATASET_ROOT")
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/task34_baselines}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_AMP="${USE_AMP:-true}"
ACT_STEPS="${ACT_STEPS:-50000}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100000}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-lerobot-train}"

"$LEROBOT_TRAIN" \
  "${DATASET_ARGS[@]}" \
  --policy.type=act \
  --policy.device="$POLICY_DEVICE" \
  --policy.push_to_hub=false \
  --policy.use_amp="$USE_AMP" \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.n_obs_steps=1 \
  --steps="$ACT_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --output_dir="$OUTPUT_ROOT/act" \
  --job_name=libero_task34_act

"$LEROBOT_TRAIN" \
  "${DATASET_ARGS[@]}" \
  --policy.type=diffusion \
  --policy.device="$POLICY_DEVICE" \
  --policy.push_to_hub=false \
  --policy.use_amp="$USE_AMP" \
  --policy.horizon=64 \
  --policy.n_action_steps=10 \
  --policy.n_obs_steps=2 \
  --policy.num_train_timesteps=100 \
  --steps="$DIFFUSION_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --output_dir="$OUTPUT_ROOT/diffusion" \
  --job_name=libero_task34_diffusion
