#!/usr/bin/env bash
# Run official LeRobot ACT and Diffusion Policy checkpoints with the locked
# LIBERO task34 rollout protocol.  Training is intentionally separate because
# it can take hours and must be resumed from the user's selected checkpoint.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:?set LIBERO_ASSETS_DIR to the official asset directory}"
ACT_CHECKPOINT="${ACT_CHECKPOINT:?set ACT_CHECKPOINT to a trained LeRobot ACT checkpoint}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:?set DIFFUSION_CHECKPOINT to a trained LeRobot Diffusion checkpoint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/rollout/baselines_task34}"
EPISODES="${EPISODES:-30}"
START_SEED="${START_SEED:-0}"
TORCH_SEED="${TORCH_SEED:-123}"
MAX_STEPS="${MAX_STEPS:-280}"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"

common=(
  "$PROJECT_ROOT/scripts/run_libero_rollout.py"
  --suite libero_spatial
  --task-id 0
  --protocol "$PROJECT_ROOT/configs/evaluation_protocol.toml"
  --dataset-task-index 34
  --episodes "$EPISODES"
  --start-seed "$START_SEED"
  --torch-seed "$TORCH_SEED"
  --max-steps "$MAX_STEPS"
  --assets-dir "$LIBERO_ASSETS_DIR"
  --gripper-polarity positive_open
)

"$PYTHON_BIN" "${common[@]}" \
  --policy-type act \
  --checkpoint "$ACT_CHECKPOINT" \
  --mode sync \
  --output "$OUTPUT_DIR/act_sync.json"

"$PYTHON_BIN" "${common[@]}" \
  --policy-type act \
  --checkpoint "$ACT_CHECKPOINT" \
  --mode async \
  --disable-rtc \
  --overlap-steps 10 \
  --output "$OUTPUT_DIR/act_async_overlap.json"

"$PYTHON_BIN" "${common[@]}" \
  --policy-type diffusion \
  --checkpoint "$DIFFUSION_CHECKPOINT" \
  --mode sync \
  --output "$OUTPUT_DIR/diffusion_sync.json"

"$PYTHON_BIN" "${common[@]}" \
  --policy-type diffusion \
  --checkpoint "$DIFFUSION_CHECKPOINT" \
  --mode async \
  --disable-rtc \
  --overlap-steps 10 \
  --output "$OUTPUT_DIR/diffusion_async_overlap.json"

echo "Baseline rollout matrix written to $OUTPUT_DIR"
