#!/usr/bin/env bash
# Run the current SmolVLA LIBERO acceptance matrix on the GPU host.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
ADAPTER_PATH="${ADAPTER_PATH:?set ADAPTER_PATH to the distilled 2-step action expert}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:?set LIBERO_ASSETS_DIR to the official asset directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/rollout/current_seed123}"
EPISODES="${EPISODES:-5}"
START_SEED="${START_SEED:-0}"
TORCH_SEED="${TORCH_SEED:-123}"
MAX_STEPS="${MAX_STEPS:-}"
BENCHMARK_JSON="${BENCHMARK_JSON:-}"
DISTILLATION_METRICS_JSON="${DISTILLATION_METRICS_JSON:-}"

if [[ ! -f "$ADAPTER_PATH" ]]; then
  echo "missing distilled adapter: $ADAPTER_PATH" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"

common=(
  "$PROJECT_ROOT/scripts/run_libero_rollout.py"
  --checkpoint "$CHECKPOINT"
  --suite libero_spatial
  --task-id 0
  --dataset-task-index 34
  --episodes "$EPISODES"
  --start-seed "$START_SEED"
  --torch-seed "$TORCH_SEED"
  --assets-dir "$LIBERO_ASSETS_DIR"
)
if [[ -n "$MAX_STEPS" ]]; then
  common+=(--max-steps "$MAX_STEPS")
fi

"$PYTHON_BIN" "${common[@]}" \
  --mode sync --flow-steps 10 \
  --output "$OUTPUT_DIR/native_teacher_sync.json"

"$PYTHON_BIN" "${common[@]}" \
  --mode sync --flow-steps 2 \
  --output "$OUTPUT_DIR/undistilled_two_step_sync.json"

"$PYTHON_BIN" "${common[@]}" \
  --mode sync --flow-steps 2 --adapter "$ADAPTER_PATH" \
  --output "$OUTPUT_DIR/distilled_two_step_sync.json"

"$PYTHON_BIN" "${common[@]}" \
  --mode async --flow-steps 2 --adapter "$ADAPTER_PATH" \
  --overlap-steps 10 \
  --output "$OUTPUT_DIR/distilled_two_step_async_rtc.json"

"$PYTHON_BIN" "${common[@]}" \
  --mode async --flow-steps 2 --adapter "$ADAPTER_PATH" \
  --disable-rtc --overlap-steps 0 \
  --output "$OUTPUT_DIR/distilled_two_step_async_unfused.json"

benchmark_args=()
if [[ -n "$BENCHMARK_JSON" ]]; then
  benchmark_args=(--benchmark "$BENCHMARK_JSON")
fi

distillation_metrics_args=()
if [[ -n "$DISTILLATION_METRICS_JSON" ]]; then
  distillation_metrics_args=(--distillation-metrics "$DISTILLATION_METRICS_JSON")
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_acceptance_metrics.py" \
  --rollout "$OUTPUT_DIR/distilled_two_step_sync.json" \
  --rollout "$OUTPUT_DIR/distilled_two_step_async_rtc.json" \
  "${benchmark_args[@]}" \
  "${distillation_metrics_args[@]}" \
  --fused-rollout "$OUTPUT_DIR/distilled_two_step_async_rtc.json" \
  --unfused-rollout "$OUTPUT_DIR/distilled_two_step_async_unfused.json" \
  --output "$OUTPUT_DIR/acceptance_metrics.json"

echo "Acceptance matrix written to $OUTPUT_DIR"
