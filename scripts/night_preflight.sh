#!/usr/bin/env bash
set -u

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
PREFLIGHT_DIR=${PREFLIGHT_DIR:-${PROJECT_ROOT}/artifacts/preflight}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data}
FALLBACK_DATA_FILE=${DATA_DIR}/file-055.parquet
TASK0_DATA_FILE=${DATA_DIR}/file-309.parquet
WINDOWS_DOWNLOAD_TARGET=${WINDOWS_DOWNLOAD_TARGET:-\\\\wsl.localhost\\Ubuntu${TASK0_DATA_FILE}}
LOG_FILE=${PREFLIGHT_DIR}/night_preflight.log
STATUS_FILE=${PREFLIGHT_DIR}/night_preflight.status
PID_FILE=${PREFLIGHT_DIR}/night_preflight.pid
BASELINE_RESULT_FILE=${PREFLIGHT_DIR}/teacher_forward_episode137.json
RESULT_FILE=${PREFLIGHT_DIR}/teacher_forward_task0.json
RTC_RESULT_FILE=${PREFLIGHT_DIR}/rtc_smoke.json

mkdir -p "${PREFLIGHT_DIR}"
echo "$$" > "${PID_FILE}"
echo running > "${STATUS_FILE}"
exec >> "${LOG_FILE}" 2>&1

finish() {
    exit_code=$?
    if [ "${exit_code}" -eq 0 ]; then
        echo complete > "${STATUS_FILE}"
    else
        echo "failed:${exit_code}" > "${STATUS_FILE}"
    fi
    echo "night_preflight exit_code=${exit_code} finished_at=$(date --iso-8601=seconds)"
}
trap finish EXIT

echo "night_preflight pid=$$ started_at=$(date --iso-8601=seconds)"
cd "${PROJECT_ROOT}"
export PYTHONPATH=${PROJECT_ROOT}/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH=${PREFLIGHT_DIR}/libero_config

if [ ! -f "${FALLBACK_DATA_FILE}" ]; then
    echo "missing fallback parquet: ${FALLBACK_DATA_FILE}"
    exit 20
fi

echo "phase=episode137_latency_baseline"
"${PYTHON_BIN}" -u scripts/preflight_teacher_forward.py \
    --parquet "${FALLBACK_DATA_FILE}" \
    --episode-index 137 \
    --task "put both the alphabet soup and the tomato sauce in the basket" \
    --benchmark-flow-steps 10 5 2 \
    --benchmark-warmup 5 \
    --benchmark-repeats 10 \
    --output "${BASELINE_RESULT_FILE}"

expected_sha=ac7a824d26e07681a521d868e539561b1081222e9be777034f653a0bcdf18f22
current_sha=""
if [ -f "${TASK0_DATA_FILE}" ]; then
    current_sha=$(sha256sum "${TASK0_DATA_FILE}" | cut -d ' ' -f 1)
fi
if [ "${current_sha}" != "${expected_sha}" ]; then
    echo "phase=task0_download current_sha=${current_sha:-missing}"
    mkdir -p "${DATA_DIR}"
    powershell.exe -NoProfile -NonInteractive -Command \
        "Invoke-WebRequest -Uri 'https://huggingface.co/datasets/HuggingFaceVLA/libero/resolve/main/data/chunk-000/file-309.parquet?download=true' -OutFile '${WINDOWS_DOWNLOAD_TARGET}'" || true
fi

if [ -f "${TASK0_DATA_FILE}" ]; then
    current_sha=$(sha256sum "${TASK0_DATA_FILE}" | cut -d ' ' -f 1)
fi
if [ "${current_sha}" = "${expected_sha}" ]; then
    echo "phase=task0_forward bytes=$(stat -c %s "${TASK0_DATA_FILE}") sha256=${current_sha}"
    "${PYTHON_BIN}" -u scripts/preflight_teacher_forward.py \
        --parquet "${TASK0_DATA_FILE}" \
        --episode-index 1272 \
        --task "pick up the black bowl between the plate and the ramekin and place it on the plate" \
        --benchmark-flow-steps 10 5 2 \
        --benchmark-warmup 5 \
        --benchmark-repeats 10 \
        --output "${RESULT_FILE}" || true
else
    echo "task0_download_failed expected_sha=${expected_sha} actual_sha=${current_sha:-missing}"
fi

"${PYTHON_BIN}" -u scripts/preflight_rtc_smoke.py --output "${RTC_RESULT_FILE}" || true
echo "night_preflight completed"
