#!/usr/bin/env bash
set -euo pipefail

# Download only the parquet shards that contain the locked LIBERO task34
# episodes. The source dataset metadata is staged separately by the caller.
ROOT="${TASK34_DATA_ROOT:-$HOME/datasets/libero_task34_modelscope}"
mkdir -p "$ROOT/data/chunk-000"

# ModelScope's current parquet layout contains task34 in these files. The
# older preflight episode metadata refers to a stale file numbering scheme.
for idx in 309 310 311 314 318 319 321 322 325 327 329 330 331 334 338 344 345 346 347 348 350 351 355 356 357 358 359 360 361 362 364 368 371 372 373 374; do
  file="$ROOT/data/chunk-000/file-${idx}.parquet"
  url="https://www.modelscope.cn/datasets/HuggingFaceVLA/libero/resolve/master/data/chunk-000/file-${idx}.parquet"
  echo "[$(date -Is)] downloading $idx"
  curl --fail --location --retry 8 --retry-delay 3 --retry-all-errors \
    --continue-at - --output "$file" "$url"
  test -s "$file"
done

echo "[$(date -Is)] DOWNLOAD_DONE"
