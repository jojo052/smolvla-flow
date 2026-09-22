#!/usr/bin/env bash
set -euo pipefail

ROOT="${TASK34_DATA_ROOT:-$HOME/datasets/libero_task34_source}"
OUT="${TASK34_SPLIT_ROOT:-$ROOT/split}"

# Locked task34 split from artifacts/preflight/libero_spatial_task0_split.json.
SPLITS='{"train":[1272,1273,1282,1300,1327,1330,1344,1352,1368,1379,1382,1397,1401,1425,1491,1493,1504,1532,1560,1563,1579,1583,1589,1597,1602,1615,1617,1660,1669,1672,1679],"validation":[1347,1395,1449,1450,1500,1567,1642],"test":[1275,1410,1506,1517,1527,1568,1574]}'

python "$(dirname "$0")/materialize_task34_split.py" \
  --source-root "$ROOT" \
  --manifest "${TASK34_SPLIT_MANIFEST:-$HOME/projects/smolvla-flow/project-current/artifacts/preflight/libero_spatial_task0_split.json}" \
  --output-root "$OUT"

echo "TASK34_SPLIT_ROOT=$OUT"
