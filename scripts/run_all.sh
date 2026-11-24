#!/usr/bin/env bash
# Runs the TensorRT benchmark 3 times with a cooldown between rounds.
set -euo pipefail

for r in 1 2 3; do
  echo "=== Round $r ==="
  export BENCH_RUN=$r
  bash scripts/bench_trt.sh
  python3 scripts/parse_trt.py
  if [ "$r" -lt 3 ]; then sleep 90; fi
done
