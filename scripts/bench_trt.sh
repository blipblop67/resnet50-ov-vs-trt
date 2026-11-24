#!/usr/bin/env bash
set -euo pipefail

TRTEXEC=/usr/src/tensorrt/bin/trtexec
RUN=${BENCH_RUN:-1}
mkdir -p results/raw

for p in fp32 fp16 int8; do
  echo "Benchmarking $p (run $RUN)..."
  $TRTEXEC --loadEngine=models/r50_$p.engine --shapes=input:1x3x224x224 \
    --warmUp=500 --duration=0 --iterations=500 --avgRuns=100 \
    --exportTimes=results/raw/trt_${p}_run${RUN}.json \
    > results/raw/bench_${p}_run${RUN}.log 2>&1
done
