#!/usr/bin/env bash
set -euo pipefail

TRTEXEC=/usr/src/tensorrt/bin/trtexec
ONNX=models/resnet50.onnx
SHAPES=input:1x3x224x224

echo "Building FP32 engine..."
$TRTEXEC --onnx=$ONNX --shapes=$SHAPES \
  --saveEngine=models/r50_fp32.engine > models/build_fp32.log 2>&1

echo "Building FP16 engine..."
$TRTEXEC --onnx=$ONNX --shapes=$SHAPES --fp16 \
  --saveEngine=models/r50_fp16.engine > models/build_fp16.log 2>&1

echo "Building INT8 engine (no calibration data: speed test only)..."
$TRTEXEC --onnx=$ONNX --shapes=$SHAPES --int8 --fp16 \
  --saveEngine=models/r50_int8.engine > models/build_int8.log 2>&1

echo "Done."
ls -lh models/*.engine
