#!/usr/bin/env bash
# Logs board power with tegrastats while each TensorRT engine runs for 30 s.
set -euo pipefail

TRTEXEC=/usr/src/tensorrt/bin/trtexec
LOGDIR="$PWD/results/raw"
mkdir -p "$LOGDIR"

stop_tegrastats() { sudo tegrastats --stop >/dev/null 2>&1 || true; }
trap stop_tegrastats EXIT
stop_tegrastats

measure() {  # $1 = label, remaining args = command to run while logging
  local label=$1; shift
  rm -f "$LOGDIR/tegra_$label.log"
  sudo tegrastats --interval 500 --logfile "$LOGDIR/tegra_$label.log" --start
  sleep 2
  "$@"
  sleep 2
  stop_tegrastats
}

echo "Idle baseline (20 s)..."
measure idle sleep 20
sleep 20

for p in fp32 fp16 int8; do
  echo "Power run: $p (30 s)..."
  measure "$p" $TRTEXEC --loadEngine=models/r50_$p.engine \
    --shapes=input:1x3x224x224 --warmUp=500 --duration=30 \
    > "$LOGDIR/power_$p.log" 2>&1
  sleep 20
done
echo "Done."
