# Runs the whole laptop benchmark suite 3 times with a cooldown between rounds.
foreach ($r in 1..3) {
    $env:BENCH_RUN = $r
    Write-Host "=== Round $r ==="
    python scripts\bench_ort.py
    python scripts\bench_ov.py fp32 fp16 int8
    if ($r -lt 3) { Start-Sleep -Seconds 90 }
}