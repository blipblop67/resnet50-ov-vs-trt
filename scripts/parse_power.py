import csv
import re

PWR = re.compile(r"VDD_IN (\d+)mW/")
GPU = re.compile(r"GR3D_FREQ (\d+)%")
TJ = re.compile(r"tj@([\d.]+)C")

rows = []
for cfg in ["idle", "fp32", "fp16", "int8"]:
    samples = []
    with open(f"results/raw/tegra_{cfg}.log") as f:
        for line in f:
            p, g, t = PWR.search(line), GPU.search(line), TJ.search(line)
            if p and g and t:
                samples.append((int(p.group(1)), int(g.group(1)), float(t.group(1))))
    if cfg != "idle":
        samples = [s for s in samples if s[1] >= 50]  # GPU busy only
    if not samples:
        raise SystemExit(f"no usable samples for {cfg}")
    n = len(samples)
    rows.append(dict(
        config=cfg,
        samples=n,
        mean_power_w=round(sum(s[0] for s in samples) / n / 1000, 2),
        mean_gpu_pct=round(sum(s[1] for s in samples) / n, 1),
        peak_tj_c=max(s[2] for s in samples),
    ))

idle_w = rows[0]["mean_power_w"]
for r in rows:
    r["above_idle_w"] = round(r["mean_power_w"] - idle_w, 2)

with open("results/jetson_power.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(r)
