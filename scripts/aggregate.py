import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

LAPTOP = "Laptop (i7-8665U)"
JETSON = "Jetson Orin Nano"
NAMES = {"DESKTOP-7RHCGIG": LAPTOP, "ubuntu": JETSON}
COLORS = {LAPTOP: "#4C78A8", JETSON: "#F58518"}

df = pd.concat([pd.read_csv("results/laptop.csv"), pd.read_csv("results/jetson.csv")])
df["system"] = df["machine"].map(NAMES).fillna(df["machine"])
df["config"] = df["runtime"] + "-" + df["device"] + " " + df["precision"]

summary = df.groupby(["system", "config"], sort=False).agg(
    precision=("precision", "first"),
    p50_ms=("p50_ms", "median"),
    p50_min=("p50_ms", "min"),
    p50_max=("p50_ms", "max"),
    p95_ms=("p95_ms", "median"),
    fps=("fps", "median"),
    rounds=("run", "nunique"),
).reset_index()

# Jetson power (module input power while the GPU is busy)
power = pd.read_csv("results/jetson_power.csv").set_index("config")["mean_power_w"]
summary["power_w"] = summary.apply(
    lambda r: power.get(r["precision"]) if r["system"] == JETSON else float("nan"), axis=1)
summary["fps_per_w"] = summary["fps"] / summary["power_w"]

# TensorRT INT8 was built without calibration data: speed only
speed_only = (summary["system"] == JETSON) & (summary["precision"] == "int8")
summary["label"] = summary["config"] + speed_only.map({True: " *", False: ""})

summary = summary.sort_values("p50_ms").reset_index(drop=True)
summary.to_csv("results/summary.csv", index=False)

# Markdown table
lines = ["| System | Config | p50 (ms) | p50 range | p95 (ms) | FPS | Power (W) | FPS/W |",
         "|---|---|---|---|---|---|---|---|"]
for r in summary.itertuples():
    pw = "" if pd.isna(r.power_w) else f"{r.power_w:.1f}"
    fw = "" if pd.isna(r.fps_per_w) else f"{r.fps_per_w:.1f}"
    lines.append(f"| {r.system} | {r.label} | {r.p50_ms:.2f} | {r.p50_min:.2f}-{r.p50_max:.2f} "
                 f"| {r.p95_ms:.2f} | {r.fps:.1f} | {pw} | {fw} |")
lines += ["", "\\* TensorRT INT8 engine built without calibration data: speed only, accuracy not valid.",
          f"Medians over {int(summary['rounds'].min())}-{int(summary['rounds'].max())} rounds; "
          "range = min-max of per-round p50."]
with open("results/summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
y = list(range(len(summary)))
colors = summary["system"].map(COLORS)
ax1.barh(y, summary["p50_ms"], color=colors,
         xerr=[summary["p50_ms"] - summary["p50_min"], summary["p50_max"] - summary["p50_ms"]],
         capsize=3)
ax1.set_xscale("log")
ax1.set_xlabel("p50 latency, ms (log scale) - lower is better")
ax2.barh(y, summary["fps"], color=colors)
ax2.set_xlabel("Throughput, FPS - higher is better")
ax1.set_yticks(y)
ax1.set_yticklabels(summary["label"])
ax1.invert_yaxis()
ax2.legend(handles=[Patch(color=c, label=n) for n, c in COLORS.items()], loc="lower right")
fig.suptitle("ResNet50, batch 1: OpenVINO on laptop vs TensorRT on Jetson Orin Nano")
fig.text(0.01, 0.01, "* TensorRT INT8 uncalibrated: speed only", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/chart.png", dpi=150)
print(open("results/summary.md", encoding="utf-8").read())