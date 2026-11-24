import json
import os

import numpy as np
from common import log_result

run = os.environ.get("BENCH_RUN", "1")
for p in ["fp32", "fp16", "int8"]:
    with open(f"results/raw/trt_{p}_run{run}.json") as f:
        entries = json.load(f)
    times = np.array([e["latencyMs"] for e in entries])
    print(f"{p}: {len(times)} timed iterations")
    log_result("results/jetson.csv", "tensorrt", "gpu", p, 1, times,
               notes="trtexec latencyMs (H2D+compute+D2H)")
