import sys

import numpy as np
import openvino as ov
from common import time_fn, log_result

core = ov.Core()
x = np.random.rand(1, 3, 224, 224).astype(np.float32)

# (device, model precision, execution precision hint)
runs = [
    ("CPU", "fp32", "f32"),
    ("GPU", "fp32", "f32"),
    ("GPU", "fp16", "f16"),
    ("CPU", "int8", "f32"),
    ("GPU", "int8", "f16"),
]
precisions = sys.argv[1:] or ["fp32", "fp16"]

for dev, prec, hint in runs:
    if prec not in precisions:
        continue
    compiled = core.compile_model(
        f"models/r50_{prec}.xml", dev,
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": hint},
    )
    actual = compiled.get_property("INFERENCE_PRECISION_HINT")
    req = compiled.create_infer_request()
    times = time_fn(lambda: req.infer({0: x}))
    log_result("results/laptop.csv", "openvino", dev.lower(), prec, 1, times,
               notes=f"precision_hint={actual}")