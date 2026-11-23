import numpy as np
import onnxruntime as ort
from common import time_fn, log_result

sess = ort.InferenceSession("models/resnet50.onnx", providers=["CPUExecutionProvider"])
x = np.random.rand(1, 3, 224, 224).astype(np.float32)

times = time_fn(lambda: sess.run(None, {"input": x}))
log_result("results/laptop.csv", "onnxruntime", "cpu", "fp32", 1, times)