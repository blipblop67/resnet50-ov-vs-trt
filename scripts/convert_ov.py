import numpy as np
import onnxruntime as ort
import openvino as ov

core = ov.Core()
print("OpenVINO devices:", core.available_devices)

model = ov.convert_model("models/resnet50.onnx")
ov.save_model(model, "models/r50_fp32.xml", compress_to_fp16=False)
ov.save_model(model, "models/r50_fp16.xml", compress_to_fp16=True)

x = np.random.rand(1, 3, 224, 224).astype(np.float32)
ref = ort.InferenceSession("models/resnet50.onnx",
                           providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]

for dev in core.available_devices:
    out = core.compile_model("models/r50_fp32.xml", dev)(x)[0]
    print(f"{dev}: max abs diff = {np.abs(ref - out).max():.2e}, "
          f"top1 match = {ref.argmax() == out.argmax()}")