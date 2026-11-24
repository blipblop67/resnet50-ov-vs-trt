import hashlib
import os

import torch
import torchvision

model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, "models/resnet50.onnx",
    opset_version=17,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)

lines = []
for p in ["models/resnet50.onnx", "models/resnet50.onnx.data"]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {p}")
with open("models/resnet50.onnx.sha256", "w", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
