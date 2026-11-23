import hashlib
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

with open("models/resnet50.onnx", "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
with open("models/resnet50.onnx.sha256", "w") as f:
    f.write(digest + "\n")
print("exported, sha256:", digest)