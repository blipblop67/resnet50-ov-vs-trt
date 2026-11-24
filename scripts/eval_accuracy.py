import glob
import random

import numpy as np
import openvino as ov
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_IDX = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
             "n03028079": 497, "n03394916": 566, "n03417042": 569,
             "n03425413": 571, "n03445777": 574, "n03888257": 701}
ROOT = "data/imagenette2-160"


def load(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    a = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)[None]


files = glob.glob(f"{ROOT}/val/*/*.JPEG")
random.Random(1).shuffle(files)
files = files[:500]
labels = [CLASS_IDX[f.replace("\\", "/").split("/")[-2]] for f in files]
data = [load(f) for f in files]

core = ov.Core()
runs = [("CPU", "fp32", "f32"), ("GPU", "fp32", "f32"), ("GPU", "fp16", "f16"),
        ("CPU", "int8", "f32"), ("GPU", "int8", "f16")]

lines = ["| Device | Model | Precision hint | Top-1 (500 val images) |", "|---|---|---|---|"]
for dev, prec, hint in runs:
    cm = core.compile_model(f"models/r50_{prec}.xml", dev,
                            {"INFERENCE_PRECISION_HINT": hint})
    hits = sum(int(cm(x)[0].argmax()) == l for x, l in zip(data, labels))
    lines.append(f"| {dev} | {prec} | {hint} | {hits / len(files):.1%} |")
    print(lines[-1])

with open("results/accuracy_by_config.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")