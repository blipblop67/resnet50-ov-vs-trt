import glob
import random
import nncf
import numpy as np
import openvino as ov
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Imagenette folder -> ImageNet-1k class index
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
    return a.transpose(2, 0, 1)[None]  # 1x3x224x224


def sample(split, n, seed):
    files = glob.glob(f"{ROOT}/{split}/*/*.JPEG")
    random.Random(seed).shuffle(files)
    files = files[:n]
    labels = [CLASS_IDX[f.replace("\\", "/").split("/")[-2]] for f in files]
    return files, labels


def accuracy(core, model_path, files, labels):
    compiled = core.compile_model(model_path, "CPU")
    hits = sum(int(compiled(load(f))[0].argmax()) == l for f, l in zip(files, labels))
    return hits / len(files)


core = ov.Core()
model = core.read_model("models/r50_fp32.xml")
model.reshape([1, 3, 224, 224])

calib_files, _ = sample("train", 300, seed=0)
calib = nncf.Dataset([load(f) for f in calib_files])
int8 = nncf.quantize(model, calib)
ov.save_model(int8, "models/r50_int8.xml")

val_files, val_labels = sample("val", 500, seed=1)
acc32 = accuracy(core, "models/r50_fp32.xml", val_files, val_labels)
acc8 = accuracy(core, "models/r50_int8.xml", val_files, val_labels)
text = (f"Top-1 accuracy on {len(val_files)} Imagenette val images (1000-way ImageNet head)\n"
        f"- FP32: {acc32:.1%}\n- INT8: {acc8:.1%}\n- Drop: {(acc32 - acc8) * 100:.1f} points\n")
print(text)
with open("results/accuracy.md", "w") as f:
    f.write(text)