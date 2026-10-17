"""Run in a Colab T4 cell. Downloads a JSON receipt; never trains on smoke inputs."""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.4.0", "onnxruntime==1.31.0"], check=True)
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from transformers import ViTForImageClassification
from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
REVISION = "ac6ee457bea904a373065754107451793b56db00"
EXPECTED = "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1"
assert torch.cuda.is_available(), "A CUDA GPU is required. Select Runtime > Change runtime type > T4 GPU."
gpu = torch.cuda.get_device_name(0)
assert "T4" in gpu, f"Expected T4, got {gpu}"
print("GPU:", gpu, flush=True)
model = ViTForImageClassification.from_pretrained(MODEL_ID, revision=REVISION, use_safetensors=True).eval().cuda()
processor = CLIPImageProcessorPil.from_pretrained(MODEL_ID, revision=REVISION)
path = hf_hub_download(MODEL_ID, "onnx/model.onnx", revision=REVISION)
digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest == EXPECTED
options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def prepare(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = 440 / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 384) // 2, (image.height - 384) // 2
    image = image.crop((left, top, left + 384, top + 384))
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    return np.ascontiguousarray(((np.asarray(image, np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])


rng = np.random.default_rng(42)
inputs = [("gray", Image.new("RGB", (384, 384), (128, 128, 128))),
          ("rectangular_noise", Image.fromarray(rng.integers(0, 256, (320, 640, 3), dtype=np.uint8))),
          ("portrait_noise", Image.fromarray(rng.integers(0, 256, (640, 320, 3), dtype=np.uint8)))]
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
r = requests.get(url, timeout=60)
r.raise_for_status()
inputs.append(("public_cat_photo", Image.open(io.BytesIO(r.content)).convert("RGB")))
records = []
for name, image in inputs:
    x = prepare(image)
    reference = processor(image, return_tensors="np")["pixel_values"]
    preprocessing_error = float(np.max(np.abs(x - reference)))
    assert preprocessing_error < 1e-5, (name, "preprocess mismatch", preprocessing_error)
    tensor = torch.from_numpy(x).cuda()
    with torch.inference_mode():
        model(pixel_values=tensor)
        torch.cuda.synchronize()
        start = time.perf_counter()
        logit = model(pixel_values=tensor).logits
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - start) * 1000
        p_gpu = torch.sigmoid(logit).item()
    start = time.perf_counter()
    raw = float(session.run(None, {"pixel_values": x})[0].reshape(-1)[0])
    cpu_ms = (time.perf_counter() - start) * 1000
    p_cpu = float(1 / (1 + np.exp(-np.clip(raw, -80, 80))))
    row = {"input": name, "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
           "gpu_score": p_gpu, "onnx_score": p_cpu, "absolute_error": abs(p_gpu-p_cpu),
           "preprocessing_max_error": preprocessing_error, "gpu_ms": gpu_ms, "cpu_ms": cpu_ms}
    records.append(row)
    print(json.dumps(row), flush=True)
    assert row["absolute_error"] < 1e-4, "GPU/ONNX parity failed"
report = {"status": "passed", "kind": "runtime_smoke_and_numerical_parity", "gpu": gpu,
          "utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
          "onnxruntime": ort.__version__, "preprocessor": "CLIPImageProcessorPil", "model": MODEL_ID, "revision": REVISION,
          "model_sha256": digest, "inputs": records,
          "limitations": "Four smoke inputs; no detection accuracy, calibration, face-swap or audio validation."}
Path("authenfy-colab-validation.json").write_text(json.dumps(report, indent=2))
print("AUTHENFY_T4_VALIDATION_PASSED", json.dumps(report), flush=True)
from google.colab import files
files.download("authenfy-colab-validation.json")
