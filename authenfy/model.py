"""Pinned FP32 Community Forensics inference, with no remote code execution."""

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .errors import AnalysisError

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
REVISION = "ac6ee457bea904a373065754107451793b56db00"
MODEL_SHA256 = "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if min(w, h) < 32 or max(w, h) / min(w, h) > 8:
        raise AnalysisError("Use media at least 32 pixels wide and tall, with aspect ratio at most 8:1.")
    scale = 440 / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 384) // 2, (image.height - 384) // 2
    image = image.crop((left, top, left + 384, top + 384))
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


class Detector:
    def __init__(self, path):
        self.path = Path(path)
        self._session = None
        self._lock = threading.Lock()
        from .faces import FaceDetector

        self.faces = FaceDetector(self.path.parent)
        self.identity = {
            "id": MODEL_ID,
            "revision": REVISION,
            "sha256": MODEL_SHA256,
            "format": "ONNX FP32",
            "preprocessing": "PIL bicubic; shortest edge 440; center crop 384; CLIP normalization",
            "scope": "AI-generated image screening",
        }

    @property
    def available(self):
        return self.path.is_file()

    def validate(self):
        if not self.available:
            raise AnalysisError("Model unavailable. Run python scripts/download_model.py.", 503, "model_unavailable")
        if sha256_file(self.path) != MODEL_SHA256:
            raise AnalysisError("Model checksum mismatch. Re-download the pinned model.", 503, "model_integrity")

    def analyze_file(self, path, timeout=90):
        # Decode untrusted media and run inference in a killable child process.
        # A stalled native codec must not occupy the web worker indefinitely.
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "authenfy.worker", str(path), str(self.path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
        except subprocess.TimeoutExpired as error:
            raise AnalysisError(
                "Analysis exceeded 90 seconds. Try a shorter or smaller file.", 504, "analysis_timeout"
            ) from error
        if completed.returncode != 0:
            raise AnalysisError("The isolated analyzer failed. Check the server logs.", 503, "inference_failed")
        payload = json.loads(completed.stdout)
        if "error" in payload:
            raise AnalysisError(payload["error"], payload["status"], payload["code"])
        return payload

    def _load(self):
        with self._lock:
            if self._session is None:
                self.validate()
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                self._session = ort.InferenceSession(
                    str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
                )
        return self._session

    def predict(self, images):
        session = self._load()
        scores = []
        for image in images:
            logit = float(np.asarray(session.run(None, {"pixel_values": preprocess(image)})[0]).reshape(-1)[0])
            if not np.isfinite(logit):
                raise AnalysisError("Model returned a non-finite score.", 503, "inference_failed")
            scores.append(float(1 / (1 + np.exp(-np.clip(logit, -80, 80)))))
        return scores
