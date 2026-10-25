"""CPU ONNX serving for the newly trained EfficientNet/DCT/pHash architecture."""

import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np

from .errors import AnalysisError
from .fusion_features import FEATURE_VERSION
from .integrity import sha256_file


class FusionDetector:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "efficientnet-fusion.onnx"
        self.card_path = self.directory / "model-card.json"
        self._session = None
        self._lock = threading.Lock()
        self.identity = {
            "id": "authenfy/efficientnet-b0-dct-phash-mlp",
            "feature_version": FEATURE_VERSION,
            "scope": "FF++ facial manipulation screening",
        }
        self.card = None
        if self.card_path.is_file():
            try:
                self.card = json.loads(self.card_path.read_text())
                self.identity.update(
                    sha256=self.card["model_sha256"],
                    training_manifest_sha256=self.card["training_manifest_sha256"],
                    format="ONNX FP32",
                )
            except (ValueError, KeyError, TypeError, OSError):
                self.card = None

    @property
    def available(self):
        return self.path.is_file() and self.card is not None

    def validate(self):
        if not self.available:
            raise AnalysisError(
                "Trained fusion model unavailable. Install a completed training/export bundle.",
                503,
                "model_unavailable",
            )
        if self.card.get("status") != "validated" or self.card.get("feature_version") != FEATURE_VERSION:
            raise AnalysisError("Fusion feature version or export validation mismatch.", 503, "model_integrity")
        if sha256_file(self.path) != self.card["model_sha256"]:
            raise AnalysisError("Fusion model checksum mismatch.", 503, "model_integrity")

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

    def predict_inputs(self, pixels, handcrafted):
        result = np.asarray(
            self._load().run(None, {"pixels": np.stack(pixels), "handcrafted": np.stack(handcrafted)})[0]
        ).reshape(-1)
        if len(result) != len(pixels) or not np.isfinite(result).all():
            raise AnalysisError("Fusion model returned invalid scores.", 503, "inference_failed")
        return (1 / (1 + np.exp(-np.clip(result, -80, 80)))).tolist()

    def analyze_file(self, path, timeout=90):
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "authenfy.fusion_worker", str(path), str(self.directory)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
        except subprocess.TimeoutExpired as error:
            raise AnalysisError(
                "Analysis exceeded 90 seconds. Use a shorter video.", 504, "analysis_timeout"
            ) from error
        if completed.returncode != 0:
            raise AnalysisError("The isolated fusion analyzer failed.", 503, "inference_failed")
        try:
            result = json.loads(completed.stdout)
        except ValueError as error:
            raise AnalysisError("Invalid analyzer response.", 503, "inference_failed") from error
        if "error" in result:
            raise AnalysisError(result["error"], result["status"], result["code"])
        return result
