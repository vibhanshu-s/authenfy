"""Landmark alignment and optional specialist ONNX model. Missing coverage abstains."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .errors import AnalysisError
from .model import MEAN, STD, sha256_file

FACE_MODEL_ID = "yermandy/deepfake-detection"
FACE_REVISION = "9a6857ec642deb57373c5437be803a199468b8c6"
FACE_SOURCE_SHA = "55c54d3f4d301ce07f0d08126fc4202572d20ea6952b039bc7a949a43714157d"
YUNET_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def align_face(rgb, landmarks):
    source = np.asarray(landmarks, np.float32).reshape(5, 2).copy()
    source[:2] = source[:2][np.argsort(source[:2, 0])]
    source[3:] = source[3:][np.argsort(source[3:, 0])]
    if np.linalg.norm(source[0] - source[1]) < 8:
        raise AnalysisError("Detected face landmarks are too small for reliable alignment.")
    target = np.array(
        [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]], np.float32
    )
    target[:, 0] += 8
    target *= 256 / 112
    target = (target + 256 * 0.3 / 2) / 1.3
    # Solve the deterministic least-squares similarity transform using all landmarks.
    rows, values = [], []
    for (x, y), (u, v) in zip(source, target):
        rows.extend([[x, -y, 1, 0], [y, x, 0, 1]])
        values.extend([u, v])
    a, b, tx, ty = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
    matrix = np.array([[a, -b, tx], [b, a, ty]], np.float32)
    if not np.isfinite(matrix).all():
        raise AnalysisError("Face alignment failed.")
    return Image.fromarray(cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR))


def preprocess_face(image):
    image = image.convert("RGB")
    scale = 224 / min(image.size)
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    array = (np.asarray(image, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


class FaceDetector:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "face-specialist.onnx"
        self.yunet = self.directory / "yunet.onnx"
        local_manifest = self.directory / "specialist-validation.json"
        self.manifest_path = (
            local_manifest
            if local_manifest.is_file()
            else Path(__file__).resolve().parents[1] / "reports/specialist-validation.json"
        )
        self._session = None
        self._face_detector = None
        self.identity = {
            "id": FACE_MODEL_ID,
            "revision": FACE_REVISION,
            "source_sha256": FACE_SOURCE_SHA,
            "scope": "facial manipulation screening",
            "alignment": "YuNet five-point approximation of DeepfakeBench alignment",
            "available": self.available,
        }

    @property
    def available(self):
        return self.path.is_file() and self.yunet.is_file() and self.manifest_path.is_file()

    def _load(self):
        if self._session is None:
            manifest = json.loads(self.manifest_path.read_text())
            valid = (
                manifest["status"] == "passed"
                and manifest["source_model"] == FACE_MODEL_ID
                and manifest["source_revision"] == FACE_REVISION
                and manifest["source_sha256"] == FACE_SOURCE_SHA
                and sha256_file(self.path) == manifest["onnx_sha256"]
                and sha256_file(self.yunet) == YUNET_SHA
            )
            if not valid:
                raise AnalysisError("Specialist model integrity check failed.", 503, "model_integrity")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._face_detector = cv2.FaceDetectorYN.create(str(self.yunet), "", (320, 320), 0.8, 0.3, 100)
            self.identity["sha256"] = manifest["onnx_sha256"]

    def predict(self, images):
        if not self.available:
            return [{"score": None, "count": 0, "coverage": "model_unavailable"} for image in images]
        if self._face_detector is None:
            if sha256_file(self.yunet) != YUNET_SHA:
                raise AnalysisError("Face detector integrity check failed.", 503, "model_integrity")
            self._face_detector = cv2.FaceDetectorYN.create(str(self.yunet), "", (320, 320), 0.8, 0.3, 100)
        results = []
        for image in images:
            scale = min(1.0, 1280 / max(image.size))
            resized = image.resize((max(32, int(image.width * scale)), max(32, int(image.height * scale))))
            rgb = np.asarray(resized.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._face_detector.setInputSize((resized.width, resized.height))
            _, detections = self._face_detector.detect(bgr)
            if detections is None:
                results.append({"score": None, "count": 0, "coverage": "no_face_detected"})
                continue
            detections = sorted(detections, key=lambda face: float(face[2] * face[3]), reverse=True)
            scores = []
            for face in detections[:4]:
                if min(face[2], face[3]) < 64:
                    continue
                try:
                    crop = align_face(rgb, face[4:14].reshape(5, 2))
                except AnalysisError:
                    continue
                self._load()
                logits = self._session.run(None, {"pixel_values": preprocess_face(crop)})[0].reshape(-1)
                if logits.shape != (2,) or not np.isfinite(logits).all():
                    raise AnalysisError("Invalid specialist model score.", 503, "inference_failed")
                probability = np.exp(logits - logits.max())
                scores.append(float(probability[1] / probability.sum()))
            complete = len(scores) == len(detections) and len(scores) <= 4
            results.append(
                {
                    "score": max(scores) if scores else None,
                    "count": len(scores),
                    "coverage": "detected_faces_analyzed" if complete else "partial_face_coverage",
                }
            )
        return results
