import math
import time

import numpy as np

from .errors import AnalysisError
from .media import IMAGE_EXTENSIONS, decode_image, decode_video, mark_exact_duplicates, phash

POLICY = "screening-v2-coverage-aware"
LIMITATIONS = [
    "This is an uncalibrated AI-generated image screening model; scores are not probabilities of authenticity.",
    "Face swaps, local edits and unseen generators may be missed. No-face, partial-coverage and unavailable-specialist cases cannot receive a low-signal verdict.",
    "Video analysis samples at most 16 frames. Audio, lip sync and rPPG are not analyzed.",
    "The center crop may exclude manipulated regions. Human review is required.",
    "YuNet alignment approximates the facial specialist's original dlib pipeline; this preprocessing requires broader independent validation.",
]


def analyze(path, detector):
    start = time.perf_counter()
    frames, metadata = decode_image(path) if path.suffix.lower() in IMAGE_EXTENSIONS else decode_video(path)
    mark_exact_duplicates(frames)
    unique = [frame for frame in frames if frame.duplicate_of is None]
    values = detector.predict([frame.image for frame in unique])
    facial = (
        detector.faces.predict([frame.image for frame in unique])
        if hasattr(detector, "faces")
        else [{"score": None, "count": 0, "coverage": "model_unavailable"} for frame in unique]
    )
    if len(values) != len(unique) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise AnalysisError("Detector returned invalid scores.", 503, "inference_failed")
    scores, rows, cursor = [], [], 0
    coverage_complete = True
    for index, frame in enumerate(frames):
        if frame.duplicate_of is not None:
            score = scores[frame.duplicate_of]
            generated_score = rows[frame.duplicate_of]["generated_image_score"]
            face_result = {
                "score": rows[frame.duplicate_of]["face_manipulation_score"],
                "count": rows[frame.duplicate_of]["faces_analyzed"],
                "coverage": rows[frame.duplicate_of]["face_coverage"],
            }
        else:
            generated_score = float(values[cursor])
            face_result = facial[cursor]
            score = max(generated_score, face_result["score"] or 0)
            cursor += 1
        coverage_complete &= face_result["coverage"] == "detected_faces_analyzed"
        scores.append(score)
        rows.append(
            {
                "index": index,
                "timestamp_seconds": frame.timestamp,
                "synthetic_score": round(score, 6),
                "duplicate_of": frame.duplicate_of,
                "generated_image_score": round(generated_score, 6),
                "face_manipulation_score": None if face_result["score"] is None else round(face_result["score"], 6),
                "faces_analyzed": face_result["count"],
                "face_coverage": face_result["coverage"],
                "perceptual_hash": phash(frame.image),
            }
        )
    average, maximum, minimum = float(np.mean(scores)), max(scores), min(scores)
    # These conservative operating bands are an explicit unvalidated policy, not calibrated thresholds.
    verdict = "synthetic_signals" if average >= 0.8 else "low_synthetic_signals" if maximum <= 0.2 else "inconclusive"
    if maximum - minimum > 0.5:
        verdict = "inconclusive"
    if verdict == "low_synthetic_signals" and not coverage_complete:
        verdict = "inconclusive"
    return {
        "schema_version": 1,
        "policy": POLICY,
        "verdict": verdict,
        "review_required": True,
        "synthetic_score": round(average, 6),
        "score_range": [round(minimum, 6), round(maximum, 6)],
        "model": detector.identity,
        "specialist_model": detector.faces.identity if hasattr(detector, "faces") else None,
        "coverage_complete": coverage_complete,
        "aggregation": "maximum of generated-image and facial-manipulation scores per frame, then temporal mean",
        "media": metadata,
        "frames": rows,
        "unique_frames_analyzed": len(unique),
        "limitations": LIMITATIONS,
        "processing_seconds": round(time.perf_counter() - start, 3),
        "provenance": {"kind": "local_signed_receipt", "blockchain_anchored": False},
    }
