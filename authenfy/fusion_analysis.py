"""All-frame streaming application analysis with bounded approximate pHash reuse."""

import math
import time

import cv2
import numpy as np
from PIL import Image

from .errors import AnalysisError
from .fusion_features import FrameSelector, frequency_features, model_inputs, rgb_input
from .media import IMAGE_EXTENSIONS, check_dimensions, decode_image


def service_video(path, metadata):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise AnalysisError("The video could not be decoded.")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        check_dimensions(width, height)
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count < 1:
            raise AnalysisError("The video has no reliable frame count or rate.")
        if count / fps > 120 or count > 3600:
            raise AnalysisError("Use a video no longer than 120 seconds and 3,600 frames.")
        metadata.update(
            kind="video",
            width=width,
            height=height,
            duration_seconds=round(count / fps, 3),
            sampled_frames=int(count),
            frame_policy="every decoded frame",
        )
        for index in range(int(count)):
            ok, bgr = capture.read()
            if not ok:
                raise AnalysisError("A video frame could not be decoded; analysis stopped.")
            check_dimensions(bgr.shape[1], bgr.shape[0])
            yield index, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), index / fps
        ok, _ = capture.read()
        if ok:
            raise AnalysisError("Video frame count is inconsistent; analysis stopped.")
    finally:
        capture.release()


def analyze_fusion(path, detector):
    start = time.perf_counter()
    metadata = {}
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        images, metadata = decode_image(path)
        iterator = [(0, images[0].image, 0.0)]
    else:
        iterator = service_video(path, metadata)
    selector, rows, scores = FrameSelector(), [], {}
    pixels, crafted, indices = [], [], []

    def flush():
        if pixels:
            predictions = detector.predict_inputs(pixels, crafted)
            for index, value in zip(indices, predictions, strict=True):
                scores[index] = value
            pixels.clear()
            crafted.clear()
            indices.clear()

    for index, image, timestamp in iterator:
        anchor = selector.select(image, index)
        bits = frequency_features(rgb_input(image))[63:]
        rows.append(
            dict(
                index=index,
                timestamp_seconds=round(timestamp, 4),
                duplicate_of=None,
                reused_from=anchor,
                reuse_kind="approximate_phash" if anchor is not None else None,
                face_coverage="full_frame_screening",
                faces_analyzed=0,
                generated_image_score=None,
                perceptual_hash="".join(str(int(bit)) for bit in bits),
            )
        )
        if anchor is None:
            x, h = model_inputs(image)
            pixels.append(x)
            crafted.append(h)
            indices.append(index)
        if len(pixels) == 8:
            flush()
    flush()
    values = []
    for row in rows:
        value = float(scores[row["index"] if row["reused_from"] is None else row["reused_from"]])
        values.append(value)
        row.update(
            synthetic_score=round(value, 6), fusion_score=round(value, 6), face_manipulation_score=round(value, 6)
        )
    average, maximum, minimum = float(np.mean(values)), max(values), min(values)
    reused = sum(row["reused_from"] is not None for row in rows)
    verdict = (
        "synthetic_signals"
        if average >= 0.8
        else "low_synthetic_signals"
        if maximum <= 0.2 and not reused
        else "inconclusive"
    )
    if maximum - minimum > 0.5:
        verdict = "inconclusive"
    return dict(
        schema_version=2,
        engine_version="efficientnet-fusion-v1",
        policy="fusion-v1-review-bands",
        verdict=verdict,
        review_required=True,
        synthetic_score=round(average, 6),
        score_range=[round(minimum, 6), round(maximum, 6)],
        model=detector.identity,
        specialist_model=None,
        coverage_complete=not bool(reused),
        aggregation="mean over every decoded frame; reused frames retain their temporal weight",
        media=metadata,
        frames=rows,
        unique_frames_analyzed=len(scores),
        phash_reused_frames=reused,
        limitations=[
            "EfficientNet-B0 + DCT + pHash fusion is trained for facial manipulation; fully AI-generated video detection is not yet enabled.",
            "Scores are uncalibrated. FF++ source-pair evaluation does not establish generalization to unseen identities or generators.",
            "Approximate pHash reuse can miss brief or subtle edits. A reused-frame video cannot receive a low-signal verdict.",
            "Full-frame visual screening only. Faces are not localized; audio, lip sync and rPPG are not analyzed. Human review is required.",
        ],
        processing_seconds=round(time.perf_counter() - start, 3),
        provenance=dict(kind="local_signed_receipt", blockchain_anchored=False),
    )
