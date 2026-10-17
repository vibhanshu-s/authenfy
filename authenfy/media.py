"""Bounded decoding and uniform temporal sampling. Never reuse pHash as a verdict."""

from dataclasses import dataclass
import math
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import AnalysisError

Image.MAX_IMAGE_PIXELS = 20_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


@dataclass
class Frame:
    image: Image.Image
    timestamp: float
    duplicate_of: int | None = None


def check_dimensions(width, height):
    if min(width, height) < 32 or width * height > Image.MAX_IMAGE_PIXELS:
        raise AnalysisError("Media dimensions must be at least 32 x 32 and at most 20 megapixels.")
    if max(width, height) / min(width, height) > 8:
        raise AnalysisError("Media aspect ratio must be at most 8:1.")


def decode_image(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
                    raise AnalysisError("Use a static JPEG, PNG or WebP image.")
                check_dimensions(*source.size)
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
        return [Frame(image, 0)], {
            "kind": "image",
            "width": image.width,
            "height": image.height,
            "duration_seconds": None,
            "sampled_frames": 1,
        }
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        ValueError,
    ) as error:
        raise AnalysisError("The image is corrupt, unsupported, or exceeds the pixel limit.") from error


def decode_video(path, max_frames=16, max_duration=120):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise AnalysisError("The video could not be decoded.")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        check_dimensions(width, height)
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count < 1:
            raise AnalysisError("The video has no reliable frame count or frame rate.")
        duration = count / fps
        if duration > max_duration:
            raise AnalysisError(f"Use a video no longer than {max_duration} seconds.")
        frames = []
        for index in np.linspace(0, int(count) - 1, min(max_frames, int(count)), dtype=int):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                raise AnalysisError("The video does not support reliable frame sampling.")
            success, bgr = capture.read()
            if not success:
                raise AnalysisError("A sampled video frame could not be decoded; analysis was stopped.")
            check_dimensions(bgr.shape[1], bgr.shape[0])
            frames.append(Frame(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), round(float(index / fps), 3)))
        return frames, {
            "kind": "video",
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 3),
            "sampled_frames": len(frames),
        }
    finally:
        capture.release()


def phash(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    block = cv2.dct(gray)[:8, :8].flatten()[1:]
    return "".join("1" if x > np.median(block) else "0" for x in block)


def mark_exact_duplicates(frames):
    # pHash collisions can hide small manipulated regions. Only byte-identical RGB frames skip inference.
    import hashlib

    seen = {}
    for index, frame in enumerate(frames):
        digest = hashlib.sha256(str(frame.image.size).encode() + frame.image.tobytes()).hexdigest()
        frame.duplicate_of = seen.get(digest)
        seen.setdefault(digest, index)
