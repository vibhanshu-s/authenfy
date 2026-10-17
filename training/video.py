"""Research decoding: sequential all-frame evaluation, bounded sampled training frames."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from authenfy.media import IMAGE_EXTENSIONS


def iter_frames(path):
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            yield 0, image.convert("RGB")
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {path.name}")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            yield index, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            index += 1
        if index == 0 or (expected > 0 and index != expected):
            raise ValueError(f"Incomplete decode: {path.name}, {index}/{expected}")
    finally:
        capture.release()


def sample_frames(path, count=8):
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return [image for _, image in iter_frames(path)]
    capture = cv2.VideoCapture(str(path))
    try:
        size = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or size < 1:
            raise ValueError(f"Cannot decode {path.name}")
        result = []
        for index in np.linspace(0, size - 1, min(size, count), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode training frame {path.name}:{index}")
            result.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return result
    finally:
        capture.release()
