"""Versioned, shared training/serving features. No learned or label-derived values."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

FEATURE_VERSION = "efficientnet-b0-1280+dct63+phash63-v1"
DEEP_DIM = 1280
HANDCRAFTED_DIM = 126
FUSION_DIM = DEEP_DIM + HANDCRAFTED_DIM
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rgb_input(image):
    """Shortest edge 256/bicubic, floor-centered 224 crop, shared by train/serve."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = 256 / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    return image.crop((left, top, left + 224, top + 224))


def tensor_input(image):
    return normalize_crop(rgb_input(image))


def normalize_crop(crop):
    array = (np.asarray(crop, dtype=np.float32) / 255 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def rgb_thumbnail(crop):
    return np.asarray(crop.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255


def frequency_features(image):
    """63 signed-log AC coefficients + 63 thresholded bits from the same RGB crop."""
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    ac = cv2.dct(gray)[:8, :8].ravel()[1:]
    dct = np.sign(ac) * np.log1p(np.abs(ac))
    bits = (ac > np.median(ac)).astype(np.float32)
    return np.concatenate([dct, bits]).astype(np.float32)


def model_inputs(image):
    crop = rgb_input(image)
    return normalize_crop(crop), frequency_features(crop)


@dataclass(frozen=True)
class ReusePolicy:
    """Approximate reuse is a measured experiment, never an authenticity lookup."""

    hamming: int = 2
    max_gap: int = 12
    max_rgb_mae: float = 0.01
    max_dct_mae: float = 0.015


class FrameSelector:
    def __init__(self, policy=None):
        self.policy = policy or ReusePolicy()
        self.anchor = None
        self.anchor_index = None

    def select(self, image, index):
        crop = rgb_input(image)
        vector = frequency_features(crop)
        return self.select_prepared(vector, rgb_thumbnail(crop), index)

    def select_prepared(self, vector, thumb, index):
        reuse = False
        if self.anchor is not None:
            old_vector, old_thumb = self.anchor
            reuse = (
                index - self.anchor_index < self.policy.max_gap
                and np.count_nonzero(vector[63:] != old_vector[63:]) <= self.policy.hamming
                and np.abs(thumb - old_thumb).mean() <= self.policy.max_rgb_mae
                and np.abs(vector[:63] - old_vector[:63]).mean() <= self.policy.max_dct_mae
            )
        if reuse:
            return self.anchor_index
        self.anchor, self.anchor_index = (vector, thumb), index
        return None
