"""Bounded still-image decoding shared by the web service and fusion pipeline."""

import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import AnalysisError

Image.MAX_IMAGE_PIXELS = 20_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


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
        return image, {
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
