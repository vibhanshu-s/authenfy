from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from authenfy.analysis import analyze
from authenfy.errors import AnalysisError
from authenfy.media import Frame, mark_exact_duplicates, phash
from authenfy.model import preprocess


def test_preprocess_shape_and_normalization():
    image = Image.new("RGB", (640, 320), (128, 128, 128))
    result = preprocess(image)
    assert result.shape == (1, 3, 384, 384)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result[0, 0, 0, 0] == pytest.approx((128 / 255 - 0.48145466) / 0.26862954, abs=1e-6)


def test_phash_collision_does_not_skip_inference():
    red, blue = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")
    assert phash(red) == phash(blue)
    frames = [Frame(red, 0), Frame(blue, 1), Frame(red.copy(), 2)]
    mark_exact_duplicates(frames)
    assert [frame.duplicate_of for frame in frames] == [None, None, 0]


@pytest.mark.parametrize("score,verdict", [(0.1, "inconclusive"), (0.5, "inconclusive"), (0.9, "synthetic_signals")])
def test_review_bands_are_explicit(tmp_path, score, verdict):
    from test_app import FakeDetector

    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "red").save(path)
    result = analyze(path, FakeDetector(score))
    assert result["verdict"] == verdict
    assert result["review_required"] is True


def test_low_signal_requires_specialist_coverage(tmp_path):
    from types import SimpleNamespace
    from test_app import FakeDetector

    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "red").save(path)
    model = FakeDetector(0.1)
    for coverage, expected in [
        ("detected_faces_analyzed", "low_synthetic_signals"),
        ("partial_face_coverage", "inconclusive"),
        ("no_face_detected", "inconclusive"),
        ("model_unavailable", "inconclusive"),
    ]:
        model.faces = SimpleNamespace(
            identity={"id": "test"},
            predict=lambda images: [{"score": 0.1, "count": 1, "coverage": coverage} for image in images],
        )
        assert analyze(path, model)["verdict"] == expected


def test_specialist_signal_is_not_diluted_by_generic_false_negative(tmp_path):
    from types import SimpleNamespace
    from test_app import FakeDetector

    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "red").save(path)
    model = FakeDetector(0.001)
    model.faces = SimpleNamespace(
        identity={"id": "test"},
        predict=lambda images: [{"score": 0.99, "count": 1, "coverage": "detected_faces_analyzed"} for image in images],
    )
    result = analyze(path, model)
    assert result["verdict"] == "synthetic_signals"
    assert result["frames"][0]["generated_image_score"] == 0.001
    assert result["frames"][0]["face_manipulation_score"] == 0.99


def test_face_preprocessing_shape_and_alignment():
    from authenfy.faces import align_face, preprocess_face

    image = Image.new("RGB", (256, 256), "red")
    points = [[70, 80], [180, 80], [125, 125], [80, 170], [170, 170]]
    crop = align_face(np.asarray(image), points)
    assert crop.size == (256, 256)
    assert preprocess_face(crop).shape == (1, 3, 224, 224)
    with pytest.raises(AnalysisError):
        align_face(np.asarray(image), [[1, 1]] * 5)


def test_nonfinite_model_result_is_rejected(tmp_path):
    from test_app import FakeDetector

    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "red").save(path)
    with pytest.raises(AnalysisError):
        analyze(path, FakeDetector(float("nan")))


def test_real_model_smoke():
    from authenfy.model import Detector

    path = Path(__file__).resolve().parents[1] / "artifacts/model.onnx"
    if not path.exists():
        pytest.skip("Download the pinned model to run integration smoke")
    result = Detector(path).predict([Image.new("RGB", (384, 384), (128, 128, 128))])
    assert result[0] == pytest.approx(0.003220907, abs=1e-5)
