"""Feature and split regressions run without optional training dependencies."""

import numpy as np
from PIL import Image
import pytest

from authenfy.fusion_features import FrameSelector, ReusePolicy, frequency_features, model_inputs
from training.manifest import FIELDS, file_sha, load_manifest, write_manifest


def test_feature_layout_and_shared_crop():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 255, (321, 640, 3), dtype=np.uint8))
    pixels, handcrafted = model_inputs(image)
    assert pixels.shape == (3, 224, 224)
    assert pixels.dtype == handcrafted.dtype == np.float32
    assert handcrafted.shape == (126,)
    assert set(handcrafted[63:]).issubset({0, 1})
    assert np.isfinite(pixels).all() and np.isfinite(handcrafted).all()


def test_phash_collision_color_guard_and_forced_refresh():
    selector = FrameSelector(ReusePolicy(max_gap=3))
    red, blue = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")
    assert np.array_equal(frequency_features(red)[63:], frequency_features(blue)[63:])
    assert selector.select(red, 0) is None
    assert selector.select(red, 1) == 0
    assert selector.select(blue, 2) is None
    assert selector.select(blue, 3) == 2
    assert selector.select(blue, 5) is None


def make_manifest(tmp_path):
    rows = []
    for split in ["train", "val", "test"]:
        for label in [0, 1]:
            path = tmp_path / f"{split}-{label}.png"
            Image.new("RGB", (32, 32), (len(rows) * 30, 0, 0)).save(path)
            rows.append(
                dict(
                    zip(
                        FIELDS,
                        [
                            path.name,
                            label,
                            "manipulated" if label else "real",
                            split,
                            split,
                            split,
                            "test",
                            file_sha(path),
                        ],
                    )
                )
            )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    return manifest, rows


def test_manifest_accepts_source_disjoint_pairs(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    assert len(load_manifest(manifest)) == 6


def test_manifest_rejects_shared_source_and_identical_media(tmp_path):
    manifest, rows = make_manifest(tmp_path)
    rows[-1]["source_ids"] = "train"
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Source leakage"):
        load_manifest(manifest)
    rows[-1]["source_ids"] = "test"
    (tmp_path / rows[-1]["path"]).write_bytes((tmp_path / rows[0]["path"]).read_bytes())
    rows[-1]["sha256"] = rows[0]["sha256"]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Identical media"):
        load_manifest(manifest)


def test_manifest_rejects_hash_mismatch(tmp_path):
    manifest, rows = make_manifest(tmp_path)
    rows[-1]["sha256"] = "0" * 64
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(manifest)


def test_normalizer_uses_only_provided_training_features():
    torch = pytest.importorskip("torch")
    from training.networks import FusionHead

    head = FusionHead(2)
    head.fit_normalizer(torch.tensor([[1.0, 5.0], [3.0, 5.0]]))
    assert torch.equal(head.mean, torch.tensor([2.0, 5.0]))
    assert torch.isfinite(head(torch.tensor([[100.0, -50.0]]))).all()


def test_metrics_record_errors_and_calibration():
    pytest.importorskip("sklearn")
    from training.metrics import binary_metrics

    result = binary_metrics([0, 0, 1, 1], [0.0, 0.6, 0.4, 1.0], include_curves=True)
    assert result["accuracy"] == 0.5
    assert result["precision"] == result["recall"] == result["specificity"] == 0.5
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["brier_score"] == pytest.approx(0.18)
    assert sum(b["count"] for b in result["calibration_bins"]) == 4
    import json

    json.dumps(result, allow_nan=False)


def test_efficientnet_onnx_export_parity(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    import onnxruntime as ort
    from training.networks import FusionModel

    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = FusionModel().eval()
    pixels = torch.randn(2, 3, 224, 224)
    handcrafted = torch.randn(2, 126)
    path = tmp_path / "fusion.onnx"
    torch.onnx.export(
        model,
        (pixels, handcrafted),
        str(path),
        input_names=["pixels", "handcrafted"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(pixels, handcrafted).numpy()
    actual = session.run(None, {"pixels": pixels.numpy(), "handcrafted": handcrafted.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_all_frame_benchmark_and_temporal_weighting(tmp_path):
    torch = pytest.importorskip("torch")
    import cv2
    from training.benchmark import run_video

    class SmallModel:
        def __call__(self, pixels):
            return pixels.mean((1, 2, 3)) * 0

        def features(self, pixels):
            return torch.ones((len(pixels), 1280))

    class Head:
        def __call__(self, features):
            return features[:, 0] * 0

    path = tmp_path / "repeat.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for _ in range(25):
        writer.write(np.full((64, 64, 3), 128, np.uint8))
    writer.release()
    for arm, expected in [("baseline_all", 25), ("fusion_all", 25), ("fusion_phash", 3)]:
        result = run_video(path, SmallModel(), {"fusion": Head()}, arm, torch.device("cpu"), batch_size=8)
        assert result["frames"] == 25
        assert result["inferred_frames"] == expected
        assert result["score"] == 0.5
