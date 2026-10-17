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
