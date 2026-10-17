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
    assert torch.equal(head.scale, torch.tensor([1.0, 1.0]))
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


def test_fusion_service_streams_every_frame_and_discloses_reuse(tmp_path):
    import cv2
    from types import SimpleNamespace
    from authenfy.fusion_analysis import analyze_fusion

    path = tmp_path / "stream.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for _ in range(25):
        writer.write(np.full((64, 64, 3), 128, np.uint8))
    writer.release()
    sizes = []

    def predict(pixels, crafted):
        sizes.append(len(pixels))
        assert len(pixels) == len(crafted)
        return [0.1] * len(pixels)

    engine = SimpleNamespace(identity={"id": "unit-test-stub"}, predict_inputs=predict)
    result = analyze_fusion(path, engine)
    assert len(result["frames"]) == 25
    assert result["unique_frames_analyzed"] == 3
    assert result["phash_reused_frames"] == 22
    assert result["frames"][-1]["timestamp_seconds"] == 2.4
    assert result["verdict"] == "inconclusive"
    assert sum(sizes) == 3


def test_fusion_mode_never_silently_falls_back_to_legacy(tmp_path):
    from app import create_app

    app = create_app(
        dict(
            TESTING=True,
            ENGINE="fusion",
            FUSION_DIR=str(tmp_path / "absent"),
            DATABASE=str(tmp_path / "receipts.db"),
            SIGNING_KEY_PATH=str(tmp_path / "key"),
        )
    )
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json["code"] == "model_unavailable"
    assert b"EfficientNet-B0" in client.get("/").data


@pytest.mark.parametrize("card", [None, [], "invalid", 17])
def test_invalid_fusion_card_reports_unavailable(tmp_path, card):
    import json
    from authenfy.fusion_model import FusionDetector
    from authenfy.errors import AnalysisError

    (tmp_path / "model-card.json").write_text(json.dumps(card))
    engine = FusionDetector(tmp_path)
    assert not engine.available
    with pytest.raises(AnalysisError, match="unavailable"):
        engine.validate()


def test_installer_rejects_modified_training_provenance(tmp_path):
    import json
    from scripts.install_fusion import install
    from authenfy.fusion_features import FEATURE_VERSION

    source, target = tmp_path / "run", tmp_path / "installed"
    source.mkdir()
    (source / "efficientnet-fusion.onnx").write_bytes(b"integrity-test-content")
    training = dict(status="trained", manifest_sha256="manifest", weights_sha256={"fusion.pt": "weights"})
    (source / "training.json").write_text(json.dumps(training))
    benchmark = dict(status="measured", manifest_sha256="manifest", weights_sha256=training["weights_sha256"])
    (source / "benchmark.json").write_text(json.dumps(benchmark))
    card = dict(
        status="validated",
        feature_version=FEATURE_VERSION,
        model_sha256=file_sha(source / "efficientnet-fusion.onnx"),
        benchmark_sha256=file_sha(source / "benchmark.json"),
        training_sha256=file_sha(source / "training.json"),
        training_manifest_sha256="manifest",
        training_weights_sha256=training["weights_sha256"],
        task="facial_manipulation",
    )
    (source / "model-card.json").write_text(json.dumps(card))
    install(source, target)
    assert (target / "training.json").read_bytes() == (source / "training.json").read_bytes()
    training["manifest_sha256"] = "different-run"
    (source / "training.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="provenance mismatch"):
        install(source, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_benchmark_resume_preserves_completed_frames_and_rejects_corruption(tmp_path):
    from training.recording import VideoLedger

    identity = dict(arms=["baseline_all", "fusion_phash"], revision="reference", repeats=3)
    ledger = VideoLedger(tmp_path, identity)
    frames = dict(baseline_all=[0.1, 0.2, 0.3], fusion_phash=[0.1, 0.1, 0.3])
    record = dict(path="video.mp4", measurements={arm: [dict(frames=3)] for arm in identity["arms"]})
    ledger.append(record, frames)
    resumed = VideoLedger(tmp_path, identity, resume=True)
    assert len(resumed.records) == 1
    for arm in identity["arms"]:
        np.testing.assert_array_equal(resumed.frames(0)[arm], frames[arm])
    with pytest.raises(ValueError, match="use --resume"):
        VideoLedger(tmp_path, identity)
    with pytest.raises(ValueError, match="runtime changed"):
        VideoLedger(tmp_path, dict(identity, repeats=1), resume=True)
    (tmp_path / "00000.npz").write_bytes(b"interrupted or corrupted transfer")
    with pytest.raises(ValueError, match="missing or corrupt"):
        VideoLedger(tmp_path, identity, resume=True)


def test_review_policy_records_abstention_separately():
    pytest.importorskip("sklearn")
    from scripts.summarize_fusion import review_policy

    def row(label, average, minimum, maximum, inferred=10):
        return dict(
            label=label,
            fusion_phash=average,
            measurements={
                "fusion_phash": [dict(min_score=minimum, max_score=maximum, frames=10, inferred_frames=inferred)]
            },
        )

    result = review_policy(
        [row(0, 0.1, 0.1, 0.1), row(0, 0.1, 0.1, 0.1, 1), row(1, 0.9, 0.9, 0.9), row(1, 0.8, 0.2, 0.9)],
        "fusion_phash",
    )
    assert result["counts"] == dict(synthetic_signals=1, low_synthetic_signals=1, inconclusive=2)
    assert result["decisive_fraction"] == 0.5
    assert result["selective_accuracy"] == 1.0


def test_training_resume_matches_uninterrupted_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from training import train
    from training.networks import EfficientBaseline

    torch.set_num_threads(2)
    manifest, _ = make_manifest(tmp_path)
    # Exercise the real optimizer/model with deterministic small code fixtures.
    monkeypatch.setattr(train, "EfficientBaseline", lambda pretrained: EfficientBaseline(pretrained=False))

    def loader(*args, **kwargs):
        return DataLoader(*args, **dict(kwargs, num_workers=0))

    monkeypatch.setattr(train, "DataLoader", loader)
    args = SimpleNamespace(
        manifest=manifest,
        output=tmp_path / "continuous",
        seed=7,
        require_t4=False,
        frames=1,
        epochs=2,
        head_epochs=1,
        batch_size=2,
        resume=False,
    )
    train.main(args)
    args.output = tmp_path / "interrupted"
    save = train.atomic_save

    def interrupt_after_snapshot(value, path):
        save(value, path)
        if path.name == "baseline-training-state.pt":
            raise RuntimeError("simulated interruption after durable snapshot")

    monkeypatch.setattr(train, "atomic_save", interrupt_after_snapshot)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.main(args)
    monkeypatch.setattr(train, "atomic_save", save)
    args.resume = True
    train.main(args)
    for name in ["baseline.pt", "deep_mlp.pt", "fusion.pt"]:
        expected = torch.load(tmp_path / "continuous" / name, weights_only=True)
        actual = torch.load(args.output / name, weights_only=True)
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected), name
