import io
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from PIL import Image

from app import create_app
from authenfy.errors import AnalysisError


class FakeDetector:
    available = True
    identity = {"id": "test-double", "revision": "test", "sha256": "test"}

    def __init__(self, score=0.5):
        self.score = score
        self.calls = []

    def predict(self, images):
        self.calls.append(len(images))
        return [self.score] * len(images)


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "DATABASE": str(tmp_path / "receipts.sqlite3"),
            "SIGNING_KEY_PATH": str(tmp_path / "receipts.key"),
        },
        FakeDetector(),
    )


def png(color="red"):
    handle = io.BytesIO()
    Image.new("RGB", (80, 64), color).save(handle, format="PNG")
    return handle.getvalue()


def upload(client, data=None, name="test.png"):
    return client.post("/analyze", data={"media": (io.BytesIO(png() if data is None else data), name)})


@pytest.mark.parametrize("route", ["/", "/upload", "/about", "/contact", "/form", "/health"])
def test_routes_render(app, route):
    assert app.test_client().get(route).status_code == 200


def test_receipt_and_verify(app):
    client = app.test_client()
    result = upload(client).get_json()
    assert result["verdict"] == "inconclusive"
    assert result["review_required"] is True
    assert result["model"]["id"] == "test-double"
    assert not result["provenance"]["blockchain_anchored"]
    assert "test.png" not in str(result)
    receipt_id = result["receipt_id"]
    response = client.get(f"/receipts/{receipt_id}")
    assert response.get_json() == result
    assert response.headers["Cache-Control"] == "no-store"
    match = client.post("/verify", data={"receipt_id": receipt_id, "media": (io.BytesIO(png()), "same.png")})
    assert match.get_json()["file_matches"]
    mismatch = client.post("/verify", data={"receipt_id": receipt_id, "media": (io.BytesIO(png("blue")), "same.png")})
    assert not mismatch.get_json()["file_matches"]
    assert not list(__import__("pathlib").Path(app.config["UPLOAD_FOLDER"]).iterdir())


@pytest.mark.parametrize(
    "data,name,status", [(b"", "a.png", 400), (b"junk", "a.png", 422), (b"junk", "a.mp4", 422), (b"junk", "a.exe", 415)]
)
def test_invalid_uploads(app, data, name, status):
    assert upload(app.test_client(), data, name).status_code == status
    assert not list(__import__("pathlib").Path(app.config["UPLOAD_FOLDER"]).iterdir())


def test_missing_and_multiple_media(app):
    client = app.test_client()
    assert client.post("/analyze").status_code == 400
    response = client.post("/analyze", data={"media": [(io.BytesIO(png()), "a.png"), (io.BytesIO(png()), "b.png")]})
    assert response.status_code == 400


def test_size_limit_is_json(app):
    app.config["MAX_CONTENT_LENGTH"] = 100
    response = upload(app.test_client())
    assert response.status_code == 413
    assert response.is_json


def test_media_limit_is_separate_from_multipart_overhead(app):
    payload = png()
    app.config["MAX_MEDIA_BYTES"] = len(payload)
    assert upload(app.test_client(), payload).status_code == 200
    app.config["MAX_MEDIA_BYTES"] = len(payload) - 1
    assert upload(app.test_client(), payload).status_code == 413
    assert not list(__import__("pathlib").Path(app.config["UPLOAD_FOLDER"]).iterdir())


def test_traversal_filename_is_not_a_path(app):
    assert upload(app.test_client(), name="../../../../evil.png").status_code == 200
    assert not (__import__("pathlib").Path(app.config["UPLOAD_FOLDER"]).parent / "evil.png").exists()


def test_integrity_tampering_detected(app):
    client = app.test_client()
    receipt = upload(client).get_json()
    with sqlite3.connect(app.config["DATABASE"]) as connection:
        connection.execute("UPDATE receipts SET payload = replace(payload, 'inconclusive', 'fake')")
    assert client.get(f"/receipts/{receipt['receipt_id']}").status_code == 409


def test_server_errors_do_not_expose_internals_and_cleanup(app):
    def fail(_):
        raise RuntimeError("private-server-file-path")

    app.extensions["detector"].predict = fail
    response = upload(app.test_client())
    assert response.status_code == 500
    assert b"private-server-file-path" not in response.data
    assert not list(__import__("pathlib").Path(app.config["UPLOAD_FOLDER"]).iterdir())


def test_missing_model_is_not_fake_prediction(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(tmp_path / "missing.onnx"),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "DATABASE": str(tmp_path / "db"),
            "SIGNING_KEY_PATH": str(tmp_path / "key"),
        }
    )
    assert app.test_client().get("/health").status_code == 503
    assert upload(app.test_client()).status_code == 503


def test_cross_origin_rejected(app):
    assert app.test_client().post("/analyze", headers={"Origin": "https://hostile.example"}).status_code == 403


def test_busy_analyzer_does_not_mix_uploads(app):
    entered, release = threading.Event(), threading.Event()

    def wait_predict(images):
        entered.set()
        assert release.wait(5)
        return [0.5] * len(images)

    app.extensions["detector"].predict = wait_predict
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(lambda: upload(app.test_client(), name="same.png"))
        assert entered.wait(5)
        try:
            assert upload(app.test_client(), name="same.png").status_code == 429
        finally:
            release.set()
        assert first.result().status_code == 200


def test_video_temporal_coverage_and_exact_duplicates(app, tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((64, 64, 3), 128, np.uint8))
    writer.release()
    result = upload(app.test_client(), path.read_bytes(), "clip.avi").get_json()
    assert result["media"]["kind"] == "video"
    assert result["media"]["sampled_frames"] == 16
    assert result["frames"][0]["timestamp_seconds"] == 0
    assert result["frames"][-1]["timestamp_seconds"] == 1.9
    assert result["unique_frames_analyzed"] == 1
    assert app.extensions["detector"].calls == [1]


def test_wrong_model_checksum_fails(tmp_path):
    from authenfy.model import Detector

    path = tmp_path / "bad.onnx"
    path.write_bytes(b"wrong weights")
    with pytest.raises(AnalysisError, match="checksum"):
        Detector(path).predict([Image.new("RGB", (64, 64))])


def test_native_worker_timeout_cleans_upload(tmp_path, monkeypatch):
    import subprocess
    from authenfy.model import Detector

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("isolated-analyzer", 90)

    monkeypatch.setattr("authenfy.model.subprocess.run", timeout)
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "DATABASE": str(tmp_path / "db"),
            "SIGNING_KEY_PATH": str(tmp_path / "key"),
        },
        Detector(tmp_path / "model.onnx"),
    )
    response = upload(app.test_client())
    assert response.status_code == 504
    assert response.get_json()["code"] == "analysis_timeout"
    assert not list((tmp_path / "uploads").iterdir())


def test_receipts_survive_application_restart(app):
    receipt = upload(app.test_client()).get_json()
    second = create_app(dict(app.config), FakeDetector())
    fetched = second.test_client().get(f"/receipts/{receipt['receipt_id']}").get_json()
    assert fetched == receipt
    assert len(fetched["receipt_commitment"]) == 64
