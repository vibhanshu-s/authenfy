"""Application entry point. Run with waitress-serve --host=127.0.0.1 app:app."""

import os
import re
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from authenfy.analysis import analyze
from authenfy.errors import AnalysisError
from authenfy.media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from authenfy.model import Detector, sha256_file
from authenfy.receipts import ReceiptStore

ROOT = Path(__file__).resolve().parent


def create_app(config=None, detector=None):
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=64 * 1024 * 1024 + 128 * 1024,
        MAX_MEDIA_BYTES=64 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=128 * 1024,
        MAX_FORM_PARTS=8,
        UPLOAD_FOLDER=str(ROOT / "uploads"),
        MODEL_PATH=os.environ.get("AUTHENFY_MODEL_PATH", str(ROOT / "artifacts/model.onnx")),
        DATABASE=str(ROOT / "instance/receipts.sqlite3"),
        SIGNING_KEY_PATH=str(ROOT / "instance/receipts.key"),
    )
    app.config.update(config or {})
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    engine = detector or Detector(app.config["MODEL_PATH"])
    receipts = ReceiptStore(app.config["DATABASE"], app.config["SIGNING_KEY_PATH"])
    slots = threading.BoundedSemaphore(1)
    app.extensions.update(detector=engine, receipts=receipts)

    @app.before_request
    def check_origin():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and urlparse(origin).netloc != request.host:
                raise AnalysisError("Cross-origin uploads are not allowed.", 403, "origin_rejected")

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.path.startswith(("/analyze", "/verify", "/receipts")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(AnalysisError)
    def analysis_error(error):
        return jsonify(error=str(error), code=error.code), error.status

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description, code=error.name.lower().replace(" ", "_")), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("Analysis request failed")
        return jsonify(error="Analysis failed. Please retry or check the server logs.", code="internal_error"), 500

    @app.get("/")
    def home():
        return render_template("home.html")

    @app.get("/upload")
    def upload_page():
        return render_template("upload.html")

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/contact")
    def contact():
        return render_template("contact.html")

    @app.get("/form")
    def form():
        return render_template("form.html")

    @app.get("/health")
    def health():
        if isinstance(engine, Detector):
            engine.validate()
        return jsonify(
            status="ready" if engine.available else "model_unavailable",
            model=engine.identity,
            model_available=engine.available,
            facial_specialist_available=engine.faces.available if hasattr(engine, "faces") else False,
            blockchain_anchored=False,
        ), 200 if engine.available else 503

    def save_upload():
        files = request.files.getlist("media")
        if len(files) != 1 or not files[0].filename:
            raise AnalysisError("Choose exactly one media file.", 400, "missing_media")
        media = files[0]
        suffix = Path(media.filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise AnalysisError("Use JPEG, PNG, WebP, MP4, MOV, AVI, WebM or MKV.", 415, "unsupported_type")
        # Random names isolate concurrent uploads; user filenames never become filesystem paths.
        descriptor, filename = tempfile.mkstemp(suffix=suffix, dir=app.config["UPLOAD_FOLDER"])
        path = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                media.save(handle)
            if path.stat().st_size == 0:
                raise AnalysisError("The uploaded file is empty.", 400, "empty_media")
            if path.stat().st_size > app.config["MAX_MEDIA_BYTES"]:
                raise AnalysisError("The file exceeds 64 MiB.", 413, "file_too_large")
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @app.post("/analyze")
    def analyze_video():
        if not slots.acquire(blocking=False):
            raise AnalysisError("The analyzer is busy. Retry shortly.", 429, "busy")
        path = None
        try:
            path = save_upload()
            result = engine.analyze_file(path) if isinstance(engine, Detector) else analyze(path, engine)
            receipt = receipts.put(result, sha256_file(path))
            return jsonify(receipt)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            slots.release()

    def get_receipt(receipt_id):
        if not re.fullmatch(r"[0-9a-f]{32}", receipt_id):
            raise AnalysisError("Receipt not found.", 404, "not_found")
        try:
            receipt = receipts.get(receipt_id)
        except ValueError as error:
            raise AnalysisError("Receipt integrity check failed.", 409, "receipt_integrity") from error
        if receipt is None:
            raise AnalysisError("Receipt not found.", 404, "not_found")
        return receipt

    @app.get("/receipts/<receipt_id>")
    def receipt_download(receipt_id):
        response = jsonify(get_receipt(receipt_id))
        response.headers["Content-Disposition"] = f'attachment; filename="authenfy-{receipt_id}.json"'
        return response

    @app.post("/verify")
    def verify():
        receipt = get_receipt(request.form.get("receipt_id", ""))
        path = save_upload()
        try:
            matches = sha256_file(path) == receipt["file_sha256"]
            return jsonify(
                file_matches=matches,
                receipt_integrity=True,
                receipt_id=receipt["receipt_id"],
                blockchain_anchored=False,
                meaning="Exact-byte match to a local analysis receipt. This does not prove authenticity.",
            )
        finally:
            path.unlink(missing_ok=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
