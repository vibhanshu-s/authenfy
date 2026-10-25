"""Application entry point. Run with waitress-serve --host=127.0.0.1 app:app."""

import os
import re
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from authenfy.errors import AnalysisError
from authenfy.fusion_analysis import analyze_fusion
from authenfy.fusion_model import FusionDetector
from authenfy.integrity import sha256_file
from authenfy.media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from authenfy.receipts import ReceiptStore

ROOT = Path(__file__).resolve().parent


def create_app(config=None, detector=None):
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=64 * 1024 * 1024 + 128 * 1024,
        MAX_MEDIA_BYTES=64 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=128 * 1024,
        MAX_FORM_PARTS=8,
        UPLOAD_FOLDER=os.environ.get("AUTHENFY_UPLOAD_FOLDER", str(ROOT / "uploads")),
        FUSION_DIR=os.environ.get("AUTHENFY_FUSION_DIR", str(ROOT / "artifacts/efficientnet-v1")),
        DATABASE=os.environ.get("AUTHENFY_DATABASE", str(ROOT / "instance/receipts.sqlite3")),
        SIGNING_KEY_PATH=os.environ.get("AUTHENFY_SIGNING_KEY_PATH", str(ROOT / "instance/receipts.key")),
        BLOCKCHAIN_REGISTRY_ADDRESS=os.environ.get("AUTHENFY_REGISTRY_ADDRESS", ""),
        BLOCKCHAIN_CHAIN_ID=os.environ.get("AUTHENFY_CHAIN_ID", ""),
        BLOCKCHAIN_RPC_URL=os.environ.get("AUTHENFY_RPC_URL", ""),
        BLOCKCHAIN_CONFIRMATIONS=os.environ.get("AUTHENFY_CONFIRMATIONS", "2"),
    )
    app.config.update(config or {})
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    engine = detector or FusionDetector(app.config["FUSION_DIR"])
    receipts = ReceiptStore(app.config["DATABASE"], app.config["SIGNING_KEY_PATH"])
    registry = str(app.config["BLOCKCHAIN_REGISTRY_ADDRESS"]).strip()
    rpc_url = str(app.config["BLOCKCHAIN_RPC_URL"]).strip()
    try:
        chain_id = int(app.config["BLOCKCHAIN_CHAIN_ID"])
    except (TypeError, ValueError):
        chain_id = 0
    try:
        confirmations_required = max(1, int(app.config["BLOCKCHAIN_CONFIRMATIONS"]))
    except (TypeError, ValueError):
        confirmations_required = 2
    blockchain_enabled = bool(
        re.fullmatch(r"0x[a-fA-F0-9]{40}", registry)
        and chain_id > 0
        and urlparse(rpc_url).scheme in {"http", "https"}
        and urlparse(rpc_url).netloc
    )
    slots = threading.BoundedSemaphore(1)
    app.extensions.update(detector=engine, receipts=receipts, blockchain_enabled=blockchain_enabled)

    @app.context_processor
    def engine_context():
        return {
            "engine_mode": "fusion",
            "blockchain_config": {
                "enabled": blockchain_enabled,
                "registry": registry.lower() if blockchain_enabled else "",
                "chain_id": chain_id if blockchain_enabled else None,
                "confirmations": confirmations_required if blockchain_enabled else None,
            },
        }

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
        if request.path.startswith(("/analyze", "/verify", "/receipts", "/anchors")):
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
        if hasattr(engine, "validate"):
            engine.validate()
        return jsonify(
            status="ready" if engine.available else "model_unavailable",
            model=engine.identity,
            model_available=engine.available,
            blockchain_enabled=blockchain_enabled,
            engine="efficientnet-fusion-v1",
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
            result = engine.analyze_file(path) if hasattr(engine, "analyze_file") else analyze_fusion(path, engine)
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
        receipt = get_receipt(receipt_id)
        anchor = receipts.get_anchor("receipt", receipt_id)
        if anchor:
            receipt["blockchain_anchor"] = anchor
        response = jsonify(receipt)
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
                blockchain_anchored=receipts.get_anchor("receipt", receipt["receipt_id"]) is not None,
                blockchain_anchor=receipts.get_anchor("receipt", receipt["receipt_id"]),
                meaning="Exact-byte match to a local analysis receipt. This does not prove authenticity.",
            )
        finally:
            path.unlink(missing_ok=True)

    @app.get("/evidence")
    def evidence_page():
        path = ROOT / "reports/v1-evidence/evidence.json"
        comparison_path = ROOT / "reports/v1-evidence/benchmark-150-comparison.json"
        benchmark_150_path = ROOT / "reports/v1-evidence/benchmark-150.json"
        evidence = None
        evidence_hash = None
        benchmark150 = None
        if path.is_file():
            import json

            evidence = json.loads(path.read_text(encoding="utf-8"))
            evidence_hash = sha256_file(path)
        if comparison_path.is_file():
            import json

            benchmark150 = json.loads(comparison_path.read_text(encoding="utf-8"))
            if benchmark_150_path.is_file():
                benchmark150["videos"] = json.loads(benchmark_150_path.read_text(encoding="utf-8")).get("test_videos")
        return render_template(
            "evidence.html", evidence=evidence, evidence_hash=evidence_hash, benchmark150=benchmark150
        )

    @app.get("/evidence/v1-ff-c23-100")
    def evidence_download():
        path = ROOT / "reports/v1-evidence/evidence.json"
        if not path.is_file():
            raise AnalysisError("The benchmark evidence bundle is not installed.", 404, "evidence_unavailable")
        return app.response_class(path.read_bytes(), mimetype="application/json", headers={"Content-Disposition": 'attachment; filename="authenfy-v1-evidence.json"'})

    @app.get("/evidence/v1-ff-c23-100/benchmark.json")
    def benchmark_download():
        path = ROOT / "reports/v1-evidence/benchmark.json"
        if not path.is_file():
            raise AnalysisError("The benchmark evidence bundle is not installed.", 404, "evidence_unavailable")
        return app.response_class(path.read_bytes(), mimetype="application/json", headers={"Content-Disposition": 'attachment; filename="authenfy-v1-benchmark.json"'})

    @app.get("/evidence/v1-ff-c23-150/comparison.json")
    def benchmark_150_comparison_download():
        path = ROOT / "reports/v1-evidence/benchmark-150-comparison.json"
        if not path.is_file():
            raise AnalysisError("The 150-video benchmark report is not installed.", 404, "evidence_unavailable")
        return app.response_class(
            path.read_bytes(),
            mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="authenfy-v1-benchmark-150-comparison.json"'},
        )

    @app.get("/evidence/v1-ff-c23-150/benchmark.json")
    def benchmark_150_download():
        path = ROOT / "reports/v1-evidence/benchmark-150.json"
        if not path.is_file():
            raise AnalysisError("The 150-video benchmark report is not installed.", 404, "evidence_unavailable")
        return app.response_class(
            path.read_bytes(),
            mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="authenfy-v1-benchmark-150.json"'},
        )

    @app.get("/evidence/v1-ff-c23-150/manifest.json")
    def benchmark_150_manifest_download():
        path = ROOT / "reports/v1-evidence/evidence-150.json"
        if not path.is_file():
            raise AnalysisError("The 150-video evidence manifest is not installed.", 404, "evidence_unavailable")
        return app.response_class(
            path.read_bytes(),
            mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="authenfy-v1-evidence-150.json"'},
        )

    @app.get("/evidence/v1-ff-c23-150/manifest.sha256")
    def benchmark_150_manifest_hash_download():
        path = ROOT / "reports/v1-evidence/evidence-150.json.sha256"
        if not path.is_file():
            raise AnalysisError("The 150-video evidence hash is not installed.", 404, "evidence_unavailable")
        return app.response_class(
            path.read_bytes(),
            mimetype="text/plain",
            headers={"Content-Disposition": 'attachment; filename="authenfy-v1-evidence-150.json.sha256"'},
        )

    @app.get("/evidence/v1-ff-c23-100/bundle.zip")
    def evidence_bundle_download():
        path = ROOT / "reports/v1-evidence/evidence-bundle.zip"
        if not path.is_file():
            raise AnalysisError("The benchmark evidence bundle is not installed.", 404, "evidence_unavailable")
        return app.response_class(path.read_bytes(), mimetype="application/zip", headers={"Content-Disposition": 'attachment; filename="authenfy-v1-evidence.zip"'})

    def evidence_commitments(kind, subject_id):
        if kind == "receipt" and re.fullmatch(r"[0-9a-f]{32}", subject_id):
            receipt = get_receipt(subject_id)
            return receipt["file_sha256"], receipt["receipt_commitment"]
        if kind == "evidence" and subject_id == "v1-ff-c23-100":
            root = ROOT / "reports/v1-evidence"
            manifest_path, benchmark_path = root / "evidence.json", root / "benchmark.json"
            if manifest_path.is_file() and benchmark_path.is_file():
                return sha256_file(benchmark_path), sha256_file(manifest_path)
            raise AnalysisError("The benchmark evidence bundle is not installed.", 404, "evidence_unavailable")
        raise AnalysisError("Anchor target not found.", 404, "anchor_target_not_found")

    def rpc(method, params):
        try:
            response = requests.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=6)
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            raise AnalysisError("Could not verify the transaction with the configured blockchain RPC.", 502, "rpc_unavailable") from error
        if result.get("error") or "result" not in result:
            raise AnalysisError("The blockchain RPC could not verify this transaction.", 502, "rpc_error")
        return result["result"]

    @app.post("/anchors/confirm")
    def confirm_anchor():
        if not blockchain_enabled:
            raise AnalysisError("Blockchain anchoring is not configured on this deployment.", 503, "blockchain_unavailable")
        if request.content_length and request.content_length > 4096:
            raise AnalysisError("The confirmation request is too large.", 413, "request_too_large")
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            raise AnalysisError("A JSON object is required.", 400, "invalid_anchor_target")
        kind, subject_id, tx_hash = body.get("kind"), body.get("subject_id"), body.get("transaction_hash", "")
        if kind not in {"receipt", "evidence"} or not isinstance(subject_id, str):
            raise AnalysisError("Choose a valid receipt or evidence item.", 400, "invalid_anchor_target")
        if not isinstance(tx_hash, str) or not re.fullmatch(r"0x[a-fA-F0-9]{64}", tx_hash):
            raise AnalysisError("A transaction hash is required.", 400, "invalid_transaction_hash")
        file_hash, commitment = evidence_commitments(kind, subject_id)
        expected_input = "0xcda1c3b9" + file_hash.lower().removeprefix("0x") + commitment.lower().removeprefix("0x")
        tx = rpc("eth_getTransactionByHash", [tx_hash])
        mined = rpc("eth_getTransactionReceipt", [tx_hash])
        if not tx or not mined:
            raise AnalysisError("The transaction is not confirmed yet.", 409, "transaction_pending")
        if int(rpc("eth_chainId", []), 16) != chain_id:
            raise AnalysisError("The configured RPC is connected to a different chain ID.", 502, "chain_mismatch")
        block_number = int(mined["blockNumber"], 16)
        latest_block = int(rpc("eth_blockNumber", []), 16)
        confirmation_count = latest_block - block_number + 1
        if confirmation_count < confirmations_required:
            raise AnalysisError(
                f"The transaction has {confirmation_count} of {confirmations_required} required confirmations.",
                409,
                "transaction_pending",
            )
        if (
            str(tx.get("to", "")).lower() != registry.lower()
            or str(tx.get("input", tx.get("data", ""))).lower() != expected_input
            or str(mined.get("status", "")).lower() != "0x1"
            or str(mined.get("to", "")).lower() != registry.lower()
        ):
            raise AnalysisError("The transaction does not match this file and receipt commitment.", 409, "anchor_mismatch")
        publisher = rpc("eth_call", [{"to": registry, "data": "0x8da5cb5b"}, mined.get("blockNumber", "latest")])
        if not isinstance(publisher, str) or len(publisher) != 66 or str(tx.get("from", "")).lower() != ("0x" + publisher[-40:]).lower():
            raise AnalysisError("The transaction sender is not the registry publisher.", 409, "publisher_mismatch")
        event_topic = "0xe7e560a3f83a100a96b69bee4a2060f4ace35f3c25fa188088ece2dc435f6853"
        expected_file_topic = "0x" + file_hash.lower().removeprefix("0x").rjust(64, "0")
        expected_commitment_topic = "0x" + commitment.lower().removeprefix("0x").rjust(64, "0")
        found = any(
            str(log.get("address", "")).lower() == registry.lower()
            and len(log.get("topics", [])) == 3
            and [str(x).lower() for x in log["topics"]] == [event_topic, expected_file_topic, expected_commitment_topic]
            for log in mined.get("logs", [])
        )
        if not found:
            raise AnalysisError("The confirmed transaction has no matching registry event.", 409, "anchor_event_missing")
        try:
            anchor = receipts.put_anchor(kind, subject_id, tx_hash.lower(), registry, chain_id, block_number)
        except (ValueError, KeyError) as error:
            raise AnalysisError(str(error), 409, "anchor_conflict") from error
        return jsonify(verified=True, kind=kind, subject_id=subject_id, blockchain_anchor=anchor)

    @app.get("/anchors/<kind>/<subject_id>")
    def get_anchor(kind, subject_id):
        if kind not in {"receipt", "evidence"}:
            raise AnalysisError("Anchor target not found.", 404, "anchor_target_not_found")
        evidence_commitments(kind, subject_id)
        return jsonify(blockchain_anchor=receipts.get_anchor(kind, subject_id))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
