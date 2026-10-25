# Authenfy v1

Authenfy screens images and videos for facial-manipulation signals with fine-tuned EfficientNet-B0 features, DCT/pHash features and an MLP. Its versioned benchmark compares an all-frame EfficientNet baseline, a deep-feature MLP, full-frame fusion and fusion with guarded pHash frame reuse. AI-generated video detection remains out of scope for this version.

## Build and results

The implementation, training workflow, benchmark protocol, detailed metric definitions, limitations, and current measured results are documented in [docs/BUILD_AND_METRICS.md](docs/BUILD_AND_METRICS.md). The experiment uses Kaggle [FF++ C23](https://www.kaggle.com/datasets/xdxd003/ff-c23). Its source-pair test is preliminary evidence for facial-manipulation screening; it does not establish AI-generated-video detection or generalization to unseen identities.

The complete evidence bundle is in [reports/v1-evidence](reports/v1-evidence) and downloadable from `/evidence`. It includes benchmark JSON, training records, model card, per-video and frame scores, timing repeats and evaluation source files, with SHA-256 indexes. Verify it with:

```powershell
.\.venv\Scripts\python.exe scripts\record_benchmark_evidence.py --verify
.\.venv\Scripts\python.exe scripts\record_benchmark_evidence.py --verify-150
```

The evidence page at `/evidence` summarizes the run and offers wallet anchoring. The bundle can also be checked offline. Anchoring records a benchmark file hash and evidence-manifest hash; it cannot prove the dataset or content is truthful.

## Run locally

Python 3.12 is supported. Install the trained bundle first; model artifacts are not stored in Git:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\install_fusion.py C:/path/to/extracted/run
.\.venv\Scripts\waitress-serve.exe --host=127.0.0.1 --port=5000 app:app
```

The installer validates the ONNX model, training metadata, benchmark and model-card hashes. The fusion service uses CPU ONNX independently of the GPU benchmark. It decodes every video frame, batches inference, and can reuse scores for visually similar frames under conservative RGB/DCT checks. Uploads are limited to 64 MiB, 120 seconds, 3,600 frames and 20 MP per frame. One analysis runs at a time. Scores are uncalibrated screening signals and every case needs human review.

## Blockchain timestamping

`contracts/AnalysisRegistry.sol` stores immutable `(file SHA-256, commitment SHA-256)` pairs. The app can anchor local analysis receipts and the benchmark evidence manifest using an injected EIP-1193 browser wallet. The contract publisher is fixed at deployment; the connected wallet must be that publisher. The private key stays in the wallet. The server uses a read-only RPC endpoint to verify chain ID, sender, call data, transaction status and the emitted event before marking an item anchored.

Install the development dependencies, then deploy from a trusted shell to a chain you control. Protect the deployer key and never commit it:

```powershell
npm ci
$env:AUTHENFY_RPC_URL = 'https://your-rpc-endpoint'
$env:AUTHENFY_DEPLOYER_PRIVATE_KEY = '0x...'
npm run deploy:contract
```

Copy the returned registry address and chain ID into the Authenfy deployment configuration. Set `AUTHENFY_RPC_URL` to a read-only RPC endpoint for that same chain, then restart the service. In the browser, add the same chain to the wallet and connect the registry publisher account. The registry contract tests run on a disposable in-process Ganache chain; no public deployment is included in this repository. Public transactions require gas and an RPC/wallet you provide.

## Container deployment

Install the model bundle at `artifacts/efficientnet-v1` before building. Then:

```powershell
Copy-Item deploy/.env.example .env
# Optional: fill registry address, chain ID and read-only RPC URL in .env.
docker compose up --build -d
```

Open `http://localhost:5000`. Compose runs as a non-root user, mounts model files read-only, makes the container root filesystem read-only, drops Linux capabilities and persists the SQLite database and HMAC signing key in a named volume. Back up that volume securely; losing the HMAC key invalidates local receipt signatures. Put a public deployment behind HTTPS and a proxy with appropriate upload/body and request timeouts. The built-in analyzer caps still apply.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model availability and blockchain configuration status |
| `POST /analyze` | Analyze multipart `media`; return scores, evidence and signed receipt |
| `GET /receipts/<receipt_id>` | Download receipt and verified anchor metadata, if recorded |
| `POST /verify` | Verify a receipt and exact uploaded file bytes |
| `GET /evidence` | Inspect and anchor the benchmark evidence |
| `GET /evidence/v1-ff-c23-100` | Download the SHA-256 indexed evidence manifest |
| `GET /evidence/v1-ff-c23-100/bundle.zip` | Download the complete proof bundle and hash sidecars |
| `GET /evidence/v1-ff-c23-150/benchmark.json` | Download the full later benchmark record |
| `GET /evidence/v1-ff-c23-150/comparison.json` | Download the concise comparison and paired uncertainty summary |
| `GET /evidence/v1-ff-c23-150/manifest.json` | Download the 150-video file-hash manifest |
| `GET /evidence/v1-ff-c23-150/manifest.sha256` | Download its SHA-256 sidecar |
| `POST /anchors/confirm` | Verify a mined wallet transaction against the configured registry |
| `GET /anchors/<kind>/<subject_id>` | Retrieve the server-verified chain anchor |

Receipts are HMAC-SHA256 signed and stored in SQLite. Keep the database and signing key together. Blockchain timestamping proves that a particular hash pair was recorded at a chain position; it does not prove a media file is authentic or an analysis is correct.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check app.py authenfy tests scripts training
npm ci
npm run test:contract
npm run test:browser
```

Browser tests need Chrome and a running local app. Tests validate code behavior; benchmark results are separate model evidence.
