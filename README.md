# Authenfy v1

Facial deepfake screening with **fine-tuned EfficientNet-B0 + DCT + pHash features fed into an MLP**. A paired Colab T4 experiment compares the EfficientNet all-frame baseline, a deep-only MLP control, full-frame fusion and fusion with guarded pHash reuse.

The T4 runtime reset during preparation and its temporary run files were lost. Dataset restoration is in progress; fine-tuning and held-out accuracy/speed results remain pending. Persistent Drive recovery is implemented, with the active mount awaiting Google's authorization. Passing code tests or a small pipeline smoke run does not establish model performance. AI-video development is deferred.

## Architecture

EfficientNet-B0 starts from ImageNet-1K weights and is fine-tuned for facial manipulation. Its best validation checkpoint supplies 1,280 pooled features. A shared 224-pixel RGB crop supplies 63 signed-log DCT coefficients and 63 pHash bits. The concatenated 1,406 values pass through train-only normalization and a 256/64/1 MLP. The backbone is frozen while fitting the fusion and deep-only heads.

Every evaluation frame is decoded. The pHash arm can reuse an earlier score within two hash bits, subject to RGB/DCT residual limits and a refresh every 12 frames. Reused frames retain their temporal weight. pHash similarity does not prove authenticity; reuse can miss brief or subtle edits.

## Train and measure on a T4

Open [Authenfy_EfficientNet_Fusion_T4.ipynb](notebooks/Authenfy_EfficientNet_Fusion_T4.ipynb) in Colab, select a T4 and run the cells. The [experiment protocol](docs/EFFICIENTNET_EXPERIMENT.md) documents features, splits, training settings, timing and limitations.

The experiment uses Kaggle [FF++ C23](https://www.kaggle.com/datasets/xdxd003/ff-c23), version 1: all 4,320 training and 840 validation videos, plus a fixed 240-video test subset. Both manipulation directions and their original sources stay in the same official source-pair split. This does not prove identity-disjoint generalization. Test data never selects checkpoints or the fixed 0.5 classification threshold.

The test subset contains 40 real and 200 manipulated videos; predicting every video as fake gives 83.33% raw accuracy. Balanced accuracy, AUROC, class-specific errors and a constant-prior control accompany raw accuracy.

The downloaded run bundle contains:

- `training.json`: epoch histories, full validation metrics, hyperparameters, hardware, software versions, seed, durations and checkpoint hashes.
- `benchmark.json` / `benchmark.md`: all four arms' video/frame/method metrics, calibration, confusion matrices, latency distributions, throughput, frame reuse and measured resources.
- `video-predictions.csv`, `frame-predictions.npz`, `timings.csv`: individual scores and all three timing repeats for every clip and arm.
- `additional-metrics.json`: source-group bootstrap confidence intervals, paired accuracy/speed comparisons, descriptive ROC operating points and the constant-prior control.
- `figures/`: learning curves, ROC/PR/calibration, confusion matrices and accuracy/speed/reuse plots.
- Three trained `.pt` checkpoints, `efficientnet-fusion.onnx`, `model-card.json`, manifest, dataset provenance and execution logs.

## Run the trained application

After downloading and extracting a completed measured run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/install_fusion.py C:/path/to/extracted/run
$env:AUTHENFY_ENGINE = 'fusion'
.\.venv\Scripts\waitress-serve.exe --host=127.0.0.1 --port=5000 app:app
```

Open http://127.0.0.1:5000. The installer checks the matching ONNX, training, benchmark and model-card hashes. Selecting `fusion` reports missing or corrupt weights explicitly. Model binaries, datasets and local receipt keys are excluded from Git.

The application runs CPU ONNX inference, independently from the T4 speed measurements. It inspects every decoded video frame with batches of eight and discloses approximate score reuse. Limits are 64 MiB, 120 seconds, 3,600 frames and 20 MP per frame; the isolated analyzer has a 90-second deadline. Uploads are removed after the request, and one analysis is admitted at a time.

Scores are uncalibrated screening signals. Every result requires review. The service uses full-frame center crops; it does not localize faces or analyze audio, lip sync or rPPG. Fully AI-generated video detection is not enabled. A reused-frame video cannot receive a low-signal verdict.

## API and receipts

| Endpoint | Purpose |
|---|---|
| `GET /health` | Engine identity and model integrity; 503 when unavailable |
| `POST /analyze` | Multipart `media`; scores, coverage, frames, hashes and signed receipt |
| `GET /receipts/<receipt_id>` | Download a verified local JSON receipt |
| `POST /verify` | Multipart `receipt_id` and `media`; exact-file SHA-256 and signature check |

Receipts use server-held HMAC-SHA256 keys and SQLite. Protect and back up the key with the database. These are local server-verifiable records. The tested Ethereum registry contract is included, but the application reports `blockchain_anchored: false` and does not submit public-chain transactions.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check app.py authenfy tests scripts training
npm ci
npm run test:contract
npm run test:browser
```

CI checks Windows/Linux application behavior, feature and source-split integrity, CPU ONNX parity and Ethereum contract behavior. Browser checks require a running local server and Chrome. Test success is separate from trained-model evaluation.

The earlier pretrained ViT/CLIP engine remains explicitly selectable with `AUTHENFY_ENGINE=legacy`; its setup and small correlated audit are archived in [LEGACY_ENGINE.md](docs/LEGACY_ENGINE.md). Those results are not the fine-tuned v1 benchmark.
