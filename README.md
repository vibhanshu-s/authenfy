# Authenfy

The project's clarified EfficientNet-B0 + DCT + pHash -> MLP implementation and controlled T4 experiment are in [docs/EFFICIENTNET_EXPERIMENT.md](docs/EFFICIENTNET_EXPERIMENT.md) and [notebooks/Authenfy_EfficientNet_Fusion_T4.ipynb](notebooks/Authenfy_EfficientNet_Fusion_T4.ipynb). Training/evaluation results will be recorded before promotion; the descriptions below document the earlier engine. Select `AUTHENFY_ENGINE=fusion` to serve an installed validated fusion model.

A local media inspection app with separate generated-image and facial-manipulation screening, sampled video evidence, and signed exact-file receipts. Flask + native HTML/CSS/JavaScript; ONNX inference runs on the server. Model development and numerical checks use a Colab T4.

**An analysis score is not proof that media is real or fake.** The original repository could not start because its detector was missing and its root template did not exist. This rebuild supplies a functioning baseline, bounded decoding, honest coverage states and reproducible verification.

## Run locally (Python 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\waitress-serve.exe --host=127.0.0.1 --port=5000 app:app
```

On Linux/macOS use `.venv/bin/python` and `.venv/bin/waitress-serve`. Open http://127.0.0.1:5000. The app remains usable without a facial specialist, but missing coverage cannot receive a low-signal verdict. To install the specialist, use the T4 notebook below, then:

```powershell
.\.venv\Scripts\python.exe scripts/install_specialist.py C:/path/to/face-specialist.onnx
```

The installer verifies this ONNX export against the committed validation report and installs the pinned YuNet face detector. For your own trusted reproduction, pass `--validation path/to/authenfy-specialist-validation.json`; build hashes can differ across exporter environments. Keep the model and its matching validation report together. Large model binaries are deliberately excluded from Git. The generic model installer downloads 87 MB from its immutable Hugging Face revision and checks SHA-256. The specialist export is about 1.2 GB. At least 4 GB free RAM is recommended for local inference; run one application process because its admission semaphore is process-local.

## Colab T4

Open `notebooks/Authenfy_T4_Validation.ipynb` in Colab and select **Runtime > Change runtime type > T4 GPU**. Run the cells in order. They:

1. Assert that the device is a Tesla T4; validate pinned generated-image GPU inference against CPU ONNX and explicit PIL preprocessing.
2. Audit the same 32 labeled example crops without fitting weights or choosing thresholds.
3. Load the checksum-verified published specialist tensor archive, reconstruct it using standard Transformers layers and verify it against the source graph.
4. Export a local CPU ONNX model, verify numerical parity, and download the report and model.

This is inference, export and regression work, not model training. The 32 crops come from 8 correlated video/method groups and include FF++ examples that may overlap specialist training. They are **not** an independent benchmark or a production accuracy estimate. The original generic model missed every manipulated crop in that audit; see `reports/face-audit-baseline.json`. The exported specialist detected 19/20 manipulated crops with 4/12 false positives at 0.5 on the supplied prealigned inputs. The actual YuNet detection/alignment path detected 19/20 with 5/12 false positives and zero missed faces on the same 32 examples; see `reports/local-face-alignment-audit.json`. These small, correlated results must not be presented as general accuracy.

## API

| Endpoint | Contract |
| --- | --- |
| `GET /health` | Model presence/integrity and specialist availability; 503 on missing/corrupt generic weights |
| `POST /analyze` | Multipart `media`; returns scores, coverage, frames, model hashes, signed receipt and receipt commitment |
| `GET /receipts/<receipt_id>` | Download a verified local JSON receipt; the random ID is its access token |
| `POST /verify` | Multipart `receipt_id` plus `media`; checks signature integrity and exact SHA-256 bytes |

Accepted static images: JPEG, PNG and WebP. Accepted video containers: MP4, MOV, AVI, WebM and MKV, subject to the installed codec's ability to decode. Maximum upload: 64 MiB. Maximum clip: 120 seconds. Maximum image/frame size: 20 MP, minimum dimensions 32 pixels, maximum aspect ratio 8:1. Up to 16 frames are sampled uniformly, including both endpoints. Decode failures abort rather than silently reducing coverage.

The server isolates decoding and inference in a killable process with a 90-second deadline. It accepts one analysis at a time and returns 429 when busy. Uploads use random temporary paths and are deleted in `finally` blocks. A server crash can leave an orphan; operators should clean the upload directory while the server is stopped.

## Models and decision policy

- **Generated images:** corrected Community Forensics ViT-Small FP32. Aspect-preserving shortest-edge resize 440, PIL bicubic, center crop 384, exact CLIP normalization, sigmoid output. Revision and SHA-256 are pinned in `authenfy/model.py`.
- **Facial manipulation:** pinned Yermakov/Cech/Matas CLIP-L specialist. YuNet finds up to four faces and five landmarks; a least-squares similarity transform approximates the specialist's DeepfakeBench alignment. This approximation still needs independent validation on arbitrary raw images and videos.
- **Aggregation:** take the maximum of the two component scores for each sampled frame, then the temporal mean. A mean >= 0.8 yields synthetic signals; a maximum <= 0.2 yields low signals only with complete detected-face coverage. Intermediate scores, disagreement > 0.5, no-face cases and incomplete low-score coverage yield inconclusive. These bands are policy defaults, not calibrated probabilities.
- Only byte-identical RGB frames reuse inference. pHash is descriptive evidence and never an authenticity key.
- All results require human review. Component scores and coverage are returned separately; no verdict is labeled “authentic.”

## Receipts and Ethereum

Local receipts use HMAC-SHA256 over canonical JSON, signed with a server-held random 32-byte key. SQLite stores receipts, not uploads or filenames. `receipt_commitment` is SHA-256 of the signed payload before signature/commitment fields are attached. Protect and back up `instance/receipts.key` with the database. Key loss prevents checking old receipts. HMAC is server-verifiable, not independent public attestation.

`contracts/AnalysisRegistry.sol` implements an immutable publisher-restricted Ethereum registry for `(file SHA-256, receipt commitment)` pairs. It rejects unauthorized writes, zero commitments and overwrites, while allowing multiple distinct analyses of the same file. It has been exercised on an in-process Ganache chain. **The running application does not submit public-chain transactions and reports `blockchain_anchored: false`.** Public chain deployment, issuer management, RPC finality checks and UI integration remain separate work. Hash registration cannot prove a depicted event is true.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check app.py authenfy tests scripts
npm ci
npm run test:contract
# Start the local server before the browser test; Chrome must be installed.
npm run test:browser
```

To reproduce the local alignment audit after installing the specialist, run `python scripts/fetch_face_audit.py` and `python scripts/local_face_audit.py private/face-audit/inputs.json`. These fetch checksum-verified public research examples into the ignored private directory.

Python tests cover routing, malformed/oversized uploads, missing/corrupt weights, tampered receipts, concurrent requests, temporal sampling, duplicate/pHash collisions and coverage-aware decisions. Browser checks cover actual local inference, receipt verification, filename injection, error recovery and mobile overflow. CI runs Python checks on Windows and Ubuntu plus registry tests. CI evidence must be checked on GitHub after a push; local checks alone do not establish a green remote run.

## Scope and remaining gaps

See `docs/GAPS.md` and the machine-readable receipts in `reports/`. Audio detection, audiovisual synchronization, rPPG, Grad-CAM localization, user accounts, production deployment and representative independent calibration are not implemented. The app is bound to loopback by default; do not expose it publicly without authentication, process/container resource limits, proxy limits and a retention policy.

## Sources and licenses

- [Community Forensics model](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (MIT per model card) and [original research](https://github.com/JeongsooP/Community-Forensics).
- [Facial deepfake model](https://huggingface.co/yermandy/deepfake-detection) (MIT per model card) and [official implementation](https://github.com/yermandy/deepfake-detection).
- [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (see its directory license).
- [DeepfakeBench preprocessing](https://github.com/SCLBD/DeepfakeBench/blob/fb6171a8e1db2ae0f017d9f3a12be31fd9e0a3fb/preprocessing/preprocess.py) defines the alignment reference. YuNet landmark locations differ from dlib's; the substitution is explicitly disclosed.

Example face images are fetched only for the small research audit and are not redistributed in this repository or used to train a model.
