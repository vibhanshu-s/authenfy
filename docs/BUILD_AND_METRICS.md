# Authenfy v1: build, training, and measured results

## What this project does

Authenfy is an educational and research prototype for screening images and videos for facial-manipulation signals. V1 uses a fine-tuned EfficientNet-B0 feature extractor with frequency-domain features and an MLP classifier. It is not a forensic certification system. Its scores are uncalibrated signals for human review.

V1 was trained and evaluated for facial manipulation in FaceForensics++ C23. It does not detect fully AI-generated video as a separately validated task. The [FaceForensics++ terms of use](https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf) limit use to non-commercial research and educational purposes. The model repository is private; do not redistribute the dataset or weights without confirming the applicable rights.

## Architecture

### Training and compared models

The experiment compares four inference arms on the same held-out videos:

1. **EfficientNet baseline:** Fine-tuned EfficientNet-B0 with a binary classification head; every decoded frame is passed through the network.
2. **Deep-feature MLP:** The fine-tuned EfficientNet-B0 backbone supplies 1,280 pooled features to an MLP on every frame.
3. **Feature-fusion MLP:** The 1,280 EfficientNet features are concatenated with 63 signed-log low-frequency DCT coefficients and 63 pHash bits, then classified by a 256/64/1 MLP.
4. **Fusion with pHash reuse:** The same fusion head reuses a preceding score for frames that pass conservative pHash, RGB, and DCT similarity checks. It refreshes at least every 12 frames. Every decoded frame still contributes to video aggregation.

The control and fusion MLPs share the frozen fine-tuned EfficientNet backbone. Feature normalization is fit on training data only. The application and training code share crop, normalization, DCT, and pHash implementations. The exported CPU ONNX model is checked against the PyTorch implementation for preprocessing parity.

### Application flow

`app.py` accepts image/video uploads and delegates analysis to `authenfy/fusion_analysis.py`. `authenfy/media.py` validates and decodes bounded media. `authenfy/fusion_features.py` prepares normalized image tensors and handcrafted features. `authenfy/fusion_model.py` loads the versioned ONNX artifact and returns frame scores. Video reports preserve timestamps, frame-level scores, reuse decisions, model identity, and known limitations. Receipt signing, local storage, and optional blockchain timestamp verification are implemented separately from model inference.

Uploads are limited to 64 MiB, 120 seconds, 3,600 frames, and 20 megapixels per frame. The local service uses CPU ONNX inference; benchmark GPU speed is not the expected speed of a deployed CPU instance.

## Data, split, and training

The dataset is Kaggle [FF++ C23](https://www.kaggle.com/datasets/xdxd003/ff-c23), dataset version 1. Preparation retains original/manipulated source pairs in the same split to reduce source leakage. The manifest records file hashes, labels, manipulation methods, groups, and source IDs. This is a source-pair split, not an identity-disjoint or original-YouTube-source-disjoint split.

The selected evaluation pool contained 240 videos. The final benchmark used the recorded 150-video held-out selection (26 real and 124 manipulated), covering 13 source groups, and 74,871 decoded frames. Each arm received three timing repeats. Accuracy and AUROC are video-level metrics; per-frame results are descriptive because frames from one video are correlated.

The baseline was trained for eight epochs. MLP heads were trained for 20 epochs each. Training uses eight sampled frames per video, class-weighted binary cross entropy, a fixed seed, and validation-video log loss for checkpoint selection. Test classification uses the fixed 0.5 threshold; no threshold was tuned on test data. The exact run metadata, code/runtime identity, split hashes, and checkpoints are recorded in the training artifacts.

The Colab notebook is `notebooks/Authenfy_EfficientNet_Fusion_T4.ipynb`. Dataset preparation is in `scripts/prepare_ff_c23.py`; training and recovery are in `training/`; the benchmark is `training/benchmark.py`; ONNX export is `training/export.py`. A completed local run can be reproduced with the same pinned dataset manifest, code revision, settings, and runtime:

Create a separate training environment with `pip install -r requirements-training.txt`, using the PyTorch CUDA wheel channel appropriate to the host. The recorded run used PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128. The Colab notebook uses its preinstalled CUDA build and installs the pinned supporting packages listed in that file.

```text
python scripts/prepare_ff_c23.py data/ff-c23 --test-pairs 20
python -m training.train data/ff-c23/manifest.csv runs/ff-c23
python -m training.benchmark data/ff-c23/manifest.csv runs/ff-c23 --limit-videos 150
python -m training.export data/ff-c23/manifest.csv runs/ff-c23
python -m training.plots runs/ff-c23
python scripts/summarize_fusion.py runs/ff-c23
```

For a Colab T4 run, add `--require-t4` to training and benchmark commands. To resume across a Colab reset, mount Drive and pass the same `--recovery-dir` to the training or campaign command. Recovery checks hashes and runtime identity before restoring. The provided notebook is the recommended end-to-end entry point.

## Recorded 150-video benchmark

These are the measured results in `reports/v1-evidence/benchmark-150.json` and `benchmark-150-comparison.json`. Timings were collected locally on an NVIDIA RTX 4060 Laptop GPU. FPS is aggregated over measured frames; each timing arm had three repeats. Speed percentage is relative to the all-frame EfficientNet baseline.

| Inference arm | Accuracy | AUROC | F1 | Mean seconds/video | FPS | Speed vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet baseline, all frames | 88.67% | 0.9318 | 0.9300 | 6.0624 | 82.33 | Reference |
| Deep-feature MLP, all frames | 92.67% | 0.9429 | 0.9572 | 6.0736 | 82.18 | -0.18% |
| Feature fusion, all frames | 92.67% | 0.9417 | 0.9572 | 6.2511 | 79.85 | -3.02% |
| Feature fusion with pHash reuse | 92.67% | 0.9432 | 0.9572 | 5.4183 | 92.12 | +11.89% |

The fusion+pHash arm improved observed accuracy by 4 percentage points and was 11.89% faster than the baseline in this run. Its paired source-group bootstrap interval for the accuracy difference versus baseline was -2.67 to +12.50 percentage points and included zero. This run therefore does not establish a statistically conclusive accuracy improvement. The all-frame fusion arm was 3.02% slower than the baseline; the deep-feature MLP was 0.18% slower. The pHash arm reused approximately 63.51% of frame scores.

The earlier 100-video evidence run is retained separately in `benchmark.json` and `benchmark.md`; it is not overwritten by the later 150-video run. `benchmark-150.md` is the concise summary, and the JSON files retain predictions, per-method metrics, timing repeats, and comparison details. Do not combine the two runs as if they were one test set.

## Metric definitions and evidence

- **Accuracy:** Correct video classifications divided by the number of videos at the fixed 0.5 threshold.
- **AUROC:** Area under the video-level receiver operating characteristic curve; threshold independent.
- **F1:** Harmonic mean of video-level precision and recall at the fixed threshold.
- **FPS:** Total measured frames divided by total timed inference wall time for that arm.
- **Speed vs baseline:** `(arm FPS / baseline FPS - 1) × 100%`; a positive value means more frames per second.
- **Mean seconds/video:** Average timed wall time for a held-out video, including decode, preprocessing, and inference as recorded by the benchmark.

The raw run also records confusion matrices, per-class metrics, log loss, calibration summaries, method-specific metrics, group-bootstrap comparisons, per-video predictions, per-frame predictions, per-video timings, resource observations, and training metadata. Full source and hashes are kept in `reports/v1-evidence`. Validate the 100-video signed evidence bundle with:

```powershell
.\.venv\Scripts\python.exe scripts\record_benchmark_evidence.py --verify
```

The later benchmark's JSON and comparison files are preserved alongside the bundle. SHA-256 hashes establish file identity after recording; they do not prove that the dataset, labels, or measurements were independently audited. Blockchain anchoring can timestamp those hashes, but cannot establish model correctness or media authenticity.

## Model files and installation

The trained model weights are hosted in the private Hugging Face repository [`vibhxnshu/authenfy-efficientnet-fusion-v1`](https://huggingface.co/vibhxnshu/authenfy-efficientnet-fusion-v1). Weight files are excluded from Git. Download the matching release bundle from that repository and install it locally:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\install_fusion.py C:/path/to/extracted/run
.\.venv\Scripts\waitress-serve.exe --host=127.0.0.1 --port=5000 app:app
```

Keep the Hugging Face access token in a local environment variable or credential store. Never put tokens, `.env` files, model weights, datasets, virtual environments, or `node_modules` in Git. The repository `.gitignore` excludes those paths and file types.

## Known limits and next work

- The measured task is FF++ facial manipulation, not fully AI-generated video detection.
- The held-out test has 26 real and 124 manipulated videos from 13 source groups; results are preliminary and may not transfer to other datasets, identities, compression levels, or generators.
- Group bootstrap uncertainty includes no accuracy improvement for fusion+pHash versus baseline.
- The observed speed is from the RTX 4060 Laptop GPU and benchmark harness, not a hosted CPU service.
- pHash reuse is approximate and can miss brief or subtle changes. It does not identify an authentic source or certify media.
- Scores are uncalibrated. A human must review consequential decisions.
- V2 AI-generated video detection requires a separate labeled dataset, source/prompt-disjoint evaluation, and independent metrics. It is intentionally not part of this release.
