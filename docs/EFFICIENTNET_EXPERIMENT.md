# EfficientNet feature-fusion experiment

This supersedes the presentation's Xception/Swin design with the user's clarified
project architecture. It does not reuse the previous app's ViT/CLIP detector.

1. Start EfficientNet-B0 from torchvision's ImageNet-1K V1 weights. Replace its
   classifier with a binary facial-manipulation head and fine-tune the backbone.
2. Select the baseline checkpoint by validation video log loss. Freeze that same
   backbone for both MLP experiments, so changing the features is the comparison.
3. Concatenate 1,280 pooled EfficientNet features, 63 signed-log low-frequency DCT
   AC coefficients and 63 pHash bits. Fit a 256/64/1 MLP with train-only feature
   normalization. The control MLP has the same hidden layers with 1,280 inputs.
4. Compare `baseline_all`, `deep_mlp_all`, `fusion_all` and `fusion_phash` on the
   exact same held-out complete videos. Every frame is decoded. The pHash arm
   reuses an anchor's score only within two hash bits, RGB/DCT residual guards,
   and a forced refresh every 12 frames. All decoded frames retain temporal weight.

The feature branches all describe visual content. There is no audio branch.
pHash is a similarity signal, not proof of authenticity or a unique identifier.
Approximate reuse is an experiment whose missed edits must be measured.

## Data and separation

Dataset: https://www.kaggle.com/datasets/xdxd003/ff-c23, version 1.
The preparation script uses pinned official FF++ source pairs, including both
directions of a manipulation and the corresponding originals in the same split.
It includes original, Deepfakes, Face2Face, FaceSwap, NeuralTextures and FaceShifter.
The separate Google DeepFakeDetection actor collection is excluded from this
experiment because its identifiers require a separate split mapping.

The manifest records file SHA-256, label, method, group and both source IDs.
Validation rejects cross-split source IDs, repeated paths and identical file bytes.
These source-pair splits do not prove identity or original YouTube-source separation.
The manifest and split hashes are preserved with every run. No random frame split.

`--train-pairs 0 --val-pairs 0 --test-pairs 20` uses all 360 training pairs and all
70 validation pairs, with a fixed 20-pair held-out test subset for the expensive
all-frame comparison: 4,320 training, 840 validation and 240 test videos. This is
not the entire Kaggle collection or the complete official test set.

## Reproduction on a Colab T4

Use the preinstalled matching CUDA torch/torchvision pair, then install the small
dependencies listed by the notebook. Exact versions and device are recorded.

```text
python scripts/prepare_ff_c23.py data/ff-c23 --test-pairs 20
python -m training.train data/ff-c23/manifest.csv runs/ff-c23 --require-t4
python -m training.benchmark data/ff-c23/manifest.csv runs/ff-c23 --require-t4
python -m training.export data/ff-c23/manifest.csv runs/ff-c23
python -m training.plots runs/ff-c23
python scripts/summarize_fusion.py runs/ff-c23
```

Training samples eight frames per video, uses class-weighted binary cross entropy,
and chooses checkpoints on validation data only. Test aggregation is the mean of
every frame's sigmoid score at a threshold fixed at 0.5. No test threshold tuning.

Each baseline epoch saves an atomic recovery snapshot with current and best
weights, optimizer, AMP scaler, completed history and Python/NumPy/Torch/CUDA RNG
states. `training.train --resume` requires the same manifest, settings, code and
runtime. It restores baseline epochs; the short MLP stages restart from their
fixed seed if interrupted. The CPU regression verifies exact resumed vs continuous
weights for the backbone and both heads. `training.campaign --resume` applies both
training and benchmark recovery, then exports the matching final artifacts.

Colab's `/content` disk is temporary. An epoch snapshot on that disk does not
survive runtime replacement. After authorizing Google Drive in the notebook, pass
`--recovery-dir /content/drive/MyDrive/Authenfy/v1-recovery` to training or the
campaign. Training publishes checksummed archives of every 128 prepared videos
and verified checkpoint copies after each completed baseline epoch and after both
heads finish. Partial archives never enter the committed backup index. Recovery
copies add overhead to the recorded training duration. The campaign also backs up
completed stage artifacts; no copying occurs during benchmark clip timings.

The benchmark also accepts `--recovery-dir`. After every fully measured video it
copies immutable frame-score chunks to content-addressed backup objects and
commits a checksummed ledger index. Repeated snapshots reuse existing chunks and
do not duplicate the model weights. Restore applies the newest completed-video
ledger after the trained checkpoint snapshot. A partially measured clip is rerun.
All copies occur between videos, outside every timed inference interval, and the
report discloses this backup policy. `training.campaign --benchmark-only --resume`
continues evaluation without rerunning or resuming optimizer training. It still
validates the manifest and checkpoint hashes. Benchmark code/runtime identity must
match the saved ledger; a code change requires a separately archived attempt.

After a reset, check out the same pinned code revision, prepare the same dataset,
and restore into an empty local run directory:

```text
python -m training.recovery /content/drive/MyDrive/Authenfy/v1-recovery runs/ff-c23
python -m training.campaign data/ff-c23/manifest.csv runs/ff-c23 --resume --recovery-dir /content/drive/MyDrive/Authenfy/v1-recovery
```

Recovery validates file hashes and refuses to overwrite a populated local run.
The training resume checks still require matching code, software, hardware and
settings. A changed environment fails explicitly. Keep the saved revision and
runtime record; do not silently combine incompatible runs. Dataset source videos
are downloaded again when needed; the persistent backup contains sampled crops
and model/results files. Check Drive capacity before a full run. The tests exercise
restoring into a new directory, completed training recovery, corrupt backups and
unsafe archive paths. They are code checks, not held-out accuracy measurements.

The benchmark records accuracy, balanced accuracy, AUROC, F1, log loss, confusion
matrix, per-method results, group-bootstrap paired accuracy difference, median and
p95 clip latency, aggregate FPS and inferred/skipped frame counts. Warm FP32 wall
timing includes decoding, preprocessing, transfer and inference, with GPU
synchronization and rotating arm order. Three repeats are saved individually.
Model loading is separate. Baseline does not compute DCT or pHash.

The benchmark saves a checksummed frame-score chunk and timing record after every
complete video. An interrupted benchmark can use `--resume` with the same code,
manifest, checkpoints, settings and runtime. It rejects changed inputs or corrupt
chunks rather than silently combining runs. Completed clips retain their original
measurements; the report records how many were reused across sessions. A fresh
benchmark requires a new output directory or removal of its own previous records.

Artifacts: `training.json`, three `.pt` checkpoints, `benchmark.json`,
`benchmark.md`, `efficientnet-fusion.onnx`, and `model-card.json`.
An empty result or a code smoke test must never be reported as measured accuracy.

## Complete metric record

`benchmark.json` includes video and descriptive frame metrics: accuracy, balanced
accuracy, precision, recall/sensitivity, specificity, negative predictive value,
binary/macro/weighted F1, AUROC, average precision, trapezoidal PR-AUC, log loss,
Brier score, 10-bin expected calibration error, calibration bins, MCC, Cohen kappa,
per-class report, class counts and confusion matrix. Video ROC and PR curves are
stored with the scores, and method-specific video summaries use the same real
control videos. Frames are correlated and are not independent statistical trials.

`video-predictions.csv` contains every held-out video, source group and model
score. `frame-predictions.npz` preserves all first-repeat frame scores, labels and
video offsets, with a hash in the report. `timings.csv` records every video/arm/
repeat: decode, visual features, synchronized model/transfer time, CPU time,
decoded and inferred frames, wall time and FPS. Summaries include p50/p90/p95/p99
clip latency, mean/std/min/max, aggregate FPS, skip fraction and component totals.

Resource records include PyTorch peak allocated/reserved GPU memory and observed
process RSS. These are explicitly not full-machine peak memory. Energy and GPU
utilization are marked unmeasured because extra polling would perturb timings.
Training saves per-epoch loss, full validation metrics, learning rates, duration,
resource snapshots, parameter counts, class counts, checkpoint sizes and hashes,
preprocessing/split versions, seed, augmentation, optimizer settings, AMP status,
software versions and hardware. `python -m training.plots runs/ff-c23` exports
learning curves, ROC/PR/calibration plots, confusion matrices and speed comparisons.

`additional-metrics.json` also separates the application's 0.2/0.8 review bands
from the benchmark's fixed-0.5 classification: inconclusive/low/flagged counts,
decisive fraction, selective accuracy, flagged precision and real-video flag rate.
Disagreement and approximate-reuse abstention follow the application's policy.
Selective accuracy is conditional on decisive cases, not whole-dataset accuracy.

## Deferred V2

AI-video development is deferred at the user's request. FF++ contains facial manipulation examples; it cannot establish detection of fully
AI-generated videos. V2 needs a separately labeled real/generated video dataset,
source/prompt-disjoint splits, temporal pHash features and its own held-out report.
Do not rename the FF++ result as AI-video accuracy. Preserve this baseline and
evaluate V2 separately before promotion.

## Application installation

After the complete measured bundle has been downloaded:

```powershell
python scripts/install_fusion.py C:/path/to/extracted/run
$env:AUTHENFY_ENGINE = 'fusion'
waitress-serve --host=127.0.0.1 --port=5000 app:app
```

The installer checks the ONNX, card, training and benchmark hashes. The selected
fusion engine fails explicitly when its model is missing rather than falling
back to the earlier ViT/CLIP service. Application inference uses CPU ONNX, all
decoded video frames, batches of eight, guarded pHash reuse and the existing
isolated 90-second deadline. Videos are capped at 3,600 frames and 120 seconds.
Application CPU time is separate from the T4 experiment's GPU speed measurements.
