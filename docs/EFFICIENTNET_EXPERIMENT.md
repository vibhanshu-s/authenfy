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
```

Training samples eight frames per video, uses class-weighted binary cross entropy,
and chooses checkpoints on validation data only. Test aggregation is the mean of
every frame's sigmoid score at a threshold fixed at 0.5. No test threshold tuning.

The benchmark records accuracy, balanced accuracy, AUROC, F1, log loss, confusion
matrix, per-method results, group-bootstrap paired accuracy difference, median and
p95 clip latency, aggregate FPS and inferred/skipped frame counts. Warm FP32 wall
timing includes decoding, preprocessing, transfer and inference, with GPU
synchronization and rotating arm order. Three repeats are saved individually.
Model loading is separate. Baseline does not compute DCT or pHash.

Artifacts: `training.json`, three `.pt` checkpoints, `benchmark.json`,
`benchmark.md`, `efficientnet-fusion.onnx`, and `model-card.json`.
An empty result or a code smoke test must never be reported as measured accuracy.

## V2 scope

FF++ contains facial manipulation examples; it cannot establish detection of fully
AI-generated videos. V2 needs a separately labeled real/generated video dataset,
source/prompt-disjoint splits, temporal pHash features and its own held-out report.
Do not rename the FF++ result as AI-video accuracy. Preserve this baseline and
evaluate V2 separately before promotion.
