# EfficientNet feature-fusion experiment

The completed v1 implementation and measured metrics are documented in
[BUILD_AND_METRICS.md](BUILD_AND_METRICS.md). This file remains as a short
architecture entry point for the experiment.

Authenfy compares a fine-tuned EfficientNet-B0 baseline, a deep-feature MLP, a
fusion MLP with DCT/pHash features, and the fusion MLP with guarded pHash frame
reuse. The benchmark uses Kaggle FF++ C23 with source-pair-aware splits. The
application decodes every frame and applies a conservative RGB/DCT/pHash reuse
policy while retaining each frame's temporal weight.

Training and recovery code lives in `training/`; dataset preparation is
`scripts/prepare_ff_c23.py`; the Colab notebook is
`notebooks/Authenfy_EfficientNet_Fusion_T4.ipynb`. The 150-video comparison and
the earlier 100-video run are preserved as separate records in
`reports/v1-evidence`. See the main build and metrics guide for commands,
results, metric definitions, evidence verification, and limitations.

AI-generated video detection was deferred for this release. FF++ results are
facial-manipulation screening results and must not be reported as AI-video
detection accuracy.
