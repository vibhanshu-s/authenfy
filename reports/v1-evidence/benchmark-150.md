# Authenfy v1 benchmark comparison

Measured on NVIDIA GeForce RTX 4060 Laptop GPU using 3 timing repeats per video.

Accuracy, AUROC, and F1 are compared with the EfficientNet-B0 all-frame baseline. Speed change is based on measured throughput; positive means faster than baseline.

| Model | Accuracy | Accuracy change vs baseline | AUROC | AUROC change vs baseline | F1 | F1 change vs baseline | Throughput | Speed change vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_all | 88.67% | +0.00 pp | 0.9318 | +0.0000 | 0.9300 | +0.0000 | 82.33 FPS | +0.00% |
| deep_mlp_all | 92.67% | +4.00 pp | 0.9429 | +0.0112 | 0.9572 | +0.0272 | 82.18 FPS | -0.18% |
| fusion_all | 92.67% | +4.00 pp | 0.9417 | +0.0099 | 0.9572 | +0.0272 | 79.85 FPS | -3.02% |
| fusion_phash | 92.67% | +4.00 pp | 0.9432 | +0.0115 | 0.9572 | +0.0272 | 92.12 FPS | +11.89% |

Fusion+pHash measured **+11.89% speed** versus baseline. Its accuracy difference 95% source-group bootstrap interval is **[-2.67, +12.50] percentage points**, which includes zero; therefore the accuracy comparison is not conclusive at the 95% level.

The full raw metrics and per-video records remain in `benchmark-150.json`. The comparison-only metrics are in `benchmark-150-comparison.json`.
