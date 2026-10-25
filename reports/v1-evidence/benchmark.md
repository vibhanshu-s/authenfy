# FF++ C23 paired benchmark

Measured on NVIDIA GeForce RTX 4060 Laptop GPU

| Arm | Accuracy | Balanced accuracy | AUROC | FPS | Skipped frames |
|---|---:|---:|---:|---:|---:|
| baseline_all | 0.8700 | 0.8557 | 0.9282 | 97.55 | 0.00% |
| deep_mlp_all | 0.9300 | 0.8056 | 0.9465 | 97.40 | 0.00% |
| fusion_all | 0.9300 | 0.8056 | 0.9438 | 94.78 | 0.00% |
| fusion_phash | 0.9300 | 0.8056 | 0.9438 | 113.77 | 66.82% |

These results cover 100 held-out videos out of 240 in the selected test split, not the entire Kaggle collection.

One training seed; source-pair splits do not prove identity-disjoint generalization.
FF++ facial manipulations do not establish detection of fully AI-generated videos.
Warm inference excludes model loading; filesystem page cache may be warm.
If resumed_videos is nonzero, completed video measurements were checksummed and retained across sessions with identical code/settings/runtime.
Optional Drive recovery I/O runs between videos and is excluded from inference latency; source media are still decoded afresh for every arm/repeat.
Approximate pHash reuse can miss short or subtle edits; compare against fusion_all.