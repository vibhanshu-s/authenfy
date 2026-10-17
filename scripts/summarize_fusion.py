"""Add group-bootstrap uncertainty and ROC operating points to saved V1 results.

No model fitting, threshold selection or inference is performed by this script.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.manifest import file_sha

ARMS = ["baseline_all", "deep_mlp_all", "fusion_all", "fusion_phash"]


def scalars(y, p):
    predicted = p >= 0.5
    tp, tn = np.sum(predicted & (y == 1)), np.sum(~predicted & (y == 0))
    fp, fn = np.sum(predicted & (y == 0)), np.sum(~predicted & (y == 1))
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        accuracy=float(np.mean(predicted == y)),
        balanced_accuracy=float((tp / (tp + fn) + tn / (tn + fp)) / 2),
        precision=float(tp / (tp + fp)) if tp + fp else 0.0,
        recall=float(tp / (tp + fn)),
        f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        false_positive_rate=float(fp / (fp + tn)),
        false_negative_rate=float(fn / (fn + tp)),
        auroc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        brier_score=float(np.mean((p - y) ** 2)),
        log_loss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
    )


def summarize(root, repeats=2000):
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    path = root / "benchmark.json"
    report = json.loads(path.read_text())
    if report.get("status") != "measured":
        raise ValueError("Complete the held-out benchmark first")
    records = report["predictions"]
    y = np.array([r["label"] for r in records])
    groups = sorted({r["group"] for r in records})
    indices = {g: np.array([i for i, r in enumerate(records) if r["group"] == g]) for g in groups}
    predictions = {a: np.array([r[a] for r in records]) for a in ARMS}
    times = {
        a: np.array([np.median([x["total_seconds"] for x in r["measurements"][a]]) for r in records]) for a in ARMS
    }
    random = np.random.default_rng(20261017)
    draws = {a: [] for a in ARMS}
    speedups, differences = [], []
    for _ in range(repeats):
        ix = np.concatenate([indices[g] for g in random.choice(groups, len(groups), replace=True)])
        if len(np.unique(y[ix])) < 2:
            continue
        for arm in ARMS:
            draws[arm].append(scalars(y[ix], predictions[arm][ix]))
        speedups.append(float(times["baseline_all"][ix].sum() / times["fusion_phash"][ix].sum()))
        differences.append(
            float(
                np.mean((predictions["fusion_phash"][ix] >= 0.5) == y[ix])
                - np.mean((predictions["baseline_all"][ix] >= 0.5) == y[ix])
            )
        )
    intervals = {
        a: {key: [float(v) for v in np.quantile([d[key] for d in draws[a]], [0.025, 0.975])] for key in draws[a][0]}
        for a in ARMS
    }
    points = {}
    for arm, p in predictions.items():
        fpr, tpr, thresholds = roc_curve(y, p)
        eer_index = int(np.argmin(np.abs(fpr - (1 - tpr))))
        points[arm] = dict(
            approximate_equal_error_rate=float((fpr[eer_index] + 1 - tpr[eer_index]) / 2),
            eer_threshold=float(thresholds[eer_index]) if np.isfinite(thresholds[eer_index]) else None,
            empirical_tpr_at_fpr={str(limit): float(np.max(tpr[fpr <= limit])) for limit in [0.01, 0.05, 0.10]},
        )
    training = json.loads((root / "training.json").read_text())
    classes = training["split_classes"]["train"]
    prior = classes["manipulated"] / (classes["manipulated"] + classes["real"])
    result = dict(
        schema_version=1,
        benchmark_sha256=file_sha(path),
        source_groups=len(groups),
        group_bootstrap_repeats=repeats,
        usable_repeats=len(speedups),
        seed=20261017,
        ci95_by_arm=intervals,
        descriptive_roc_operating_points=points,
        paired_fusion_minus_baseline_accuracy_ci95=[float(v) for v in np.quantile(differences, [0.025, 0.975])],
        paired_speedup_baseline_over_fusion_ci95=[float(v) for v in np.quantile(speedups, [0.025, 0.975])],
        constant_training_prior_control=scalars(y, np.full(len(y), prior)),
        operating_points_used_for_deployment=False,
        notes=[
            "Source groups, not correlated frames, are resampled.",
            "Empirical ROC operating points are descriptive; test thresholds are not deployed.",
            f"There are {int(np.sum(y == 0))} real test videos; empirical FPR resolution is limited by this count.",
            "One training seed; bootstrap uncertainty does not include seed or dataset-shift uncertainty.",
        ],
    )
    (root / "additional-metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    summarize(args.run, args.repeats)
