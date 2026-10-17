"""Video-level metrics and group bootstrap: frames are never independent trials."""

import numpy as np


def binary_metrics(labels, scores, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return dict(
        count=len(labels),
        accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        auroc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        f1=float(f1_score(labels, predicted, zero_division=0)),
        log_loss=float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1])),
        threshold=threshold,
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        false_positive_rate=float(fp / (fp + tn)) if fp + tn else None,
        false_negative_rate=float(fn / (fn + tp)) if fn + tp else None,
    )


def paired_accuracy_interval(rows, left, right, repeats=1000):
    """Difference right-minus-left, bootstrap source groups, fixed threshold .5."""
    groups = sorted({r["group"] for r in rows})
    by_group = {g: [r for r in rows if r["group"] == g] for g in groups}
    random = np.random.default_rng(20261017)
    deltas = []
    for _ in range(repeats):
        sampled = [r for g in random.choice(groups, len(groups), replace=True) for r in by_group[g]]
        deltas.append(
            np.mean([int((r[right] >= 0.5) == r["label"]) - int((r[left] >= 0.5) == r["label"]) for r in sampled])
        )
    return [float(x) for x in np.quantile(deltas, [0.025, 0.975])]
