"""Export standalone scientific figures from saved measured artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(root):
    report = json.loads((root / "benchmark.json").read_text())
    training = json.loads((root / "training.json").read_text())
    arms = report["arms"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, metrics in arms.items():
        curves = metrics["curves"]
        axes[0].plot(curves["roc"]["fpr"], curves["roc"]["tpr"], label=f"{name} AUC={metrics['auroc']:.3f}")
        axes[1].plot(
            curves["pr"]["recall"], curves["pr"]["precision"], label=f"{name} AP={metrics['average_precision']:.3f}"
        )
        bins = [b for b in metrics["calibration_bins"] if b["count"]]
        axes[2].plot([b["mean_score"] for b in bins], [b["fraction_positive"] for b in bins], marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)
    for ax, title, x, y in zip(
        axes,
        ["ROC", "Precision–recall", "Reliability"],
        ["False-positive rate", "Recall", "Mean score"],
        ["True-positive rate", "Precision", "Observed fake fraction"],
        strict=True,
    ):
        ax.set(title=title, xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.suptitle(f"FF++ C23 held-out videos — {report['runtime']['device']}")
    fig.tight_layout()
    fig.savefig(figures / "roc-pr-calibration.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(arms), figsize=(14, 3.5))
    for ax, (name, metrics) in zip(axes, arms.items(), strict=True):
        cm = metrics["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(
            title=name,
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            xlabel="Predicted",
            ylabel="Actual",
        )
    fig.tight_layout()
    fig.savefig(figures / "confusion-matrices.png")
    plt.close(fig)
    names = list(arms)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(
        axes,
        ["balanced_accuracy", "throughput_fps", "skipped_fraction"],
        ["Video balanced accuracy", "End-to-end decoded FPS", "Skipped inference fraction"],
        strict=True,
    ):
        ax.bar(names, [arms[name][metric] for name in names])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures / "accuracy-speed-reuse.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    histories = {"baseline": training["baseline_history"], **training["head_history"]}
    for name, history in histories.items():
        axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label=name)
        axes[1].plot([r["epoch"] for r in history], [r["val"]["log_loss"] for r in history], label=name)
    for ax, title in zip(axes, ["Training weighted BCE", "Validation video log loss"], strict=True):
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "learning-curves.png")
    plt.close(fig)
    print("Figures saved to", figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
