"""Paired held-out video accuracy and actual all-frame synchronized wall timing.

Each arm reopens and decodes the same complete clip. No cached features, decoding
exclusions, artificial repeated frames, or model loading are included as inference.
Cold model loading is recorded separately. Timing repeats rotate arm order.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from authenfy.fusion_features import FrameSelector, model_inputs, tensor_input
from training.manifest import file_sha, load_manifest
from training.metrics import binary_metrics, paired_accuracy_interval
from training.networks import EfficientBaseline, FusionHead
from training.train import runtime_identity
from training.video import iter_frames

ARMS = ["baseline_all", "deep_mlp_all", "fusion_all", "fusion_phash"]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def run_video(path, baseline, heads, arm, device, batch_size=32):
    selector = FrameSelector() if arm == "fusion_phash" else None
    scores, assignments, pending_pixels, pending_handcrafted = {}, [], [], []
    pending_indices = []
    timings = dict(decode_seconds=0.0, feature_seconds=0.0, model_seconds=0.0)
    synchronize(device)
    start = time.perf_counter()

    def flush():
        if not pending_pixels:
            return
        synchronize(device)
        t = time.perf_counter()
        pixels = torch.from_numpy(np.stack(pending_pixels)).to(device)
        if arm == "baseline_all":
            predictions = baseline(pixels)
        else:
            features = baseline.features(pixels)
            if arm == "deep_mlp_all":
                predictions = heads["deep_mlp"](features)
            else:
                crafted = torch.from_numpy(np.stack(pending_handcrafted)).to(device)
                predictions = heads["fusion"](torch.cat([features, crafted], 1))
        values = predictions.sigmoid().cpu().numpy()
        synchronize(device)
        timings["model_seconds"] += time.perf_counter() - t
        for index, score in zip(pending_indices, values, strict=True):
            scores[index] = float(score)
        pending_pixels.clear()
        pending_handcrafted.clear()
        pending_indices.clear()

    iterator = iter(iter_frames(path))
    while True:
        t = time.perf_counter()
        try:
            index, image = next(iterator)
        except StopIteration:
            break
        timings["decode_seconds"] += time.perf_counter() - t
        t = time.perf_counter()
        anchor = selector.select(image, index) if selector else None
        assignments.append(index if anchor is None else anchor)
        if anchor is None:
            if arm in {"baseline_all", "deep_mlp_all"}:
                pixels, handcrafted = tensor_input(image), None
            else:
                pixels, handcrafted = model_inputs(image)
            pending_pixels.append(pixels)
            pending_handcrafted.append(handcrafted)
            pending_indices.append(index)
        timings["feature_seconds"] += time.perf_counter() - t
        if len(pending_pixels) == batch_size:
            flush()
    flush()
    synchronize(device)
    elapsed = time.perf_counter() - start
    expanded = [scores[index] for index in assignments]
    if not expanded:
        raise ValueError("No frames decoded")
    return dict(
        score=float(np.mean(expanded)),
        frames=len(expanded),
        inferred_frames=len(scores),
        total_seconds=elapsed,
        frames_per_second=len(expanded) / elapsed,
        **timings,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity = runtime_identity()
    if args.require_t4 and "T4" not in identity["device"]:
        raise RuntimeError("Benchmark must run on a Colab T4")
    training = json.loads((args.run / "training.json").read_text())
    if training["status"] != "trained" or file_sha(args.manifest) != training["manifest_sha256"]:
        raise ValueError("Training/benchmark manifest or status mismatch")
    for name, digest in training["weights_sha256"].items():
        if file_sha(args.run / name) != digest:
            raise ValueError("Checkpoint integrity mismatch")
    rows = [r for r in load_manifest(args.manifest) if r["split"] == "test"]
    load_start = time.perf_counter()
    baseline = EfficientBaseline().to(device)
    baseline.load_state_dict(torch.load(args.run / "baseline.pt", map_location=device, weights_only=True))
    heads = {name: FusionHead(dim).to(device) for name, dim in [("deep_mlp", 1280), ("fusion", 1406)]}
    for name, head in heads.items():
        head.load_state_dict(torch.load(args.run / f"{name}.pt", map_location=device, weights_only=True))
        head.eval()
    baseline.eval()
    synchronize(device)
    cold = time.perf_counter() - load_start
    # Warm each exact graph before timing; no test predictions used for selection.
    with torch.inference_mode():
        x = torch.zeros(args.batch_size, 3, 224, 224, device=device)
        for _ in range(3):
            f = baseline.features(x)
            baseline.net.classifier(f)
            heads["deep_mlp"](f)
            heads["fusion"](torch.cat([f, torch.zeros(args.batch_size, 126, device=device)], 1))
    synchronize(device)
    results = []
    for video_index, row in enumerate(rows):
        record = {k: row[k] for k in ["path", "label", "group", "method", "sha256"]}
        samples = {arm: [] for arm in ARMS}
        for repeat in range(args.repeats):
            offset = (video_index + repeat) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                samples[arm].append(run_video(row["absolute_path"], baseline, heads, arm, device, args.batch_size))
        for arm in ARMS:
            if len({r["frames"] for r in samples[arm]}) != 1:
                raise ValueError("Frame count changed between repeats")
            record[arm] = samples[arm][0]["score"]
        if len({samples[arm][0]["frames"] for arm in ARMS}) != 1:
            raise ValueError("Arms did not decode the same number of frames")
        record["measurements"] = samples
        results.append(record)
        (args.run / "benchmark-progress.json").write_text(json.dumps(results, indent=2))
        print(f"benchmarked {video_index + 1}/{len(rows)} {row['path']}", flush=True)
    summary = {}
    for arm in ARMS:
        per_video = [r["measurements"][arm] for r in results]
        seconds = [statistics.median(x["total_seconds"] for x in values) for values in per_video]
        frames = sum(values[0]["frames"] for values in per_video)
        inferred = sum(values[0]["inferred_frames"] for values in per_video)
        summary[arm] = dict(
            **binary_metrics([r["label"] for r in results], [r[arm] for r in results]),
            median_clip_seconds=statistics.median(seconds),
            p95_clip_seconds=float(np.quantile(seconds, 0.95)),
            total_seconds=sum(seconds),
            frames=frames,
            inferred_frames=inferred,
            skipped_fraction=1 - inferred / frames,
            throughput_fps=frames / sum(seconds),
            by_method={
                method: binary_metrics(
                    [r["label"] for r in results if r["method"] in {"original", method}],
                    [r[arm] for r in results if r["method"] in {"original", method}],
                )
                for method in sorted({r["method"] for r in results} - {"original"})
            },
        )
    report = dict(
        schema_version=1,
        status="measured",
        task="facial_manipulation",
        manifest_sha256=training["manifest_sha256"],
        weights_sha256=training["weights_sha256"],
        runtime=identity,
        repeats=args.repeats,
        batch_size=args.batch_size,
        model_loading_seconds=cold,
        protocol="Sequentially decode EVERY frame independently per arm; warmed FP32 inference; GPU synchronization; rotated order; temporal frame-weighted mean; threshold 0.5 fixed before test.",
        pHash_policy=dict(hamming=2, max_gap=12, max_rgb_mae=0.01, max_dct_mae=0.015),
        accuracy_difference_ci95_group_bootstrap=paired_accuracy_interval(results, "baseline_all", "fusion_phash"),
        scope=training["dataset_provenance"],
        arms=summary,
        speedup_baseline_over_fusion=summary["baseline_all"]["total_seconds"]
        / summary["fusion_phash"]["total_seconds"],
        predictions=results,
        limitations=[
            "One training seed; source-pair splits do not prove identity-disjoint generalization.",
            "FF++ facial manipulations do not establish detection of fully AI-generated videos.",
            "Warm inference excludes model loading; filesystem page cache may be warm.",
            "Approximate pHash reuse can miss short or subtle edits; compare against fusion_all.",
        ],
    )
    (args.run / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = [
        "# FF++ C23 paired benchmark",
        "",
        "Measured on " + identity["device"],
        "",
        "| Arm | Accuracy | Balanced accuracy | AUROC | FPS | Skipped frames |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary.items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['auroc']:.4f} | {metrics['throughput_fps']:.2f} | {metrics['skipped_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "These results cover the manifest's selected held-out source pairs, not the entire Kaggle collection.",
            "",
            *report["limitations"],
        ]
    )
    (args.run / "benchmark.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-t4", action="store_true")
    main(parser.parse_args())
