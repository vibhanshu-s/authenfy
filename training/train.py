"""Fine-tune B0, freeze the best backbone, then train deep-only/fused MLPs.

Usage: python -m training.train data/ff-c23/manifest.csv runs/ff-c23 --require-t4
Test media hashes are verified; test features/scores are never used by training.
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from authenfy.fusion_features import FEATURE_VERSION, MEAN, STD, frequency_features, rgb_input
from training.manifest import file_sha, load_manifest
from training.checkpoints import atomic_save, restore_rng, rng_state
from training.metrics import binary_metrics
from training.networks import EfficientBaseline, FusionHead
from training.telemetry import hardware, resources
from training.video import sample_frames


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_identity():
    return dict(
        python=platform.python_version(),
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        opencv=cv2.__version__,
        hardware=hardware(),
        packages={
            k: importlib.metadata.version(k)
            for k in ["torch", "torchvision", "numpy", "Pillow", "scikit-learn", "onnx", "onnxruntime"]
        },
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
    )


def prepare_cache(rows, root, count):
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def cache_one(row):
        directory = root / (row["sha256"] + f"-{count}")
        marker = directory / "complete.json"
        try:
            paths = json.loads(marker.read_text())
            valid = (
                isinstance(paths, list)
                and 0 < len(paths) <= count
                and all(isinstance(p, str) and Path(p).name == p and (directory / p).is_file() for p in paths)
            )
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            directory.mkdir(exist_ok=True)
            frames = sample_frames(row["absolute_path"], count)
            paths = []
            for i, frame in enumerate(frames):
                path = directory / f"{i:03}.png"
                rgb_input(frame).save(path, compress_level=1)
                paths.append(path.name)
            temporary = marker.with_suffix(".partial")
            temporary.write_text(json.dumps(paths))
            temporary.replace(marker)
        return row["sha256"], [str(directory / p) for p in paths]

    unique_rows = list({row["sha256"]: row for row in rows}.values())
    cached = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for number, (digest, paths) in enumerate(pool.map(cache_one, unique_rows), 1):
            cached[digest] = paths
            if number % 25 == 0 or number == len(unique_rows):
                progress = dict(
                    stage="crop_cache",
                    completed=number,
                    total=len(unique_rows),
                    elapsed_seconds=time.perf_counter() - started,
                )
                (root.parent / "cache-progress.json").write_text(json.dumps(progress))
                print(f"cached {number}/{len(unique_rows)} unique training/validation videos", flush=True)
    samples = [
        dict(path=p, label=row["label"], video=row["path"], split=row["split"])
        for row in rows
        for p in cached[row["sha256"]]
    ]
    return samples


class CropDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            pixels = (np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD
            handcrafted = frequency_features(image)
        return (
            torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
            torch.from_numpy(handcrafted),
            torch.tensor(float(row["label"])),
        )


def video_metrics(rows, scores):
    pooled = {}
    for row, score in zip(rows, scores, strict=True):
        pooled.setdefault(row["video"], [row["label"], []])[1].append(float(score))
    return binary_metrics([x[0] for x in pooled.values()], [np.mean(x[1]) for x in pooled.values()])


@torch.inference_mode()
def predict_baseline(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).sigmoid().cpu() for x, _, _ in loader]).numpy()


@torch.inference_mode()
def extract(model, loader, device):
    model.eval()
    vectors, labels = [], []
    for x, handcrafted, y in loader:
        deep = model.features(x.to(device)).cpu()
        vectors.append(torch.cat([deep, handcrafted], 1))
        labels.append(y)
    return torch.cat(vectors), torch.cat(labels)


def train_head(x, y, val_x, val_y, val_rows, dimension, epochs, device, seed):
    seed_everything(seed)
    head = FusionHead(dimension).to(device)
    head.fit_normalizer(x[:, :dimension].to(device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum()), device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(x[:, :dimension], y), batch_size=128, shuffle=True)
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        head.train()
        losses = []
        for features, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            predictions = head(val_x[:, :dimension].to(device)).sigmoid().cpu().numpy()
        metrics = video_metrics(val_rows, predictions)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=float(np.mean(losses)),
                val=metrics,
                epoch_seconds=time.perf_counter() - epoch_start,
                learning_rate=optimizer.param_groups[0]["lr"],
                resources=resources(),
            )
        )
        if metrics["log_loss"] < best:
            best, best_state = metrics["log_loss"], copy.deepcopy(head.cpu().state_dict())
            head.to(device)
        print(
            json.dumps(
                dict(
                    stage=f"head-{dimension}",
                    epoch=epoch + 1,
                    train_loss=history[-1]["train_loss"],
                    val_accuracy=metrics["accuracy"],
                    val_auroc=metrics["auroc"],
                    val_log_loss=metrics["log_loss"],
                )
            ),
            flush=True,
        )
    head.load_state_dict(best_state)
    return head.cpu(), history


def main(args):
    if min(args.frames, args.epochs, args.head_epochs, args.batch_size) < 1:
        raise ValueError("Frame, epoch and batch counts must be positive")
    seed_everything(args.seed)
    runtime = runtime_identity()
    if args.require_t4 and "T4" not in runtime["device"]:
        raise RuntimeError(f"A Colab T4 is required; got {runtime['device']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance_path = args.manifest.parent / "provenance.json"
    config = dict(
        seed=args.seed,
        feature_version=FEATURE_VERSION,
        frames_per_train_video=args.frames,
        baseline_epochs=args.epochs,
        head_epochs=args.head_epochs,
        batch_size=args.batch_size,
        manifest_sha256=file_sha(args.manifest),
        runtime=runtime,
        dataset_provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else None,
        split_counts={s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
        status="training",
        threshold=0.5,
        test_used_for_selection=False,
        split_classes={
            s: {kind: sum(r["split"] == s and r["kind"] == kind for r in rows) for kind in ["real", "manipulated"]}
            for s in ["train", "val", "test"]
        },
        optimizer={
            "name": "AdamW",
            "backbone_lr": 1e-4,
            "classifier_lr": 5e-4,
            "weight_decay": 1e-4,
            "head_lr": 3e-4,
            "head_weight_decay": 1e-3,
        },
        augmentation="50% batch horizontal flip during baseline training only",
        training_precision="CUDA automatic mixed precision" if device.type == "cuda" else "FP32",
        initialized_from="torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
        resources_start=resources(),
    )
    state_path = args.output / "baseline-training-state.pt"
    previous = None
    if args.resume:
        previous_config = json.loads((args.output / "training.json").read_text())
        for key in [
            "seed",
            "feature_version",
            "frames_per_train_video",
            "baseline_epochs",
            "head_epochs",
            "batch_size",
            "manifest_sha256",
            "runtime",
            "optimizer",
            "augmentation",
        ]:
            if previous_config[key] != config[key]:
                raise ValueError(f"Cannot resume: training {key} changed")
        if previous_config["status"] == "trained":
            for name, digest in previous_config["weights_sha256"].items():
                if file_sha(args.output / name) != digest:
                    raise ValueError("Completed training checkpoint integrity mismatch")
            print("TRAINING_ALREADY_COMPLETE", flush=True)
            return
        if state_path.exists():
            previous = torch.load(state_path, map_location="cpu", weights_only=True)
            if (
                previous.get("schema_version") != 1
                or previous["config"]["manifest_sha256"] != config["manifest_sha256"]
                or previous["config"]["runtime"] != config["runtime"]
            ):
                raise ValueError("Training snapshot identity mismatch")
            config = previous["config"]
    elif state_path.exists() or (args.output / "baseline.pt").exists():
        raise ValueError("Training progress exists; use --resume or a new output directory")
    (args.output / "training.json").write_text(json.dumps(config, indent=2))
    train_val = [r for r in rows if r["split"] != "test"]
    cache_start = time.perf_counter()
    cache = prepare_cache(train_val, args.output / "crop-cache", args.frames)
    config.setdefault("cache_preparation_attempt_seconds", []).append(time.perf_counter() - cache_start)
    config["crop_cache_seconds"] = sum(config["cache_preparation_attempt_seconds"])
    split_rows = {s: [r for r in cache if r["split"] == s] for s in ["train", "val"]}
    loaders = {
        s: DataLoader(CropDataset(split_rows[s]), batch_size=args.batch_size, shuffle=False, num_workers=2)
        for s in split_rows
    }
    training_loader = DataLoader(
        CropDataset(split_rows["train"]), batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    model = EfficientBaseline(pretrained=True).to(device)
    config["baseline_parameter_count"] = sum(p.numel() for p in model.parameters())
    config["sampled_frame_counts"] = {s: len(v) for s, v in split_rows.items()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.net.features.parameters(), lr=1e-4),
            dict(params=model.net.classifier.parameters(), lr=5e-4),
        ],
        weight_decay=1e-4,
    )
    train_labels = [r["label"] for r in split_rows["train"]]
    weight = train_labels.count(0) / train_labels.count(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best, best_epoch = [], float("inf"), None
    best_state, completed, elapsed_before = None, 0, 0.0
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        history, best, best_epoch = previous["history"], previous["best"], previous["best_epoch"]
        best_state, completed = previous["best_model"], previous["completed_epochs"]
        elapsed_before = previous["active_training_seconds"]
        # The snapshot is authoritative if a crash occurred between file writes.
        atomic_save(best_state, args.output / "baseline.pt")
        restore_rng(previous["rng"])
        print(f"RESUMED_BASELINE_AFTER_EPOCH_{completed}", flush=True)
    start = time.perf_counter()
    for epoch in range(completed, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for pixels, _, labels in training_loader:
            pixels, labels = pixels.to(device), labels.to(device)
            if random.random() < 0.5:
                pixels = pixels.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(pixels), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        metrics = video_metrics(split_rows["val"], predict_baseline(model, loaders["val"], device))
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=float(np.mean(losses)),
                val=metrics,
                epoch_seconds=time.perf_counter() - epoch_start,
                learning_rates=[g["lr"] for g in optimizer.param_groups],
                resources=resources(),
            )
        )
        if metrics["log_loss"] < best:
            best, best_epoch = metrics["log_loss"], epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            atomic_save(best_state, args.output / "baseline.pt")
        config["baseline_history"] = history
        atomic_save(
            dict(
                schema_version=1,
                config=config,
                model={key: value.detach().cpu() for key, value in model.state_dict().items()},
                best_model=best_state,
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
                best=best,
                best_epoch=best_epoch,
                completed_epochs=epoch + 1,
                active_training_seconds=elapsed_before + time.perf_counter() - start,
                rng=rng_state(),
            ),
            state_path,
        )
        (args.output / "training.json").write_text(json.dumps(config, indent=2))
        print(
            json.dumps(
                dict(
                    stage="baseline",
                    epoch=epoch + 1,
                    train_loss=history[-1]["train_loss"],
                    val_accuracy=metrics["accuracy"],
                    val_auroc=metrics["auroc"],
                    val_log_loss=metrics["log_loss"],
                    epoch_seconds=history[-1]["epoch_seconds"],
                )
            ),
            flush=True,
        )
    model.load_state_dict(torch.load(args.output / "baseline.pt", weights_only=True, map_location=device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = extract(model, loaders["train"], device)
    val_x, val_y = extract(model, loaders["val"], device)
    head_history = {}
    for name, dimension in [("deep_mlp", 1280), ("fusion", 1406)]:
        head, head_history[name] = train_head(
            train_x, train_y, val_x, val_y, split_rows["val"], dimension, args.head_epochs, device, args.seed
        )
        atomic_save(head.state_dict(), args.output / f"{name}.pt")
    config.update(
        status="trained",
        best_baseline_epoch=best_epoch,
        training_seconds=round(elapsed_before + time.perf_counter() - start, 3),
        baseline_history=history,
        head_history=head_history,
        resources_end=resources(),
        checkpoint_bytes={
            f"{name}.pt": (args.output / f"{name}.pt").stat().st_size for name in ["baseline", "deep_mlp", "fusion"]
        },
        weights_sha256={
            f"{name}.pt": file_sha(args.output / f"{name}.pt") for name in ["baseline", "deep_mlp", "fusion"]
        },
        resume_snapshot_sha256=file_sha(state_path),
        resume_snapshot_bytes=state_path.stat().st_size,
    )
    (args.output / "training.json").write_text(json.dumps(config, indent=2))
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--require-t4", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore completed baseline epochs with matching inputs/settings/runtime; MLP stages restart from their fixed seed",
    )
    main(parser.parse_args())
