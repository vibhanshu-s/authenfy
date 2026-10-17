"""Fine-tune B0, freeze the best backbone, then train deep-only/fused MLPs.

Usage: python -m training.train data/ff-c23/manifest.csv runs/ff-c23 --require-t4
Test data is never opened by this command. Test evaluation is a separate command.
"""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from authenfy.fusion_features import FEATURE_VERSION, MEAN, STD, frequency_features, rgb_input
from training.manifest import file_sha, load_manifest
from training.metrics import binary_metrics
from training.networks import EfficientBaseline, FusionHead
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
        packages={
            k: importlib.metadata.version(k) for k in ["torch", "torchvision", "numpy", "Pillow", "scikit-learn"]
        },
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
    )


def prepare_cache(rows, root, count):
    root.mkdir(parents=True, exist_ok=True)
    samples = []
    for number, row in enumerate(rows):
        directory = root / (row["sha256"] + f"-{count}")
        marker = directory / "complete.json"
        if not marker.exists():
            directory.mkdir(exist_ok=True)
            frames = sample_frames(row["absolute_path"], count)
            paths = []
            for i, frame in enumerate(frames):
                path = directory / f"{i:03}.png"
                rgb_input(frame).save(path)
                paths.append(path.name)
            marker.write_text(json.dumps(paths))
        paths = json.loads(marker.read_text())
        samples.extend(
            dict(path=str(directory / p), label=row["label"], video=row["path"], split=row["split"]) for p in paths
        )
        if (number + 1) % 100 == 0:
            print(f"cached {number + 1}/{len(rows)} training/validation videos", flush=True)
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
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val=metrics))
        if metrics["log_loss"] < best:
            best, best_state = metrics["log_loss"], copy.deepcopy(head.cpu().state_dict())
            head.to(device)
        print(json.dumps(dict(stage=f"head-{dimension}", **history[-1])), flush=True)
    head.load_state_dict(best_state)
    return head.cpu(), history


def main(args):
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
    )
    (args.output / "training.json").write_text(json.dumps(config, indent=2))
    train_val = [r for r in rows if r["split"] != "test"]
    cache = prepare_cache(train_val, args.output / "crop-cache", args.frames)
    split_rows = {s: [r for r in cache if r["split"] == s] for s in ["train", "val"]}
    loaders = {
        s: DataLoader(CropDataset(split_rows[s]), batch_size=args.batch_size, shuffle=False, num_workers=2)
        for s in split_rows
    }
    training_loader = DataLoader(
        CropDataset(split_rows["train"]), batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    model = EfficientBaseline(pretrained=True).to(device)
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
    start = time.perf_counter()
    for epoch in range(args.epochs):
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
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val=metrics))
        if metrics["log_loss"] < best:
            best, best_epoch = metrics["log_loss"], epoch + 1
            torch.save(model.cpu().state_dict(), args.output / "baseline.pt")
            model.to(device)
        print(json.dumps(dict(stage="baseline", **history[-1])), flush=True)
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
        torch.save(head.state_dict(), args.output / f"{name}.pt")
    config.update(
        status="trained",
        best_baseline_epoch=best_epoch,
        training_seconds=round(time.perf_counter() - start, 3),
        baseline_history=history,
        head_history=head_history,
        weights_sha256={
            f"{name}.pt": file_sha(args.output / f"{name}.pt") for name in ["baseline", "deep_mlp", "fusion"]
        },
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
    main(parser.parse_args())
