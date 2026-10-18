"""Install a trained ONNX model with its matching measured benchmark and card."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from authenfy.fusion_features import FEATURE_VERSION
from authenfy.model import sha256_file


def install(source, target):
    card = json.loads((source / "model-card.json").read_text())
    benchmark = json.loads((source / "benchmark.json").read_text())
    training = json.loads((source / "training.json").read_text())
    if card.get("status") != "validated" or card.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Need a validated matching feature architecture")
    if benchmark.get("status") != "measured" or card["benchmark_sha256"] != sha256_file(source / "benchmark.json"):
        raise ValueError("Need the exact measured benchmark associated with this export")
    if (
        card["training_manifest_sha256"] != benchmark["manifest_sha256"]
        or card["training_weights_sha256"] != benchmark["weights_sha256"]
        or training["status"] != "trained"
        or card["training_sha256"] != sha256_file(source / "training.json")
        or training["manifest_sha256"] != benchmark["manifest_sha256"]
        or training["weights_sha256"] != benchmark["weights_sha256"]
    ):
        raise ValueError("Training/export/benchmark provenance mismatch")
    if sha256_file(source / "efficientnet-fusion.onnx") != card["model_sha256"]:
        raise ValueError("ONNX checksum mismatch")
    target.mkdir(parents=True, exist_ok=True)
    existing = target / "efficientnet-fusion.onnx"
    if existing.exists() and sha256_file(existing) != card["model_sha256"]:
        raise ValueError("Target already contains a different model; use a new version directory")
    for name in ["efficientnet-fusion.onnx", "model-card.json", "benchmark.json", "training.json"]:
        path = target / name
        temporary = path.with_suffix(path.suffix + ".partial")
        shutil.copyfile(source / name, temporary)
        temporary.replace(path)
    print(json.dumps(dict(installed=str(target.resolve()), sha256=card["model_sha256"], task=card["task"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=Path, default=Path("artifacts/efficientnet-v1"))
    args = parser.parse_args()
    install(args.source, args.target)
