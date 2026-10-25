"""Copy a completed FF++ run into a hash-verifiable, redistributable evidence bundle."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = (
    "benchmark.json",
    "benchmark.md",
    "training.json",
    "model-card.json",
    "video-predictions.csv",
    "frame-predictions.npz",
    "timings.csv",
)
CODE_FILES = (
    "training/benchmark.py",
    "training/metrics.py",
    "training/video.py",
    "training/manifest.py",
    "training/recording.py",
    "authenfy/fusion_features.py",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(output):
    folder = Path(output)
    manifest = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))
    errors = []
    for relative, expected in manifest["files"].items():
        path = folder / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif digest(path) != expected["sha256"]:
            errors.append(f"hash mismatch: {relative}")
        elif path.stat().st_size != expected["bytes"]:
            errors.append(f"size mismatch: {relative}")
    actual_manifest_hash = digest(folder / "evidence.json")
    expected_sidecar = (folder / "evidence.json.sha256").read_text(encoding="utf-8").split()[0]
    if actual_manifest_hash != expected_sidecar:
        errors.append("hash mismatch: evidence.json")
    bundle = folder / "evidence-bundle.zip"
    bundle_sidecar = folder / "evidence-bundle.zip.sha256"
    if not bundle.is_file() or not bundle_sidecar.is_file() or digest(bundle) != bundle_sidecar.read_text(encoding="ascii").split()[0]:
        errors.append("hash mismatch: evidence-bundle.zip")
    elif not errors:
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            expected_names = set(manifest["files"]) | {"evidence.json", "evidence.json.sha256"}
            if names != expected_names:
                errors.append("bundle file list mismatch")
            for relative, expected in manifest["files"].items():
                if hashlib.sha256(archive.read(relative)).hexdigest() != expected["sha256"]:
                    errors.append(f"bundle hash mismatch: {relative}")
    if errors:
        raise SystemExit("Evidence verification failed: " + "; ".join(errors))
    print(json.dumps({"status": "verified", "files": len(manifest["files"]), "evidence_sha256": actual_manifest_hash}))


def verify_150(output):
    folder = Path(output)
    manifest_path = folder / "evidence-150.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = folder / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        elif digest(path) != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']}")
    for relative, expected in (
        ("benchmark-150.json", manifest["benchmark_sha256"]),
        ("benchmark-150-comparison.json", manifest["comparison_sha256"]),
    ):
        if not (folder / relative).is_file() or digest(folder / relative) != expected:
            errors.append(f"hash mismatch: {relative}")
    benchmark = json.loads((folder / "benchmark-150.json").read_text(encoding="utf-8"))
    if benchmark.get("status") != "measured" or benchmark.get("test_videos") != 150 or benchmark.get("repeats") != 3:
        errors.append("benchmark scope mismatch")
    actual_manifest_hash = digest(manifest_path)
    sidecar = folder / "evidence-150.json.sha256"
    if not sidecar.is_file() or actual_manifest_hash != sidecar.read_text(encoding="ascii").split()[0]:
        errors.append("hash mismatch: evidence-150.json")
    if errors:
        raise SystemExit("150-video evidence verification failed: " + "; ".join(errors))
    print(json.dumps({"status": "verified", "files": len(manifest["files"]), "evidence_sha256": actual_manifest_hash}))


def record(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for relative in RUN_FILES:
        origin = source / relative
        if not origin.is_file():
            raise SystemExit(f"Required result is missing: {origin}")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        files[relative] = {"sha256": digest(target), "bytes": target.stat().st_size}
    for relative in CODE_FILES:
        origin = ROOT / relative
        target_relative = Path("source") / relative
        target = output / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        files[target_relative.as_posix()] = {"sha256": digest(target), "bytes": target.stat().st_size}

    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    training = json.loads((output / "training.json").read_text(encoding="utf-8"))
    if benchmark.get("status") != "measured" or benchmark.get("test_videos") != 100:
        raise SystemExit("Expected a measured benchmark covering exactly 100 held-out videos")
    manifest = {
        "schema_version": 1,
        "evidence_id": "v1-ff-c23-100",
        "title": "Authenfy EfficientNet fusion v1: 100-video FF++ C23 benchmark",
        "dataset": "Kaggle FF++ C23, xdxd003/ff-c23, dataset version 1",
        "dataset_url": "https://www.kaggle.com/datasets/xdxd003/ff-c23",
        "scope": "100 selected held-out videos out of a 240-video test pool; 18 real and 82 manipulated; three timing repeats per arm",
        "device": benchmark["runtime"]["device"],
        "runtime": benchmark["runtime"],
        "git_commit": benchmark["runtime"].get("git_commit"),
        "seed": training.get("seed"),
        "training_manifest_sha256": training.get("manifest_sha256"),
        "test_manifest_sha256": benchmark.get("manifest_sha256"),
        "weights_sha256": benchmark["weights_sha256"],
        "accuracy_difference_ci95_group_bootstrap": benchmark.get("accuracy_difference_ci95_group_bootstrap"),
        "speedup_baseline_over_fusion": benchmark.get("speedup_baseline_over_fusion"),
        "limitations": benchmark.get("limitations", []),
        "metric_notes": benchmark.get("metric_notes", []),
        "arms": benchmark["arms"],
        "files": files,
        "verification": "Recompute every files.*.sha256 value and compare evidence.json.sha256; an Ethereum AnalysisRegistry anchor can timestamp the benchmark.json hash and this manifest hash.",
        "proof_limit": "Hashes establish that later files match this recorded bundle. Public-chain anchoring adds a tamper-evident timestamp after deployment; neither proves dataset authenticity or model truthfulness.",
    }
    manifest_path = output / "evidence.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "evidence.json.sha256").write_text(f"{digest(manifest_path)}  evidence.json\n", encoding="ascii")
    bundle_path = output / "evidence-bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in [*files, "evidence.json", "evidence.json.sha256"]:
            archive.write(output / relative, arcname=relative)
    (output / "evidence-bundle.zip.sha256").write_text(f"{digest(bundle_path)}  evidence-bundle.zip\n", encoding="ascii")
    print(json.dumps({"status": "recorded", "files": len(files), "evidence_sha256": digest(manifest_path), "benchmark_sha256": files["benchmark.json"]["sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/ff-c23-pc")
    parser.add_argument("--output", default="reports/v1-evidence")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-150", action="store_true")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if args.verify and args.verify_150:
        parser.error("Choose either --verify or --verify-150")
    if args.verify_150:
        verify_150(destination)
    elif args.verify:
        verify(destination)
    else:
        record((ROOT / args.source).resolve(), destination)
