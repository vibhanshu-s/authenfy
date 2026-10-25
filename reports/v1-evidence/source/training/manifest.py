"""Explicit source-level splits; never randomly split frames from the same video."""

import csv
import hashlib
from pathlib import Path

LABELS = {"real": 0, "manipulated": 1, "ai_generated": 1}
FIELDS = ["path", "label", "kind", "split", "group", "source_ids", "method", "sha256"]


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path, root=None, task="deepfake", verify=True):
    path = Path(path)
    root = Path(root or path.parent).resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(FIELDS).issubset(rows[0]):
        raise ValueError(f"Manifest must contain {FIELDS}")
    seen_paths, seen_hashes, seen_sources = {}, {}, {}
    for row in rows:
        if row["kind"] not in LABELS or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown label kind or split")
        row["label"] = int(row["label"])
        if row["label"] != LABELS[row["kind"]] or not row["group"] or not row["source_ids"]:
            raise ValueError("Label/group/source provenance missing or inconsistent")
        media = (root / row["path"]).resolve()
        if not media.is_relative_to(root) or not media.is_file():
            raise ValueError(f"Missing/outside-root media: {row['path']}")
        if str(media) in seen_paths:
            raise ValueError("Repeated media path")
        seen_paths[str(media)] = row["split"]
        if len(row["sha256"]) != 64 or (verify and file_sha(media) != row["sha256"]):
            raise ValueError(f"Invalid checksum: {row['path']}")
        for key in ["group:" + row["group"], *["source:" + s for s in row["source_ids"].split(";")]]:
            if key in seen_sources and seen_sources[key] != row["split"]:
                raise ValueError(f"Source leakage across splits: {key}")
            seen_sources[key] = row["split"]
        if row["sha256"] in seen_hashes and seen_hashes[row["sha256"]] != row["split"]:
            raise ValueError("Identical media bytes cross splits")
        seen_hashes[row["sha256"]] = row["split"]
        row["absolute_path"] = str(media)
    if task not in {"deepfake", "ai_video"}:
        raise ValueError("Unknown training task")
    allowed = {"real", "manipulated"} if task == "deepfake" else {"real", "ai_generated"}
    selected = [r for r in rows if r["kind"] in allowed]
    for split in ["train", "val", "test"]:
        if {r["label"] for r in selected if r["split"] == split} != {0, 1}:
            raise ValueError(f"Each {task} split needs both classes: {split}")
    return selected


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in rows)
