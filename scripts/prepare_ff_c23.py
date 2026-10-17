"""Download the user's Kaggle FF++ C23 files with official pair-disjoint splits.

Google's separate DeepFakeDetection subset is excluded: its actor IDs are not FF++ IDs.
Download is file-by-file, no credentials required for this public dataset.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.manifest import file_sha, write_manifest

DATASET = "xdxd003/ff-c23"
VERSION = 1
SPLITS_REVISION = "b952e41cba017eb37593c39e12bd884a934791e1"
SPLIT_HASHES = {
    "train": "e59386911255e6fb0a79a7808ec2210536b8c0474268342694c4bca766d1b347",
    "val": "b48cc511f66938e05356aaa9c67e150b1e3638c355db3d745225b0022884b36e",
    "test": "886f5a0da623c25820692e0d8dc33d197ddb1db527a7f1cfcb9bcbca60fe4f40",
}
METHODS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


def download(url, path):
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Dataset download returned an HTML page, not media")
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            partial.replace(path)
            return
        except (requests.RequestException, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def prepare(root, pair_limits=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    revision = SPLITS_REVISION
    rows, split_hashes = [], {}
    used = set()
    for split, filename in [("train", "train.json"), ("val", "val.json"), ("test", "test.json")]:
        target = root / "splits" / filename
        download(
            f"https://raw.githubusercontent.com/ondyari/FaceForensics/{revision}/dataset/splits/{filename}", target
        )
        pairs = json.loads(target.read_text())
        split_hashes[split] = file_sha(target)
        if split_hashes[split] != SPLIT_HASHES[split]:
            raise ValueError("Official split checksum mismatch")
        limit = (pair_limits or {}).get(split, 0)
        if limit:
            # Fixed order independent of media, labels and downstream scores.
            import random

            pairs = sorted(pairs)
            random.Random(20261017).shuffle(pairs)
            pairs = pairs[:limit]
        for a, b in pairs:
            a, b = str(a).zfill(3), str(b).zfill(3)
            if a in used or b in used:
                raise ValueError("Official source IDs cross splits or repeat")
            used.update([a, b])
            group = "ffpp:" + "_".join(sorted([a, b]))
            for method in METHODS:
                names = [a, b] if method == "original" else [f"{a}_{b}", f"{b}_{a}"]
                for name in names:
                    path = f"{method}/{name}.mp4"
                    rows.append(
                        dict(
                            path=path,
                            label=0 if method == "original" else 1,
                            kind="real" if method == "original" else "manipulated",
                            split=split,
                            group=group,
                            source_ids=f"ffpp:{a};ffpp:{b}",
                            method=method,
                            sha256="",
                        )
                    )

    def fetch(row):
        encoded = quote("FaceForensics++_C23/" + row["path"], safe="")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{encoded}?datasetVersionNumber={VERSION}"
        target = root / row["path"]
        download(url, target)
        row["sha256"] = file_sha(target)
        return row

    with ThreadPoolExecutor(max_workers=8) as executor:
        for count, _ in enumerate(executor.map(fetch, rows), 1):
            if count % 24 == 0 or count == len(rows):
                print(f"downloaded {count}/{len(rows)}", flush=True)
    write_manifest(root / "manifest.csv", rows)
    provenance = dict(
        dataset=DATASET,
        dataset_version=VERSION,
        official_splits_revision=revision,
        split_sha256=split_hashes,
        selected_pair_limits=pair_limits,
        methods=METHODS,
        videos=len(rows),
        excluded=["DeepFakeDetection"],
        limitation="Source-pair disjoint, not verified identity-disjoint; uploader videos may share identities or YouTube sources.",
    )
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--train-pairs", type=int, default=0, help="0 uses every official pair")
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--test-pairs", type=int, default=0)
    args = parser.parse_args()
    prepare(args.root, {"train": args.train_pairs, "val": args.val_pairs, "test": args.test_pairs})
