"""Validate the actual YuNet alignment + ONNX path on verified private audit inputs.

Usage: python scripts/local_face_audit.py private/face-audit/inputs.json
The input manifest contains local_path, path, label_fake and sha256 fields.
This is a correlated regression audit, not independent validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from authenfy.faces import FaceDetector
from authenfy.model import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    images = []
    for row in records:
        assert sha256_file(row["local_path"]) == row["sha256"]
        with Image.open(row["local_path"]) as source:
            images.append(source.convert("RGB"))
    root = Path(__file__).resolve().parents[1]
    detector = FaceDetector(root / "artifacts")
    assert detector.available, "Install the verified T4 specialist export first"
    start = time.perf_counter()
    results = detector.predict(images)
    output = [
        {"path": row["path"], "sha256": row["sha256"], "label_fake": row["label_fake"], **result}
        for row, result in zip(records, results)
    ]
    report = {
        "status": "completed",
        "kind": "actual_yunet_alignment_path_tiny_regression_audit",
        "n_images": len(output),
        "faces_assessed": sum(row["count"] for row in output),
        "missing_coverage": sum(row["coverage"] != "detected_faces_analyzed" for row in output),
        "fake_detected_at_0_5": sum(
            row["label_fake"] == 1 and row["score"] is not None and row["score"] >= 0.5 for row in output
        ),
        "false_positive_at_0_5": sum(
            row["label_fake"] == 0 and row["score"] is not None and row["score"] >= 0.5 for row in output
        ),
        "fake_high_at_0_8": sum(
            row["label_fake"] == 1 and row["score"] is not None and row["score"] >= 0.8 for row in output
        ),
        "false_positive_at_0_8": sum(
            row["label_fake"] == 0 and row["score"] is not None and row["score"] >= 0.8 for row in output
        ),
        "seconds": round(time.perf_counter() - start, 2),
        "model": detector.identity,
        "limitations": "Same 32 correlated prealigned example crops; this checks the application's extra detection/alignment step, not generalization to raw real-world media. No tuning performed.",
        "records": output,
    }
    (root / "reports/local-face-alignment-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
