"""Install a Colab-exported specialist only if it matches the selected validation receipt."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from authenfy.faces import YUNET_REVISION, YUNET_SHA, FACE_MODEL_ID, FACE_REVISION, FACE_SOURCE_SHA
from authenfy.model import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="Path to face-specialist.onnx downloaded from the T4 notebook")
    parser.add_argument(
        "--validation", type=Path, help="Validation report from your own trusted reproduction of the T4 notebook"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report_path = args.validation or root / "reports/specialist-validation.json"
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "passed"
        or report.get("source_model") != FACE_MODEL_ID
        or report.get("source_revision") != FACE_REVISION
        or report.get("source_sha256") != FACE_SOURCE_SHA
    ):
        raise SystemExit("Validation report has incorrect source model provenance or failed status.")
    if sha256_file(args.onnx) != report["onnx_sha256"]:
        raise SystemExit("The ONNX file does not match the selected T4 export receipt.")
    directory = root / "artifacts"
    directory.mkdir(exist_ok=True)
    target = directory / "face-specialist.onnx"
    if args.onnx.resolve() != target.resolve():
        shutil.copyfile(args.onnx, target)
    (directory / "specialist-validation.json").write_text(json.dumps(report, indent=2))
    url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{YUNET_REVISION}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    temporary = directory / "yunet.part"
    try:
        temporary.write_bytes(response.content)
        if sha256_file(temporary) != YUNET_SHA:
            raise SystemExit("YuNet checksum mismatch")
        temporary.replace(directory / "yunet.onnx")
    finally:
        temporary.unlink(missing_ok=True)
    print("Specialist model and face detector verified and installed.")


if __name__ == "__main__":
    main()
