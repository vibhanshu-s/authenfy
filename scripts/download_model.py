"""Download the immutable model artifact, verify bytes, then install atomically."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from authenfy.model import MODEL_ID, REVISION, MODEL_SHA256, sha256_file


def main():
    target = Path(__file__).resolve().parents[1] / "artifacts/model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == MODEL_SHA256:
        print("Pinned model already verified:", target)
        return
    temporary = target.with_suffix(".part")
    try:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/onnx/model.onnx"
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        if sha256_file(temporary) != MODEL_SHA256:
            raise RuntimeError("Downloaded model failed SHA-256 verification")
        temporary.replace(target)
        print("Verified model:", target, MODEL_SHA256)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
