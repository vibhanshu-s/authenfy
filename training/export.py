"""Export a trained fusion model only after a measured benchmark, with FP32 parity."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from authenfy.fusion_features import FEATURE_VERSION, model_inputs
from training.manifest import file_sha, load_manifest
from training.networks import EfficientBaseline, FusionHead, FusionModel
from training.train import runtime_identity
from training.video import sample_frames


def main(args):
    training = json.loads((args.run / "training.json").read_text())
    benchmark = json.loads((args.run / "benchmark.json").read_text())
    if training["status"] != "trained" or benchmark["status"] != "measured":
        raise ValueError("Need completed training and benchmark")
    if file_sha(args.manifest) != training["manifest_sha256"]:
        raise ValueError("Manifest mismatch")
    if (
        benchmark["manifest_sha256"] != training["manifest_sha256"]
        or benchmark["weights_sha256"] != training["weights_sha256"]
    ):
        raise ValueError("Training/benchmark provenance mismatch")
    baseline, head = EfficientBaseline(), FusionHead()
    for name, model in [("baseline", baseline), ("fusion", head)]:
        path = args.run / f"{name}.pt"
        if file_sha(path) != training["weights_sha256"][path.name]:
            raise ValueError("Weight hash mismatch")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = FusionModel(baseline, head).eval()
    output = args.run / "efficientnet-fusion.onnx"
    examples = (torch.zeros(1, 3, 224, 224), torch.zeros(1, 126))
    torch.onnx.export(
        model,
        examples,
        str(output),
        input_names=["pixels", "handcrafted"],
        output_names=["logits"],
        dynamic_axes={"pixels": {0: "batch"}, "handcrafted": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    # Validation-only export checks do not touch the test set or select the model.
    rows = [r for r in load_manifest(args.manifest) if r["split"] == "val"][:12]
    errors = []
    for row in rows:
        inputs = [model_inputs(image) for image in sample_frames(row["absolute_path"], 2)]
        pixels = np.stack([v[0] for v in inputs])
        crafted = np.stack([v[1] for v in inputs])
        with torch.inference_mode():
            expected = model(torch.from_numpy(pixels), torch.from_numpy(crafted)).numpy()
        actual = session.run(None, {"pixels": pixels, "handcrafted": crafted})[0]
        errors.append(float(np.abs(expected - actual).max()))
    if not errors or max(errors) > 1e-3:
        raise ValueError(f"Export parity failed: {errors}")
    report = dict(
        schema_version=1,
        architecture="EfficientNet-B0 + DCT63 + pHash63 -> MLP256/64/1",
        feature_version=FEATURE_VERSION,
        model_sha256=file_sha(output),
        status="validated",
        training_manifest_sha256=training["manifest_sha256"],
        training_weights_sha256=training["weights_sha256"],
        training_runtime=training["runtime"],
        training_sha256=file_sha(args.run / "training.json"),
        export_runtime=runtime_identity(),
        benchmark_sha256=file_sha(args.run / "benchmark.json"),
        parity_max_logit_error=max(errors),
        parity_inputs=len(rows) * 2,
        task="facial_manipulation",
        threshold=0.5,
        limitation="Source-pair FF++ evaluation only; no evidence for fully generated videos or calibrated probabilities.",
    )
    (args.run / "model-card.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run", type=Path)
    main(parser.parse_args())
