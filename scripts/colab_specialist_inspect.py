"""Inspect the pinned published facial-manipulation model's tensor structure."""

FACE_MODEL_ID = "yermandy/deepfake-detection"
FACE_REVISION = "9a6857ec642deb57373c5437be803a199468b8c6"
FACE_SHA = "55c54d3f4d301ce07f0d08126fc4202572d20ea6952b039bc7a949a43714157d"
face_path = hf_hub_download(FACE_MODEL_ID, "model.torchscript", revision=FACE_REVISION)
assert hashlib.sha256(Path(face_path).read_bytes()).hexdigest() == FACE_SHA
face_archive = torch.jit.load(face_path, map_location="cpu")
face_state = face_archive.state_dict()
print(
    "FACE_TENSOR_STRUCTURE",
    json.dumps(
        {
            key: list(value.shape)
            for key, value in face_state.items()
            if "linear" in key or "embeddings" in key or "layer_norm" in key or "ln_" in key
        }
    )[:6000],
    flush=True,
)
print("FACE_FIRST_KEYS", list(face_state)[:20], flush=True)
print("FACE_GRAPH", str(face_archive.code)[:2000], flush=True)
