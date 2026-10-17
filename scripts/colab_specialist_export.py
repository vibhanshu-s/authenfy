"""Reconstruct pinned tensor weights using registry-installed Transformers; export FP32 ONNX."""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx==1.20.1"], check=True)
from transformers import CLIPVisionConfig, CLIPVisionModel
import torch.nn.functional as F

print("FACE_HEAD_GRAPH", face_archive.model.code, flush=True)
config = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                         num_attention_heads=16, image_size=224, patch_size=14,
                         hidden_act="quick_gelu", layer_norm_eps=1e-5, attention_dropout=0.0)
config._attn_implementation = "eager"
vision = CLIPVisionModel(config)
converted = {}
prefix = "feature_extractor.base_model.model.vision_model."
for key, value in face_state.items():
    if not key.startswith(prefix) or key.endswith("position_ids") or ".base_layer." in key:
        continue
    name = key[len(prefix):].replace(".ln_tuning_layers.default.", ".")
    converted[name] = value.float()
vision.vision_model.load_state_dict(converted, strict=True)


class FaceClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = vision
        self.linear = torch.nn.Linear(1024, 2)
        self.linear.weight.data.copy_(face_state["model.linear.weight"].float())
        self.linear.bias.data.copy_(face_state["model.linear.bias"].float())

    def forward(self, pixel_values):
        features = self.vision(pixel_values=pixel_values).pooler_output
        return self.linear(F.normalize(features, p=2, dim=1))


def prepare_face(image):
    image = image.convert("RGB")
    w, h = image.size
    scale = 224 / min(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    mean = np.array([0.48145466, .4578275, .40821073], np.float32)
    std = np.array([.26862954, .26130258, .27577711], np.float32)
    return np.ascontiguousarray(((np.asarray(image, np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])


face_model = FaceClassifier().eval().cuda().float()
# The published trace hard-codes BF16 casts. Replay it in FP32 for an apples-to-apples
# architecture check, rather than comparing different numerical precision paths.
face_archive = torch.jit.load(face_path, map_location="cpu").eval().cuda().float()
for module in face_archive.modules():
    if module._c._has_method("forward"):
        graph = module.graph
        for node in list(graph.nodes()):
            if node.kind() == "aten::to":
                for index, value in enumerate(list(node.inputs())[1:3], start=1):
                    if value.node().kind() == "prim::Constant" and value.toIValue() == 15:
                        replacement = graph.insertConstant(6)
                        replacement.node().moveBefore(node)
                        node.replaceInput(index, replacement)
specialist_records = []
with torch.inference_mode():
    for path, image, label, digest in samples:
        x = torch.from_numpy(prepare_face(image)).cuda()
        expected = face_archive(x).softmax(dim=1)[0, 1].item()
        actual = face_model(x).softmax(dim=1)[0, 1].item()
        assert abs(expected-actual) < 1e-4, (path, expected, actual, "reconstruction mismatch")
        specialist_records.append({"path":path, "label_fake":label, "sha256":digest,
                                   "published_graph_fp32_score":expected, "reconstructed_score":actual})
print("FACE_RECONSTRUCTION_PASSED", flush=True)
del face_archive, face_state, converted
import gc
gc.collect()
torch.cuda.empty_cache()
dummy = torch.zeros(1, 3, 224, 224, device="cuda")
torch.onnx.export(face_model, dummy, "face-specialist.onnx", input_names=["pixel_values"],
                  output_names=["logits"], opset_version=17, dynamo=False, external_data=False)
print("FACE_ONNX_EXPORTED", flush=True)
onnx_options = ort.SessionOptions()
onnx_options.intra_op_num_threads = 2
face_onnx = ort.InferenceSession("face-specialist.onnx", sess_options=onnx_options, providers=["CPUExecutionProvider"])
for row, (_, image, _, _) in zip(specialist_records, samples):
    logits = face_onnx.run(None, {"pixel_values":prepare_face(image)})[0].reshape(-1)
    p = np.exp(logits-logits.max()); score=float(p[1]/p.sum())
    row["onnx_score"] = score
    row["onnx_absolute_error"] = abs(score-row["reconstructed_score"])
    assert row["onnx_absolute_error"] < 1e-4, row
specialist_labels = np.array([row["label_fake"] for row in specialist_records])
specialist_scores = np.array([row["onnx_score"] for row in specialist_records])
model_digest = hashlib.sha256(Path("face-specialist.onnx").read_bytes()).hexdigest()
specialist_report = {"status":"passed", "gpu":gpu, "kind":"specialist_reconstruction_export_and_tiny_audit",
                    "utc":datetime.now(timezone.utc).isoformat(), "source_model":FACE_MODEL_ID,
                    "source_revision":FACE_REVISION, "source_sha256":FACE_SHA, "reference_precision":"Published trace replayed in FP32, promoting explicit BF16 casts to FP32", "onnx_sha256":model_digest,
                    "torch":torch.__version__, "onnxruntime":ort.__version__, "n_images":len(samples),
                    "sample_source":REPO, "sample_commit":COMMIT, "n_groups":8,
                    "accuracy_at_0_5":float(((specialist_scores>=.5)==specialist_labels).mean()),
                    "fake_detected_at_0_5":int(((specialist_scores>=.5)&(specialist_labels==1)).sum()),
                    "false_positive_at_0_5":int(((specialist_scores>=.5)&(specialist_labels==0)).sum()),
                    "limitations":"32 correlated example crops, FF++ may overlap training. This is a regression audit, not independent performance or calibration evidence. YuNet alignment on arbitrary uploads is an approximation of original dlib preprocessing and requires separate validation.",
                    "records":specialist_records}
Path("authenfy-specialist-validation.json").write_text(json.dumps(specialist_report, indent=2))
print("AUTHENFY_SPECIALIST_VALIDATION", json.dumps({k:v for k,v in specialist_report.items() if k!='records'}), flush=True)
files.download("authenfy-specialist-validation.json")
files.download("face-specialist.onnx")
