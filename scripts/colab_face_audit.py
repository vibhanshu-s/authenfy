"""Small labeled face-swap audit after colab_validate.py, not a benchmark."""
from concurrent.futures import ThreadPoolExecutor

REPO = "yermandy/deepfake-detection"
COMMIT = "0b872f7903e72d2af8c5fff262d2fc2ca99bb44f"
tree_response = requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", timeout=60)
tree_response.raise_for_status()
paths = sorted(row["path"] for row in tree_response.json()["tree"] if row["path"].endswith(".png") and row["path"].startswith("datasets/"))


def fetch(path):
    response = requests.get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}", timeout=60)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    label = 0 if "/real/" in path or "/Celeb-real/" in path or "/YouTube-real/" in path else 1
    return path, image, label, hashlib.sha256(response.content).hexdigest()


with ThreadPoolExecutor(max_workers=4) as pool:
    samples = list(pool.map(fetch, paths))
face_records = []
for path, image, label, digest in samples:
    x = torch.from_numpy(prepare(image)).cuda()
    with torch.inference_mode():
        score = torch.sigmoid(model(pixel_values=x).logits).item()
    face_records.append({"path": path, "label_fake": label, "score": score, "sha256": digest,
                         "group": "/".join(path.split("/")[:-1])})
labels = np.array([row["label_fake"] for row in face_records])
scores = np.array([row["score"] for row in face_records])
assert len(face_records) == 32 and set(labels) == {0, 1}
tp = int(((scores >= .5) & (labels == 1)).sum())
tn = int(((scores < .5) & (labels == 0)).sum())
fp = int(((scores >= .5) & (labels == 0)).sum())
fn = int(((scores < .5) & (labels == 1)).sum())
face_report = {"kind": "tiny_labeled_face_swap_gap_audit", "gpu": gpu, "utc": datetime.now(timezone.utc).isoformat(),
               "model": MODEL_ID, "model_revision": REVISION, "sample_source": REPO, "sample_commit": COMMIT,
               "n_images": len(face_records), "n_groups": len(set(row["group"] for row in face_records)),
               "fake_count": int(labels.sum()), "real_count": int((labels == 0).sum()),
               "accuracy_at_0_5": (tp+tn)/len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "false_low_signal_fake_images": int(((scores <= .2) & (labels == 1)).sum()),
               "high_signal_fake_images": int(((scores >= .8) & (labels == 1)).sum()),
               "limitations": "32 correlated example crops from 8 video/method groups; unknown training overlap; not an independent test, calibration set, or production accuracy estimate. No model fitting or threshold tuning.",
               "records": face_records}
Path("authenfy-face-audit.json").write_text(json.dumps(face_report, indent=2))
print("AUTHENFY_FACE_AUDIT", json.dumps({key:value for key,value in face_report.items() if key != "records"}), flush=True)
files.download("authenfy-face-audit.json")
