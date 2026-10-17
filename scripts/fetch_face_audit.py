"""Download only the public audit examples and verify every committed input hash."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/face-audit-baseline.json").read_text())
    base = f"https://raw.githubusercontent.com/{report['sample_source']}/{report['sample_commit']}/"
    folder = root / "private/face-audit"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        response = requests.get(base + row["path"], timeout=40)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != row["sha256"]:
            raise RuntimeError("Audit example checksum mismatch")
        target = folder / (hashlib.sha256(row["path"].encode()).hexdigest()[:16] + ".png")
        target.write_bytes(response.content)
        return {**row, "local_path": str(target)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, report["records"]))
    (folder / "inputs.json").write_text(json.dumps(records, indent=2))
    print(f"Verified {len(records)} audit examples; manifest: {folder / 'inputs.json'}")


if __name__ == "__main__":
    main()
