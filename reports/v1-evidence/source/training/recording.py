"""Durable per-video benchmark records, with strict identity and frame checksums."""

import json
from pathlib import Path

import numpy as np

from training.manifest import file_sha


class VideoLedger:
    def __init__(self, root, identity, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"
        self.records = []
        self.identity = identity
        if self.path.exists():
            if not resume:
                raise ValueError("Benchmark progress exists; use --resume or a new run directory")
            state = json.loads(self.path.read_text())
            if state.get("schema_version") != 1 or state.get("identity") != identity:
                raise ValueError("Cannot resume: benchmark code, inputs, settings or runtime changed")
            self.records = state["records"]
            # Validate every completed chunk before continuing or using saved scores.
            for index in range(len(self.records)):
                self.frames(index)

    def frames(self, index):
        record = self.records[index]
        path = self.root / f"{index:05}.npz"
        if not path.is_file() or file_sha(path) != record["frame_chunk_sha256"]:
            raise ValueError("Saved benchmark frame chunk is missing or corrupt")
        with np.load(path, allow_pickle=False) as chunk:
            result = {arm: chunk[arm].copy() for arm in self.identity["arms"]}
        count = record["measurements"][self.identity["arms"][0]][0]["frames"]
        if any(values.shape != (count,) or not np.isfinite(values).all() for values in result.values()):
            raise ValueError("Saved benchmark frame count or scores are invalid")
        return result

    def append(self, record, frames):
        index = len(self.records)
        path = self.root / f"{index:05}.npz"
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **{arm: np.asarray(values, dtype=np.float64) for arm, values in frames.items()})
        temporary.replace(path)
        saved = dict(record, frame_chunk_sha256=file_sha(path))
        self.records.append(saved)
        self.frames(index)
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(dict(schema_version=1, identity=self.identity, records=self.records), indent=2, allow_nan=False)
        )
        temporary.replace(self.path)
        return saved
