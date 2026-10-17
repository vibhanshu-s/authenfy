"""Local HMAC receipts. Not blockchain, not proof of factual truth."""

import hashlib
import hmac
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ReceiptStore:
    def __init__(self, database, key_path):
        self.database = str(database)
        Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        key_path = Path(key_path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os

            with key_path.open("xb") as handle:
                handle.write(os.urandom(32))
        except FileExistsError:
            pass
        self.key = key_path.read_bytes()
        if len(self.key) != 32:
            raise RuntimeError("Invalid receipt signing key")
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, payload TEXT NOT NULL, signature TEXT NOT NULL)"
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def sign(self, payload):
        return hmac.new(self.key, canonical(payload).encode(), hashlib.sha256).hexdigest()

    def put(self, result, file_hash):
        payload = {
            **result,
            "receipt_id": uuid.uuid4().hex,
            "file_sha256": file_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        signature = self.sign(payload)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO receipts VALUES (?, ?, ?)", (payload["receipt_id"], canonical(payload), signature)
            )
        return {
            **payload,
            "signature": signature,
            "receipt_commitment": hashlib.sha256(canonical(payload).encode()).hexdigest(),
        }

    def get(self, receipt_id):
        with self.connect() as connection:
            row = connection.execute("SELECT payload, signature FROM receipts WHERE id = ?", (receipt_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        valid = hmac.compare_digest(self.sign(payload), row[1])
        if not valid:
            raise ValueError("Receipt integrity check failed")
        return {
            **payload,
            "signature": row[1],
            "receipt_commitment": hashlib.sha256(canonical(payload).encode()).hexdigest(),
        }
