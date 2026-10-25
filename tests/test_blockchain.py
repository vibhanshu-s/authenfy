import io

import app as app_module
from app import create_app
from tests.test_app import FakeDetector, png


def test_browser_transaction_is_verified_against_registry_and_saved(tmp_path, monkeypatch):
    publisher = "0x1111111111111111111111111111111111111111"
    registry = "0x2222222222222222222222222222222222222222"
    tx_hash = "0x" + "ab" * 32
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "DATABASE": str(tmp_path / "receipts.sqlite3"),
            "SIGNING_KEY_PATH": str(tmp_path / "receipts.key"),
            "BLOCKCHAIN_REGISTRY_ADDRESS": registry,
            "BLOCKCHAIN_CHAIN_ID": "1337",
            "BLOCKCHAIN_RPC_URL": "http://rpc.example.test",
        },
        FakeDetector(),
    )
    receipt = app.test_client().post(
        "/analyze", data={"media": (io.BytesIO(png()), "test.png")}
    ).get_json()
    event = "0xe7e560a3f83a100a96b69bee4a2060f4ace35f3c25fa188088ece2dc435f6853"
    file_topic = "0x" + receipt["file_sha256"]
    commitment_topic = "0x" + receipt["receipt_commitment"]
    calls = {
        "eth_getTransactionByHash": {
            "to": registry,
            "from": publisher,
            "input": "0xcda1c3b9" + receipt["file_sha256"] + receipt["receipt_commitment"],
        },
        "eth_getTransactionReceipt": {
            "to": registry,
            "status": "0x1",
            "blockNumber": "0x64",
            "logs": [{"address": registry, "topics": [event, file_topic, commitment_topic]}],
        },
        "eth_chainId": "0x539",
        "eth_blockNumber": "0x65",
        "eth_call": "0x" + "0" * 24 + publisher[2:],
    }

    class Response:
        def __init__(self, result):
            self.result = result

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": self.result}

    monkeypatch.setattr(app_module.requests, "post", lambda _url, json, timeout: Response(calls[json["method"]]))
    client = app.test_client()
    response = client.post(
        "/anchors/confirm",
        json={"kind": "receipt", "subject_id": receipt["receipt_id"], "transaction_hash": tx_hash},
    )
    assert response.status_code == 200
    assert response.get_json()["blockchain_anchor"]["block_number"] == 100
    assert client.get(f"/receipts/{receipt['receipt_id']}").get_json()["blockchain_anchor"]["tx_hash"] == tx_hash


def test_anchor_refuses_unrelated_transaction_event(tmp_path, monkeypatch):
    publisher = "0x1111111111111111111111111111111111111111"
    registry = "0x2222222222222222222222222222222222222222"
    tx_hash = "0x" + "cd" * 32
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "DATABASE": str(tmp_path / "receipts.sqlite3"),
            "SIGNING_KEY_PATH": str(tmp_path / "receipts.key"),
            "BLOCKCHAIN_REGISTRY_ADDRESS": registry,
            "BLOCKCHAIN_CHAIN_ID": "1337",
            "BLOCKCHAIN_RPC_URL": "http://rpc.example.test",
        },
        FakeDetector(),
    )
    receipt = app.test_client().post("/analyze", data={"media": (io.BytesIO(png()), "test.png")}).get_json()
    event = "0xe7e560a3f83a100a96b69bee4a2060f4ace35f3c25fa188088ece2dc435f6853"
    calls = {
        "eth_getTransactionByHash": {
            "to": registry,
            "from": publisher,
            "input": "0xcda1c3b9" + receipt["file_sha256"] + receipt["receipt_commitment"],
        },
        "eth_getTransactionReceipt": {
            "to": registry,
            "status": "0x1",
            "blockNumber": "0x64",
            "logs": [{"address": registry, "topics": [event, "0x" + "0" * 64, "0x" + receipt["receipt_commitment"]]}],
        },
        "eth_chainId": "0x539",
        "eth_blockNumber": "0x65",
        "eth_call": "0x" + "0" * 24 + publisher[2:],
    }

    class Response:
        def __init__(self, result):
            self.result = result

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": self.result}

    monkeypatch.setattr(app_module.requests, "post", lambda _url, json, timeout: Response(calls[json["method"]]))
    result = app.test_client().post(
        "/anchors/confirm",
        json={"kind": "receipt", "subject_id": receipt["receipt_id"], "transaction_hash": tx_hash},
    )
    assert result.status_code == 409
    assert result.get_json()["code"] == "anchor_event_missing"
    assert app.extensions["receipts"].get_anchor("receipt", receipt["receipt_id"]) is None


def test_anchor_endpoint_disabled_without_deployment_config(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "DATABASE": str(tmp_path / "receipts.sqlite3"),
            "SIGNING_KEY_PATH": str(tmp_path / "receipts.key"),
        },
        FakeDetector(),
    )
    response = app.test_client().post(
        "/anchors/confirm", json={"kind": "evidence", "subject_id": "v1-ff-c23-100", "transaction_hash": "0x" + "00" * 32}
    )
    assert response.status_code == 503
