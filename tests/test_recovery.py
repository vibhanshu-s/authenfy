import json
from zipfile import ZipFile

import pytest

from training.manifest import file_sha
from training.recovery import RecoveryStore


def test_persistent_backup_restores_without_original_runtime(tmp_path):
    run = tmp_path / 'ephemeral'
    cache = run / 'crop-cache' / 'digest-8'
    cache.mkdir(parents=True)
    (cache / '000.png').write_bytes(b'prepared frame fixture')
    (cache / 'complete.json').write_text('["000.png"]')
    (run / 'baseline-training-state.pt').write_bytes(b'checkpoint fixture')
    (run / 'training.json').write_text('{"status":"training"}')
    backup = tmp_path / 'persistent'
    store = RecoveryStore(backup)
    store.cache(cache)
    store.flush_cache()
    store.snapshot(run)
    # A new store and empty target simulate replacement of /content.
    replacement = tmp_path / 'replacement'
    restored = RecoveryStore(backup)
    restored.restore(replacement)
    for relative in ['training.json', 'baseline-training-state.pt', 'crop-cache/digest-8/000.png']:
        assert (replacement / relative).read_bytes() == (run / relative).read_bytes()
    restored.cache(replacement / 'crop-cache' / 'digest-8')
    restored.flush_cache()
    assert len(json.loads((backup / 'cache-index.json').read_text())) == 1
    with pytest.raises(ValueError, match='empty'):
        restored.restore(replacement)


def test_persistent_backup_rejects_corrupt_checkpoint(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'baseline.pt').write_bytes(b'weights')
    store = RecoveryStore(tmp_path / 'backup')
    store.snapshot(run)
    latest = json.loads((store.directory / 'latest.json').read_text())
    (store.directory / latest['folder'] / 'baseline.pt').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='checksum'):
        store.restore(tmp_path / 'restored')


def test_persistent_backup_rejects_archive_path_escape(tmp_path):
    backup = tmp_path / 'backup'
    backup.mkdir()
    archive = backup / 'cache.zip'
    with ZipFile(archive, 'w') as output:
        output.writestr('../escaped', b'unsafe')
    (backup / 'cache-index.json').write_text(json.dumps([
        dict(file='cache.zip', sha256=file_sha(archive), directories=['digest-8'])
    ]))
    with pytest.raises(ValueError, match='path'):
        RecoveryStore(backup).restore(tmp_path / 'restored')
    assert not (tmp_path / 'escaped').exists()
