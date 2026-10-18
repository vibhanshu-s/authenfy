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


def test_benchmark_backup_restores_latest_committed_videos_and_deduplicates(tmp_path):
    import numpy as np
    from training.recording import VideoLedger

    run = tmp_path / 'run'
    run.mkdir()
    store = RecoveryStore(tmp_path / 'backup')
    (run / 'training.json').write_text('{"status":"trained"}')
    store.snapshot(run)
    identity = dict(arms=['baseline_all'])
    ledger = VideoLedger(run / 'benchmark-records', identity)
    for i in range(2):
        ledger.append(dict(measurements={'baseline_all': [dict(frames=2)]}),
                      {'baseline_all': np.array([0.1+i/10, 0.5])})
        (run / 'benchmark-progress.json').write_text(json.dumps(ledger.records))
        store.snapshot_benchmark(run)
    before = {p.name for p in (store.directory / 'benchmark-objects').iterdir()}
    store.snapshot_benchmark(run)
    assert before == {p.name for p in (store.directory / 'benchmark-objects').iterdir()}
    restored = tmp_path / 'replacement'
    RecoveryStore(store.directory).restore(restored)
    recovered = VideoLedger(restored / 'benchmark-records', identity, resume=True)
    assert len(recovered.records) == 2
    np.testing.assert_array_equal(recovered.frames(1)['baseline_all'], [0.2, 0.5])
    assert (restored / 'training.json').read_text() == '{"status":"trained"}'
    latest = json.loads((store.directory / 'benchmark-latest.json').read_text())
    digest = latest['files']['benchmark-records/00000.npz']
    (store.directory / 'benchmark-objects' / digest).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        store.snapshot_benchmark(run)
    with pytest.raises(ValueError, match='checksum'):
        RecoveryStore(store.directory).restore(tmp_path / 'bad-restore')
