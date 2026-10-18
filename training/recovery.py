"""Checksummed recovery copies on persistent storage; training stays on local disk."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4
from zipfile import ZipFile, ZIP_STORED

from training.manifest import file_sha


def write_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class RecoveryStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / 'cache-index.json'
        self.chunks = json.loads(self.index_path.read_text()) if self.index_path.exists() else []
        self.saved = {name for chunk in self.chunks for name in chunk['directories']}
        self.pending = []

    def cache(self, directory):
        directory = Path(directory)
        if directory.name not in self.saved:
            self.pending.append(directory)
            self.saved.add(directory.name)
        if len(self.pending) >= 128:
            self.flush_cache()

    def flush_cache(self):
        if not self.pending:
            return
        name = f'cache-{uuid4().hex}.zip'
        # Build locally, then publish and verify before committing the index.
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / name
            with ZipFile(archive, 'w', compression=ZIP_STORED) as output:
                for directory in self.pending:
                    files = json.loads((directory / 'complete.json').read_text())
                    for filename in [*files, 'complete.json']:
                        if Path(filename).name != filename:
                            raise ValueError('Invalid cache filename')
                        output.write(directory / filename, f'{directory.name}/{filename}')
            digest = file_sha(archive)
            target = self.directory / name
            shutil.copyfile(archive, target.with_suffix('.partial'))
            target.with_suffix('.partial').replace(target)
            if file_sha(target) != digest:
                raise ValueError('Persistent cache backup checksum mismatch')
        self.chunks.append(dict(file=name, sha256=digest, directories=[p.name for p in self.pending]))
        write_json(self.index_path, self.chunks)
        self.pending.clear()
        print(f'RECOVERY_CACHE_VIDEOS {len(self.saved)}', flush=True)

    def snapshot(self, root):
        root = Path(root)
        folder = self.directory / ('snapshot-' + uuid4().hex)
        folder.mkdir()
        files = {}
        for source in sorted(root.rglob('*')):
            relative = source.relative_to(root)
            if not source.is_file() or 'crop-cache' in relative.parts or source.suffix == '.partial':
                continue
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = file_sha(target)
            # Logs may grow while copying; preserve their copied prefix. Model,
            # configuration and metric files must match their source exactly.
            if source.suffix != '.log' and file_sha(source) != digest:
                raise ValueError('Persistent checkpoint backup checksum mismatch')
            files[relative.as_posix()] = digest
        write_json(self.directory / 'latest.json', dict(
            folder=folder.name, files=files, saved_utc=datetime.now(timezone.utc).isoformat()
        ))
        print(f'RECOVERY_SNAPSHOT {folder.name}', flush=True)

    def restore(self, root):
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise ValueError('Restore requires an empty local run directory')
        root.mkdir(parents=True, exist_ok=True)
        for chunk in self.chunks:
            if Path(chunk['file']).name != chunk['file'] or not chunk['file'].endswith('.zip'):
                raise ValueError('Invalid recovery archive path')
            archive = self.directory / chunk['file']
            if file_sha(archive) != chunk['sha256']:
                raise ValueError('Persistent cache archive checksum mismatch')
            with ZipFile(archive) as source:
                for member in source.infolist():
                    parts = Path(member.filename).parts
                    if (
                        len(parts) != 2 or parts[0] not in chunk['directories'] or '..' in parts
                        or ':' in member.filename or '\\' in member.filename
                    ):
                        raise ValueError('Invalid recovery archive path')
                source.extractall(root / 'crop-cache')
        latest = self.directory / 'latest.json'
        if latest.exists():
            record = json.loads(latest.read_text())
            folder = self.directory / record['folder']
            if folder.parent != self.directory or not folder.name.startswith('snapshot-'):
                raise ValueError('Invalid recovery snapshot path')
            for name, digest in record['files'].items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid recovery file path')
                source = folder / relative
                if file_sha(source) != digest:
                    raise ValueError('Persistent checkpoint checksum mismatch')
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

        self.restore_benchmark(root)

    def snapshot_benchmark(self, root):
        """Publish completed videos between timings, copying immutable chunks once."""
        root = Path(root)
        state = root / 'benchmark-records' / 'state.json'
        if not state.exists():
            return
        record = json.loads(state.read_text())
        files = [state]
        progress = root / 'benchmark-progress.json'
        if progress.exists():
            files.append(progress)
        files.extend(state.parent / f'{i:05}.npz' for i in range(len(record['records'])))
        objects = self.directory / 'benchmark-objects'
        objects.mkdir(exist_ok=True)
        entries = {}
        for source in files:
            digest = file_sha(source)
            target = objects / digest
            if not target.exists():
                temporary = objects / (digest + '.partial')
                shutil.copyfile(source, temporary)
                if file_sha(temporary) != digest:
                    raise ValueError('Benchmark recovery checksum mismatch')
                temporary.replace(target)
            elif file_sha(target) != digest:
                raise ValueError('Existing benchmark recovery object is corrupt')
            entries[source.relative_to(root).as_posix()] = digest
        write_json(self.directory / 'benchmark-latest.json', dict(
            files=entries, completed_videos=len(record['records']),
            saved_utc=datetime.now(timezone.utc).isoformat()
        ))
        print(f'RECOVERY_BENCHMARK_VIDEOS {len(record["records"])}', flush=True)

    def restore_benchmark(self, root):
        latest = self.directory / 'benchmark-latest.json'
        if not latest.exists():
            return
        record = json.loads(latest.read_text())
        for name, digest in record['files'].items():
            relative = Path(name)
            if (
                relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or not (name == 'benchmark-progress.json' or name.startswith('benchmark-records/'))
            ):
                raise ValueError('Invalid benchmark recovery path')
            source = self.directory / 'benchmark-objects' / digest
            if file_sha(source) != digest:
                raise ValueError('Benchmark recovery checksum mismatch')
            target = Path(root) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup', type=Path)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    RecoveryStore(args.backup).restore(args.run)
