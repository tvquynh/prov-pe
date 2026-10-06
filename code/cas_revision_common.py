"""Read-only input helpers and fresh-output receipts for the bounded CAS revision."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone


class CheckFailed(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise CheckFailed(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def fresh(path, protected=()):
    output = Path(path).resolve()
    for item in protected:
        source = Path(item).resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output), 'OUTPUT_OVERLAPS_INPUT: ' + str(source))
    require(not output.exists(), 'OUTPUT_ALREADY_EXISTS: ' + str(output))
    output.mkdir(parents=True)
    return output


def environment(names=()):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'versions': versions,
            'utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv}


def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name != 'INPUT_MANIFEST.json'}
